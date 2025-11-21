import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import numpy as np
import struct
import math

async def reset_dut(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 1)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)

def saturate_to_s8(x):
    return max(-128, min(127, int(x)))

def float_to_fp8_e4m3(value: float) -> int:
    """
    Convert a Python float (assumed float32 range) to an FP8 e4m3 bit-pattern (8-bit int).
    This implements a reasonable round-to-nearest conversion (not IEEE-perfect for all edge cases).
    """
    # Use a small lookup to pick the closest representable FP8 (e4m3) value.
    # Build table on first call and cache it in the function attribute.
    if not hasattr(float_to_fp8_e4m3, "_table"):
        table = []
        bias = 7
        for u in range(256):
            sign = (u >> 7) & 0x1
            exp = (u >> 3) & 0xF
            mant = u & 0x7
            if exp == 0:
                if mant == 0:
                    val = -0.0 if sign else 0.0
                else:
                    # subnormal: mant / 2^3 * 2^(1-bias)
                    val = ((-1.0) ** sign) * (mant / 8.0) * (2.0 ** (1 - bias))
            elif exp == 0xF:
                if mant == 0:
                    val = float('-inf') if sign else float('inf')
                else:
                    val = float('nan')
            else:
                e = exp - bias
                mantissa = 1.0 + (mant / 8.0)
                val = ((-1.0) ** sign) * mantissa * (2.0 ** e)
            table.append(val)
        float_to_fp8_e4m3._table = table

    # Handle NaN/inf explicitly to pick a canonical bit pattern
    if math.isnan(value):
        return 0x7F
    if math.isinf(value):
        return 0xFC if value < 0 else 0x7C

    # Find nearest table entry (round-to-nearest; tie -> lower index)
    tbl = float_to_fp8_e4m3._table
    # Use absolute difference; for -0.0 match +0.0 as nearest
    best = 0
    best_diff = abs(tbl[0] - value)
    for i in range(1, 256):
        d = abs(tbl[i] - value)
        if d < best_diff:
            best_diff = d
            best = i
    return int(best)


def fp8_to_float(u8: int) -> float:
    """Decode FP8 e4m3 byte to Python float using same table."""
    if not hasattr(float_to_fp8_e4m3, "_table"):
        # ensure table built
        float_to_fp8_e4m3(0.0)
    return float_to_fp8_e4m3._table[int(u8) & 0xFF]

def bf16_to_float(u16: int) -> float:
    """Convert a 16-bit BF16 bitpattern (as integer 0..65535) to Python float32."""
    # Ensure unsigned 16-bit
    u16 = int(u16) & 0xFFFF
    u32 = (u16 << 16) & 0xFFFFFFFF
    return struct.unpack('>f', struct.pack('>I', u32))[0]


def float_to_bf16_bits(value: float) -> int:
    """Round Python float to BF16 bitpattern using round-to-nearest-even."""
    u32 = struct.unpack('>I', struct.pack('>f', float(value)))[0]
    upper = (u32 >> 16) & 0xFFFF
    lower = u32 & 0xFFFF

    if lower > 0x8000:
        upper = (upper + 1) & 0xFFFF
    elif lower < 0x8000:
        pass
    else:
        # tie: round to even
        if (upper & 1) == 1:
            upper = (upper + 1) & 0xFFFF

    return int(upper)

def get_expected_matmul(A, B, transpose=False, relu=False):
    # Simulate DUT datapath: inputs are converted to FP8 (e4m3),
    # internal arithmetic in float32, outputs rounded to BF16.
    A_vals = [fp8_to_float(float_to_fp8_e4m3(x)) for x in A]
    B_vals = [fp8_to_float(float_to_fp8_e4m3(x)) for x in B]

    A_mat = np.array(A_vals).reshape(2, 2)
    B_mat = np.array(B_vals).reshape(2, 2)
    if transpose:
        B_mat = B_mat.T
    result = A_mat @ B_mat
    if relu:
        result = np.maximum(result, 0)

    # Round to BF16 and return the decoded float values to compare with DUT reads
    out = []
    for v in result.flatten().tolist():
        bf = float_to_bf16_bits(v)
        out.append(bf16_to_float(bf))
    return out

async def load_matrix(dut, matrix, transpose=0, relu=0):
    for i in range(4):
        # Convert Python float/int to FP8 e4m3 representation before driving DUT
        dut.ui_in.value = float_to_fp8_e4m3(matrix[i])
        dut._log.info(f"Loading=0x{float_to_fp8_e4m3(matrix[i]):04x} originally {matrix[i]}")
        dut.uio_in.value = (transpose << 1) | (relu << 2) | 1  # load_en=1, load_sel_ab=sel, load_index
        await RisingEdge(dut.clk)

async def parallel_load_read(dut, A, B, transpose=0, relu=0):
    results = []
    dut.uio_in.value = (transpose << 1) | (relu << 2) | 1  # load_en=1

    for inputs in [A, B]:
        for i in range(2):
            idx0 = i * 2
            idx1 = i * 2 + 1
            # Feed either real data or dummy zeros
            dut.ui_in.value = float_to_fp8_e4m3(inputs[idx0]) if inputs else 0
            await ClockCycles(dut.clk, 1)
            high = dut.uo_out.value.integer & 0xFF

            dut.ui_in.value = float_to_fp8_e4m3(inputs[idx1]) if inputs else 0
            await ClockCycles(dut.clk, 1)
            low = dut.uo_out.value.integer & 0xFF

            combined = (high << 8) | low
            # Diagnostic: report multiple interpretations of the two bytes
            signed16 = combined if combined < 0x8000 else combined - 0x10000
            bf16_float = bf16_to_float(combined)
            swapped = (low << 8) | high
            swapped_signed16 = swapped if swapped < 0x8000 else swapped - 0x10000
            swapped_bf16_float = bf16_to_float(swapped)

            results.append(bf16_float)
            dut._log.info(f"Read bytes high=0x{high:02x} low=0x{low:02x} comb=0x{combined:04x} signed16={signed16} bf16={bf16_float}")
            dut._log.info(f"Swapped comb=0x{swapped:04x} signed16={swapped_signed16} bf16={swapped_bf16_float}")
    return results

@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    await reset_dut(dut)

    # ------------------------------
    # STEP 1: Load matrix A
    # A = [[1, 2],
    #      [3, 4]]
    A = [1, 2, 3, 4]  # row-major

    # ------------------------------
    # STEP 2: Load matrix B
    # B = [[5, 6],
    #      [7, 8]]
    B = [5, 6, 7, 8]  # row-major: [B00, B01, B10, B11]
    
    await load_matrix(dut, A)
    await load_matrix(dut, B)

    # ------------------------------
    # STEP 4: Read outputs
    expected = get_expected_matmul(A, B)
    results = []
    
    # Test 2 matrices
    A = [2, -1, 3, 5]  # row-major
    B = [2, 6, 5, 8]  # row-major: [B00, B01, B10, B11]

    # Read test 1 matrices
    results = await parallel_load_read(dut, A, B)

    # ------------------------------
    # STEP 5: Check results of test 1
    for i in range(4):
        assert np.isclose(results[i], expected[i], atol=1e-3), f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 1 passed!")
    
    #######################################
    ##### TEST RUN 2 - CHECK CLEARING #####

    # ------------------------------
    # STEP 4: Get expected of test 2
    expected = get_expected_matmul(A, B)
    results = []

    A = [5, -6, 7, 8]  # row-major
    B = [1, 2, 3, -4]  # row-major: [B00, B01, B10, B11]

    # Read test 2 outputs + load test 3 inputs
    results = await parallel_load_read(dut, A, B)

    # ------------------------------
    # STEP 5: Check results of test 2
    for i in range(4):
        assert np.isclose(results[i], expected[i], atol=1e-3), f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("Test 2 passed!")