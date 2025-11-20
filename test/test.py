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
    # Handle NaN/inf
    if math.isnan(value):
        return 0x7F  # NaN -> set exponent all ones and non-zero mantissa
    if math.isinf(value):
        return 0x7C if value > 0 else 0xFC  # +Inf / -Inf

    # Work in float32 bit representation
    f32 = struct.unpack('>I', struct.pack('>f', float(value)))[0]
    sign = (f32 >> 31) & 0x1
    exp32 = (f32 >> 23) & 0xFF
    mant32 = f32 & 0x7FFFFF

    if exp32 == 0:  # zero or subnormal in float32
        return (sign << 7)

    # Compute unbiased exponent
    e32 = exp32 - 127
    e8 = e32 + 7  # bias for e4 is 7

    if e8 <= 0:
        # Underflow to zero (we do not implement fp8 subnormals here)
        return (sign << 7)
    if e8 >= 0xF:
        # Overflow to inf
        return (sign << 7) | (0xF << 3)

    # Create 3-bit mantissa by shifting and rounding
    # mant32 has 23 bits; we need top 3 bits (bits 22..20)
    shift = 23 - 3
    mant3 = (mant32 >> shift) & 0x7

    # Round-to-nearest: check the next bit
    next_bit = (mant32 >> (shift - 1)) & 0x1 if shift - 1 >= 0 else 0
    if next_bit == 1:
        mant3 += 1
        if mant3 == 0x8:  # mantissa overflow
            mant3 = 0
            e8 += 1
            if e8 >= 0xF:
                return (sign << 7) | (0xF << 3)

    fp8 = (sign << 7) | ((e8 & 0xF) << 3) | (mant3 & 0x7)
    return int(fp8)


def bf16_to_float(u16: int) -> float:
    """Convert a 16-bit BF16 bitpattern (as integer 0..65535) to Python float32."""
    # Ensure unsigned 16-bit
    u16 = int(u16) & 0xFFFF
    u32 = (u16 << 16) & 0xFFFFFFFF
    return struct.unpack('>f', struct.pack('>I', u32))[0]

def get_expected_matmul(A, B, transpose=False, relu=False):
    A_mat = np.array(A).reshape(2, 2)
    B_mat = np.array(B).reshape(2, 2)
    if transpose:
        B_mat = B_mat.T
    result = A_mat @ B_mat
    if relu:
        result = np.maximum(result, 0)
    return result.flatten().tolist()

async def load_matrix(dut, matrix, transpose=0, relu=0):
    for i in range(4):
        # Convert Python float/int to FP8 e4m3 representation before driving DUT
        dut.ui_in.value = float_to_fp8_e4m3(matrix[i])
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
            # Interpret combined as BF16 bitpattern and convert to float
            val_f = bf16_to_float(combined)

            results.append(val_f)
            dut._log.info(f"Read BF16 bits=0x{combined:04x} -> {val_f}")
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
    A = [79, -10, 7, 8]  # row-major
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