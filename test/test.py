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


def fp8_to_bf16_bits_from_byte(u8: int) -> int:
    """Convert FP8 e4m3 byte to BF16 bitpattern following PE.v rules.
    Implements: if exp==0 -> zero; else exp16=exp8+120; mant16=mant8<<5.
    """
    u8 = int(u8) & 0xFF
    sign = (u8 >> 7) & 0x1
    exp8 = (u8 >> 3) & 0xF
    mant8 = u8 & 0x7

    if exp8 == 0:
        exp16 = 0
        mant16 = 0
    else:
        exp16 = (exp8 + 120) & 0xFF
        mant16 = (mant8 << 5) & 0x7F

    return (sign << 15) | (exp16 << 7) | mant16


def bf16_mul_bits(a_bits: int, b_bits: int) -> int:
    """Multiply two BF16 bitpatterns following PE.v bf16_mul implementation."""
    a = int(a_bits) & 0xFFFF
    b = int(b_bits) & 0xFFFF

    sa = (a >> 15) & 1
    sb = (b >> 15) & 1
    ea = (a >> 7) & 0xFF
    eb = (b >> 7) & 0xFF
    ma = a & 0x7F
    mb = b & 0x7F

    sp = sa ^ sb

    # Special cases: zero or inf
    if ea == 0 or eb == 0:
        return (sp << 15)  # zero
    if ea == 0xFF or eb == 0xFF:
        return (sp << 15) | (0xFF << 7)  # inf

    ep = ea + eb - 127

    # mp = {1,ma} * {1,mb} where ma,mb are 7-bit -> implied leading 1
    A = (1 << 7) | ma
    B = (1 << 7) | mb
    mp = A * B  # up to 8*8 bits -> up to 16-bit, but verilog uses 14 bits

    # Determine if mp[13] is set (counting from 0)
    if (mp >> 13) & 1:
        mp_final = (mp >> 6) & 0x7F  # mp[12:6]
        round_bit = (mp >> 5) & 1    # mp[5]
        ep_final = ep + 1
    else:
        mp_final = (mp >> 5) & 0x7F  # mp[11:5]
        round_bit = (mp >> 4) & 1    # mp[4]
        ep_final = ep

    # Determine lower bits for tie-breaking: mp[3:0]
    lower_bits = mp & 0xF

    # Round-to-nearest-even
    if round_bit and ((mp_final & 1) or (lower_bits != 0)):
        mp_final = (mp_final + 1) & 0x7F

    if ep_final >= 0xFF:
        return (sp << 15) | (0xFF << 7)
    return (sp << 15) | ((ep_final & 0xFF) << 7) | (mp_final & 0x7F)


def bf16_add_bits(a_bits: int, b_bits: int) -> int:
    """Add two BF16 bitpatterns following PE.v bf16_add implementation."""
    a = int(a_bits) & 0xFFFF
    b = int(b_bits) & 0xFFFF

    sa = (a >> 15) & 1
    sb = (b >> 15) & 1
    ea = (a >> 7) & 0xFF
    eb = (b >> 7) & 0xFF
    # mant_a,mant_b include implicit 1 and extra guard bit as in Verilog: {1, a[6:0], 0}
    mant_a = ((1 << 8) | ((a & 0x7F) << 1))  # 1(implicit) + 7 bits + trailing 0 -> 9 bits
    mant_b = ((1 << 8) | ((b & 0x7F) << 1))

    if ea > eb:
        e_max = ea
        shift = ea - eb
        mant_b = mant_b >> shift
    else:
        e_max = eb
        shift = eb - ea
        mant_a = mant_a >> shift

    if sa == sb:
        mant_sum = mant_a + mant_b
        result_sign = sa
    else:
        # mant_sum = (sa ? mant_b - mant_a : mant_a - mant_b);
        if sa:
            mant_sum = mant_b - mant_a
            result_sign = sb
        else:
            mant_sum = mant_a - mant_b
            result_sign = sa

    # Normalize
    if (mant_sum >> 8) & 1:
        m_result = (mant_sum >> 2) & 0x7F
        e_result = e_max + 1
    else:
        m_result = (mant_sum >> 1) & 0x7F
        e_result = e_max

    if e_result >= 255:
        return (result_sign << 15) | (0xFF << 7)
    return (result_sign << 15) | ((e_result & 0xFF) << 7) | (m_result & 0x7F)

def get_expected_matmul(A, B, transpose=False, relu=False):
    # Bit-exact simulation of PE.v: FP8 -> BF16 conversion, BF16 multiply, BF16 add
    # A and B are length-4 lists in row-major order
    # Convert inputs A,B (which may be ints) to FP8 bytes then to BF16 bitpatterns
    A_bits = [fp8_to_bf16_bits_from_byte(float_to_fp8_e4m3(x)) for x in A]
    B_bits = [fp8_to_bf16_bits_from_byte(float_to_fp8_e4m3(x)) for x in B]

    # Build 2x2 matrices
    # A_bits[i,k] where i row, k col
    # indexes: row-major A = [A00, A01, A10, A11]
    out_bits = []
    for i in range(2):
        for j in range(2):
            accum = 0
            for k in range(2):
                a_idx = i*2 + k
                b_idx = k*2 + j
                a_b = A_bits[a_idx]
                b_b = B_bits[b_idx]
                prod = bf16_mul_bits(a_b, b_b)
                if k == 0:
                    accum = prod
                else:
                    accum = bf16_add_bits(accum, prod)
            out_bits.append(accum)

    # Decode BF16 bits to floats for comparison
    out = [bf16_to_float(x) for x in out_bits]

    if relu:
        out = [max(0.0, v) for v in out]

    return out


def get_expected_matmul_bits(A, B, transpose=False, relu=False):
    """Return expected BF16 bitpatterns (ints) for A,B using PE.v semantics."""
    A_bits = [fp8_to_bf16_bits_from_byte(float_to_fp8_e4m3(x)) for x in A]
    B_bits = [fp8_to_bf16_bits_from_byte(float_to_fp8_e4m3(x)) for x in B]

    out_bits = []
    for i in range(2):
        for j in range(2):
            accum = 0
            for k in range(2):
                a_idx = i*2 + k
                b_idx = k*2 + j
                a_b = A_bits[a_idx]
                b_b = B_bits[b_idx]
                prod = bf16_mul_bits(a_b, b_b)
                if k == 0:
                    accum = prod
                else:
                    accum = bf16_add_bits(accum, prod)
            out_bits.append(accum)

    if relu:
        # apply ReLU on decoded floats then re-encode to bits
        floats = [bf16_to_float(x) for x in out_bits]
        res = []
        for v in floats:
            v2 = max(0.0, v)
            res.append(float_to_bf16_bits(v2))
        return res

    return out_bits

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
            if inputs:
                enc0 = float_to_fp8_e4m3(inputs[idx0])
                dec0 = fp8_to_float(enc0)
                dut._log.info(f"Feeding A/B value idx{idx0}: orig={inputs[idx0]} fp8=0x{enc0:02x} -> {dec0}")
                dut.ui_in.value = enc0
            else:
                dut.ui_in.value = 0
            await ClockCycles(dut.clk, 1)
            high = dut.uo_out.value.integer & 0xFF

            if inputs:
                enc1 = float_to_fp8_e4m3(inputs[idx1])
                dec1 = fp8_to_float(enc1)
                dut._log.info(f"Feeding A/B value idx{idx1}: orig={inputs[idx1]} fp8=0x{enc1:02x} -> {dec1}")
                dut.ui_in.value = enc1
            else:
                dut.ui_in.value = 0
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
    results = []

    # Test 1 matrices (new inputs for read)
    A = [79, -10, 7, 8]  # row-major
    B = [2, 6, 5, 8]  # row-major: [B00, B01, B10, B11]

    # Compute expected for these test matrices (simulate FP8->BF16 path)
    expected = get_expected_matmul(A, B)
    expected_bits = get_expected_matmul_bits(A, B)
    for idx, eb in enumerate(expected_bits):
        dut._log.info(f"Expected BF16 idx{idx} = 0x{eb:04x} -> {bf16_to_float(eb)}")

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