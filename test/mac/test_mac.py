import cocotb
from cocotb.triggers import RisingEdge, ClockCycles
from cocotb.clock import Clock
import random
import math
import struct
# ---------- helpers to emulate RTL bit-exact behavior ----------

FP8_BIAS = 7
INT_ACC_OFFSET = 18

def bf16_to_float(bf16: int) -> float:
    i = bf16 << 16
    b = struct.pack(">I", i)
    return struct.unpack(">f", b)[0]

def fp8_e4m3_encode(x: float) -> int:
    if math.isnan(x):
        return 0x7F
    if math.isinf(x):
        return 0x7F if x > 0 else 0xFF

    sign = 0
    if x < 0:
        sign = 1
        x = -x

    if x == 0:
        return sign << 7

    exp = math.floor(math.log2(x))
    
    # FP8 E4M3 bias = 7, min normal exp = -6
    exp_fp8 = exp + 7

    # Denormal handling
    if exp_fp8 <= 0:
        # Denormal: exp_fp8 = 0, effective exponent = -6
        # Value = 2^(-6) * (0.mantissa)
        # So: x = 2^(-6) * (mantissa_bits / 8)
        # => mantissa_bits = x * 2^6 * 8 = x * 512
        mant_fp8 = int(round(x * 512))
        if mant_fp8 == 0 or mant_fp8 >= 8:
            return sign << 7  # underflow to zero
        return (sign << 7) | mant_fp8
    
    # Normal numbers
    if exp_fp8 >= 0xF:
        return (sign << 7) | 0x7F  # overflow

    mant = x / (2 ** exp) - 1.0
    mant_fp8 = int(round(mant * 8))

    if mant_fp8 == 8:  # rounding overflow
        mant_fp8 = 0
        exp_fp8 += 1
        if exp_fp8 >= 0xF:
            return (sign << 7) | 0x7F

    return (sign << 7) | (exp_fp8 << 3) | mant_fp8

async def reset_accumulator(dut):
    dut.a_in.value = 0
    dut.b_in.value = 0
    dut.clear.value = 1
    await RisingEdge(dut.clk)
    dut.clear.value = 0

# ---------- Cocotb tests --- -------------------------------
@cocotb.test()
async def test_pe_deviation(dut):
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())
    dut.rst.value = 1
    dut.clear.value = 1
    dut.a_in.value = 0
    dut.b_in.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    NUM_TESTS = 100
    random.seed(42)

    for i in range(NUM_TESTS):
        fa = random.uniform(-10.0, 10.0)
        fb = random.uniform(-10.0, 10.0)
        expected = fa * fb

        fp8_a = fp8_e4m3_encode(fa)
        fp8_b = fp8_e4m3_encode(fb)

        await reset_accumulator(dut)
        
        dut.a_in.value = fp8_a
        dut.b_in.value = fp8_b
        await RisingEdge(dut.clk)

        bf16_raw = int(dut.c_out.value) & 0xFFFF
        got_float = bf16_to_float(bf16_raw)

        abs_err = abs(got_float - expected)
        rel_err = abs_err / abs(expected) if expected != 0 else float('inf')

        assert rel_err < 0.1, f"Test {i} failed: rel_err={rel_err:.6e}"

        dut._log.info(
            f"[{i}] a={fa:.6f}  b={fb:.6f}  expected={expected:.6f}  "
            f"bf16={got_float:.6f}  abs_err={abs_err:.6e}  rel_err={rel_err:.6e}"
        )
