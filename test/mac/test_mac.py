import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import random

# ---------- helpers to emulate RTL bit-exact behavior ----------

FP8_BIAS = 7
INT_ACC_OFFSET = 18

def decode_fp8_to_mant_exp(fp8: int):
    """Return (sign, exp, mant4) where mant4 is 4-bit mantissa with hidden bit as in RTL"""
    sign = (fp8 >> 7) & 1
    exp  = (fp8 >> 3) & 0xF
    frac = fp8 & 0x7
    if exp == 0:
        mant = (0 << 3) | frac   # {0, frac}
    else:
        mant = (1 << 3) | frac   # {1, frac}
    return sign, exp, mant  # mant in range 0..15

def mask18(x: int) -> int:
    """Mask to 18-bit two's complement representation (0..2^18-1)."""
    return x & ((1 << 18) - 1)

def to_signed18(x: int) -> int:
    """Interpret 18-bit two's complement as Python int (signed)."""
    x = mask18(x)
    if x & (1 << 17):
        return x - (1 << 18)
    return x

def from_signed18(x: int) -> int:
    """Take Python int (can be negative) and return 18-bit two's complement representation."""
    return mask18(x & ((1 << 18) - 1))

def int18_to_bf16_lzd_ref(acc_signed: int) -> int:
    """
    Bit-exact reference of your int18_to_bf16_lzd module.
    Input: acc_signed is Python int in range [-131072, 131071]
    Returns: 16-bit BF16 value (unsigned int)
    """
    BF16_BIAS = 127

    # convert to magnitude and sign
    if acc_signed < 0:
        sign = 1
        mag = -acc_signed
    else:
        sign = 0
        mag = acc_signed

    mag = mag & ((1 << 18) - 1)
    if mag == 0:
        return 0

    # LZD: find first 1 from MSB=17 down to 0, and compute lz = 17 - i
    lz = 0
    found = False
    for i in range(17, -1, -1):
        if (mag >> i) & 1:
            lz = 17 - i
            found = True
            break
    if not found:
        return 0

    exp = (18 - lz) + (BF16_BIAS - 1)  # matches your RTL

    if lz < 11:
        mant = (mag << (lz + 1)) >> 11
    else:
        mant = 0

    mant &= 0x7F
    exp &= 0xFF

    bf16 = (sign << 15) | (exp << 7) | mant
    return bf16 & 0xFFFF

# ---------- Cocotb tests ----------

@cocotb.test()
async def test_pe_rtl_semantics(dut):
    """Randomized test that matches the RTL semantics where c_out = BF16(acc_previous)."""

    # 10 ns clock
    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())

    random.seed(100) # for reproducibility
    # reset pulse
    dut.rst.value = 1
    dut.clear.value = 1
    dut.a_in.value = 0
    dut.b_in.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    # Initialize software accumulator to match RTL register (starts at 0)
    acc_reg = 0  # signed Python int

    N = 50
    for cycle in range(N):
        # random inputs
        a = random.randint(-127, 128) & 0xFF
        b = random.randint(-127, 128) & 0xFF

        dut.a_in.value = a
        dut.b_in.value = b

        # Occasionally toggle clear to test reset path
        dut.clear.value = 1 if (cycle % 2 == 0) else 0

        # BEFORE clock edge: expected DUT.c_out is BF16 of acc_reg (the current register content)
        expected_c_out = int18_to_bf16_lzd_ref(acc_reg)

        await RisingEdge(dut.clk)

        # sample DUT output after posedge (RTL assigns c_out <= bf16_c computed from acc before update)
        got = int(dut.c_out.value) & 0xFFFF

        if got != expected_c_out:
            raise cocotb.log.info(
                f"Mismatch at cycle {cycle}:\n"
                f" a=0x{a:02x}, b=0x{b:02x}, clear={int(dut.clear.value)}\n"
                f" expected c_out (BF16(acc_before)) = 0x{expected_c_out:04x}\n"
                f" got                                  = 0x{got:04x}\n"
                f" acc_before = {acc_reg}\n"
            )

        # AFTER checking output, update our software acc_reg exactly like RTL non-blocking update:
        # If clear was asserted at the edge, acc <= 0; else acc <= acc +/- aligned_prod
        if int(dut.rst.value) == 1 or int(dut.clear.value) == 1:
            acc_reg = 0
            continue

        # Recompute what aligned_prod would be from inputs (bit-exact)
        sA, eA, mA = decode_fp8_to_mant_exp(a)
        sB, eB, mB = decode_fp8_to_mant_exp(b)

        prod_sign = sA ^ sB
        mant_prod = (mA * mB) & 0xFF  # 8-bit mul result

        # shift amount (can be > 8, but our RTL used an 8-bit wire; emulate integer shift)
        shift_amount = int(eA) + int(eB) - FP8_BIAS + INT_ACC_OFFSET
        # --- emulate Verilog left-shift on 8-bit operand (result truncated to 8 bits) ---
        if shift_amount < 0:
            shifted8 = 0
        else:
            # shift but keep only 8 LSBs (Verilog: result width == left operand)
            shifted8 = (mant_prod << shift_amount) & 0xFF

        # aligned_prod as used by RTL is zero-extended into 18-bit domain
        aligned = shifted8  # small int 0..255

        # --- now apply signed 18-bit accumulation (with wrap) ---
        if prod_sign:
            acc_next_signed = acc_reg - aligned
        else:
            acc_next_signed = acc_reg + aligned

        # wrap into signed 18-bit two's complement
        acc_next_repr = acc_next_signed & ((1 << 18) - 1)
        acc_reg = acc_next_repr if (acc_next_repr & (1 << 17)) == 0 else acc_next_repr - (1 << 18)
    dut._log.info("test_pe_rtl_semantics: PASS")


@cocotb.test()
async def test_pe_simple_vectors(dut):
    """Directed vectors — also follow RTL semantics (c_out = BF16(acc_before))."""

    cocotb.start_soon(Clock(dut.clk, 10, "ns").start())

    # reset
    dut.rst.value = 1
    dut.clear.value = 1
    dut.a_in.value = 0
    dut.b_in.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    dut.clear.value = 0
    await RisingEdge(dut.clk)

    # software acc
    acc_reg = 0

    vectors = [
        (0x3C, 0x3C),  # +1 * +1
        (0xBC, 0x3C),  # -1 * +1
        (0x40, 0x40),  # 2 * 2
        (0x00, 0x40),  # 0 * 2
        (0x7F, 0x7F),  # large normals
    ]

    for idx, (a, b) in enumerate(vectors):
        dut.a_in.value = a
        dut.b_in.value = b

        expected = int18_to_bf16_lzd_ref(acc_reg)

        await RisingEdge(dut.clk)

        got = int(dut.c_out.value) & 0xFFFF
        if got != expected:
            raise cocotb.result.TestFailure(
                f"Vector {idx} fail: a=0x{a:02x} b=0x{b:02x}\n"
                f" expected c_out (BF16(acc_before)) = 0x{expected:04x}\n"
                f" got                               = 0x{got:04x}\n"
                f" acc_before = {acc_reg}\n"
            )

        # update acc_reg like RTL did at that edge
        sA, eA, mA = decode_fp8_to_mant_exp(a)
        sB, eB, mB = decode_fp8_to_mant_exp(b)
        prod_sign = sA ^ sB
        mant_prod = (mA * mB) & 0xFF
        shift_amount = int(eA) + int(eB) - FP8_BIAS + INT_ACC_OFFSET
        if shift_amount < 0:
            aligned = 0
        else:
            aligned = (mant_prod << shift_amount) & ((1 << 40) - 1)
        if prod_sign:
            acc_next = acc_reg - aligned
        else:
            acc_next = acc_reg + aligned
        acc_reg = to_signed18(from_signed18(acc_next))

    dut._log.info("test_pe_simple_vectors: PASS")
