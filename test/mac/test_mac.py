import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
import struct
import math

def float_to_fp8_e4m3(f: float) -> int:
    """Converts a standard Python float (f32) to an 8-bit FP8 E4M3 integer."""
    if f == 0.0: return 0x00
    if f < 0.0: sign, f = 1, -f
    else: sign = 0
    if f > 240.0: return (sign << 7) | 0x7E
    
    f_32_bits = struct.unpack('<I', struct.pack('<f', f))[0]
    e_f32 = (f_32_bits >> 23) & 0xFF
    m_f32 = f_32_bits & 0x7FFFFF
    e_unbiased = e_f32 - 127
    e_fp8_biased = e_unbiased + 7

    if e_unbiased < -6: return 0x00
    
    if 1 <= e_fp8_biased <= 14:
        m_fp8 = m_f32 >> 20
        if (m_f32 >> 19) & 0x1: m_fp8 += 1
        return (sign << 7) | (e_fp8_biased << 3) | (m_fp8 & 0x7)
    elif e_fp8_biased >= 15:
        return (sign << 7) | 0x7E
    return 0x00

def bf16_to_float(bf16_int: int) -> float:
    """Converts a 16-bit BF16 integer to a standard Python float (f32)."""
    f32_int = bf16_int << 16
    return struct.unpack('<f', struct.pack('<I', f32_int))[0]


@cocotb.test()
async def test_mac_fp8_bf16(dut):
    """Test the PE MAC for FP8 multiplication and BF16 accumulation."""
    dut._log.info("🚀 Starting FP8 MAC test with BF16 Accumulator")
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # --- Reset ---
    dut.rst.value = 1
    dut.clear.value = 0
    dut.a_in.value = 0
    dut.b_in.value = 0
    await Timer(1, units='ns')  # Let values settle
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    dut._log.info("Reset complete.")

    # --- Test Vectors ---
    test_vectors = [
        (2.5, 3.2),     # P0: 8.00
        (4.0, 2.25),    # P1: 9.00
        (-1.125, 7.0),  # P2: -7.875
        (3.0, -2.0),    # P3: -6.00
    ]
    
    expected_acc_float = sum(a * b for a, b in test_vectors)
    
    # --- Feed inputs with proper timing ---
    for i, (a_float, b_float) in enumerate(test_vectors):
        a_fp8 = float_to_fp8_e4m3(a_float)
        b_fp8 = float_to_fp8_e4m3(b_float)
        
        # Apply inputs at beginning of cycle
        dut.a_in.value = int(a_fp8)
        dut.b_in.value = int(b_fp8)
        dut.clear.value = 1 if i == 0 else 0
        
        # Wait for values to settle
        await Timer(1, units='ns')
        
        # Verify what we set
        readback_a = int(dut.a_in.value)
        readback_b = int(dut.b_in.value)
        
        dut._log.info(f"P{i}: {a_float} × {b_float} = {a_float*b_float:.4f}")
        dut._log.info(f"    Set:  a=0x{a_fp8:02x} b=0x{b_fp8:02x} clear={1 if i==0 else 0}")
        dut._log.info(f"    Read: a=0x{readback_a:02x} b=0x{readback_b:02x}")
        
        if readback_a != a_fp8 or readback_b != b_fp8:
            dut._log.error(f"    ❌ INPUT MISMATCH!")
            assert False, "Input values not applied correctly"
        
        # Clock the values in
        await RisingEdge(dut.clk)
        
        # After clock edge, check accumulated output from PREVIOUS cycles
        if i > 0:
            await Timer(1, units='ns')  # Let outputs settle
            
            c_out_int = int(dut.c_out.value)
            c_out_float = bf16_to_float(c_out_int)
            current_expected = sum(v[0] * v[1] for v in test_vectors[:i])
            
            tolerance = 0.2
            diff = abs(c_out_float - current_expected)
            
            if diff <= tolerance:
                dut._log.info(f"    ✅ ACC after P{i-1}: {c_out_float:.4f} (exp: {current_expected:.4f})")
            else:
                dut._log.error(f"    ❌ ACC after P{i-1}: {c_out_float:.4f} (exp: {current_expected:.4f}, diff: {diff:.4f})")
                dut._log.error(f"       c_out = 0x{c_out_int:04x}")
                assert False, f"Mismatch after P{i-1}"
    
    # Final cycle to get last result
    await RisingEdge(dut.clk)
    await Timer(1, units='ns')
    
    c_out_int = int(dut.c_out.value)
    c_out_float = bf16_to_float(c_out_int)
    
    tolerance = 0.5
    diff = abs(c_out_float - expected_acc_float)
    
    if diff <= tolerance:
        dut._log.info(f"🎉 FINAL: {c_out_float:.4f} (expected: {expected_acc_float:.4f})")
        dut._log.info("✅ All tests PASSED!")
    else:
        dut._log.error(f"❌ FINAL: {c_out_float:.4f} (expected: {expected_acc_float:.4f}, diff: {diff:.4f})")
        dut._log.error(f"   c_out = 0x{c_out_int:04x}")
        assert False, "Final accumulation mismatch"