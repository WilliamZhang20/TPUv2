import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
import math
import struct

def bf16_to_float(bf16: int) -> float:
    i = bf16 << 16
    b = struct.pack(">I", i)
    return struct.unpack(">f", b)[0]

def fp8_e4m3_encode(x: float) -> int:
    if math.isnan(x):
        return 0x7F  # closest representation
    if math.isinf(x):
        return 0x7F if x > 0 else 0xFF

    sign = 0
    if x < 0:
        sign = 1
        x = -x

    if x == 0:
        return sign << 7

    exp = math.floor(math.log2(x))
    mant = x / (2 ** exp) - 1.0

    # FP8 exponent bias = 7
    exp_fp8 = exp + 7

    # Handle underflow/overflow
    if exp_fp8 <= 0:
        return sign << 7
    if exp_fp8 >= 0xF:
        return (sign << 7) | 0x7F  # max finite

    mant_fp8 = int(round(mant * 8))  # 3 bits mantissa (2^3=8)

    if mant_fp8 == 8:  # rounding overflow
        mant_fp8 = 0
        exp_fp8 += 1
        if exp_fp8 >= 0xF:
            return (sign << 7) | 0x7F

    return (sign << 7) | (exp_fp8 << 3) | mant_fp8

def get_expected_matmul(A, B, transpose=False, relu=False):
    A_mat = np.array(A).reshape(2, 2)
    B_mat = np.array(B).reshape(2, 2)
    if transpose:
        B_mat = B_mat.T
    result = A_mat @ B_mat
    if relu:
        result = np.maximum(result, 0)
    return result.flatten().tolist()

async def reset_dut(dut):
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 1)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)

async def load_matrix(dut, matrix, transpose=0, relu=0):
    for i in range(4):
        dut.ui_in.value = fp8_e4m3_encode(matrix[i])
        dut.uio_in.value = (transpose << 1) | (relu << 2) | 1
        await RisingEdge(dut.clk)

async def parallel_load_read(dut, A, B, transpose=0, relu=0):
    results = []
    dut.uio_in.value = (transpose << 1) | (relu << 2) | 1

    for inputs in [A, B]:
        for i in range(2):
            idx0 = i * 2
            idx1 = i * 2 + 1
            # Feed either real data or dummy zeros
            dut.ui_in.value = fp8_e4m3_encode(inputs[idx0]) if inputs else 0
            await ClockCycles(dut.clk, 1)
            high = dut.uo_out.value.integer
            dut._log.info(f"Read high value = {high}")

            dut.ui_in.value = fp8_e4m3_encode(inputs[idx1]) if inputs else 0
            await ClockCycles(dut.clk, 1)
            low = dut.uo_out.value.integer
            dut._log.info(f"Read low value = {low}")

            combined = (high << 8) | low
            float_val = bf16_to_float(combined)

            results.append(float_val)
    return results

@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    await reset_dut(dut)

    A = [1, 2, 3, 4]

    B = [5, 6, 7, 8]

    await load_matrix(dut, A)
    await load_matrix(dut, B)

    # ------------------------------
    # STEP 4: Read outputs
    expected = get_expected_matmul(A, B)
    results = []

    A = [7.9, -10, 3.5, 8]  # row-major
    B = [2.8, 6.4, 5.3, 1.2]  # row-major: [B00, B01, B10, B11]

    # Read test 1 matrices
    results = await parallel_load_read(dut, A, B)

    print(results)
    print(expected)
    # for i in range(4):
    #    assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    expected = get_expected_matmul(A, B)

    results = await parallel_load_read(dut, [], [])

    print(results)
    print(expected)
    """
    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"
    """
    dut._log.info("End of TEST")