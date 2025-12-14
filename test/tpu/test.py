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

def get_expected_output(A, B, transpose=False, hadamard=False, relu=False):
    A_mat = np.array(A).reshape(2, 2)
    B_mat = np.array(B).reshape(2, 2)
    if transpose:
        B_mat = B_mat.T
    if hadamard:
        result = np.multiply(A_mat, B_mat)
    else:
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

async def load_matrix(dut, matrix, hadamard=0, transpose=0, relu=0):
    for i in range(4):
        dut.ui_in.value = fp8_e4m3_encode(matrix[i])
        dut.uio_in.value = (1 << 4) | (hadamard << 3) | (transpose << 1) | (relu << 2) | 1
        await RisingEdge(dut.clk)

async def read_output(dut, hadamard=0):
    dut.uio_in.value = (1 << 4) | (hadamard << 3) | 0
    results = []
    for _ in range(4):
        await RisingEdge(dut.clk)
        high = dut.uo_out.value.integer
        await RisingEdge(dut.clk)
        low = dut.uo_out.value.integer
        combined = (high << 8) | low
        float_val = bf16_to_float(combined)
        results.append(float_val)
    return results

async def parallel_load_read(dut, A, B, hadamard=0, transpose=0, relu=0):
    results = []
    dut.uio_in.value = (1 << 4) | (hadamard << 3) | (transpose << 1) | (relu << 2) | 1
    
    for inputs in [A, B]:
        for i in range(2):
            idx0 = i * 2
            idx1 = i * 2 + 1
            # Feed either real data or dummy zeros
            dut.ui_in.value = fp8_e4m3_encode(inputs[idx0]) if inputs else 0
            await ClockCycles(dut.clk, 1)
            high = dut.uo_out.value.integer
            
            dut.ui_in.value = fp8_e4m3_encode(inputs[idx1]) if inputs else 0
            await ClockCycles(dut.clk, 1)
            low = dut.uo_out.value.integer

            combined = (high << 8) | low
            float_val = bf16_to_float(combined)

            results.append(float_val)
    return results

@cocotb.test()
async def test_hadamard(dut):
    dut._log.info("Start")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    await reset_dut(dut)

    A = [1, 2, 3, 4]  # row-major
    B = [5, 6, 7, 8]  # row-major: [B00, B01, B10, B11]

    await load_matrix(dut, A, hadamard=1)
    await load_matrix(dut, B, hadamard=1)

    # ------------------------------
    # STEP 4: Read outputs
    expected = get_expected_output(A, B, hadamard=True)
    results = []

    # Read test 1 matrices
    results = await read_output(dut, hadamard=1)

    print(results)
    print(expected)
    for i in range(4):
        rel_err = abs(results[i] - expected[i]) / abs(expected[i])
        assert rel_err <= 0.12, (
            f"C[{i//2}][{i%2}] = {results[i]} "
            f"!= expected {expected[i]} (relative error {rel_err:.4f})"
        )
    dut._log.info("Elementwise product passed")

@cocotb.test()
async def test_gemm(dut):
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
    expected = get_expected_output(A, B)
    results = []

    A = [7.9, -10, 3.5, 8]  # row-major
    B = [2.8, 6.4, 5.3, 1.2]  # row-major: [B00, B01, B10, B11]

    # Read test 1 matrices
    results = await parallel_load_read(dut, A, B)

    print(results)
    print(expected)
    for i in range(4):
        rel_err = abs(results[i] - expected[i]) / abs(expected[i])
        assert rel_err <= 0.12, (
            f"C[{i//2}][{i%2}] = {results[i]} "
            f"!= expected {expected[i]} (relative error {rel_err:.4f})"
        )
    dut._log.info("Test 1 passed")

    expected = get_expected_output(A, B)

    results = await parallel_load_read(dut, [], [])

    print(results)
    print(expected)
    for i in range(4):
        rel_err = abs(results[i] - expected[i]) / abs(expected[i])
        assert rel_err <= 0.12, (
            f"C[{i//2}][{i%2}] = {results[i]} "
            f"!= expected {expected[i]} (relative error {rel_err:.4f})"
        )
    dut._log.info("Test 2 passed")

def get_expected_large_matmul(A, B, transpose=0, relu=0):
    if transpose:
        B = B.T
    
    result = A @ B

    if relu:
        result = np.maximum(result, 0)

    return result

def check_expected(A, B, result, transpose=0, relu=0):
    """
    Check DUT results against expected matrix multiplication, for big matrices
    """
    expected = get_expected_large_matmul(A, B, transpose, relu)
    np.testing.assert_array_equal(result, expected, err_msg="Matrix multiplication result does not match expected")

async def accumulate_matrix_output(dut, results_large, i, j, transpose=0, A_block=None, B_block=None):
    """
    Serially loads A_block and B_block (1 value per cycle),
    and reads interleaved output (1 byte per cycle: high, low, high, low, ...).
    Accumulates output into results_large[i:i+2, j:j+2].
    """
    # Full interleaved stream of 8 input values: A0-A3, then B0-B3
    input_stream = (A_block + B_block) if (A_block and B_block) else [0]*8

    dut.uio_in.value = (transpose << 1) | 1  # load_en=1

    partial_outputs = []

    for idx in range(8):
        dut.ui_in.value = input_stream[idx]
        await ClockCycles(dut.clk, 1)
        val = dut.uo_out.value.integer
        partial_outputs.append(val)

    # Now decode high/low bytes
    combined_outputs = []
    for ii in range(0, 8, 2):
        high = partial_outputs[ii]
        low = partial_outputs[ii + 1]
        val = (high << 8) | low
        if val >= 0x8000:
            val -= 0x10000
        combined_outputs.append(val)

    results_large[i,   j  ] += combined_outputs[0]  # C00
    results_large[i,   j+1] += combined_outputs[1]  # C01
    results_large[i+1, j  ] += combined_outputs[2]  # C10
    results_large[i+1, j+1] += combined_outputs[3]  # C11

    return combined_outputs

async def matmul(dut, A, B, transpose=False, relu=False):
    """
    Fully pipelined systolic matrix multiplication using 2x2 blocks.
    Accumulates partial results across k dimension for each (i,j) tile.
    Loads A and B in parallel with reading previous output.
    """
    m, n = A.shape
    n_b, p = B.shape
    if (transpose):
        assert n == p, "Reminder: you are computing A*B^T"
    else:
        assert n == n_b, "Matrix dimension mismatch"

    # Pad dimensions to multiples of 2
    m_p = ((m + 1) // 2) * 2
    n_p = ((n + 1) // 2) * 2
    n_bp = ((n_b + 1) // 2) * 2
    p_p = ((p + 1) // 2) * 2

    A_padded = np.zeros((m_p, n_p), dtype=int)
    B_padded = np.zeros((n_bp, p_p), dtype=int)
    
    A_padded[:m, :n] = A
    B_padded[:n_b, :p] = B
    results_large = np.zeros((m_p, n_bp), dtype=int) if transpose else np.zeros((m_p, p_p), dtype=int)

    # Generate tile coordinates (i, j, k)
    if transpose:
        # Order: j, i, k for transpose case
        tile_coords = [
            (i, j, k)
            for i in range(0, m_p, 2)
            for j in range(0, n_bp, 2)
            for k in range(0, p_p, 2)
        ]
    else:
        # Original order: i, j, k
        tile_coords = [
            (i, j, k)
            for i in range(0, m_p, 2)
            for j in range(0, p_p, 2)
            for k in range(0, n_p, 2)
        ]

    # Step 1: Load first tile only (no output yet)
    i0, j0, k0 = tile_coords[0]
    A_block = A_padded[i0:i0+2, k0:k0+2].flatten().tolist()
    B_block = B_padded[k0:k0+2, j0:j0+2].flatten().tolist()

    await load_matrix(dut, A_block, transpose=0, relu=relu)
    await load_matrix(dut, B_block, transpose=transpose, relu=relu)

    # Step 2: Pipelined main loop
    for coord in tile_coords[1:]:
        i1, j1, k1 = coord
        A_next = A_padded[i1:i1+2, k1:k1+2].flatten().tolist()
        B_next = B_padded[j1:j1+2, k1:k1+2].flatten().tolist() if transpose else B_padded[k1:k1+2, j1:j1+2].flatten().tolist()
        # Read output from previous tile while loading next
        await accumulate_matrix_output(dut, results_large, i0, j0, transpose, A_next, B_next)

        # Slide to next
        i0, j0, k0 = i1, j1, k1
        A_block = A_next
        B_block = B_next

    # Final tile read (no further input)
    await accumulate_matrix_output(dut, results_large, i0, j0, transpose)

    # Apply ReLU if enabled
    if relu:
        results_large = np.maximum(results_large, 0)

    return results_large[:m, :n_b] if transpose else results_large[:m, :p]
