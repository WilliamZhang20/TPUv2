import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
from cocotb.utils import get_sim_time
import math
import struct
import itertools

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

async def parallel_load_read(dut, A, B, instr=(0, 0, 0), next_instr=(0, 0, 0)):
    results = []
    hadamard, transpose, relu = instr
    next_hadamard, next_transpose, next_relu = next_instr
    dut.uio_in.value = (1 << 4) | (hadamard << 3) | (transpose << 1) | (relu << 2) | 1
    cycle = 0

    for inputs in [A, B]:
        for i in range(2):
            cycle += 1
            idx0 = i * 2
            idx1 = i * 2 + 1
            if cycle == 3:
                dut.uio_in.value = (1 << 4) | (next_hadamard << 3) | (next_transpose << 1) | (next_relu << 2) | 1
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

    for i in range(4):
        rel_err = abs(results[i] - expected[i]) / abs(expected[i])
        assert rel_err <= 0.12, (
            f"C[{i//2}][{i%2}] = {results[i]} "
            f"!= expected {expected[i]} (relative error {rel_err:.4f})"
        )
    dut._log.info("Test 1 passed")

    expected = get_expected_output(A, B)

    A = [5, -6, 7, 8]  # row-major
    B = [8, 9, 6, 8]  # row-major: [B00, B01, B10, B11]

    results = await parallel_load_read(dut, A, B, next_instr=(0, 1, 1))

    for i in range(4):
        rel_err = abs(results[i] - expected[i]) / abs(expected[i])
        assert rel_err <= 0.12, (
            f"C[{i//2}][{i%2}] = {results[i]} "
            f"!= expected {expected[i]} (relative error {rel_err:.4f})"
        )
    dut._log.info("Test 2 passed")

    expected = get_expected_output(A, B, transpose=True, relu=True)
    results = await parallel_load_read(dut, [], [], instr=(0, 1, 1))

    for i in range(4):
        assert results[i] == expected[i], f"C[{i//2}][{i%2}] = {results[i]} != expected {expected[i]}"

    dut._log.info("ReLU + Transpose test passed!")

async def load_stationary_weights(dut, weights):
    """Load weights in stationary mode (stat_weights=1, load_weights=1)"""
    for i in range(4):
        dut.ui_in.value = fp8_e4m3_encode(weights[i])
        # stat_weights=1 (bit 5), load_weights=1 (bit 6), enable=1 (bit 4), load_en=1 (bit 0)
        dut.uio_in.value = (1 << 6) | (1 << 5) | (1 << 4) | 1
        await RisingEdge(dut.clk)

async def load_inputs_stationary(dut, inputs):
    """Load inputs when using stationary weights (stat_weights=1, load_weights=0)"""
    for i in range(4):
        dut.ui_in.value = fp8_e4m3_encode(inputs[i])
        # stat_weights=1 (bit 5), load_weights=0, enable=1 (bit 4), load_en=1 (bit 0)
        dut.uio_in.value = (1 << 5) | (1 << 4) | 1
        await RisingEdge(dut.clk)

async def parallel_rw_stationary(dut, inputs, load_weights=0):
    # stat_weights=1 (bit 5), enable=1 (bit 4)
    dut.uio_in.value = (load_weights << 6) | (1 << 5) | (1 << 4) | 1
    results = []
    
    for i in range(4):
        dut.ui_in.value = fp8_e4m3_encode(inputs[i])
        await RisingEdge(dut.clk)
        high = dut.uo_out.value.integer
        low = dut.uio_out.value.integer
        combined = (high << 8) | low
        float_val = bf16_to_float(combined)
        results.append(float_val)
    
    return results

@cocotb.test()
async def test_stationary_weights(dut):
    """Test stationary weights mode: load weights once, reuse for multiple input matrices"""
    dut._log.info("Testing stationary weights mode")
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    await reset_dut(dut)

    # Define stationary weights 
    weights = [2.0, 1.0, 0.5, 3.0] 
    inputs0 = [1.0, 2.0, 3.0, 4.0]
    
    # Load weights once
    dut._log.info("Loading stationary weights")
    await load_stationary_weights(dut, weights)
    await load_inputs_stationary(dut, inputs0)
    
    # Test with multiple different input matrices
    test_inputs = [
        [1.0, -2.0, 3.0, 4.0],   
        [0.5, 1.5, 2.5, 3.5], 
        [-1.0, 2.0, -3.0, 4.0],
    ]
    
    for idx, inputs in enumerate(test_inputs):
        dut._log.info(f"Testing with input matrix {idx + 1}")
        
        # Read results
        results = await parallel_rw_stationary(dut, test_inputs[idx])
        
        # Calculate expected output: weights @ inputs
        expected = get_expected_output(weights, inputs0 if idx == 0 else test_inputs[idx-1])
        
        dut._log.info(f"Results:  {results}")
        dut._log.info(f"Expected: {expected}")
        
        # Verify results with tolerance
        for i in range(4):
            if expected[i] == 0:
                # For zero expected values, check absolute error
                abs_err = abs(results[i] - expected[i])
                assert abs_err <= 0.5, (
                    f"Input set {idx + 1}: C[{i//2}][{i%2}] = {results[i]} "
                    f"!= expected {expected[i]} (absolute error {abs_err:.4f})"
                )
            else:
                # For non-zero values, check relative error
                rel_err = abs(results[i] - expected[i]) / abs(expected[i])
                assert rel_err <= 0.12, (
                    f"Input set {idx + 1}: C[{i//2}][{i%2}] = {results[i]} "
                    f"!= expected {expected[i]} (relative error {rel_err:.4f})"
                )
        
        dut._log.info(f"Input matrix {idx + 1} passed")

    results = await parallel_rw_stationary(dut, [0, 0, 0, 0])
    expected = get_expected_output(weights, test_inputs[-1])

    for i in range(4):
        if expected[i] == 0:
            # For zero expected values, check absolute error
            abs_err = abs(results[i] - expected[i])
            assert abs_err <= 0.5, (
                f"Input set {idx + 1}: C[{i//2}][{i%2}] = {results[i]} "
                f"!= expected {expected[i]} (absolute error {abs_err:.4f})"
            )
        else:
            # For non-zero values, check relative error
            rel_err = abs(results[i] - expected[i]) / abs(expected[i])
            assert rel_err <= 0.12, (
                f"Input set {idx + 1}: C[{i//2}][{i%2}] = {results[i]} "
                f"!= expected {expected[i]} (relative error {rel_err:.4f})"
            )

    dut._log.info("All stationary weights tests passed!")

async def accumulate_matrix_output(dut, results_large, i, j, transpose=0, A_block=None, B_block=None):
    """
    Serially loads A_block and B_block (1 value per cycle),
    and reads interleaved output (1 byte per cycle: high, low, high, low, ...).
    Accumulates output into results_large[i:i+2, j:j+2].
    """
    # Full interleaved stream of 8 input values: A0-A3, then B0-B3
     # Prepare input stream (A then B), or zeros if flushing
    if A_block is not None and B_block is not None:
        input_stream = (
            [fp8_e4m3_encode(x) for x in A_block] +
            [fp8_e4m3_encode(x) for x in B_block]
        )
    else:
        input_stream = [0] * 8

    dut.uio_in.value = (1 << 4) | (transpose << 1) | 1  # load_en=1

    partial_outputs = []

    # 8 cycles: input + output interleaved
    for idx in range(8):
        dut.ui_in.value = input_stream[idx]
        await ClockCycles(dut.clk, 1)
        partial_outputs.append(dut.uo_out.value.integer)

    # Decode BF16 outputs → float
    combined_outputs = []
    for ii in range(0, 8, 2):
        high = partial_outputs[ii]
        low  = partial_outputs[ii + 1]
        bf16 = (high << 8) | low
        float_val = bf16_to_float(bf16)
        combined_outputs.append(float_val)

    # Accumulate into output matrix (floating point)
    results_large[i,   j  ] += combined_outputs[0]  # C00
    results_large[i,   j+1] += combined_outputs[1]  # C01
    results_large[i+1, j  ] += combined_outputs[2]  # C10
    results_large[i+1, j+1] += combined_outputs[3]  # C11

    return combined_outputs

async def matmul(dut, A, B, transpose=False, relu=False):
    import torch
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

    A_padded = torch.zeros((m_p, n_p), dtype=torch.float32)
    B_padded = torch.zeros((n_bp, p_p), dtype=torch.float32)
    
    A_padded[:m, :n] = A
    B_padded[:n_b, :p] = B
    results_large = torch.zeros((m_p, n_bp), dtype=torch.float32) if transpose else torch.zeros((m_p, p_p), dtype=torch.float32)

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
        results_large = torch.maximum(results_large, torch.tensor(0.0))

    return results_large[:m, :n_b] if transpose else results_large[:m, :p]

async def matmul_faster(dut, A, B, transpose=False, relu=False):
    """
    True dyadic A-stationary matmul.
    Reuses stationary A tiles across all j tiles.
    Legal and faster when n <= 2.
    """
    import torch

    m, n = A.shape
    n_b, p = B.shape

    if transpose:
        assert n == p
    else:
        assert n == n_b

    # Fallback if we cannot exploit stationarity
    if n > 2:
        return await matmul(dut, A, B, transpose=transpose, relu=relu)

    # ---- Padding ----
    m_p = ((m + 1) // 2) * 2
    p_p = ((p + 1) // 2) * 2

    A_p = torch.zeros((m_p, 2), dtype=torch.float32)
    B_p = torch.zeros((2, p_p), dtype=torch.float32)

    A_p[:m, :n] = A
    B_p[:n, :p] = B

    C = torch.zeros((m_p, p_p), dtype=torch.float32)

    # ---- Dyadic schedule ----
    for i in range(0, m_p, 2):
        # A(i,k) is stationary for entire j sweep
        A_block = A_p[i:i+2, :2].flatten().tolist()
        await load_stationary_weights(dut, A_block)

        # ---- Prime pipeline with first B tile ----
        j0 = 0
        B_block = B_p[:2, j0:j0+2].flatten().tolist()
        await load_inputs_stationary(dut, B_block)

        # ---- Stream remaining B tiles ----
        for j in range(2, p_p, 2):
            B_next = B_p[:2, j:j+2].flatten().tolist()
            results = await parallel_rw_stationary(dut, B_next)

            C[i,   j-2] += results[0]
            C[i,   j-1] += results[1]
            C[i+1, j-2] += results[2]
            C[i+1, j-1] += results[3]

        # ---- Drain final output ----
        results = await parallel_rw_stationary(dut, [0, 0, 0, 0])

        C[i,   p_p-2] += results[0]
        C[i,   p_p-1] += results[1]
        C[i+1, p_p-2] += results[2]
        C[i+1, p_p-1] += results[3]

    if relu:
        C = torch.maximum(C, torch.tensor(0.0))

    return C[:m, :p]

@cocotb.test()
async def test_matmul_faster_matches_reference(dut):
    import torch
    dut._log.info("Testing matmul_faster vs reference matmul")

    # Clock
    clock = Clock(dut.clk, 20, units="ns")
    cocotb.start_soon(clock.start())

    # Reset
    await reset_dut(dut)

    # Test configurations
    test_cases = [
        # (m, n, p, transpose, relu)
        (4, 4, 4, False, False),
        (5, 3, 6, False, False),
        (6, 5, 3, False, True),
        (4, 6, 5, True, False),
        (7, 7, 7, True, True),
        (10, 10, 10, False, False),
        (20, 20, 20, False, False)
    ]

    torch.manual_seed(0)
    
    for idx, (m, n, p, transpose, relu) in enumerate(test_cases):
        dut._log.info(
            f"Case {idx+1}: A={m}x{n}, B={n}x{p}, "
            f"transpose={transpose}, relu={relu}"
        )

        # Generate random inputs (FP8-safe range)
        A = (torch.rand(m, n) * 6.0 - 3.0).float()
        if transpose:
            B = (torch.rand(p, n) * 6.0 - 3.0).float()
        else:
            B = (torch.rand(n, p) * 6.0 - 3.0).float()

        t0 = get_sim_time(units="ns")
        ref_out = await matmul(
            dut,
            A,
            B,
            transpose=transpose,
            relu=relu,
        )
        t1 = get_sim_time(units="ns")
        ref_time = t1 - t0

        # Reset DUT to avoid state contamination
        await reset_dut(dut)

        t2 = get_sim_time(units="ns")
        fast_out = await matmul_faster(
            dut,
            A,
            B,
            transpose=transpose,
            relu=relu,
        )
        t3 = get_sim_time(units="ns")
        fast_time = t3 - t2

        ref_out = ref_out.cpu()
        fast_out = fast_out.cpu()

        assert ref_out.shape == fast_out.shape

        for i in range(ref_out.shape[0]):
            for j in range(ref_out.shape[1]):
                ref_val = ref_out[i, j].item()
                fast_val = fast_out[i, j].item()
                assert ref_val == fast_val, (
                    f"Mismatch at ({i},{j}): ref={ref_val}, fast={fast_val}"
                )

        speedup = ref_time / fast_time if fast_time > 0 else float("inf")

        dut._log.info(
            f"Case {idx+1} timing: "
            f"ref={ref_time:.0f} ns, "
            f"fast={fast_time:.0f} ns, "
            f"speedup={speedup:.2f}×"
        )
        
        dut._log.info(f"Case {idx+1} passed ✔")

    dut._log.info("All matmul_faster correctness + benchmark tests passed 🎉")