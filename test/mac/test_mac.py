import cocotb
from cocotb.triggers import RisingEdge, ClockCycles
from cocotb.clock import Clock
import random
import math
import matplotlib.pyplot as plt
import numpy as np
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
        await RisingEdge(dut.clk) # one cycle if output is assigned combinationally (testbench has reg input)
        await RisingEdge(dut.clk) # two cycles if output is registered inside PE

        bf16_raw = int(dut.c_out.value) & 0xFFFF
        got_float = bf16_to_float(bf16_raw)

        abs_err = abs(got_float - expected)
        rel_err = abs_err / abs(expected) if expected != 0 else float('inf')

        assert rel_err < 0.1, f"Test {i} failed: rel_err={rel_err:.6e}"

        dut._log.info(
            f"[{i}] a={fa:.6f}  b={fb:.6f}  expected={expected:.6f}  "
            f"bf16={got_float:.6f}  abs_err={abs_err:.6e}  rel_err={rel_err:.6e}"
        )

async def perform_multiplication(dut, fa: float, fb: float):
    expected = fa * fb

    fp8_a = fp8_e4m3_encode(fa)
    fp8_b = fp8_e4m3_encode(fb)

    await reset_accumulator(dut)
    
    dut.a_in.value = fp8_a
    dut.b_in.value = fp8_b
    await RisingEdge(dut.clk) # one cycle if output is assigned combinationally
    await RisingEdge(dut.clk) # two cycles if output is registered inside PE

    bf16_raw = int(dut.c_out.value) & 0xFFFF
    got_float = bf16_to_float(bf16_raw)

    abs_err = abs(got_float - expected)
    # Using relative error or a variant of it (like ULP) is often better for floats, 
    # but based on the previous test and request for 'error', we'll use Absolute Error for the heatmap.
    return abs_err

# Function to setup the clock and reset for the tests
async def setup_dut(dut):
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

@cocotb.test()
async def test_pe_error_heatmap(dut):
    await setup_dut(dut)

    # --- Plotting parameters ---
    MIN_VAL = -10.0
    MAX_VAL = 10.0
    NUM_STEPS = 100 # Increase for finer resolution, decrease for faster test
    
    a_values = np.linspace(MIN_VAL, MAX_VAL, NUM_STEPS)
    b_values = np.linspace(MIN_VAL, MAX_VAL, NUM_STEPS)

    epsilon_grid = 1e-6
    
    # Initialize the error matrix
    error_matrix = np.zeros((NUM_STEPS, NUM_STEPS))

    dut._log.info(f"Starting heatmap generation with {NUM_STEPS*NUM_STEPS} points...")
    
    # --- Data Collection Loop ---
    for i in range(NUM_STEPS):
        fa = a_values[i]
        for j in range(NUM_STEPS):
            fb = b_values[j]
            
            # The perform_multiplication function now handles the RTL interaction
            abs_err = await perform_multiplication(dut, fa, fb)

            expected = fa * fb

            if abs(expected) < epsilon_grid:
                # If the product is essentially zero, relative error is undefined/infinite.
                # We'll assign a max error value for plotting purposes.
                rel_err = 1.0 # Max expected relative error
            else:
                rel_err = abs_err / abs(expected)

            error_matrix[i, j] = rel_err

    dut._log.info("Data collection complete. Generating plot...")

    # --- Matplotlib Plotting ---
    
    # Create the X and Y meshgrid for the plot
    X, Y = np.meshgrid(a_values, b_values)
    
    # Create the figure and axes
    plt.figure(figsize=(10, 8))
    
    # Plot the heatmap (using imshow for 2D array)
    epsilon = 1e-15
    log_error_matrix = error_matrix + epsilon
    
    # Plot the log10 of the absolute error
    plt.imshow(log_error_matrix, origin='lower', aspect='auto', 
               extent=[MIN_VAL, MAX_VAL, MIN_VAL, MAX_VAL], 
               cmap='viridis') # 'viridis' or 'inferno' are good choices

    # Add a color bar to show the error scale
    cbar = plt.colorbar()
    cbar.set_label('Relative Error') # 
    
    # Add labels and title
    plt.xlabel('Multiplicand A (a_in)')
    plt.ylabel('Multiplicand B (b_in)')
    plt.title(f'FP8 E4M3 Multiplication Relative Error Heatmap ({NUM_STEPS}x{NUM_STEPS} Grid)')
    
    # Save the plot
    plot_filename = "fp8_multiplication_error_heatmap.png"
    plt.savefig(plot_filename)
    dut._log.info(f"Error heatmap saved to: {plot_filename}")

    # Optionally, you can assert that the max error is below a certain threshold
    max_err = np.max(error_matrix)
    dut._log.info(f"Maximum absolute error in grid: {max_err:.6e}")
    # assert max_err < 1e-2, f"Max absolute error {max_err} exceeds threshold"