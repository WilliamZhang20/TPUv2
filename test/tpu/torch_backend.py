import torch
from test_tpu import matmul, reset_dut
from torch._inductor.compile_fx import compile_fx
from torch._dynamo import register_backend
from torch.library import custom_op, register_fake
from torch import Tensor
from typing import Optional
from cocotb.triggers import RisingEdge
import asyncio
import cocotb
import concurrent
import torch.fx

dut = None  # Global variable to hold the DUT reference

@custom_op("tpu::matmul_fp8", mutates_args=())
def tpu_matmul_fp8(a_fp8: Tensor, b_fp8: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    FP8 E4M3 matrix multiplication on TPU hardware.
    
    The hardware:
    - Accepts FP8 E4M3 (float8_e4m3fn) inputs
    - Performs 2x2 tile accumulation in BF16
    - Returns FP32 output
    
    Args:
        a_fp8: Input tensor in FP8 E4M3 format (or will be converted)
        b_fp8: Weight tensor in FP8 E4M3 format (or will be converted)
        bias: Optional bias tensor (FP32)
    
    Returns:
        FP32 output tensor
    """
    
    # Convert to FP8 E4M3 if not already
    if a_fp8.dtype != torch.float8_e4m3fn:
        a_fp8 = a_fp8.to(torch.float8_e4m3fn)
    if b_fp8.dtype != torch.float8_e4m3fn:
        b_fp8 = b_fp8.to(torch.float8_e4m3fn)
    
    # Call TPU hardware with FP8 matrices
    # Hardware performs: matmul in FP8 -> BF16 accumulation (2x2 tiles) -> FP32 output
    future = concurrent.futures.Future()
    async def wrapper():
        try:
            await reset_dut(dut)
            # matmul expects FP8 E4M3 tensors, returns FP32
            result_fp32 = await matmul(dut, a_fp8, b_fp8, 
                                       transpose=True, is_torch=True)
            future.set_result(result_fp32)
        except Exception as e:
            future.set_exception(e)
    cocotb.start_soon(wrapper())
    result_fp32 = future.result()
    
    # Add bias
    if bias is not None:
        result_fp32 = result_fp32 + bias
        
    return result_fp32

@register_fake("tpu::matmul_fp8")
def tpu_matmul_fp8_abstract(a: Tensor, b: Tensor, bias: Optional[Tensor] = None, 
                            *args, **kwargs) -> Tensor:
    """
    Abstract/fake implementation for shape inference.
    
    Args:
        a: (..., K) input features
        b: (N, K) weight matrix (will be transposed)
        
    Returns:
        (..., N) output in FP32
    """
    N = b.shape[0]
    out_shape = list(a.shape[:-1]) + [N]
    out = torch.zeros(out_shape, dtype=torch.float32, device=a.device)
    
    if bias is not None:
        out = out + bias
    return out

def make_backend(dut_arg):
    global dut
    dut = dut_arg  # Set global DUT

    @register_backend(name=f"tpu_net")
    def _backend(gm: torch.fx.GraphModule, example_inputs):
        print("\n=== FX graph received ===")
        gm.graph.print_tabular()

        # Find and replace torchao Float8 linear operations
        # The pattern in the graph is complex because torchao does internal casting
        # We need to decompose the trampoline_autograd_apply and inject our TPU op
        
        # Import the actual Float8 matmul function to replace it
        from torchao.float8.float8_linear import _ToFloat8ConstrFunc
        
        for node in list(gm.graph.nodes):
            # Look for the torchao autograd apply function
            if (node.op == 'call_function' and 
                hasattr(node.target, '__name__') and
                'trampoline_autograd_apply' in node.target.__name__):
                
                print(f"\nFound Float8 linear operation: {node.name}")
                print(f"  Args: {[arg.name if isinstance(arg, torch.fx.Node) else arg for arg in node.args]}")
                
                # Extract input (arg 0), transposed weight (arg 1), and configs
                if len(node.args) >= 4:
                    input_node = node.args[0]
                    weight_t_node = node.args[1]
                    linear_mm_config = node.args[2]
                    float8_config = node.args[3]
                    
                    # Find the bias add operation that follows
                    bias_node = None
                    add_node = None
                    for user in list(node.users):
                        if user.op == 'call_function' and user.target in (torch.ops.aten.add.default, torch.ops.aten.add.Tensor):
                            # This is the bias add
                            add_node = user
                            # The bias is the other argument (not the linear output)
                            for arg in user.args:
                                if arg != node and isinstance(arg, torch.fx.Node):
                                    bias_node = arg
                            break
                    
                    # Find the original weight (before transpose)
                    weight_node = None
                    if (isinstance(weight_t_node, torch.fx.Node) and
                        weight_t_node.op == 'call_method' and 
                        weight_t_node.target == 't' and 
                        len(weight_t_node.args) > 0):
                        weight_node = weight_t_node.args[0]
                    
                    if weight_node is not None:
                        print(f"  Input: {input_node.name}")
                        print(f"  Weight: {weight_node.name}")
                        print(f"  Bias: {bias_node.name if bias_node else None}")
                        
                        # Replace with TPU FP8 matmul wrapped in FP8 conversion
                        with gm.graph.inserting_before(node):
                            # We need to cast FP32 inputs to FP8 before calling TPU
                            # Get cast configs from float8_config
                            input_cast_config = None
                            weight_cast_config = None
                            
                            # The float8_config node should have cast_config_input and cast_config_weight
                            for config_node in gm.graph.nodes:
                                if config_node == float8_config:
                                    # Look at the kwargs
                                    if 'cast_config_input' in config_node.kwargs:
                                        input_cast_config = config_node.kwargs['cast_config_input']
                                    if 'cast_config_weight' in config_node.kwargs:
                                        weight_cast_config = config_node.kwargs['cast_config_weight']
                            
                            # Insert FP8 casting operations
                            # Cast input to FP8 E4M3
                            input_fp8 = gm.graph.call_function(
                                torch.ops.aten._to_copy.default,
                                args=(input_node,),
                                kwargs={'dtype': torch.float8_e4m3fn}
                            )
                            
                            # Cast weight to FP8 E4M3 (weight is already a parameter)
                            weight_fp8 = gm.graph.call_function(
                                torch.ops.aten._to_copy.default,
                                args=(weight_node,),
                                kwargs={'dtype': torch.float8_e4m3fn}
                            )
                            
                            # Create TPU matmul call with FP8 inputs
                            tpu_node = gm.graph.call_function(
                                torch.ops.tpu.matmul_fp8,
                                args=(input_fp8, weight_fp8, bias_node),
                            )
                            
                            print(f"  Replacing with: {tpu_node.name}")
                            
                            # Replace uses
                            if add_node is not None:
                                # Replace the add node (which includes bias)
                                add_node.replace_all_uses_with(tpu_node)
                                gm.graph.erase_node(add_node)
                            else:
                                # No bias, just replace the linear
                                node.replace_all_uses_with(tpu_node)
                            
                            # Clean up the original linear node
                            if len(list(node.users)) == 0:
                                gm.graph.erase_node(node)
                            
                            # Clean up transpose node if no longer used
                            if isinstance(weight_t_node, torch.fx.Node) and len(list(weight_t_node.users)) == 0:
                                gm.graph.erase_node(weight_t_node)

        gm.recompile()
        print("\n=== Modified graph ===")
        gm.graph.print_tabular()

        # Let Inductor compile the rest
        return compile_fx(gm, example_inputs)

    return _backend