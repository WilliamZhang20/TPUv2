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
                                       transpose=True)
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
        # print("\n=== FX graph received ===")
        # gm.graph.print_tabular()

        # Find and replace torchao Float8 linear operations
        # The trampoline_autograd_apply appears as call_function nodes
        
        replaced_nodes = []
        
        for node in list(gm.graph.nodes):
            # Look for the torchao autograd apply function
            if (node.op == 'call_function' and 
                callable(node.target) and
                hasattr(node.target, '__name__') and
                'trampoline_autograd_apply' in node.target.__name__):
                
                # print(f"\n✓ Found Float8 linear operation: {node.name}")
                # print(f"  Target: {node.target.__name__}")
                # print(f"  Args: {len(node.args)}")
                
                # Extract input (arg 0), transposed weight (arg 1), and configs
                if len(node.args) >= 4:
                    input_node = node.args[0]
                    weight_t_node = node.args[1]
                    linear_mm_config = node.args[2]
                    float8_config = node.args[3]
                    
                    # print(f"  Input: {input_node.name if isinstance(input_node, torch.fx.Node) else input_node}")
                    # print(f"  Weight_t: {weight_t_node.name if isinstance(weight_t_node, torch.fx.Node) else weight_t_node}")
                    
                    # Find the bias add operation that follows
                    bias_node = None
                    add_node = None
                    for user in list(node.users):
                        # print(f"  User: {user.name} target={user.target}")
                        if user.op == 'call_function' and user.target in (torch.ops.aten.add.default, torch.ops.aten.add.Tensor):
                            # This is the bias add
                            add_node = user
                            # The bias is the other argument (not the linear output)
                            for arg in user.args:
                                if arg != node and isinstance(arg, torch.fx.Node):
                                    bias_node = arg
                                    break
                            break
                        # Also check for built-in add function
                        if callable(user.target) and user.target.__name__ == 'add':
                            add_node = user
                            for arg in user.args:
                                if arg != node and isinstance(arg, torch.fx.Node):
                                    bias_node = arg
                                    break
                            break
                    
                    # Find the original weight (before transpose)
                    weight_node = None
                    if (isinstance(weight_t_node, torch.fx.Node) and
                        weight_t_node.op == 'call_method' and 
                        weight_t_node.target == 't' and 
                        len(weight_t_node.args) > 0):
                        weight_node = weight_t_node.args[0]
                    
                    if weight_node is not None:
                        """
                        print(f"  → Input: {input_node.name}")
                        print(f"  → Weight: {weight_node.name}")
                        print(f"  → Bias: {bias_node.name if bias_node else 'None'}")
                        """
                        
                        # Insert nodes in correct order: casts first, then matmul
                        # We need to insert them one by one to maintain order
                        if bias_node is not None:
                            # Insert input cast after bias
                            with gm.graph.inserting_after(bias_node):
                                input_fp8 = gm.graph.call_function(
                                    torch.ops.aten._to_copy.default,
                                    args=(input_node,),
                                    kwargs={'dtype': torch.float8_e4m3fn}
                                )
                            
                            # Insert weight cast after input cast
                            with gm.graph.inserting_after(input_fp8):
                                weight_fp8 = gm.graph.call_function(
                                    torch.ops.aten._to_copy.default,
                                    args=(weight_node,),
                                    kwargs={'dtype': torch.float8_e4m3fn}
                                )
                            
                            # Insert matmul after weight cast
                            with gm.graph.inserting_after(weight_fp8):
                                tpu_node = gm.graph.call_function(
                                    torch.ops.tpu.matmul_fp8,
                                    args=(input_fp8, weight_fp8, bias_node),
                                )
                        else:
                            # No bias case - insert before the node that uses the result
                            with gm.graph.inserting_before(node):
                                input_fp8 = gm.graph.call_function(
                                    torch.ops.aten._to_copy.default,
                                    args=(input_node,),
                                    kwargs={'dtype': torch.float8_e4m3fn}
                                )
                                
                                weight_fp8 = gm.graph.call_function(
                                    torch.ops.aten._to_copy.default,
                                    args=(weight_node,),
                                    kwargs={'dtype': torch.float8_e4m3fn}
                                )
                                
                                tpu_node = gm.graph.call_function(
                                    torch.ops.tpu.matmul_fp8,
                                    args=(input_fp8, weight_fp8, None),
                                )
                        
                        # print(f"  ✓ Created TPU node: {tpu_node.name}")
                        
                        # Replace uses
                        if add_node is not None:
                            # Replace the add node (which includes bias)
                            # print(f"  ✓ Replacing add node: {add_node.name}")
                            add_node.replace_all_uses_with(tpu_node)
                            replaced_nodes.append(add_node)
                        else:
                            # No bias, just replace the linear
                            # print(f"  ✓ Replacing linear node directly")
                            node.replace_all_uses_with(tpu_node)
                        
                        # Mark nodes for deletion
                        replaced_nodes.append(node)
                        if isinstance(weight_t_node, torch.fx.Node):
                            replaced_nodes.append(weight_t_node)
        
        # Clean up replaced nodes
        for node in replaced_nodes:
            if len(list(node.users)) == 0:
                # print(f"  Erasing: {node.name}")
                gm.graph.erase_node(node)

        gm.recompile()
        """
        print("\n=== Modified graph ===")
        gm.graph.print_tabular()
        """
        # Let Inductor compile the rest
        return compile_fx(gm, example_inputs)

    return _backend