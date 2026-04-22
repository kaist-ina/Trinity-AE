"""
Load/store and multi-tensor memory helpers.

Extracted from ops.py to keep responsibilities smaller and isolated.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, TYPE_CHECKING

from ...AstNode import ASTNode
from ...NodeType import NodeType

if TYPE_CHECKING:
    from ...state import CodeGenState
    from ....TritonGen import TritonCodeGen


class MemoryOps:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def generate_load(self, node: ASTNode) -> str:
        """Generate load operation"""
        # (load tensor index)
        tensor_node = node.children[0]
        index_node = node.children[1]

        tensor_name = tensor_node.children[0].value
        node.tensor_shape = self.state.tensor_shapes.get(tensor_name)
        node.tensor_shape = self.state.tensor_shapes.get(tensor_name)

        # Skip checking for temporary variables - we don't create them anymore

        # For input/output tensors and cross-kernel tensors, use normal load operation
        # Handle case where index_node is directly FULLTILE (not wrapped in INDEX)
        if index_node.node_type == NodeType.FULLTILE:
            # Wrap the FULLTILE in an INDEX node for proper handling
            index_node = ASTNode(NodeType.INDEX, [index_node])

        node.block_shape = self.gen.indexer.infer_block_shape_from_index(index_node, tensor_name)

        # Check if this is a kernel accumulator - if so, just return the accumulator variable
        # UNLESS it's also a cross-sloop memory tensor (needs actual load)
        if tensor_name in self.state.kernel_accumulators and tensor_name not in self.state.cross_sloop_memory_tensors:
            # Accumulator loads always return fp32 — no eager cast here.
            # fp16 conversion happens at the actual point of use:
            #   - global memory store: generate_store adds .to(fp16)
            #   - tl.dot input: promote_dot_operands adds .to(fp16)
            #   - non-transform intermediate store: generate_store adds .to(fp16)
            # Eager casting here would cause overflow in downstream x*x patterns
            # (fp16 max ≈ 65504, values > 256 overflow when squared).
            node.temp_var = tensor_name
            return ""

        # Check if this is a local intermediate tensor (not cross-kernel and not cross-sloop memory)
        if (tensor_name in self.state.intermediate_tensors and
            tensor_name not in self.state.cross_kernel_tensors and
            not (hasattr(self.state, 'cross_sloop_memory_tensors') and tensor_name in self.state.cross_sloop_memory_tensors)):
            # For local intermediate tensors, directly reference without reshape.
            # Accumulators return fp32 — see comment above.
            node.temp_var = tensor_name
            return ""

        offset_expr = self.gen.indexer.generate_index(index_node, tensor_name)

        # Create a cache key from tensor name and offset expression
        cache_key = (tensor_name, offset_expr)

        # Check if we've already loaded this exact tensor+offset combination
        if cache_key in self.state.load_cache:
            # Reuse the existing temp variable
            node.temp_var = self.state.load_cache[cache_key]
            return ""  # No code to generate, just reuse existing variable

        # Generate unique variable names for offset
        offset_var = f"offset_{self.state.offset_counter}"
        self.state.offset_counter += 1

        # Use temp_{counter} for safer variable naming
        var_name = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1

        # Store the variable mapping for later use
        node.temp_var = var_name

        # Cache this load operation
        self.state.load_cache[cache_key] = var_name

        # Generate code with separate offset variable
        # Use proper indentation based on current context
        indent_str = '    ' * self.state.indent_level
        code = f"{indent_str}{offset_var} = {offset_expr}\n"

        # Generate mask if needed
        mask_code, mask_var = self.gen.masking.generate_mask_for_index(index_node, tensor_name)
        # Follow inductor's precision rule: cross-kernel / cross-sloop / input
        # globals are all fp16, but register compute is fp32. Promote on load
        # so the register value is fp32; tl.dot inputs get downcast back to
        # fp16 later via promote_dot_operands, global stores downcast to fp16
        # via the tl.store path. Removes the need for a separate fp32-identity
        # analysis (transform_fp32_tensors) and eliminates cross-kernel dtype
        # mismatch bugs.
        if mask_var:  # Check if mask_var exists, not just mask_code
            if mask_code:
                code += mask_code
            code += f"{indent_str}{var_name} = tl.load({tensor_name}_ptr + {offset_var}, mask={mask_var}, other=0.0).to(tl.float32)"
        else:
            code += f"{indent_str}{var_name} = tl.load({tensor_name}_ptr + {offset_var}).to(tl.float32)"

        # No need for expand_dims anymore - proper offset calculation handles all dimensions
        # The loaded tensor will have the correct shape based on the offset dimensions

        return code

    def replace_multi_tensor_loads(self, node: ASTNode, index: int) -> ASTNode:
        """Replace multi-tensor loads with single tensor at given index"""
        import copy

        # Deep copy the node to avoid modifying the original
        new_node = copy.deepcopy(node)

        def replace_loads(n):
            if n.node_type == NodeType.LOAD:
                tensor_node = n.children[0]
                # Check if this load has multiple tensor names
                if len(tensor_node.children) > 1 and index < len(tensor_node.children):
                    # Replace with single tensor at index
                    single_child = tensor_node.children[index]
                    n.children[0] = ASTNode(tensor_node.node_type, [single_child])

            # Recursively process children
            for child in n.children:
                if isinstance(child, ASTNode):
                    replace_loads(child)

        replace_loads(new_node)
        return new_node

    def generate_store(self, node: ASTNode) -> str:
        """Generate store operation"""
        # (store tensor val index)
        tensor_node = node.children[0]
        val_node = node.children[1]
        index_node = node.children[2]

        # Check if tensor_node has multiple children (comma-separated tensors)
        if len(tensor_node.children) > 1:
            # Multiple tensors - generate a store for each one
            code = ""
            for i, tensor_child in enumerate(tensor_node.children):
                # Create a new store node for each tensor
                single_tensor_node = ASTNode(tensor_node.node_type, [tensor_child])

                # Create a modified value node that uses the corresponding input/tensor
                modified_val_node = self.replace_multi_tensor_loads(val_node, i)

                single_store_node = ASTNode(node.node_type, [single_tensor_node, modified_val_node, index_node])
                code += self.generate_store(single_store_node)
                if code and not code.endswith('\n'):
                    code += '\n'
            return code.rstrip('\n')  # Remove trailing newline

        tensor_name = tensor_node.children[0].value
        tensor_type = tensor_node.node_type

        # Skip store operation if this is a kernel accumulator
        # UNLESS it's also a cross-sloop memory tensor (needs immediate store)
        # OR if the store's index pattern matches current loop variables

        skip_this_store = False

        if tensor_name in self.state.kernel_accumulators and tensor_name not in self.state.cross_sloop_memory_tensors:
            # Check if this store's index pattern uses current loop variables
            # If it does, we need to generate the store (not accumulate)
            index_uses_current_loop_var = False

            if index_node and hasattr(self.state, 'current_sloop_info') and self.state.current_sloop_info:
                current_loop_var = self.state.current_sloop_info[0]
                # Check if index contains tile with current loop variable
                for child in index_node.children:
                    if child.node_type == NodeType.TILE:
                        if child.children and child.children[0].node_type == NodeType.VAR:
                            if child.children[0].value == current_loop_var:
                                index_uses_current_loop_var = True
                                break

            if index_uses_current_loop_var:
                # This store should happen now, not accumulate
                skip_this_store = False
            else:
                # Regular accumulator behavior - skip store
                skip_this_store = True

        if skip_this_store:
            # Still need to generate the value expression for accumulation
            # For accumulation patterns, we need to update the accumulator in place
            code = ""
            if self.gen.expressions.contains_loads(val_node) or self.gen.expressions.contains_reduce_sum(val_node):
                code = self.gen.expressions.generate_loads_separately(val_node)
                value_expr = self.gen.expressions.generate_node_without_loads(val_node)
            else:
                value_code = self.gen.dispatch.generate_node(val_node)
                if hasattr(val_node, 'temp_var'):
                    if value_code:
                        code += value_code
                        if not code.endswith('\n'):
                            code += '\n'
                    value_expr = val_node.temp_var
                else:
                    value_expr = value_code

            indent_str = '    ' * self.state.indent_level
            # Generate accumulation update (not a store operation)
            code += f"{indent_str}{tensor_name} = {value_expr}\n"
            return code.rstrip('\n')

        # First generate any load operations in the value expression
        code = ""

        # Delegate transform operations to transforms.py via dispatch
        if val_node.node_type in [NodeType.PERMUTE3, NodeType.TRANSPOSE, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            val_code = self.gen.dispatch.generate_node(val_node)
            if val_code:
                code += val_code
                if not code.endswith('\n'):
                    code += '\n'
            if hasattr(val_node, 'temp_var'):
                value_expr = val_node.temp_var
            else:
                raise ValueError(f"Expected temp_var for {val_node.node_type} node after dispatch")

        # Handle other expression types
        elif self.gen.expressions.contains_loads(val_node) or self.gen.expressions.contains_reduce_sum(val_node):
            code += self.gen.expressions.generate_loads_separately(val_node)

            # Check if we have nested matmuls and handle them
            if self.gen.matmul.contains_nested_matmul(val_node):
                # Find the nested matmul: X @ (A @ B)
                # We need to generate A @ B as a temporary first
                code += self.gen.matmul.generate_nested_matmul_temps(val_node)

            value_expr = self.gen.expressions.generate_node_without_loads(val_node)
        else:
            value_code = self.gen.dispatch.generate_node(val_node)
            if hasattr(val_node, 'temp_var'):
                if value_code:
                    code += value_code
                    if not code.endswith('\n'):
                        code += '\n'
                value_expr = val_node.temp_var
            else:
                value_expr = value_code

        # Use proper indentation based on current context
        indent_str = '    ' * self.state.indent_level

        # Check if this is an output tensor, input tensor, cross-kernel intermediate tensor, or cross-sloop memory tensor
        # Also check if this tensor is used with different index patterns in sloops
        is_cross_sloop_memory = (hasattr(self.state, 'cross_sloop_memory_tensors') and
                                tensor_name in self.state.cross_sloop_memory_tensors)


        if (tensor_type == NodeType.OUTPUT or
            tensor_type == NodeType.INPUT or
            (tensor_type == NodeType.TENSOR and tensor_name in self.state.cross_kernel_tensors) or
            (tensor_type == NodeType.TENSOR and is_cross_sloop_memory)):
            # For output tensors, input tensors, and cross-kernel intermediates, use tl.store
            # Global memory stores are always fp16 to match PyTorch behavior
            if not value_expr.endswith('.to(tl.float16)'):
                value_expr = f"{value_expr}.to(tl.float16)"

            # Mark this accumulator as stored if it's in a loop
            if tensor_name in self.state.kernel_accumulators and hasattr(self.state, 'current_sloop_info') and self.state.current_sloop_info:
                self.state.stored_accumulators.add(tensor_name)
            offset_expr = self.gen.indexer.generate_index(index_node, tensor_name)

            # Generate unique variable names for offset
            offset_var = f"offset_{self.state.offset_counter}"
            self.state.offset_counter += 1

            # Generate code with separate offset variable
            code += f"{indent_str}{offset_var} = {offset_expr}\n"

            # Generate mask if needed for store
            mask_code, mask_var = self.gen.masking.generate_mask_for_index(index_node, tensor_name)
            if mask_var:  # Check if mask_var exists, not just mask_code
                if mask_code:
                    code += mask_code
                code += f"{indent_str}tl.store({tensor_name}_ptr + {offset_var}, {value_expr}, mask={mask_var})"
            else:
                code += f"{indent_str}tl.store({tensor_name}_ptr + {offset_var}, {value_expr})"
        else:
            # Register intermediate store: value_expr stays fp32 (inductor-style
            # precision rule). Load-time promotion (generate_load adds
            # `.to(tl.float32)`), fp32 compute, and `promote_dot_operands`
            # downcasting at tl.dot inputs keep the register dtype invariant
            # at fp32 everywhere except tensor-core matmul operand slots.
            # The only dtype cast we still apply is at the global-store branch
            # above, which downcasts to fp16 to match PyTorch's storage dtype.
            pass

            # For intermediate tensors (pre-allocated with tl.zeros), use direct assignment
            # Check if index has any tiles or if it's a simple element access
            has_tiles = any(child.node_type in [NodeType.TILE, NodeType.FULLTILE]
                          for child in index_node.children)

            # Check if we're in a sloop and need to create a temporary variable
            # Skip temporary variable creation for intermediate tensors in sloop
            # They should be stored directly to memory if they are cross-sloop tensors
            if has_tiles:
                # For tiled access, we need to handle accumulation patterns
                # Check if this is an accumulation pattern (common in matrix multiply)
                if val_node.node_type == NodeType.ADD and len(val_node.children) > 0:
                    # Check for (+ (x 1 (load tensor ...)) ...) or (+ (x (load tensor ...) 1) ...)  pattern
                    first_child = val_node.children[0]
                    if (first_child.node_type == NodeType.MUL and
                        len(first_child.children) == 2):
                        # Check if it's (x 1 load) or (x load 1)
                        is_accumulation = False
                        if (first_child.children[0].node_type == NodeType.NUM and
                            first_child.children[0].value == '1' and
                            first_child.children[1].node_type == NodeType.LOAD):
                            # Pattern: (x 1 load)
                            is_accumulation = True
                        elif (first_child.children[1].node_type == NodeType.NUM and
                              first_child.children[1].value == '1' and
                              first_child.children[0].node_type == NodeType.LOAD):
                            # Pattern: (x load 1)
                            is_accumulation = True

                        if is_accumulation:
                            # This is an accumulation pattern - skip the "* 1" part
                            if len(val_node.children) > 1:
                                new_value = self.gen.expressions.generate_node_without_loads(val_node.children[1])
                                code += f"{indent_str}{tensor_name} += {new_value}"
                            else:
                                # Only (x 1 load) with no addition - just assign
                                code += f"{indent_str}{tensor_name} = {value_expr}"
                        else:
                            # Regular addition but not accumulation pattern
                            code += f"{indent_str}{tensor_name} = {value_expr}"
                    else:
                        # Regular addition but not accumulation pattern
                        code += f"{indent_str}{tensor_name} = {value_expr}"
                else:
                    # Direct assignment for non-accumulation patterns
                    code += f"{indent_str}{tensor_name} = {value_expr}"
            else:
                # For element access, direct assignment
                code += f"{indent_str}{tensor_name} = {value_expr}"

        return code
