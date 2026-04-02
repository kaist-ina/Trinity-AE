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
            node.temp_var = tensor_name
            return ""

        # Check if this is a local intermediate tensor (not cross-kernel and not cross-sloop memory)
        if (tensor_name in self.state.intermediate_tensors and
            tensor_name not in self.state.cross_kernel_tensors and
            not (hasattr(self.state, 'cross_sloop_memory_tensors') and tensor_name in self.state.cross_sloop_memory_tensors)):
            # For local intermediate tensors, directly reference without reshape
            # tl.dot can handle 3D tensors with batch dimension
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
        if mask_var:  # Check if mask_var exists, not just mask_code
            if mask_code:
                code += mask_code
            code += f"{indent_str}{var_name} = tl.load({tensor_name}_ptr + {offset_var}, mask={mask_var}, other=0.0)"
        else:
            code += f"{indent_str}{var_name} = tl.load({tensor_name}_ptr + {offset_var})"

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

        # Set current store tensor context for binary/unary ops
        self.state.current_store_tensor = tensor_name

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

        # Check if value is a simple transformation operation (permute, squeeze, unsqueeze)
        # that we can generate inline without temp variable
        if val_node.node_type in [NodeType.PERMUTE3, NodeType.TRANSPOSE, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            # For these operations, generate them inline and assign directly to target tensor
            if val_node.node_type == NodeType.PERMUTE3:
                child = val_node.children[0]
                # Generate child if needed (could be LOAD, UNSQUEEZE, or other operations)
                if not hasattr(child, 'temp_var'):
                    # For UNSQUEEZE nodes, we need special handling to avoid inline generation
                    if child.node_type == NodeType.UNSQUEEZE:
                        # Generate the unsqueeze operation properly
                        unsqueeze_child = child.children[0]
                        if not hasattr(unsqueeze_child, 'temp_var'):
                            child_code = self.gen.dispatch.generate_node(unsqueeze_child)
                            if child_code:
                                code += child_code
                                if not code.endswith('\n'):
                                    code += '\n'

                        # Now generate the unsqueeze itself
                        child_code = self.gen.dispatch.generate_node(child)
                        if child_code:
                            code += child_code
                            if not code.endswith('\n'):
                                code += '\n'
                    else:
                        child_code = self.gen.dispatch.generate_node(child)
                        if child_code:  # If code was generated, append it
                            code += child_code
                            if not code.endswith('\n'):
                                code += '\n'

                # Get tensor expression - at this point child should have temp_var
                if hasattr(child, 'temp_var'):
                    tensor_expr = child.temp_var
                else:
                    # This should not happen anymore with proper generation
                    raise ValueError(f"Expected temp_var for {child.node_type} node")

                # Get permutation dimensions
                perm_strs = []
                for i in range(len(val_node.children)-1):
                    dim = self.gen.dispatch.generate_node(val_node.children[i+1])
                    perm_strs.append(str(dim))
                perm_str = f"({', '.join(perm_strs)})"
                value_expr = f"tl.permute({tensor_expr}, {perm_str})"
            elif val_node.node_type == NodeType.TRANSPOSE:
                child = val_node.children[0]
                # Generate child if needed
                if not hasattr(child, 'temp_var'):
                    if child.node_type == NodeType.UNSQUEEZE:
                        unsqueeze_child = child.children[0]
                        if not hasattr(unsqueeze_child, 'temp_var'):
                            child_code = self.gen.dispatch.generate_node(unsqueeze_child)
                            if child_code:
                                code += child_code
                                if not code.endswith('\n'):
                                    code += '\n'

                        child_code = self.gen.dispatch.generate_node(child)
                        if child_code:
                            code += child_code
                            if not code.endswith('\n'):
                                code += '\n'
                    else:
                        child_code = self.gen.dispatch.generate_node(child)
                        if child_code:
                            code += child_code
                            if not code.endswith('\n'):
                                code += '\n'

                if hasattr(child, 'temp_var'):
                    tensor_expr = child.temp_var
                else:
                    raise ValueError(f"Expected temp_var for {child.node_type} node")

                perm_dims, perm_str = self.gen.transforms.build_transpose_permutation(child, val_node)
                if len(perm_dims) == 2 and perm_dims == (1, 0):
                    value_expr = f"tl.trans({tensor_expr})"
                else:
                    value_expr = f"tl.permute({tensor_expr}, {perm_str})"

            elif val_node.node_type == NodeType.UNSQUEEZE:
                child = val_node.children[0]
                # Generate child if needed
                if child.node_type == NodeType.LOAD and not hasattr(child, 'temp_var'):
                    code += self.gen.dispatch.generate_node(child) + "\n"

                # Get tensor expression
                tensor_expr = child.temp_var if hasattr(child, 'temp_var') else self.gen.dispatch.generate_node(child)

                # Get dimension
                dim = self.gen.dispatch.generate_node(val_node.children[1])
                dim_value = None
                try:
                    dim_value = int(dim)
                except (TypeError, ValueError):
                    dim_value = None

                value_expr = f"tl.expand_dims({tensor_expr}, {dim})"
                self.state.debug_log(
                    f"inline unsqueeze child={child.node_type} dim={dim} dim_value={dim_value} "
                    f"child.block_shape={getattr(child, 'block_shape', None)} "
                    f"child.tensor_shape={getattr(child, 'tensor_shape', None)}"
                )
                if hasattr(child, "tensor_shape") and child.tensor_shape and dim_value is not None:
                    child_shape = list(child.tensor_shape)
                    if dim_value < 0:
                        dim_value += len(child_shape) + 1
                    if 0 <= dim_value <= len(child_shape):
                        child_shape.insert(dim_value, 1)
                        val_node.tensor_shape = tuple(child_shape)
                if hasattr(child, "block_shape") and child.block_shape and dim_value is not None:
                    block_shape = list(child.block_shape)
                    dim_index = dim_value
                    if dim_index < 0:
                        dim_index += len(block_shape) + 1
                    if 0 <= dim_index <= len(block_shape):
                        block_shape.insert(dim_index, 1)
                        val_node.block_shape = tuple(block_shape)
                if dim_value is not None:
                    val_node.unsqueeze_dim = dim_value
                self.state.debug_log(
                    f"inline unsqueeze result block_shape={getattr(val_node, 'block_shape', None)} "
                    f"tensor_shape={getattr(val_node, 'tensor_shape', None)}"
                )

            elif val_node.node_type == NodeType.SQUEEZE:
                child = val_node.children[0]
                squeeze_handled = False
                # Generate child if needed
                if child.node_type in [NodeType.LOAD, NodeType.PERMUTE3, NodeType.TRANSPOSE, NodeType.UNSQUEEZE] and not hasattr(child, 'temp_var'):
                    code += self.gen.dispatch.generate_node(child)
                    if not code.endswith('\n'):
                        code += '\n'

                # Get tensor expression
                if hasattr(child, 'temp_var'):
                    tensor_expr = child.temp_var
                else:
                    raise ValueError(f"Expected temp_var for {child.node_type} node in squeeze")

                # Get dimension (for squeeze, we use reshape instead)
                # Infer the source tensor name for dimension info
                source_tensor_name = self.gen.transforms.infer_tensor_name(child)
                squeeze_dim = None
                try:
                    squeeze_dim = int(self.gen.dispatch.generate_node(val_node.children[1]))
                except (TypeError, ValueError):
                    squeeze_dim = None

                self.state.debug_log(
                    f"inline squeeze child={child.node_type} dim={squeeze_dim} "
                    f"child.block_shape={getattr(child, 'block_shape', None)} "
                    f"child.tensor_shape={getattr(child, 'tensor_shape', None)}"
                )

                if hasattr(child, "block_shape") and child.block_shape and squeeze_dim is not None:
                    block_shape = list(child.block_shape)
                    dim_index = squeeze_dim
                    if dim_index < 0:
                        dim_index += len(block_shape)
                    if 0 <= dim_index < len(block_shape):
                        del block_shape[dim_index]
                        shape_parts = [str(dim) for dim in block_shape]
                        shape_str = f"({', '.join(shape_parts)})"
                        value_expr = f"tl.reshape({tensor_expr}, {shape_str})"
                        val_node.block_shape = tuple(block_shape)
                        if hasattr(child, "tensor_shape") and child.tensor_shape:
                            tensor_shape = list(child.tensor_shape)
                            dim_idx = squeeze_dim
                            if dim_idx < 0:
                                dim_idx += len(tensor_shape)
                            if 0 <= dim_idx < len(tensor_shape):
                                del tensor_shape[dim_idx]
                                val_node.tensor_shape = tuple(tensor_shape)
                        # Skip tensor_shape-based reshape handling
                        source_tensor_name = None
                        squeeze_handled = True
                        self.state.debug_log(
                            f"inline squeeze used block_shape -> {val_node.block_shape} "
                            f"tensor_shape={getattr(val_node, 'tensor_shape', None)}"
                        )

                # For squeeze operations, we need to use the source tensor's dimensions
                # after removing the squeezed dimension
                if not squeeze_handled and source_tensor_name and source_tensor_name in self.state.tensor_shapes:
                    # Check if child was a permute operation to map dimensions correctly
                    if child.node_type in [NodeType.PERMUTE3, NodeType.TRANSPOSE] and hasattr(child, 'permute_dims'):
                        # Get the permuted dimension order
                        perm_dims = child.permute_dims  # e.g., (1, 0, 2)
                        shape_parts = []

                        # Build shape from source tensor dimensions, excluding squeezed dim
                        for i in range(len(perm_dims)):
                            if i != squeeze_dim:  # Skip the dimension to be squeezed
                                orig_dim = perm_dims[i]
                                # Get the dimension value from tensor_shapes
                                shape = self.state.tensor_shapes[source_tensor_name]
                                if orig_dim < len(shape):
                                    dim_value = shape[orig_dim]
                                    if isinstance(dim_value, str):
                                        # Symbolic dimension
                                        shape_parts.append(dim_value)
                                    else:
                                        # Literal number
                                        shape_parts.append(str(dim_value))
                                else:
                                    # Fallback
                                    shape_parts.append(f"{source_tensor_name}_dim{orig_dim}")
                    else:
                        # No permutation, use original dimension order
                        shape_parts = []
                        num_dims = len(self.state.tensor_shapes[source_tensor_name])
                        shape = self.state.tensor_shapes[source_tensor_name]
                        for i in range(num_dims):
                            if i != squeeze_dim:  # Skip the dimension to be squeezed
                                dim_value = shape[i]
                                if isinstance(dim_value, str):
                                    # Symbolic dimension
                                    shape_parts.append(dim_value)
                                else:
                                    # Literal number
                                    shape_parts.append(str(dim_value))

                    shape_str = f"({', '.join(shape_parts)})"
                    padded_parts, _ = self.gen.shape_utils.get_padded_shape(shape_values) if shape_values else (shape_parts, shape_values)
                    padded_shape_str = f"({', '.join(padded_parts)})"
                    value_expr = f"tl.reshape({tensor_expr}, {padded_shape_str})"
                elif not squeeze_handled and hasattr(child, "tensor_shape") and child.tensor_shape and squeeze_dim is not None:
                    child_shape = child.tensor_shape
                    if squeeze_dim < 0:
                        squeeze_dim += len(child_shape)
                    if 0 <= squeeze_dim < len(child_shape):
                        shape_parts = []
                        shape_values = []
                        for i, dim_value in enumerate(child_shape):
                            if i == squeeze_dim:
                                continue
                            if isinstance(dim_value, str):
                                shape_parts.append(dim_value)
                            else:
                                shape_parts.append(str(dim_value))
                            shape_values.append(dim_value)
                        shape_str = f"({', '.join(shape_parts)})"
                        padded_parts, _ = self.gen.shape_utils.get_padded_shape(shape_values) if shape_values else (shape_parts, shape_values)
                        padded_shape_str = f"({', '.join(padded_parts)})"
                        value_expr = f"tl.reshape({tensor_expr}, {padded_shape_str})"
                        if shape_values:
                            val_node.tensor_shape = tuple(shape_values)
                elif not squeeze_handled:
                    # Fallback: use source tensor shape after squeeze
                    # For squeeze operation, we need to use the dimensions of the source tensor O
                    # not the output tensor O2
                    if source_tensor_name and source_tensor_name in self.state.tensor_shapes:
                        # Get source tensor shape and remove squeezed dimension
                        shape_parts = []
                        source_shape = self.state.tensor_shapes[source_tensor_name]

                        for i in range(len(source_shape)):
                            if i != squeeze_dim:  # Skip the squeezed dimension
                                dim_value = source_shape[i]
                                if isinstance(dim_value, str):
                                    shape_parts.append(dim_value)
                                else:
                                    shape_parts.append(str(dim_value))

                        shape_str = f"({', '.join(shape_parts)})"
                        padded_parts, _ = self.gen.shape_utils.get_padded_shape(shape_values) if shape_values else (shape_parts, shape_values)
                        padded_shape_str = f"({', '.join(padded_parts)})"
                        value_expr = f"tl.reshape({tensor_expr}, {padded_shape_str})"
                    elif tensor_name in self.state.tensor_shapes:
                        # If we still don't have source tensor info, use output tensor dimensions
                        shape_dims = []
                        for i in range(len(self.state.tensor_shapes[tensor_name])):
                            dim_value = self.state.tensor_shapes[tensor_name][i]
                            if isinstance(dim_value, str):
                                # Symbolic dimension - use it directly
                                shape_dims.append(dim_value)
                            else:
                                # Literal number
                                shape_dims.append(str(dim_value))
                        shape_str = f"({', '.join(shape_dims)})"
                        padded_parts, _ = self.gen.shape_utils.get_padded_shape(shape_dims) if shape_dims else (shape_dims, shape_dims)
                        padded_shape_str = f"({', '.join(padded_parts)})"
                        value_expr = f"tl.reshape({tensor_expr}, {padded_shape_str})"
                    else:
                        # Last resort: tensor already reshaped in squeeze operation
                        value_expr = f"{tensor_expr}"

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

        fp32_tensors = self.state.get_fp32_tensors()
        if tensor_name in fp32_tensors:
            value_expr = value_expr.replace('.to(tl.float16)', '')

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
            dest_dtype = "tl.float32" if tensor_name in fp32_tensors else "tl.float16"
            if dest_dtype == "tl.float16":
                if not value_expr.endswith('.to(tl.float16)'):
                    value_expr = f"{value_expr}.to(tl.float16)"
            else:
                value_expr = value_expr.replace('.to(tl.float16)', '')
                if not value_expr.endswith('.to(tl.float32)'):
                    value_expr = f"{value_expr}.to(tl.float32)"

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
