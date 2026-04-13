"""Tensor shape transform helpers for Triton code emission."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, TYPE_CHECKING

from ...AstNode import ASTNode
from ...NodeType import NodeType

if TYPE_CHECKING:
    from ...state import CodeGenState
    from ....TritonGen import TritonCodeGen


class Transforms:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def build_transpose_permutation(self, child: ASTNode, node: ASTNode) -> Tuple[Tuple[int, ...], str]:
        """Build permutation tuple and string for transpose."""
        if len(node.children) != 1:
            raise ValueError("transpose requires exactly 1 argument: tensor")

        source_name = self.infer_tensor_name(child)
        if hasattr(child, "tensor_shape") and child.tensor_shape:
            num_dims = len(child.tensor_shape)
        elif source_name in self.state.tensor_shapes:
            num_dims = len(self.state.tensor_shapes[source_name])
        else:
            num_dims = 2

        perm_dims = list(range(num_dims))
        if num_dims >= 2:
            perm_dims[-2], perm_dims[-1] = perm_dims[-1], perm_dims[-2]
        return tuple(perm_dims), f"({', '.join(str(d) for d in perm_dims)})"

    def generate_permute3(self, node: ASTNode) -> str:
        """Generate permute3 operation for 3D or 4D tensor

        permute3 in the IR takes:
        1. The tensor to permute
        2-4/5. Three or four dimension indices for the permutation

        Example:
        - (permute3 tensor 1 0 2) means permute 3D tensor dimensions as (1, 0, 2)
        - (permute3 tensor 0 2 1 3) means permute 4D tensor dimensions as (0, 2, 1, 3)
        """
        if len(node.children) not in [4, 5]:
            raise ValueError("permute3 requires exactly 4 or 5 arguments: tensor, dim0, dim1, dim2, [dim3]")

        # Check if we need to generate the child first
        child = node.children[0]
        child_code = ""

        # If child doesn't have temp_var yet, generate it.
        if not hasattr(child, 'temp_var'):
            child_code = self.gen.dispatch.generate_node(child)
            if child_code and not child_code.endswith('\n'):
                child_code += '\n'

        # Get the tensor expression
        if hasattr(child, 'temp_var'):
            tensor_expr = child.temp_var
        else:
            # This should not happen anymore with proper generation
            raise ValueError(f"Expected temp_var for {child.node_type} node in permute3")

        # Dynamic permutation dimensions
        perm_dims = []
        perm_strs = []

        for i in range(len(node.children)-1):  # Skip the first child (tensor)
            dim = self.gen.dispatch.generate_node(node.children[i + 1])
            perm_dims.append(int(dim))
            perm_strs.append(str(dim))

        perm_dims = tuple(perm_dims)
        perm_str = f"({', '.join(perm_strs)})"

        # If used inline (e.g., in a store), return the expression directly
        if hasattr(self, '_generating_inline') and self._generating_inline:
            return f"tl.permute({tensor_expr}, {perm_str})"

        # Generate a temporary variable for the result
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1

        # In Triton, use tl.permute
        indent = '    ' * self.state.indent_level
        code = child_code
        code += f"{indent}{temp_var} = tl.permute({tensor_expr}, {perm_str})\n"

        # Store temp var in node for parent operations
        node.temp_var = temp_var

        # Store the permutation dimensions for later use (e.g., in squeeze)
        node.permute_dims = perm_dims
        child_shape = getattr(child, "tensor_shape", None)
        if child_shape and len(child_shape) == len(perm_dims):
            node.tensor_shape = tuple(child_shape[i] for i in perm_dims)
        child_block = getattr(child, "block_shape", None)
        if child_block and len(child_block) == len(perm_dims):
            node.block_shape = tuple(child_block[i] for i in perm_dims)

        return code

    def generate_transpose(self, node: ASTNode) -> str:
        """Generate transpose operation for tensors.

        transpose in the IR takes:
        1. The tensor to transpose
        2-3. Optional dimension indices to swap

        Examples:
        - (transpose tensor) swaps the last two dimensions
        - (transpose tensor 0 1) swaps dimensions 0 and 1
        """
        if len(node.children) != 1:
            raise ValueError("transpose requires exactly 1 argument: tensor")

        child = node.children[0]
        child_code = ""

        if not hasattr(child, 'temp_var'):
            # if child.node_type in [NodeType.LOAD, NodeType.UNSQUEEZE, NodeType.SQUEEZE, NodeType.PERMUTE3, NodeType.TRANSPOSE]:
            child_code = self.gen.dispatch.generate_node(child)
            if child_code and not child_code.endswith('\n'):
                child_code += '\n'

        if hasattr(child, 'temp_var'):
            tensor_expr = child.temp_var
        else:
            raise ValueError(f"Expected temp_var for {child.node_type} node in transpose")

        perm_dims, perm_str = self.build_transpose_permutation(child, node)

        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1

        indent = '    ' * self.state.indent_level
        code = child_code
        if len(perm_dims) == 2 and perm_dims == (1, 0):
            code += f"{indent}{temp_var} = tl.trans({tensor_expr})\n"
        else:
            raise ValueError(f"Expected 2 dimenssion in transpose")
            # code += f"{indent}{temp_var} = tl.permute({tensor_expr}, {perm_str})\n"

        node.temp_var = temp_var
        node.permute_dims = perm_dims
        child_shape = getattr(child, "tensor_shape", None)
        if child_shape and len(child_shape) == len(perm_dims):
            node.tensor_shape = tuple(child_shape[i] for i in perm_dims)
        child_block = getattr(child, "block_shape", None)
        if child_block and len(child_block) == len(perm_dims):
            node.block_shape = tuple(child_block[i] for i in perm_dims)

        return code

    def generate_squeeze(self, node: ASTNode) -> str:
        """Generate squeeze operation to remove a dimension

        squeeze in the IR takes:
        1. The tensor to squeeze
        2. The dimension to remove

        Example: (squeeze tensor 1) means remove dimension 1
        """
        if len(node.children) != 2:
            raise ValueError("squeeze requires exactly 2 arguments: tensor, dim")

        # Check if we need to generate the child first
        child = node.children[0]
        child_code = ""

        # If child needs generation, do it first
        child_code = self.gen.dispatch.generate_node(child)
        if not child_code.endswith('\n'):
            child_code += '\n'

        # Get the tensor expression
        if hasattr(child, 'temp_var'):
            tensor_expr = child.temp_var

        # Get the dimension to squeeze
        dim = self.gen.dispatch.generate_node(node.children[1])
        dim_value = None
        try:
            dim_value = int(dim)
        except (TypeError, ValueError):
            dim_value = None

        self.state.debug_log(
            f"squeeze child={child.node_type} dim={dim} dim_value={dim_value} "
            f"child.block_shape={getattr(child, 'block_shape', None)} "
            f"child.tensor_shape={getattr(child, 'tensor_shape', None)}"
        )

        # Generate a temporary variable for the result
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1

        # Try to infer the source tensor name for dimension information
        source_tensor_name = self.infer_tensor_name(child)

        indent = '    ' * self.state.indent_level
        code = child_code

        if hasattr(child, "block_shape") and child.block_shape and dim_value is not None:
            block_shape = list(child.block_shape)
            dim_index = dim_value
            if dim_index < 0:
                dim_index += len(block_shape)
            if 0 <= dim_index < len(block_shape):
                del block_shape[dim_index]
                shape_parts = [str(dim) for dim in block_shape]
                shape_tuple = f"({', '.join(shape_parts)})"
                code += f"{indent}{temp_var} = tl.reshape({tensor_expr}, {shape_tuple})\n"
                node.block_shape = tuple(block_shape)
                if hasattr(child, "tensor_shape") and child.tensor_shape:
                    tensor_shape = list(child.tensor_shape)
                    dim_idx = dim_value
                    if dim_idx < 0:
                        dim_idx += len(tensor_shape)
                    if 0 <= dim_idx < len(tensor_shape):
                        del tensor_shape[dim_idx]
                node.tensor_shape = tuple(tensor_shape)
                node.temp_var = temp_var
                node.squeeze_dim = int(dim)
                self.state.debug_log(
                    f"squeeze used block_shape -> {node.block_shape} tensor_shape={getattr(node, 'tensor_shape', None)}"
                )
                return code

        if hasattr(child, "tensor_shape") and child.tensor_shape and dim_value is not None:
            result = self._apply_squeeze_from_shape(
                child.tensor_shape,
                dim_value,
                tensor_expr,
                temp_var,
                indent,
                node,
            )
            if result is not None:
                code += result
                node.temp_var = temp_var
                node.squeeze_dim = int(dim)
                self.state.debug_log(
                    f"squeeze used tensor_shape -> {getattr(node, 'tensor_shape', None)} block_shape={getattr(node, 'block_shape', None)}"
                )
                return code
        if source_tensor_name and source_tensor_name in self.state.tensor_shapes:
            # Use tensor dimension information to construct new shape for reshape
            # Build new shape by excluding the squeezed dimension
            code += f"{indent}# Squeeze dimension {dim} from {source_tensor_name}\n"

            # Get the number of dimensions from tensor_shapes if available
            num_dims = len(self.state.tensor_shapes[source_tensor_name])

            # Check if the child was a permute operation
            if child.node_type in [NodeType.PERMUTE3, NodeType.TRANSPOSE] and hasattr(child, 'permute_dims'):
                # Get the permuted dimension order
                perm_dims = child.permute_dims  # e.g., (1, 0, 2)
                shape_parts = []
                shape_values = []

                # Map the original dimensions through the permutation
                for i in range(num_dims):
                    if i != int(dim):  # Skip the dimension to be squeezed
                        # Find which original dimension this corresponds to
                        orig_dim = perm_dims[i]
                        dim_value = self.state.tensor_shapes[source_tensor_name][orig_dim]
                        if isinstance(dim_value, str):
                            shape_parts.append(dim_value)
                        else:
                            shape_parts.append(str(dim_value))
                        if orig_dim < len(self.state.tensor_shapes[source_tensor_name]):
                            shape_values.append(self.state.tensor_shapes[source_tensor_name][orig_dim])
            else:
                # No permutation, use original dimension order
                shape_parts = []
                shape_values = []
                for i in range(num_dims):
                    if i != int(dim):  # Skip the dimension to be squeezed
                        dim_value = self.state.tensor_shapes[source_tensor_name][i]
                        if isinstance(dim_value, str):
                            shape_parts.append(dim_value)
                        else:
                            shape_parts.append(str(dim_value))
                        shape_values.append(self.state.tensor_shapes[source_tensor_name][i])

            # Pass shape directly as tuple to tl.reshape
            shape_tuple = f"({', '.join(shape_parts)})"
            padded_parts, padded_values = self.gen.shape_utils.get_padded_shape(shape_values) if shape_values else (shape_parts, shape_values)
            padded_shape_tuple = f"({', '.join(padded_parts)})"
            code += f"{indent}{temp_var} = tl.reshape({tensor_expr}, {padded_shape_tuple})\n"
            if shape_values:
                node.tensor_shape = tuple(shape_values)
                node.block_shape = tuple(padded_values)
            self.state.debug_log(
                f"squeeze used tensor_shapes[{source_tensor_name}] -> "
                f"tensor_shape={getattr(node, 'tensor_shape', None)} "
                f"block_shape={getattr(node, 'block_shape', None)}"
            )
        else:
            # Fallback: simple assignment with comment
            code += f"{indent}{temp_var} = {tensor_expr}  # squeeze dim {dim} (fallback)\n"
            self.state.debug_log("squeeze fallback: no tensor_shape/block_shape available")

        # Store temp var in node for parent operations
        node.temp_var = temp_var

        # Store squeeze dimension for parent operations
        node.squeeze_dim = int(dim)

        return code

    def _apply_squeeze_from_shape(
        self,
        child_shape,
        dim_value: int,
        tensor_expr: str,
        temp_var: str,
        indent: str,
        node: ASTNode,
    ) -> str | None:
        if dim_value < 0:
            dim_value += len(child_shape)
        if not (0 <= dim_value < len(child_shape)):
            return None
        shape_parts = []
        shape_values = []
        for i, dim_entry in enumerate(child_shape):
            if i == dim_value:
                continue
            if isinstance(dim_entry, str):
                shape_parts.append(dim_entry)
            else:
                shape_parts.append(str(dim_entry))
            shape_values.append(dim_entry)
        shape_tuple = f"({', '.join(shape_parts)})"
        padded_parts, padded_values = self.gen.shape_utils.get_padded_shape(shape_values) if shape_values else (shape_parts, shape_values)
        padded_shape_tuple = f"({', '.join(padded_parts)})"
        code = f"{indent}{temp_var} = tl.reshape({tensor_expr}, {padded_shape_tuple})\n"
        if shape_values:
            node.tensor_shape = tuple(shape_values)
            node.block_shape = tuple(padded_values)
        return code

    def infer_tensor_name(self, node: ASTNode) -> str:
        """Infer the original tensor name from a node (for accessing dimension info)"""
        if node.node_type == NodeType.LOAD:
            # Direct load operation
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT]:
                return tensor_node.children[0].value
            elif tensor_node.node_type == NodeType.TENSOR:
                # Handle (tensor name) case
                return tensor_node.children[0].value
        elif node.node_type in [NodeType.PERMUTE3, NodeType.TRANSPOSE]:
            # Permute operation - infer from its child
            return self.infer_tensor_name(node.children[0])
        elif node.node_type == NodeType.UNSQUEEZE:
            # Unsqueeze operation - infer from its child
            return self.infer_tensor_name(node.children[0])
        elif hasattr(node, 'temp_var') and hasattr(node, 'original_tensor'):
            # If we stored the original tensor name
            return node.original_tensor

        return None

    def generate_unsqueeze(self, node: ASTNode) -> str:
        """Generate unsqueeze operation to add a dimension

        unsqueeze in the IR takes:
        1. The tensor to unsqueeze
        2. The dimension where to insert new dimension

        Example: (unsqueeze tensor 1) means add a dimension at position 1
        """
        if len(node.children) != 2:
            raise ValueError("unsqueeze requires exactly 2 arguments: tensor, dim")

        # Check if we need to generate the child first
        child = node.children[0]
        child_code = ""

        # Unsqueeze can wrap arbitrary tensor expressions such as sqrt/load/reduce.
        # Generate the child unconditionally when it has not been materialized yet.
        if not hasattr(child, 'temp_var'):
            child_code = self.gen.dispatch.generate_node(child)
            if child_code and not child_code.endswith('\n'):
                child_code += '\n'

        # Get the tensor expression
        if hasattr(child, 'temp_var'):
            tensor_expr = child.temp_var
        else:
            raise ValueError(f"Expected temp_var for {child.node_type} node in unsqueeze")

        # Get the dimension to unsqueeze
        dim = self.gen.dispatch.generate_node(node.children[1])
        dim_value = None
        try:
            dim_value = int(dim)
        except (TypeError, ValueError):
            dim_value = None

        self.state.debug_log(
            f"unsqueeze child={child.node_type} dim={dim} dim_value={dim_value} "
            f"child.block_shape={getattr(child, 'block_shape', None)} "
            f"child.tensor_shape={getattr(child, 'tensor_shape', None)}"
        )

        # Generate a temporary variable for the result
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1

        # In Triton, use tl.expand_dims
        # Accumulator fp16 cast is handled at load level (generate_load)

        indent = '    ' * self.state.indent_level
        code = child_code
        code += f"{indent}{temp_var} = tl.expand_dims({tensor_expr}, {dim})\n"

        # Store temp var in node for parent operations
        node.temp_var = temp_var
        child_shape = getattr(child, "tensor_shape", None)
        dim_value = None
        try:
            dim_value = int(dim)
        except (TypeError, ValueError):
            dim_value = None
        if child_shape and dim_value is not None:
            if dim_value < 0:
                dim_value += len(child_shape) + 1
            if 0 <= dim_value <= len(child_shape):
                new_shape = list(child_shape)
                new_shape.insert(dim_value, 1)
                node.tensor_shape = tuple(new_shape)

        # Store unsqueeze dimension for parent operations
        if dim_value is not None:
            node.unsqueeze_dim = dim_value
        if hasattr(child, "tensor_shape") and child.tensor_shape and dim_value is not None:
            child_shape = list(child.tensor_shape)
            if dim_value < 0:
                dim_value += len(child_shape) + 1
            if 0 <= dim_value <= len(child_shape):
                child_shape.insert(dim_value, 1)
                node.tensor_shape = tuple(child_shape)
        if hasattr(child, "block_shape") and child.block_shape and dim_value is not None:
            block_shape = list(child.block_shape)
            dim_index = dim_value
            if dim_index < 0:
                dim_index += len(block_shape) + 1
            if 0 <= dim_index <= len(block_shape):
                block_shape.insert(dim_index, 1)
                node.block_shape = tuple(block_shape)
        self.state.debug_log(
            f"unsqueeze result block_shape={getattr(node, 'block_shape', None)} "
            f"tensor_shape={getattr(node, 'tensor_shape', None)}"
        )

        return code
