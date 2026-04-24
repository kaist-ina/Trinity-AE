"""
Scalar, unary, cast, broadcast, and reduction helpers.

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


class ScalarOps:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def ensure_temp_var(self, child: ASTNode, context: str) -> tuple[str, str]:
        child_code = ""
        if not hasattr(child, 'temp_var'):
            child_code = self.gen.dispatch.generate_node(child)
            if child_code and not child_code.endswith('\n'):
                child_code += '\n'
        if not hasattr(child, 'temp_var'):
            raise ValueError(f"Expected temp_var for {child.node_type} node in {context}")
        return child_code, child.temp_var

    def get_operand_expr(self, child: ASTNode, context: str) -> tuple[str, str]:
        if child.node_type in [NodeType.NUM, NodeType.VAR]:
            return "", self.gen.dispatch.generate_node(child)
        return self.ensure_temp_var(child, context)

    def generate_binary_op(self, node: ASTNode, op: str) -> str:
        """Generate binary operation"""

        if node.node_type == NodeType.SQR:
            # For SQR, generate the operand only once
            child = node.children[0]
            child_code, operand = self.get_operand_expr(child, "sqr")
            node.tensor_shape = getattr(child, "tensor_shape", None)
            indent_str = '    ' * self.state.indent_level
            temp_var = f"temp_{self.state.temp_counter}"
            self.state.temp_counter += 1
            node.temp_var = temp_var
            expr = f"({operand} * {operand})"
            return f"{child_code}{indent_str}{temp_var} = {expr}"

        # For other binary operations
        left_child = node.children[0]
        right_child = node.children[1]

        left_code, left = self.get_operand_expr(left_child, "binary op")
        right_code, right = self.get_operand_expr(right_child, "binary op")
        child_code = f"{left_code}{right_code}"

        left_shape = getattr(left_child, "tensor_shape", None)
        right_shape = getattr(right_child, "tensor_shape", None)
        if left_shape == right_shape:
            node.tensor_shape = left_shape
        elif left_shape is None:
            node.tensor_shape = right_shape
        elif right_shape is None:
            node.tensor_shape = left_shape
        else:
            node.tensor_shape = None

        left_block_shape = getattr(left_child, "block_shape", None)
        right_block_shape = getattr(right_child, "block_shape", None)
        if left_block_shape == right_block_shape:
            node.block_shape = left_block_shape
        elif left_block_shape is None:
            node.block_shape = right_block_shape
        elif right_block_shape is None:
            node.block_shape = left_block_shape
        else:
            node.block_shape = None

        indent_str = '    ' * self.state.indent_level
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1
        node.temp_var = temp_var

        expr = f"({left} {op} {right})"
        return f"{child_code}{indent_str}{temp_var} = {expr}"

    def generate_binary_func(self, node: ASTNode, func: str) -> str:
        """Generate binary function call (e.g., tl.maximum, tl.minimum)."""
        left_child = node.children[0]
        right_child = node.children[1]

        left_code, left = self.get_operand_expr(left_child, "binary func")
        right_code, right = self.get_operand_expr(right_child, "binary func")
        child_code = f"{left_code}{right_code}"

        left_shape = getattr(left_child, "tensor_shape", None)
        right_shape = getattr(right_child, "tensor_shape", None)
        if left_shape == right_shape:
            node.tensor_shape = left_shape
        elif left_shape is None:
            node.tensor_shape = right_shape
        elif right_shape is None:
            node.tensor_shape = left_shape
        else:
            node.tensor_shape = None

        left_block_shape = getattr(left_child, "block_shape", None)
        right_block_shape = getattr(right_child, "block_shape", None)
        if left_block_shape == right_block_shape:
            node.block_shape = left_block_shape
        elif left_block_shape is None:
            node.block_shape = right_block_shape
        elif right_block_shape is None:
            node.block_shape = left_block_shape
        else:
            node.block_shape = None

        indent_str = '    ' * self.state.indent_level
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1
        node.temp_var = temp_var

        expr = f"{func}({left}, {right})"
        return f"{child_code}{indent_str}{temp_var} = {expr}"

    def generate_unary_op(self, node: ASTNode, op: str) -> str:
        """Generate unary operation"""
        child = node.children[0]

        child_code, operand = self.get_operand_expr(child, "unary op")

        node.tensor_shape = getattr(child, "tensor_shape", None)
        node.block_shape = getattr(child, "block_shape", None)

        indent_str = '    ' * self.state.indent_level
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1
        node.temp_var = temp_var

        if op in ["tl.exp", "tl.sqrt", "tl.sigmoid", "tl.math.erf"]:
            # Triton math ops require fp32 input; the explicit cast stays
            # defensively even though the register is already fp32.
            expr = f"{op}({operand}.to(tl.float32))"
        else:
            expr = f"{op}({operand})"
        return f"{child_code}{indent_str}{temp_var} = {expr}"

    def map_cast_dtype(self, dtype: str) -> str:
        """Map IR dtype string to a Triton dtype expression."""
        if dtype.startswith("tl."):
            return dtype

        dtype_map = {
            "float16": "tl.float16",
            "float32": "tl.float32",
            "float64": "tl.float64",
            "int32": "tl.int32",
            "int64": "tl.int64",
            "bool": "tl.int1",
        }
        return dtype_map.get(dtype, dtype)

    def generate_cast(self, node: ASTNode) -> str:
        """Generate cast operation"""
        if len(node.children) != 2:
            raise ValueError("cast requires exactly 2 arguments: dtype and value")

        dtype_node = node.children[0]
        value_node = node.children[1]

        dtype = self.gen.dispatch.generate_node(dtype_node)
        dtype = self.map_cast_dtype(dtype)

        value_code, value_expr = self.get_operand_expr(value_node, "cast")
        indent_str = '    ' * self.state.indent_level
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1
        node.temp_var = temp_var

        node.tensor_shape = getattr(value_node, "tensor_shape", None)
        node.block_shape = getattr(value_node, "block_shape", None)
        return f"{value_code}{indent_str}{temp_var} = {value_expr}.to({dtype})"

    def generate_reduce_sum(self, node: ASTNode) -> str:
        """Generate reduce sum operation"""
        # Check if the child is a load operation
        child = node.children[0]
        axis = self.gen.dispatch.generate_node(node.children[1])

        child_code, tensor = self.ensure_temp_var(child, "reduce sum")
        indent_str = '    ' * self.state.indent_level
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1
        node.temp_var = temp_var
        # Reduction accumulator always fp32 under the inductor-style rule.
        return f"{child_code}{indent_str}{temp_var} = tl.sum({tensor}, axis={axis}, dtype=tl.float32)"

    def generate_reduce_max(self, node: ASTNode) -> str:
        """Generate reduce max operation"""
        child = node.children[0]
        axis = self.gen.dispatch.generate_node(node.children[1])

        child_code, tensor = self.ensure_temp_var(child, "reduce max")
        indent_str = '    ' * self.state.indent_level
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1
        node.temp_var = temp_var
        return f"{child_code}{indent_str}{temp_var} = tl.max({tensor}, axis={axis})"

    def generate_reduce_min(self, node: ASTNode) -> str:
        """Generate reduce min operation"""
        child = node.children[0]
        axis = self.gen.dispatch.generate_node(node.children[1])

        child_code, tensor = self.ensure_temp_var(child, "reduce min")
        indent_str = '    ' * self.state.indent_level
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1
        node.temp_var = temp_var
        return f"{child_code}{indent_str}{temp_var} = tl.min({tensor}, axis={axis})"

    def generate_broadcast(self, node: ASTNode) -> str:
        """Generate broadcast operation

        The bcast node has two children:
        1. The tensor to broadcast (usually a load operation)
        2. The dimension along which to broadcast

        In Triton, broadcasting a 1D tensor to 2D is done by adding [:, None] or [None, :]
        depending on the broadcast dimension.
        """
        # Get the tensor to broadcast
        tensor_child = node.children[0]
        broadcast_dim = int(self.gen.dispatch.generate_node(node.children[1]))

        # Check if the child is a load operation with a temp_var
        child_code, tensor_expr = self.ensure_temp_var(tensor_child, "broadcast")
        indent_str = '    ' * self.state.indent_level
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1
        node.temp_var = temp_var

        # In Triton, to broadcast:
        # - Along dimension 0: use [None, :] or [None, :, :] or [None, :, :, :]
        # - Along dimension 1: use [:, None] or [:, None, :] or [:, None, :, :]
        # - Along dimension 2: use [:, :, None] or [:, :, None, :]
        # - Along dimension 3: use [:, :, :, None]
        if broadcast_dim == 0:
            return f"{child_code}{indent_str}{temp_var} = {tensor_expr}[None, :]"
        elif broadcast_dim == 1:
            return f"{child_code}{indent_str}{temp_var} = {tensor_expr}[:, None]"
        elif broadcast_dim == 2:
            return f"{child_code}{indent_str}{temp_var} = {tensor_expr}[:, :, None]"
        elif broadcast_dim == 3:
            return f"{child_code}{indent_str}{temp_var} = {tensor_expr}[:, :, :, None]"
        else:
            # For higher dimensions, we'd need more complex indexing
            raise NotImplementedError(f"Broadcast along dimension {broadcast_dim} not yet implemented")
