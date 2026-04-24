"""
Expression lowering helpers for staged load generation.

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


class ExpressionLowerer:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def contains_loads(self, node: ASTNode) -> bool:
        """Check if node contains load operations or operations that need temp vars"""
        if node.node_type in [NodeType.LOAD, NodeType.PERMUTE3, NodeType.TRANSPOSE, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            return True
        for child in node.children:
            if isinstance(child, ASTNode) and self.contains_loads(child):
                return True
        return False

    def contains_reduce_sum(self, node: ASTNode) -> bool:
        """Check if node contains reduce operations"""
        if node.node_type in [NodeType.RSUM, NodeType.RMAX, NodeType.RMIN]:
            return True
        for child in node.children:
            if isinstance(child, ASTNode) and self.contains_reduce_sum(child):
                return True
        return False

    def generate_loads_separately(self, node: ASTNode) -> str:
        """Generate load operations and tensor operations separately before using them"""
        code = ""
        if node.node_type == NodeType.LOAD:
            # Generate load without additional indentation
            load_code = self.gen.dispatch.generate_node(node)
            if load_code:  # Only add if there's actual code
                code += load_code
                if not load_code.endswith('\n'):
                    code += '\n'
        elif node.node_type in [NodeType.PERMUTE3, NodeType.TRANSPOSE, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            # Generate these operations as well since they produce temp vars
            op_code = self.gen.dispatch.generate_node(node)
            if op_code:  # Only add if there's actual code
                code += op_code
                if not op_code.endswith('\n'):
                    code += '\n'
        else:
            for child in node.children:
                if isinstance(child, ASTNode):
                    code += self.generate_loads_separately(child)
        return code

    def generate_node_without_loads(self, node: ASTNode) -> str:
        """Generate node expression using temp variables for loads and tensor ops"""
        if node.node_type == NodeType.LOAD:
            # Return the temp variable assigned during load generation
            return node.temp_var if hasattr(node, 'temp_var') else self.gen.dispatch.generate_node(node)
        elif node.node_type in [NodeType.PERMUTE3, NodeType.TRANSPOSE, NodeType.SQUEEZE, NodeType.UNSQUEEZE]:
            # Return the temp variable assigned during operation generation
            return node.temp_var if hasattr(node, 'temp_var') else self.gen.dispatch.generate_node(node)
        elif node.node_type in [NodeType.ADD, NodeType.SUB, NodeType.MUL, NodeType.DIV, NodeType.LE]:
            op_map = {
                NodeType.ADD: '+',
                NodeType.SUB: '-',
                NodeType.MUL: '*',
                NodeType.DIV: '/',
                NodeType.LE: '<=',
            }
            left_expr = self.generate_node_without_loads(node.children[0])
            right_expr = self.generate_node_without_loads(node.children[1])
            return f"({left_expr} {op_map[node.node_type]} {right_expr})"
        elif node.node_type == NodeType.MAX:
            left_expr = self.generate_node_without_loads(node.children[0])
            right_expr = self.generate_node_without_loads(node.children[1])
            return f"tl.maximum({left_expr}, {right_expr})"
        elif node.node_type == NodeType.MIN:
            left_expr = self.generate_node_without_loads(node.children[0])
            right_expr = self.generate_node_without_loads(node.children[1])
            return f"tl.minimum({left_expr}, {right_expr})"
        elif node.node_type == NodeType.MATMUL:
            # Handle matmul specially - check if it has a temp_var from nested processing
            if hasattr(node, 'temp_var'):
                return node.temp_var
            else:
                left = node.children[0]
                right = node.children[1]
                left_expr = self.generate_node_without_loads(left)
                right_expr = self.generate_node_without_loads(right)
                # promote_dot_operands downcasts both operands to fp16 so the
                # tensor-core HMMA path can be used; tl.dot returns fp32.
                left_expr, right_expr = self.state.promote_dot_operands(
                    left_expr, right_expr, left, right,
                )
                return f"tl.dot({left_expr}, {right_expr})"
        elif node.node_type == NodeType.RSUM:
            # Handle reduce sum specially. Under the inductor-style precision
            # rule the reduction accumulator is always fp32 (same as the
            # register compute dtype); the fp16 cast for global storage is
            # applied later at the store site.
            child = node.children[0]
            axis = self.gen.dispatch.generate_node(node.children[1])
            if child.node_type == NodeType.LOAD and hasattr(child, 'temp_var'):
                return f"tl.sum({child.temp_var}, axis={axis}, dtype=tl.float32)"
            else:
                child_expr = self.generate_node_without_loads(child)
                return f"tl.sum({child_expr}, axis={axis}, dtype=tl.float32)"
        elif node.node_type == NodeType.RMAX:
            child = node.children[0]
            axis = self.gen.dispatch.generate_node(node.children[1])
            if child.node_type == NodeType.LOAD and hasattr(child, 'temp_var'):
                return f"tl.max({child.temp_var}, axis={axis})"
            else:
                child_expr = self.generate_node_without_loads(child)
                return f"tl.max({child_expr}, axis={axis})"
        elif node.node_type == NodeType.RMIN:
            child = node.children[0]
            axis = self.gen.dispatch.generate_node(node.children[1])
            if child.node_type == NodeType.LOAD and hasattr(child, 'temp_var'):
                return f"tl.min({child.temp_var}, axis={axis})"
            else:
                child_expr = self.generate_node_without_loads(child)
                return f"tl.min({child_expr}, axis={axis})"
        elif node.node_type == NodeType.BCAST:
            # Handle broadcast specially
            tensor_expr = self.generate_node_without_loads(node.children[0])
            broadcast_dim = int(self.gen.dispatch.generate_node(node.children[1]))
            if broadcast_dim == 0:
                return f"{tensor_expr}[None, :]"
            elif broadcast_dim == 1:
                return f"{tensor_expr}[:, None]"
            elif broadcast_dim == 2:
                return f"{tensor_expr}[:, :, None]"
            elif broadcast_dim == 3:
                return f"{tensor_expr}[:, :, :, None]"
            raise NotImplementedError(f"Broadcast along dimension {broadcast_dim} not yet implemented")
        elif node.node_type == NodeType.EXP:
            operand = self.generate_node_without_loads(node.children[0])
            return f"tl.exp({operand}.to(tl.float32))"
        elif node.node_type == NodeType.SQR:
            operand = self.generate_node_without_loads(node.children[0])
            return f"({operand} * {operand})"
        elif node.node_type == NodeType.SQRT:
            operand = self.generate_node_without_loads(node.children[0])
            return f"tl.sqrt({operand}.to(tl.float32))"
        elif node.node_type == NodeType.SIGMOID:
            operand = self.generate_node_without_loads(node.children[0])
            return f"tl.sigmoid({operand}.to(tl.float32))"
        elif node.node_type == NodeType.ERF:
            operand = self.generate_node_without_loads(node.children[0])
            return f"tl.math.erf({operand}.to(tl.float32))"
        elif node.node_type == NodeType.CAST:
            dtype_node = node.children[0]
            value_node = node.children[1]
            dtype = self.gen.dispatch.generate_node(dtype_node)
            dtype = self.gen.scalar_ops.map_cast_dtype(dtype)
            value_expr = self.generate_node_without_loads(value_node)
            return f"{value_expr}.to({dtype})"
        elif node.node_type == NodeType.ABS:
            operand = self.generate_node_without_loads(node.children[0])
            return f"tl.abs({operand})"
        else:
            return self.gen.dispatch.generate_node(node)
