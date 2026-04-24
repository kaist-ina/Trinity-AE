"""
Matmul and nested matmul helpers.

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


class MatmulOps:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def generate_matmul(self, node: ASTNode) -> str:
        """Generate matrix multiplication"""
        # Check if children are load operations that have temp_var
        left_child = node.children[0]
        right_child = node.children[1]

        # Check if either child is a concat operation
        left_is_concat = (hasattr(left_child, 'is_concat') and left_child.is_concat) or \
                         (left_child.node_type == NodeType.CONCAT)
        right_is_concat = (hasattr(right_child, 'is_concat') and right_child.is_concat) or \
                          (right_child.node_type == NodeType.CONCAT)

        # Handle the special case where both operands are concat
        # This represents a block matrix multiplication
        if left_is_concat and right_is_concat:
            # Pattern: (* (concat A B axis1) (concat C D axis2))
            # This can be decomposed based on the axes

            # For the LoRA pattern:
            # (* (concat X X 1) (concat (A@B) W 0))
            # = X @ (A@B) + X @ W

            # Get axis information
            if hasattr(left_child, 'concat_axis'):
                left_axis = left_child.concat_axis
                left_children = left_child.concat_children
            else:
                # Direct NodeType.CONCAT node
                left_axis = int(self.gen.dispatch.generate_node(left_child.children[-1]))
                left_children = left_child.children[:-1]

            if hasattr(right_child, 'concat_axis'):
                right_axis = right_child.concat_axis
                right_children = right_child.concat_children
            else:
                # Direct NodeType.CONCAT node
                right_axis = int(self.gen.dispatch.generate_node(right_child.children[-1]))
                right_children = right_child.children[:-1]

            if left_axis == 1 and right_axis == 0:
                # This is the pattern we see in the LoRA expressions
                # Generate separate matmuls and add them

                # Get the concatenated tensors
                left_tensors = []
                child_code = ""
                for child in left_children:
                    child_part, temp_var = self.gen.scalar_ops.ensure_temp_var(child, "matmul concat")
                    child_code += child_part
                    left_tensors.append(temp_var)

                right_tensors = []
                for child in right_children:
                    child_part, temp_var = self.gen.scalar_ops.ensure_temp_var(child, "matmul concat")
                    child_code += child_part
                    right_tensors.append(temp_var)

                # For the pattern (concat X X 1) @ (concat T1 T2 0)
                # Result = X @ T1 + X @ T2
                if len(left_tensors) == 2 and len(right_tensors) == 2:
                    left0, right0 = self.state.promote_dot_operands(
                        left_tensors[0], right_tensors[0],
                        left_children[0], right_children[0],
                    )
                    left1, right1 = self.state.promote_dot_operands(
                        left_tensors[1], right_tensors[1],
                        left_children[1], right_children[1],
                    )
                    indent_str = '    ' * self.state.indent_level
                    temp_var = f"temp_{self.state.temp_counter}"
                    self.state.temp_counter += 1
                    node.temp_var = temp_var
                    expr = f"(tl.dot({left0}, {right0}) + tl.dot({left1}, {right1}))"
                    return f"{child_code}{indent_str}{temp_var} = {expr}"
                else:
                    # Fallback for other concat patterns
                    raise NotImplementedError(f"Concat pattern with {len(left_tensors)} x {len(right_tensors)} tensors not supported")
            else:
                raise NotImplementedError(f"Concat with axes {left_axis}, {right_axis} not supported in matmul")

        # Normal matmul without concat
        left_code, left = self.gen.scalar_ops.ensure_temp_var(left_child, "matmul")
        right_code, right = self.gen.scalar_ops.ensure_temp_var(right_child, "matmul")
        child_code = f"{left_code}{right_code}"
        left, right = self.state.promote_dot_operands(
            left, right, left_child, right_child,
        )
        indent_str = '    ' * self.state.indent_level
        temp_var = f"temp_{self.state.temp_counter}"
        self.state.temp_counter += 1
        node.temp_var = temp_var
        return f"{child_code}{indent_str}{temp_var} = tl.dot({left}, {right})"

    def contains_nested_matmul(self, node: ASTNode) -> bool:
        """Check if node contains nested matmul operations"""
        if node.node_type == NodeType.MATMUL:
            # Check if any child is also a matmul
            for child in node.children:
                if isinstance(child, ASTNode) and child.node_type == NodeType.MATMUL:
                    return True

        # Check children recursively
        for child in node.children:
            if isinstance(child, ASTNode) and self.contains_nested_matmul(child):
                return True

        return False

    def _find_nested_matmuls(self, node: ASTNode, matmuls=None):
        """Find all matmul nodes that have nested matmuls"""
        if matmuls is None:
            matmuls = []

        if node.node_type == NodeType.MATMUL:
            # Check if any child is also a matmul
            for child in node.children:
                if isinstance(child, ASTNode) and child.node_type == NodeType.MATMUL:
                    matmuls.append(node)
                    break

        # Recurse into children
        for child in node.children:
            if isinstance(child, ASTNode):
                self._find_nested_matmuls(child, matmuls)

        return matmuls

    def _find_temp_var_index(self, node: ASTNode) -> int:
        """Find the temp variable index for a load node"""
        if hasattr(node, 'temp_var'):
            # Extract number from temp_N
            return int(node.temp_var.split('_')[-1])
        return 0

    def generate_nested_matmul_temps(self, node: ASTNode) -> str:
        """Generate temporary variables for nested matmul operations"""
        code = ""
        indent_str = '    ' * self.state.indent_level

        # Find matmul nodes with nested matmuls as children
        def find_and_process_matmuls(n):
            if n.node_type == NodeType.MATMUL:
                # Check if right child is also a matmul
                if len(n.children) > 1 and n.children[1].node_type == NodeType.MATMUL:
                    inner_matmul = n.children[1]

                    # Only process if not already processed
                    if not hasattr(inner_matmul, 'temp_var'):
                        # Generate temp variable for inner matmul (A @ B)
                        temp_name = f"matmul_temp_{self.state.temp_counter}"
                        self.state.temp_counter += 1

                        # Get operands
                        left_child = inner_matmul.children[0]
                        right_child = inner_matmul.children[1]

                        # Generate expressions for operands
                        left_expr = left_child.temp_var if hasattr(left_child, 'temp_var') else self.gen.expressions.generate_node_without_loads(left_child)
                        right_expr = right_child.temp_var if hasattr(right_child, 'temp_var') else self.gen.expressions.generate_node_without_loads(right_child)

                        left_expr, right_expr = self.state.promote_dot_operands(
                            left_expr, right_expr, left_child, right_child,
                        )
                        code_line = f"{indent_str}{temp_name} = tl.dot({left_expr}, {right_expr})\n"

                        # Store temp_var for later use
                        inner_matmul.temp_var = temp_name

                        return code_line

            # Recurse into children
            for child in n.children:
                if isinstance(child, ASTNode):
                    result = find_and_process_matmuls(child)
                    if result:
                        return result
            return ""

        # Process the tree
        code += find_and_process_matmuls(node)

        return code
