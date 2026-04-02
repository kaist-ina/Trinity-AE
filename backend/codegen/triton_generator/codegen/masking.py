"""Mask generation helpers for Triton code emission."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple, TYPE_CHECKING

from ...AstNode import ASTNode
from ...NodeType import NodeType

if TYPE_CHECKING:
    from ...state import CodeGenState
    from ....TritonGen import TritonCodeGen


class MaskGenerator:
    def __init__(self, state: CodeGenState, gen: TritonCodeGen) -> None:
        self.state = state
        self.gen = gen

    def generate_mask_for_index(self, index_node: ASTNode, tensor_name: str) -> tuple:
        """Generate mask code for tensor access if needed
        Returns: (mask_code, mask_var_name or None)
        """
        # Collect tiles that need masking
        tiles_needing_mask = []

        for i, child in enumerate(index_node.children):
            if child.node_type == NodeType.TILE:
                loop_var = child.children[0].value
                if loop_var in self.state.loop_vars:
                    # Check if we need mask for this dimension
                    if tensor_name in self.state.tensor_shapes and i < len(self.state.tensor_shapes[tensor_name]):
                        dim_size = self.state.tensor_shapes[tensor_name][i]
                        # Check if dimension size is known and not divisible by block size
                        if isinstance(dim_size, (int, float)):
                            block_param = f"BLOCK_{loop_var.upper()}"
                            # We need mask if size is not divisible by block
                            # But we can't check this at code generation time with symbolic BLOCK_K
                            # So we always generate mask for safety when using tiles
                            tiles_needing_mask.append((loop_var, i, block_param, dim_size))
                        elif isinstance(dim_size, str):
                            # Symbolic dimension - always need mask for safety
                            block_param = f"BLOCK_{loop_var.upper()}"
                            tiles_needing_mask.append((loop_var, i, block_param, dim_size))
            elif child.node_type == NodeType.FULLTILE:
                # For fulltile, check if the full dimension needs masking
                if tensor_name in self.state.tensor_shapes and i < len(self.state.tensor_shapes[tensor_name]):
                    dim_size = self.state.tensor_shapes[tensor_name][i]
                    # Fulltile uses the entire dimension, but we still need to check bounds
                    # Create a pseudo loop_var for the fulltile dimension
                    pseudo_var = f"fulltile_{i}"
                    # Get the actual range size used in fulltile
                    if isinstance(dim_size, (int, float)):
                        # For fulltile with numeric size, we need mask if the size is not aligned
                        padded_size, needs_padding = self.gen.shape_utils.get_padded_block_size(dim_size)
                        if needs_padding or dim_size != padded_size:
                            tiles_needing_mask.append((pseudo_var, i, str(padded_size), dim_size))
                    elif isinstance(dim_size, str):
                        # For symbolic dimensions, always need mask for safety
                        if dim_size in self.state.constants:
                            actual_size = self.state.constants[dim_size]
                            padded_size, needs_padding = self.gen.shape_utils.get_padded_block_size(actual_size)
                            if needs_padding or actual_size != padded_size:
                                tiles_needing_mask.append((pseudo_var, i, str(padded_size), dim_size))
                        else:
                            # Unknown symbolic size, need mask
                            tiles_needing_mask.append((pseudo_var, i, dim_size, dim_size))
            elif child.node_type == NodeType.ELEM:
                if (
                    child.children
                    and child.children[0].node_type == NodeType.VAR
                    and tensor_name in self.state.tensor_shapes
                    and i < len(self.state.tensor_shapes[tensor_name])
                ):
                    elem_var = child.children[0].value
                    dim_size = self.state.tensor_shapes[tensor_name][i]
                    block_param = f"BLOCK_{elem_var.upper()}"
                    tiles_needing_mask.append((f"elem_{elem_var}", i, block_param, dim_size))

        if not tiles_needing_mask:
            return "", None

        # Generate mask code
        indent_str = '    ' * self.state.indent_level
        mask_code = ""
        mask_conditions = []

        # Check if we've already generated this mask combination
        # Include tensor name in the key to avoid reusing masks for different tensors
        mask_key = (tensor_name,) + tuple((loop_var, dim_idx) for loop_var, dim_idx, _, _ in tiles_needing_mask)
        if mask_key in self.state.generated_masks:
            # Check if the mask was defined at a higher or equal scope level
            existing_mask_var = self.state.generated_masks[mask_key]
            existing_scope = self.state.mask_scope_level.get(existing_mask_var, 0)
            existing_loop_instance = self.state.mask_loop_instance.get(existing_mask_var, None)

            # If the existing mask is at the current scope level or higher (lower number),
            # AND it's from the same loop instance (or no loop), reuse it
            if existing_scope <= self.state.indent_level and existing_loop_instance == self.state.current_loop_instance:
                return "", existing_mask_var
            else:
                # The mask was defined in a deeper scope or different loop instance, need to regenerate
                pass

        # Generate new mask
        mask_var = f"mask_{self.state.mask_counter}"
        self.state.mask_counter += 1
        self.state.generated_masks[mask_key] = mask_var
        self.state.mask_scope_level[mask_var] = self.state.indent_level
        self.state.mask_loop_instance[mask_var] = self.state.current_loop_instance

        for loop_var, dim_idx, block_param, dim_size in tiles_needing_mask:
            # Handle fulltile differently from regular tiles
            if loop_var.startswith("fulltile_"):
                # For fulltile, generate indices directly using tl.arange
                indices_var = f"{loop_var}_indices"

                # Check if indices were already generated and are still in scope
                need_generate_indices = True
                if indices_var in self.state.generated_indices:
                    existing_scope = self.state.indices_scope_level.get(indices_var, 0)
                    existing_loop_instance = self.state.indices_loop_instance.get(indices_var, None)

                    # For fulltile indices, we need to check if they're accessible from current scope
                    # If we're at a higher indent level (deeper scope) than where it was defined,
                    # and it's from a different loop instance, we need to regenerate
                    if existing_scope <= self.state.indent_level and (existing_loop_instance == self.state.current_loop_instance or existing_loop_instance is None):
                        need_generate_indices = False

                if need_generate_indices:
                    # Generate indices for fulltile
                    if isinstance(block_param, str) and block_param.isdigit():
                        # Numeric block param (padded size)
                        mask_code += f"{indent_str}{indices_var} = tl.arange(0, {block_param})\n"
                    else:
                        # Symbolic block param
                        mask_code += f"{indent_str}{indices_var} = tl.arange(0, {block_param})\n"

                    self.state.generated_indices[indices_var] = indices_var
                    self.state.indices_scope_level[indices_var] = self.state.indent_level
                    self.state.indices_loop_instance[indices_var] = self.state.current_loop_instance

                # Add mask condition for fulltile
                mask_conditions.append(f"{indices_var} < {dim_size}")
            elif loop_var.startswith("elem_"):
                elem_var = loop_var[len("elem_"):]
                indices_var = f"{loop_var}_indices"
                need_generate_indices = True
                if indices_var in self.state.generated_indices:
                    existing_scope = self.state.indices_scope_level.get(indices_var, 0)
                    existing_loop_instance = self.state.indices_loop_instance.get(indices_var, None)
                    if existing_scope <= self.state.indent_level and existing_loop_instance == self.state.current_loop_instance:
                        need_generate_indices = False

                if need_generate_indices:
                    mask_code += (
                        f"{indent_str}{indices_var} = "
                        f"(({elem_var} // {block_param}) + tl.arange(0, 1))\n"
                    )
                    self.state.generated_indices[indices_var] = indices_var
                    self.state.indices_scope_level[indices_var] = self.state.indent_level
                    self.state.indices_loop_instance[indices_var] = self.state.current_loop_instance

                mask_conditions.append(f"{indices_var} < {dim_size}")
            else:
                # Regular tile handling
                indices_var = f"{loop_var}_indices"

                # Check if indices were already generated and are still in scope
                need_generate_indices = True
                if loop_var in self.state.generated_indices:
                    existing_indices_var = self.state.generated_indices[loop_var]
                    existing_scope = self.state.indices_scope_level.get(existing_indices_var, 0)
                    existing_loop_instance = self.state.indices_loop_instance.get(existing_indices_var, None)

                    # If the existing indices are at the current scope level or higher (lower number),
                    # AND from the same loop instance, reuse them
                    if existing_scope <= self.state.indent_level and existing_loop_instance == self.state.current_loop_instance:
                        indices_var = existing_indices_var
                        need_generate_indices = False

                # Generate indices if needed
                if need_generate_indices:
                    # Get padded block size and check if padding is needed
                    padded_block, needs_padding = self.gen.shape_utils.get_padded_block_size(block_param)
                    if needs_padding:
                        # Use padded size for tl.arange
                        mask_code += f"{indent_str}{indices_var} = {loop_var} + tl.arange(0, {padded_block})\n"
                    else:
                        # Use original size
                        mask_code += f"{indent_str}{indices_var} = {loop_var} + tl.arange(0, {block_param})\n"
                    self.state.generated_indices[loop_var] = indices_var
                    self.state.indices_scope_level[indices_var] = self.state.indent_level
                    self.state.indices_loop_instance[indices_var] = self.state.current_loop_instance

                # Add mask condition
                # dim_size is already the shape value from tensor_shapes
                # If we used padding, we need to ensure mask conditions use original size
                if need_generate_indices:
                    # Check if we used padding for this block
                    padded_block, needs_padding = self.gen.shape_utils.get_padded_block_size(block_param)
                    if needs_padding:
                        # Add both bounds check and original size check
                        mask_conditions.append(f"({indices_var} < {dim_size}) & ({indices_var} >= {loop_var})")
                    else:
                        mask_conditions.append(f"{indices_var} < {dim_size}")
                else:
                    # Reusing existing indices, still need size check
                    mask_conditions.append(f"{indices_var} < {dim_size}")

        # Combine conditions
        if len(mask_conditions) == 1:
            mask_expr = mask_conditions[0]
        else:
            # For multi-dimensional masks, we need to handle broadcasting properly
            if len(tiles_needing_mask) > 1:
                # Multiple dimensions need masking
                # Create individual masks with proper broadcasting first
                individual_masks = []
                for i, (tile_var, dim_idx, shape_value, comparison_value) in enumerate(tiles_needing_mask):
                    indices_var = f"{tile_var}_indices"
                    # Create the comparison
                    individual_mask = f"({indices_var} < {comparison_value})"
                    # Add broadcasting based on dimension
                    broadcast_parts = []
                    for j in range(len(index_node.children)):
                        if j == dim_idx:
                            broadcast_parts.append(":")
                        else:
                            broadcast_parts.append("None")
                    broadcast_str = f"[{', '.join(broadcast_parts)}]"
                    individual_masks.append(f"{individual_mask}{broadcast_str}")

                # Combine the broadcasted masks
                mask_expr = " & ".join(individual_masks)
            else:
                mask_expr = " & ".join(f"({cond})" for cond in mask_conditions)

        # Handle broadcasting for multi-dimensional access
        num_dims = len(index_node.children)
        if num_dims > 1 and len(tiles_needing_mask) == 1:
            # Single dimension mask needs broadcasting
            tile_var, dim_idx, _, _ = tiles_needing_mask[0]
            broadcast_parts = []
            for i in range(num_dims):
                if i == dim_idx:
                    broadcast_parts.append(":")
                else:
                    broadcast_parts.append("None")

            mask_broadcast = f"[{', '.join(broadcast_parts)}]"
            mask_code += f"{indent_str}{mask_var} = ({mask_expr}){mask_broadcast}\n"
        else:
            mask_code += f"{indent_str}{mask_var} = {mask_expr}\n"

        return mask_code, mask_var
