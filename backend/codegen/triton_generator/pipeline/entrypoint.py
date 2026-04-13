"""Pipeline entrypoint for Triton code generation."""

from __future__ import annotations

from typing import Dict, Tuple

from ...AstNode import ASTNode
from ...NodeType import NodeType


class PipelineEntryPoint:
    def generate(
        self,
        ast: ASTNode,
        tensor_shapes: Dict[str, Tuple[int, ...]] = None,
        constants: Dict[str, int] = None,
    ) -> str:
        """Generate Triton kernel code from AST."""
        self.state.current_ast = ast

        if tensor_shapes:
            self.state.tensor_shapes = tensor_shapes
        if constants:
            self.state.constants = constants

        self.state.kernel_counter = 0
        self.state.generated_kernels = []
        self.state.cross_kernel_tensors = set()

        if ast.node_type == NodeType.SEQ:
            self.gen.analyzer.identify_cross_kernel_tensors(ast)
            return self.generate_seq_kernels(ast)

        all_code = ""
        kernel_name = "kernel_0"

        cross_sloop_memory_tensors = self.gen.analyzer.identify_cross_sloop_memory_tensors(ast)
        kernel_code = self.generate_single_kernel(ast, kernel_name=kernel_name)

        block_params = []
        for loop_var, _, _, tile_size in self.state.all_loops:
            if isinstance(tile_size, str) and not tile_size.isdigit():
                block_params.append(f"BLOCK_{loop_var.upper()}")

        all_code += """import triton
import triton.language as tl
import torch
"""
        if block_params:
            all_code += "\n" + self.gen.kernel.generate_autotune_decorator(block_params) + "\n"

        kernel_code = kernel_code.replace("import triton\nimport triton.language as tl\nimport torch\n\n", "")

        all_code += kernel_code + "\n\n"

        self.state.generated_kernels.append(
            (
                kernel_name,
                kernel_code,
                self.state.tensors_used.copy(),
                self.state.parallel_dims.copy(),
                self.state.all_loops.copy(),
                self.state.stored_tensor_dims.copy(),
                self.state.intermediate_tensors.copy(),
                cross_sloop_memory_tensors.copy(),
            )
        )

        local_intermediate_tensors = self.state.intermediate_tensors - self.state.cross_kernel_tensors
        all_code += self.generate_wrapper_function(local_intermediate_tensors)

        return all_code
