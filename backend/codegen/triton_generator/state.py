"""Shared mutable state for the Triton code generator."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..TritonGen import TritonCodeGen


class CodeGenState:
    """Holds all mutable state shared across code-generation components."""

    def __init__(self):
        self.gen: TritonCodeGen | None = None  # back-reference, set by mediator

        self.tensor_shapes = {}
        self.loop_vars = {}
        self.temp_counter = 0
        self.indent_level = 0
        self.parallel_dims = []
        self.all_loops = []
        self.program_id_counter = 0
        self.tensors_used = set()
        self.offset_counter = 0
        self.loop_var_to_tensor_dim = {}
        self.kernel_counter = 0
        self.intermediate_tensors = set()
        self.generated_kernels = []
        self.cross_kernel_tensors = set()
        self.cross_sloop_memory_tensors = set()
        self.stored_tensor_dims = {}
        self.load_cache = {}
        self.constants = {}
        self.mask_counter = 0
        self.generated_masks = {}
        self.intermediate_tensor_indices = {}
        self.kernel_accumulators = set()
        self.sloop_temp_vars = {}
        self.current_sloop_info = None
        self.input_tensors = set()
        self.output_tensors = set()
        self.mask_scope_level = {}
        self.indices_scope_level = {}
        self.generated_indices = {}
        self.loop_instance_counter = 0
        self.current_loop_instance = None
        self.mask_loop_instance = {}
        self.indices_loop_instance = {}
        self.sloop_depth = 0
        self.stored_accumulators = set()
        self.current_store_tensor = None
        self.fp32_tensors = set()
        self.global_fp32_tensors = set()
        self.exp_tensors = set()
        self.current_ast = None
        self.debug = bool(os.environ.get("TRITON_GEN_DEBUG"))

    # ------------------------------------------------------------------
    # Utility helpers that only depend on state attributes
    # ------------------------------------------------------------------

    def debug_log(self, message: str) -> None:
        if self.debug:
            print(f"[TritonGen] {message}")

    def get_fp32_tensors(self) -> set:
        fp32_tensors = getattr(self, "fp32_tensors", None)
        if fp32_tensors is not None:
            return fp32_tensors
        return getattr(self, "exp_tensors", set())

    def is_fp32_tensor(self, tensor_name: str | None) -> bool:
        return bool(tensor_name) and tensor_name in self.get_fp32_tensors()

    def current_store_requires_fp32(self) -> bool:
        return self.is_fp32_tensor(getattr(self, "current_store_tensor", None))

    def node_requires_fp32(self, node) -> bool:
        if node is None:
            return False
        fp32_tensors = self.get_fp32_tensors()
        uses_fp32_tensor = False
        contains_fp32_op = False
        if self.gen is not None:
            uses_fp32_tensor = self.gen.analyzer.contains_fp32_tensor_load(node, fp32_tensors)
            contains_fp32_op = self.gen.analyzer.contains_fp32_promoting_operation(node)
        return uses_fp32_tensor or contains_fp32_op

    def cast_expression(self, expr: str, *, keep_fp32: bool, dtype: str = "tl.float16") -> str:
        if keep_fp32:
            return expr
        return f"{expr}.to({dtype})"

    def promote_dot_operands(
        self,
        left: str,
        right: str,
        left_node,
        right_node,
        *,
        force_fp32: bool = False,
    ) -> tuple[str, str, bool]:
        left_requires_fp32 = self.node_requires_fp32(left_node)
        right_requires_fp32 = self.node_requires_fp32(right_node)
        store_requires_fp32 = self.current_store_requires_fp32()
        keep_fp32 = (
            force_fp32
            or store_requires_fp32
            or left_requires_fp32
            or right_requires_fp32
        )
        if keep_fp32:
            if left_requires_fp32 and not right_requires_fp32:
                right = f"{right}.to(tl.float32)"
            elif right_requires_fp32 and not left_requires_fp32:
                left = f"{left}.to(tl.float32)"
            elif force_fp32 or store_requires_fp32:
                if not left_requires_fp32:
                    left = f"{left}.to(tl.float32)"
                if not right_requires_fp32:
                    right = f"{right}.to(tl.float32)"
        return left, right, keep_fp32
