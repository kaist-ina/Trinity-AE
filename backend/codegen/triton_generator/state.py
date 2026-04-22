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
        self.current_ast = None
        self.debug = bool(os.environ.get("TRITON_GEN_DEBUG"))

    # ------------------------------------------------------------------
    # Utility helpers that only depend on state attributes
    # ------------------------------------------------------------------

    def debug_log(self, message: str) -> None:
        if self.debug:
            print(f"[TritonGen] {message}")

    def promote_dot_operands(
        self,
        left: str,
        right: str,
        left_node,
        right_node,
    ) -> tuple[str, str]:
        # tl.dot requires both operands to be fp16 for tensor core utilisation.
        # Check the OUTERMOST type via endswith — ``in`` matches inner
        # sub-expressions (e.g. tl.exp(tl.dot(...).to(fp32))) and misses the
        # fact that the outermost expression is still fp32. If already fp16,
        # the extra cast is a Triton no-op.
        if not left.endswith('.to(tl.float16)'):
            left = f"({left}).to(tl.float16)"
        if not right.endswith('.to(tl.float16)'):
            right = f"({right}).to(tl.float16)"
        return left, right
