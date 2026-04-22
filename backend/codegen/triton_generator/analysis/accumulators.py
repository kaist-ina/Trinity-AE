"""Accumulator and fp32-sensitivity analysis helpers."""

from __future__ import annotations

from ...AstNode import ASTNode
from ...NodeType import NodeType


class AccumulatorAnalysis:
    def find_nested_sloop_accumulators(self, ast: ASTNode, loop_var: str = None) -> set:
        """Find accumulators in nested sloops that need per-iteration re-init.

        Called in two contexts:
          * Without ``loop_var`` (from ``AllocationPlanner``): conservatively
            identify cross-sloop-memory accumulators to skip at kernel-top
            allocation. Behavior preserved — only inspects
            ``state.kernel_accumulators``.
          * With ``loop_var`` (from ``LoopEmitter.generate_sloop``): decide
            which accumulators to zero-init at the START of the enclosing
            sloop's body. Two cases qualify:
              (1) the accumulator's index uses ``loop_var`` — per-tile
                  scratch buffer that differs each outer iteration.
              (2) neither the index nor the rhs depends on ``loop_var`` —
                  the accumulator is outer-loop-invariant, so the inner
                  reduction must restart each outer iteration (otherwise
                  eq-saturation으로 도입된 redundant outer loop가 결과를
                  배수로 누적시켜 정확도를 깬다).
            The excluded case (index-independent, rhs-dependent on
            ``loop_var``) is a genuine cross-outer-loop accumulator (e.g.
            softmax denominator accumulated across m-tiles); keep its init
            at the enclosing scope.
        """
        accumulators = set()

        # Register-only accumulators (not in ``kernel_accumulators``) only
        # need per-iteration re-init when a specific enclosing ``loop_var`` is
        # supplied. For the no-``loop_var`` call site, keep the original
        # conservative behavior so kernel-top allocation is not disturbed.
        consider_register_accs = loop_var is not None
        all_accs = getattr(self.state, "all_accumulators", set())

        def find_in_sloop(node: ASTNode, in_nested: bool = False, depth: int = 0):
            if node.node_type == NodeType.SLOOP:
                find_in_sloop(node.children[4], in_nested=True, depth=depth + 1)
            elif node.node_type == NodeType.STORE and in_nested:
                tensor_node = node.children[0]
                val_node = node.children[1]
                index_node = node.children[2] if len(node.children) > 2 else None

                if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    for child in tensor_node.children:
                        if child.node_type != NodeType.VAR:
                            continue
                        tensor_name = child.value
                        if not self.expression_contains_tensor(val_node, tensor_name):
                            continue
                        if not self.is_accumulation_pattern(val_node, tensor_name):
                            continue
                        is_acc = tensor_name in self.state.kernel_accumulators
                        if consider_register_accs and tensor_name in all_accs:
                            is_acc = True
                        if not is_acc:
                            continue
                        if loop_var and index_node:
                            uses_loop_var = any(
                                self.index_uses_loop_var(idx_child, loop_var)
                                for idx_child in index_node.children
                            )
                            rhs_dep = self._rhs_depends_on_loop_var(val_node, loop_var)
                            if uses_loop_var or not rhs_dep:
                                accumulators.add(tensor_name)
                        elif not loop_var:
                            accumulators.add(tensor_name)
            else:
                for child in node.children:
                    if isinstance(child, ASTNode):
                        find_in_sloop(child, in_nested, depth)

        find_in_sloop(ast)
        return accumulators

    def _rhs_depends_on_loop_var(self, expr: ASTNode, loop_var: str) -> bool:
        """Whether any load's index in ``expr`` references ``loop_var``.

        Used to distinguish a genuine cross-outer-loop accumulator (rhs
        actually depends on the outer loop var through some load's index)
        from an outer-loop-invariant one (rhs ignores the outer loop).
        """
        if expr.node_type == NodeType.LOAD and len(expr.children) >= 2:
            index_node = expr.children[1]
            if index_node is not None and hasattr(index_node, "children"):
                for idx_child in index_node.children:
                    if self.index_uses_loop_var(idx_child, loop_var):
                        return True
            return False
        for child in expr.children:
            if isinstance(child, ASTNode):
                if self._rhs_depends_on_loop_var(child, loop_var):
                    return True
        return False

    def generate_nested_accumulator_init(self, accumulators: set) -> str:
        """Generate initialization code for nested accumulators."""
        if not accumulators:
            return ""

        code = ""
        indent = "    " * self.state.indent_level

        for tensor_name in sorted(accumulators):
            if tensor_name in self.state.tensor_shapes:
                shape = self.state.tensor_shapes[tensor_name]
                shape_params = []

                if (
                    hasattr(self.state, "intermediate_tensor_indices")
                    and tensor_name in self.state.intermediate_tensor_indices
                ):
                    index_node = self.state.intermediate_tensor_indices[tensor_name]
                    for i, child in enumerate(index_node.children):
                        if child.node_type == NodeType.FULLTILE:
                            if i < len(shape):
                                dim = shape[i]
                                if isinstance(dim, str):
                                    shape_params.append(dim)
                                else:
                                    shape_params.append(str(dim))
                            else:
                                shape_params.append(f"{tensor_name}_dim{i}")
                        elif child.node_type == NodeType.TILE:
                            if child.children and child.children[0].node_type == NodeType.VAR:
                                loop_var = child.children[0].value
                                is_loop_var = any(
                                    lv == loop_var for lv, _, _, _ in self.state.all_loops
                                )
                                if is_loop_var:
                                    shape_params.append(f"BLOCK_{loop_var.upper()}")
                                else:
                                    shape_params.append(f"BLOCK_{loop_var.upper()}")
                            else:
                                if i < len(shape):
                                    dim = shape[i]
                                    if isinstance(dim, str):
                                        shape_params.append(dim)
                                    else:
                                        shape_params.append(str(dim))
                        elif child.node_type == NodeType.ELEM:
                            shape_params.append("1")
                else:
                    for dim in shape:
                        if isinstance(dim, str):
                            if dim in self.state.constants:
                                shape_params.append(dim)
                            elif any(op in dim for op in ["+", "-", "*", "//"]):
                                resolved = self.gen.shape_utils.resolve_value(dim)
                                shape_params.append(resolved)
                            elif dim.startswith("BLOCK_"):
                                shape_params.append(dim)
                            else:
                                shape_params.append(dim)
                        else:
                            shape_params.append(str(dim))

                if len(shape_params) == 1:
                    shape_str = f"({shape_params[0]},)"
                else:
                    shape_str = f"({', '.join(shape_params)})"

                # Accumulators always use fp32 for precision during loop accumulation
                code += f"{indent}{tensor_name} = tl.zeros({shape_str}, dtype=tl.float32)\n"

        return code

    def identify_accumulators(self, ast: ASTNode) -> set:
        """Identify which intermediate tensors are accumulators."""
        accumulators = set()

        def analyze_store(node: ASTNode, in_sloop: bool = False):
            if node.node_type == NodeType.STORE and len(node.children) >= 2:
                tensor_node = node.children[0]
                if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    for i, child in enumerate(tensor_node.children):
                        if child.node_type == NodeType.VAR:
                            tensor_name = child.value
                            val_expr = node.children[1]
                            if len(tensor_node.children) > 1:
                                modified_expr = self.gen.memory.replace_multi_tensor_loads(
                                    val_expr, i
                                )
                                if self.expression_contains_tensor(modified_expr, tensor_name):
                                    if self.is_accumulation_pattern(modified_expr, tensor_name):
                                        accumulators.add(tensor_name)
                                    elif in_sloop and self.expression_contains_tensor(
                                        modified_expr, tensor_name
                                    ):
                                        accumulators.add(tensor_name)
                            else:
                                if self.expression_contains_tensor(val_expr, tensor_name):
                                    if self.is_accumulation_pattern(val_expr, tensor_name):
                                        accumulators.add(tensor_name)
                                    elif in_sloop and self.expression_contains_tensor(
                                        val_expr, tensor_name
                                    ):
                                        accumulators.add(tensor_name)

        def traverse(node: ASTNode, in_sloop: bool = False):
            if node.node_type == NodeType.STORE:
                analyze_store(node, in_sloop)
            elif node.node_type == NodeType.SLOOP:
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop=True)
            elif node.node_type in [NodeType.SEQ, NodeType.PLOOP]:
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, in_sloop)

        traverse(ast)
        return accumulators

    def generate_kernel_accumulator_init(self) -> str:
        """Generate initialization code for kernel accumulators.
        All accumulators use fp32 for precision during loop accumulation."""
        if not self.state.kernel_accumulators:
            return ""

        code = ""
        indent_str = "    " * self.state.indent_level
        code += f"{indent_str}# Initialize kernel accumulators\n"

        for tensor in sorted(self.state.kernel_accumulators):
            if tensor in self.state.cross_sloop_memory_tensors:
                if tensor in self.state.intermediate_tensor_indices:
                    index_node = self.state.intermediate_tensor_indices[tensor]
                    uses_tile = any(
                        child.node_type == NodeType.TILE for child in index_node.children
                    )
                    if uses_tile:
                        continue
                else:
                    continue

            if tensor in self.state.intermediate_tensor_indices:
                index_node = self.state.intermediate_tensor_indices[tensor]
                actual_shape_dims = []

                for i, idx_child in enumerate(index_node.children):
                    if idx_child.node_type == NodeType.FULLTILE:
                        if tensor in self.state.tensor_shapes and i < len(
                            self.state.tensor_shapes[tensor]
                        ):
                            dim = self.state.tensor_shapes[tensor][i]
                            if isinstance(dim, str) and dim in self.state.constants:
                                actual_shape_dims.append(str(self.state.constants[dim]))
                            elif isinstance(dim, str):
                                actual_shape_dims.append(dim)
                            else:
                                actual_shape_dims.append(str(dim))
                    elif idx_child.node_type == NodeType.TILE:
                        tile_var = idx_child.children[0].value
                        actual_shape_dims.append(f"BLOCK_{tile_var.upper()}")
                    elif idx_child.node_type == NodeType.ELEM:
                        actual_shape_dims.append("1")

                shape_str = ", ".join(actual_shape_dims)
                dtype = "tl.float32"
                if len(actual_shape_dims) == 1:
                    code += f"{indent_str}{tensor} = tl.zeros(({shape_str},), dtype={dtype})\n"
                else:
                    code += f"{indent_str}{tensor} = tl.zeros(({shape_str}), dtype={dtype})\n"
            elif tensor in self.state.tensor_shapes:
                shape = self.state.tensor_shapes[tensor]
                padded_shape_dims = []
                for dim in shape:
                    if isinstance(dim, str):
                        if dim in self.state.constants:
                            actual_size = self.state.constants[dim]
                            padded_size = 1
                            while padded_size < actual_size:
                                padded_size *= 2
                            padded_shape_dims.append(str(padded_size))
                        else:
                            padded_shape_dims.append(dim)
                    else:
                        padded_size = 1
                        while padded_size < dim:
                            padded_size *= 2
                        padded_shape_dims.append(str(padded_size))

                padded_shape_str = ", ".join(padded_shape_dims)
                dtype = "tl.float32"
                if len(padded_shape_dims) == 1:
                    code += (
                        f"{indent_str}{tensor} = tl.zeros(({padded_shape_str},), dtype={dtype})\n"
                    )
                else:
                    code += (
                        f"{indent_str}{tensor} = tl.zeros(({padded_shape_str}), dtype={dtype})\n"
                    )
            else:
                dtype = "tl.float32"
                code += f"{indent_str}{tensor} = tl.zeros((1,), dtype={dtype})[0]\n"

        return code

    def generate_kernel_accumulator_stores(self) -> str:
        """Generate code to store kernel accumulators at the end."""
        if not self.state.kernel_accumulators:
            return ""

        remaining_accumulators = self.state.kernel_accumulators - self.state.stored_accumulators
        if not remaining_accumulators:
            return ""

        code = ""
        indent_str = "    " * self.state.indent_level
        code += f"{indent_str}# Store kernel accumulators\n"

        for tensor in sorted(remaining_accumulators):
            if tensor in self.state.cross_sloop_memory_tensors:
                if tensor in self.state.intermediate_tensor_indices:
                    index_node = self.state.intermediate_tensor_indices[tensor]
                    uses_tile = any(
                        child.node_type == NodeType.TILE for child in index_node.children
                    )
                    if uses_tile:
                        continue
                else:
                    continue

            if tensor in self.state.intermediate_tensor_indices:
                index_node = self.state.intermediate_tensor_indices[tensor]
                index_code = self.gen.indexer.generate_index(index_node, tensor)

                code += f"{indent_str}offset_{self.state.offset_counter} = {index_code}\n"

                mask_code, mask_var = self.gen.masking.generate_mask_for_index(index_node, tensor)
                if mask_var:
                    if mask_code:
                        code += mask_code
                    if tensor in self.state.tensor_shapes:
                        shape = self.state.tensor_shapes[tensor]
                        slice_exprs = []
                        for dim in shape:
                            if isinstance(dim, str):
                                if dim in self.state.constants:
                                    padded_dim, needs_padding = (
                                        self.gen.shape_utils.get_padded_block_size(
                                            self.state.constants[dim]
                                        )
                                    )
                                    slice_exprs.append(f":tl.arange(0, {padded_dim})")
                                else:
                                    padded_dim, needs_padding = (
                                        self.gen.shape_utils.get_padded_block_size(dim)
                                    )
                                    slice_exprs.append(f":tl.arange(0, {padded_dim})")
                            else:
                                padded_dim, needs_padding = self.gen.shape_utils.get_padded_block_size(
                                    dim
                                )
                                slice_exprs.append(f":tl.arange(0, {padded_dim})")

                        # Kernel accumulators are fp32 registers. Global storage
                        # is fp16, so always downcast on store.
                        code += (
                            f"{indent_str}tl.store({tensor}_ptr + offset_{self.state.offset_counter}, "
                            f"{tensor}.to(tl.float16), mask={mask_var})\n"
                        )
                    else:
                        code += (
                            f"{indent_str}tl.store({tensor}_ptr + offset_{self.state.offset_counter}, "
                            f"{tensor}.to(tl.float16), mask={mask_var})\n"
                        )
                else:
                    code += (
                        f"{indent_str}tl.store({tensor}_ptr + offset_{self.state.offset_counter}, "
                        f"{tensor}.to(tl.float16))\n"
                    )
                self.state.offset_counter += 1
            else:
                code += f"{indent_str}tl.store({tensor}_ptr, {tensor})\n"

        return code

    def expression_contains_tensor(self, expr: ASTNode, tensor_name: str) -> bool:
        """Check if an expression contains a load of the given tensor."""
        if expr.node_type == NodeType.LOAD:
            tensor_node = expr.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR and child.value == tensor_name:
                        return True
                return False

        for child in expr.children:
            if isinstance(child, ASTNode) and self.expression_contains_tensor(child, tensor_name):
                return True
        return False

    def is_accumulation_pattern(self, expr: ASTNode, tensor_name: str) -> bool:
        """Check if expression is an accumulation pattern."""
        if expr.node_type == NodeType.ADD:
            for child in expr.children:
                if isinstance(child, ASTNode) and self.expression_contains_tensor(child, tensor_name):
                    return True

        for child in expr.children:
            if isinstance(child, ASTNode) and self.is_accumulation_pattern(child, tensor_name):
                return True

        return False
