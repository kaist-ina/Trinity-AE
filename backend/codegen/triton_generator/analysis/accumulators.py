"""Accumulator and fp32-sensitivity analysis helpers."""

from __future__ import annotations

from ...AstNode import ASTNode
from ...NodeType import NodeType


class AccumulatorAnalysis:
    def find_nested_sloop_accumulators(self, ast: ASTNode, loop_var: str = None) -> set:
        """Find accumulators that are used in nested sloops within this AST."""
        accumulators = set()

        def find_in_sloop(node: ASTNode, in_nested: bool = False, depth: int = 0):
            if node.node_type == NodeType.SLOOP:
                find_in_sloop(node.children[4], in_nested=True, depth=depth + 1)
            elif node.node_type == NodeType.STORE and in_nested:
                tensor_node = node.children[0]
                val_node = node.children[1]
                index_node = node.children[2] if len(node.children) > 2 else None

                if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    for child in tensor_node.children:
                        if child.node_type == NodeType.VAR:
                            tensor_name = child.value
                            if self.expression_contains_tensor(val_node, tensor_name):
                                if self.is_accumulation_pattern(val_node, tensor_name):
                                    if tensor_name in self.state.kernel_accumulators:
                                        if loop_var and index_node:
                                            uses_loop_var = False
                                            for idx_child in index_node.children:
                                                if self.index_uses_loop_var(idx_child, loop_var):
                                                    uses_loop_var = True
                                                    break
                                            if uses_loop_var:
                                                accumulators.add(tensor_name)
                                        elif not loop_var:
                                            accumulators.add(tensor_name)
            else:
                for child in node.children:
                    if isinstance(child, ASTNode):
                        find_in_sloop(child, in_nested, depth)

        find_in_sloop(ast)
        return accumulators

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

                code += f"{indent}{tensor_name} = tl.zeros({shape_str}, dtype=tl.float16)\n"

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

    def identify_fp32_tensors(self, ast: ASTNode) -> set:
        """Identify intermediate tensors that should stay in fp32."""
        fp32_tensors = set()
        fp32_dependent_tensors = set()

        def find_fp32_stores(node: ASTNode):
            if node.node_type == NodeType.STORE and len(node.children) >= 2:
                tensor_node = node.children[0]
                value_node = node.children[1]

                if self.contains_fp32_promoting_operation(value_node):
                    if tensor_node.node_type == NodeType.TENSOR:
                        for child in tensor_node.children:
                            if child.node_type == NodeType.VAR:
                                fp32_tensors.add(child.value)

            for child in node.children:
                if isinstance(child, ASTNode):
                    find_fp32_stores(child)

        def find_fp32_dependencies(node: ASTNode):
            if node.node_type == NodeType.STORE and len(node.children) >= 2:
                tensor_node = node.children[0]
                value_node = node.children[1]

                if tensor_node.node_type == NodeType.TENSOR:
                    for i, child in enumerate(tensor_node.children):
                        if child.node_type != NodeType.VAR:
                            continue

                        stored_tensor = child.value
                        if len(tensor_node.children) > 1:
                            relevant_value = self.gen.memory.replace_multi_tensor_loads(
                                value_node, i
                            )
                        else:
                            relevant_value = value_node

                        if self.uses_fp32_tensors(relevant_value, fp32_tensors):
                            fp32_dependent_tensors.add(stored_tensor)

            for child in node.children:
                if isinstance(child, ASTNode):
                    find_fp32_dependencies(child)

        find_fp32_stores(ast)

        prev_size = 0
        while len(fp32_tensors) != prev_size:
            prev_size = len(fp32_tensors)
            find_fp32_dependencies(ast)
            fp32_tensors.update(fp32_dependent_tensors)
            fp32_dependent_tensors.clear()

        return fp32_tensors

    def identify_exponentials(self, ast: ASTNode) -> set:
        """Alias for the fp32-sensitive tensor analysis helper."""
        return self.identify_fp32_tensors(ast)

    def contains_fp32_promoting_operation(self, node: ASTNode) -> bool:
        """Check if a node contains an op that promotes computation to fp32."""
        if node.node_type in [NodeType.EXP, NodeType.SQRT, NodeType.SIGMOID, NodeType.ERF]:
            return True

        for child in node.children:
            if isinstance(child, ASTNode) and self.contains_fp32_promoting_operation(child):
                return True

        return False

    def contains_exp_operation(self, node: ASTNode) -> bool:
        """Alias for fp32-promoting op detection."""
        return self.contains_fp32_promoting_operation(node)

    def uses_fp32_tensors(self, node: ASTNode, fp32_tensor_set: set) -> bool:
        """Check if expression uses any tensor from the fp32-sensitive set."""
        if node.node_type == NodeType.LOAD and len(node.children) >= 1:
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR and child.value in fp32_tensor_set:
                        return True

        for child in node.children:
            if isinstance(child, ASTNode) and self.uses_fp32_tensors(child, fp32_tensor_set):
                return True
        return False

    def uses_exp_tensors(self, node: ASTNode, exp_tensor_set: set) -> bool:
        """Backward-compatible alias for fp32-sensitive tensor usage checks."""
        return self.uses_fp32_tensors(node, exp_tensor_set)

    def contains_fp32_tensor_load(self, node: ASTNode, fp32_tensor_set: set) -> bool:
        """Check if a node contains a load of a fp32-sensitive tensor."""
        if node.node_type == NodeType.LOAD and len(node.children) >= 1:
            tensor_node = node.children[0]
            if tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                for child in tensor_node.children:
                    if child.node_type == NodeType.VAR and child.value in fp32_tensor_set:
                        return True

        for child in node.children:
            if isinstance(child, ASTNode) and self.contains_fp32_tensor_load(child, fp32_tensor_set):
                return True
        return False

    def contains_exp_tensor_load(self, node: ASTNode, exp_tensor_set: set) -> bool:
        """Backward-compatible alias for fp32-sensitive tensor load checks."""
        return self.contains_fp32_tensor_load(node, exp_tensor_set)

    def generate_kernel_accumulator_init(self) -> str:
        """Generate initialization code for kernel accumulators."""
        if not self.state.kernel_accumulators:
            return ""

        fp32_tensors = self.state.get_fp32_tensors()

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
                dtype = "tl.float32" if tensor in fp32_tensors else "tl.float16"
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
                dtype = "tl.float32" if tensor in fp32_tensors else "tl.float16"
                if len(padded_shape_dims) == 1:
                    code += (
                        f"{indent_str}{tensor} = tl.zeros(({padded_shape_str},), dtype={dtype})\n"
                    )
                else:
                    code += (
                        f"{indent_str}{tensor} = tl.zeros(({padded_shape_str}), dtype={dtype})\n"
                    )
            else:
                dtype = "tl.float32" if tensor in fp32_tensors else "tl.float16"
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

                        fp32_tensors = self.state.get_fp32_tensors()
                        store_value = f"{tensor}.to(tl.float16)" if tensor in fp32_tensors else tensor
                        code += (
                            f"{indent_str}tl.store({tensor}_ptr + offset_{self.state.offset_counter}, "
                            f"{store_value}, mask={mask_var})\n"
                        )
                    else:
                        fp32_tensors = self.state.get_fp32_tensors()
                        store_value = f"{tensor}.to(tl.float16)" if tensor in fp32_tensors else tensor
                        code += (
                            f"{indent_str}tl.store({tensor}_ptr + offset_{self.state.offset_counter}, "
                            f"{store_value}, mask={mask_var})\n"
                        )
                else:
                    fp32_tensors = self.state.get_fp32_tensors()
                    store_value = f"{tensor}.to(tl.float16)" if tensor in fp32_tensors else tensor
                    code += (
                        f"{indent_str}tl.store({tensor}_ptr + offset_{self.state.offset_counter}, "
                        f"{store_value})\n"
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
