"""Intermediate allocation planning helpers."""

from __future__ import annotations

from ...AstNode import ASTNode
from ...NodeType import NodeType


class AllocationPlanner:
    def generate_intermediate_allocations(self, ast: ASTNode = None) -> str:
        """Generate tl.zeros allocations for intermediate tensors."""
        accumulators = set()
        cross_sloop_tensors = set()
        sloop_intermediate_tensors = set()
        cross_sloop_memory_tensors = set()
        fp32_tensors = set()
        if ast:
            accumulators = self.identify_accumulators(ast)
            cross_sloop_tensors = self.identify_cross_sloop_tensors(ast)
            sloop_intermediate_tensors = self.identify_sloop_intermediate_tensors(ast)
            cross_sloop_memory_tensors = self.identify_cross_sloop_memory_tensors(ast)
            fp32_tensors = self.identify_fp32_tensors(ast)

            non_cross_sloop_accumulators = accumulators - cross_sloop_tensors
            cross_sloop_memory_tensors -= non_cross_sloop_accumulators

            self.state.intermediate_tensors -= cross_sloop_memory_tensors
            self.state.tensors_used.update(cross_sloop_memory_tensors)
            if (
                not hasattr(self.state, "cross_sloop_memory_tensors")
                or not self.state.cross_sloop_memory_tensors
            ):
                self.state.cross_sloop_memory_tensors = cross_sloop_memory_tensors

        local_intermediates = (
            self.state.intermediate_tensors
            - self.state.cross_kernel_tensors
            - cross_sloop_memory_tensors
        )

        if not local_intermediates:
            return ""

        nested_sloop_accumulators = set()

        def find_sloop_accumulators(node: ASTNode):
            if node.node_type == NodeType.SLOOP:
                sloop_accums = self.find_nested_sloop_accumulators(node.children[4])
                nested_sloop_accumulators.update(sloop_accums)
            else:
                for child in node.children:
                    if isinstance(child, ASTNode):
                        find_sloop_accumulators(child)

        if ast:
            find_sloop_accumulators(ast)

        local_intermediates = local_intermediates - nested_sloop_accumulators

        if not local_intermediates:
            return ""

        code = "    # Allocate intermediate tensors\n"

        for tensor_name in sorted(local_intermediates):
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

                if (
                    tensor_name in accumulators
                    or tensor_name in cross_sloop_tensors
                    or tensor_name in sloop_intermediate_tensors
                ):
                    dtype = "tl.float32" if tensor_name in fp32_tensors else "tl.float16"
                    code += f"    {tensor_name} = tl.zeros({shape_str}, dtype={dtype})\n"

        code += "\n"
        return code
