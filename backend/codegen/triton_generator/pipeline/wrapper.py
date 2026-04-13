"""Wrapper generation helpers."""

from __future__ import annotations


class WrapperPipeline:
    def generate_wrapper_function(self, all_local_intermediate_tensors: set) -> str:
        """Generate a wrapper function that calls all kernels sequentially."""
        all_tensors = set()
        all_block_params = set()

        for _, _, tensors_used, _, all_loops, _, _, _ in self.state.generated_kernels:
            all_tensors.update(tensors_used)
            for loop_var, _, _, tile_size in all_loops:
                if isinstance(tile_size, str) and not tile_size.isdigit():
                    all_block_params.add(f"block_{loop_var}")

        forward_params = []
        for tensor_name in sorted(all_tensors):
            if tensor_name not in all_local_intermediate_tensors:
                forward_params.append(tensor_name)

        code = "# Metadata for benchmark.py\n"
        code += f"TENSOR_PARAMS = {forward_params}\n"
        code += f"BLOCK_PARAMS = {sorted(list(all_block_params))}\n\n"

        code += "def forward("

        params = []
        for tensor_name in forward_params:
            params.append(f"{tensor_name}")

        for block_param in sorted(all_block_params):
            params.append(f"{block_param}=16")

        code += ", ".join(params) + "):\n"
        code += '    """\n'
        code += '    Wrapper function that executes all kernels sequentially.\n'
        code += '    """\n'

        for (
            kernel_name,
            _,
            tensors_used,
            parallel_dims,
            all_loops,
            _stored_tensor_dims,
            intermediate_tensors,
            cross_sloop_memory_tensors,
        ) in self.state.generated_kernels:
            kernel_block_params = []
            for loop_var, _, _, tile_size in all_loops:
                if isinstance(tile_size, str) and not tile_size.isdigit():
                    block_param = f"block_{loop_var}"
                    if block_param in all_block_params:
                        kernel_block_params.append((loop_var, block_param))

            if parallel_dims:
                has_block_params = False
                grid_lambda_parts = []

                for loop_var, start, end, tile_size in parallel_dims:
                    if isinstance(tile_size, str) and tile_size.isdigit():
                        end_param = end
                        grid_lambda_parts.append(
                            f"({end_param} - {start} + {tile_size} - 1) // {tile_size}"
                        )
                    else:
                        has_block_params = True
                        block_var = None
                        for lv, _block_param in kernel_block_params:
                            if lv == loop_var:
                                block_var = f'meta["BLOCK_{lv.upper()}"]'
                                break

                        if block_var:
                            end_param = end
                            grid_lambda_parts.append(
                                f"({end_param} - {start} + {block_var} - 1) // {block_var}"
                            )

                if grid_lambda_parts:
                    if has_block_params:
                        lambda_expr = f"lambda meta: ({', '.join(grid_lambda_parts)},)"
                        code += f"    {kernel_name}[{lambda_expr}](\n"
                    else:
                        code += f"    {kernel_name}[({', '.join(grid_lambda_parts)},)](\n"
                else:
                    code += f"    {kernel_name}[(1,)](\n"
            else:
                code += f"    {kernel_name}[(1,)](\n"

            kernel_params = []

            non_intermediate_tensors = sorted(
                [
                    t
                    for t in tensors_used
                    if t not in intermediate_tensors
                    or (t in self.state.cross_kernel_tensors and t in tensors_used)
                    or t in cross_sloop_memory_tensors
                ]
            )

            for tensor in non_intermediate_tensors:
                kernel_params.append(f"        {tensor}")

                if tensor in self.state.tensor_shapes:
                    num_dims = len(self.state.tensor_shapes[tensor])
                else:
                    num_dims = max(len(parallel_dims), 2)

                for dim in range(num_dims):
                    kernel_params.append(f"        {tensor}.stride({dim})")

            code += ",\n".join(kernel_params)

            kernel_autotuned_blocks = set()
            for loop_var, _, _, tile_size in all_loops:
                if isinstance(tile_size, str) and not tile_size.isdigit():
                    kernel_autotuned_blocks.add(f"BLOCK_{loop_var.upper()}")

            if kernel_autotuned_blocks:
                sorted_autotuned = sorted(list(kernel_autotuned_blocks))
                code += f",\n        # {', '.join(sorted_autotuned)} are provided by autotune"

            keyword_params = []

            for loop_var, _, _, tile_size in all_loops:
                block_param_name = f"BLOCK_{loop_var.upper()}"
                if isinstance(tile_size, str) and not tile_size.isdigit():
                    if block_param_name in kernel_autotuned_blocks:
                        keyword_params.append(
                            f"        # {block_param_name} is automatically set by autotune"
                        )
                    else:
                        keyword_params.append(f"        {block_param_name}=block_{loop_var}")
                else:
                    keyword_params.append(f"        {block_param_name}={tile_size}")

            for const_name in sorted(self.state.constants.keys()):
                keyword_params.append(f"        {const_name}={self.state.constants[const_name]}")

            if keyword_params:
                code += ",\n" + ",\n".join(keyword_params)

            code += "\n    )\n\n"

        code += "    # Return output tensors if needed\n"
        code += "    # This depends on your specific use case\n"
        code += "    pass\n"

        return code
