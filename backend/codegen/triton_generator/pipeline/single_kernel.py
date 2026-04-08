"""Single-kernel generation helpers."""

from __future__ import annotations

from ...AstNode import ASTNode


class SingleKernelPipeline:
    def generate_single_kernel(self, ast: ASTNode, kernel_name: str = "kernel") -> str:
        """Generate a single Triton kernel from AST."""
        self.state.kernel_accumulators = set()

        self.state.parallel_dims = []
        self.state.all_loops = []
        self.state.program_id_counter = 0
        self.state.tensors_used = set()
        self.state.temp_counter = 0
        self.state.offset_counter = 0
        self.state.loop_var_to_tensor_dim = {}
        self.state.loop_vars = {}

        self.state.stored_tensor_dims = {}
        self.state.load_cache = {}
        self.state.intermediate_tensor_indices = {}
        self.state.sloop_temp_vars = {}
        self.state.generated_masks = {}
        self.state.mask_scope_level = {}
        self.state.generated_indices = {}
        self.state.indices_scope_level = {}
        self.state.current_sloop_info = None
        self.state.kernel_accumulators = set()
        self.state.output_tensors = set()
        self.state.loop_instance_counter = 0
        self.state.stored_accumulators = set()
        self.state.current_loop_instance = None
        self.state.mask_loop_instance = {}
        self.state.indices_loop_instance = {}

        self.gen.analyzer.collect_tensors(ast)
        self.gen.analyzer.collect_intermediate_tensors(ast, in_ploop=False, ploop_var=None)
        self.gen.loops.collect_all_loops(ast)
        self.gen.indexer.analyze_loop_contexts(ast)
        self.gen.analyzer.analyze_parallel_store_dims(ast)

        cross_sloop_memory_tensors = self.gen.analyzer.identify_cross_sloop_memory_tensors(ast)
        accumulators = self.gen.analyzer.identify_accumulators(ast)

        fp32_tensors = self.gen.analyzer.identify_fp32_tensors(ast)
        self.state.fp32_tensors = fp32_tensors
        self.state.exp_tensors = fp32_tensors

        cross_sloop_tensors = self.gen.analyzer.identify_cross_sloop_tensors(ast)

        non_cross_sloop_accumulators = accumulators - cross_sloop_tensors
        cross_sloop_memory_tensors -= non_cross_sloop_accumulators

        self.state.intermediate_tensors -= cross_sloop_memory_tensors
        self.state.tensors_used.update(cross_sloop_memory_tensors)
        self.state.cross_sloop_memory_tensors = cross_sloop_memory_tensors

        kernel_code = self.gen.kernel.generate_header(kernel_name)

        self.state.indent_level = 1

        kernel_code += self.gen.analyzer.generate_intermediate_allocations(ast)

        all_accumulators = self.gen.analyzer.identify_accumulators(ast)

        for tensor in all_accumulators:
            must_use_memory = False

            if tensor in self.state.input_tensors or tensor in self.state.output_tensors:
                must_use_memory = True
            elif tensor in self.state.cross_kernel_tensors:
                must_use_memory = True
            elif tensor in self.state.cross_sloop_memory_tensors:
                must_use_memory = True

            if must_use_memory:
                self.state.kernel_accumulators.add(tensor)

        kernel_code += self.gen.analyzer.generate_kernel_accumulator_init()

        kernel_code += self.gen.dispatch.generate_node(ast)

        kernel_code += self.gen.analyzer.generate_kernel_accumulator_stores()
        self.state.indent_level = 0

        return kernel_code
