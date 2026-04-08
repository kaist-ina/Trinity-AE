import torch
from torch.export import export
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program


def _patch_relax_block_builder():
    block_builder_cls = relax.BlockBuilder

    if getattr(block_builder_cls, "_trinity_patch_applied", False):
        return

    func_stacks = {}

    def _stack(self):
        return func_stacks.setdefault(id(self), [])

    def patched_init(self, mod=None):
        self.__init_handle_by_constructor__(
            relax.block_builder._ffi_api.BlockBuilderCreate, mod
        )
        _stack(self)

    def patched_current_func(self):
        stack = _stack(self)
        if stack:
            return stack[-1]
        raise RuntimeError(
            "Cannot access BlockBuilder._func when outside a bb.function() block"
        )

    def patched_enter_function_scope(self, func_scope):
        block_builder_cls._stack.append(self)
        _stack(self).append(func_scope)
        self.begin_scope(func_scope._params)
        self._begin_binding_block()

    def patched_exit_function_scope(self, exc_type, exc_val, exc_tb):
        current_func = patched_current_func(self)
        is_emit_func_output_called = current_func._is_emit_func_output_called
        _stack(self).pop()

        assert block_builder_cls._stack
        assert block_builder_cls._stack[-1] is self
        block_builder_cls._stack.pop()

        if exc_type is None and not is_emit_func_output_called:
            raise RuntimeError("emit_func_output must be called in a relax function.")

    block_builder_cls.__init__ = patched_init
    block_builder_cls._func = property(patched_current_func)
    block_builder_cls._enter_function_scope = patched_enter_function_scope
    block_builder_cls._exit_function_scope = patched_exit_function_scope
    block_builder_cls._trinity_patch_applied = True


def _build_param_first_use_order(exported):
    """Return list of (pytorch_name, shape) in the order constants are
    first *used* in the graph — this matches the order build_main_func
    will assign const_1, const_2, …"""
    from torch.export.graph_signature import InputKind

    # Map placeholder arg name → (pytorch_name, shape)
    placeholder_info: dict[str, tuple[str, list[int]]] = {}
    user_input_names: set[str] = set()
    for spec in exported.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            user_input_names.add(spec.arg.name)
        elif spec.target is not None:
            # Get shape from the placeholder's fake tensor
            param = None
            if hasattr(exported, "state_dict"):
                param = exported.state_dict.get(spec.target)
            if param is None and hasattr(exported, "constants"):
                param = exported.constants.get(spec.target)
            shape = list(param.shape) if param is not None else []
            placeholder_info[spec.arg.name] = (spec.target, shape)

    # Walk graph ops to find first-use order of constant placeholders
    seen: set[str] = set()
    ordered: list[tuple[str, list[int]]] = []
    for node in exported.graph_module.graph.nodes:
        if node.op == "placeholder":
            continue
        for inp in node.all_input_nodes:
            if inp.op == "placeholder" and inp.name in placeholder_info and inp.name not in seen:
                seen.add(inp.name)
                ordered.append(placeholder_info[inp.name])
    # Append any remaining placeholders not referenced in ops
    for arg_name, info in placeholder_info.items():
        if arg_name not in seen:
            ordered.append(info)

    return ordered, user_input_names


def to_relax(model, example_input):
    """PyTorch 모델을 Relax IR로 변환"""
    _patch_relax_block_builder()
    model.eval()
    with torch.no_grad():
        # export 함수는 tuple of tensors를 받아야 함
        if not isinstance(example_input, tuple):
            example_input = (example_input,)
        exported = export(model, example_input)
    user_output_count = len(getattr(exported.graph_signature, "user_outputs", []) or [])

    # Build mapping: pytorch parameter name → IR const_N name
    param_order, user_input_names = _build_param_first_use_order(exported)
    param_name_map: dict[str, str] = {}
    for idx, (pytorch_name, _shape) in enumerate(param_order):
        param_name_map[pytorch_name] = f"const_{idx + 1}"
    # Map user inputs (keep their name as-is)
    for spec in exported.graph_signature.input_specs:
        if spec.arg.name in user_input_names:
            param_name_map[spec.arg.name] = spec.arg.name

    relax_mod = from_exported_program(exported, keep_params_as_input=True)
    return relax_mod, user_output_count, param_name_map

def to_tir(relax_mod):
    """Relax IR을 TIR로 lowering"""
    return relax.transform.LegalizeOps()(relax_mod)
