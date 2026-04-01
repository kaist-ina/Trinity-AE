import inspect
import os
from typing import Any

import ir.AST as T
from core.from_tir import build_primfunc_nodes
from core.from_rir import build_main_func
from core.sequantialize import sequentialize_main_func
from core.to_ir import filter_identity_and_apply_alias
from utils.io_utils import export_main_func
from utils.ir_utils import (
    bind_main_func_calls,
    eliminate_dead_calls,
    eliminate_dead_seq_stores,
    inline_elementwise_op_calls,
    inline_shape_op_calls,
    normalize_main_func_axes,
    plan_fusion_groups,
    simplify_redundant_shape_ops,
    validate_fusion_groups,
)
from utils.test_utils import validate_main_func_errors
from utils.tir_utils import to_relax, to_tir


def export_model_ir(
    model: Any,
    example_inputs: Any,
    basename: str | None = None,
    output_dir: str = "./outputs",
    context: str | None = None,
    inline_shape_op: bool = True,
    inline_elementwise_op: bool = True,
    remove_short_loop_threshold: int = 64,
    decompose_nested_op_ratio: float = 0.3,
    flat_output: bool = False,
) -> tuple[T.MainFunc, list[str]]:
    """
    Convert a PyTorch model to Trinity IR and save artifacts under output_dir.
    """
    if basename is None or context is None:
        module_name = getattr(model.__class__, "__module__", "") or ""
        if module_name == "__main__":
            source_path = inspect.getsourcefile(model.__class__) or ""
            inferred = os.path.splitext(os.path.basename(source_path))[0] if source_path else "model"
        else:
            inferred = module_name.split(".")[-1] or "model"
        if basename is None:
            basename = inferred
        if context is None:
            context = inferred

    relax_mod, user_output_count = to_relax(model, example_inputs)
    tir_mod = to_tir(relax_mod)

    os.makedirs(f"{output_dir}/tvm", exist_ok=True)
    os.makedirs(f"{output_dir}/trinity", exist_ok=True)

    tir_path = f"{output_dir}/tvm/{basename}_tir.py"
    with open(tir_path, "w") as f:
        f.write(tir_mod.script())
    print(f"TIR saved to: {tir_path}")

    primfunc_nodes = build_primfunc_nodes(
        tir_mod,
        tmp_ratio=decompose_nested_op_ratio,
        remove_short_loop_threshold=remove_short_loop_threshold,
    )

    main_func_ir = build_main_func(tir_mod, primfunc_nodes, user_output_count=user_output_count)
    main_func_ir = sequentialize_main_func(main_func_ir)
    main_func_ir = bind_main_func_calls(main_func_ir)
    main_func_ir = normalize_main_func_axes(main_func_ir)
    main_func_ir = simplify_redundant_shape_ops(main_func_ir)
    main_func_ir = filter_identity_and_apply_alias(main_func_ir)
    if inline_shape_op:
        main_func_ir = inline_shape_op_calls(main_func_ir)
        main_func_ir = simplify_redundant_shape_ops(main_func_ir)
    if inline_elementwise_op:
        main_func_ir = inline_elementwise_op_calls(main_func_ir)
        main_func_ir = simplify_redundant_shape_ops(main_func_ir)
    main_func_ir = eliminate_dead_seq_stores(main_func_ir)
    main_func_ir = eliminate_dead_calls(main_func_ir)

    fusion_groups = plan_fusion_groups(main_func_ir)
    errors = validate_main_func_errors(main_func_ir, context=context or basename)
    errors.extend(validate_fusion_groups(main_func_ir, fusion_groups))
    export_main_func(main_func_ir, output_dir, basename, fusion_groups=fusion_groups, flat_output=flat_output)
    return main_func_ir, errors
