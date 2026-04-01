from typing import List, Optional

import tvm
try:
    from tvm import tir, relax
except ImportError:
    from tvm import tirx as tir, relax
import ir.AST as T

def build_main_func(
    relax_mod,
    primfunc_nodes: List[T.PrimFunc],
    apply_alloc: bool = True,
    user_output_count: Optional[int] = None,
) -> T.MainFunc:
    primfunc_map = {pf.name: pf for pf in primfunc_nodes}
    main_func = relax_mod["main"]

    input_tensors: List[T.TensorInfo] = []
    calls: List[T.PrimFuncCall] = []
    intermediate_tensors: List[T.TensorInfo] = []
    value_map = {}
    const_counter = [0]
    const_map = {}
    call_counts: dict[str, int] = {}

    def _fulltile_index(rank: int) -> T.Index:
        return T.Index([T.FullTile() for _ in range(rank)])

    def _make_identity_primfunc(name: str, input_info: T.TensorInfo, output_info: T.TensorInfo) -> T.PrimFunc:
        rank = len(output_info.shape)
        out_index = _fulltile_index(rank)
        if rank != len(input_info.shape):
            in_index = _fulltile_index(len(input_info.shape))
        else:
            in_index = out_index
        store = T.Store(T.Tensor(output_info.name), T.Load(T.Tensor(input_info.name), in_index), out_index)
        return T.PrimFunc(
            name=name,
            input_tensors=[input_info],
            output_tensor=output_info,
            spatial_axes=[],
            root_node=store,
            allocated_tensors=[],
        )

    for param in main_func.params:
        info = _struct_info_to_tensor_info(param.struct_info, param.name_hint)
        input_tensors.append(info)
        value_map[param.name_hint] = info

    body = main_func.body
    blocks = body.blocks if isinstance(body, relax.SeqExpr) else []
    for block in blocks:
        for binding in block.bindings:
            if not hasattr(binding, "var") or not hasattr(binding, "value"):
                continue
            if not isinstance(binding.var, relax.Var):
                continue
            call = binding.value
            if isinstance(call, relax.Tuple):
                tuple_values: List[T.TensorInfo] = []
                for field in call.fields:
                    val = _expr_to_value(field, value_map, const_counter, const_map, input_tensors)
                    if val is None:
                        if isinstance(field, relax.Var):
                            val = _struct_info_to_tensor_info(field.struct_info, field.name_hint)
                        else:
                            raise TypeError(
                                f"Unsupported tuple field type: {type(field).__name__} in relax.Tuple"
                            )
                    tuple_values.append(val)
                value_map[binding.var.name_hint] = tuple_values
                continue
            if isinstance(call, relax.Constant):
                const_info = _expr_to_value(call, value_map, const_counter, const_map, input_tensors)
                if const_info is None:
                    continue
                out_var = binding.var
                out_info = _struct_info_to_tensor_info(out_var.struct_info, out_var.name_hint)
                intermediate_tensors.append(out_info)
                primfunc = _make_identity_primfunc(f"const_assign_{out_info.name}", const_info, out_info)
                calls.append(
                    T.PrimFuncCall(
                        primfunc=primfunc,
                        out_var_tensor=out_info,
                        input_tensors=[const_info],
                        call_index=1,
                    )
                )
                value_map[out_info.name] = out_info
                continue
            if not isinstance(call, relax.Call):
                continue
            call_info = _extract_call_tir(call)
            if call_info is None:
                continue
            func_ref, args = call_info
            func_name = getattr(func_ref, "name_hint", None) or str(func_ref)
            primfunc = primfunc_map.get(func_name)
            if primfunc is None:
                continue

            out_var = binding.var
            out_info = _struct_info_to_tensor_info(out_var.struct_info, out_var.name_hint)
            intermediate_tensors.append(out_info)

            input_values: List[T.TensorInfo] = []
            for arg in args:
                val = _expr_to_value(arg, value_map, const_counter, const_map, input_tensors)
                if val is None:
                    if isinstance(arg, relax.Var):
                        val = _struct_info_to_tensor_info(arg.struct_info, arg.name_hint)
                    else:
                        val = T.TensorInfo(name="unknown", shape=[], dtype="unknown")
                input_values.append(val)

            call_counts[primfunc.name] = call_counts.get(primfunc.name, 0) + 1
            call_index = call_counts[primfunc.name]
            calls.append(
                T.PrimFuncCall(
                    primfunc=primfunc,
                    out_var_tensor=out_info,
                    input_tensors=input_values,
                    call_index=call_index,
                )
            )
            value_map[out_info.name] = out_info

    output_tensors: List[T.TensorInfo] = []
    ret_expr = body.body if isinstance(body, relax.SeqExpr) else body

    def _add_output(expr):
        if isinstance(expr, relax.Var):
            val = value_map.get(expr.name_hint)
            if isinstance(val, list):
                for item in val:
                    output_tensors.append(item)
                return
            if isinstance(val, T.TensorInfo):
                output_tensors.append(val)
            else:
                output_tensors.append(_struct_info_to_tensor_info(expr.struct_info, expr.name_hint))
        elif isinstance(expr, relax.Tuple):
            for field in expr.fields:
                _add_output(field)

    _add_output(ret_expr)
    if user_output_count is not None and user_output_count >= 0:
        output_tensors = output_tensors[-user_output_count:] if user_output_count > 0 else []

    main_func_ir = T.MainFunc(
        calls=calls,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        intermediate_tensors=intermediate_tensors,
    )

    return _apply_alloc_renames(main_func_ir) if apply_alloc else main_func_ir

def _apply_alloc_renames(main_func: T.MainFunc) -> T.MainFunc:
    updated_calls: List[T.PrimFuncCall] = []
    intermediate_tensors: List[T.TensorInfo] = list(main_func.intermediate_tensors)

    for call in main_func.calls:
        primfunc = call.primfunc
        if primfunc.allocated_tensors:
            if call.call_index == 1:
                intermediate_tensors.extend(primfunc.allocated_tensors)
            else:
                primfunc = _clone_primfunc_with_renamed_allocs(primfunc, call.call_index)
                intermediate_tensors.extend(primfunc.allocated_tensors)

        updated_calls.append(
            T.PrimFuncCall(
                primfunc=primfunc,
                out_var_tensor=call.out_var_tensor,
                input_tensors=call.input_tensors,
                call_index=call.call_index,
            )
        )

    return T.MainFunc(
        calls=updated_calls,
        input_tensors=main_func.input_tensors,
        output_tensors=main_func.output_tensors,
        intermediate_tensors=intermediate_tensors,
    )

def _clone_primfunc_with_renamed_allocs(primfunc: T.PrimFunc, call_index: int) -> T.PrimFunc:
    rename_map = {
        tensor.name: f"{tensor.name}_call_{call_index}"
        for tensor in primfunc.allocated_tensors
    }
    new_allocs = [_rename_tensor_info(t, rename_map) for t in primfunc.allocated_tensors]
    new_root = _rename_tensor_nodes(primfunc.root_node, rename_map)
    return T.PrimFunc(
        name=primfunc.name,
        input_tensors=primfunc.input_tensors,
        output_tensor=primfunc.output_tensor,
        spatial_axes=primfunc.spatial_axes,
        root_node=new_root,
        allocated_tensors=new_allocs,
    )

def _rename_tensor_info(tensor: T.TensorInfo, rename_map: dict[str, str]) -> T.TensorInfo:
    new_name = rename_map.get(tensor.name, tensor.name)
    return T.TensorInfo(name=new_name, shape=tensor.shape, dtype=tensor.dtype)

def _rename_tensor_nodes(node: T.ASTNode, rename_map: dict[str, str]) -> T.ASTNode:
    if isinstance(node, T.Tensor):
        return T.Tensor(rename_map.get(node.name, node.name))
    if isinstance(node, T.Index):
        return T.Index([_rename_tensor_nodes(idx, rename_map) for idx in node.indices])
    if isinstance(node, T.Load):
        return T.Load(_rename_tensor_nodes(node.tensor, rename_map), _rename_tensor_nodes(node.index, rename_map))
    if isinstance(node, T.Store):
        return T.Store(
            _rename_tensor_nodes(node.tensor, rename_map),
            _rename_tensor_nodes(node.value, rename_map),
            _rename_tensor_nodes(node.index, rename_map),
        )
    if isinstance(node, T.Loop):
        return T.Loop(
            _rename_tensor_nodes(node.start, rename_map),
            _rename_tensor_nodes(node.end, rename_map),
            node.tile_name,
            node.loop_var,
            _rename_tensor_nodes(node.body, rename_map),
        )
    if isinstance(node, T.Block):
        return T.Block([_rename_tensor_nodes(stmt, rename_map) for stmt in node.stmts])
    if isinstance(node, T.Seq):
        return T.Seq(
            _rename_tensor_nodes(node.left, rename_map),
            _rename_tensor_nodes(node.right, rename_map),
        )
    if isinstance(node, T.If):
        return T.If(
            _rename_tensor_nodes(node.cond, rename_map),
            _rename_tensor_nodes(node.then_branch, rename_map),
            _rename_tensor_nodes(node.else_branch, rename_map) if node.else_branch else None,
        )
    if isinstance(node, T.Let):
        return T.Let(
            _rename_tensor_nodes(node.tensor, rename_map),
            _rename_tensor_nodes(node.value, rename_map),
            _rename_tensor_nodes(node.body, rename_map),
        )
    if isinstance(node, T.Add):
        return T.Add(_rename_tensor_nodes(node.left, rename_map), _rename_tensor_nodes(node.right, rename_map))
    if isinstance(node, T.Sub):
        return T.Sub(_rename_tensor_nodes(node.left, rename_map), _rename_tensor_nodes(node.right, rename_map))
    if isinstance(node, T.Mul):
        return T.Mul(_rename_tensor_nodes(node.left, rename_map), _rename_tensor_nodes(node.right, rename_map))
    if isinstance(node, T.Div):
        return T.Div(_rename_tensor_nodes(node.left, rename_map), _rename_tensor_nodes(node.right, rename_map))
    if isinstance(node, T.Exp):
        return T.Exp(_rename_tensor_nodes(node.val, rename_map))
    if isinstance(node, T.Sqr):
        return T.Sqr(_rename_tensor_nodes(node.val, rename_map))
    if isinstance(node, T.Sqrt):
        return T.Sqrt(_rename_tensor_nodes(node.val, rename_map))
    if isinstance(node, T.Sigmoid):
        return T.Sigmoid(_rename_tensor_nodes(node.val, rename_map))
    if isinstance(node, T.Matmul):
        return T.Matmul(_rename_tensor_nodes(node.left, rename_map), _rename_tensor_nodes(node.right, rename_map))
    if isinstance(node, T.Take):
        return T.Take(
            _rename_tensor_nodes(node.data, rename_map),
            _rename_tensor_nodes(node.indices, rename_map),
            node.axis,
            _rename_tensor_nodes(node.index, rename_map),
        )
    if isinstance(node, T.ReduceSum):
        return T.ReduceSum(_rename_tensor_nodes(node.val, rename_map), node.axis)
    if isinstance(node, T.ReduceMax):
        return T.ReduceMax(_rename_tensor_nodes(node.val, rename_map), node.axis)
    if isinstance(node, T.ReduceMin):
        return T.ReduceMin(_rename_tensor_nodes(node.val, rename_map), node.axis)
    if isinstance(node, T.Concat):
        return T.Concat(
            _rename_tensor_nodes(node.a, rename_map),
            _rename_tensor_nodes(node.b, rename_map),
            node.axis,
        )
    if isinstance(node, T.Broadcast):
        return T.Broadcast(_rename_tensor_nodes(node.val, rename_map), node.axis)
    if isinstance(node, T.Permute3):
        return T.Permute3(
            _rename_tensor_nodes(node.val, rename_map),
            node.d0,
            node.d1,
            node.d2,
        )
    if isinstance(node, T.Squeeze):
        return T.Squeeze(_rename_tensor_nodes(node.val, rename_map), node.axis)
    if isinstance(node, T.Unsqueeze):
        return T.Unsqueeze(_rename_tensor_nodes(node.val, rename_map), node.axis)
    if isinstance(node, T.GenericBinary):
        return T.GenericBinary(
            node.op,
            _rename_tensor_nodes(node.left, rename_map),
            _rename_tensor_nodes(node.right, rename_map),
        )
    if isinstance(node, T.GenericCall):
        return T.GenericCall(node.func_name, [_rename_tensor_nodes(arg, rename_map) for arg in node.args])
    if isinstance(node, T.Cast):
        return T.Cast(node.dtype, _rename_tensor_nodes(node.val, rename_map))
    return node

def _extract_call_tir(call: relax.Call):
    if not isinstance(call, relax.Call):
        return None
    op_name = getattr(call.op, "name", None)
    if "call_tir" not in op_name:
        return None
    if not call.args:
        return None
    func_ref = call.args[0]
    args_expr = call.args[1] if len(call.args) > 1 else relax.Tuple([])
    if isinstance(args_expr, relax.Tuple):
        args = list(args_expr.fields)
    else:
        args = [args_expr]
    return func_ref, args

def _expr_to_value(expr, value_map, const_counter, const_map, input_tensors):
    if isinstance(expr, relax.Var):
        return value_map.get(expr.name_hint, _struct_info_to_tensor_info(expr.struct_info, expr.name_hint))
    if isinstance(expr, relax.Constant):
        try:
            const_key = tvm.ir.structural_hash(expr)
        except Exception:
            const_key = id(expr)
        if const_key in const_map:
            return const_map[const_key]
        const_counter[0] += 1
        name = f"const_{const_counter[0]}"
        info = _struct_info_to_tensor_info(expr.struct_info, name)
        const_map[const_key] = info
        input_tensors.append(info)
        return info
    if isinstance(expr, relax.TupleGetItem):
        base = expr.tuple_value
        if isinstance(base, relax.Var) and base.name_hint in value_map:
            return value_map[base.name_hint]
    return None

def _struct_info_to_tensor_info(struct_info, name_hint: str) -> T.TensorInfo:
    if isinstance(struct_info, relax.TensorStructInfo):
        shape = _relax_shape_to_list(struct_info.shape)
        return T.TensorInfo(name=name_hint, shape=shape, dtype=str(struct_info.dtype))
    return T.TensorInfo(name=name_hint, shape=[], dtype="unknown")

def _relax_shape_to_list(shape_expr) -> List[int]:
    if shape_expr is None:
        return []
    if isinstance(shape_expr, relax.ShapeExpr):
        dims = list(shape_expr.values)
    elif isinstance(shape_expr, (list, tuple)):
        dims = list(shape_expr)
    elif isinstance(shape_expr, tvm.ir.container.Array):
        dims = list(shape_expr)
    else:
        return []
    shape: List[int] = []
    for dim in dims:
        dim_val = _primexpr_to_int(dim)
        shape.append(dim_val if dim_val is not None else dim)
    return shape

def _primexpr_to_int(expr) -> Optional[int]:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if isinstance(expr, int):
        return expr
    return None
