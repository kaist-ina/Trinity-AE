from typing import List, Optional
try:
    from tvm import tir
except ImportError:
    from tvm import tirx as tir
import ir.AST as T
from utils.ir_utils import (
    remove_short_loop_nodes,
    remove_smallest_inner_loops,
    remove_let_nodes,
    decompose_operations,
)

def build_primfunc_nodes(
    tir_mod,
    tmp_ratio: float = 0.3,
    remove_short_loop_threshold: int = 64,
    decompose_ops: bool = True,
    debug: bool = False,
) -> List[T.PrimFunc]:
    primfunc_nodes: List[T.PrimFunc] = []

    for global_var, func in tir_mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue

        func_name = global_var.name_hint
        pattern_tracker = {"hit": False, "name_patterns": [], "block_patterns": []}
        root_node = _convert_to_ast(
            func,
            func_name=func_name,
            debug=debug,
            pattern_tracker=pattern_tracker,
        )
        if debug:
            name_patterns = pattern_tracker["name_patterns"] or ["none"]
            block_patterns = pattern_tracker["block_patterns"] or ["none"]
            print(func_name)
            print(f"name_pattern: {', '.join(name_patterns)}")
            print(f"block_pattern: {', '.join(block_patterns)}")
            print()
        root_node = remove_let_nodes(root_node)

        axes = _extract_axes(func)
        input_tensors = _get_input_tensor_infos(func)
        output_tensor = _get_output_tensor_info(func)
        allocated_tensors = _collect_allocated_tensors(func)
        tensor_info_map = {t.name: t for t in input_tensors + [output_tensor] + allocated_tensors}

        tmp_tensors: List[T.TensorInfo] = []
        if decompose_ops:
            func_name_lower = func_name.lower()
            exclude_ops = (
                "reshape",
                "sum",
                "matmul",
                "broadcast",
                "transpose",
                "expand_dims",
                "multiply",
                "divide",
                "add",
                "tir_exp",
                "concat",
                "mean",
            )
            if not any(op in func_name_lower for op in exclude_ops):
                root_node, tmp_tensors = decompose_operations(
                    root_node, tensor_info_map, ratio=tmp_ratio
                )
            if tmp_tensors:
                allocated_tensors.extend(tmp_tensors)
        root_node = remove_short_loop_nodes(root_node, threshold=remove_short_loop_threshold)
        root_node = remove_smallest_inner_loops(root_node, max_extent=384)


        primfunc_nodes.append(
            T.PrimFunc(
                name=func_name,
                input_tensors=input_tensors,
                output_tensor=output_tensor,
                spatial_axes=axes,
                root_node=root_node,
                allocated_tensors=allocated_tensors,
            )
        )

    return primfunc_nodes


def _extract_axes(prim_func: tir.PrimFunc) -> List[str]:
    axes: List[str] = []
    seen = set()

    def visit(stmt):
        if not isinstance(stmt, tir.SBlock):
            return
        for iter_var in stmt.iter_vars:
            var_name = iter_var.var.name
            if var_name in seen:
                continue
            seen.add(var_name)
            axes.append(var_name)

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    return axes


def _accumulate_with_identity(acc_value: T.ASTNode, update_value: T.ASTNode) -> T.ASTNode:
    """Make reduction-style accumulation explicit as (+ (* acc 1) update)."""
    return T.Add(T.Mul(acc_value, T.Const(1)), update_value)


def _get_output_tensor_info(prim_func: tir.PrimFunc) -> T.TensorInfo:
    if hasattr(prim_func, "ret_buffer") and prim_func.ret_buffer is not None:
        return _buffer_to_tensor_info(prim_func.ret_buffer)

    buffer_params = [p for p in prim_func.params if p in prim_func.buffer_map]
    if not buffer_params:
        return T.TensorInfo(name="unknown", shape=[], dtype="unknown")
    return _buffer_to_tensor_info(prim_func.buffer_map[buffer_params[-1]])

def _get_input_tensor_infos(prim_func: tir.PrimFunc) -> List[T.TensorInfo]:
    buffer_params = [p for p in prim_func.params if p in prim_func.buffer_map]
    if not buffer_params:
        return []
    input_params = buffer_params[:-1]
    return [_buffer_to_tensor_info(prim_func.buffer_map[p]) for p in input_params]

def _collect_allocated_tensors(prim_func: tir.PrimFunc) -> List[T.TensorInfo]:
    allocated: dict[str, T.TensorInfo] = {}
    alloc_stmt_types = tuple(
        t for t in (getattr(tir, "Allocate", None), getattr(tir, "AllocBuffer", None)) if t is not None
    )
    decl_buffer_type = getattr(tir, "DeclBuffer", None)

    def _register_alloc(name: str, shape: List[int], dtype: str) -> None:
        safe_name = name.replace(".", "_")
        info = T.TensorInfo(name=safe_name, shape=shape, dtype=dtype)
        allocated[safe_name] = info
        allocated[name] = info

    def visit(stmt):
        if isinstance(stmt, tir.SBlock):
            for buf in stmt.alloc_buffers:
                info = _buffer_to_tensor_info(buf)
                _register_alloc(info.name, info.shape, info.dtype)
        elif alloc_stmt_types and isinstance(stmt, alloc_stmt_types):
            if hasattr(stmt, "buffer"):
                info = _buffer_to_tensor_info(stmt.buffer)
                _register_alloc(info.name, info.shape, info.dtype)
            else:
                shape: List[int] = []
                for ext in stmt.extents:
                    ext_val = _primexpr_to_int(ext)
                    if ext_val is None:
                        shape.append(ext)
                    else:
                        shape.append(ext_val)
                _register_alloc(stmt.buffer_var.name, shape, str(stmt.dtype))
        elif decl_buffer_type is not None and isinstance(stmt, decl_buffer_type):
            info = _buffer_to_tensor_info(stmt.buffer)
            _register_alloc(info.name, info.shape, info.dtype)

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    return list(allocated.values())

def _primexpr_to_int(expr) -> Optional[int]:
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if isinstance(expr, int):
        return expr
    return None

def _buffer_to_tensor_info(buf: tir.Buffer) -> T.TensorInfo:
    shape: List[int] = []
    for dim in buf.shape:
        dim_val = _primexpr_to_int(dim)
        if dim_val is None:
            shape.append(dim)
        else:
            shape.append(dim_val)
    safe_name = buf.name.replace(".", "_")
    return T.TensorInfo(name=safe_name, shape=shape, dtype=str(buf.dtype))

def _get_input_output_tensors(prim_func: tir.PrimFunc) -> tuple[T.TensorInfo, T.TensorInfo]:
    buffer_params = [p for p in prim_func.params if p in prim_func.buffer_map]
    if not buffer_params:
        unknown = T.TensorInfo(name="unknown", shape=[], dtype="unknown")
        return unknown, unknown
    input_buf = prim_func.buffer_map[buffer_params[0]]
    output_buf = prim_func.buffer_map[buffer_params[-1]]
    return _buffer_to_tensor_info(input_buf), _buffer_to_tensor_info(output_buf)

def _find_first_store(prim_func: tir.PrimFunc) -> Optional[tir.BufferStore]:
    store_ref: Optional[tir.BufferStore] = None

    def visit(stmt):
        nonlocal store_ref
        if store_ref is None and isinstance(stmt, tir.BufferStore):
            store_ref = stmt

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    return store_ref

def _find_first_store_in_stmt(stmt) -> Optional[tir.BufferStore]:
    store_ref: Optional[tir.BufferStore] = None

    def visit(node):
        nonlocal store_ref
        if store_ref is None and isinstance(node, tir.BufferStore):
            store_ref = node

    tir.stmt_functor.post_order_visit(stmt, visit)
    return store_ref

def _find_first_load(expr: tir.PrimExpr) -> Optional[tir.BufferLoad]:
    load_ref: Optional[tir.BufferLoad] = None

    def visit(node):
        nonlocal load_ref
        if load_ref is None and isinstance(node, tir.BufferLoad):
            load_ref = node

    tir.stmt_functor.post_order_visit(tir.Evaluate(expr), visit)
    return load_ref

def _infer_op_name_from_blocks(prim_func: tir.PrimFunc, output_name: str) -> Optional[str]:
    hint: Optional[str] = None

    def visit(stmt):
        nonlocal hint
        if hint is not None:
            return
        if not isinstance(stmt, tir.SBlock):
            return
        block_name = getattr(stmt, "name_hint", None) or getattr(stmt, "name", None) or ""
        block_name = str(block_name).lower()
        if not block_name:
            return
        writes = getattr(stmt, "writes", None) or []
        writes_output = any(
            getattr(region.buffer, "name", None) == output_name
            for region in writes
            if hasattr(region, "buffer")
        )
        if not writes_output:
            return
        if "reshape" in block_name:
            hint = "reshape"
        elif "broadcast" in block_name:
            hint = "broadcast"
        elif "transpose" in block_name:
            hint = "transpose"
        elif "concat" in block_name:
            hint = "concat"

    tir.stmt_functor.post_order_visit(prim_func.body, visit)
    return hint

def _block_io_tensors(block: tir.SBlock) -> Optional[tuple[T.TensorInfo, T.TensorInfo]]:
    reads = getattr(block, "reads", None) or []
    writes = getattr(block, "writes", None) or []
    if not reads or not writes:
        return None
    input_buf = reads[0].buffer
    output_buf = writes[0].buffer
    return _buffer_to_tensor_info(input_buf), _buffer_to_tensor_info(output_buf)

def _block_concat_tensors(block: tir.SBlock) -> Optional[tuple[T.TensorInfo, T.TensorInfo, T.TensorInfo]]:
    reads = getattr(block, "reads", None) or []
    writes = getattr(block, "writes", None) or []
    if len(reads) < 2 or not writes:
        return None
    return (
        _buffer_to_tensor_info(reads[0].buffer),
        _buffer_to_tensor_info(reads[1].buffer),
        _buffer_to_tensor_info(writes[0].buffer),
    )

def _try_convert_special_primfunc(
    prim_func: tir.PrimFunc,
    func_name: Optional[str] = None,
    debug: bool = False,
    pattern_tracker: Optional[dict] = None,
) -> Optional[T.ASTNode]:
    func_name = func_name or str(prim_func.attrs.get("global_symbol", ""))
    func_name_lower = func_name.lower()

    if "concat" in func_name_lower:
        input_tensors = _get_input_tensor_infos(prim_func)
        if len(input_tensors) < 2:
            return None
        a_tensor, b_tensor = input_tensors[0], input_tensors[1]
        out_tensor = _get_output_tensor_info(prim_func)
        concat_loop = _concat_to_loop_ast(a_tensor, b_tensor, out_tensor)
        if concat_loop is None:
            return None
        if pattern_tracker is not None:
            pattern_tracker["hit"] = True
            pattern_tracker["name_patterns"].append("concat")
        return concat_loop

    if "slice_scatter" in func_name_lower:
        input_tensors = _get_input_tensor_infos(prim_func)
        if len(input_tensors) < 2:
            return None
        out_tensor = _get_output_tensor_info(prim_func)
        scatter_loop = _slice_scatter_to_loop_ast(input_tensors, out_tensor)
        if scatter_loop is None:
            return None
        if pattern_tracker is not None:
            pattern_tracker["hit"] = True
            pattern_tracker["name_patterns"].append("slice_scatter")
        return scatter_loop

    if "strided_slice" in func_name_lower:
        io_tensors = _get_input_output_tensors(prim_func)
        input_tensor, output_tensor = io_tensors
        slice_loop = _strided_slice_to_loop_ast(input_tensor, output_tensor)
        if slice_loop is None:
            return None
        if pattern_tracker is not None:
            pattern_tracker["hit"] = True
            pattern_tracker["name_patterns"].append("strided_slice")
        return slice_loop

    if "tir_abs" in func_name_lower or func_name_lower == "abs":
        input_tensors = _get_input_tensor_infos(prim_func)
        if len(input_tensors) < 1:
            return None
        output_tensor = _get_output_tensor_info(prim_func)
        abs_loop = _unary_to_loop_ast(input_tensors[0], output_tensor, "abs")
        if abs_loop is None:
            return None
        if pattern_tracker is not None:
            pattern_tracker["hit"] = True
            pattern_tracker["name_patterns"].append("tir_abs")
        return abs_loop

    import re
    if re.match(r"^mean\d*$", func_name_lower):
        input_tensors = _get_input_tensor_infos(prim_func)
        if len(input_tensors) < 1:
            return None
        input_tensor = input_tensors[0]
        output_tensor = _get_output_tensor_info(prim_func)
        mean_loop = _mean_to_loop_ast(prim_func, input_tensor, output_tensor)
        if mean_loop is None:
            return None
        if pattern_tracker is not None:
            pattern_tracker["hit"] = True
            pattern_tracker["name_patterns"].append("mean")
        return mean_loop

    return None

def _try_convert_special_block_loop(
    stmt: tir.For,
    prim_func: Optional[tir.PrimFunc] = None,
    func_name: Optional[str] = None,
    debug: bool = False,
    pattern_tracker: Optional[dict] = None,
) -> Optional[T.ASTNode]:
    cur = stmt
    loop_meta: List[tuple[str, int]] = []
    while isinstance(cur, tir.For):
        extent = _primexpr_to_int(cur.extent)
        if extent is None:
            return None
        loop_meta.append((cur.loop_var.name, extent))
        cur = cur.body

    if isinstance(cur, tir.SBlockRealize):
        block = cur.block
    elif isinstance(cur, tir.SBlock):
        block = cur
    else:
        return None

    block_name = getattr(block, "name_hint", None) or getattr(block, "name", None) or ""
    block_name = str(block_name).lower()
    op_name = None
    special_keys = ("transpose", "broadcast", "reshape", "expand_dims")
    for key in special_keys:
        if key in block_name:
            op_name = key
            break

    if op_name is None and prim_func is not None:
        name = (func_name or str(prim_func.attrs.get("global_symbol", ""))).lower()
        for key in special_keys:
            if key in name:
                op_name = key
                break
        if op_name is None:
            output_tensor = _get_output_tensor_info(prim_func)
            block_hint = _infer_op_name_from_blocks(prim_func, output_tensor.name)
            if block_hint:
                op_name = block_hint

    def _resolve_io() -> Optional[tuple[T.TensorInfo, T.TensorInfo]]:
        if prim_func is not None:
            return _get_input_output_tensors(prim_func)
        return _block_io_tensors(block)

    if op_name == "transpose":
        io_tensors = _resolve_io()
        if io_tensors is None:
            return None
        input_tensor, output_tensor = io_tensors
        store = _find_first_store_in_stmt(block.body)
        permutation = _infer_permutation(store) if store is not None else []
        transpose_loop = _transpose_to_loop_ast(input_tensor, output_tensor, permutation)
        if transpose_loop is None:
            raise RuntimeError("transpose conversion failed")
        if pattern_tracker is not None:
            pattern_tracker["hit"] = True
            pattern_tracker["name_patterns"].append("transpose")
        return transpose_loop
    if op_name == "broadcast":
        io_tensors = _resolve_io()
        if io_tensors is None:
            return None
        input_tensor, output_tensor = io_tensors
        broadcast_loop = _broadcast_to_loop_ast(input_tensor, output_tensor)
        if broadcast_loop is None:
            raise RuntimeError("broadcast conversion failed")
        if pattern_tracker is not None:
            pattern_tracker["hit"] = True
            pattern_tracker["name_patterns"].append("broadcast")
        return broadcast_loop
    if op_name in ("reshape", "expand_dims"):
        io_tensors = _resolve_io()
        if io_tensors is None:
            return None
        input_tensor, output_tensor = io_tensors
        reshape_loop = _reshape_to_loop_ast(input_tensor, output_tensor)
        if reshape_loop is None:
            raise RuntimeError(f"{op_name} conversion failed")
        if pattern_tracker is not None:
            pattern_tracker["hit"] = True
            pattern_tracker["name_patterns"].append(op_name)
        return reshape_loop
    if "take" in block_name:
        store = _find_first_store_in_stmt(block.body)
        if store is None or not isinstance(store.value, tir.BufferLoad):
            return None

        data_load = store.value
        data_buf = data_load.buffer
        axis = None
        indices_buf = None
        indices_load = None
        for idx_pos, idx_expr in enumerate(data_load.indices):
            if isinstance(idx_expr, tir.BufferLoad):
                indices_load = idx_expr
                indices_buf = idx_expr.buffer
                axis = idx_pos
                break

        if axis is None or indices_buf is None or indices_load is None:
            return None
        if len(indices_load.indices) != 1:
            return None

        out_indices: List[T.ASTNode] = []
        for idx in store.indices:
            if isinstance(idx, tir.Var):
                out_indices.append(T.Tile(idx.name))
            elif isinstance(idx, (tir.IntImm, tir.FloatImm)):
                out_indices.append(T.Const(idx.value))
            else:
                return None

        out_index = T.Index(out_indices)
        data_tensor = T.Tensor(data_buf.name.replace(".", "_"))
        indices_tensor = T.Tensor(indices_buf.name.replace(".", "_"))
        take_value = T.Take(data_tensor, indices_tensor, axis, out_index)
        out_tensor = T.Tensor(store.buffer.name.replace(".", "_"))
        store_node = T.Store(out_tensor, take_value, out_index)
        for loop_var, extent in reversed(loop_meta):
            store_node = T.Loop(T.Const(0), T.Const(extent), f"tile_{loop_var}", loop_var, store_node)
        return store_node
    return None

def _try_convert_matmul_block(block: tir.SBlock, visit_expr) -> Optional[T.ASTNode]:
    if not isinstance(block.body, tir.BufferStore):
        return None

    body_store = block.body
    if not isinstance(body_store.value, tir.Add):
        return None

    add_left = body_store.value.a
    add_right = body_store.value.b
    out_load = None
    other = None
    if isinstance(add_left, tir.BufferLoad) and add_left.buffer == body_store.buffer and list(add_left.indices) == list(body_store.indices):
        out_load = add_left
        other = add_right
    elif isinstance(add_right, tir.BufferLoad) and add_right.buffer == body_store.buffer and list(add_right.indices) == list(body_store.indices):
        out_load = add_right
        other = add_left
    else:
        return None

    if not isinstance(other, tir.Mul):
        return None

    def _pick_load(expr) -> Optional[tir.BufferLoad]:
        return expr if isinstance(expr, tir.BufferLoad) else None

    lhs = _pick_load(other.a)
    rhs = _pick_load(other.b)
    if lhs is None or rhs is None:
        return None

    reduce_axes = []
    for iter_var in block.iter_vars:
        iter_type = iter_var.iter_type
        reduce_kinds = {
            getattr(tir.IterVar, "Reduce", None),
            getattr(tir.IterVar, "CommReduce", None),
        }
        if iter_type in reduce_kinds or "reduce" in str(iter_type).lower():
            reduce_axes.append(iter_var.var)

    if len(reduce_axes) != 1:
        return None
    reduce_var = reduce_axes[0]

    def _index_expr_to_ast(idx_expr: tir.PrimExpr) -> T.ASTNode:
        if isinstance(idx_expr, tir.Var):
            return T.Tile(idx_expr.name)
        if isinstance(idx_expr, tir.IntImm) and int(idx_expr.value) == 0:
            return T.FullTile()
        if isinstance(idx_expr, tir.Sub) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value) * -1)
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value))
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.b, tir.Var) and isinstance(idx_expr.a, tir.IntImm):
            return T.TileOffset(idx_expr.b.name, int(idx_expr.a.value))
        return visit_expr(idx_expr)

    if any(isinstance(idx, tir.Var) and idx.same_as(reduce_var) for idx in body_store.indices):
        return None

    if not any(isinstance(idx, tir.Var) and idx.same_as(reduce_var) for idx in lhs.indices):
        return None
    if not any(isinstance(idx, tir.Var) and idx.same_as(reduce_var) for idx in rhs.indices):
        return None

    out_index_nodes = [_index_expr_to_ast(idx) for idx in body_store.indices]
    lhs_index_nodes = [_index_expr_to_ast(idx) for idx in lhs.indices]
    rhs_index_nodes = [_index_expr_to_ast(idx) for idx in rhs.indices]

    out_tensor = T.Tensor(body_store.buffer.name.replace(".", "_"))
    out_index = T.Index(out_index_nodes)
    out_load_node = T.Load(out_tensor, out_index)

    lhs_tensor = T.Tensor(lhs.buffer.name.replace(".", "_"))
    rhs_tensor = T.Tensor(rhs.buffer.name.replace(".", "_"))
    lhs_load = T.Load(lhs_tensor, T.Index(lhs_index_nodes))
    rhs_load = T.Load(rhs_tensor, T.Index(rhs_index_nodes))

    matmul_value = T.Matmul(lhs_load, rhs_load)
    return T.Store(out_tensor, _accumulate_with_identity(out_load_node, matmul_value), out_index)

def _try_convert_arange_block(block: tir.SBlock) -> Optional[T.ASTNode]:
    if not isinstance(block.body, tir.BufferStore):
        return None

    if len(block.iter_vars) != 1:
        return None
    iter_var = block.iter_vars[0].var

    store = block.body
    if len(store.indices) != 1:
        return None
    if not isinstance(store.indices[0], tir.Var):
        return None
    if not store.indices[0].same_as(iter_var):
        return None

    value = store.value
    if isinstance(value, tir.Cast):
        value = value.value
    if not isinstance(value, tir.Var) or not value.same_as(iter_var):
        return None

    out_tensor = T.Tensor(store.buffer.name.replace(".", "_"))
    return T.Store(out_tensor, T.Arange(iter_var.name), T.Index([T.Tile(iter_var.name)]))

def _strip_cast_load(expr):
    if isinstance(expr, tir.BufferLoad):
        return expr
    if isinstance(expr, tir.Cast) and isinstance(expr.value, tir.BufferLoad):
        return expr.value
    return None

def _collect_let_bindings(stmt) -> dict[str, tir.PrimExpr]:
    bindings: dict[str, tir.PrimExpr] = {}
    if stmt is None:
        return bindings
    let_stmt_type = getattr(tir, "LetStmt", None)
    if let_stmt_type is None:
        return bindings

    def visit(s):
        if isinstance(s, let_stmt_type):
            bindings[s.var.name] = s.value

    tir.stmt_functor.post_order_visit(stmt, visit)
    return bindings

def _find_reduce_axis_in_expr(expr: tir.PrimExpr, reduce_var: tir.Var) -> Optional[int]:
    axis_index = None

    def visit(e):
        nonlocal axis_index
        if axis_index is not None:
            return
        if isinstance(e, tir.BufferLoad):
            for idx, axis in enumerate(e.indices):
                if isinstance(axis, tir.Var) and axis.same_as(reduce_var):
                    axis_index = idx
                    return
        if isinstance(e, tir.PrimExpr):
            if hasattr(e, "indices"):
                for child in e.indices:
                    visit(child)
            if hasattr(e, "args"):
                for child in e.args:
                    visit(child)
            if hasattr(e, "a"):
                visit(e.a)
            if hasattr(e, "b"):
                visit(e.b)
            if hasattr(e, "value"):
                visit(e.value)
            if hasattr(e, "index"):
                visit(e.index)

    visit(expr)
    return axis_index

def _try_convert_reducesum_block(block: tir.SBlock, visit_expr) -> Optional[T.ASTNode]:
    if block.init is None:
        return None
    if not isinstance(block.body, tir.BufferStore):
        return None

    let_bindings = _collect_let_bindings(block.body)

    init_store = block.init
    if isinstance(init_store, tir.SBlockRealize):
        init_store = init_store.block.body
    if isinstance(init_store, tir.SeqStmt):
        init_store = init_store.seq[0] if init_store.seq else None
    if not isinstance(init_store, tir.BufferStore):
        return None

    body_store = block.body
    value_expr = body_store.value
    if isinstance(value_expr, tir.Var):
        value_expr = let_bindings.get(value_expr.name, value_expr)
    if not isinstance(value_expr, tir.Add):
        return None

    add_left = value_expr.a
    add_right = value_expr.b

    out_load = None
    in_load = None
    left_load = _strip_cast_load(add_left)
    right_load = _strip_cast_load(add_right)
    if isinstance(left_load, tir.BufferLoad) and left_load.buffer == body_store.buffer and list(left_load.indices) == list(body_store.indices):
        out_load = left_load
        in_load = add_right
    elif isinstance(right_load, tir.BufferLoad) and right_load.buffer == body_store.buffer and list(right_load.indices) == list(body_store.indices):
        out_load = right_load
        in_load = add_left

    if out_load is None:
        return None

    reduce_axes = []
    spatial_axes = []
    for iter_var in block.iter_vars:
        iter_type = iter_var.iter_type
        reduce_kinds = {
            getattr(tir.IterVar, "Reduce", None),
            getattr(tir.IterVar, "CommReduce", None),
        }
        if iter_type in reduce_kinds or "reduce" in str(iter_type).lower():
            reduce_axes.append(iter_var.var)
        else:
            spatial_axes.append(iter_var.var)

    if len(reduce_axes) != 1:
        return None

    reduce_var = reduce_axes[0]
    def _index_expr_to_ast(idx_expr: tir.PrimExpr) -> T.ASTNode:
        if isinstance(idx_expr, tir.Var):
            return T.Tile(idx_expr.name)
        if isinstance(idx_expr, tir.IntImm) and int(idx_expr.value) == 0:
            return T.FullTile()
        if isinstance(idx_expr, tir.Sub) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value) * -1)
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value))
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.b, tir.Var) and isinstance(idx_expr.a, tir.IntImm):
            return T.TileOffset(idx_expr.b.name, int(idx_expr.a.value))
        return visit_expr(idx_expr)

    out_index_nodes: List[T.ASTNode] = []
    for idx in body_store.indices:
        if isinstance(idx, tir.Var) and idx.same_as(reduce_var):
            return None
        out_index_nodes.append(_index_expr_to_ast(idx))

    reduce_axis = _find_reduce_axis_in_expr(in_load, reduce_var)
    if reduce_axis is None:
        return None
    reduce_value = T.ReduceSum(visit_expr(in_load), reduce_axis)
    out_tensor = T.Tensor(body_store.buffer.name.replace(".", "_"))
    out_index = T.Index(out_index_nodes)
    out_load = T.Load(out_tensor, out_index)
    return T.Store(out_tensor, _accumulate_with_identity(out_load, reduce_value), out_index)

def _try_convert_multi_reducesum_block(block: tir.SBlock, visit_expr) -> Optional[T.ASTNode]:
    if block.init is None:
        return None

    def _collect_buffer_stores(stmt) -> List[tir.BufferStore]:
        stores: List[tir.BufferStore] = []
        if stmt is None:
            return stores

        def visit(s):
            if isinstance(s, tir.BufferStore):
                stores.append(s)

        tir.stmt_functor.post_order_visit(stmt, visit)
        return stores

    init_store = block.init
    if isinstance(init_store, tir.SBlockRealize):
        init_store = init_store.block.body
    init_stores = _collect_buffer_stores(init_store)
    if not init_stores:
        return None

    let_bindings = _collect_let_bindings(block.body)

    init_buffers = {s.buffer for s in init_stores}
    body_stores = _collect_buffer_stores(block.body)
    target_stores = [s for s in body_stores if s.buffer in init_buffers]
    if not target_stores:
        return None

    reduce_axes = []
    for iter_var in block.iter_vars:
        iter_type = iter_var.iter_type
        reduce_kinds = {
            getattr(tir.IterVar, "Reduce", None),
            getattr(tir.IterVar, "CommReduce", None),
        }
        if iter_type in reduce_kinds or "reduce" in str(iter_type).lower():
            reduce_axes.append(iter_var.var)

    if len(reduce_axes) != 1:
        return None
    reduce_var = reduce_axes[0]

    def _index_expr_to_ast(idx_expr: tir.PrimExpr) -> T.ASTNode:
        if isinstance(idx_expr, tir.Var):
            return T.Tile(idx_expr.name)
        if isinstance(idx_expr, tir.IntImm) and int(idx_expr.value) == 0:
            return T.FullTile()
        if isinstance(idx_expr, tir.Sub) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value) * -1)
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value))
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.b, tir.Var) and isinstance(idx_expr.a, tir.IntImm):
            return T.TileOffset(idx_expr.b.name, int(idx_expr.a.value))
        return visit_expr(idx_expr)

    stores: List[T.ASTNode] = []
    for body_store in target_stores:
        value_expr = body_store.value
        if isinstance(value_expr, tir.Var):
            value_expr = let_bindings.get(value_expr.name, value_expr)
        if not isinstance(value_expr, tir.Add):
            return None
        add_left = value_expr.a
        add_right = value_expr.b

        out_load = None
        in_expr = None
        left_load = _strip_cast_load(add_left)
        right_load = _strip_cast_load(add_right)
        if isinstance(left_load, tir.BufferLoad) and left_load.buffer == body_store.buffer and list(left_load.indices) == list(body_store.indices):
            out_load = left_load
            in_expr = add_right
        elif isinstance(right_load, tir.BufferLoad) and right_load.buffer == body_store.buffer and list(right_load.indices) == list(body_store.indices):
            out_load = right_load
            in_expr = add_left

        if out_load is None or in_expr is None:
            return None

        if any(isinstance(idx, tir.Var) and idx.same_as(reduce_var) for idx in body_store.indices):
            return None

        reduce_axis = _find_reduce_axis_in_expr(in_expr, reduce_var)
        if reduce_axis is None:
            return None

        out_index_nodes = [_index_expr_to_ast(idx) for idx in body_store.indices]
        out_tensor = T.Tensor(body_store.buffer.name.replace(".", "_"))
        out_index = T.Index(out_index_nodes)
        out_load_node = T.Load(out_tensor, out_index)
        reduce_value = T.ReduceSum(visit_expr(in_expr), reduce_axis)
        stores.append(T.Store(out_tensor, _accumulate_with_identity(out_load_node, reduce_value), out_index))

    if not stores:
        return None
    if len(stores) == 1:
        return stores[0]
    return T.Block(stores)

def _try_convert_reducemaxmin_block(block: tir.SBlock, visit_expr) -> Optional[T.ASTNode]:
    if block.init is None:
        return None
    if not isinstance(block.body, tir.BufferStore):
        return None

    let_bindings = _collect_let_bindings(block.body)

    init_store = block.init
    if isinstance(init_store, tir.SBlockRealize):
        init_store = init_store.block.body
    if isinstance(init_store, tir.SeqStmt):
        init_store = init_store.seq[0] if init_store.seq else None
    if not isinstance(init_store, tir.BufferStore):
        return None

    body_store = block.body
    value_expr = body_store.value
    if isinstance(value_expr, tir.Var):
        value_expr = let_bindings.get(value_expr.name, value_expr)
    if not isinstance(value_expr, tir.Max) and not isinstance(value_expr, tir.Min):
        return None

    is_max = isinstance(value_expr, tir.Max)
    op_left = value_expr.a
    op_right = value_expr.b

    out_load = None
    in_expr = None
    left_load = _strip_cast_load(op_left)
    right_load = _strip_cast_load(op_right)
    if isinstance(left_load, tir.BufferLoad) and left_load.buffer == body_store.buffer and list(left_load.indices) == list(body_store.indices):
        out_load = left_load
        in_expr = op_right
    elif isinstance(right_load, tir.BufferLoad) and right_load.buffer == body_store.buffer and list(right_load.indices) == list(body_store.indices):
        out_load = right_load
        in_expr = op_left

    if out_load is None or in_expr is None:
        return None

    reduce_axes = []
    for iter_var in block.iter_vars:
        iter_type = iter_var.iter_type
        reduce_kinds = {
            getattr(tir.IterVar, "Reduce", None),
            getattr(tir.IterVar, "CommReduce", None),
        }
        if iter_type in reduce_kinds or "reduce" in str(iter_type).lower():
            reduce_axes.append(iter_var.var)

    if len(reduce_axes) != 1:
        return None
    reduce_var = reduce_axes[0]

    def _index_expr_to_ast(idx_expr: tir.PrimExpr) -> T.ASTNode:
        if isinstance(idx_expr, tir.Var):
            return T.Tile(idx_expr.name)
        if isinstance(idx_expr, tir.IntImm) and int(idx_expr.value) == 0:
            return T.FullTile()
        if isinstance(idx_expr, tir.Sub) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value) * -1)
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value))
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.b, tir.Var) and isinstance(idx_expr.a, tir.IntImm):
            return T.TileOffset(idx_expr.b.name, int(idx_expr.a.value))
        return visit_expr(idx_expr)

    if any(isinstance(idx, tir.Var) and idx.same_as(reduce_var) for idx in body_store.indices):
        return None

    reduce_axis = _find_reduce_axis_in_expr(in_expr, reduce_var)
    if reduce_axis is None:
        return None

    out_index_nodes = [_index_expr_to_ast(idx) for idx in body_store.indices]

    out_tensor = T.Tensor(body_store.buffer.name.replace(".", "_"))
    out_index = T.Index(out_index_nodes)
    out_load_node = T.Load(out_tensor, out_index)

    if is_max:
        reduce_value = T.ReduceMax(visit_expr(in_expr), reduce_axis)
        combine = T.GenericCall("max", [out_load_node, reduce_value])
    else:
        reduce_value = T.ReduceMin(visit_expr(in_expr), reduce_axis)
        combine = T.GenericCall("min", [out_load_node, reduce_value])
    return T.Store(out_tensor, combine, out_index)

def _try_convert_multi_reducemaxmin_block(block: tir.SBlock, visit_expr) -> Optional[T.ASTNode]:
    if block.init is None:
        return None

    def _collect_buffer_stores(stmt) -> List[tir.BufferStore]:
        stores: List[tir.BufferStore] = []
        if stmt is None:
            return stores

        def visit(s):
            if isinstance(s, tir.BufferStore):
                stores.append(s)

        tir.stmt_functor.post_order_visit(stmt, visit)
        return stores

    init_store = block.init
    if isinstance(init_store, tir.SBlockRealize):
        init_store = init_store.block.body
    init_stores = _collect_buffer_stores(init_store)
    if not init_stores:
        return None

    init_buffers = {s.buffer for s in init_stores}
    body_stores = _collect_buffer_stores(block.body)
    target_stores = [s for s in body_stores if s.buffer in init_buffers]
    if not target_stores:
        return None

    let_bindings = _collect_let_bindings(block.body)

    reduce_axes = []
    for iter_var in block.iter_vars:
        iter_type = iter_var.iter_type
        reduce_kinds = {
            getattr(tir.IterVar, "Reduce", None),
            getattr(tir.IterVar, "CommReduce", None),
        }
        if iter_type in reduce_kinds or "reduce" in str(iter_type).lower():
            reduce_axes.append(iter_var.var)

    if len(reduce_axes) != 1:
        return None
    reduce_var = reduce_axes[0]

    def _index_expr_to_ast(idx_expr: tir.PrimExpr) -> T.ASTNode:
        if isinstance(idx_expr, tir.Var):
            return T.Tile(idx_expr.name)
        if isinstance(idx_expr, tir.IntImm) and int(idx_expr.value) == 0:
            return T.FullTile()
        if isinstance(idx_expr, tir.Sub) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value) * -1)
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.a, tir.Var) and isinstance(idx_expr.b, tir.IntImm):
            return T.TileOffset(idx_expr.a.name, int(idx_expr.b.value))
        if isinstance(idx_expr, tir.Add) and isinstance(idx_expr.b, tir.Var) and isinstance(idx_expr.a, tir.IntImm):
            return T.TileOffset(idx_expr.b.name, int(idx_expr.a.value))
        return visit_expr(idx_expr)

    stores: List[T.ASTNode] = []
    for body_store in target_stores:
        value_expr = body_store.value
        if isinstance(value_expr, tir.Var):
            value_expr = let_bindings.get(value_expr.name, value_expr)
        if not isinstance(value_expr, (tir.Max, tir.Min)):
            return None
        is_max = isinstance(value_expr, tir.Max)
        op_left = value_expr.a
        op_right = value_expr.b

        out_load = None
        in_expr = None
        left_load = _strip_cast_load(op_left)
        right_load = _strip_cast_load(op_right)
        if isinstance(left_load, tir.BufferLoad) and left_load.buffer == body_store.buffer and list(left_load.indices) == list(body_store.indices):
            out_load = left_load
            in_expr = op_right
        elif isinstance(right_load, tir.BufferLoad) and right_load.buffer == body_store.buffer and list(right_load.indices) == list(body_store.indices):
            out_load = right_load
            in_expr = op_left

        if out_load is None or in_expr is None:
            return None

        if any(isinstance(idx, tir.Var) and idx.same_as(reduce_var) for idx in body_store.indices):
            return None

        reduce_axis = _find_reduce_axis_in_expr(in_expr, reduce_var)
        if reduce_axis is None:
            return None

        out_index_nodes = [_index_expr_to_ast(idx) for idx in body_store.indices]
        out_tensor = T.Tensor(body_store.buffer.name.replace(".", "_"))
        out_index = T.Index(out_index_nodes)
        out_load_node = T.Load(out_tensor, out_index)
        if is_max:
            reduce_value = T.ReduceMax(visit_expr(in_expr), reduce_axis)
            combine = T.GenericCall("max", [out_load_node, reduce_value])
        else:
            reduce_value = T.ReduceMin(visit_expr(in_expr), reduce_axis)
            combine = T.GenericCall("min", [out_load_node, reduce_value])
        stores.append(T.Store(out_tensor, combine, out_index))

    if not stores:
        return None
    if len(stores) == 1:
        return stores[0]
    return T.Block(stores)

def _infer_permutation(store: tir.BufferStore) -> List[int]:
    load = _find_first_load(store.value)
    if load is None:
        return []

    store_indices = list(store.indices)
    load_indices = list(load.indices)
    if len(store_indices) != len(load_indices):
        return []

    if not all(isinstance(idx, tir.Var) for idx in store_indices):
        return []
    if not all(isinstance(idx, tir.Var) for idx in load_indices):
        return []

    store_vars = [idx.name for idx in store_indices]
    load_vars = [idx.name for idx in load_indices]
    if set(store_vars) != set(load_vars):
        return []

    permutation: List[int] = []
    for var in store_vars:
        permutation.append(load_vars.index(var))
    inv = [0]*len(permutation)
    for out_axis, in_axis in enumerate(permutation):
        inv[in_axis] = out_axis
    return inv

def _build_loop_nest(
    extents: List[int],
    body: T.ASTNode,
    var_prefix: str = "ax",
) -> T.ASTNode:
    node = body
    for idx in reversed(range(len(extents))):
        var = f"{var_prefix}{idx}"
        node = T.Loop(T.Const(0), T.Const(extents[idx]), f"tile_{var}", var, node)
    return node

def _unary_to_loop_ast(
    input_tensor: T.TensorInfo,
    output_tensor: T.TensorInfo,
    op_name: str,
) -> Optional[T.ASTNode]:
    in_dims = _shape_to_ints(input_tensor.shape)
    out_dims = _shape_to_ints(output_tensor.shape)
    if in_dims is None or out_dims is None:
        return None
    if in_dims != out_dims:
        return None

    loop_vars = [f"ax{i}" for i in range(len(out_dims))]
    indices = [T.Tile(v) for v in loop_vars]
    load = T.Load(T.Tensor(input_tensor.name), T.Index(indices))
    value = T.GenericCall(op_name, [load])
    store = T.Store(T.Tensor(output_tensor.name), value, T.Index(indices))
    return _build_loop_nest(out_dims, store)

def _transpose_to_loop_ast(
    input_tensor: T.TensorInfo,
    output_tensor: T.TensorInfo,
    permutation: List[int],
) -> Optional[T.ASTNode]:
    in_dims = _shape_to_ints(input_tensor.shape)
    out_dims = _shape_to_ints(output_tensor.shape)
    if in_dims is None or out_dims is None:
        return None
    if len(permutation) != len(out_dims) or len(in_dims) != len(out_dims):
        return None

    loop_vars = [f"ax{i}" for i in range(len(out_dims))]
    out_indices = [T.Tile(v) for v in loop_vars]

    load_indices = [out_indices[p] for p in permutation]
    load = T.Load(T.Tensor(input_tensor.name), T.Index(load_indices))
    inv_perm = [0] * len(permutation)
    for out_axis, in_axis in enumerate(permutation):
        inv_perm[in_axis] = out_axis
    if len(inv_perm) == 3:
        value = T.Permute3(load, inv_perm[0], inv_perm[1], inv_perm[2])
    else:
        if len(inv_perm) == 2:
            value = T.GenericCall("transpose", [load])
        else:
            perm_args = [T.Const(p) for p in inv_perm]
            value = T.GenericCall("transpose", [load] + perm_args)
    store = T.Store(T.Tensor(output_tensor.name), value, T.Index(out_indices))
    return _build_loop_nest(out_dims, store)

def _broadcast_to_loop_ast(
    input_tensor: T.TensorInfo,
    output_tensor: T.TensorInfo,
) -> Optional[T.ASTNode]:
    in_dims = _shape_to_ints(input_tensor.shape)
    out_dims = _shape_to_ints(output_tensor.shape)
    if in_dims is None or out_dims is None:
        return None
    if len(in_dims) > len(out_dims):
        return None

    offset = len(out_dims) - len(in_dims)
    loop_vars = [f"ax{i}" for i in range(len(out_dims))]
    out_indices = [T.Tile(v) for v in loop_vars]

    input_indices: List[T.ASTNode] = []
    broadcast_axes: List[int] = []
    for in_idx, dim in enumerate(in_dims):
        out_idx = in_idx + offset
        if dim == out_dims[out_idx]:
            input_indices.append(T.Tile(loop_vars[out_idx]))
        elif dim == 1:
            input_indices.append(T.FullTile())
            if out_dims[out_idx] > 1:
                broadcast_axes.append(out_idx)
        else:
            return None
    if offset > 0:
        broadcast_axes.extend(list(range(offset)))

    load = T.Load(T.Tensor(input_tensor.name), T.Index(input_indices))
    value: T.ASTNode = load
    squeeze_axes = [axis for axis in broadcast_axes if axis >= offset]
    for axis in sorted(set(squeeze_axes), reverse=True):
        value = T.Squeeze(value, axis)
    for axis in sorted(set(broadcast_axes)):
        value = T.Broadcast(value, axis)
    store = T.Store(T.Tensor(output_tensor.name), value, T.Index(out_indices))
    return _build_loop_nest(out_dims, store)

def _concat_to_loop_ast(
    a_tensor: T.TensorInfo,
    b_tensor: T.TensorInfo,
    output_tensor: T.TensorInfo,
) -> Optional[T.ASTNode]:
    a_dims = _shape_to_ints(a_tensor.shape)
    b_dims = _shape_to_ints(b_tensor.shape)
    out_dims = _shape_to_ints(output_tensor.shape)
    if a_dims is None or b_dims is None or out_dims is None:
        return None
    if len(a_dims) != len(b_dims) or len(a_dims) != len(out_dims):
        return None

    axis = None
    for i, (ad, bd, od) in enumerate(zip(a_dims, b_dims, out_dims)):
        if ad == od and bd == od:
            continue
        if ad + bd == od:
            if axis is not None:
                return None
            axis = i
        else:
            return None

    if axis is None:
        return None
    a_dim = a_dims[axis]
    b_dim = b_dims[axis]
    if a_dim <= 0 or b_dim <= 0:
        return None

    loop_vars = {dim: f"ax{dim}" for dim in range(len(out_dims))}

    # First loop: write the front (a_tensor) segment.
    out_indices_a: List[T.ASTNode] = []
    a_indices: List[T.ASTNode] = []
    for dim in range(len(out_dims)):
        var = loop_vars[dim]
        if dim == axis:
            out_indices_a.append(T.Tile(var))
            a_indices.append(T.Tile(var))
        else:
            out_indices_a.append(T.Tile(var))
            a_indices.append(T.Tile(var))

    a_load = T.Load(T.Tensor(a_tensor.name), T.Index(a_indices))
    store_a = T.Store(T.Tensor(output_tensor.name), a_load, T.Index(out_indices_a))
    node_a: T.ASTNode = store_a
    for dim in reversed(range(len(out_dims))):
        extent = a_dim if dim == axis else out_dims[dim]
        var = loop_vars[dim]
        node_a = T.Loop(T.Const(0), T.Const(extent), f"tile_{var}", var, node_a)

    # Second loop: write the tail (b_tensor) segment using ConstTile.
    out_indices_b: List[T.ASTNode] = []
    b_indices: List[T.ASTNode] = []
    for dim in range(len(out_dims)):
        var = loop_vars[dim]
        if dim == axis:
            out_indices_b.append(T.ConstTile(a_dim, b_dim))
            b_indices.append(T.FullTile())
        else:
            out_indices_b.append(T.Tile(var))
            b_indices.append(T.Tile(var))

    b_load = T.Load(T.Tensor(b_tensor.name), T.Index(b_indices))
    store_b = T.Store(T.Tensor(output_tensor.name), b_load, T.Index(out_indices_b))
    node_b: T.ASTNode = store_b
    for dim in reversed(range(len(out_dims))):
        if dim == axis:
            continue
        extent = out_dims[dim]
        var = loop_vars[dim]
        node_b = T.Loop(T.Const(0), T.Const(extent), f"tile_{var}", var, node_b)

    return T.Block([node_a, node_b])

def _slice_scatter_to_loop_ast(
    input_tensors: List[T.TensorInfo],
    output_tensor: T.TensorInfo,
) -> Optional[T.ASTNode]:
    if len(input_tensors) < 2:
        return None

    out_dims = _shape_to_ints(output_tensor.shape)
    if out_dims is None:
        return None

    base_tensor = None
    update_tensor = None
    update_dims = None
    axis = None

    for candidate in input_tensors:
        cand_dims = _shape_to_ints(candidate.shape)
        if cand_dims is None:
            continue
        if cand_dims == out_dims and base_tensor is None:
            base_tensor = candidate
            continue
        if len(cand_dims) != len(out_dims):
            continue
        diff_axis = None
        for i, (od, cd) in enumerate(zip(out_dims, cand_dims)):
            if od == cd:
                continue
            if cd < od and diff_axis is None:
                diff_axis = i
                continue
            diff_axis = None
            break
        if diff_axis is not None:
            update_tensor = candidate
            update_dims = cand_dims
            axis = diff_axis

    if base_tensor is None or update_tensor is None or update_dims is None or axis is None:
        return None

    start = out_dims[axis] - update_dims[axis]
    if start < 0:
        return None

    loop_vars = {dim: f"ax{dim}" for dim in range(len(out_dims))}

    # First loop: copy base tensor into output.
    out_indices_base: List[T.ASTNode] = []
    base_indices: List[T.ASTNode] = []
    for dim in range(len(out_dims)):
        var = loop_vars[dim]
        out_indices_base.append(T.Tile(var))
        base_indices.append(T.Tile(var))

    base_load = T.Load(T.Tensor(base_tensor.name), T.Index(base_indices))
    store_base = T.Store(T.Tensor(output_tensor.name), base_load, T.Index(out_indices_base))
    node_base: T.ASTNode = store_base
    for dim in reversed(range(len(out_dims))):
        extent = out_dims[dim]
        var = loop_vars[dim]
        node_base = T.Loop(T.Const(0), T.Const(extent), f"tile_{var}", var, node_base)

    # Second loop: overwrite the tail slice with update tensor.
    out_indices_update: List[T.ASTNode] = []
    update_indices: List[T.ASTNode] = []
    for dim in range(len(out_dims)):
        var = loop_vars[dim]
        if dim == axis:
            out_indices_update.append(T.ConstTile(start, update_dims[axis]))
            update_indices.append(T.FullTile())
        else:
            out_indices_update.append(T.Tile(var))
            update_indices.append(T.Tile(var))

    update_load = T.Load(T.Tensor(update_tensor.name), T.Index(update_indices))
    store_update = T.Store(T.Tensor(output_tensor.name), update_load, T.Index(out_indices_update))
    node_update: T.ASTNode = store_update
    for dim in reversed(range(len(out_dims))):
        if dim == axis:
            continue
        extent = out_dims[dim]
        var = loop_vars[dim]
        node_update = T.Loop(T.Const(0), T.Const(extent), f"tile_{var}", var, node_update)

    return T.Block([node_base, node_update])

def _mean_to_loop_ast(
    prim_func: tir.PrimFunc,
    input_tensor: T.TensorInfo,
    output_tensor: T.TensorInfo,
) -> Optional[T.ASTNode]:
    in_dims = _shape_to_ints(input_tensor.shape)
    out_dims = _shape_to_ints(output_tensor.shape)
    if in_dims is None or out_dims is None:
        return None
    if len(in_dims) != len(out_dims):
        return None

    axis = None
    for i, (idim, odim) in enumerate(zip(in_dims, out_dims)):
        if idim == odim:
            continue
        if odim == 1 and idim > 1 and axis is None:
            axis = i
            continue
        axis = None
        break

    if axis is None:
        return None

    reduce_size = in_dims[axis]
    if reduce_size <= 0:
        return None

    alloc_tensors = _collect_allocated_tensors(prim_func)
    acc_tensor = None
    for tensor in alloc_tensors:
        if tensor.name == output_tensor.name:
            continue
        if _shape_to_ints(tensor.shape) == out_dims:
            acc_tensor = tensor
            break

    if acc_tensor is None:
        return None

    loop_var = "j"
    input_indices: List[T.ASTNode] = []
    for dim in range(len(in_dims)):
        if dim == axis:
            input_indices.append(T.Tile(loop_var))
        else:
            input_indices.append(T.FullTile())

    out_indices = [T.FullTile() for _ in out_dims]
    rsum_value = T.ReduceSum(T.Load(T.Tensor(input_tensor.name), T.Index(input_indices)), axis)
    rsum_value = T.Unsqueeze(rsum_value, axis)
    acc_tensor_node = T.Tensor(acc_tensor.name)
    out_index = T.Index(out_indices)
    acc_load = T.Load(acc_tensor_node, out_index)
    acc_store = T.Store(
        acc_tensor_node,
        _accumulate_with_identity(acc_load, rsum_value),
        out_index,
    )
    loop_node = T.Loop(T.Const(0), T.Const(reduce_size), f"tile_{loop_var}", loop_var, acc_store)
    out_store = T.Store(
        T.Tensor(output_tensor.name),
        T.Div(T.Load(acc_tensor_node, out_index), T.Const(float(reduce_size))),
        out_index,
    )
    return T.Block([loop_node, out_store])

def _strided_slice_to_loop_ast(
    input_tensor: T.TensorInfo,
    output_tensor: T.TensorInfo,
) -> Optional[T.ASTNode]:
    in_dims = _shape_to_ints(input_tensor.shape)
    out_dims = _shape_to_ints(output_tensor.shape)
    if in_dims is None or out_dims is None:
        return None
    if len(in_dims) != len(out_dims):
        return None

    axis = None
    for i, (idim, odim) in enumerate(zip(in_dims, out_dims)):
        if idim == odim:
            continue
        if odim < idim and axis is None:
            axis = i
            continue
        axis = None
        break

    if axis is None:
        return None

    start = in_dims[axis] - out_dims[axis]
    if start < 0:
        return None

    loop_vars = {dim: f"ax{dim}" for dim in range(len(out_dims))}
    out_indices: List[T.ASTNode] = []
    in_indices: List[T.ASTNode] = []
    for dim in range(len(out_dims)):
        var = loop_vars[dim]
        out_indices.append(T.Tile(var))
        if dim == axis:
            in_indices.append(T.ConstTile(start, out_dims[axis]))
        else:
            in_indices.append(T.Tile(var))

    load = T.Load(T.Tensor(input_tensor.name), T.Index(in_indices))
    store = T.Store(T.Tensor(output_tensor.name), load, T.Index(out_indices))
    node = store
    for dim in reversed(range(len(out_dims))):
        extent = out_dims[dim]
        var = loop_vars[dim]
        node = T.Loop(T.Const(0), T.Const(extent), f"tile_{var}", var, node)
    return node

def _expand_dims_to_loop_ast(
    input_tensor: T.TensorInfo,
    output_tensor: T.TensorInfo,
) -> Optional[T.ASTNode]:
    in_dims = _shape_to_ints(input_tensor.shape)
    out_dims = _shape_to_ints(output_tensor.shape)
    if in_dims is None or out_dims is None:
        return None
    if len(in_dims) >= len(out_dims):
        return None

    loop_vars = [f"ax{i}" for i in range(len(out_dims))]
    out_indices = [T.Tile(v) for v in loop_vars]

    input_indices: List[T.ASTNode] = []
    insert_axes: List[int] = []
    in_idx = 0
    for out_idx, out_dim in enumerate(out_dims):
        if in_idx < len(in_dims) and in_dims[in_idx] == out_dim:
            input_indices.append(T.Tile(loop_vars[out_idx]))
            in_idx += 1
        elif out_dim == 1:
            insert_axes.append(out_idx)
            continue
        else:
            return None

    if in_idx != len(in_dims):
        return None

    load = T.Load(T.Tensor(input_tensor.name), T.Index(input_indices))
    value: T.ASTNode = load
    for axis in sorted(insert_axes):
        value = T.Unsqueeze(value, axis)
    store = T.Store(T.Tensor(output_tensor.name), value, T.Index(out_indices))
    return _build_loop_nest(out_dims, store)

def _shape_to_ints(shape: List[object]) -> Optional[List[int]]:
    dims: List[int] = []
    for dim in shape:
        dim_val = _primexpr_to_int(dim)
        if dim_val is None:
            return None
        dims.append(dim_val)
    return dims

def _infer_single_split_shape(
    input_shape: List[object],
    output_shape: List[object],
) -> Optional[tuple[int, int, int, int]]:
    in_dims = _shape_to_ints(input_shape)
    out_dims = _shape_to_ints(output_shape)
    if in_dims is None or out_dims is None:
        return None
    if len(out_dims) != len(in_dims) + 1:
        return None

    split: Optional[tuple[int, int, int, int]] = None
    i = 0
    j = 0
    while i < len(in_dims) and j < len(out_dims):
        if in_dims[i] == out_dims[j]:
            i += 1
            j += 1
            continue
        if split is not None or j + 1 >= len(out_dims):
            return None
        if in_dims[i] != out_dims[j] * out_dims[j + 1]:
            return None
        split = (i, j, out_dims[j], out_dims[j + 1])
        i += 1
        j += 2

    if i == len(in_dims) and j == len(out_dims) and split is not None:
        return split
    return None

def _infer_single_merge_shape(
    input_shape: List[object],
    output_shape: List[object],
) -> Optional[tuple[int, int, int, int]]:
    in_dims = _shape_to_ints(input_shape)
    out_dims = _shape_to_ints(output_shape)
    if in_dims is None or out_dims is None:
        return None
    if len(in_dims) != len(out_dims) + 1:
        return None

    merge: Optional[tuple[int, int, int, int]] = None
    i = 0
    j = 0
    while i < len(in_dims) and j < len(out_dims):
        if in_dims[i] == out_dims[j]:
            i += 1
            j += 1
            continue
        if merge is not None or i + 1 >= len(in_dims):
            return None
        if in_dims[i] * in_dims[i + 1] != out_dims[j]:
            return None
        merge = (i, j, in_dims[i], in_dims[i + 1])
        i += 2
        j += 1

    if i == len(in_dims) and j == len(out_dims) and merge is not None:
        return merge
    return None

def _reshape_to_loop_ast(
    input_tensor: T.TensorInfo,
    output_tensor: T.TensorInfo,
) -> Optional[T.ASTNode]:
    in_dims = _shape_to_ints(input_tensor.shape)
    out_dims = _shape_to_ints(output_tensor.shape)
    split = _infer_single_split_shape(input_tensor.shape, output_tensor.shape)
    if split is not None:
        in_idx, out_idx, _, inner = split
        loop_var = "i"
        tile_name = str(inner)

        in_indices: List[T.ASTNode] = []
        for idx in range(len(input_tensor.shape)):
            if idx == in_idx:
                in_indices.append(T.Tile(loop_var))
            else:
                in_indices.append(T.FullTile())

        out_indices: List[T.ASTNode] = []
        for idx in range(len(output_tensor.shape)):
            if idx == out_idx:
                out_indices.append(T.Elem(loop_var))
            elif idx == out_idx + 1:
                out_indices.append(T.FullTile())
            else:
                out_indices.append(T.FullTile())

        load = T.Load(T.Tensor(input_tensor.name), T.Index(in_indices))
        value = T.Unsqueeze(load, out_idx)
        store = T.Store(T.Tensor(output_tensor.name), value, T.Index(out_indices))
        return T.Loop(T.Const(0), T.Const(input_tensor.shape[in_idx]), tile_name, loop_var, store)

    merge = _infer_single_merge_shape(input_tensor.shape, output_tensor.shape)
    if merge is None:
        if in_dims is not None and out_dims is not None:
            diff = len(in_dims) - len(out_dims)
            if diff > 1:
                squeeze_axes = [i for i, dim in enumerate(in_dims) if dim == 1]
                needed = diff - 1
                if len(squeeze_axes) >= needed:
                    squeeze_axes = squeeze_axes[:needed]
                    reduced_in_dims = [
                        dim for i, dim in enumerate(in_dims) if i not in squeeze_axes
                    ]
                    merge = _infer_single_merge_shape(reduced_in_dims, out_dims)
                    if merge is not None:
                        in_idx, out_idx, _, inner = merge
                        loop_var = "i"
                        tile_name = str(inner)

                        reduced_to_orig = [
                            i for i in range(len(in_dims)) if i not in squeeze_axes
                        ]

                        in_indices: List[T.ASTNode] = []
                        for orig_idx in range(len(in_dims)):
                            if orig_idx in squeeze_axes:
                                in_indices.append(T.FullTile())
                                continue
                            reduced_idx = reduced_to_orig.index(orig_idx)
                            if reduced_idx == in_idx:
                                in_indices.append(T.Elem(loop_var))
                            elif reduced_idx == in_idx + 1:
                                in_indices.append(T.FullTile())
                            else:
                                in_indices.append(T.FullTile())

                        out_indices: List[T.ASTNode] = []
                        for idx in range(len(out_dims)):
                            if idx == out_idx:
                                out_indices.append(T.Tile(loop_var))
                            else:
                                out_indices.append(T.FullTile())

                        value: T.ASTNode = T.Load(
                            T.Tensor(input_tensor.name), T.Index(in_indices)
                        )
                        for axis in sorted(squeeze_axes, reverse=True):
                            value = T.Squeeze(value, axis)
                        value = T.Squeeze(value, in_idx)
                        store = T.Store(
                            T.Tensor(output_tensor.name), value, T.Index(out_indices)
                        )
                        return T.Loop(
                            T.Const(0),
                            T.Const(output_tensor.shape[out_idx]),
                            tile_name,
                            loop_var,
                            store,
                        )

        expanded = _expand_dims_to_loop_ast(input_tensor, output_tensor)
        if expanded is not None:
            return expanded
        return _reshape_with_ones_to_loop_ast(input_tensor, output_tensor)
    in_idx, out_idx, _, inner = merge
    loop_var = "i"
    tile_name = str(inner)

    in_indices: List[T.ASTNode] = []
    for idx in range(len(input_tensor.shape)):
        if idx == in_idx:
            in_indices.append(T.Elem(loop_var))
        elif idx == in_idx + 1:
            in_indices.append(T.FullTile())
        else:
            in_indices.append(T.FullTile())

    out_indices: List[T.ASTNode] = []
    for idx in range(len(output_tensor.shape)):
        if idx == out_idx:
            out_indices.append(T.Tile(loop_var))
        else:
            out_indices.append(T.FullTile())

    load = T.Load(T.Tensor(input_tensor.name), T.Index(in_indices))
    value = T.Squeeze(load, in_idx)
    store = T.Store(T.Tensor(output_tensor.name), value, T.Index(out_indices))
    return T.Loop(T.Const(0), T.Const(output_tensor.shape[out_idx]), tile_name, loop_var, store)

def _reshape_with_ones_to_loop_ast(
    input_tensor: T.TensorInfo,
    output_tensor: T.TensorInfo,
) -> Optional[T.ASTNode]:
    in_dims = _shape_to_ints(input_tensor.shape)
    out_dims = _shape_to_ints(output_tensor.shape)
    if in_dims is None or out_dims is None:
        return None
    if len(out_dims) < len(in_dims):
        return None

    loop_vars = [f"ax{i}" for i in range(len(out_dims))]
    out_indices: List[T.ASTNode] = []
    for idx, dim in enumerate(out_dims):
        if dim == 1:
            out_indices.append(T.FullTile())
        else:
            out_indices.append(T.Tile(loop_vars[idx]))

    # Map input dims onto output dims, allowing extra 1s in output.
    mapping: List[tuple[int, ...]] = []
    out_idx = 0
    for in_dim in in_dims:
        while out_idx < len(out_dims) and out_dims[out_idx] == 1 and in_dim != 1:
            out_idx += 1
        if out_idx >= len(out_dims):
            return None
        if in_dim == 1:
            if out_dims[out_idx] != 1:
                return None
            mapping.append((out_idx,))
            out_idx += 1
            continue
        if out_dims[out_idx] == in_dim:
            mapping.append((out_idx,))
            out_idx += 1
            continue
        if out_idx + 1 < len(out_dims) and out_dims[out_idx] * out_dims[out_idx + 1] == in_dim:
            mapping.append((out_idx, out_idx + 1))
            out_idx += 2
            continue
        return None

    if any(dim != 1 for dim in out_dims[out_idx:]):
        return None

    split_mappings = [(i, mapped) for i, mapped in enumerate(mapping) if len(mapped) == 2]
    if split_mappings:
        # Use split-style representation to avoid index arithmetic.
        if len(split_mappings) > 1:
            return None
        split_in_idx, (outer, inner) = split_mappings[0]
        mapped_out = {outer, inner}
        for idx, dim in enumerate(out_dims):
            if dim != 1 and idx not in mapped_out:
                return None

        loop_var = "i"
        in_indices: List[T.ASTNode] = []
        for idx in range(len(in_dims)):
            if idx == split_in_idx:
                in_indices.append(T.Tile(loop_var))
            else:
                in_indices.append(T.FullTile())

        out_indices: List[T.ASTNode] = []
        for idx, dim in enumerate(out_dims):
            if idx == outer:
                out_indices.append(T.Elem(loop_var))
            else:
                out_indices.append(T.FullTile())

        value: T.ASTNode = T.Load(T.Tensor(input_tensor.name), T.Index(in_indices))
        diff = len(out_dims) - len(in_dims)
        candidates = [i for i, dim in enumerate(out_dims) if dim == 1]
        inserted_axes = candidates[:diff]
        if diff > 0 and outer not in inserted_axes and inserted_axes:
            inserted_axes[-1] = outer
        for axis in sorted(inserted_axes):
            value = T.Unsqueeze(value, axis)

        store = T.Store(T.Tensor(output_tensor.name), value, T.Index(out_indices))
        return T.Loop(T.Const(0), T.Const(in_dims[split_in_idx]), str(out_dims[inner]), loop_var, store)

    in_indices: List[T.ASTNode] = []
    for in_dim, mapped in zip(in_dims, mapping):
        if len(mapped) == 1:
            out_pos = mapped[0]
            if out_dims[out_pos] == 1:
                in_indices.append(T.FullTile())
            else:
                in_indices.append(T.Tile(loop_vars[out_pos]))
        elif len(mapped) == 2:
            outer, inner = mapped
            inner_dim = out_dims[inner]
            idx_expr = T.Add(
                T.Mul(T.Tile(loop_vars[outer]), T.Const(inner_dim)),
                T.Tile(loop_vars[inner]),
            )
            in_indices.append(idx_expr)
        else:
            return None

    load = T.Load(T.Tensor(input_tensor.name), T.Index(in_indices))
    store = T.Store(T.Tensor(output_tensor.name), load, T.Index(out_indices))
    return _build_loop_nest(out_dims, store)

def _convert_to_ast(
    prim_func: tir.PrimFunc,
    func_name: Optional[str] = None,
    debug: bool = False,
    pattern_tracker: Optional[dict] = None,
) -> T.ASTNode:
    special_primfunc = _try_convert_special_primfunc(
        prim_func,
        func_name=func_name,
        debug=debug,
        pattern_tracker=pattern_tracker,
    )
    if special_primfunc is not None:
        return special_primfunc
    tensor_shape_map: dict[str, List[object]] = {}
    for buf in prim_func.buffer_map.values():
        safe_name = buf.name.replace(".", "_")
        tensor_shape_map[safe_name] = list(buf.shape)

    # -----------------------------------------------------------
    # Expression Visitor: Returns ASTNode instances (Expr)
    # -----------------------------------------------------------
    def visit_expr(expr):
        # 1. Arithmetic
        if isinstance(expr, tir.Add):
            return T.Add(visit_expr(expr.a), visit_expr(expr.b))
        elif isinstance(expr, tir.Sub):
            return T.Sub(visit_expr(expr.a), visit_expr(expr.b))
        elif isinstance(expr, tir.Mul):
            return T.Mul(visit_expr(expr.a), visit_expr(expr.b))
        elif isinstance(expr, tir.Div):
            return T.Div(visit_expr(expr.a), visit_expr(expr.b))
        
        # Operations not in strict user list -> Map to GenericBinary or Specifics
        elif isinstance(expr, tir.FloorDiv):
            return T.GenericBinary("//", visit_expr(expr.a), visit_expr(expr.b))
        elif isinstance(expr, tir.Mod):
            return T.GenericBinary("%", visit_expr(expr.a), visit_expr(expr.b))
        elif isinstance(expr, tir.FloorMod):
            return T.GenericBinary("%", visit_expr(expr.a), visit_expr(expr.b))
        
        # 2. Logic & Comparison
        elif isinstance(expr, (tir.LT, tir.LE, tir.GT, tir.GE, tir.EQ, tir.NE)):
            op_map = {tir.LT: "<", tir.LE: "<=", tir.GT: ">", tir.GE: ">=", tir.EQ: "==", tir.NE: "!="}
            return T.GenericBinary(op_map[type(expr)], visit_expr(expr.a), visit_expr(expr.b))
        elif isinstance(expr, tir.And):
            return T.GenericBinary("and", visit_expr(expr.a), visit_expr(expr.b))
        elif isinstance(expr, tir.Or):
            return T.GenericBinary("or", visit_expr(expr.a), visit_expr(expr.b))
        
        # 3. Call / Intrinsics
        elif isinstance(expr, tir.Call):
            op_name = expr.op.name if hasattr(expr.op, 'name') else str(expr.op)
            op_name = op_name.replace("tir.", "")
            op_name = op_name.replace("tirx.", "")
            op_name = op_name.replace("s_tir.", "")
            args = [visit_expr(arg) for arg in expr.args]
            
            # Mapping known intrinsics to User AST
            if op_name == "exp":
                return T.Exp(args[0])
            elif op_name == "abs":
                return T.GenericCall("abs", [args[0]])
            elif op_name == "sqrt":
                return T.Sqrt(args[0])
            elif op_name == "sigmoid":
                return T.Sigmoid(args[0])
            elif op_name == "pow":
                if len(expr.args) != 2:
                    raise RuntimeError("pow expects exactly two arguments")
                exponent = expr.args[1]
                exp_val = None
                if isinstance(exponent, (tir.IntImm, int)):
                    exp_val = int(exponent)
                elif isinstance(exponent, tir.FloatImm):
                    if float(exponent.value).is_integer():
                        exp_val = int(exponent.value)
                if exp_val is not None and exp_val >= 0:
                    if exp_val == 0:
                        return T.Const(1)
                    base = visit_expr(expr.args[0])
                    result = base
                    for _ in range(exp_val - 1):
                        result = T.Mul(result, base)
                    return result
                raise NotImplementedError("pow only supports non-negative integer exponents")
            elif op_name == "rsqrt":
                # User list doesn't have Rsqrt, mapping to 1 / Sqrt(x) or Generic
                return T.Div(T.Const(1.0), T.Sqrt(args[0]))
            else:
                return T.GenericCall(op_name, args)

        elif isinstance(expr, tir.Select):
            return T.GenericCall("select", [visit_expr(expr.condition), visit_expr(expr.true_value), visit_expr(expr.false_value)])
        elif isinstance(expr, tir.Max):
            return T.GenericCall("max", [visit_expr(expr.a), visit_expr(expr.b)])
        elif isinstance(expr, tir.Min):
            return T.GenericCall("min", [visit_expr(expr.a), visit_expr(expr.b)])
            
        elif isinstance(expr, tir.Cast):
            return T.Cast(str(expr.dtype), visit_expr(expr.value))

        # 4. Memory Access
        elif isinstance(expr, tir.BufferLoad):
            # Indices handling: Try to wrap in Tile if simple var, else raw Expr
            idx_nodes = []
            for idx in expr.indices:
                if isinstance(idx, tir.Var):
                    idx_nodes.append(T.Tile(idx.name))
                elif isinstance(idx, tir.IntImm) and int(idx.value) == 0:
                    idx_nodes.append(T.FullTile())
                elif isinstance(idx, tir.Sub) and isinstance(idx.a, tir.Var) and isinstance(idx.b, tir.IntImm):
                    idx_nodes.append(T.TileOffset(idx.a.name, int(idx.b.value) * -1))
                elif isinstance(idx, tir.Add) and isinstance(idx.a, tir.Var) and isinstance(idx.b, tir.IntImm):
                    idx_nodes.append(T.TileOffset(idx.a.name, int(idx.b.value)))
                elif isinstance(idx, tir.Add) and isinstance(idx.b, tir.Var) and isinstance(idx.a, tir.IntImm):
                    idx_nodes.append(T.TileOffset(idx.b.name, int(idx.a.value)))
                else:
                    idx_nodes.append(visit_expr(idx))
            
            return T.Load(visit_expr(expr.buffer), T.Index(idx_nodes))

        # 5. Atomic
        elif isinstance(expr, tir.Var) or isinstance(expr, tir.Buffer):
            safe_name = expr.name.replace(".", "_")
            return T.Tensor(safe_name)
        elif isinstance(expr, (tir.IntImm, tir.FloatImm)):
            return T.Const(expr.value)
        else:
            return T.Const(f"UNKNOWN({type(expr)})")

    # -----------------------------------------------------------
    # Statement Visitor: Returns ASTNode instances (Stmt)
    # -----------------------------------------------------------
    def visit_stmt(stmt):
        def _index_key(idx: T.ASTNode) -> tuple:
            if isinstance(idx, T.Tile):
                return ("tile", idx.name)
            if isinstance(idx, T.Elem):
                return ("elem", idx.name)
            if isinstance(idx, T.FullTile):
                return ("full",)
            if isinstance(idx, T.ConstTile):
                return ("const_tile", idx.start_index, idx.interval)
            if isinstance(idx, T.Const):
                return ("const", idx.value)
            return ("other", str(idx))

        def _insert_broadcasts(
            node: T.ASTNode,
            out_indices: List[T.ASTNode],
            out_shape: Optional[List[object]],
        ) -> T.ASTNode:
            def _axis_is_one(shape: Optional[List[object]], axis: int) -> bool:
                if shape is None or axis >= len(shape):
                    return False
                dim_val = _primexpr_to_int(shape[axis])
                return dim_val == 1

            def _axis_gt_one(shape: Optional[List[object]], axis: int) -> bool:
                if shape is None or axis >= len(shape):
                    return False
                dim_val = _primexpr_to_int(shape[axis])
                return dim_val is not None and dim_val > 1

            if isinstance(node, T.Load):
                load_indices = node.index.indices
                if len(load_indices) < len(out_indices):
                    mapping: List[int] = []
                    out_keys = [_index_key(idx) for idx in out_indices]
                    out_pos = 0
                    for li in load_indices:
                        li_key = _index_key(li)
                        while out_pos < len(out_keys) and out_keys[out_pos] != li_key:
                            out_pos += 1
                        if out_pos == len(out_keys):
                            mapping = list(range(len(load_indices)))
                            break
                        mapping.append(out_pos)
                        out_pos += 1
                    missing = [i for i in range(len(out_indices)) if i not in mapping]
                    value: T.ASTNode = node
                    for axis in missing:
                        value = T.Broadcast(value, axis)
                    return value
                if len(load_indices) == len(out_indices):
                    const_axes = []
                    fulltile_axes = []
                    input_shape = tensor_shape_map.get(node.tensor.name)
                    for axis, (load_idx, out_idx) in enumerate(zip(load_indices, out_indices)):
                        if (
                            isinstance(load_idx, T.Const)
                            and load_idx.value == 0
                            and not isinstance(out_idx, T.Const)
                            and not _axis_is_one(out_shape, axis)
                        ):
                            const_axes.append(axis)
                        elif (
                            isinstance(load_idx, T.FullTile)
                            and _axis_gt_one(out_shape, axis)
                            and _axis_is_one(input_shape, axis)
                            and not isinstance(out_idx, T.Const)
                        ):
                            fulltile_axes.append(axis)
                    if const_axes or fulltile_axes:
                        new_indices: List[T.ASTNode] = []
                        for axis, idx in enumerate(load_indices):
                            if axis in const_axes:
                                new_indices.append(T.FullTile())
                            else:
                                new_indices.append(idx)
                        value: T.ASTNode = T.Load(node.tensor, T.Index(new_indices))
                        for axis in sorted(const_axes + fulltile_axes, reverse=True):
                            value = T.Squeeze(value, axis)
                        for axis in sorted(const_axes + fulltile_axes):
                            value = T.Broadcast(value, axis)
                        return value
                return node
            if isinstance(node, (T.Add, T.Sub, T.Mul, T.Div, T.Matmul, T.GenericBinary)):
                left = _insert_broadcasts(node.left, out_indices, out_shape)
                right = _insert_broadcasts(node.right, out_indices, out_shape)
                if isinstance(node, T.Add):
                    return T.Add(left, right)
                if isinstance(node, T.Sub):
                    return T.Sub(left, right)
                if isinstance(node, T.Mul):
                    return T.Mul(left, right)
                if isinstance(node, T.Div):
                    return T.Div(left, right)
                if isinstance(node, T.Matmul):
                    return T.Matmul(left, right)
                return T.GenericBinary(node.op, left, right)
            if isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
                return node.__class__(_insert_broadcasts(node.val, out_indices, out_shape))
            if isinstance(node, T.Cast):
                return T.Cast(node.dtype, _insert_broadcasts(node.val, out_indices, out_shape))
            if isinstance(node, T.Unsqueeze):
                return T.Unsqueeze(_insert_broadcasts(node.val, out_indices, out_shape), node.axis)
            if isinstance(node, T.Squeeze):
                return T.Squeeze(_insert_broadcasts(node.val, out_indices, out_shape), node.axis)
            if isinstance(node, T.Broadcast):
                return T.Broadcast(_insert_broadcasts(node.val, out_indices, out_shape), node.axis)
            if isinstance(node, T.GenericCall):
                return T.GenericCall(
                    node.func_name,
                    [_insert_broadcasts(arg, out_indices, out_shape) for arg in node.args],
                )
            if isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin, T.Take)):
                return node
            return node

        if isinstance(stmt, tir.For):
            special_loop = _try_convert_special_block_loop(
                stmt,
                prim_func=prim_func,
                func_name=func_name,
                debug=debug,
                pattern_tracker=pattern_tracker,
            )
            if special_loop is not None:
                return special_loop
            # Loop(start, end, tile_name, loop_var, body)
            loop_var = stmt.loop_var.name
            start = visit_expr(stmt.min)
            # extent가 길이이므로 end = min + extent (단, 여기서는 구조상 extent를 그대로 넘기거나 합쳐야 함)
            # 보통 Loop IR에서 end는 exclusive limit을 의미하므로 extent가 맞을 수 있고, start+extent일 수 있음.
            # 여기서는 편의상 extent 값을 그대로 사용하거나 AST 정의에 따라 맞춤.
            # 문맥상 end를 요구하므로 (start + extent) 로직을 적용하거나 단순화.
            if isinstance(stmt.min, tir.IntImm) and stmt.min.value == 0:
                 end = visit_expr(stmt.extent)
            else:
                 end = T.Add(visit_expr(stmt.min), visit_expr(stmt.extent))
            
            tile_name = f"tile_{loop_var}"
            
            body_node = visit_stmt(stmt.body)
            return T.Loop(start, end, tile_name, loop_var, body_node)

        elif isinstance(stmt, tir.SBlockRealize):
            return visit_stmt(stmt.block)

        elif isinstance(stmt, tir.SBlock):
            arange_node = _try_convert_arange_block(stmt)
            if arange_node is not None:
                if pattern_tracker is not None:
                    pattern_tracker["block_patterns"].append("arange")
                return arange_node
            matmul_node = _try_convert_matmul_block(stmt, visit_expr)
            if matmul_node is not None:
                if pattern_tracker is not None:
                    pattern_tracker["block_patterns"].append("matmul")
                return matmul_node
            reduc_mm_node = _try_convert_reducemaxmin_block(stmt, visit_expr)
            if reduc_mm_node is not None:
                if pattern_tracker is not None:
                    pattern_tracker["block_patterns"].append("reducemaxmin")
                return reduc_mm_node
            reduce_multi_mm_node = _try_convert_multi_reducemaxmin_block(stmt, visit_expr)
            if reduce_multi_mm_node is not None:
                if pattern_tracker is not None:
                    pattern_tracker["block_patterns"].append("multi_reducemaxmin")
                return reduce_multi_mm_node
            reduce_multi_node = _try_convert_multi_reducesum_block(stmt, visit_expr)
            if reduce_multi_node is not None:
                if pattern_tracker is not None:
                    pattern_tracker["block_patterns"].append("multi_reducesum")
                return reduce_multi_node
            reduce_node = _try_convert_reducesum_block(stmt, visit_expr)
            if reduce_node is not None:
                if pattern_tracker is not None:
                    pattern_tracker["block_patterns"].append("reducesum")
                return reduce_node
            # Block body processing
            # Init is often skipped in tiling IR unless explicit.
            # If init exists, we might need a Block container.
            
            stmts = []
            if stmt.init:
                init_res = visit_stmt(stmt.init)
                if isinstance(init_res, T.Block):
                    stmts.extend(init_res.stmts)
                else:
                    stmts.append(init_res)
            
            body_res = visit_stmt(stmt.body)
            if isinstance(body_res, T.Block):
                stmts.extend(body_res.stmts)
            else:
                stmts.append(body_res)
            
            if len(stmts) == 1:
                return stmts[0]
            return T.Block(stmts)

        elif isinstance(stmt, tir.BufferStore):
            # Store(tensor, value, index)
            idx_nodes = []
            for idx in stmt.indices:
                if isinstance(idx, tir.Var):
                    idx_nodes.append(T.Tile(idx.name))
                elif isinstance(idx, tir.Sub) and isinstance(idx.a, tir.Var) and isinstance(idx.b, tir.IntImm):
                    idx_nodes.append(T.TileOffset(idx.a.name, int(idx.b.value) * -1))
                elif isinstance(idx, tir.Add) and isinstance(idx.a, tir.Var) and isinstance(idx.b, tir.IntImm):
                    idx_nodes.append(T.TileOffset(idx.a.name, int(idx.b.value)))
                elif isinstance(idx, tir.Add) and isinstance(idx.b, tir.Var) and isinstance(idx.a, tir.IntImm):
                    idx_nodes.append(T.TileOffset(idx.b.name, int(idx.a.value)))
                else:
                    idx_nodes.append(visit_expr(idx))
            value_node = visit_expr(stmt.value)
            out_shape = getattr(stmt.buffer, "shape", None)
            value_node = _insert_broadcasts(value_node, idx_nodes, out_shape)
            return T.Store(visit_expr(stmt.buffer), value_node, T.Index(idx_nodes))
            
        elif isinstance(stmt, tir.SeqStmt):
            stmt_list = [visit_stmt(s) for s in stmt.seq]
            # Flatten blocks if necessary
            flat_list = []
            for s in stmt_list:
                if isinstance(s, T.Block):
                    flat_list.extend(s.stmts)
                else:
                    flat_list.append(s)
            return T.Block(flat_list)
        
        elif isinstance(stmt, tir.IfThenElse):
            cond = visit_expr(stmt.condition)
            then_b = visit_stmt(stmt.then_case)
            else_b = visit_stmt(stmt.else_case) if stmt.else_case else None
            return T.If(cond, then_b, else_b)

        elif isinstance(stmt, tir.Evaluate):
            return visit_expr(stmt.value)
            
        elif getattr(tir, "LetStmt", None) is not None and isinstance(stmt, tir.LetStmt):
            var_name = stmt.var.name
            val_node = visit_expr(stmt.value)
            body_node = visit_stmt(stmt.body)
            return T.Let(visit_expr(stmt.var), val_node, body_node)
        
        else:
            raise RuntimeError(f"unsupported tir stmt: {type(stmt)}")

    return visit_stmt(prim_func.body)

class ExpressionFlattener:
    # ... (ExpressionFlattener 클래스는 변경 없이 그대로 두세요) ...
    # (위의 답변에 있는 코드를 그대로 사용하시면 됩니다)
    def __init__(self, ops_list):
        self.ops = ops_list

    def visit(self, expr):
        if isinstance(expr, tir.Add):
            lhs_idx = self.visit(expr.a)
            rhs_idx = self.visit(expr.b)
            self.ops.append(BinaryOpNode("Add", lhs_idx, rhs_idx))
            return len(self.ops) - 1
        elif isinstance(expr, tir.Sub):
            lhs_idx = self.visit(expr.a)
            rhs_idx = self.visit(expr.b)
            self.ops.append(BinaryOpNode("Sub", lhs_idx, rhs_idx))
            return len(self.ops) - 1
        elif isinstance(expr, tir.Mul):
            lhs_idx = self.visit(expr.a)
            rhs_idx = self.visit(expr.b)
            self.ops.append(BinaryOpNode("Mul", lhs_idx, rhs_idx))
            return len(self.ops) - 1
        elif isinstance(expr, tir.Div):
            lhs_idx = self.visit(expr.a)
            rhs_idx = self.visit(expr.b)
            self.ops.append(BinaryOpNode("Div", lhs_idx, rhs_idx))
            return len(self.ops) - 1
        elif isinstance(expr, tir.Max):
            lhs_idx = self.visit(expr.a)
            rhs_idx = self.visit(expr.b)
            self.ops.append(BinaryOpNode("Max", lhs_idx, rhs_idx))
            return len(self.ops) - 1
        elif isinstance(expr, tir.Min):
            lhs_idx = self.visit(expr.a)
            rhs_idx = self.visit(expr.b)
            self.ops.append(BinaryOpNode("Min", lhs_idx, rhs_idx))
            return len(self.ops) - 1
        elif isinstance(expr, tir.Call):
            if expr.op.name == "tir.exp":
                arg_idx = self.visit(expr.args[0])
                self.ops.append(UnaryOpNode("Exp", arg_idx))
                return len(self.ops) - 1
            pass
        elif isinstance(expr, tir.BufferLoad):
            indices = [str(idx) for idx in expr.indices]
            self.ops.append(BufferAccessNode(expr.buffer.name, indices))
            return len(self.ops) - 1
        elif isinstance(expr, (tir.IntImm, tir.FloatImm)):
            self.ops.append(ConstNode(expr.value))
            return len(self.ops) - 1
        elif isinstance(expr, tir.Cast):
            return self.visit(expr.value)
        
        self.ops.append(ConstNode(str(expr)))
        return len(self.ops) - 1

    def process_stmt(self, stmt):
        if isinstance(stmt, tir.BufferStore):
            val_idx = self.visit(stmt.value)
            dest_node = BufferAccessNode(stmt.buffer.name, [str(idx) for idx in stmt.indices])
            self.ops.append(dest_node)
            dest_idx = len(self.ops) - 1
            self.ops.append(StoreNode(val_idx, dest_idx))
        elif isinstance(stmt, tir.SeqStmt):
            for s in stmt.seq:
                self.process_stmt(s)
        elif isinstance(stmt, tir.Evaluate):
            self.visit(stmt.value)
