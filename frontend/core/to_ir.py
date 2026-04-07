import os
from typing import List
import ir.AST as T


_IR_DEBUG = os.getenv("TRINITY_DEBUG_IR", "").lower() in {"1", "true", "yes", "on"}

def ast_to_lisp(node, level=0, role_map=None):
    """
    AST 노드를 재귀적으로 순회하여 Lisp 스타일 문자열로 변환합니다.
    """
    indent = "  " * level
    
    # --- Helper for inline formatting (indentation 없이 반환) ---
    def to_inline(n):
        return ast_to_lisp(n, level=-1, role_map=role_map).strip()

    # level이 -1이면 들여쓰기를 하지 않음 (수식 내부 등)
    prefix = indent if level >= 0 else ""
    nl = "\n" if level >= 0 else ""

    # 1. Control Flow
    if isinstance(node, T.Loop):
        # (loop start end tile_name loop_var body)
        start_str = to_inline(node.start)
        end_str = to_inline(node.end)
        body_str = ast_to_lisp(node.body, level + 1, role_map=role_map)
        return f"{prefix}(loop {start_str} {end_str} {node.tile_name} {node.loop_var}{nl}{body_str}{nl}{prefix})"

    elif isinstance(node, T.Seq):
        if level < 0:
            return f"(seq {to_inline(node.left)} {to_inline(node.right)})"
        left_str = ast_to_lisp(node.left, level + 1, role_map=role_map)
        right_str = ast_to_lisp(node.right, level, role_map=role_map)
        return f"{prefix}(seq{nl}{left_str}{nl}{right_str}{nl}{prefix})"

    elif isinstance(node, T.Block):
        # 여러 구문이 있는 경우 줄바꿈으로 연결
        return "\n".join([ast_to_lisp(stmt, level, role_map=role_map) for stmt in node.stmts])

    elif isinstance(node, T.If):
        cond_str = to_inline(node.cond)
        then_str = ast_to_lisp(node.then_branch, level + 1, role_map=role_map)
        res = f"{prefix}(if {cond_str}{nl}{then_str}"
        if node.else_branch:
            else_str = ast_to_lisp(node.else_branch, level + 1, role_map=role_map)
            res += f"{nl}{prefix}  (else{nl}{else_str}{nl}{prefix}  )"
        res += f"{nl}{prefix})"
        return res
    
    elif isinstance(node, T.Let):
        tensor_str = to_inline(node.tensor)
        val_str = to_inline(node.value) # 수식 (예: (+ (load...) (load...)))
        
        # Let은 body가 깊어질 수 있으므로 들여쓰기 적용
        body_str = ast_to_lisp(node.body, level + 1, role_map=role_map)
        
        # Lisp 스타일: (let (var value) body)
        return f"{prefix}(let ({tensor_str} {val_str}){nl}{body_str}{nl}{prefix})"

    # 2. Memory Ops
    elif isinstance(node, T.Store):
        # (store (tensor name) value index)
        tensor_str = to_inline(node.tensor)
        # Value와 Index는 복잡할 수 있으므로 들여쓰기 적용
        val_str = ast_to_lisp(node.value, level + 1, role_map=role_map)
        idx_str = ast_to_lisp(node.index, level + 1, role_map=role_map)
        return f"{prefix}(store {tensor_str}{nl}{val_str}{nl}{idx_str}{nl}{prefix})"

    elif isinstance(node, T.Load):
        # (load (tensor name) index)
        tensor_str = to_inline(node.tensor)
        idx_str = to_inline(node.index)
        return f"{prefix}(load {tensor_str} {idx_str})"

    elif isinstance(node, T.Tensor):
        role = role_map.get(node.name) if role_map else None
        if role in ("input", "output"):
            return f"{prefix}({role} {node.name})"
        if role:
            return f"{prefix}(tensor {node.name} {role})"
        return f"{prefix}(tensor {node.name})"

    elif isinstance(node, T.VarRef):
        return f"{prefix}{node.name}"

    elif isinstance(node, T.Index):
        # (index (tile x) (tile y) ...)
        # 인덱스 내부 요소들은 한 줄에 나열
        indices_str = " ".join([to_inline(idx) for idx in node.indices])
        return f"{prefix}(index {indices_str})"

    elif isinstance(node, T.FullTile):
        # 들여쓰기나 괄호 없이 순수 문자열 반환 (Index 내부 등에서 사용)
        return "fulltile"

    elif isinstance(node, T.Tile):
        clean_name = _get_clean_name(node.name)
        return f"(tile {clean_name})"
    elif isinstance(node, T.TileOffset):
        clean_name = _get_clean_name(node.name)
        return f"(shifted_tile {clean_name} {node.offset})"
    elif isinstance(node, T.ConstTile):
        return f"(const_tile {node.start_index} {node.interval})"
    elif isinstance(node, T.Elem):
        clean_name = _get_clean_name(node.name)
        return f"(elem {clean_name})"

    elif isinstance(node, T.Const):
        return f"{prefix}{node.value}"
    # 3. Arithmetic & Logic (Prefix Notation)
    # Binary Ops mapping
    op_symbol = None
    if isinstance(node, T.Add): op_symbol = "+"
    elif isinstance(node, T.Sub): op_symbol = "-"
    elif isinstance(node, T.Mul): op_symbol = "*"  # Scalar multiply
    elif isinstance(node, T.Div): op_symbol = "/"
    elif isinstance(node, T.Max): op_symbol = "max"
    elif isinstance(node, T.Min): op_symbol = "min"
    elif isinstance(node, T.GenericBinary): op_symbol = node.op
    elif isinstance(node, T.Matmul): op_symbol = "@" # Tensor matmul
    
    if op_symbol:
        return f"{prefix}({op_symbol} {to_inline(node.left)} {to_inline(node.right)})"

    # Unary / Function Calls
    if isinstance(node, T.Exp): return f"{prefix}(exp {to_inline(node.val)})"
    elif isinstance(node, T.Sqr): return f"{prefix}(sqr {to_inline(node.val)})"
    elif isinstance(node, T.Sqrt): return f"{prefix}(sqrt {to_inline(node.val)})"
    elif isinstance(node, T.Sigmoid): return f"{prefix}(sigmoid {to_inline(node.val)})"
    elif isinstance(node, T.Cast): return f"{prefix}(cast {node.dtype} {to_inline(node.val)})"
    elif isinstance(node, T.Arange): return f"{prefix}(arange {node.axis})"
    elif isinstance(node, T.Unsqueeze): return f"{prefix}(unsqueeze {to_inline(node.val)} {node.axis})"
    elif isinstance(node, T.Squeeze): return f"{prefix}(squeeze {to_inline(node.val)} {node.axis})"
    elif isinstance(node, T.ReduceSum): return f"{prefix}(rsum {to_inline(node.val)} {node.axis})"
    elif isinstance(node, T.ReduceMax): return f"{prefix}(rmax {to_inline(node.val)} {node.axis})"
    elif isinstance(node, T.ReduceMin): return f"{prefix}(rmin {to_inline(node.val)} {node.axis})"
    elif isinstance(node, T.Broadcast): return f"{prefix}(bcast {to_inline(node.val)} {node.axis})"
    elif isinstance(node, T.Concat): return f"{prefix}(concat {to_inline(node.a)} {to_inline(node.b)} {node.axis})"
    
    elif isinstance(node, T.Permute3):
        return f"{prefix}(permute3 {to_inline(node.val)} {node.d0} {node.d1} {node.d2})"
    elif isinstance(node, T.GenericCall):
        if node.func_name == "transpose":
            args = node.args
            if len(args) == 5:
                return (
                    f"{prefix}(permute4 {to_inline(args[0])} "
                    f"{to_inline(args[1])} {to_inline(args[2])} "
                    f"{to_inline(args[3])} {to_inline(args[4])})"
                )
            if len(args) == 4:
                return (
                    f"{prefix}(permute3 {to_inline(args[0])} "
                    f"{to_inline(args[1])} {to_inline(args[2])} {to_inline(args[3])})"
                )
            if len(args) > 5:
                perm_rank = len(args) - 1
                args_str = " ".join([to_inline(arg) for arg in args])
                return f"{prefix}(permute{perm_rank} {args_str})"
        args_str = " ".join([to_inline(arg) for arg in node.args])
        return f"{prefix}({node.func_name} {args_str})"
    elif isinstance(node, T.Take):
        return f"{prefix}(take {to_inline(node.data)} {to_inline(node.indices)} {node.axis} {to_inline(node.index)})"

    # Fallback
    return f"{prefix}; UnknownNode({type(node)})"


def primfunc_call_to_lisp(call: T.PrimFuncCall, level: int = 0, role_map=None) -> str:
    return primfunc_call_to_lisp_with_root(call, call.primfunc.root_node, level, role_map=role_map)


def primfunc_call_to_lisp_with_root(
    call: T.PrimFuncCall,
    root_node: T.ASTNode,
    level: int = 0,
    role_map=None,
) -> str:
    prefix = "  " * level if level >= 0 else ""
    inputs = " ".join([t.name for t in call.input_tensors])
    head = f"{prefix}(call {call.primfunc.name} {call.out_var_tensor.name}"
    if inputs:
        head = f"{head} {inputs}"
    body = ast_to_lisp(root_node, level + 1, role_map=role_map)
    return f"{head}\n{body}\n{prefix})"


def _fulltile_index(rank: int) -> T.Index:
    return T.Index([T.FullTile() for _ in range(rank)])


def _extract_single_store(node: T.ASTNode) -> T.Store | None:
    if isinstance(node, T.Store):
        return node
    if isinstance(node, T.Seq):
        return None
    if isinstance(node, T.Loop):
        return _extract_single_store(node.body)
    if isinstance(node, T.Block):
        if len(node.stmts) != 1:
            return None
        return _extract_single_store(node.stmts[0])
    return None


def _is_simple_identity_index(node: T.ASTNode) -> bool:
    if isinstance(node, T.FullTile):
        return True
    if isinstance(node, T.Tile):
        return True
    if isinstance(node, T.Index):
        return all(_is_simple_identity_index(idx) for idx in node.indices)
    return False


def _is_expand_dims_alias(call: T.PrimFuncCall) -> bool:
    """Return True if the call is an expand_dims (unsqueeze) that only adds
    a size-1 dimension.  These are redundant under Triton semantics where
    reductions already drop the axis and consumers broadcast directly."""
    if len(call.input_tensors) != 1:
        return False
    in_shape = call.input_tensors[0].shape
    out_shape = call.out_var_tensor.shape
    if len(out_shape) != len(in_shape) + 1:
        return False
    # The extra dimension must be size 1.
    store = _extract_single_store(call.primfunc.root_node)
    if store is None:
        return False
    value = store.value
    if not isinstance(value, T.Unsqueeze):
        return False
    axis = value.axis
    if axis < 0 or axis > len(in_shape):
        return False
    expected = list(in_shape)
    expected.insert(axis, 1)
    out_ints = [d if isinstance(d, int) else None for d in out_shape]
    if out_ints != expected:
        return False
    return True


def _analyze_identity_call(call: T.PrimFuncCall) -> tuple[bool, str]:
    store = _extract_single_store(call.primfunc.root_node)
    if store is None:
        return False, "no_single_store"
    value = store.value
    if isinstance(value, T.GenericCall) and value.func_name == "transpose":
        if not value.args:
            return False, "transpose_without_args"
        load = value.args[0]
        if not isinstance(load, T.Load):
            return False, "transpose_base_not_load"
        if len(value.args) == 1:
            perm = list(reversed(range(len(load.index.indices))))
        else:
            perm = []
            for arg in value.args[1:]:
                if not isinstance(arg, T.Const):
                    return False, "transpose_perm_not_const"
                if not isinstance(arg.value, int):
                    return False, "transpose_perm_not_int"
                perm.append(arg.value)
        if perm != list(range(len(perm))):
            return False, f"transpose_not_identity:{perm}"
        value = load
    if isinstance(value, T.Permute3):
        if (value.d0, value.d1, value.d2) != (0, 1, 2):
            return False, f"permute3_not_identity:{(value.d0, value.d1, value.d2)}"
        if not isinstance(value.val, T.Load):
            return False, "permute3_base_not_load"
        value = value.val
    if not isinstance(value, T.Load):
        return False, f"value_not_load:{type(value).__name__}"
    if len(call.input_tensors) != 1:
        return False, f"input_count:{len(call.input_tensors)}"
    input_tensor = call.input_tensors[0]
    if value.tensor.name != input_tensor.name:
        return False, f"load_tensor_mismatch:{value.tensor.name}!={input_tensor.name}"
    if input_tensor.shape != call.out_var_tensor.shape:
        return False, f"shape_mismatch:{input_tensor.shape}!={call.out_var_tensor.shape}"
    # Only treat as identity if indices are plain tiles/fulltile.
    if not _is_simple_identity_index(store.index):
        return False, "store_index_not_simple_identity"
    if not _is_simple_identity_index(value.index):
        return False, "load_index_not_simple_identity"
    idx_lhs = ast_to_lisp(store.index, level=-1)
    idx_rhs = ast_to_lisp(value.index, level=-1)
    if idx_lhs != idx_rhs:
        return False, f"index_mismatch:{idx_lhs}!={idx_rhs}"
    if value.tensor.name != input_tensor.name:
        return False, f"final_name_mismatch:{value.tensor.name}!={input_tensor.name}"
    return True, "identity"


def _analyze_copy_then_const_tile_overwrite_call(
    call: T.PrimFuncCall,
) -> tuple[bool, str, str | None, str | None, T.ASTNode | None]:
    root = call.primfunc.root_node
    if not isinstance(root, T.Seq):
        return False, "root_not_seq", None, None, None

    copy_store = _extract_single_store(root.left)
    overwrite_store = _extract_single_store(root.right)
    if copy_store is None or overwrite_store is None:
        return False, "seq_branch_not_single_store", None, None, None

    if copy_store.tensor.name != call.out_var_tensor.name:
        return False, f"copy_target_mismatch:{copy_store.tensor.name}!={call.out_var_tensor.name}", None, None, None
    if overwrite_store.tensor.name != call.out_var_tensor.name:
        return False, f"overwrite_target_mismatch:{overwrite_store.tensor.name}!={call.out_var_tensor.name}", None, None, None

    copy_value = copy_store.value
    overwrite_value = overwrite_store.value
    if not isinstance(copy_value, T.Load):
        return False, f"copy_value_not_load:{type(copy_value).__name__}", None, None, None
    if not isinstance(overwrite_value, T.Load):
        return False, f"overwrite_value_not_load:{type(overwrite_value).__name__}", None, None, None

    if not isinstance(copy_store.index, T.Index) or not isinstance(copy_value.index, T.Index):
        return False, "copy_index_not_index", None, None, None
    if not isinstance(overwrite_store.index, T.Index) or not isinstance(overwrite_value.index, T.Index):
        return False, "overwrite_index_not_index", None, None, None

    if len(copy_store.index.indices) != len(copy_value.index.indices):
        return False, "copy_rank_mismatch", None, None, None
    if len(overwrite_store.index.indices) != len(overwrite_value.index.indices):
        return False, "overwrite_rank_mismatch", None, None, None

    copy_dst = ast_to_lisp(copy_store.index, level=-1)
    copy_src = ast_to_lisp(copy_value.index, level=-1)
    if copy_dst != copy_src:
        return False, f"copy_index_mismatch:{copy_dst}!={copy_src}", None, None, None

    overwrite_const_dim = None
    overwrite_const_tile = None
    for dim, (dst_idx, src_idx) in enumerate(zip(overwrite_store.index.indices, overwrite_value.index.indices)):
        if isinstance(dst_idx, T.ConstTile):
            if overwrite_const_dim is not None:
                return False, "multiple_const_tile_dims", None, None, None
            if not isinstance(src_idx, T.FullTile):
                return False, "const_tile_src_not_fulltile", None, None, None
            overwrite_const_dim = dim
            overwrite_const_tile = dst_idx
            continue

        if ast_to_lisp(dst_idx, level=-1) != ast_to_lisp(src_idx, level=-1):
            return False, f"overwrite_index_mismatch_dim:{dim}", None, None, None

    if overwrite_const_dim is None or overwrite_const_tile is None:
        return False, "no_const_tile_dim", None, None, None

    copy_src_name = copy_value.tensor.name
    overwrite_src_name = overwrite_value.tensor.name
    if copy_src_name == overwrite_src_name:
        return False, "copy_and_overwrite_same_source", None, None, None

    return (
        True,
        f"copy_then_const_tile_overwrite:base={copy_src_name},update={overwrite_src_name},"
        f"dim={overwrite_const_dim},start={overwrite_const_tile.start_index},"
        f"interval={overwrite_const_tile.interval}",
        copy_src_name,
        overwrite_src_name,
        root.right,
    )


def _is_identity_call(call: T.PrimFuncCall) -> bool:
    is_identity, _ = _analyze_identity_call(call)
    return is_identity


def _apply_tensor_alias(node: T.ASTNode, alias: dict[str, str]) -> T.ASTNode:
    if isinstance(node, T.Tensor):
        return T.Tensor(alias.get(node.name, node.name))
    if isinstance(node, T.Index):
        return node
    if isinstance(node, T.Load):
        return T.Load(_apply_tensor_alias(node.tensor, alias), node.index)
    if isinstance(node, T.Store):
        return T.Store(
            _apply_tensor_alias(node.tensor, alias),
            _apply_tensor_alias(node.value, alias),
            node.index
        )
    if isinstance(node, T.Seq):
        return T.Seq(
            _apply_tensor_alias(node.left, alias),
            _apply_tensor_alias(node.right, alias),
        )
    if isinstance(node, T.Block):
        return T.Block([_apply_tensor_alias(stmt, alias) for stmt in node.stmts])
    if isinstance(node, T.Loop):
        return T.Loop(
            node.start,
            node.end,
            node.tile_name,
            node.loop_var,
            _apply_tensor_alias(node.body, alias),
        )
    if isinstance(node, T.If):
        return T.If(
            _apply_tensor_alias(node.cond, alias),
            _apply_tensor_alias(node.then_branch, alias),
            _apply_tensor_alias(node.else_branch, alias) if node.else_branch else None,
        )
    if isinstance(node, T.Let):
        return T.Let(
            _apply_tensor_alias(node.tensor, alias),
            _apply_tensor_alias(node.value, alias),
            _apply_tensor_alias(node.body, alias),
        )
    if isinstance(node, T.Add):
        return T.Add(_apply_tensor_alias(node.left, alias), _apply_tensor_alias(node.right, alias))
    if isinstance(node, T.Sub):
        return T.Sub(_apply_tensor_alias(node.left, alias), _apply_tensor_alias(node.right, alias))
    if isinstance(node, T.Mul):
        return T.Mul(_apply_tensor_alias(node.left, alias), _apply_tensor_alias(node.right, alias))
    if isinstance(node, T.Div):
        return T.Div(_apply_tensor_alias(node.left, alias), _apply_tensor_alias(node.right, alias))
    if isinstance(node, T.Exp):
        return T.Exp(_apply_tensor_alias(node.val, alias))
    if isinstance(node, T.Sqr):
        return T.Sqr(_apply_tensor_alias(node.val, alias))
    if isinstance(node, T.Sqrt):
        return T.Sqrt(_apply_tensor_alias(node.val, alias))
    if isinstance(node, T.Sigmoid):
        return T.Sigmoid(_apply_tensor_alias(node.val, alias))
    if isinstance(node, T.Matmul):
        return T.Matmul(_apply_tensor_alias(node.left, alias), _apply_tensor_alias(node.right, alias))
    if isinstance(node, T.Take):
        return T.Take(
            _apply_tensor_alias(node.data, alias),
            _apply_tensor_alias(node.indices, alias),
            node.axis,
            _apply_tensor_alias(node.index, alias),
        )
    if isinstance(node, T.ReduceSum):
        return T.ReduceSum(_apply_tensor_alias(node.val, alias), node.axis)
    if isinstance(node, T.ReduceMax):
        return T.ReduceMax(_apply_tensor_alias(node.val, alias), node.axis)
    if isinstance(node, T.ReduceMin):
        return T.ReduceMin(_apply_tensor_alias(node.val, alias), node.axis)
    if isinstance(node, T.Concat):
        return T.Concat(
            _apply_tensor_alias(node.a, alias),
            _apply_tensor_alias(node.b, alias),
            node.axis,
        )
    if isinstance(node, T.Broadcast):
        return T.Broadcast(_apply_tensor_alias(node.val, alias), node.axis)
    if isinstance(node, T.Permute3):
        return T.Permute3(_apply_tensor_alias(node.val, alias), node.d0, node.d1, node.d2)
    if isinstance(node, T.Squeeze):
        return T.Squeeze(_apply_tensor_alias(node.val, alias), node.axis)
    if isinstance(node, T.Unsqueeze):
        return T.Unsqueeze(_apply_tensor_alias(node.val, alias), node.axis)
    if isinstance(node, T.GenericBinary):
        return T.GenericBinary(
            node.op,
            _apply_tensor_alias(node.left, alias),
            _apply_tensor_alias(node.right, alias),
        )
    if isinstance(node, T.GenericCall):
        return T.GenericCall(node.func_name, [_apply_tensor_alias(arg, alias) for arg in node.args])
    if isinstance(node, T.Cast):
        return T.Cast(node.dtype, _apply_tensor_alias(node.val, alias))
    return node


def _apply_alias_to_call(call: T.PrimFuncCall, alias: dict[str, str], root_override: T.ASTNode | None = None) -> T.PrimFuncCall:
    if not alias:
        if root_override is None:
            return call
        return T.PrimFuncCall(
            primfunc=T.PrimFunc(
                name=call.primfunc.name,
                input_tensors=call.primfunc.input_tensors,
                output_tensor=call.primfunc.output_tensor,
                spatial_axes=call.primfunc.spatial_axes,
                root_node=root_override,
                allocated_tensors=call.primfunc.allocated_tensors,
            ),
            out_var_tensor=call.out_var_tensor,
            input_tensors=call.input_tensors,
            call_index=call.call_index,
        )

    new_inputs = [
        T.TensorInfo(alias.get(t.name, t.name), t.shape, t.dtype) for t in call.input_tensors
    ]
    out_name = alias.get(call.out_var_tensor.name, call.out_var_tensor.name)
    new_out = T.TensorInfo(out_name, call.out_var_tensor.shape, call.out_var_tensor.dtype)
    root_node = root_override or call.primfunc.root_node
    new_root = _apply_tensor_alias(root_node, alias)
    primfunc_inputs = [
        T.TensorInfo(alias.get(t.name, t.name), t.shape, t.dtype) for t in call.primfunc.input_tensors
    ]
    primfunc_output_name = alias.get(call.primfunc.output_tensor.name, call.primfunc.output_tensor.name)
    primfunc_output = T.TensorInfo(
        primfunc_output_name,
        call.primfunc.output_tensor.shape,
        call.primfunc.output_tensor.dtype,
    )
    primfunc_allocated = [
        T.TensorInfo(alias.get(t.name, t.name), t.shape, t.dtype)
        for t in call.primfunc.allocated_tensors
    ]
    return T.PrimFuncCall(
        primfunc=T.PrimFunc(
            name=call.primfunc.name,
            input_tensors=primfunc_inputs,
            output_tensor=primfunc_output,
            spatial_axes=call.primfunc.spatial_axes,
            root_node=new_root,
            allocated_tensors=primfunc_allocated,
        ),
        out_var_tensor=new_out,
        input_tensors=new_inputs,
        call_index=call.call_index,
    )


def _inline_transpose_root(call: T.PrimFuncCall) -> T.ASTNode | None:
    store = _extract_single_store(call.primfunc.root_node)
    if store is None:
        return None
    value = store.value
    out_rank = len(call.out_var_tensor.shape)
    in_rank = len(call.input_tensors[0].shape) if call.input_tensors else 0
    load_rank = in_rank or out_rank
    if isinstance(value, T.Permute3):
        load = T.Load(T.Tensor(call.input_tensors[0].name), _fulltile_index(load_rank))
        value = T.Permute3(load, value.d0, value.d1, value.d2)
    elif isinstance(value, T.GenericCall) and value.func_name == "transpose":
        load = T.Load(T.Tensor(call.input_tensors[0].name), _fulltile_index(load_rank))
        value = T.GenericCall("transpose", [load] + value.args[1:])
    else:
        return None
    out_index = _fulltile_index(out_rank)
    return T.Store(T.Tensor(call.out_var_tensor.name), value, out_index)


def _calls_to_seq_lisp(
    calls: List[T.PrimFuncCall],
    level: int = 0,
    call_to_lisp=primfunc_call_to_lisp,
) -> str:
    if not calls:
        return ""
    if len(calls) == 1:
        return call_to_lisp(calls[0], level)
    prefix = "  " * level if level >= 0 else ""
    left = call_to_lisp(calls[0], level + 1)
    right = _calls_to_seq_lisp(calls[1:], 0, call_to_lisp=call_to_lisp)
    return f"{prefix}(seq\n{left}\n{right}\n{prefix})"


def primfunc_call_root_to_lisp(call: T.PrimFuncCall, level: int = 0, role_map=None) -> str:
    return ast_to_lisp(call.primfunc.root_node, level, role_map=role_map)


def _filter_identity_and_apply_alias_calls(
    calls: List[T.PrimFuncCall],
) -> tuple[List[T.PrimFuncCall], dict[str, str]]:
    filtered: List[T.PrimFuncCall] = []
    alias: dict[str, str] = {}
    for call in calls:
        is_identity, reason = _analyze_identity_call(call)
        if is_identity:
            src = call.input_tensors[0].name
            dst = call.out_var_tensor.name
            if _IR_DEBUG:
                print(f"[filter_identity] alias {dst} -> {src} ({call.primfunc.name}, call {call.call_index})")
            alias[dst] = alias.get(src, src)
            continue
        # Treat expand_dims (unsqueeze adding a size-1 dim) as alias-able:
        # Triton semantics drops the reduction axis, so the expand_dims
        # that re-adds it is redundant — consumers can broadcast directly.
        if _is_expand_dims_alias(call):
            src = call.input_tensors[0].name
            dst = call.out_var_tensor.name
            if _IR_DEBUG:
                print(f"[filter_identity] alias {dst} -> {src} ({call.primfunc.name}, call {call.call_index}, expand_dims)")
            alias[dst] = alias.get(src, src)
            continue
        scatter_like, scatter_reason, scatter_base, _, scatter_root = _analyze_copy_then_const_tile_overwrite_call(call)
        if scatter_like:
            resolved_base = alias.get(scatter_base, scatter_base) if scatter_base else None
            if resolved_base and resolved_base.startswith("const_") and scatter_root is not None:
                dst = call.out_var_tensor.name
                if _IR_DEBUG:
                    print(
                        f"[filter_identity] alias {dst} -> {resolved_base} "
                        f"({call.primfunc.name}, call {call.call_index}, scatter-base-keep-overwrite)"
                    )
                alias[dst] = resolved_base
                filtered.append(_apply_alias_to_call(call, alias, root_override=scatter_root))
                continue
            if _IR_DEBUG:
                print(
                    f"[filter_identity] recognize {call.out_var_tensor.name} "
                    f"({call.primfunc.name}, call {call.call_index}): {scatter_reason}"
                )
            filtered.append(_apply_alias_to_call(call, alias))
            continue
        if call.primfunc.name == "strided_slice":
            if _IR_DEBUG:
                print(
                    f"[filter_identity] keep {call.out_var_tensor.name} "
                    f"({call.primfunc.name}, call {call.call_index}): {reason}"
                )
        filtered.append(_apply_alias_to_call(call, alias))
    return filtered, alias


def _filter_identity_and_apply_alias(main_func: T.MainFunc) -> T.MainFunc:
    filtered, alias = _filter_identity_and_apply_alias_calls(main_func.calls)
    if alias and filtered:
        first_call = filtered[0]
        # root = _inline_transpose_root(first_call) or first_call.primfunc.root_node
        root = first_call.primfunc.root_node
        filtered = [_apply_alias_to_call(first_call, alias, root)] + filtered[1:]
    input_tensors, output_tensors, intermediate_tensors = _rewrite_tensor_metadata(
        main_func,
        filtered,
        alias,
    )
    return T.MainFunc(
        calls=filtered,
        # input_tensors=main_func.input_tensors,
        # output_tensors=main_func.output_tensors,
        # intermediate_tensors=main_func.intermediate_tensors,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        intermediate_tensors=intermediate_tensors,
    )


def filter_identity_and_apply_alias(main_func: T.MainFunc) -> T.MainFunc:
    return _filter_identity_and_apply_alias(main_func)


def _rewrite_tensor_metadata(
    main_func: T.MainFunc,
    calls: List[T.PrimFuncCall],
    alias: dict[str, str],
) -> tuple[List[T.TensorInfo], List[T.TensorInfo], List[T.TensorInfo]]:
    latest_by_name: dict[str, T.TensorInfo] = {}

    def remember(tensor: T.TensorInfo) -> None:
        latest_by_name[tensor.name] = T.TensorInfo(tensor.name, list(tensor.shape), tensor.dtype)

    def rewrite_tensor(tensor: T.TensorInfo) -> T.TensorInfo:
        aliased_name = alias.get(tensor.name, tensor.name)
        latest = latest_by_name.get(aliased_name)
        if latest is not None:
            return T.TensorInfo(latest.name, list(latest.shape), latest.dtype)
        return T.TensorInfo(aliased_name, list(tensor.shape), tensor.dtype)

    def dedupe_keep_last(tensors: List[T.TensorInfo]) -> List[T.TensorInfo]:
        ordered: dict[str, T.TensorInfo] = {}
        for tensor in tensors:
            ordered[tensor.name] = tensor
        return list(ordered.values())

    for call in calls:
        for tensor in call.input_tensors:
            remember(tensor)
        remember(call.out_var_tensor)
        for tensor in call.primfunc.input_tensors:
            remember(tensor)
        remember(call.primfunc.output_tensor)
        for tensor in call.primfunc.allocated_tensors:
            remember(tensor)

    input_tensors = dedupe_keep_last([rewrite_tensor(t) for t in main_func.input_tensors])
    output_tensors = dedupe_keep_last([rewrite_tensor(t) for t in main_func.output_tensors])

    reserved_names = {tensor.name for tensor in input_tensors}
    reserved_names.update(tensor.name for tensor in output_tensors)

    rewritten_intermediates = [rewrite_tensor(t) for t in main_func.intermediate_tensors]
    intermediate_tensors = dedupe_keep_last(
        [tensor for tensor in rewritten_intermediates if tensor.name not in reserved_names]
    )

    return input_tensors, output_tensors, intermediate_tensors


def calls_to_ir(main_func: T.MainFunc, level: int = 0, role_map=None) -> str:
    """
    Convert MainFunc to Trinity IR.
    """
    calls = main_func.calls
    if role_map is None:
        role_map = {t.name: "input" for t in main_func.input_tensors}
        for tensor in main_func.output_tensors:
            role_map[tensor.name] = "output"

    if not calls:
        return ""
    
    def call_to_lisp(call: T.PrimFuncCall, local_level: int = 0) -> str:
        return primfunc_call_root_to_lisp(call, local_level, role_map=role_map)

    return _calls_to_seq_lisp(
        calls,
        level,
        call_to_lisp=call_to_lisp,
    )


def calls_to_ir_with_groups(main_func: T.MainFunc, groups, level: int = 0, role_map=None) -> str:
    calls = main_func.calls
    if role_map is None:
        role_map = {t.name: "input" for t in main_func.input_tensors}
        for tensor in main_func.output_tensors:
            role_map[tensor.name] = "output"

    group_by_start = {group.call_indices[0]: group for group in groups}
    skipped = {idx for group in groups for idx in group.call_indices[1:]}
    exprs: List[str] = []

    for idx, call in enumerate(calls):
        if idx in skipped:
            continue
        group = group_by_start.get(idx)
        if group is None:
            exprs.append(primfunc_call_root_to_lisp(call, 0, role_map=role_map))
            continue
        exprs.append(_group_to_lisp(call, group, role_map=role_map))

    return _exprs_to_seq_lisp(exprs, level)

def _get_clean_name(raw_name: str) -> str:
    return raw_name.replace("v_", "")


def _group_to_lisp(call: T.PrimFuncCall, group, role_map=None) -> str:
    local_role_map = dict(role_map or {})

    def register_joined_role(joined_name: str, names: list[str]) -> None:
        roles = {local_role_map.get(name) for name in names}
        roles.discard(None)
        if len(roles) == 1:
            local_role_map[joined_name] = next(iter(roles))

    read_slot_map: dict[str, int] = {}
    write_slot_map: dict[str, int] = {}

    for names in group.varying_read_slots.values():
        register_joined_role(",".join(names), names)
    for names in group.write_slots.values():
        register_joined_role(",".join(names), names)

    def assign_read(name: str) -> int:
        if name not in read_slot_map:
            read_slot_map[name] = len(read_slot_map)
        return read_slot_map[name]

    def assign_write(name: str) -> int:
        if name not in write_slot_map:
            write_slot_map[name] = len(write_slot_map)
        return write_slot_map[name]

    def rewrite(node: T.ASTNode) -> T.ASTNode:
        if isinstance(node, T.Load):
            slot = assign_read(node.tensor.name)
            if slot in group.exact_read_slots:
                tensor_name = group.exact_read_slots[slot]
            else:
                tensor_name = ",".join(group.varying_read_slots[slot])
            return T.Load(T.Tensor(tensor_name), rewrite(node.index))
        if isinstance(node, T.Store):
            slot = assign_write(node.tensor.name)
            tensor_name = ",".join(group.write_slots[slot])
            return T.Store(T.Tensor(tensor_name), rewrite(node.value), rewrite(node.index))
        if isinstance(node, T.Seq):
            return T.Seq(rewrite(node.left), rewrite(node.right))
        if isinstance(node, T.Block):
            return T.Block([rewrite(stmt) for stmt in node.stmts])
        if isinstance(node, T.Loop):
            return T.Loop(node.start, node.end, node.tile_name, node.loop_var, rewrite(node.body))
        if isinstance(node, T.If):
            else_branch = rewrite(node.else_branch) if node.else_branch else None
            return T.If(rewrite(node.cond), rewrite(node.then_branch), else_branch)
        if isinstance(node, T.Let):
            return T.Let(rewrite(node.tensor), rewrite(node.value), rewrite(node.body))
        if isinstance(node, T.Index):
            return T.Index([rewrite(idx) for idx in node.indices])
        if isinstance(node, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min, T.Matmul)):
            return node.__class__(rewrite(node.left), rewrite(node.right))
        if isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
            return node.__class__(rewrite(node.val))
        if isinstance(node, T.Cast):
            return T.Cast(node.dtype, rewrite(node.val))
        if isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin, T.Broadcast)):
            return node.__class__(rewrite(node.val), node.axis)
        if isinstance(node, T.Concat):
            return T.Concat(rewrite(node.a), rewrite(node.b), node.axis)
        if isinstance(node, T.Permute3):
            return T.Permute3(rewrite(node.val), node.d0, node.d1, node.d2)
        if isinstance(node, T.Squeeze):
            return T.Squeeze(rewrite(node.val), node.axis)
        if isinstance(node, T.Unsqueeze):
            return T.Unsqueeze(rewrite(node.val), node.axis)
        if isinstance(node, T.GenericBinary):
            return T.GenericBinary(node.op, rewrite(node.left), rewrite(node.right))
        if isinstance(node, T.GenericCall):
            return T.GenericCall(node.func_name, [rewrite(arg) for arg in node.args])
        if isinstance(node, T.Take):
            return T.Take(rewrite(node.data), rewrite(node.indices), node.axis, rewrite(node.index))
        return node

    new_root = rewrite(call.primfunc.root_node)
    return ast_to_lisp(new_root, 0, role_map=local_role_map)


def _exprs_to_seq_lisp(exprs: List[str], level: int = 0) -> str:
    if not exprs:
        return ""
    if len(exprs) == 1:
        return _indent_block(exprs[0], level)
    prefix = "  " * level if level >= 0 else ""
    left = _indent_block(exprs[0], level + 1)
    right = _exprs_to_seq_lisp(exprs[1:], 0)
    return f"{prefix}(seq\n{left}\n{right}\n{prefix})"


def _indent_block(text: str, level: int) -> str:
    prefix = "  " * level
    return "\n".join(f"{prefix}{line}" if line else line for line in text.splitlines())
