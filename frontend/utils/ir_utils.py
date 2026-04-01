import copy
import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import ir.AST as T
import os 

_IR_DEBUG = os.getenv("TRINITY_DEBUG_IR", "").lower() in {"1", "true", "yes", "on"}

@dataclass
class FusionGroup:
    call_indices: List[int]
    signature: str
    exact_read_slots: Dict[int, str]
    varying_read_slots: Dict[int, List[str]]
    write_slots: Dict[int, List[str]]

def bind_primfunc_to_call(call: T.PrimFuncCall) -> T.PrimFunc:
    """
    Rename primfunc input/output tensors to match the call's global tensors.
    """
    primfunc = call.primfunc
    rename_map: dict[str, str] = {}
    for idx, input_tensor in enumerate(primfunc.input_tensors):
        if idx < len(call.input_tensors):
            rename_map[input_tensor.name] = call.input_tensors[idx].name
    rename_map[primfunc.output_tensor.name] = call.out_var_tensor.name

    new_inputs = [_rename_tensor_info(t, rename_map) for t in primfunc.input_tensors]
    new_output = _rename_tensor_info(primfunc.output_tensor, rename_map)
    new_root = _rename_tensor_nodes(primfunc.root_node, rename_map)
    return T.PrimFunc(
        name=primfunc.name,
        input_tensors=new_inputs,
        output_tensor=new_output,
        spatial_axes=primfunc.spatial_axes,
        root_node=new_root,
        allocated_tensors=primfunc.allocated_tensors,
    )

def bind_main_func_calls(main_func: T.MainFunc) -> T.MainFunc:
    bound_calls: list[T.PrimFuncCall] = []
    for call in main_func.calls:
        bound_calls.append(
            T.PrimFuncCall(
                primfunc=bind_primfunc_to_call(call),
                out_var_tensor=call.out_var_tensor,
                input_tensors=call.input_tensors,
                call_index=call.call_index,
            )
        )
    return T.MainFunc(
        calls=bound_calls,
        input_tensors=main_func.input_tensors,
        output_tensors=main_func.output_tensors,
        intermediate_tensors=main_func.intermediate_tensors,
    )

def analyze_axis_access_patterns(primfunc: T.PrimFunc) -> dict[str, dict[str, set[str]]]:
    """
    Analyze axis usage from tensor access patterns.
    Returns: {axis: {"reads": set(sig), "writes": set(sig)}} where
    sig = "tensor[dim_index]=expr_signature".
    """
    patterns: dict[str, dict[str, set[str]]] = {}

    def record(axis: str, kind: str, tensor: str, dim_index: int, expr_sig: str):
        axis_key = _clean_var_name(axis)
        entry = patterns.setdefault(axis_key, {"reads": set(), "writes": set()})
        entry[kind].add(f"{tensor}[{dim_index}]={expr_sig}")

    def collect_axes(expr: T.ASTNode) -> set[str]:
        axes: set[str] = set()
        if isinstance(expr, T.Tile):
            axes.add(expr.name)
        elif isinstance(expr, T.VarRef):
            axes.add(expr.name)
        elif isinstance(expr, T.TileOffset):
            axes.add(expr.name)
        elif isinstance(expr, T.ConstTile):
            return axes
        elif isinstance(expr, T.Arange):
            axes.add(expr.axis)
        elif isinstance(expr, T.Index):
            for idx in expr.indices:
                axes.update(collect_axes(idx))
        elif isinstance(expr, (T.Add, T.Sub, T.Mul, T.Div, T.Matmul)):
            axes.update(collect_axes(expr.left))
            axes.update(collect_axes(expr.right))
        elif isinstance(expr, T.Take):
            axes.update(collect_axes(expr.data))
            axes.update(collect_axes(expr.indices))
            axes.update(collect_axes(expr.index))
        elif isinstance(expr, T.GenericBinary):
            axes.update(collect_axes(expr.left))
            axes.update(collect_axes(expr.right))
        elif isinstance(expr, T.GenericCall):
            for arg in expr.args:
                axes.update(collect_axes(arg))
        elif isinstance(expr, T.Cast):
            axes.update(collect_axes(expr.val))
        elif isinstance(expr, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
            axes.update(collect_axes(expr.val))
        elif isinstance(expr, T.Broadcast):
            axes.update(collect_axes(expr.val))
        elif isinstance(expr, T.Concat):
            axes.update(collect_axes(expr.a))
            axes.update(collect_axes(expr.b))
        return axes

    def expr_signature(expr: T.ASTNode) -> str:
        if isinstance(expr, T.Tile):
            return f"tile:{_clean_var_name(expr.name)}"
        if isinstance(expr, T.VarRef):
            return f"var:{_clean_var_name(expr.name)}"
        if isinstance(expr, T.TileOffset):
            return f"tile_offset:{_clean_var_name(expr.name)}:{expr.offset}"
        if isinstance(expr, T.ConstTile):
            return f"const_tile:{expr.start_index}:{expr.interval}"
        if isinstance(expr, T.Elem):
            return f"elem:{_clean_var_name(expr.name)}"
        if isinstance(expr, T.FullTile):
            return "fulltile"
        if isinstance(expr, T.Const):
            return str(expr.value)
        if isinstance(expr, T.Add):
            return f"(+ {expr_signature(expr.left)} {expr_signature(expr.right)})"
        if isinstance(expr, T.Sub):
            return f"(- {expr_signature(expr.left)} {expr_signature(expr.right)})"
        if isinstance(expr, T.Mul):
            return f"(* {expr_signature(expr.left)} {expr_signature(expr.right)})"
        if isinstance(expr, T.Matmul):
            return f"(@ {expr_signature(expr.left)} {expr_signature(expr.right)})"
        if isinstance(expr, T.Take):
            return f"(take {expr_signature(expr.data)} {expr_signature(expr.indices)} {expr.axis} {expr_signature(expr.index)})"
        if isinstance(expr, T.ReduceSum):
            return f"(rsum {expr_signature(expr.val)} {expr.axis})"
        if isinstance(expr, T.ReduceMax):
            return f"(rmax {expr_signature(expr.val)} {expr.axis})"
        if isinstance(expr, T.ReduceMin):
            return f"(rmin {expr_signature(expr.val)} {expr.axis})"
        if isinstance(expr, T.Concat):
            return f"(concat {expr_signature(expr.a)} {expr_signature(expr.b)} {expr.axis})"
        if isinstance(expr, T.Arange):
            return f"(arange {expr.axis})"
        if isinstance(expr, T.Div):
            return f"(/ {expr_signature(expr.left)} {expr_signature(expr.right)})"
        if isinstance(expr, T.GenericBinary):
            return f"({expr.op} {expr_signature(expr.left)} {expr_signature(expr.right)})"
        if isinstance(expr, T.Cast):
            return f"(cast {expr.dtype} {expr_signature(expr.val)})"
        if isinstance(expr, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
            name = expr.__class__.__name__.lower()
            return f"({name} {expr_signature(expr.val)})"
        if isinstance(expr, T.Broadcast):
            return f"(bcast {expr_signature(expr.val)} {expr.axis})"
        if isinstance(expr, T.GenericCall):
            args = " ".join([expr_signature(arg) for arg in expr.args])
            return f"({expr.func_name} {args})"
        if isinstance(expr, T.Index):
            parts = " ".join([expr_signature(idx) for idx in expr.indices])
            return f"(index {parts})"
        return expr.__class__.__name__

    def visit(node: T.ASTNode):
        if isinstance(node, T.Store):
            tensor_name = node.tensor.name
            for dim, idx in enumerate(node.index.indices):
                sig = expr_signature(idx)
                for axis in collect_axes(idx):
                    record(axis, "writes", tensor_name, dim, sig)
            visit(node.value)
            return
        if isinstance(node, T.Load):
            tensor_name = node.tensor.name
            for dim, idx in enumerate(node.index.indices):
                sig = expr_signature(idx)
                for axis in collect_axes(idx):
                    record(axis, "reads", tensor_name, dim, sig)
            return
        if isinstance(node, T.Block):
            for stmt in node.stmts:
                visit(stmt)
            return
        if isinstance(node, T.Seq):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, T.Loop):
            visit(node.body)
            return
        if isinstance(node, T.If):
            visit(node.then_branch)
            if node.else_branch:
                visit(node.else_branch)
            return
        if isinstance(node, T.Let):
            visit(node.value)
            visit(node.body)
            return
        if isinstance(node, (T.Add, T.Sub, T.Mul, T.Div, T.Matmul)):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, T.Take):
            visit(node.data)
            visit(node.indices)
            visit(node.index)
            return
        if isinstance(node, T.GenericBinary):
            visit(node.left)
            visit(node.right)
            return
        if isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
            visit(node.val)
            return
        if isinstance(node, T.Broadcast):
            visit(node.val)
            return
        if isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin)):
            visit(node.val)
            return
        if isinstance(node, (T.Squeeze, T.Unsqueeze)):
            visit(node.val)
            return
        if isinstance(node, T.GenericCall):
            for arg in node.args:
                visit(arg)
            return
        if isinstance(node, T.Cast):
            visit(node.val)
            return
        if isinstance(node, T.Concat):
            visit(node.a)
            visit(node.b)
            return
        if isinstance(node, T.Arange):
            return

    visit(primfunc.root_node)
    return patterns

def normalize_main_func_axes(main_func: T.MainFunc) -> T.MainFunc:
    """
    Normalize loop vars across calls based on axis access patterns.
    Shape exprs are resolved recursively to their earliest source tensor.
    PrimFunc tensors should already be bound to call tensors.
    """
    canonical_groups: list[dict[str, object]] = []
    axis_pool = list("ijklmnopqrstuvwxyz")
    axis_index = 0

    def _collect_outermost_loop_var(root_node: T.ASTNode) -> Optional[str]:
        node = root_node
        while True:
            if isinstance(node, T.Loop):
                return _clean_var_name(node.loop_var)
            if isinstance(node, T.Seq):
                node = node.left
                continue
            if isinstance(node, T.Block) and node.stmts:
                node = node.stmts[0]
                continue
            if isinstance(node, T.Let):
                node = node.body
                continue
            return None

    def axis_access_map(patterns: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, set[int]]]:
        access_map: dict[str, dict[str, set[int]]] = {}
        for axis, entry in patterns.items():
            axis_access: dict[str, set[int]] = {}
            for kind in ("reads", "writes"):
                for sig in entry.get(kind, set()):
                    parsed = _parse_tensor_dim_sig(sig)
                    if parsed is None:
                        continue
                    tensor, dim = parsed
                    axis_access.setdefault(tensor, set()).add(dim)
            access_map[axis] = axis_access
        return access_map

    def axis_range_map(root_node: T.ASTNode) -> dict[str, tuple[Optional[int], Optional[int]]]:
        ranges: dict[str, tuple[Optional[int], Optional[int]]] = {}

        def visit(node: T.ASTNode):
            if isinstance(node, T.Loop):
                start = _eval_int(node.start)
                end = _eval_int(node.end)
                loop_key = _clean_var_name(node.loop_var)
                ranges[loop_key] = (start, end)
                visit(node.body)
                return
            if isinstance(node, T.Block):
                for stmt in node.stmts:
                    visit(stmt)
                return
            if isinstance(node, T.Seq):
                visit(node.left)
                visit(node.right)
                return
            if isinstance(node, T.If):
                visit(node.then_branch)
                if node.else_branch:
                    visit(node.else_branch)
                return
            if isinstance(node, T.Let):
                visit(node.body)
                return
            if isinstance(
                node,
                (
                    T.Store,
                    T.Load,
                    T.Add,
                    T.Sub,
                    T.Mul,
                    T.Div,
                    T.Matmul,
                    T.Take,
                    T.Exp,
                    T.Sqr,
                    T.Sqrt,
                    T.Sigmoid,
                    T.GenericBinary,
                    T.GenericCall,
                    T.Cast,
                ),
            ):
                return

        visit(root_node)
        return ranges

    def axis_tile_map(root_node: T.ASTNode) -> dict[str, Optional[str]]:
        tiles: dict[str, Optional[str]] = {}

        def visit(node: T.ASTNode):
            if isinstance(node, T.Loop):
                loop_key = _clean_var_name(node.loop_var)
                tiles[loop_key] = node.tile_name
                visit(node.body)
                return
            if isinstance(node, T.Block):
                for stmt in node.stmts:
                    visit(stmt)
                return
            if isinstance(node, T.Seq):
                visit(node.left)
                visit(node.right)
                return
            if isinstance(node, T.If):
                visit(node.then_branch)
                if node.else_branch:
                    visit(node.else_branch)
                return
            if isinstance(node, T.Let):
                visit(node.body)
                return

        visit(root_node)
        return tiles

    def _is_tile_constant(tile: str) -> bool:
        if not tile.startswith("tile_"):
            return False
        return tile[len("tile_"):].isdigit()

    def tile_compatible(axis_tile: Optional[str], group_tiles: set[Optional[str]]) -> bool:
        if axis_tile is None:
            return None in group_tiles
        if _is_tile_constant(axis_tile):
            return axis_tile in group_tiles
        # Treat any tile variable as compatible with other tile variables.
        return any(t is not None and not _is_tile_constant(t) for t in group_tiles)

    def access_compatible(
        access: dict[str, set[int]],
        group_access: dict[str, set[int]],
    ) -> bool:
        shared = set(access.keys()) & set(group_access.keys())
        for tensor in shared:
            if access[tensor] != group_access[tensor]:
                return False
        return True

    def _collect_loop_vars(root_node: T.ASTNode) -> list[str]:
        vars_seen: list[str] = []
        seen = set()

        def visit(node: T.ASTNode):
            if isinstance(node, T.Loop):
                if node.loop_var not in seen:
                    seen.add(node.loop_var)
                    vars_seen.append(node.loop_var)
                visit(node.body)
                return
            if isinstance(node, T.Block):
                for stmt in node.stmts:
                    visit(stmt)
                return
            if isinstance(node, T.Seq):
                visit(node.left)
                visit(node.right)
                return
            if isinstance(node, T.If):
                visit(node.then_branch)
                if node.else_branch:
                    visit(node.else_branch)
                return
            if isinstance(node, T.Let):
                visit(node.body)
                return

        visit(root_node)
        return vars_seen

    updated_calls: list[T.PrimFuncCall] = []
    for call in main_func.calls:
        bound_primfunc = call.primfunc
        patterns = analyze_axis_access_patterns(bound_primfunc)
        access_map = axis_access_map(patterns)
        range_map = axis_range_map(bound_primfunc.root_node)
        tile_map = axis_tile_map(bound_primfunc.root_node)
        outermost_axis = _collect_outermost_loop_var(bound_primfunc.root_node)

        rename_map: dict[str, str] = {}
        loop_axes = _collect_loop_vars(bound_primfunc.root_node)
        all_axes = list(dict.fromkeys(list(bound_primfunc.spatial_axes) + loop_axes))
        for axis in all_axes:
            axis_key = _clean_var_name(axis)
            access = access_map.get(axis_key, {})
            axis_range = range_map.get(axis_key, (None, None))
            axis_tile = tile_map.get(axis_key)
            axis_outer_count = 1 if outermost_axis == axis_key else 0
            scored_candidates: list[
                tuple[int, str, int, int, tuple[Optional[int], Optional[int]], Optional[str], dict[str, set[int]]]
            ] = []

            best_idx = None
            best_score = -1
            best_outer_score = -1
            for idx, group in enumerate(canonical_groups):
                if axis_range not in group["ranges"]:
                    continue
                if not tile_compatible(axis_tile, group["tiles"]):
                    continue
                if not access_compatible(access, group["access"]):
                    continue
                shared = set(access.keys()) & set(group["access"].keys())
                score = len(shared)
                outer_score = group.get("outer_count", 0)
                scored_candidates.append(
                    (
                        idx,
                        group["name"],
                        score,
                        outer_score,
                        axis_range,
                        axis_tile,
                        access,
                    )
                )
                if score > best_score or (score == best_score and outer_score > best_outer_score):
                    best_score = score
                    best_outer_score = outer_score
                    best_idx = idx

            if best_idx is None:
                name = axis_pool[axis_index] if axis_index < len(axis_pool) else f"a{axis_index}"
                axis_index += 1
                canonical_groups.append(
                    {
                        "name": name,
                        "access": {k: set(v) for k, v in access.items()},
                        "ranges": {axis_range},
                        "tiles": {axis_tile},
                        "outer_count": axis_outer_count,
                    }
                )
                rename_map[axis] = name
                if axis_key != axis:
                    rename_map[axis_key] = name
            else:
                group = canonical_groups[best_idx]
                rename_map[axis] = group["name"]
                if axis_key != axis:
                    rename_map[axis_key] = group["name"]
                group["outer_count"] = group.get("outer_count", 0) + axis_outer_count
                for tensor, dims in access.items():
                    if tensor not in group["access"]:
                        group["access"][tensor] = set(dims)
                group["ranges"].add(axis_range)
                group["tiles"].add(axis_tile)

        new_root = rename_loop_vars(bound_primfunc.root_node, rename_map)
        new_spatial = [rename_map.get(a, a) for a in bound_primfunc.spatial_axes]

        updated_calls.append(
            T.PrimFuncCall(
                primfunc=T.PrimFunc(
                    name=bound_primfunc.name,
                    input_tensors=bound_primfunc.input_tensors,
                    output_tensor=bound_primfunc.output_tensor,
                    spatial_axes=new_spatial,
                    root_node=new_root,
                    allocated_tensors=bound_primfunc.allocated_tensors,
                ),
                out_var_tensor=call.out_var_tensor,
                input_tensors=call.input_tensors,
                call_index=call.call_index,
            )
        )

    return T.MainFunc(
        calls=updated_calls,
        input_tensors=main_func.input_tensors,
        output_tensors=main_func.output_tensors,
        intermediate_tensors=main_func.intermediate_tensors,
    )

def rename_loop_vars(node: T.ASTNode, rename_map: dict[str, str]) -> T.ASTNode:
    if isinstance(node, T.Tile):
        name = rename_map.get(node.name, node.name)
        return T.Tile(name)
    if isinstance(node, T.Elem):
        name = rename_map.get(node.name, node.name)
        return T.Elem(name)
    if isinstance(node, T.FullTile):
        return node
    if isinstance(node, T.ConstTile):
        return node
    if isinstance(node, T.Tensor):
        return node
    if isinstance(node, T.Const):
        return node
    if isinstance(node, T.Index):
        return T.Index([rename_loop_vars(idx, rename_map) for idx in node.indices])
    if isinstance(node, T.Load):
        return T.Load(
            rename_loop_vars(node.tensor, rename_map),
            rename_loop_vars(node.index, rename_map),
        )
    if isinstance(node, T.Store):
        return T.Store(
            rename_loop_vars(node.tensor, rename_map),
            rename_loop_vars(node.value, rename_map),
            rename_loop_vars(node.index, rename_map),
        )
    if isinstance(node, T.Block):
        return T.Block([rename_loop_vars(stmt, rename_map) for stmt in node.stmts])
    if isinstance(node, T.Seq):
        return T.Seq(
            rename_loop_vars(node.left, rename_map),
            rename_loop_vars(node.right, rename_map),
        )
    if isinstance(node, T.If):
        return T.If(
            rename_loop_vars(node.cond, rename_map),
            rename_loop_vars(node.then_branch, rename_map),
            rename_loop_vars(node.else_branch, rename_map) if node.else_branch else None,
        )
    if isinstance(node, T.Let):
        return T.Let(
            rename_loop_vars(node.tensor, rename_map),
            rename_loop_vars(node.value, rename_map),
            rename_loop_vars(node.body, rename_map),
        )
    if isinstance(node, T.Loop):
        new_loop_var = rename_map.get(node.loop_var, node.loop_var)
        tile_name = node.tile_name
        if tile_name.startswith("tile_"):
            tile_name = f"tile_{new_loop_var}"
        return T.Loop(
            rename_loop_vars(node.start, rename_map),
            rename_loop_vars(node.end, rename_map),
            tile_name,
            new_loop_var,
            rename_loop_vars(node.body, rename_map),
        )
    if isinstance(node, T.Add):
        return T.Add(rename_loop_vars(node.left, rename_map), rename_loop_vars(node.right, rename_map))
    if isinstance(node, T.Sub):
        return T.Sub(rename_loop_vars(node.left, rename_map), rename_loop_vars(node.right, rename_map))
    if isinstance(node, T.Mul):
        return T.Mul(rename_loop_vars(node.left, rename_map), rename_loop_vars(node.right, rename_map))
    if isinstance(node, T.Div):
        return T.Div(rename_loop_vars(node.left, rename_map), rename_loop_vars(node.right, rename_map))
    if isinstance(node, T.VarRef):
        return T.VarRef(rename_map.get(node.name, node.name))
    if isinstance(node, T.TileOffset):
        return T.TileOffset(rename_map.get(node.name, node.name), node.offset)
    if isinstance(node, T.Arange):
        return T.Arange(rename_map.get(node.axis, node.axis))
    if isinstance(node, T.Exp):
        return T.Exp(rename_loop_vars(node.val, rename_map))
    if isinstance(node, T.Sqr):
        return T.Sqr(rename_loop_vars(node.val, rename_map))
    if isinstance(node, T.Sqrt):
        return T.Sqrt(rename_loop_vars(node.val, rename_map))
    if isinstance(node, T.Sigmoid):
        return T.Sigmoid(rename_loop_vars(node.val, rename_map))
    if isinstance(node, T.Matmul):
        return T.Matmul(rename_loop_vars(node.left, rename_map), rename_loop_vars(node.right, rename_map))
    if isinstance(node, T.Take):
        return T.Take(
            rename_loop_vars(node.data, rename_map),
            rename_loop_vars(node.indices, rename_map),
            node.axis,
            rename_loop_vars(node.index, rename_map),
        )
    if isinstance(node, T.ReduceSum):
        return T.ReduceSum(rename_loop_vars(node.val, rename_map), node.axis)
    if isinstance(node, T.ReduceMax):
        return T.ReduceMax(rename_loop_vars(node.val, rename_map), node.axis)
    if isinstance(node, T.ReduceMin):
        return T.ReduceMin(rename_loop_vars(node.val, rename_map), node.axis)
    if isinstance(node, T.Concat):
        return T.Concat(
            rename_loop_vars(node.a, rename_map),
            rename_loop_vars(node.b, rename_map),
            node.axis,
        )
    if isinstance(node, T.Broadcast):
        return T.Broadcast(rename_loop_vars(node.val, rename_map), node.axis)
    if isinstance(node, T.Permute3):
        return T.Permute3(
            rename_loop_vars(node.val, rename_map),
            node.d0,
            node.d1,
            node.d2,
        )
    if isinstance(node, T.Squeeze):
        return T.Squeeze(rename_loop_vars(node.val, rename_map), node.axis)
    if isinstance(node, T.Unsqueeze):
        return T.Unsqueeze(rename_loop_vars(node.val, rename_map), node.axis)
    if isinstance(node, T.GenericBinary):
        return T.GenericBinary(
            node.op,
            rename_loop_vars(node.left, rename_map),
            rename_loop_vars(node.right, rename_map),
        )
    if isinstance(node, T.GenericCall):
        return T.GenericCall(node.func_name, [rename_loop_vars(arg, rename_map) for arg in node.args])
    if isinstance(node, T.Cast):
        return T.Cast(node.dtype, rename_loop_vars(node.val, rename_map))
    return node

def _rename_tensor_info(tensor: T.TensorInfo, rename_map: dict[str, str]) -> T.TensorInfo:
    new_name = rename_map.get(tensor.name, tensor.name)
    return T.TensorInfo(name=new_name, shape=tensor.shape, dtype=tensor.dtype)

def _rename_tensor_nodes(node: T.ASTNode, rename_map: dict[str, str]) -> T.ASTNode:
    if isinstance(node, T.Tensor):
        return T.Tensor(rename_map.get(node.name, node.name))
    if isinstance(node, T.Load):
        return T.Load(_rename_tensor_nodes(node.tensor, rename_map), node.index)
    if isinstance(node, T.Store):
        return T.Store(
            _rename_tensor_nodes(node.tensor, rename_map),
            _rename_tensor_nodes(node.value, rename_map),
            node.index
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
    if isinstance(node, T.Loop):
        return T.Loop(
            _rename_tensor_nodes(node.start, rename_map),
            _rename_tensor_nodes(node.end, rename_map),
            node.tile_name,
            node.loop_var,
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
    if isinstance(node, T.Arange):
        return node
    return node

def remove_short_loop_nodes(node, eliminated_vars=None, threshold=64):
    """
    AST 또는 Lisp 텍스트를 순회하며 크기가 threshold 이하인 Loop를 제거하고
    내부의 Tile 참조를 fulltile로 변경합니다.
    """
    if not isinstance(node, str):
        transformed = _replace_short_loops_with_fulltile(node, eliminated_vars or set(), threshold)
        return _remove_redundant_self_accumulations(transformed)

    tokens = _tokenize_lisp(node)
    exprs = _parse_many(tokens)
    transformed = _transform_seq(exprs, eliminated_vars or set(), threshold)
    return _to_lisp_program(transformed)


def remove_smallest_inner_loops(node: T.ASTNode, max_extent: int = 384) -> T.ASTNode:
    """
    For loop chains with depth >= 2, repeatedly remove the smallest non-outermost loop
    whose extent is <= max_extent. Stop when the chain becomes 1-deep or no eligible
    inner loop remains.
    """
    transformed = _remove_smallest_inner_loops(node, max_extent=max_extent)
    return _remove_redundant_self_accumulations(transformed)

def remove_let_nodes(root: T.ASTNode) -> T.ASTNode:
    """
    Let stmt를 제거하고 바인딩된 값을 body에 직접 치환한 AST를 반환합니다.
    """
    return _remove_let_nodes(root, {}, substitute_tensor=True)

def inline_shape_op_calls(main_func: T.MainFunc) -> T.MainFunc:
    """
    Inline single-op producer calls (transpose/permute/squeeze/unsqueeze)
    into the immediate consumer when the producer output is used only there.
    """
    calls = list(main_func.calls)
    if len(calls) < 2:
        return main_func

    output_names = {t.name for t in main_func.output_tensors}
    intermediate_tensors = list(main_func.intermediate_tensors)
    tensor_info_map = _build_tensor_info_map(main_func)

    def _index_has_numeric_tile(index: T.Index) -> bool:
        for idx in index.indices:
            if isinstance(idx, (T.Tile, T.TileOffset)):
                clean = _clean_var_name(idx.name)
                if clean.isdigit():
                    return True
            if isinstance(idx, T.ConstTile):
                return True
        return False

    def _has_numeric_tile_name(node: T.ASTNode) -> bool:
        if isinstance(node, T.Loop):
            if node.tile_name.isdigit():
                return True
            return _has_numeric_tile_name(node.body)
        if isinstance(node, T.Seq):
            return _has_numeric_tile_name(node.left) or _has_numeric_tile_name(node.right)
        if isinstance(node, T.Block):
            return any(_has_numeric_tile_name(stmt) for stmt in node.stmts)
        if isinstance(node, T.If):
            if _has_numeric_tile_name(node.then_branch):
                return True
            if node.else_branch and _has_numeric_tile_name(node.else_branch):
                return True
            return False
        if isinstance(node, T.Let):
            return _has_numeric_tile_name(node.body)
        return False

    def _extract_inline_op(store: T.Store) -> dict[str, object] | None:
        if not isinstance(store.index, T.Index):
            return None
        if isinstance(store.value, T.Permute3):
            load = store.value.val
            if not isinstance(load, T.Load):
                return None
            if not isinstance(load.index, T.Index):
                return None
            if _index_has_numeric_tile(store.index) or _index_has_numeric_tile(load.index):
                return None
            inv_perm = [store.value.d0, store.value.d1, store.value.d2]
            return {
                "op": "permute3",
                "src": load.tensor.name,
                "inv_perm": inv_perm,
                "axis": None,
            }
        if isinstance(store.value, T.GenericCall) and store.value.func_name == "transpose":
            if not store.value.args:
                return None
            load = store.value.args[0]
            if not isinstance(load, T.Load):
                return None
            if not isinstance(load.index, T.Index):
                return None
            if _index_has_numeric_tile(store.index) or _index_has_numeric_tile(load.index):
                return None
            inv_perm: list[int] | None = None
            if len(store.value.args) > 1:
                inv_perm = []
                for arg in store.value.args[1:]:
                    if not isinstance(arg, T.Const):
                        return None
                    if not isinstance(arg.value, int):
                        return None
                    inv_perm.append(arg.value)
            return {
                "op": "transpose",
                "src": load.tensor.name,
                "inv_perm": inv_perm,
                "axis": None,
            }
        if isinstance(store.value, T.Unsqueeze):
            load = store.value.val
            if not isinstance(load, T.Load):
                return None
            if not isinstance(load.index, T.Index):
                return None
            if _index_has_numeric_tile(store.index) or _index_has_numeric_tile(load.index):
                return None
            return {
                "op": "unsqueeze",
                "src": load.tensor.name,
                "inv_perm": None,
                "axis": store.value.axis,
            }
        if isinstance(store.value, T.Squeeze):
            load = store.value.val
            if not isinstance(load, T.Load):
                return None
            if not isinstance(load.index, T.Index):
                return None
            if _index_has_numeric_tile(store.index) or _index_has_numeric_tile(load.index):
                return None
            return {
                "op": "squeeze",
                "src": load.tensor.name,
                "inv_perm": None,
                "axis": store.value.axis,
            }
        return None

    def _invert_perm(inv_perm: list[int]) -> list[int] | None:
        size = len(inv_perm)
        perm = [0] * size
        for out_axis, in_axis in enumerate(inv_perm):
            if in_axis < 0 or in_axis >= size:
                return None
            perm[in_axis] = out_axis
        return perm

    def _inline_expr(dst_index: T.Index, op_info: dict[str, object]) -> T.ASTNode | None:
        dst_indices = list(dst_index.indices)
        op = op_info["op"]
        src_name = op_info["src"]
        if not isinstance(src_name, str):
            return None
        if op in ("transpose", "permute3"):
            inv_perm = op_info.get("inv_perm")
            if inv_perm is None:
                if len(dst_indices) < 2:
                    return None
                inv_perm = list(reversed(range(len(dst_indices))))
            if not isinstance(inv_perm, list):
                return None
            perm = _invert_perm(inv_perm)
            if perm is None or len(perm) != len(dst_indices):
                return None
            src_indices = [dst_indices[p] for p in perm]
            src_load = T.Load(T.Tensor(src_name), T.Index(src_indices))
            if op == "permute3":
                return T.Permute3(src_load, inv_perm[0], inv_perm[1], inv_perm[2])
            if len(inv_perm) == 2 and op_info.get("inv_perm") is None:
                return T.GenericCall("transpose", [src_load])
            if len(inv_perm) == 2 and op_info.get("inv_perm") is not None:
                return T.GenericCall("transpose", [src_load])
            perm_args = [T.Const(p) for p in inv_perm]
            return T.GenericCall("transpose", [src_load] + perm_args)
        if op == "unsqueeze":
            axis = op_info["axis"]
            if not isinstance(axis, int):
                return None
            if axis < 0:
                axis += len(dst_indices)
            if axis < 0 or axis >= len(dst_indices):
                return None
            src_indices = dst_indices[:axis] + dst_indices[axis + 1 :]
            src_load = T.Load(T.Tensor(src_name), T.Index(src_indices))
            return T.Unsqueeze(src_load, axis)
        if op == "squeeze":
            axis = op_info["axis"]
            if not isinstance(axis, int):
                return None
            src_rank = len(dst_indices) + 1
            if axis < 0:
                axis += src_rank
            if axis < 0 or axis >= src_rank:
                return None
            src_indices = list(dst_indices)
            src_indices.insert(axis, T.FullTile())
            src_load = T.Load(T.Tensor(src_name), T.Index(src_indices))
            return T.Squeeze(src_load, axis)
        return None

    def _inline_loads_with_op(node: T.ASTNode, dst_name: str, op_info: dict[str, object]) -> T.ASTNode:
        if isinstance(node, T.Load) and node.tensor.name == dst_name:
            if isinstance(node.index, T.Index):
                expr = _inline_expr(node.index, op_info)
                if expr is not None:
                    return expr
            return node
        if isinstance(node, T.Store):
            return T.Store(
                node.tensor,
                _inline_loads_with_op(node.value, dst_name, op_info),
                node.index,
            )
        if isinstance(node, T.Seq):
            return T.Seq(
                _inline_loads_with_op(node.left, dst_name, op_info),
                _inline_loads_with_op(node.right, dst_name, op_info),
            )
        if isinstance(node, T.Block):
            return T.Block([_inline_loads_with_op(stmt, dst_name, op_info) for stmt in node.stmts])
        if isinstance(node, T.Loop):
            return T.Loop(
                node.start,
                node.end,
                node.tile_name,
                node.loop_var,
                _inline_loads_with_op(node.body, dst_name, op_info),
            )
        if isinstance(node, T.If):
            else_branch = (
                _inline_loads_with_op(node.else_branch, dst_name, op_info) if node.else_branch else None
            )
            return T.If(
                node.cond,
                _inline_loads_with_op(node.then_branch, dst_name, op_info),
                else_branch,
            )
        if isinstance(node, T.Let):
            return T.Let(
                node.tensor,
                _inline_loads_with_op(node.value, dst_name, op_info),
                _inline_loads_with_op(node.body, dst_name, op_info),
            )
        if isinstance(node, (T.Add, T.Sub, T.Mul, T.Div)):
            return node.__class__(
                _inline_loads_with_op(node.left, dst_name, op_info),
                _inline_loads_with_op(node.right, dst_name, op_info),
            )
        if isinstance(node, T.Matmul):
            return T.Matmul(
                _inline_loads_with_op(node.left, dst_name, op_info),
                _inline_loads_with_op(node.right, dst_name, op_info),
            )
        if isinstance(node, T.GenericBinary):
            return T.GenericBinary(
                node.op,
                _inline_loads_with_op(node.left, dst_name, op_info),
                _inline_loads_with_op(node.right, dst_name, op_info),
            )
        if isinstance(node, T.Exp):
            return T.Exp(_inline_loads_with_op(node.val, dst_name, op_info))
        if isinstance(node, T.Sqr):
            return T.Sqr(_inline_loads_with_op(node.val, dst_name, op_info))
        if isinstance(node, T.Sqrt):
            return T.Sqrt(_inline_loads_with_op(node.val, dst_name, op_info))
        if isinstance(node, T.Sigmoid):
            return T.Sigmoid(_inline_loads_with_op(node.val, dst_name, op_info))
        if isinstance(node, T.Cast):
            return T.Cast(node.dtype, _inline_loads_with_op(node.val, dst_name, op_info))
        if isinstance(node, T.Broadcast):
            return T.Broadcast(_inline_loads_with_op(node.val, dst_name, op_info), node.axis)
        if isinstance(node, T.ReduceSum):
            return T.ReduceSum(_inline_loads_with_op(node.val, dst_name, op_info), node.axis)
        if isinstance(node, T.ReduceMax):
            return T.ReduceMax(_inline_loads_with_op(node.val, dst_name, op_info), node.axis)
        if isinstance(node, T.ReduceMin):
            return T.ReduceMin(_inline_loads_with_op(node.val, dst_name, op_info), node.axis)
        if isinstance(node, T.Take):
            return T.Take(
                _inline_loads_with_op(node.data, dst_name, op_info),
                _inline_loads_with_op(node.indices, dst_name, op_info),
                node.axis,
                _inline_loads_with_op(node.index, dst_name, op_info),
            )
        if isinstance(node, T.Concat):
            return T.Concat(
                _inline_loads_with_op(node.a, dst_name, op_info),
                _inline_loads_with_op(node.b, dst_name, op_info),
                node.axis,
            )
        if isinstance(node, T.Permute3):
            return T.Permute3(
                _inline_loads_with_op(node.val, dst_name, op_info),
                node.d0,
                node.d1,
                node.d2,
            )
        if isinstance(node, T.Squeeze):
            return T.Squeeze(_inline_loads_with_op(node.val, dst_name, op_info), node.axis)
        if isinstance(node, T.Unsqueeze):
            return T.Unsqueeze(_inline_loads_with_op(node.val, dst_name, op_info), node.axis)
        if isinstance(node, T.GenericCall):
            return T.GenericCall(
                node.func_name,
                [_inline_loads_with_op(arg, dst_name, op_info) for arg in node.args],
            )
        return node

    def _has_tensor_load(node: T.ASTNode, tensor_name: str) -> bool:
        return tensor_name in _collect_load_tensor_names(node)

    def _matmul_use_info(
        node: T.ASTNode,
        tensor_name: str,
        in_matmul: bool = False,
    ) -> tuple[bool, bool]:
        if isinstance(node, T.Load) and node.tensor.name == tensor_name:
            return True, in_matmul

        next_in_matmul = in_matmul or isinstance(node, T.Matmul)

        if isinstance(node, T.Store):
            return _matmul_use_info(node.value, tensor_name, next_in_matmul)
        if isinstance(node, T.Seq):
            left_found, left_ok = _matmul_use_info(node.left, tensor_name, next_in_matmul)
            right_found, right_ok = _matmul_use_info(node.right, tensor_name, next_in_matmul)
            found = left_found or right_found
            ok = (not left_found or left_ok) and (not right_found or right_ok)
            return found, ok
        if isinstance(node, T.Block):
            found = False
            ok = True
            for stmt in node.stmts:
                stmt_found, stmt_ok = _matmul_use_info(stmt, tensor_name, next_in_matmul)
                found = found or stmt_found
                ok = ok and (not stmt_found or stmt_ok)
            return found, ok
        if isinstance(node, T.Loop):
            return _matmul_use_info(node.body, tensor_name, next_in_matmul)
        if isinstance(node, T.If):
            cond_found, cond_ok = _matmul_use_info(node.cond, tensor_name, next_in_matmul)
            then_found, then_ok = _matmul_use_info(node.then_branch, tensor_name, next_in_matmul)
            else_found, else_ok = (
                _matmul_use_info(node.else_branch, tensor_name, next_in_matmul)
                if node.else_branch
                else (False, True)
            )
            found = cond_found or then_found or else_found
            ok = (
                (not cond_found or cond_ok)
                and (not then_found or then_ok)
                and (not else_found or else_ok)
            )
            return found, ok
        if isinstance(node, T.Let):
            value_found, value_ok = _matmul_use_info(node.value, tensor_name, next_in_matmul)
            body_found, body_ok = _matmul_use_info(node.body, tensor_name, next_in_matmul)
            found = value_found or body_found
            ok = (not value_found or value_ok) and (not body_found or body_ok)
            return found, ok

        child_attrs = (
            "left",
            "right",
            "val",
            "a",
            "b",
            "data",
            "indices",
            "index",
            "tensor",
        )
        found = False
        ok = True
        for attr in child_attrs:
            child = getattr(node, attr, None)
            if isinstance(child, T.ASTNode):
                child_found, child_ok = _matmul_use_info(child, tensor_name, next_in_matmul)
                found = found or child_found
                ok = ok and (not child_found or child_ok)
        if isinstance(node, T.GenericCall):
            for arg in node.args:
                arg_found, arg_ok = _matmul_use_info(arg, tensor_name, next_in_matmul)
                found = found or arg_found
                ok = ok and (not arg_found or arg_ok)
        return found, ok

    analyses = [_analyze_call_pattern(call) for call in calls]

    def _can_inline_all_uses(idx: int) -> bool:
        call = calls[idx]
        if "concat" in call.primfunc.name:
            return False
        if _has_numeric_tile_name(call.primfunc.root_node):
            return False
        store = _extract_single_store(call.primfunc.root_node)
        if store is None:
            return False
        out_name = store.tensor.name
        if out_name != call.out_var_tensor.name:
            return False
        if out_name in output_names:
            return False
        op_info = _extract_inline_op(store)
        if op_info is None:
            return False

        use_found = False
        for j in range(idx + 1, len(calls)):
            call_j = calls[j]
            if not _has_tensor_load(call_j.primfunc.root_node, out_name):
                continue
            use_found = True
            if op_info["op"] in {"permute3", "transpose"}:
                found_in_matmul, only_in_matmul = _matmul_use_info(call_j.primfunc.root_node, out_name)
                if not found_in_matmul or not only_in_matmul:
                    return False
            new_root = _inline_loads_with_op(call_j.primfunc.root_node, out_name, op_info)
            if new_root == call_j.primfunc.root_node:
                return False
        return use_found

    shape_family_indices: dict[str, list[int]] = {}
    for idx, call in enumerate(calls):
        store = _extract_single_store(call.primfunc.root_node)
        if store is None:
            continue
        if _extract_inline_op(store) is None:
            continue
        shape_family_indices.setdefault(analyses[idx].signature, []).append(idx)

    blocked_family_indices: set[int] = set()
    for indices in shape_family_indices.values():
        if len(indices) < 2:
            continue
        if any(not _can_inline_all_uses(idx) for idx in indices):
            blocked_family_indices.update(indices)

    i = 0
    while i < len(calls) - 1:
        call = calls[i]
        if i in blocked_family_indices:
            i += 1
            continue
        if "concat" in call.primfunc.name:
            i += 1
            continue
        if _has_numeric_tile_name(call.primfunc.root_node):
            if call.primfunc.name == "transpose":
                print(f"[inline_shape_op_calls] skip {call.primfunc.name}: numeric_tile_name")
            i += 1
            continue
        store = _extract_single_store(call.primfunc.root_node)
        if store is None:
            i += 1
            continue
        out_name = store.tensor.name
        if out_name != call.out_var_tensor.name:
            i += 1
            continue
        if out_name in output_names:
            i += 1
            continue
        op_info = _extract_inline_op(store)
        if op_info is None:
            i += 1
            continue

        total_uses = any(_has_tensor_load(c.primfunc.root_node, out_name) for c in calls[i + 1 :])
        if not total_uses:
            i += 1
            continue

        replaced_any = False
        for j in range(i + 1, len(calls)):
            call_j = calls[j]
            uses = _has_tensor_load(call_j.primfunc.root_node, out_name)
            if not uses:
                continue
            if op_info["op"] in {"permute3", "transpose"}:
                found_in_matmul, only_in_matmul = _matmul_use_info(call_j.primfunc.root_node, out_name)
                if not found_in_matmul or not only_in_matmul:
                    continue
            new_root = _inline_loads_with_op(call_j.primfunc.root_node, out_name, op_info)
            if new_root == call_j.primfunc.root_node:
                continue
            replaced_any = True

            needed_inputs = {op_info["src"]}
            merged_inputs = _merge_input_tensors(
                call_j.input_tensors,
                needed_inputs,
                tensor_info_map,
            )
            merged_primfunc_inputs = _merge_input_tensors(
                call_j.primfunc.input_tensors,
                needed_inputs,
                tensor_info_map,
            )
            used_names = _collect_load_tensor_names(new_root)
            merged_inputs = [t for t in merged_inputs if t.name in used_names]
            merged_primfunc_inputs = [t for t in merged_primfunc_inputs if t.name in used_names]
            new_primfunc = dataclasses.replace(call_j.primfunc, root_node=new_root)
            calls[j] = T.PrimFuncCall(
                primfunc=dataclasses.replace(new_primfunc, input_tensors=merged_primfunc_inputs),
                out_var_tensor=call_j.out_var_tensor,
                input_tensors=merged_inputs,
                call_index=call_j.call_index,
            )

        if not replaced_any:
            i += 1
            continue

        remaining_uses = _count_loads_in_calls(calls[i + 1 :], out_name)
        if remaining_uses == 0:
            calls.pop(i)
            intermediate_tensors = [t for t in intermediate_tensors if t.name != out_name]
            continue

        i += 1
        continue

    return T.MainFunc(
        calls=calls,
        input_tensors=main_func.input_tensors,
        output_tensors=main_func.output_tensors,
        intermediate_tensors=intermediate_tensors,
    )


def inline_elementwise_op_calls(main_func: T.MainFunc) -> T.MainFunc:
    """
    Inline elementwise producer calls (add/sub/mul/div/sigmoid/etc.)
    into all consumers when indices match and output is used only via loads.

    Exp producers are intentionally excluded to avoid duplicating expensive
    elementwise work across multiple downstream consumers.
    """
    calls = list(main_func.calls)
    if len(calls) < 2:
        return main_func

    output_names = {t.name for t in main_func.output_tensors}
    intermediate_tensors = list(main_func.intermediate_tensors)
    tensor_info_map = _build_tensor_info_map(main_func)

    def _extract_elementwise_op(store: T.Store) -> dict[str, object] | None:
        if not isinstance(store.index, T.Index):
            return None
        store_index = store.index
        needed_inputs: set[str] = set()

        def contains_exp(expr: T.ASTNode) -> bool:
            if isinstance(expr, T.Exp):
                return True
            if isinstance(expr, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min, T.Matmul, T.GenericBinary)):
                return contains_exp(expr.left) or contains_exp(expr.right)
            if isinstance(expr, (T.Sqr, T.Sqrt, T.Sigmoid, T.Cast, T.Broadcast, T.ReduceSum, T.ReduceMax, T.ReduceMin, T.Squeeze, T.Unsqueeze, T.Permute3)):
                return contains_exp(expr.val)
            if isinstance(expr, T.Take):
                return (
                    contains_exp(expr.data)
                    or contains_exp(expr.indices)
                    or contains_exp(expr.index)
                )
            if isinstance(expr, T.Concat):
                return contains_exp(expr.a) or contains_exp(expr.b)
            if isinstance(expr, T.GenericCall):
                return any(contains_exp(arg) for arg in expr.args)
            if isinstance(expr, T.Load):
                return False
            if isinstance(expr, T.Const):
                return False
            return False

        if contains_exp(store.value):
            return None

        def visit(expr: T.ASTNode):
            if isinstance(expr, T.Load):
                if expr.index != store_index:
                    return None
                needed_inputs.add(expr.tensor.name)
                return ("load", expr.tensor.name)
            if isinstance(expr, T.Const):
                return ("const", expr.value)
            if isinstance(expr, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min, T.GenericBinary)):
                left = visit(expr.left)
                right = visit(expr.right)
                if left is None or right is None:
                    return None
                return ("binary", type(expr), getattr(expr, "op", None), left, right)
            if isinstance(expr, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid, T.Cast)):
                inner = visit(expr.val)
                if inner is None:
                    return None
                return ("unary", type(expr), expr.dtype if isinstance(expr, T.Cast) else None, inner)
            if isinstance(expr, T.GenericCall):
                if len(expr.args) != 1:
                    return None
                if expr.func_name not in ("erf", "abs"):
                    return None
                inner = visit(expr.args[0])
                if inner is None:
                    return None
                return ("generic_unary", expr.func_name, inner)
            return None

        template = visit(store.value)
        if template is None:
            return None
        return {"template": template, "needed_inputs": needed_inputs}

    def _build_expr(template: tuple, index: T.Index) -> T.ASTNode | None:
        tag = template[0]
        if tag == "load":
            return T.Load(T.Tensor(template[1]), index)
        if tag == "const":
            return T.Const(template[1])
        if tag == "binary":
            _, cls, op, left, right = template
            left_expr = _build_expr(left, index)
            right_expr = _build_expr(right, index)
            if left_expr is None or right_expr is None:
                return None
            if cls is T.GenericBinary:
                return T.GenericBinary(op, left_expr, right_expr)
            return cls(left_expr, right_expr)
        if tag == "unary":
            _, cls, dtype, inner = template
            inner_expr = _build_expr(inner, index)
            if inner_expr is None:
                return None
            if cls is T.Cast:
                return T.Cast(dtype, inner_expr)
            return cls(inner_expr)
        if tag == "generic_unary":
            _, func_name, inner = template
            inner_expr = _build_expr(inner, index)
            if inner_expr is None:
                return None
            return T.GenericCall(func_name, [inner_expr])
        return None

    def _inline_loads_with_elementwise(node: T.ASTNode, dst_name: str, op_info: dict[str, object]) -> T.ASTNode:
        if isinstance(node, T.Load) and node.tensor.name == dst_name:
            if isinstance(node.index, T.Index):
                expr = _build_expr(op_info["template"], node.index)
                if expr is not None:
                    return expr
            return node
        if isinstance(node, T.Store):
            return T.Store(
                node.tensor,
                _inline_loads_with_elementwise(node.value, dst_name, op_info),
                node.index,
            )
        if isinstance(node, T.Seq):
            return T.Seq(
                _inline_loads_with_elementwise(node.left, dst_name, op_info),
                _inline_loads_with_elementwise(node.right, dst_name, op_info),
            )
        if isinstance(node, T.Block):
            return T.Block([_inline_loads_with_elementwise(stmt, dst_name, op_info) for stmt in node.stmts])
        if isinstance(node, T.Loop):
            return T.Loop(
                node.start,
                node.end,
                node.tile_name,
                node.loop_var,
                _inline_loads_with_elementwise(node.body, dst_name, op_info),
            )
        if isinstance(node, T.If):
            else_branch = (
                _inline_loads_with_elementwise(node.else_branch, dst_name, op_info) if node.else_branch else None
            )
            return T.If(
                node.cond,
                _inline_loads_with_elementwise(node.then_branch, dst_name, op_info),
                else_branch,
            )
        if isinstance(node, T.Let):
            return T.Let(
                node.tensor,
                _inline_loads_with_elementwise(node.value, dst_name, op_info),
                _inline_loads_with_elementwise(node.body, dst_name, op_info),
            )
        if isinstance(node, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min, T.Matmul, T.GenericBinary)):
            return node.__class__(
                _inline_loads_with_elementwise(node.left, dst_name, op_info),
                _inline_loads_with_elementwise(node.right, dst_name, op_info),
            )
        if isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
            return node.__class__(_inline_loads_with_elementwise(node.val, dst_name, op_info))
        if isinstance(node, T.Cast):
            return T.Cast(node.dtype, _inline_loads_with_elementwise(node.val, dst_name, op_info))
        if isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin, T.Broadcast)):
            return node.__class__(_inline_loads_with_elementwise(node.val, dst_name, op_info), node.axis)
        if isinstance(node, T.Take):
            return T.Take(
                _inline_loads_with_elementwise(node.data, dst_name, op_info),
                _inline_loads_with_elementwise(node.indices, dst_name, op_info),
                node.axis,
                _inline_loads_with_elementwise(node.index, dst_name, op_info),
            )
        if isinstance(node, T.Concat):
            return T.Concat(
                _inline_loads_with_elementwise(node.a, dst_name, op_info),
                _inline_loads_with_elementwise(node.b, dst_name, op_info),
                node.axis,
            )
        if isinstance(node, T.Permute3):
            return T.Permute3(
                _inline_loads_with_elementwise(node.val, dst_name, op_info),
                node.d0,
                node.d1,
                node.d2,
            )
        if isinstance(node, T.Squeeze):
            return T.Squeeze(_inline_loads_with_elementwise(node.val, dst_name, op_info), node.axis)
        if isinstance(node, T.Unsqueeze):
            return T.Unsqueeze(_inline_loads_with_elementwise(node.val, dst_name, op_info), node.axis)
        if isinstance(node, T.GenericCall):
            return T.GenericCall(
                node.func_name,
                [_inline_loads_with_elementwise(arg, dst_name, op_info) for arg in node.args],
            )
        return node

    i = 0
    while i < len(calls) - 1:
        call = calls[i]
        store = _extract_single_store(call.primfunc.root_node)
        if store is None:
            i += 1
            continue
        out_name = store.tensor.name
        if out_name != call.out_var_tensor.name:
            i += 1
            continue
        if out_name in output_names:
            i += 1
            continue
        op_info = _extract_elementwise_op(store)
        if op_info is None:
            i += 1
            continue

        total_uses = any(out_name in _collect_load_tensor_names(c.primfunc.root_node) for c in calls[i + 1 :])
        if not total_uses:
            i += 1
            continue

        replaced_any = False
        for j in range(i + 1, len(calls)):
            call_j = calls[j]
            if out_name not in _collect_load_tensor_names(call_j.primfunc.root_node):
                continue
            new_root = _inline_loads_with_elementwise(call_j.primfunc.root_node, out_name, op_info)
            if new_root == call_j.primfunc.root_node:
                continue
            replaced_any = True

            needed_inputs = set(op_info["needed_inputs"])
            merged_inputs = _merge_input_tensors(
                call_j.input_tensors,
                needed_inputs,
                tensor_info_map,
            )
            merged_primfunc_inputs = _merge_input_tensors(
                call_j.primfunc.input_tensors,
                needed_inputs,
                tensor_info_map,
            )
            used_names = _collect_load_tensor_names(new_root)
            merged_inputs = [t for t in merged_inputs if t.name in used_names]
            merged_primfunc_inputs = [t for t in merged_primfunc_inputs if t.name in used_names]
            new_primfunc = dataclasses.replace(call_j.primfunc, root_node=new_root)
            calls[j] = T.PrimFuncCall(
                primfunc=dataclasses.replace(new_primfunc, input_tensors=merged_primfunc_inputs),
                out_var_tensor=call_j.out_var_tensor,
                input_tensors=merged_inputs,
                call_index=call_j.call_index,
            )

        if replaced_any:
            remaining_uses = any(out_name in _collect_load_tensor_names(c.primfunc.root_node) for c in calls[i + 1 :])
            if not remaining_uses:
                calls.pop(i)
                intermediate_tensors = [t for t in intermediate_tensors if t.name != out_name]
                continue
        i += 1

    return T.MainFunc(
        calls=calls,
        input_tensors=main_func.input_tensors,
        output_tensors=main_func.output_tensors,
        intermediate_tensors=intermediate_tensors,
    )


def eliminate_dead_seq_stores(main_func: T.MainFunc) -> T.MainFunc:
    """
    Remove stores whose destination is never used later inside the same primfunc and
    is not externally visible as a call output or an input-side-effect write.
    """
    updated_calls: list[T.PrimFuncCall] = []
    all_used_tensors = {tensor.name for tensor in main_func.input_tensors}
    all_used_tensors.update(tensor.name for tensor in main_func.output_tensors)

    for call in main_func.calls:
        preserved_writes = {call.out_var_tensor.name}
        preserved_writes.update(tensor.name for tensor in call.input_tensors)
        new_root = _eliminate_dead_stores_in_node(call.primfunc.root_node, preserved_writes)

        load_names = _collect_load_tensor_names(new_root)
        store_names = _collect_store_tensor_names(new_root)
        needed_input_names = load_names | (store_names & {tensor.name for tensor in call.input_tensors})

        updated_call = T.PrimFuncCall(
            primfunc=dataclasses.replace(
                call.primfunc,
                root_node=new_root,
                input_tensors=[tensor for tensor in call.primfunc.input_tensors if tensor.name in needed_input_names],
                allocated_tensors=[tensor for tensor in call.primfunc.allocated_tensors if tensor.name in store_names],
            ),
            out_var_tensor=call.out_var_tensor,
            input_tensors=[tensor for tensor in call.input_tensors if tensor.name in needed_input_names],
            call_index=call.call_index,
        )
        updated_calls.append(updated_call)
        all_used_tensors.update(load_names)
        all_used_tensors.update(store_names)
        all_used_tensors.add(call.out_var_tensor.name)

    return T.MainFunc(
        calls=updated_calls,
        input_tensors=main_func.input_tensors,
        output_tensors=main_func.output_tensors,
        intermediate_tensors=[
            tensor for tensor in main_func.intermediate_tensors if tensor.name in all_used_tensors
        ],
    )


def eliminate_dead_calls(main_func: T.MainFunc) -> T.MainFunc:
    """
    Remove calls whose outputs do not contribute to final outputs and that do not
    write back to observable input buffers.
    """
    live_tensors = {tensor.name for tensor in main_func.output_tensors}
    input_names = {tensor.name for tensor in main_func.input_tensors}
    kept_rev: list[T.PrimFuncCall] = []

    for call in reversed(main_func.calls):
        root = call.primfunc.root_node
        out_name = call.out_var_tensor.name
        store_names = _collect_store_tensor_names(root)
        load_names = _collect_load_tensor_names(root)
        writes_input = bool(store_names & input_names)
        is_live = out_name in live_tensors or writes_input

        if not is_live:
            if _IR_DEBUG:
                print(
                    f"[dce] drop call {call.call_index} {call.primfunc.name}: "
                    f"out={out_name}, reads={sorted(load_names)}, writes={sorted(store_names)}"
                )
            continue

        kept_rev.append(call)
        live_tensors.discard(out_name)
        live_tensors.update(load_names)
        live_tensors.update(store_names & input_names)

    kept_calls = list(reversed(kept_rev))
    kept_tensor_names = set(live_tensors)
    kept_tensor_names.update(t.name for t in main_func.output_tensors)
    kept_tensor_names.update(t.name for t in main_func.input_tensors)
    for call in kept_calls:
        kept_tensor_names.add(call.out_var_tensor.name)
        kept_tensor_names.update(_collect_store_tensor_names(call.primfunc.root_node))
        kept_tensor_names.update(_collect_load_tensor_names(call.primfunc.root_node))

    return T.MainFunc(
        calls=kept_calls,
        input_tensors=main_func.input_tensors,
        output_tensors=main_func.output_tensors,
        intermediate_tensors=[
            tensor for tensor in main_func.intermediate_tensors if tensor.name in kept_tensor_names
        ],
    )


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


def _eliminate_dead_stores_in_node(node: T.ASTNode, preserved_writes: set[str]) -> T.ASTNode:
    pruned, _ = _eliminate_dead_stores_in_stmt(node, set(), preserved_writes)
    if pruned is None:
        return T.Block([])
    return pruned


def _eliminate_dead_stores_in_stmt(
    node: T.ASTNode,
    live_after: set[str],
    preserved_writes: set[str],
) -> tuple[T.ASTNode | None, set[str]]:
    if isinstance(node, T.Store):
        target = node.tensor.name
        if target not in live_after and target not in preserved_writes:
            return None, set(live_after)
        live_before = set(live_after)
        live_before.discard(target)
        live_before.update(_collect_load_tensor_names(node.value))
        return node, live_before

    if isinstance(node, T.Seq):
        right_node, right_live = _eliminate_dead_stores_in_stmt(node.right, live_after, preserved_writes)
        left_node, left_live = _eliminate_dead_stores_in_stmt(node.left, right_live, preserved_writes)
        if left_node is None:
            return right_node, left_live
        if right_node is None:
            return left_node, left_live
        return T.Seq(left_node, right_node), left_live

    if isinstance(node, T.Block):
        live_now = set(live_after)
        kept: list[T.ASTNode] = []
        for stmt in reversed(node.stmts):
            new_stmt, live_now = _eliminate_dead_stores_in_stmt(stmt, live_now, preserved_writes)
            if new_stmt is not None:
                kept.append(new_stmt)
        kept.reverse()
        if not kept:
            return None, live_now
        if len(kept) == 1:
            return kept[0], live_now
        return T.Block(kept), live_now

    if isinstance(node, T.Loop):
        new_body, live_before = _eliminate_dead_stores_in_stmt(node.body, live_after, preserved_writes)
        if new_body is None:
            return None, set(live_after)
        return T.Loop(node.start, node.end, node.tile_name, node.loop_var, new_body), live_before

    if isinstance(node, T.If):
        then_node, then_live = _eliminate_dead_stores_in_stmt(node.then_branch, live_after, preserved_writes)
        else_node = None
        else_live = set(live_after)
        if node.else_branch is not None:
            else_node, else_live = _eliminate_dead_stores_in_stmt(node.else_branch, live_after, preserved_writes)
        if then_node is None and else_node is None:
            return None, set(live_after)
        live_before = set(then_live) | set(else_live)
        live_before.update(_collect_load_tensor_names(node.cond))
        return T.If(node.cond, then_node or T.Block([]), else_node), live_before

    if isinstance(node, T.Let):
        new_body, live_before = _eliminate_dead_stores_in_stmt(node.body, live_after, preserved_writes)
        if new_body is None:
            return None, set(live_after)
        live_before = set(live_before)
        live_before.update(_collect_load_tensor_names(node.value))
        return T.Let(node.tensor, node.value, new_body), live_before

    return node, set(live_after) | _collect_load_tensor_names(node)


def _shape_only_base(node: T.ASTNode) -> T.Load | None:
    if isinstance(node, T.Load):
        return node
    if isinstance(node, (T.Permute3, T.Squeeze, T.Unsqueeze)):
        return _shape_only_base(node.val)
    if isinstance(node, T.GenericCall) and node.func_name in ("transpose",):
        if node.args:
            return _shape_only_base(node.args[0])
    return None


def _collect_load_tensor_names(node: T.ASTNode) -> set[str]:
    names: set[str] = set()
    _collect_load_tensor_names_impl(node, names)
    return names


def _collect_load_tensor_names_impl(node: T.ASTNode, names: set[str]) -> None:
    if isinstance(node, T.Load):
        names.add(node.tensor.name)
        return
    if isinstance(node, T.Store):
        _collect_load_tensor_names_impl(node.value, names)
        return
    if isinstance(node, T.Seq):
        _collect_load_tensor_names_impl(node.left, names)
        _collect_load_tensor_names_impl(node.right, names)
        return
    if isinstance(node, T.Block):
        for stmt in node.stmts:
            _collect_load_tensor_names_impl(stmt, names)
        return
    if isinstance(node, T.Loop):
        _collect_load_tensor_names_impl(node.body, names)
        return
    if isinstance(node, T.If):
        _collect_load_tensor_names_impl(node.then_branch, names)
        if node.else_branch:
            _collect_load_tensor_names_impl(node.else_branch, names)
        return
    if isinstance(node, T.Let):
        _collect_load_tensor_names_impl(node.value, names)
        _collect_load_tensor_names_impl(node.body, names)
        return
    if isinstance(node, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min, T.Matmul)):
        _collect_load_tensor_names_impl(node.left, names)
        _collect_load_tensor_names_impl(node.right, names)
        return
    if isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid, T.Cast)):
        _collect_load_tensor_names_impl(node.val, names)
        return
    if isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin, T.Broadcast)):
        _collect_load_tensor_names_impl(node.val, names)
        return
    if isinstance(node, T.Concat):
        _collect_load_tensor_names_impl(node.a, names)
        _collect_load_tensor_names_impl(node.b, names)
        return
    if isinstance(node, (T.Permute3, T.Squeeze, T.Unsqueeze)):
        _collect_load_tensor_names_impl(node.val, names)
        return
    if isinstance(node, T.GenericBinary):
        _collect_load_tensor_names_impl(node.left, names)
        _collect_load_tensor_names_impl(node.right, names)
        return
    if isinstance(node, T.GenericCall):
        for arg in node.args:
            _collect_load_tensor_names_impl(arg, names)
        return


def _build_tensor_info_map(main_func: T.MainFunc) -> dict[str, T.TensorInfo]:
    tensor_info_map: dict[str, T.TensorInfo] = {}
    for tensor in (
        list(main_func.input_tensors)
        + list(main_func.output_tensors)
        + list(main_func.intermediate_tensors)
    ):
        tensor_info_map[tensor.name] = tensor
    return tensor_info_map


def _merge_input_tensors(
    existing: List[T.TensorInfo],
    needed_names: set[str],
    tensor_info_map: dict[str, T.TensorInfo],
) -> List[T.TensorInfo]:
    merged = list(existing)
    existing_names = {t.name for t in existing}
    for name in sorted(needed_names):
        if name in existing_names:
            continue
        info = tensor_info_map.get(name)
        if info is None:
            continue
        merged.append(info)
        existing_names.add(name)
    return merged


def _count_loads(node: T.ASTNode, tensor_name: str) -> int:
    if isinstance(node, T.Load):
        return 1 if node.tensor.name == tensor_name else 0
    if isinstance(node, T.Store):
        return _count_loads(node.value, tensor_name)
    if isinstance(node, T.Seq):
        return _count_loads(node.left, tensor_name) + _count_loads(node.right, tensor_name)
    if isinstance(node, T.Block):
        return sum(_count_loads(stmt, tensor_name) for stmt in node.stmts)
    if isinstance(node, T.Loop):
        return _count_loads(node.body, tensor_name)
    if isinstance(node, T.If):
        count = _count_loads(node.then_branch, tensor_name)
        if node.else_branch:
            count += _count_loads(node.else_branch, tensor_name)
        return count
    if isinstance(node, T.Let):
        return _count_loads(node.value, tensor_name) + _count_loads(node.body, tensor_name)
    if isinstance(node, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min)):
        return _count_loads(node.left, tensor_name) + _count_loads(node.right, tensor_name)
    if isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid, T.Cast)):
        return _count_loads(node.val, tensor_name)
    if isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin, T.Broadcast)):
        return _count_loads(node.val, tensor_name)
    if isinstance(node, T.Concat):
        return _count_loads(node.a, tensor_name) + _count_loads(node.b, tensor_name)
    if isinstance(node, (T.Permute3, T.Squeeze, T.Unsqueeze)):
        return _count_loads(node.val, tensor_name)
    if isinstance(node, T.GenericBinary):
        return _count_loads(node.left, tensor_name) + _count_loads(node.right, tensor_name)
    if isinstance(node, T.GenericCall):
        return sum(_count_loads(arg, tensor_name) for arg in node.args)
    return 0


def _count_loads_in_calls(calls: List[T.PrimFuncCall], tensor_name: str) -> int:
    return sum(_count_loads(call.primfunc.root_node, tensor_name) for call in calls)


def _collect_store_tensor_names(node: T.ASTNode) -> set[str]:
    names: set[str] = set()

    def visit(cur: T.ASTNode) -> None:
        if isinstance(cur, T.Store):
            names.add(cur.tensor.name)
            visit(cur.value)
            return
        if isinstance(cur, T.Seq):
            visit(cur.left)
            visit(cur.right)
            return
        if isinstance(cur, T.Block):
            for stmt in cur.stmts:
                visit(stmt)
            return
        if isinstance(cur, T.Loop):
            visit(cur.body)
            return
        if isinstance(cur, T.If):
            visit(cur.then_branch)
            if cur.else_branch:
                visit(cur.else_branch)
            return
        if isinstance(cur, T.Let):
            visit(cur.value)
            visit(cur.body)
            return
        if isinstance(cur, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min, T.Matmul)):
            visit(cur.left)
            visit(cur.right)
            return
        if isinstance(cur, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid, T.Cast)):
            visit(cur.val)
            return
        if isinstance(cur, (T.ReduceSum, T.ReduceMax, T.ReduceMin, T.Broadcast)):
            visit(cur.val)
            return
        if isinstance(cur, T.Concat):
            visit(cur.a)
            visit(cur.b)
            return
        if isinstance(cur, (T.Permute3, T.Squeeze, T.Unsqueeze)):
            visit(cur.val)
            return
        if isinstance(cur, T.GenericBinary):
            visit(cur.left)
            visit(cur.right)
            return
        if isinstance(cur, T.GenericCall):
            for arg in cur.args:
                visit(arg)

    visit(node)
    return names


@dataclass
class _CallPattern:
    signature: str
    read_slots: List[str]
    write_slots: List[str]
    reads: set[str]
    writes: set[str]


def _analyze_call_pattern(call: T.PrimFuncCall) -> _CallPattern:
    read_slot_ids: dict[str, int] = {}
    write_slot_ids: dict[str, int] = {}
    read_slots: List[str] = []
    write_slots: List[str] = []

    def get_read_slot(name: str) -> int:
        if name not in read_slot_ids:
            read_slot_ids[name] = len(read_slots)
            read_slots.append(name)
        return read_slot_ids[name]

    def get_write_slot(name: str) -> int:
        if name not in write_slot_ids:
            write_slot_ids[name] = len(write_slots)
            write_slots.append(name)
        return write_slot_ids[name]

    def sig_index(index: T.Index) -> str:
        return "(index " + " ".join(sig(idx) for idx in index.indices) + ")"

    def sig(node: T.ASTNode) -> str:
        if isinstance(node, T.Store):
            slot = get_write_slot(node.tensor.name)
            return f"(store W{slot} {sig(node.value)} {sig(node.index)})"
        if isinstance(node, T.Load):
            slot = get_read_slot(node.tensor.name)
            return f"(load R{slot} {sig(node.index)})"
        if isinstance(node, T.Loop):
            return f"(loop {sig(node.start)} {sig(node.end)} {node.tile_name} {node.loop_var} {sig(node.body)})"
        if isinstance(node, T.Seq):
            return f"(seq {sig(node.left)} {sig(node.right)})"
        if isinstance(node, T.Block):
            return "(block " + " ".join(sig(stmt) for stmt in node.stmts) + ")"
        if isinstance(node, T.Index):
            return sig_index(node)
        if isinstance(node, T.Tile):
            return f"(tile {node.name})"
        if isinstance(node, T.TileOffset):
            return f"(shifted_tile {node.name} {node.offset})"
        if isinstance(node, T.ConstTile):
            return f"(const_tile {node.start_index} {node.interval})"
        if isinstance(node, T.FullTile):
            return "fulltile"
        if isinstance(node, T.Elem):
            return f"(elem {node.name})"
        if isinstance(node, T.VarRef):
            return node.name
        if isinstance(node, T.Const):
            return str(node.value)
        if isinstance(node, T.Add):
            return f"(+ {sig(node.left)} {sig(node.right)})"
        if isinstance(node, T.Sub):
            return f"(- {sig(node.left)} {sig(node.right)})"
        if isinstance(node, T.Mul):
            return f"(* {sig(node.left)} {sig(node.right)})"
        if isinstance(node, T.Div):
            return f"(/ {sig(node.left)} {sig(node.right)})"
        if isinstance(node, T.Max):
            return f"(max {sig(node.left)} {sig(node.right)})"
        if isinstance(node, T.Min):
            return f"(min {sig(node.left)} {sig(node.right)})"
        if isinstance(node, T.Matmul):
            return f"(@ {sig(node.left)} {sig(node.right)})"
        if isinstance(node, T.Exp):
            return f"(exp {sig(node.val)})"
        if isinstance(node, T.Sqr):
            return f"(sqr {sig(node.val)})"
        if isinstance(node, T.Sqrt):
            return f"(sqrt {sig(node.val)})"
        if isinstance(node, T.Sigmoid):
            return f"(sigmoid {sig(node.val)})"
        if isinstance(node, T.Cast):
            return f"(cast {node.dtype} {sig(node.val)})"
        if isinstance(node, T.ReduceSum):
            return f"(rsum {sig(node.val)} {node.axis})"
        if isinstance(node, T.ReduceMax):
            return f"(rmax {sig(node.val)} {node.axis})"
        if isinstance(node, T.ReduceMin):
            return f"(rmin {sig(node.val)} {node.axis})"
        if isinstance(node, T.Broadcast):
            return f"(bcast {sig(node.val)} {node.axis})"
        if isinstance(node, T.Concat):
            return f"(concat {sig(node.a)} {sig(node.b)} {node.axis})"
        if isinstance(node, T.Permute3):
            return f"(permute3 {sig(node.val)} {node.d0} {node.d1} {node.d2})"
        if isinstance(node, T.Squeeze):
            return f"(squeeze {sig(node.val)} {node.axis})"
        if isinstance(node, T.Unsqueeze):
            return f"(unsqueeze {sig(node.val)} {node.axis})"
        if isinstance(node, T.GenericBinary):
            return f"({node.op} {sig(node.left)} {sig(node.right)})"
        if isinstance(node, T.GenericCall):
            return f"({node.func_name} {' '.join(sig(arg) for arg in node.args)})"
        if isinstance(node, T.Take):
            return f"(take {sig(node.data)} {sig(node.indices)} {node.axis} {sig(node.index)})"
        if isinstance(node, T.If):
            else_part = f" {sig(node.else_branch)}" if node.else_branch else ""
            return f"(if {sig(node.cond)} {sig(node.then_branch)}{else_part})"
        if isinstance(node, T.Let):
            return f"(let {sig(node.tensor)} {sig(node.value)} {sig(node.body)})"
        if isinstance(node, T.Arange):
            return f"(arange {node.axis})"
        if isinstance(node, T.Tensor):
            return "(tensor)"
        return node.__class__.__name__

    root = call.primfunc.root_node
    return _CallPattern(
        signature=sig(root),
        read_slots=read_slots,
        write_slots=write_slots,
        reads=_collect_load_tensor_names(root),
        writes=_collect_store_tensor_names(root),
    )


def _calls_conflict(lhs: _CallPattern, rhs: _CallPattern) -> bool:
    return bool(
        (lhs.writes & rhs.reads)
        or (rhs.writes & lhs.reads)
        or (lhs.writes & rhs.writes)
    )


def _rw_tensors(pattern: _CallPattern) -> set[str]:
    return pattern.reads | pattern.writes


def _has_intervening_dependency_on_candidate(
    analyses: List[_CallPattern],
    left_idx: int,
    right_idx: int,
) -> bool:
    if left_idx > right_idx:
        left_idx, right_idx = right_idx, left_idx
    candidate = analyses[right_idx]
    candidate_reads = candidate.reads
    candidate_writes = candidate.writes
    for mid_idx in range(left_idx + 1, right_idx):
        mid = analyses[mid_idx]
        if mid.writes & (candidate_reads | candidate_writes):
            return True
        if mid.reads & candidate_writes:
            return True
    return False


def plan_fusion_groups(main_func: T.MainFunc) -> List[FusionGroup]:
    calls = list(main_func.calls)
    if len(calls) < 2:
        return []

    analyses = [_analyze_call_pattern(call) for call in calls]
    groups: List[FusionGroup] = []
    covered: set[int] = set()
    i = 0
    while i < len(calls):
        if i in covered:
            i += 1
            continue
        base = analyses[i]
        run = [i]
        exact_slots = set(range(len(base.read_slots)))
        j = i + 1
        while j < len(calls):
            if j in covered:
                j += 1
                continue
            cur = analyses[j]
            if cur.signature != base.signature:
                if calls[i].primfunc.name == calls[j].primfunc.name:
                    if _IR_DEBUG:
                        print(
                            f"[fusion] skip pair ({i},{j}) {calls[i].primfunc.name}: "
                            "signature_mismatch"
                        )
                j += 1
                continue
            if any(_calls_conflict(analyses[k], cur) for k in run):
                if _IR_DEBUG:
                    print(
                        f"[fusion] skip pair ({run[-1]},{j}) {calls[j].primfunc.name}: "
                        f"conflict reads={sorted(cur.reads)} writes={sorted(cur.writes)}"
                    )
                j += 1
                continue
            if any(_has_intervening_dependency_on_candidate(analyses, k, j) for k in run):
                if _IR_DEBUG:
                    print(
                        f"[fusion] skip pair ({run[-1]},{j}) {calls[j].primfunc.name}: "
                        "intervening_dependency"
                    )
                j += 1
                continue
            next_exact = {
                slot
                for slot in exact_slots
                if slot < len(cur.read_slots) and cur.read_slots[slot] == base.read_slots[slot]
            }
            run.append(j)
            exact_slots = next_exact
            j += 1

        if len(run) >= 2:
            exact_read_slots = {slot: base.read_slots[slot] for slot in sorted(exact_slots)}
            varying_read_slots = {
                slot: [analyses[idx].read_slots[slot] for idx in run]
                for slot in range(len(base.read_slots))
                if slot not in exact_slots
            }
            write_slots = {
                slot: [analyses[idx].write_slots[slot] for idx in run]
                for slot in range(len(base.write_slots))
            }
            groups.append(
                FusionGroup(
                    call_indices=run,
                    signature=base.signature,
                    exact_read_slots=exact_read_slots,
                    varying_read_slots=varying_read_slots,
                    write_slots=write_slots,
                )
            )
            covered.update(run)
            i += 1
            continue
        i += 1

    return groups


def validate_fusion_groups(main_func: T.MainFunc, groups: List[FusionGroup]) -> List[str]:
    calls = list(main_func.calls)
    analyses = [_analyze_call_pattern(call) for call in calls]
    errors: List[str] = []
    covered: set[int] = set()

    for group in groups:
        indices = group.call_indices
        if len(indices) < 2:
            errors.append("fusion group must contain at least two calls")
            continue
        if any(idx < 0 or idx >= len(calls) for idx in indices):
            errors.append(f"fusion group index out of range: {indices}")
            continue
        if covered & set(indices):
            errors.append(f"fusion group overlaps another group: {indices}")
        covered.update(indices)

        base = analyses[indices[0]]
        if base.signature != group.signature:
            errors.append(f"fusion group signature mismatch at {indices[0]}")
        for idx in indices[1:]:
            cur = analyses[idx]
            if cur.signature != base.signature:
                errors.append(f"fusion group calls have different skeletons: {indices}")
                break
            if _calls_conflict(base, cur):
                errors.append(f"fusion group calls conflict: {indices}")
                break
            if _has_intervening_dependency_on_candidate(analyses, indices[0], idx):
                errors.append(f"fusion group has intervening tensor overlap: {indices}")
                break

        for pos, left_idx in enumerate(indices):
            for right_idx in indices[pos + 1 :]:
                if _calls_conflict(analyses[left_idx], analyses[right_idx]):
                    errors.append(f"fusion group calls conflict: {indices}")
                    break
                if _has_intervening_dependency_on_candidate(analyses, left_idx, right_idx):
                    errors.append(f"fusion group has intervening tensor overlap: {indices}")
                    break

        for slot, exact_name in group.exact_read_slots.items():
            for idx in indices:
                cur = analyses[idx]
                if slot >= len(cur.read_slots) or cur.read_slots[slot] != exact_name:
                    errors.append(f"fusion group exact input mismatch at slot {slot}: {indices}")
                    break

    return errors


def _inline_loads(
    node: T.ASTNode,
    tensor_name: str,
    index: T.Index,
    value: T.ASTNode,
) -> T.ASTNode:
    if isinstance(node, T.Load):
        if node.tensor.name == tensor_name and node.index == index:
            return copy.deepcopy(value)
        return node
    if isinstance(node, T.Store):
        return T.Store(node.tensor, _inline_loads(node.value, tensor_name, index, value), node.index)
    if isinstance(node, T.Seq):
        return T.Seq(
            _inline_loads(node.left, tensor_name, index, value),
            _inline_loads(node.right, tensor_name, index, value),
        )
    if isinstance(node, T.Block):
        return T.Block([_inline_loads(stmt, tensor_name, index, value) for stmt in node.stmts])
    if isinstance(node, T.Loop):
        return T.Loop(node.start, node.end, node.tile_name, node.loop_var, _inline_loads(node.body, tensor_name, index, value))
    if isinstance(node, T.If):
        else_branch = _inline_loads(node.else_branch, tensor_name, index, value) if node.else_branch else None
        return T.If(node.cond, _inline_loads(node.then_branch, tensor_name, index, value), else_branch)
    if isinstance(node, T.Let):
        return T.Let(
            node.tensor,
            _inline_loads(node.value, tensor_name, index, value),
            _inline_loads(node.body, tensor_name, index, value),
        )
    if isinstance(node, T.Add):
        return T.Add(
            _inline_loads(node.left, tensor_name, index, value),
            _inline_loads(node.right, tensor_name, index, value),
        )
    if isinstance(node, T.Sub):
        return T.Sub(
            _inline_loads(node.left, tensor_name, index, value),
            _inline_loads(node.right, tensor_name, index, value),
        )
    if isinstance(node, T.Mul):
        return T.Mul(
            _inline_loads(node.left, tensor_name, index, value),
            _inline_loads(node.right, tensor_name, index, value),
        )
    if isinstance(node, T.Div):
        return T.Div(
            _inline_loads(node.left, tensor_name, index, value),
            _inline_loads(node.right, tensor_name, index, value),
        )
    if isinstance(node, T.Max):
        return T.Max(
            _inline_loads(node.left, tensor_name, index, value),
            _inline_loads(node.right, tensor_name, index, value),
        )
    if isinstance(node, T.Min):
        return T.Min(
            _inline_loads(node.left, tensor_name, index, value),
            _inline_loads(node.right, tensor_name, index, value),
        )
    if isinstance(node, T.Exp):
        return T.Exp(_inline_loads(node.val, tensor_name, index, value))
    if isinstance(node, T.Sqr):
        return T.Sqr(_inline_loads(node.val, tensor_name, index, value))
    if isinstance(node, T.Sqrt):
        return T.Sqrt(_inline_loads(node.val, tensor_name, index, value))
    if isinstance(node, T.Sigmoid):
        return T.Sigmoid(_inline_loads(node.val, tensor_name, index, value))
    if isinstance(node, T.Cast):
        return T.Cast(node.dtype, _inline_loads(node.val, tensor_name, index, value))
    if isinstance(node, T.ReduceSum):
        return T.ReduceSum(_inline_loads(node.val, tensor_name, index, value), node.axis)
    if isinstance(node, T.ReduceMax):
        return T.ReduceMax(_inline_loads(node.val, tensor_name, index, value), node.axis)
    if isinstance(node, T.ReduceMin):
        return T.ReduceMin(_inline_loads(node.val, tensor_name, index, value), node.axis)
    if isinstance(node, T.Broadcast):
        return T.Broadcast(_inline_loads(node.val, tensor_name, index, value), node.axis)
    if isinstance(node, T.Concat):
        return T.Concat(
            _inline_loads(node.a, tensor_name, index, value),
            _inline_loads(node.b, tensor_name, index, value),
            node.axis,
        )
    if isinstance(node, T.Permute3):
        return T.Permute3(
            _inline_loads(node.val, tensor_name, index, value),
            node.d0,
            node.d1,
            node.d2,
        )
    if isinstance(node, T.Squeeze):
        return T.Squeeze(_inline_loads(node.val, tensor_name, index, value), node.axis)
    if isinstance(node, T.Unsqueeze):
        return T.Unsqueeze(_inline_loads(node.val, tensor_name, index, value), node.axis)
    if isinstance(node, T.GenericBinary):
        return T.GenericBinary(
            node.op,
            _inline_loads(node.left, tensor_name, index, value),
            _inline_loads(node.right, tensor_name, index, value),
        )
    if isinstance(node, T.GenericCall):
        return T.GenericCall(
            node.func_name,
            [_inline_loads(arg, tensor_name, index, value) for arg in node.args],
        )
    return node


def decompose_operations(
    root: T.ASTNode,
    tensor_info_map: Dict[str, T.TensorInfo],
    ratio: float = 1.0,
) -> Tuple[T.ASTNode, List[T.TensorInfo]]:
    """
    지정된 연산들을 tmp tensor store/load로 분해합니다.
    """
    ratio = max(0.0, min(1.0, ratio))
    tmp_tensors: List[T.TensorInfo] = []
    used_names = set(tensor_info_map.keys())
    counter = {"value": 0}

    decompose_types = (
        T.Add,
        T.Sub,
        T.Mul,
        T.Div,
        T.Exp,
        T.Sqr,
        T.Sqrt,
        T.Sigmoid,
        T.Matmul,
    )

    def _fresh_tmp(dest_info: T.TensorInfo) -> T.TensorInfo:
        while True:
            counter["value"] += 1
            name = f"tmp_{counter['value']}"
            if name not in used_names:
                used_names.add(name)
                info = T.TensorInfo(name=name, shape=list(dest_info.shape), dtype=dest_info.dtype)
                tmp_tensors.append(info)
                tensor_info_map[name] = info
                return info

    def _is_tmp_candidate(expr: T.ASTNode) -> bool:
        return isinstance(expr, decompose_types)

    def _list_to_block(stmts: List[T.ASTNode]) -> T.ASTNode:
        if not stmts:
            return T.Block([])
        if len(stmts) == 1:
            return stmts[0]
        return T.Block(stmts)

    def _index_drop_axis(index: T.ASTNode, axis: int) -> T.ASTNode:
        if not isinstance(index, T.Index):
            return index
        idxs = list(index.indices)
        if axis < 0:
            axis += len(idxs)
        if 0 <= axis < len(idxs):
            idxs.pop(axis)
        return T.Index(idxs)

    def _index_insert_fulltile(index: T.ASTNode, axis: int) -> T.ASTNode:
        if not isinstance(index, T.Index):
            return index
        idxs = list(index.indices)
        if axis < 0:
            axis += len(idxs) + 1
        if axis < 0:
            axis = 0
        if axis > len(idxs):
            axis = len(idxs)
        idxs.insert(axis, T.FullTile())
        return T.Index(idxs)

    def _index_permute_inverse(index: T.ASTNode, d0: int, d1: int, d2: int) -> T.ASTNode:
        if not isinstance(index, T.Index):
            return index
        idxs = list(index.indices)
        if len(idxs) != 3:
            return index
        new_idxs: List[T.ASTNode | None] = [None, None, None]
        if 0 <= d0 < 3:
            new_idxs[d0] = idxs[0]
        if 0 <= d1 < 3:
            new_idxs[d1] = idxs[1]
        if 0 <= d2 < 3:
            new_idxs[d2] = idxs[2]
        if any(v is None for v in new_idxs):
            return index
        return T.Index([v for v in new_idxs if v is not None])

    def _index_permute_inverse_2d(index: T.ASTNode, d0: int, d1: int) -> T.ASTNode:
        if not isinstance(index, T.Index):
            return index
        idxs = list(index.indices)
        if len(idxs) != 2:
            return index
        new_idxs: List[T.ASTNode | None] = [None, None]
        if 0 <= d0 < 2:
            new_idxs[d0] = idxs[0]
        if 0 <= d1 < 2:
            new_idxs[d1] = idxs[1]
        if any(v is None for v in new_idxs):
            return index
        return T.Index([v for v in new_idxs if v is not None])

    def _get_const_int(node: T.ASTNode) -> Optional[int]:
        if isinstance(node, T.Const) and isinstance(node.value, int):
            return node.value
        return None

    def _count_candidates_expr(expr: T.ASTNode, allow_tmp: bool) -> int:
        if not allow_tmp:
            return 0
        count = 1 if _is_tmp_candidate(expr) else 0
        if isinstance(expr, (T.Add, T.Sub, T.Mul, T.Div, T.Matmul, T.GenericBinary)):
            return count + _count_candidates_expr(expr.left, allow_tmp) + _count_candidates_expr(expr.right, allow_tmp)
        if isinstance(expr, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
            return count + _count_candidates_expr(expr.val, allow_tmp)
        if isinstance(expr, (T.Permute3, T.Squeeze, T.Unsqueeze)):
            return count + _count_candidates_expr(expr.val, allow_tmp)
        if isinstance(expr, T.Broadcast):
            return count + _count_candidates_expr(expr.val, allow_tmp)
        if isinstance(expr, T.Concat):
            return count + _count_candidates_expr(expr.a, allow_tmp) + _count_candidates_expr(expr.b, allow_tmp)
        if isinstance(expr, T.Take):
            return count + _count_candidates_expr(expr.data, allow_tmp) + _count_candidates_expr(expr.indices, allow_tmp)
        if isinstance(expr, T.ReduceSum):
            return count + _count_candidates_expr(expr.val, False)
        if isinstance(expr, T.ReduceMax):
            return count + _count_candidates_expr(expr.val, False)
        if isinstance(expr, T.ReduceMin):
            return count + _count_candidates_expr(expr.val, False)
        if isinstance(expr, T.GenericCall):
            if expr.func_name == "transpose" and expr.args:
                return count + _count_candidates_expr(expr.args[0], allow_tmp)
            return count + sum(_count_candidates_expr(arg, allow_tmp) for arg in expr.args)
        if isinstance(expr, T.Cast):
            return count + _count_candidates_expr(expr.val, allow_tmp)
        return count

    def _count_candidates_stmt(node: T.ASTNode) -> int:
        if isinstance(node, T.Store):
            return _count_candidates_expr(node.value, True)
        if isinstance(node, T.Loop):
            return _count_candidates_stmt(node.body)
        if isinstance(node, T.Seq):
            return _count_candidates_stmt(node.left) + _count_candidates_stmt(node.right)
        if isinstance(node, T.Block):
            return sum(_count_candidates_stmt(stmt) for stmt in node.stmts)
        if isinstance(node, T.If):
            total = _count_candidates_stmt(node.then_branch)
            if node.else_branch:
                total += _count_candidates_stmt(node.else_branch)
            return total
        if isinstance(node, T.Let):
            return _count_candidates_expr(node.value, True) + _count_candidates_stmt(node.body)
        return 0

    def _infer_shape(expr: T.ASTNode, fallback: List[object]) -> List[object]:
        if isinstance(expr, T.Load):
            info = tensor_info_map.get(expr.tensor.name)
            if info and info.shape:
                return list(info.shape)
            return list(fallback)
        if isinstance(expr, T.Tensor):
            info = tensor_info_map.get(expr.name)
            if info and info.shape:
                return list(info.shape)
            return list(fallback)
        if isinstance(expr, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
            return _infer_shape(expr.val, fallback)
        if isinstance(expr, T.Unsqueeze):
            base = _infer_shape(expr.val, fallback)
            axis = expr.axis
            if axis < 0:
                axis += len(base) + 1
            if axis < 0:
                axis = 0
            if axis > len(base):
                axis = len(base)
            return base[:axis] + [1] + base[axis:]
        if isinstance(expr, T.Squeeze):
            base = _infer_shape(expr.val, fallback)
            axis = expr.axis
            if axis < 0:
                axis += len(base)
            if 0 <= axis < len(base):
                return base[:axis] + base[axis + 1 :]
            return base
        if isinstance(expr, T.Permute3):
            base = _infer_shape(expr.val, fallback)
            if len(base) == 3:
                perm = [expr.d0, expr.d1, expr.d2]
                try:
                    return [base[i] for i in perm]
                except IndexError:
                    return base
            return base
        if isinstance(expr, T.GenericCall) and expr.func_name == "transpose":
            base = _infer_shape(expr.args[0], fallback) if expr.args else list(fallback)
            perm_vals = [_get_const_int(arg) for arg in expr.args[1:]]
            if all(v is not None for v in perm_vals) and perm_vals:
                try:
                    return [base[i] for i in perm_vals]  # type: ignore[index]
                except Exception:
                    return base
            if len(base) == 2:
                return [base[1], base[0]]
            return base
        if isinstance(expr, (T.Add, T.Sub, T.Mul, T.Div, T.Matmul, T.GenericBinary)):
            left = _infer_shape(expr.left, fallback)
            right = _infer_shape(expr.right, fallback)
            if left and right and left == right:
                return left
            return list(fallback)
        if isinstance(expr, T.Broadcast):
            return list(fallback)
        if isinstance(expr, T.Concat):
            return list(fallback)
        if isinstance(expr, T.Take):
            return list(fallback)
        if isinstance(expr, T.Cast):
            return _infer_shape(expr.val, fallback)
        return list(fallback)

    total_candidates = _count_candidates_stmt(root)
    target_count = int(round(total_candidates * ratio))
    target_count = max(0, min(total_candidates, target_count))
    selected: set[int] = set()
    if ratio >= 1.0:
        selected = set(range(total_candidates))
    elif target_count > 0:
        step = total_candidates / target_count
        for k in range(target_count):
            idx = int(k * step + step / 2)
            if idx >= total_candidates:
                idx = total_candidates - 1
            while idx in selected and idx + 1 < total_candidates:
                idx += 1
            selected.add(idx)

    candidate_index = {"value": -1}

    def _should_tmpize_candidate() -> bool:
        if ratio <= 0.0:
            return False
        if ratio >= 1.0:
            return True
        candidate_index["value"] += 1
        return candidate_index["value"] in selected

    def _decompose_expr(
        expr: T.ASTNode,
        index: T.Index,
        dest_info: T.TensorInfo,
        allow_tmp: bool,
    ) -> Tuple[T.ASTNode, List[T.ASTNode]]:
        if allow_tmp and _is_tmp_candidate(expr) and _should_tmpize_candidate():
            if isinstance(expr, (T.Add, T.Sub, T.Mul, T.Div, T.Matmul, T.GenericBinary)):
                left, left_stmts = _decompose_expr(expr.left, index, dest_info, allow_tmp)
                right, right_stmts = _decompose_expr(expr.right, index, dest_info, allow_tmp)
                if isinstance(expr, T.GenericBinary):
                    new_expr = T.GenericBinary(expr.op, left, right)
                else:
                    new_expr = type(expr)(left, right)
                inferred = _infer_shape(new_expr, list(dest_info.shape))
                tmp_info = _fresh_tmp(T.TensorInfo(name=dest_info.name, shape=inferred, dtype=dest_info.dtype))
                tmp_store = T.Store(T.Tensor(tmp_info.name), new_expr, index)
                return T.Load(T.Tensor(tmp_info.name), index), left_stmts + right_stmts + [tmp_store]
            if isinstance(expr, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid, T.Permute3, T.Squeeze, T.Unsqueeze)):
                if isinstance(expr, T.Unsqueeze):
                    child_index = _index_drop_axis(index, expr.axis)
                elif isinstance(expr, T.Squeeze):
                    child_index = _index_insert_fulltile(index, expr.axis)
                elif isinstance(expr, T.Permute3):
                    child_index = _index_permute_inverse(index, expr.d0, expr.d1, expr.d2)
                else:
                    child_index = index
                val, val_stmts = _decompose_expr(expr.val, child_index, dest_info, allow_tmp)
                if isinstance(expr, T.Permute3):
                    new_expr = T.Permute3(val, expr.d0, expr.d1, expr.d2)
                elif isinstance(expr, T.Squeeze):
                    new_expr = T.Squeeze(val, expr.axis)
                elif isinstance(expr, T.Unsqueeze):
                    new_expr = T.Unsqueeze(val, expr.axis)
                else:
                    new_expr = type(expr)(val)
                inferred = _infer_shape(new_expr, list(dest_info.shape))
                tmp_info = _fresh_tmp(T.TensorInfo(name=dest_info.name, shape=inferred, dtype=dest_info.dtype))
                tmp_store = T.Store(T.Tensor(tmp_info.name), new_expr, index)
                return T.Load(T.Tensor(tmp_info.name), index), val_stmts + [tmp_store]
            if isinstance(expr, T.GenericCall) and expr.func_name == "transpose":
                perm_vals = [_get_const_int(arg) for arg in expr.args[1:]]
                if all(v is not None for v in perm_vals) and perm_vals:
                    if len(perm_vals) == 2:
                        child_index = _index_permute_inverse_2d(index, perm_vals[0], perm_vals[1])
                    elif len(perm_vals) == 3:
                        child_index = _index_permute_inverse(index, perm_vals[0], perm_vals[1], perm_vals[2])
                    else:
                        child_index = index
                else:
                    child_index = _index_permute_inverse_2d(index, 1, 0)
                val, val_stmts = _decompose_expr(expr.args[0], child_index, dest_info, allow_tmp)
                new_expr = T.GenericCall(expr.func_name, [val] + expr.args[1:])
                inferred = _infer_shape(new_expr, list(dest_info.shape))
                tmp_info = _fresh_tmp(T.TensorInfo(name=dest_info.name, shape=inferred, dtype=dest_info.dtype))
                tmp_store = T.Store(T.Tensor(tmp_info.name), new_expr, index)
                return T.Load(T.Tensor(tmp_info.name), index), val_stmts + [tmp_store]

        if isinstance(expr, (T.Add, T.Sub, T.Mul, T.Div, T.Matmul, T.GenericBinary)):
            left, left_stmts = _decompose_expr(expr.left, index, dest_info, allow_tmp)
            right, right_stmts = _decompose_expr(expr.right, index, dest_info, allow_tmp)
            if isinstance(expr, T.GenericBinary):
                return T.GenericBinary(expr.op, left, right), left_stmts + right_stmts
            return type(expr)(left, right), left_stmts + right_stmts
        if isinstance(expr, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
            val, val_stmts = _decompose_expr(expr.val, index, dest_info, allow_tmp)
            return type(expr)(val), val_stmts
        if isinstance(expr, T.Permute3):
            child_index = _index_permute_inverse(index, expr.d0, expr.d1, expr.d2)
            val, val_stmts = _decompose_expr(expr.val, child_index, dest_info, allow_tmp)
            return T.Permute3(val, expr.d0, expr.d1, expr.d2), val_stmts
        if isinstance(expr, T.Squeeze):
            child_index = _index_insert_fulltile(index, expr.axis)
            val, val_stmts = _decompose_expr(expr.val, child_index, dest_info, allow_tmp)
            return T.Squeeze(val, expr.axis), val_stmts
        if isinstance(expr, T.Unsqueeze):
            child_index = _index_drop_axis(index, expr.axis)
            val, val_stmts = _decompose_expr(expr.val, child_index, dest_info, allow_tmp)
            return T.Unsqueeze(val, expr.axis), val_stmts
        if isinstance(expr, T.ReduceSum):
            val, _ = _decompose_expr(expr.val, index, dest_info, False)
            return T.ReduceSum(val, expr.axis), []
        if isinstance(expr, T.ReduceMax):
            val, _ = _decompose_expr(expr.val, index, dest_info, False)
            return T.ReduceMax(val, expr.axis), []
        if isinstance(expr, T.ReduceMin):
            val, _ = _decompose_expr(expr.val, index, dest_info, False)
            return T.ReduceMin(val, expr.axis), []
        if isinstance(expr, T.Broadcast):
            val, val_stmts = _decompose_expr(expr.val, index, dest_info, allow_tmp)
            return T.Broadcast(val, expr.axis), val_stmts
        if isinstance(expr, T.Concat):
            left, left_stmts = _decompose_expr(expr.a, index, dest_info, allow_tmp)
            right, right_stmts = _decompose_expr(expr.b, index, dest_info, allow_tmp)
            return T.Concat(left, right, expr.axis), left_stmts + right_stmts
        if isinstance(expr, T.Take):
            data, data_stmts = _decompose_expr(expr.data, index, dest_info, allow_tmp)
            indices, idx_stmts = _decompose_expr(expr.indices, index, dest_info, allow_tmp)
            return T.Take(data, indices, expr.axis, expr.index), data_stmts + idx_stmts
        if isinstance(expr, T.GenericCall):
            if expr.func_name == "transpose":
                perm_vals = [_get_const_int(arg) for arg in expr.args[1:]]
                if all(v is not None for v in perm_vals) and perm_vals:
                    if len(perm_vals) == 2:
                        child_index = _index_permute_inverse_2d(index, perm_vals[0], perm_vals[1])
                    elif len(perm_vals) == 3:
                        child_index = _index_permute_inverse(index, perm_vals[0], perm_vals[1], perm_vals[2])
                    else:
                        child_index = index
                else:
                    child_index = _index_permute_inverse_2d(index, 1, 0)
                val, val_stmts = _decompose_expr(expr.args[0], child_index, dest_info, allow_tmp)
                return T.GenericCall(expr.func_name, [val] + expr.args[1:]), val_stmts
            args = []
            arg_stmts: List[T.ASTNode] = []
            for arg in expr.args:
                new_arg, new_stmts = _decompose_expr(arg, index, dest_info, allow_tmp)
                args.append(new_arg)
                arg_stmts.extend(new_stmts)
            return T.GenericCall(expr.func_name, args), arg_stmts
        if isinstance(expr, T.Cast):
            val, val_stmts = _decompose_expr(expr.val, index, dest_info, allow_tmp)
            return T.Cast(expr.dtype, val), val_stmts
        if isinstance(expr, T.Load):
            return expr, []
        if isinstance(expr, (T.Tensor, T.Index, T.Tile, T.FullTile, T.ConstTile, T.Elem, T.Const, T.Arange)):
            return expr, []
        return expr, []

    def _decompose_stmt(node: T.ASTNode) -> T.ASTNode:
        if isinstance(node, T.Store):
            dest_info = tensor_info_map.get(
                node.tensor.name,
                T.TensorInfo(name=node.tensor.name, shape=[], dtype="unknown"),
            )
            value, prelude = _decompose_expr(node.value, node.index, dest_info, True)
            final_store = T.Store(node.tensor, value, node.index)
            if not prelude:
                return final_store
            return _list_to_block(prelude + [final_store])
        if isinstance(node, T.Loop):
            return T.Loop(node.start, node.end, node.tile_name, node.loop_var, _decompose_stmt(node.body))
        if isinstance(node, T.Seq):
            return T.Seq(_decompose_stmt(node.left), _decompose_stmt(node.right))
        if isinstance(node, T.Block):
            return T.Block([_decompose_stmt(stmt) for stmt in node.stmts])
        if isinstance(node, T.If):
            return T.If(
                node.cond,
                _decompose_stmt(node.then_branch),
                _decompose_stmt(node.else_branch) if node.else_branch else None,
            )
        if isinstance(node, T.Let):
            return T.Let(node.tensor, node.value, _decompose_stmt(node.body))
        return node

    return _decompose_stmt(root), tmp_tensors


def _remove_let_nodes(node: T.ASTNode, env: dict, substitute_tensor: bool) -> T.ASTNode:
    if isinstance(node, T.Tensor):
        if substitute_tensor:
            bound = env.get(node.name)
            return bound if bound is not None else node
        return node

    if isinstance(node, T.Tile):
        return node

    if isinstance(node, T.FullTile):
        return node
    if isinstance(node, T.ConstTile):
        return node

    if isinstance(node, T.Const):
        return node

    if isinstance(node, T.Index):
        return T.Index([_remove_let_nodes(idx, env, True) for idx in node.indices])

    if isinstance(node, T.Load):
        return T.Load(
            _remove_let_nodes(node.tensor, env, False),
            _remove_let_nodes(node.index, env, True),
        )

    if isinstance(node, T.Store):
        return T.Store(
            _remove_let_nodes(node.tensor, env, False),
            _remove_let_nodes(node.value, env, True),
            _remove_let_nodes(node.index, env, True),
        )
    if isinstance(node, T.Seq):
        return T.Seq(
            _remove_let_nodes(node.left, env, True),
            _remove_let_nodes(node.right, env, True),
        )

    if isinstance(node, T.Block):
        new_stmts = []
        for stmt in node.stmts:
            updated = _remove_let_nodes(stmt, env, True)
            if isinstance(updated, T.Block):
                new_stmts.extend(updated.stmts)
            else:
                new_stmts.append(updated)
        return T.Block(new_stmts)

    if isinstance(node, T.If):
        return T.If(
            _remove_let_nodes(node.cond, env, True),
            _remove_let_nodes(node.then_branch, env, True),
            _remove_let_nodes(node.else_branch, env, True) if node.else_branch else None,
        )

    if isinstance(node, T.Let):
        value = _remove_let_nodes(node.value, env, True)
        if isinstance(node.tensor, T.Tensor):
            new_env = dict(env)
            new_env[node.tensor.name] = value
            return _remove_let_nodes(node.body, new_env, True)
        return T.Let(node.tensor, value, _remove_let_nodes(node.body, env, True))

    if isinstance(node, T.Loop):
        return T.Loop(
            _remove_let_nodes(node.start, env, True),
            _remove_let_nodes(node.end, env, True),
            node.tile_name,
            node.loop_var,
            _remove_let_nodes(node.body, env, True),
        )

    if isinstance(node, T.Add):
        return T.Add(_remove_let_nodes(node.left, env, True), _remove_let_nodes(node.right, env, True))
    if isinstance(node, T.Sub):
        return T.Sub(_remove_let_nodes(node.left, env, True), _remove_let_nodes(node.right, env, True))
    if isinstance(node, T.Mul):
        return T.Mul(_remove_let_nodes(node.left, env, True), _remove_let_nodes(node.right, env, True))
    if isinstance(node, T.Div):
        return T.Div(_remove_let_nodes(node.left, env, True), _remove_let_nodes(node.right, env, True))
    if isinstance(node, T.Exp):
        return T.Exp(_remove_let_nodes(node.val, env, True))
    if isinstance(node, T.Sqr):
        return T.Sqr(_remove_let_nodes(node.val, env, True))
    if isinstance(node, T.Sqrt):
        return T.Sqrt(_remove_let_nodes(node.val, env, True))
    if isinstance(node, T.Sigmoid):
        return T.Sigmoid(_remove_let_nodes(node.val, env, True))
    if isinstance(node, T.Matmul):
        return T.Matmul(_remove_let_nodes(node.left, env, True), _remove_let_nodes(node.right, env, True))
    if isinstance(node, T.Take):
        return T.Take(
            _remove_let_nodes(node.data, env, True),
            _remove_let_nodes(node.indices, env, True),
            node.axis,
            _remove_let_nodes(node.index, env, True),
        )
    if isinstance(node, T.ReduceSum):
        return T.ReduceSum(_remove_let_nodes(node.val, env, True), node.axis)
    if isinstance(node, T.ReduceMax):
        return T.ReduceMax(_remove_let_nodes(node.val, env, True), node.axis)
    if isinstance(node, T.ReduceMin):
        return T.ReduceMin(_remove_let_nodes(node.val, env, True), node.axis)
    if isinstance(node, T.Concat):
        return T.Concat(
            _remove_let_nodes(node.a, env, True),
            _remove_let_nodes(node.b, env, True),
            node.axis,
        )
    if isinstance(node, T.Broadcast):
        return T.Broadcast(_remove_let_nodes(node.val, env, True), node.axis)
    if isinstance(node, T.Permute3):
        return T.Permute3(
            _remove_let_nodes(node.val, env, True),
            node.d0,
            node.d1,
            node.d2,
        )
    if isinstance(node, T.Squeeze):
        return T.Squeeze(_remove_let_nodes(node.val, env, True), node.axis)
    if isinstance(node, T.Unsqueeze):
        return T.Unsqueeze(_remove_let_nodes(node.val, env, True), node.axis)
    if isinstance(node, T.GenericBinary):
        return T.GenericBinary(
            node.op,
            _remove_let_nodes(node.left, env, True),
            _remove_let_nodes(node.right, env, True),
        )
    if isinstance(node, T.GenericCall):
        return T.GenericCall(node.func_name, [_remove_let_nodes(arg, env, True) for arg in node.args])
    if isinstance(node, T.Cast):
        return T.Cast(node.dtype, _remove_let_nodes(node.val, env, True))
    if isinstance(node, T.Arange):
        return node

    return node

def _const_int(node):
    if isinstance(node, T.Const) and isinstance(node.value, int):
        return node.value
    return None

def _clean_var_name(name: str) -> str:
    return name[2:] if name.startswith("v_") else name

def _parse_tensor_dim_sig(sig: str) -> Optional[tuple[str, int]]:
    left = sig.split("=", 1)[0].strip()
    if "[" not in left or not left.endswith("]"):
        return None
    tensor, dim_str = left[:-1].split("[", 1)
    try:
        dim = int(dim_str)
    except ValueError:
        return None
    return tensor, dim

def _eval_int(node):
    if isinstance(node, T.Const) and isinstance(node.value, int):
        return node.value
    if isinstance(node, T.Add):
        left = _eval_int(node.left)
        right = _eval_int(node.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, T.Sub):
        left = _eval_int(node.left)
        right = _eval_int(node.right)
        if left is None or right is None:
            return None
        return left - right
    if isinstance(node, T.Mul):
        left = _eval_int(node.left)
        right = _eval_int(node.right)
        if left is None or right is None:
            return None
        return left * right
    if isinstance(node, T.Div):
        left = _eval_int(node.left)
        right = _eval_int(node.right)
        if left is None or right is None or right == 0:
            return None
        if left % right != 0:
            return None
        return left // right
    return None

def _replace_short_loops_with_fulltile(node, eliminated_vars: set, threshold: int):
    def _contains_elem(n: T.ASTNode, loop_var: str) -> bool:
        if isinstance(n, T.Elem):
            return _clean_var_name(n.name) == _clean_var_name(loop_var)
        if isinstance(n, T.TileOffset):
            return _clean_var_name(n.name) == _clean_var_name(loop_var)
        if isinstance(n, T.Index):
            return any(_contains_elem(idx, loop_var) for idx in n.indices)
        if isinstance(n, T.Load):
            return _contains_elem(n.index, loop_var)
        if isinstance(n, T.Store):
            return _contains_elem(n.value, loop_var) or _contains_elem(n.index, loop_var)
        if isinstance(n, T.Block):
            return any(_contains_elem(stmt, loop_var) for stmt in n.stmts)
        if isinstance(n, T.Loop):
            return _contains_elem(n.body, loop_var)
        if isinstance(n, T.If):
            return (
                _contains_elem(n.cond, loop_var)
                or _contains_elem(n.then_branch, loop_var)
                or (_contains_elem(n.else_branch, loop_var) if n.else_branch else False)
            )
        if isinstance(n, T.Let):
            return _contains_elem(n.value, loop_var) or _contains_elem(n.body, loop_var)
        if isinstance(n, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min, T.Matmul, T.GenericBinary)):
            return _contains_elem(n.left, loop_var) or _contains_elem(n.right, loop_var)
        if isinstance(n, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid, T.Cast)):
            return _contains_elem(n.val, loop_var)
        if isinstance(n, (T.Unsqueeze, T.Squeeze, T.Broadcast)):
            return _contains_elem(n.val, loop_var)
        if isinstance(n, (T.ReduceSum, T.ReduceMax, T.ReduceMin)):
            return _contains_elem(n.val, loop_var)
        if isinstance(n, T.Take):
            return (
                _contains_elem(n.data, loop_var)
                or _contains_elem(n.indices, loop_var)
                or _contains_elem(n.index, loop_var)
            )
        if isinstance(n, T.Concat):
            return _contains_elem(n.a, loop_var) or _contains_elem(n.b, loop_var)
        if isinstance(n, T.GenericCall):
            return any(_contains_elem(arg, loop_var) for arg in n.args)
        return False
    if isinstance(node, T.Tile):
        clean_name = _clean_var_name(node.name)
        if node.name in eliminated_vars or clean_name in eliminated_vars:
            return T.FullTile()
        return node
    if isinstance(node, T.TileOffset):
        clean_name = _clean_var_name(node.name)
        if node.name in eliminated_vars or clean_name in eliminated_vars:
            return T.FullTile()
        return node

    if isinstance(node, T.Elem):
        clean_name = _clean_var_name(node.name)
        if node.name in eliminated_vars or clean_name in eliminated_vars:
            return T.FullTile()
        return node

    if isinstance(node, T.FullTile):
        return node
    if isinstance(node, T.ConstTile):
        return node

    if isinstance(node, T.Tensor):
        return node

    if isinstance(node, T.Const):
        return node

    if isinstance(node, T.Index):
        return T.Index([_replace_short_loops_with_fulltile(idx, eliminated_vars, threshold) for idx in node.indices])

    if isinstance(node, T.Load):
        return T.Load(
            _replace_short_loops_with_fulltile(node.tensor, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.index, eliminated_vars, threshold),
        )

    if isinstance(node, T.Store):
        return T.Store(
            _replace_short_loops_with_fulltile(node.tensor, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.value, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.index, eliminated_vars, threshold),
        )

    if isinstance(node, T.Block):
        new_stmts = []
        for stmt in node.stmts:
            updated = _replace_short_loops_with_fulltile(stmt, eliminated_vars, threshold)
            if isinstance(updated, T.Block):
                new_stmts.extend(updated.stmts)
            else:
                new_stmts.append(updated)
        return T.Block(new_stmts)

    if isinstance(node, T.If):
        return T.If(
            _replace_short_loops_with_fulltile(node.cond, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.then_branch, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.else_branch, eliminated_vars, threshold) if node.else_branch else None,
        )

    if isinstance(node, T.Let):
        return T.Let(
            _replace_short_loops_with_fulltile(node.tensor, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.value, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.body, eliminated_vars, threshold),
        )

    if isinstance(node, T.Loop):
        start_val = _eval_int(node.start)
        end_val = _eval_int(node.end)
        extent = None
        if start_val is not None and end_val is not None:
            extent = end_val - start_val

        if extent is not None and extent <= threshold and not _contains_elem(node.body, node.loop_var):
            local_elims = set(eliminated_vars)
            clean_loop_var = _clean_var_name(node.loop_var)
            local_elims.add(node.loop_var)
            local_elims.add(clean_loop_var)
            local_elims.add(f"v_{clean_loop_var}")
            return _replace_short_loops_with_fulltile(node.body, local_elims, threshold)

        return T.Loop(
            _replace_short_loops_with_fulltile(node.start, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.end, eliminated_vars, threshold),
            node.tile_name,
            node.loop_var,
            _replace_short_loops_with_fulltile(node.body, eliminated_vars, threshold),
        )

    if isinstance(node, T.Add):
        return T.Add(
            _replace_short_loops_with_fulltile(node.left, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.right, eliminated_vars, threshold),
        )
    if isinstance(node, T.Sub):
        return T.Sub(
            _replace_short_loops_with_fulltile(node.left, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.right, eliminated_vars, threshold),
        )
    if isinstance(node, T.Mul):
        return T.Mul(
            _replace_short_loops_with_fulltile(node.left, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.right, eliminated_vars, threshold),
        )
    if isinstance(node, T.Div):
        return T.Div(
            _replace_short_loops_with_fulltile(node.left, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.right, eliminated_vars, threshold),
        )
    if isinstance(node, T.Exp):
        return T.Exp(_replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold))
    if isinstance(node, T.Sqr):
        return T.Sqr(_replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold))
    if isinstance(node, T.Sqrt):
        return T.Sqrt(_replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold))
    if isinstance(node, T.Sigmoid):
        return T.Sigmoid(_replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold))
    if isinstance(node, T.Matmul):
        return T.Matmul(
            _replace_short_loops_with_fulltile(node.left, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.right, eliminated_vars, threshold),
        )
    if isinstance(node, T.Take):
        return T.Take(
            _replace_short_loops_with_fulltile(node.data, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.indices, eliminated_vars, threshold),
            node.axis,
            _replace_short_loops_with_fulltile(node.index, eliminated_vars, threshold),
        )
    if isinstance(node, T.ReduceSum):
        return T.ReduceSum(
            _replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold),
            node.axis,
        )
    if isinstance(node, T.ReduceMax):
        return T.ReduceMax(
            _replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold),
            node.axis,
        )
    if isinstance(node, T.ReduceMin):
        return T.ReduceMin(
            _replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold),
            node.axis,
        )
    if isinstance(node, T.Concat):
        return T.Concat(
            _replace_short_loops_with_fulltile(node.a, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.b, eliminated_vars, threshold),
            node.axis,
        )
    if isinstance(node, T.Broadcast):
        return T.Broadcast(
            _replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold),
            node.axis,
        )
    if isinstance(node, T.Permute3):
        return T.Permute3(
            _replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold),
            node.d0,
            node.d1,
            node.d2,
        )
    if isinstance(node, T.Squeeze):
        return T.Squeeze(_replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold), node.axis)
    if isinstance(node, T.Unsqueeze):
        return T.Unsqueeze(_replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold), node.axis)
    if isinstance(node, T.GenericBinary):
        return T.GenericBinary(
            node.op,
            _replace_short_loops_with_fulltile(node.left, eliminated_vars, threshold),
            _replace_short_loops_with_fulltile(node.right, eliminated_vars, threshold),
        )
    if isinstance(node, T.GenericCall):
        return T.GenericCall(
            node.func_name,
            [_replace_short_loops_with_fulltile(arg, eliminated_vars, threshold) for arg in node.args],
        )
    if isinstance(node, T.Cast):
        return T.Cast(node.dtype, _replace_short_loops_with_fulltile(node.val, eliminated_vars, threshold))

    return node


def _contains_elem_for_loop(n: T.ASTNode, loop_var: str) -> bool:
    if isinstance(n, T.Elem):
        return _clean_var_name(n.name) == _clean_var_name(loop_var)
    if isinstance(n, T.TileOffset):
        return _clean_var_name(n.name) == _clean_var_name(loop_var)
    if isinstance(n, T.Index):
        return any(_contains_elem_for_loop(idx, loop_var) for idx in n.indices)
    if isinstance(n, T.Load):
        return _contains_elem_for_loop(n.index, loop_var)
    if isinstance(n, T.Store):
        return _contains_elem_for_loop(n.value, loop_var) or _contains_elem_for_loop(n.index, loop_var)
    if isinstance(n, T.Block):
        return any(_contains_elem_for_loop(stmt, loop_var) for stmt in n.stmts)
    if isinstance(n, T.Loop):
        return _contains_elem_for_loop(n.body, loop_var)
    if isinstance(n, T.If):
        return (
            _contains_elem_for_loop(n.cond, loop_var)
            or _contains_elem_for_loop(n.then_branch, loop_var)
            or (_contains_elem_for_loop(n.else_branch, loop_var) if n.else_branch else False)
        )
    if isinstance(n, T.Let):
        return _contains_elem_for_loop(n.value, loop_var) or _contains_elem_for_loop(n.body, loop_var)
    if isinstance(n, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min, T.Matmul, T.GenericBinary)):
        return _contains_elem_for_loop(n.left, loop_var) or _contains_elem_for_loop(n.right, loop_var)
    if isinstance(n, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid, T.Cast)):
        return _contains_elem_for_loop(n.val, loop_var)
    if isinstance(n, (T.Unsqueeze, T.Squeeze, T.Broadcast)):
        return _contains_elem_for_loop(n.val, loop_var)
    if isinstance(n, (T.ReduceSum, T.ReduceMax, T.ReduceMin)):
        return _contains_elem_for_loop(n.val, loop_var)
    if isinstance(n, T.Take):
        return (
            _contains_elem_for_loop(n.data, loop_var)
            or _contains_elem_for_loop(n.indices, loop_var)
            or _contains_elem_for_loop(n.index, loop_var)
        )
    if isinstance(n, T.Concat):
        return _contains_elem_for_loop(n.a, loop_var) or _contains_elem_for_loop(n.b, loop_var)
    if isinstance(n, T.GenericCall):
        return any(_contains_elem_for_loop(arg, loop_var) for arg in n.args)
    return False


def _eliminate_loop_vars_with_fulltile(node: T.ASTNode, eliminated_vars: set[str]) -> T.ASTNode:
    if not eliminated_vars:
        return node
    if isinstance(node, T.Tile):
        clean_name = _clean_var_name(node.name)
        if node.name in eliminated_vars or clean_name in eliminated_vars:
            return T.FullTile()
        return node
    if isinstance(node, T.TileOffset):
        clean_name = _clean_var_name(node.name)
        if node.name in eliminated_vars or clean_name in eliminated_vars:
            return T.FullTile()
        return node
    if isinstance(node, T.Elem):
        clean_name = _clean_var_name(node.name)
        if node.name in eliminated_vars or clean_name in eliminated_vars:
            return T.FullTile()
        return node
    if isinstance(node, T.Index):
        return T.Index([_eliminate_loop_vars_with_fulltile(idx, eliminated_vars) for idx in node.indices])
    if isinstance(node, T.Load):
        return T.Load(
            _eliminate_loop_vars_with_fulltile(node.tensor, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.index, eliminated_vars),
        )
    if isinstance(node, T.Store):
        return T.Store(
            _eliminate_loop_vars_with_fulltile(node.tensor, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.value, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.index, eliminated_vars),
        )
    if isinstance(node, T.Block):
        return T.Block([_eliminate_loop_vars_with_fulltile(stmt, eliminated_vars) for stmt in node.stmts])
    if isinstance(node, T.If):
        return T.If(
            _eliminate_loop_vars_with_fulltile(node.cond, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.then_branch, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.else_branch, eliminated_vars) if node.else_branch else None,
        )
    if isinstance(node, T.Let):
        return T.Let(
            _eliminate_loop_vars_with_fulltile(node.tensor, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.value, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.body, eliminated_vars),
        )
    if isinstance(node, T.Loop):
        clean_loop_var = _clean_var_name(node.loop_var)
        if node.loop_var in eliminated_vars or clean_loop_var in eliminated_vars:
            local_elims = set(eliminated_vars)
            local_elims.add(node.loop_var)
            local_elims.add(clean_loop_var)
            local_elims.add(f"v_{clean_loop_var}")
            return _eliminate_loop_vars_with_fulltile(node.body, local_elims)
        return T.Loop(
            _eliminate_loop_vars_with_fulltile(node.start, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.end, eliminated_vars),
            node.tile_name,
            node.loop_var,
            _eliminate_loop_vars_with_fulltile(node.body, eliminated_vars),
        )
    if isinstance(node, T.Seq):
        return T.Seq(
            _eliminate_loop_vars_with_fulltile(node.left, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.right, eliminated_vars),
        )
    if isinstance(node, T.Add):
        return T.Add(
            _eliminate_loop_vars_with_fulltile(node.left, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.right, eliminated_vars),
        )
    if isinstance(node, T.Sub):
        return T.Sub(
            _eliminate_loop_vars_with_fulltile(node.left, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.right, eliminated_vars),
        )
    if isinstance(node, T.Mul):
        return T.Mul(
            _eliminate_loop_vars_with_fulltile(node.left, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.right, eliminated_vars),
        )
    if isinstance(node, T.Div):
        return T.Div(
            _eliminate_loop_vars_with_fulltile(node.left, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.right, eliminated_vars),
        )
    if isinstance(node, T.Max):
        return T.Max(
            _eliminate_loop_vars_with_fulltile(node.left, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.right, eliminated_vars),
        )
    if isinstance(node, T.Min):
        return T.Min(
            _eliminate_loop_vars_with_fulltile(node.left, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.right, eliminated_vars),
        )
    if isinstance(node, T.Matmul):
        return T.Matmul(
            _eliminate_loop_vars_with_fulltile(node.left, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.right, eliminated_vars),
        )
    if isinstance(node, T.Exp):
        return T.Exp(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars))
    if isinstance(node, T.Sqr):
        return T.Sqr(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars))
    if isinstance(node, T.Sqrt):
        return T.Sqrt(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars))
    if isinstance(node, T.Sigmoid):
        return T.Sigmoid(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars))
    if isinstance(node, T.Take):
        return T.Take(
            _eliminate_loop_vars_with_fulltile(node.data, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.indices, eliminated_vars),
            node.axis,
            _eliminate_loop_vars_with_fulltile(node.index, eliminated_vars),
        )
    if isinstance(node, T.ReduceSum):
        return T.ReduceSum(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars), node.axis)
    if isinstance(node, T.ReduceMax):
        return T.ReduceMax(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars), node.axis)
    if isinstance(node, T.ReduceMin):
        return T.ReduceMin(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars), node.axis)
    if isinstance(node, T.Concat):
        return T.Concat(
            _eliminate_loop_vars_with_fulltile(node.a, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.b, eliminated_vars),
            node.axis,
        )
    if isinstance(node, T.Broadcast):
        return T.Broadcast(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars), node.axis)
    if isinstance(node, T.Permute3):
        return T.Permute3(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars), node.d0, node.d1, node.d2)
    if isinstance(node, T.Squeeze):
        return T.Squeeze(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars), node.axis)
    if isinstance(node, T.Unsqueeze):
        return T.Unsqueeze(_eliminate_loop_vars_with_fulltile(node.val, eliminated_vars), node.axis)
    if isinstance(node, T.GenericBinary):
        return T.GenericBinary(
            node.op,
            _eliminate_loop_vars_with_fulltile(node.left, eliminated_vars),
            _eliminate_loop_vars_with_fulltile(node.right, eliminated_vars),
        )
    if isinstance(node, T.GenericCall):
        return T.GenericCall(
            node.func_name,
            [_eliminate_loop_vars_with_fulltile(arg, eliminated_vars) for arg in node.args],
        )
    if isinstance(node, T.Cast):
        return T.Cast(node.dtype, _eliminate_loop_vars_with_fulltile(node.val, eliminated_vars))
    return node


def _remove_smallest_inner_loops(node: T.ASTNode, max_extent: int) -> T.ASTNode:
    if isinstance(node, T.Seq):
        return T.Seq(
            _remove_smallest_inner_loops(node.left, max_extent),
            _remove_smallest_inner_loops(node.right, max_extent),
        )
    if isinstance(node, T.Block):
        return T.Block([_remove_smallest_inner_loops(stmt, max_extent) for stmt in node.stmts])
    if isinstance(node, T.If):
        return T.If(
            _remove_smallest_inner_loops(node.cond, max_extent),
            _remove_smallest_inner_loops(node.then_branch, max_extent),
            _remove_smallest_inner_loops(node.else_branch, max_extent) if node.else_branch else None,
        )
    if isinstance(node, T.Let):
        return T.Let(
            _remove_smallest_inner_loops(node.tensor, max_extent),
            _remove_smallest_inner_loops(node.value, max_extent),
            _remove_smallest_inner_loops(node.body, max_extent),
        )
    if not isinstance(node, T.Loop):
        return node

    transformed_root: T.ASTNode = node
    while True:
        chain: list[T.Loop] = []
        cur = transformed_root
        while isinstance(cur, T.Loop):
            chain.append(cur)
            cur = cur.body
        if len(chain) < 2:
            break

        candidates: list[tuple[int, int, str]] = []
        for idx, loop in enumerate(chain[1:], start=1):
            start_val = _eval_int(loop.start)
            end_val = _eval_int(loop.end)
            if start_val is None or end_val is None:
                continue
            extent = end_val - start_val
            if extent > max_extent:
                continue
            if _contains_elem_for_loop(loop.body, loop.loop_var):
                continue
            candidates.append((extent, idx, loop.loop_var))

        if not candidates:
            break

        _, _, loop_var = min(candidates, key=lambda item: (item[0], item[1]))
        clean_var = _clean_var_name(loop_var)
        transformed_root = _eliminate_loop_vars_with_fulltile(
            transformed_root,
            {loop_var, clean_var, f"v_{clean_var}"},
        )

    if isinstance(transformed_root, T.Loop):
        return T.Loop(
            transformed_root.start,
            transformed_root.end,
            transformed_root.tile_name,
            transformed_root.loop_var,
            _remove_smallest_inner_loops(transformed_root.body, max_extent),
        )
    return _remove_smallest_inner_loops(transformed_root, max_extent)


def _collect_loop_vars_in_expr(node: T.ASTNode) -> set[str]:
    vars_found: set[str] = set()

    def add_name(name: str) -> None:
        vars_found.add(_clean_var_name(name))

    if isinstance(node, T.Tile):
        add_name(node.name)
    elif isinstance(node, T.TileOffset):
        add_name(node.name)
    elif isinstance(node, T.Elem):
        add_name(node.name)
    elif isinstance(node, T.VarRef):
        add_name(node.name)
    elif isinstance(node, T.Arange):
        add_name(node.axis)
    elif isinstance(node, T.Index):
        for idx in node.indices:
            vars_found.update(_collect_loop_vars_in_expr(idx))
    elif isinstance(node, T.Load):
        vars_found.update(_collect_loop_vars_in_expr(node.index))
    elif isinstance(node, T.Store):
        vars_found.update(_collect_loop_vars_in_expr(node.value))
        vars_found.update(_collect_loop_vars_in_expr(node.index))
    elif isinstance(node, T.Block):
        for stmt in node.stmts:
            vars_found.update(_collect_loop_vars_in_expr(stmt))
    elif isinstance(node, T.If):
        vars_found.update(_collect_loop_vars_in_expr(node.cond))
        vars_found.update(_collect_loop_vars_in_expr(node.then_branch))
        if node.else_branch:
            vars_found.update(_collect_loop_vars_in_expr(node.else_branch))
    elif isinstance(node, T.Let):
        vars_found.update(_collect_loop_vars_in_expr(node.value))
        vars_found.update(_collect_loop_vars_in_expr(node.body))
    elif isinstance(node, T.Loop):
        vars_found.update(_collect_loop_vars_in_expr(node.start))
        vars_found.update(_collect_loop_vars_in_expr(node.end))
        vars_found.update(_collect_loop_vars_in_expr(node.body))
    elif isinstance(node, (T.Add, T.Sub, T.Mul, T.Div, T.Max, T.Min, T.Matmul, T.GenericBinary)):
        vars_found.update(_collect_loop_vars_in_expr(node.left))
        vars_found.update(_collect_loop_vars_in_expr(node.right))
    elif isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid, T.Cast)):
        vars_found.update(_collect_loop_vars_in_expr(node.val))
    elif isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin, T.Broadcast, T.Unsqueeze, T.Squeeze)):
        vars_found.update(_collect_loop_vars_in_expr(node.val))
    elif isinstance(node, T.Take):
        vars_found.update(_collect_loop_vars_in_expr(node.data))
        vars_found.update(_collect_loop_vars_in_expr(node.indices))
        vars_found.update(_collect_loop_vars_in_expr(node.index))
    elif isinstance(node, T.Concat):
        vars_found.update(_collect_loop_vars_in_expr(node.a))
        vars_found.update(_collect_loop_vars_in_expr(node.b))
    elif isinstance(node, T.Permute3):
        vars_found.update(_collect_loop_vars_in_expr(node.val))
    elif isinstance(node, T.GenericCall):
        for arg in node.args:
            vars_found.update(_collect_loop_vars_in_expr(arg))

    return vars_found


def _unwrap_self_accumulation(
    tensor: T.Tensor,
    index: T.Index,
    value: T.ASTNode,
) -> T.ASTNode | None:
    def is_identity_one(node: T.ASTNode) -> bool:
        return isinstance(node, T.Const) and node.value == 1

    def match_self_load(node: T.ASTNode) -> bool:
        return (
            isinstance(node, T.Load)
            and node.tensor == tensor
            and node.index == index
        )

    def match_identity_mul(node: T.ASTNode) -> bool:
        if not isinstance(node, T.Mul):
            return False
        return (
            (match_self_load(node.left) and is_identity_one(node.right))
            or (match_self_load(node.right) and is_identity_one(node.left))
        )

    if not isinstance(value, T.Add):
        return None
    if match_identity_mul(value.left):
        return value.right
    if match_identity_mul(value.right):
        return value.left
    return None


def _remove_redundant_self_accumulations(node: T.ASTNode) -> T.ASTNode:
    if isinstance(node, T.Store):
        value = _remove_redundant_self_accumulations(node.value)
        index = _remove_redundant_self_accumulations(node.index)
        simplified = _unwrap_self_accumulation(node.tensor, index, value) if isinstance(index, T.Index) else None
        if simplified is not None:
            update_loop_vars = _collect_loop_vars_in_expr(simplified)
            store_loop_vars = _collect_loop_vars_in_expr(index)
            if update_loop_vars.issubset(store_loop_vars):
                value = simplified
        return T.Store(node.tensor, value, index)
    if isinstance(node, T.Index):
        return T.Index([_remove_redundant_self_accumulations(idx) for idx in node.indices])
    if isinstance(node, T.Load):
        return T.Load(node.tensor, _remove_redundant_self_accumulations(node.index))
    if isinstance(node, T.Block):
        return T.Block([_remove_redundant_self_accumulations(stmt) for stmt in node.stmts])
    if isinstance(node, T.If):
        return T.If(
            _remove_redundant_self_accumulations(node.cond),
            _remove_redundant_self_accumulations(node.then_branch),
            _remove_redundant_self_accumulations(node.else_branch) if node.else_branch else None,
        )
    if isinstance(node, T.Let):
        return T.Let(
            _remove_redundant_self_accumulations(node.tensor),
            _remove_redundant_self_accumulations(node.value),
            _remove_redundant_self_accumulations(node.body),
        )
    if isinstance(node, T.Loop):
        return T.Loop(
            _remove_redundant_self_accumulations(node.start),
            _remove_redundant_self_accumulations(node.end),
            node.tile_name,
            node.loop_var,
            _remove_redundant_self_accumulations(node.body),
        )
    if isinstance(node, T.Seq):
        return T.Seq(
            _remove_redundant_self_accumulations(node.left),
            _remove_redundant_self_accumulations(node.right),
        )
    if isinstance(node, T.Add):
        return T.Add(
            _remove_redundant_self_accumulations(node.left),
            _remove_redundant_self_accumulations(node.right),
        )
    if isinstance(node, T.Sub):
        return T.Sub(
            _remove_redundant_self_accumulations(node.left),
            _remove_redundant_self_accumulations(node.right),
        )
    if isinstance(node, T.Mul):
        return T.Mul(
            _remove_redundant_self_accumulations(node.left),
            _remove_redundant_self_accumulations(node.right),
        )
    if isinstance(node, T.Div):
        return T.Div(
            _remove_redundant_self_accumulations(node.left),
            _remove_redundant_self_accumulations(node.right),
        )
    if isinstance(node, T.Max):
        return T.Max(
            _remove_redundant_self_accumulations(node.left),
            _remove_redundant_self_accumulations(node.right),
        )
    if isinstance(node, T.Min):
        return T.Min(
            _remove_redundant_self_accumulations(node.left),
            _remove_redundant_self_accumulations(node.right),
        )
    if isinstance(node, T.Matmul):
        return T.Matmul(
            _remove_redundant_self_accumulations(node.left),
            _remove_redundant_self_accumulations(node.right),
        )
    if isinstance(node, (T.Exp, T.Sqr, T.Sqrt, T.Sigmoid)):
        return node.__class__(_remove_redundant_self_accumulations(node.val))
    if isinstance(node, T.Cast):
        return T.Cast(node.dtype, _remove_redundant_self_accumulations(node.val))
    if isinstance(node, T.Take):
        return T.Take(
            _remove_redundant_self_accumulations(node.data),
            _remove_redundant_self_accumulations(node.indices),
            node.axis,
            _remove_redundant_self_accumulations(node.index),
        )
    if isinstance(node, (T.ReduceSum, T.ReduceMax, T.ReduceMin, T.Broadcast, T.Unsqueeze, T.Squeeze)):
        return node.__class__(_remove_redundant_self_accumulations(node.val), node.axis)
    if isinstance(node, T.Concat):
        return T.Concat(
            _remove_redundant_self_accumulations(node.a),
            _remove_redundant_self_accumulations(node.b),
            node.axis,
        )
    if isinstance(node, T.Permute3):
        return T.Permute3(_remove_redundant_self_accumulations(node.val), node.d0, node.d1, node.d2)
    if isinstance(node, T.GenericBinary):
        return T.GenericBinary(
            node.op,
            _remove_redundant_self_accumulations(node.left),
            _remove_redundant_self_accumulations(node.right),
        )
    if isinstance(node, T.GenericCall):
        return T.GenericCall(node.func_name, [_remove_redundant_self_accumulations(arg) for arg in node.args])
    return node

def _tokenize_lisp(text: str) -> list[str]:
    tokens: list[str] = []
    buf = []
    in_string = False

    for ch in text:
        if in_string:
            buf.append(ch)
            if ch == '"':
                in_string = False
                tokens.append("".join(buf))
                buf = []
            continue

        if ch == '"':
            if buf:
                tokens.append("".join(buf))
                buf = []
            in_string = True
            buf.append(ch)
        elif ch in ("(", ")"):
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        elif ch.isspace():
            if buf:
                tokens.append("".join(buf))
                buf = []
        else:
            buf.append(ch)

    if buf:
        tokens.append("".join(buf))
    return tokens

def _parse_many(tokens: list[str]) -> list:
    exprs = []
    while tokens:
        exprs.append(_parse_expr(tokens))
    return exprs

def _parse_expr(tokens: list[str]):
    if not tokens:
        return ""
    tok = tokens.pop(0)
    if tok == "(":
        items = []
        while tokens and tokens[0] != ")":
            items.append(_parse_expr(tokens))
        if tokens and tokens[0] == ")":
            tokens.pop(0)
        return items
    if tok == ")":
        return ""
    return tok

def _atom_to_int(atom):
    if isinstance(atom, str) and atom.isdigit():
        return int(atom)
    return None

def _is_tile_form(expr) -> bool:
    if not isinstance(expr, list) or len(expr) != 2:
        return False
    head, var = expr[0], expr[1]
    if not isinstance(head, str) or not isinstance(var, str):
        return False
    return head == "tile" or head.startswith("tile_")

def _transform_seq(seq, eliminated_vars: set, threshold: int) -> list:
    out = []
    for expr in seq:
        transformed = _transform_expr(expr, eliminated_vars, threshold)
        if isinstance(transformed, list) and transformed and transformed[0] == "__splice__":
            out.extend(transformed[1:])
        else:
            out.append(transformed)
    return out

def _transform_expr(expr, eliminated_vars: set, threshold: int):
    if isinstance(expr, str):
        return expr
    if not isinstance(expr, list):
        return expr

    if _is_tile_form(expr) and expr[1] in eliminated_vars:
        return "fulltile"

    if len(expr) >= 6 and expr[0] == "loop":
        start = expr[1]
        end = expr[2]
        loop_var = expr[4]
        start_val = _atom_to_int(start)
        end_val = _atom_to_int(end)
        extent = None
        if start_val is not None and end_val is not None:
            extent = end_val - start_val
        elif start_val == 0 and end_val is not None:
            extent = end_val

        if isinstance(loop_var, str) and extent is not None and extent <= threshold:
            local_elims = set(eliminated_vars)
            local_elims.add(loop_var)
            new_body = _transform_seq(expr[5:], local_elims, threshold)
            return ["__splice__"] + new_body

        new_body = _transform_seq(expr[5:], eliminated_vars, threshold)
        return ["loop", start, end, expr[3], loop_var] + new_body

    return [_transform_expr(e, eliminated_vars, threshold) for e in expr]

def _to_inline(expr) -> str:
    if isinstance(expr, str):
        return expr
    if not isinstance(expr, list):
        return str(expr)
    inner = " ".join(_to_inline(e) for e in expr)
    return f"({inner})"

def _to_lisp(expr, level=0) -> str:
    indent = "  " * level
    if isinstance(expr, str):
        return f"{indent}{expr}" if level >= 0 else expr

    if not isinstance(expr, list):
        return f"{indent}{expr}"

    if len(expr) >= 6 and expr[0] == "loop":
        start = _to_inline(expr[1])
        end = _to_inline(expr[2])
        tile_name = expr[3]
        loop_var = expr[4]
        body_lines = [_to_lisp(e, level + 1) for e in expr[5:]]
        body_str = "\n".join(body_lines)
        return f"{indent}(loop {start} {end} {tile_name} {loop_var}\n{body_str}\n{indent})"

    if len(expr) >= 4 and expr[0] == "store":
        tensor = _to_inline(expr[1])
        val_str = _to_lisp(expr[2], level + 1)
        idx_str = _to_lisp(expr[3], level + 1)
        return f"{indent}(store {tensor}\n{val_str}\n{idx_str}\n{indent})"

    if len(expr) >= 2 and expr[0] == "if":
        cond = _to_inline(expr[1])
        then_str = _to_lisp(expr[2], level + 1) if len(expr) > 2 else ""
        res = f"{indent}(if {cond}\n{then_str}"
        if len(expr) > 3:
            else_str = _to_lisp(expr[3], level + 1)
            res += f"\n{indent}  (else\n{else_str}\n{indent}  )"
        res += f"\n{indent})"
        return res

    if len(expr) >= 3 and expr[0] == "let":
        binding = _to_inline(expr[1])
        body = _to_lisp(expr[2], level + 1)
        return f"{indent}(let {binding}\n{body}\n{indent})"

    return f"{indent}({_to_inline(expr)})"

def _to_lisp_program(exprs: list) -> str:
    return "\n".join(_to_lisp(expr, 0) for expr in exprs)
