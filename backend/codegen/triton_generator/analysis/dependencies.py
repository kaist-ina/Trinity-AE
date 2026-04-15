"""Cross-kernel and cross-sloop dependency analysis helpers."""

from __future__ import annotations

import re

from ...AstNode import ASTNode
from ...NodeType import NodeType


class DependencyAnalysis:
    def identify_cross_sloop_tensors(self, ast: ASTNode) -> set:
        """Identify tensors that are used across different sloop scopes."""
        cross_sloop_tensors = set()
        tensor_sloop_map = {}
        sloop_counter = [0]

        def collect_tensor_usage(node: ASTNode, current_sloop_id: int = -1):
            if node.node_type == NodeType.SLOOP:
                sloop_counter[0] += 1
                new_sloop_id = sloop_counter[0]
                for child in node.children:
                    if isinstance(child, ASTNode):
                        collect_tensor_usage(child, new_sloop_id)
            elif node.node_type == NodeType.STORE:
                tensor_node = node.children[0] if node.children else None
                if tensor_node and tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    for child in tensor_node.children:
                        if child.node_type == NodeType.VAR:
                            tensor_name = child.value
                            if tensor_name not in tensor_sloop_map:
                                tensor_sloop_map[tensor_name] = set()
                            if current_sloop_id != -1:
                                tensor_sloop_map[tensor_name].add(current_sloop_id)
                if len(node.children) > 1:
                    collect_tensor_usage(node.children[1], current_sloop_id)
                for i in range(2, len(node.children)):
                    if isinstance(node.children[i], ASTNode):
                        collect_tensor_usage(node.children[i], current_sloop_id)
            elif node.node_type == NodeType.LOAD:
                tensor_node = node.children[0] if node.children else None
                if tensor_node and tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    for child in tensor_node.children:
                        if child.node_type == NodeType.VAR:
                            tensor_name = child.value
                            if tensor_name not in tensor_sloop_map:
                                tensor_sloop_map[tensor_name] = set()
                            if current_sloop_id != -1:
                                tensor_sloop_map[tensor_name].add(current_sloop_id)
            else:
                for child in node.children:
                    if isinstance(child, ASTNode):
                        collect_tensor_usage(child, current_sloop_id)

        collect_tensor_usage(ast)

        for tensor_name, sloop_ids in tensor_sloop_map.items():
            if len(sloop_ids) > 1:
                cross_sloop_tensors.add(tensor_name)

        return cross_sloop_tensors

    def identify_cross_sloop_memory_tensors(self, ast: ASTNode) -> set:
        """Identify tensors that need memory storage due to index-pattern changes."""
        memory_tensors = set()
        tensor_index_patterns = {}
        tensor_sloop_vars = {}
        loop_counter = [0]
        sloop_vars = set()

        def collect_sloop_vars(node: ASTNode):
            if node.node_type == NodeType.SLOOP:
                sloop_vars.add(node.children[3].value)
            for child in node.children:
                if isinstance(child, ASTNode):
                    collect_sloop_vars(child)

        collect_sloop_vars(ast)

        def extract_index_pattern(index_node: ASTNode) -> str:
            if not index_node or index_node.node_type != NodeType.INDEX:
                return ""

            pattern_parts = []
            for child in index_node.children:
                if child.node_type == NodeType.TILE:
                    if child.children and child.children[0].node_type == NodeType.VAR:
                        pattern_parts.append(f"tile_{child.children[0].value}")
                    else:
                        pattern_parts.append("tile")
                elif child.node_type == NodeType.FULLTILE:
                    if child.children:
                        for sub_child in child.children:
                            if sub_child.node_type == NodeType.TILE and sub_child.children:
                                if sub_child.children[0].node_type == NodeType.VAR:
                                    pattern_parts.append(f"fulltile_tile_{sub_child.children[0].value}")
                                else:
                                    pattern_parts.append("fulltile_tile")
                            else:
                                pattern_parts.append("fulltile")
                    else:
                        pattern_parts.append("fulltile")
                elif child.node_type == NodeType.ELEM:
                    if child.children and child.children[0].node_type == NodeType.VAR:
                        pattern_parts.append(f"elem_{child.children[0].value}")
                    else:
                        pattern_parts.append("elem")
                else:
                    pattern_parts.append(str(child.node_type))

            return "_".join(pattern_parts)

        def collect_tensor_patterns(node: ASTNode, current_loop_id: int = -1):
            if node.node_type in [NodeType.SLOOP, NodeType.PLOOP]:
                loop_counter[0] += 1
                new_loop_id = loop_counter[0]
                for child in node.children:
                    if isinstance(child, ASTNode):
                        collect_tensor_patterns(child, new_loop_id)
            elif node.node_type in [NodeType.STORE, NodeType.LOAD]:
                tensor_node = node.children[0] if node.children else None
                if tensor_node and tensor_node.node_type in [NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR]:
                    index_node = None
                    if node.node_type == NodeType.STORE and len(node.children) > 2:
                        index_node = node.children[2]
                    elif node.node_type == NodeType.LOAD and len(node.children) > 1:
                        index_node = node.children[1]
                        if index_node and index_node.node_type == NodeType.FULLTILE:
                            index_node = ASTNode(NodeType.INDEX, [index_node])

                    if index_node:
                        pattern = extract_index_pattern(index_node)
                        for child in tensor_node.children:
                            if child.node_type == NodeType.VAR:
                                tensor_name = child.value
                                if tensor_name not in tensor_index_patterns:
                                    tensor_index_patterns[tensor_name] = {}
                                loop_id = current_loop_id if current_loop_id != -1 else -1
                                if loop_id not in tensor_index_patterns[tensor_name]:
                                    tensor_index_patterns[tensor_name][loop_id] = set()
                                tensor_index_patterns[tensor_name][loop_id].add(pattern)

                                if tensor_name not in tensor_sloop_vars:
                                    tensor_sloop_vars[tensor_name] = set()

                                for idx_child in index_node.children:
                                    for sloop_var in sloop_vars:
                                        if self.index_uses_loop_var(idx_child, sloop_var):
                                            tensor_sloop_vars[tensor_name].add(sloop_var)

                for child in node.children:
                    if isinstance(child, ASTNode):
                        collect_tensor_patterns(child, current_loop_id)
            else:
                for child in node.children:
                    if isinstance(child, ASTNode):
                        collect_tensor_patterns(child, current_loop_id)

        collect_tensor_patterns(ast)

        for tensor_name, loop_patterns in tensor_index_patterns.items():
            all_patterns = set()
            for patterns in loop_patterns.values():
                all_patterns.update(patterns)

            if len(all_patterns) > 1:
                memory_tensors.add(tensor_name)

                tile_vars = set()
                for pattern in all_patterns:
                    if "tile_" in pattern:
                        matches = re.findall(r"tile_(\w+)", pattern)
                        tile_vars.update(matches)
                if len(tile_vars) > 1:
                    memory_tensors.add(tensor_name)

        cross_sloop_tensors = self.identify_cross_sloop_tensors(ast)
        accumulators = self.identify_accumulators(ast)

        for tensor_name in cross_sloop_tensors:
            if tensor_name in self.state.intermediate_tensors:
                uses_sloop_var = (
                    tensor_name in tensor_sloop_vars and len(tensor_sloop_vars[tensor_name]) > 0
                )
                if not uses_sloop_var:
                    continue

                if tensor_name in tensor_index_patterns:
                    all_patterns = set()
                    for patterns in tensor_index_patterns[tensor_name].values():
                        all_patterns.update(patterns)

                    only_fulltile = all(
                        all("fulltile" in part and "tile_" not in part for part in pattern.split("_"))
                        for pattern in all_patterns
                    )

                    if only_fulltile and tensor_name not in self.state.cross_kernel_tensors:
                        continue

                memory_tensors.add(tensor_name)

        return memory_tensors

    def identify_sloop_intermediate_tensors(self, ast: ASTNode) -> set:
        """Identify intermediate tensors that escape their defining sloop.

        A tensor needs pre-allocation (tl.zeros) if it is stored inside a sloop
        and loaded outside that same sloop — whether in a different sloop or
        outside all sloops. Tensors only used within the same sloop iteration
        (e.g., shape transforms consumed immediately) are excluded.
        """
        # tensor -> set of sloop IDs where it is stored
        store_scopes = {}
        # tensor -> set of sloop IDs where it is loaded (-1 = outside all sloops)
        load_scopes = {}
        sloop_counter = [0]

        def traverse(node: ASTNode, current_sloop_id: int = -1):
            if node.node_type == NodeType.SLOOP:
                sloop_counter[0] += 1
                new_id = sloop_counter[0]
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, new_id)
            elif node.node_type == NodeType.STORE:
                tensor_node = node.children[0]
                if current_sloop_id != -1 and tensor_node.node_type in [
                    NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR
                ]:
                    for child in tensor_node.children:
                        if child.node_type == NodeType.VAR:
                            store_scopes.setdefault(child.value, set()).add(current_sloop_id)
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, current_sloop_id)
            elif node.node_type == NodeType.LOAD:
                tensor_node = node.children[0]
                if tensor_node.node_type in [
                    NodeType.INPUT, NodeType.OUTPUT, NodeType.TENSOR
                ]:
                    for child in tensor_node.children:
                        if child.node_type == NodeType.VAR:
                            load_scopes.setdefault(child.value, set()).add(current_sloop_id)
            else:
                for child in node.children:
                    if isinstance(child, ASTNode):
                        traverse(child, current_sloop_id)

        traverse(ast)

        escaped = set()
        for tensor, stored_in in store_scopes.items():
            loaded_in = load_scopes.get(tensor, set())
            # If loaded anywhere outside the sloop(s) it was stored in
            if loaded_in - stored_in:
                escaped.add(tensor)

        return escaped & self.state.intermediate_tensors
