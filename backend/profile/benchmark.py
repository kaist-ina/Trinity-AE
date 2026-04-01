import torch
import triton
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import tempfile
import importlib.util
import sys
import os
import shutil
import traceback
from dataclasses import dataclass
from tqdm import tqdm
import json
import argparse
import re

try:
    # When running from repo root with -m backend.profile.benchmark
    from backend.codegen.convert_module import convert_ir_to_triton
except ModuleNotFoundError:
    # When running from backend/ with -m profile.benchmark
    from codegen.convert_module import convert_ir_to_triton


@dataclass
class BenchmarkResult:
    ir_id: int
    ir_expression: str
    execution_time: float
    error: Optional[str] = None
    benchmarked: bool = True
    source_ir_id: Optional[int] = None
    predicted_execution_time: Optional[float] = None
    selection_reason: Optional[str] = None


@dataclass
class ExpressionInfo:
    ir_id: int
    ir_expression: str
    signature: str
    features: np.ndarray


class IRBenchmark:
    _IR_KEYWORDS = {
        "seq", "loop", "store", "load", "tensor", "input", "output", "index",
        "fulltile", "elem", "tile", "const_tile", "+", "-", "*", "/", "@",
        "exp", "sqr", "sqrt", "sigmoid", "cast", "rsum", "rmax", "rmin",
        "bcast", "permute3", "permute4", "transpose", "unsqueeze", "squeeze",
        "concat", "max", "min", "arange", "dummy", "let", "if", "else",
    }

    def __init__(
        self,
        shapes_path: str,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
    ):
        """Initialize benchmark with shapes.json."""
        self.tensor_types: Dict[str, str] = {}
        self.tensor_shapes, self.tensor_types = self._load_shapes_json(shapes_path)
        self.shapes_path = shapes_path
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

        # Setup device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        print(f"GPU: {torch.cuda.get_device_name(self.device)}")
        # if self.device.type != 'cuda':
        #     raise RuntimeError("CUDA device not available. Triton requires CUDA.")
            
        # Create test tensors
        self.create_test_tensors()
        
        # Track temp files for cleanup
        self._temp_files = []

    def _load_shapes_json(self, file_path: str) -> Tuple[Dict[str, Tuple[int, ...]], Dict[str, str]]:
        """Load tensor shapes and types from shapes.json.

        Supports two formats:
        1. Flat format (frontend-generated):
           {"X": {"shape": [16, 4096], "type": "input"}, ...}
        2. Nested format (backend shapes/):
           {"config": {...}, "tensors": {"X": {"shape": [16, 4096], "type": "input"}, ...}}
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check if nested format with "tensors" key
        if "tensors" in data and isinstance(data["tensors"], dict):
            tensor_data = data["tensors"]
        else:
            tensor_data = data

        tensor_shapes: Dict[str, Tuple[int, ...]] = {}
        tensor_types: Dict[str, str] = {}

        for name, value in tensor_data.items():
            if isinstance(value, dict):
                shape = value.get("shape")
                tensor_type = value.get("type", "input")
            else:
                shape = value
                tensor_type = "input"

            if not isinstance(shape, list):
                continue

            dims = []
            valid = True
            for dim in shape:
                if isinstance(dim, int):
                    dims.append(dim)
                else:
                    valid = False
                    break

            if not valid:
                continue

            tensor_shapes[name] = tuple(dims)
            tensor_types[name] = tensor_type

        return tensor_shapes, tensor_types

    def create_test_tensors(self):
        """Create random test tensors for benchmarking."""
        self.tensors = {}
        for name, shape in self.tensor_shapes.items():
            self.tensors[name] = self._allocate_tensor(name, torch.float16)

    def _tensor_dtype(self, name: str, fp32_tensor_names: Optional[set[str]] = None) -> torch.dtype:
        if fp32_tensor_names and name in fp32_tensor_names:
            return torch.float32
        return torch.float16

    def _allocate_tensor(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        shape = self.tensor_shapes[name]
        tensor_type = self.tensor_types.get(name, "input")
        zero_init_types = {"output", "intermediate"}

        if "T_softmax_maxelem" in name:
            return torch.full(
                shape,
                float(torch.finfo(dtype).min),
                dtype=dtype,
                device=self.device,
            )
        if tensor_type in zero_init_types:
            return torch.zeros(shape, dtype=dtype, device=self.device)
        return torch.randn(shape, dtype=dtype, device=self.device).clamp(-1, 1) * 0.01

    def _ensure_tensor_storage(self, fp32_tensor_names: Optional[set[str]] = None) -> None:
        fp32_tensor_names = fp32_tensor_names or set()
        for name, shape in self.tensor_shapes.items():
            expected_dtype = self._tensor_dtype(name, fp32_tensor_names)
            tensor = self.tensors.get(name)

            if tensor is None or tuple(tensor.shape) != tuple(shape):
                self.tensors[name] = self._allocate_tensor(name, expected_dtype)
                continue

            if tensor.dtype != expected_dtype:
                tensor_type = self.tensor_types.get(name, "input")
                if tensor_type == "input":
                    self.tensors[name] = tensor.to(dtype=expected_dtype)
                else:
                    self.tensors[name] = self._allocate_tensor(name, expected_dtype)

    def parse_ir_file(self, file_path: str) -> List[Tuple[int, str]]:
        """Parse the IR expressions file and extract all expressions."""
        expressions = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and ':' in line:
                    # Extract IR ID and expression
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        id_part = parts[0]
                        try:
                            ir_id = int(id_part)
                            ir_expr = parts[1].strip()
                            if 'dummydata' not in ir_expr:
                                expressions.append((ir_id, ir_expr))
                        except:
                            continue
                            
        return expressions

    def _tokenize_ir(self, ir_expr: str) -> List[str]:
        return re.findall(r"\(|\)|[^\s()]+", ir_expr)

    def _is_number_token(self, token: str) -> bool:
        try:
            float(token)
            return True
        except (TypeError, ValueError):
            return False

    def _canonicalize_ir(self, ir_expr: str) -> str:
        mapping: Dict[str, str] = {}
        tokens = self._tokenize_ir(ir_expr)
        normalized = []
        for token in tokens:
            if token in {"(", ")"} or token in self._IR_KEYWORDS or self._is_number_token(token):
                normalized.append(token)
                continue
            alias = mapping.get(token)
            if alias is None:
                alias = f"v{len(mapping)}"
                mapping[token] = alias
            normalized.append(alias)
        return " ".join(normalized)

    def _extract_loop_depth(self, tokens: List[str]) -> int:
        stack: List[str] = []
        max_depth = 0
        for idx, token in enumerate(tokens):
            if token == "(":
                node_type = tokens[idx + 1] if idx + 1 < len(tokens) else ""
                stack.append(node_type)
                if node_type == "loop":
                    max_depth = max(max_depth, sum(1 for item in stack if item == "loop"))
            elif token == ")" and stack:
                stack.pop()
        return max_depth

    def _extract_features(self, ir_expr: str) -> np.ndarray:
        tokens = self._tokenize_ir(ir_expr)
        token_counts = {token: tokens.count(token) for token in [
            "loop", "store", "load", "@", "rsum", "rmax", "rmin",
            "exp", "permute3", "transpose", "unsqueeze", "squeeze",
            "bcast", "sqrt", "sigmoid", "concat", "cast", "fulltile",
            "elem", "const_tile",
        ]}
        numeric_tokens = [float(token) for token in tokens if self._is_number_token(token)]
        comma_tokens = sum(1 for token in tokens if "," in token)
        return np.array([
            float(len(tokens)),
            float(token_counts["loop"]),
            float(self._extract_loop_depth(tokens)),
            float(token_counts["store"]),
            float(token_counts["load"]),
            float(token_counts["@"]),
            float(token_counts["rsum"] + token_counts["rmax"] + token_counts["rmin"]),
            float(token_counts["exp"]),
            float(token_counts["permute3"] + token_counts["transpose"]),
            float(token_counts["unsqueeze"] + token_counts["squeeze"]),
            float(token_counts["bcast"]),
            float(token_counts["sqrt"]),
            float(token_counts["sigmoid"]),
            float(token_counts["concat"]),
            float(token_counts["cast"]),
            float(token_counts["fulltile"]),
            float(token_counts["elem"]),
            float(token_counts["const_tile"]),
            float(comma_tokens),
            float(sum(numeric_tokens)),
            float(max(numeric_tokens) if numeric_tokens else 0.0),
        ], dtype=np.float64)

    def build_expression_infos(self, expressions: List[Tuple[int, str]]) -> List[ExpressionInfo]:
        infos: List[ExpressionInfo] = []
        for ir_id, ir_expr in expressions:
            infos.append(
                ExpressionInfo(
                    ir_id=ir_id,
                    ir_expression=ir_expr,
                    signature=self._canonicalize_ir(ir_expr),
                    features=self._extract_features(ir_expr),
                )
            )
        return infos

    def _pick_diverse_seed_indices(self, infos: List[ExpressionInfo], count: int) -> List[int]:
        if not infos or count <= 0:
            return []
        count = min(count, len(infos))
        feature_matrix = np.stack([info.features for info in infos])
        selected = [0]
        if count == 1:
            return selected
        norms = np.linalg.norm(feature_matrix, axis=1)
        farthest = int(np.argmax(norms))
        if farthest not in selected:
            selected.append(farthest)
        while len(selected) < count:
            selected_matrix = feature_matrix[selected]
            distances = np.min(
                np.linalg.norm(feature_matrix[:, None, :] - selected_matrix[None, :, :], axis=2),
                axis=1,
            )
            for idx in selected:
                distances[idx] = -1.0
            next_idx = int(np.argmax(distances))
            if next_idx in selected:
                break
            selected.append(next_idx)
        return selected

    def _predict_candidate_score(
        self,
        candidate: ExpressionInfo,
        benchmarked_infos: List[ExpressionInfo],
        benchmarked_results: List[BenchmarkResult],
    ) -> Tuple[float, float]:
        benchmark_matrix = np.stack([info.features for info in benchmarked_infos])
        distances = np.linalg.norm(benchmark_matrix - candidate.features, axis=1)
        neighbor_order = np.argsort(distances)
        k = min(3, len(neighbor_order))
        if k == 0:
            return float("inf"), float("inf")
        top_idx = neighbor_order[:k]
        top_distances = distances[top_idx]
        top_times = np.array([benchmarked_results[idx].execution_time for idx in top_idx], dtype=np.float64)
        weights = 1.0 / np.maximum(top_distances, 1e-6)
        predicted = float(np.dot(weights, top_times) / np.sum(weights))
        uncertainty = float(np.mean(top_distances))
        return predicted, uncertainty

    def _select_optimized_expression_indices(
        self,
        unique_infos: List[ExpressionInfo],
        max_benchmarks: int,
        seed_samples: int,
    ) -> List[int]:
        if len(unique_infos) <= max_benchmarks:
            return list(range(len(unique_infos)))
        seed_count = min(max(1, seed_samples), max_benchmarks)
        selected = self._pick_diverse_seed_indices(unique_infos, seed_count)
        while len(selected) < max_benchmarks:
            benchmarked_infos = [unique_infos[idx] for idx in selected]
            remaining = [idx for idx in range(len(unique_infos)) if idx not in selected]
            if not remaining:
                break

            benchmark_matrix = np.stack([info.features for info in benchmarked_infos])
            best_idx = None
            best_distance = -1.0
            for idx in remaining:
                distances = np.linalg.norm(benchmark_matrix - unique_infos[idx].features, axis=1)
                min_distance = float(np.min(distances))
                if min_distance > best_distance:
                    best_distance = min_distance
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
        return selected

    def _select_local_refinement_indices(
        self,
        unique_infos: List[ExpressionInfo],
        selected_indices: List[int],
        benchmarked_results: List[BenchmarkResult],
        refinement_budget: int,
        top_k: int,
    ) -> List[Tuple[int, int, float]]:
        if refinement_budget <= 0 or not selected_indices or not benchmarked_results:
            return []

        ranked_positions = [
            pos
            for pos, result in sorted(
                enumerate(benchmarked_results),
                key=lambda item: (item[1].error is not None, item[1].execution_time),
            )
            if result.execution_time != float("inf")
        ]
        if not ranked_positions:
            return []

        anchor_positions = ranked_positions[: max(1, top_k)]
        remaining = [idx for idx in range(len(unique_infos)) if idx not in selected_indices]
        if not remaining:
            return []

        candidate_entries: List[Tuple[float, float, int, int]] = []
        for idx in remaining:
            best_anchor_pos = None
            best_distance = float("inf")
            for anchor_pos in anchor_positions:
                anchor_idx = selected_indices[anchor_pos]
                distance = float(
                    np.linalg.norm(unique_infos[idx].features - unique_infos[anchor_idx].features)
                )
                if distance < best_distance:
                    best_distance = distance
                    best_anchor_pos = anchor_pos

            if best_anchor_pos is None:
                continue

            anchor_result = benchmarked_results[best_anchor_pos]
            candidate_entries.append(
                (best_distance, anchor_result.execution_time, idx, best_anchor_pos)
            )

        candidate_entries.sort(key=lambda item: (item[0], item[1], item[2]))
        chosen = candidate_entries[:refinement_budget]
        return [
            (idx, selected_indices[anchor_pos], distance)
            for distance, _, idx, anchor_pos in chosen
        ]

    def _nearest_benchmarked_result(
        self,
        candidate: ExpressionInfo,
        benchmarked_infos: List[ExpressionInfo],
        benchmarked_results: List[BenchmarkResult],
    ) -> Tuple[Optional[ExpressionInfo], Optional[BenchmarkResult], float, float]:
        if not benchmarked_infos:
            return None, None, float("inf"), float("inf")

        benchmark_matrix = np.stack([info.features for info in benchmarked_infos])
        distances = np.linalg.norm(benchmark_matrix - candidate.features, axis=1)
        best_pos = int(np.argmin(distances))
        nearest_info = benchmarked_infos[best_pos]
        nearest_result = benchmarked_results[best_pos]
        predicted, uncertainty = self._predict_candidate_score(
            candidate,
            benchmarked_infos,
            benchmarked_results,
        )
        return nearest_info, nearest_result, predicted, uncertainty

    def run_optimized_benchmarks(
        self,
        expressions: List[Tuple[int, str]],
        max_benchmarks: int,
        seed_samples: int,
        local_refine_budget: int = 0,
        local_refine_top_k: int = 1,
    ) -> List[BenchmarkResult]:
        infos = self.build_expression_infos(expressions)
        if not infos:
            return []

        signature_groups: Dict[str, List[ExpressionInfo]] = {}
        for info in infos:
            signature_groups.setdefault(info.signature, []).append(info)

        unique_infos = [group[0] for group in signature_groups.values()]
        selected_unique_indices = self._select_optimized_expression_indices(
            unique_infos,
            max_benchmarks,
            seed_samples,
        )
        selected_unique_index_set = set(selected_unique_indices)

        print(
            "Optimized benchmarking:"
            f" {len(expressions)} expressions, {len(unique_infos)} unique signatures,"
            f" benchmarking {len(selected_unique_indices)} representatives"
        )

        representative_results: Dict[str, BenchmarkResult] = {}
        benchmarked_infos: List[ExpressionInfo] = []
        benchmarked_results: List[BenchmarkResult] = []

        with tqdm(total=len(selected_unique_indices), desc="Benchmarking reps", unit="IR") as pbar:
            for unique_idx, info in enumerate(unique_infos):
                if unique_idx not in selected_unique_index_set:
                    continue
                reason = "seed" if len(benchmarked_infos) < min(seed_samples, len(selected_unique_indices)) else "diverse"
                result = self.run_single_benchmark(info.ir_id, info.ir_expression)
                result.selection_reason = reason
                representative_results[info.signature] = result
                benchmarked_infos.append(info)
                benchmarked_results.append(result)
                pbar.update(1)
                pbar.set_postfix(valid=sum(1 for r in benchmarked_results if r.error is None))

        refinement_choices = self._select_local_refinement_indices(
            unique_infos,
            selected_unique_indices,
            benchmarked_results,
            refinement_budget=local_refine_budget,
            top_k=local_refine_top_k,
        )

        if refinement_choices:
            print(
                "Local refinement:"
                f" benchmarking {len(refinement_choices)} neighbors around"
                f" top-{max(1, local_refine_top_k)} representatives"
            )
            with tqdm(total=len(refinement_choices), desc="Refining local", unit="IR") as pbar:
                for refine_idx, anchor_idx, distance in refinement_choices:
                    info = unique_infos[refine_idx]
                    anchor_info = unique_infos[anchor_idx]
                    result = self.run_single_benchmark(info.ir_id, info.ir_expression)
                    result.selection_reason = (
                        f"local_refine_from={anchor_info.ir_id},distance={distance:.4f}"
                    )
                    representative_results[info.signature] = result
                    benchmarked_infos.append(info)
                    benchmarked_results.append(result)
                    pbar.update(1)
                    pbar.set_postfix(valid=sum(1 for r in benchmarked_results if r.error is None))

        results: List[BenchmarkResult] = []
        for info in infos:
            representative = representative_results.get(info.signature)
            if representative is not None:
                if info.ir_id == representative.ir_id:
                    results.append(representative)
                else:
                    results.append(
                        BenchmarkResult(
                            ir_id=info.ir_id,
                            ir_expression=info.ir_expression,
                            execution_time=representative.execution_time,
                            error=representative.error,
                            benchmarked=False,
                            source_ir_id=representative.ir_id,
                            predicted_execution_time=representative.execution_time,
                            selection_reason="dedup",
                        )
                    )
                continue

            nearest_info, nearest_result, predicted, uncertainty = self._nearest_benchmarked_result(
                info,
                benchmarked_infos,
                benchmarked_results,
            )
            predicted_error = nearest_result.error if nearest_result is not None else "No benchmark representative"
            results.append(
                BenchmarkResult(
                    ir_id=info.ir_id,
                    ir_expression=info.ir_expression,
                    execution_time=predicted if predicted != float("inf") else float("inf"),
                    error=predicted_error,
                    benchmarked=False,
                    source_ir_id=nearest_info.ir_id if nearest_info is not None else None,
                    predicted_execution_time=predicted if predicted != float("inf") else None,
                    selection_reason=f"predicted_nn_uncertainty={uncertainty:.4f}",
                )
            )

        results.sort(key=lambda item: item.ir_id)
        return results

    def generate_kernel_code(self, ir_expr: str, constants: Dict[str, int] = None) -> Optional[str]:
        """Generate Triton kernel code from IR expression.
        
        Args:
            ir_expr: IR expression string
            constants: Optional mapping of variable names to constant values
        """
        try:
            kernel_code = convert_ir_to_triton(ir_expr, self.tensor_shapes, None)

            return kernel_code
            
        except Exception as e:
            print(f"Error generating kernel: {e}")
            traceback.print_exc()
            return None

    def compile_and_load_kernel(self, kernel_code: str, kernel_id: int) -> Optional[callable]:
        """Compile Triton kernel code and return callable function."""
        try:
            # Create temporary module file
            module_name = f"llama_kernel_{kernel_id}"
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(kernel_code)
                temp_file = f.name
            
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(module_name, temp_file)
            module = importlib.util.module_from_spec(spec)
            
            # Keep the file path in the module for Triton to find source
            module.__file__ = temp_file
            
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Don't delete temp file immediately - Triton needs it
            # Store temp file path for later cleanup
            self._temp_files = getattr(self, '_temp_files', [])
            self._temp_files.append(temp_file)
            
            # Return the module instead of just the function
            # So we can access metadata
            if hasattr(module, 'forward'):
                return module
            else:
                print("Error: Cannot call the kernel")
                return None
            
        except Exception as e:
            print(f"Error compiling kernel: {e}")
            traceback.print_exc()
            if 'temp_file' in locals() and os.path.exists(temp_file):
                # Print the generated code for debugging
                with open(temp_file, 'r') as f:
                    print("Generated kernel code:")
                    print(f.read())
                os.unlink(temp_file)
            return None

    def benchmark_kernel(
        self,
        kernel_module,
        ir_id,
        warmup_runs: Optional[int] = None,
        benchmark_runs: Optional[int] = None,
    ) -> float:
        """Benchmark a single kernel and return execution time in milliseconds."""
        try:
            warmup_runs = self.warmup_runs if warmup_runs is None else warmup_runs
            benchmark_runs = self.benchmark_runs if benchmark_runs is None else benchmark_runs
            # Get metadata and forward function
            tensor_params = getattr(kernel_module, 'TENSOR_PARAMS', [])
            fp32_tensor_params = set(getattr(kernel_module, 'FP32_TENSOR_PARAMS', []))
            kernel_fn = kernel_module.forward
            
            if not tensor_params:
                raise ValueError("No TENSOR_PARAMS found in kernel module.")

            self._ensure_tensor_storage(fp32_tensor_params)

            # Reset output tensors before each benchmark
            for name in tensor_params:
                if name in self.tensors and self.tensor_types.get(name) in {"output", "intermediate"}:
                    self.tensors[name].zero_()
                    if "T_softmax_maxelem" in name:
                        self.tensors[name].fill_(float(torch.finfo(self.tensors[name].dtype).min))
            
            # Build argument list based on metadata
            args = []
            for param in tensor_params:
                if param in self.tensors:
                    args.append(self.tensors[param])
                else:
                    # Create zero tensor if not exists (for intermediate tensors)
                    if param in self.tensor_shapes:
                        dtype = self._tensor_dtype(param, fp32_tensor_params)
                        tensor = self._allocate_tensor(param, dtype)
                        self.tensors[param] = tensor
                        args.append(tensor)
                    else:
                        raise ValueError(f"Unknown tensor parameter: {param}")
            
            stream = torch.cuda.Stream(self.device)
            # First call to trigger autotune (not counted in warmup)
            kernel_fn(*args)
            torch.cuda.synchronize()
            
            # Triton Warmup - now using the best configuration from autotune
            with torch.cuda.stream(stream):
                for _ in range(warmup_runs):
                    kernel_fn(*args)
            stream.synchronize()

            # CUDA Graph Warmup
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with torch.cuda.graph(graph, stream=stream):
                    kernel_fn(*args)
            # Synchronize before timing
            stream.synchronize()
            
            # Benchmark runs
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            with torch.cuda.stream(stream):
                start_event.record()
                for _ in range(benchmark_runs):
                    graph.replay()
                end_event.record()
            stream.synchronize()
            
            # Return average time in milliseconds
            avg_time = (start_event.elapsed_time(end_event)) / benchmark_runs
            return avg_time
            
        except Exception as e:
            print(f"Error benchmarking kernel: {e}")
            traceback.print_exc()
            raise

    def run_single_benchmark(self, ir_id: int, ir_expr: str) -> BenchmarkResult:
        """Run benchmark for a single IR expression."""
        try:
            # Generate kernel code
            kernel_code = self.generate_kernel_code(ir_expr)
            if kernel_code is None:
                return BenchmarkResult(ir_id, ir_expr, float('inf'), "Failed to generate kernel")
            
            # Compile kernel
            kernel_module = self.compile_and_load_kernel(kernel_code, ir_id)
            if kernel_module is None:
                return BenchmarkResult(ir_id, ir_expr, float('inf'), "Failed to compile kernel")
            
            # Benchmark kernel
            exec_time = self.benchmark_kernel(kernel_module=kernel_module, ir_id=ir_id)
            
            # Clean up GPU memory after each benchmark
            self.cleanup_gpu()
            
            # Also clean up the loaded module to prevent memory leaks
            module_name = f"llama_kernel_{ir_id}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            return BenchmarkResult(ir_id, ir_expr, exec_time)
            
        except Exception as e:
            # Clean up GPU memory even on error
            self.cleanup_gpu()
            return BenchmarkResult(ir_id, ir_expr, float('inf'), str(e))

    def run_all_benchmarks(
        self,
        ir_file: str,
        min_expressions: Optional[int],
        num: Optional[int] = None,
        optimized: bool = False,
        max_benchmarks: int = 64,
        seed_samples: int = 16,
        local_refine_budget: int = 0,
        local_refine_top_k: int = 1,
    ) -> List[BenchmarkResult]:
        """Run benchmarks for all IR expressions in the file."""
        # Parse IR expressions
        expressions = self.parse_ir_file(ir_file)
        
        # Filter by ir_id if min_expressions is provided
        if min_expressions is not None:
            # Find expressions with ir_id >= min_expressions
            filtered_expressions = [(ir_id, expr) for ir_id, expr in expressions if ir_id >= min_expressions]
            
            # If num is specified, take only the first 'num' expressions
            if num:
                filtered_expressions = filtered_expressions[:num]
            
            expressions = filtered_expressions
        
        print(f"Found {len(expressions)} IR expressions to benchmark")
        if optimized:
            return self.run_optimized_benchmarks(
                expressions,
                max_benchmarks=max_benchmarks,
                seed_samples=seed_samples,
                local_refine_budget=local_refine_budget,
                local_refine_top_k=local_refine_top_k,
            )
        
        results = []
        # tqdm progress bar with update every 10 items
        with tqdm(total=len(expressions), desc="Benchmarking", unit="IR") as pbar:
            for i, (ir_id, ir_expr) in enumerate(expressions):
                result = self.run_single_benchmark(ir_id, ir_expr)
                results.append(result)
                
                # Update progress bar
                pbar.update(1)
                
                # Update postfix with current status every 10 items
                if (i + 1) % 10 == 0:
                    valid_so_far = sum(1 for r in results if r.error is None)
                    pbar.set_postfix(valid=valid_so_far, errors=len(results)-valid_so_far)

        return results

    def save_results(self, results: List[BenchmarkResult], output_file: str):
        """Save benchmark results to a JSON file."""
        data = []
        for r in results:
            data.append({
                'ir_id': r.ir_id,
                'ir_expression': r.ir_expression,
                'execution_time_ms': r.execution_time,
                'error': r.error,
                'benchmarked': r.benchmarked,
                'source_ir_id': r.source_ir_id,
                'predicted_execution_time_ms': r.predicted_execution_time,
                'selection_reason': r.selection_reason,
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def cleanup_gpu(self):
        """Clean up GPU memory and reset CUDA context."""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                # Force synchronization
                torch.cuda.synchronize()
                
                # Clear all allocated tensors
                self.tensors.clear()
                
                # Multiple empty_cache calls to ensure thorough cleanup
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
                
                # Small delay to ensure GPU cleanup
                time.sleep(0.1)
                
                # Recreate test tensors to ensure clean state
                self.create_test_tensors()
            
        except Exception as e:
            print(f"Warning: GPU cleanup failed: {e}")

    def clear_triton_cache(self):
        """Clear Triton's cache directory to free up disk space."""
        try:
            # Get Triton cache directory
            triton_cache_dir = os.path.expanduser("~/.triton/cache")
            
            if os.path.exists(triton_cache_dir):
                # Remove all files and subdirectories in the cache
                for item in os.listdir(triton_cache_dir):
                    item_path = os.path.join(triton_cache_dir, item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                
                print(f"  Successfully cleared Triton cache at {triton_cache_dir}")
            else:
                print(f"  Triton cache directory not found at {triton_cache_dir}")
                
        except Exception as e:
            print(f"  Warning: Failed to clear Triton cache: {e}")

    def cleanup(self):
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self._temp_files = []


def run_comprehensive_benchmark(
    shapes_path,
    ir_file,
    start_expressions,
    num_expressions,
    output_file,
    optimized=False,
    max_benchmarks=64,
    seed_samples=16,
    local_refine_budget=0,
    local_refine_top_k=1,
    warmup_runs=10,
    benchmark_runs=100,
):
    """Run benchmarks for shapes.json."""
    all_results = []
    benchmark_instances = []

    print(f"Running comprehensive benchmark with:")
    print(f"  - shapes={shapes_path}")
    print()
    
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize output file with empty list
    with open(output_file, 'w') as f:
        json.dump([], f)
    
    benchmark = IRBenchmark(
        shapes_path=shapes_path,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
    )
    benchmark_instances.append(benchmark)
    
    try:
        # Run benchmarks for this tensor configuration
        results = benchmark.run_all_benchmarks(
            ir_file,
            min_expressions=start_expressions,
            num=num_expressions,
            optimized=optimized,
            max_benchmarks=max_benchmarks,
            seed_samples=seed_samples,
            local_refine_budget=local_refine_budget,
            local_refine_top_k=local_refine_top_k,
        )
        
        # Store results with configuration info
        config_results = {
            'shapes_path': shapes_path,
            'results': results
        }
        all_results.append(config_results)
        
        # Save results incrementally
        save_incremental_results(config_results, output_file)
        print("  Saved results")
        
    except Exception as e:
        print(f"  Error in configuration: {str(e)}")
        # Save error information
        error_result = {
            'shapes_path': shapes_path,
            'error': str(e),
            'results': []
        }
        all_results.append(error_result)
        save_incremental_results(error_result, output_file)
    
    # Clean up after each tensor configuration
    benchmark.cleanup()
    
    return all_results, benchmark_instances


def save_incremental_results(config_results, output_file):
    """Append results from one configuration to the output file."""
    # Read existing data
    try:
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
    except:
        existing_data = []
    
    # Add new results
    if 'error' in config_results and config_results['error']:
        # Handle error case
        existing_data.append({
            'shapes_path': config_results.get('shapes_path'),
            'error': config_results['error'],
            'results': []
        })
    else:
        # Add all results from this configuration
        for result in config_results['results']:
            existing_data.append({
                'ir_id': result.ir_id,
                'ir_expression': result.ir_expression,
                'execution_time_ms': result.execution_time,
                'shapes_path': config_results.get('shapes_path'),
                'error': result.error,
                'benchmarked': result.benchmarked,
                'source_ir_id': result.source_ir_id,
                'predicted_execution_time_ms': result.predicted_execution_time,
                'selection_reason': result.selection_reason,
            })
    
    # Write back to file
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=2)


def print_comprehensive_report(all_results, top_k):
    """Print comprehensive report showing best kernels for each configuration."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE BENCHMARK REPORT")
    print("=" * 100)
    
    # Group results by configuration
    for config_idx, config_result in enumerate(all_results):
        results = config_result['results']
        
        print(f"\nConfiguration {config_idx + 1}:")
        if "shapes_path" in config_result:
            print(f"  Shapes: {config_result['shapes_path']}")
        
        # Find best kernels for this configuration
        valid_results = [r for r in results if r.error is None and r.execution_time != float('inf')]
        valid_results.sort(key=lambda x: x.execution_time)
        best_kernels = valid_results[:top_k]
        
        if best_kernels:
            print(f"  Top {min(len(best_kernels), top_k)} kernels:")
            for i, result in enumerate(best_kernels):
                print(f"    {i + 1}. IR {result.ir_id}: {result.execution_time:.4f} ms")
                if i == 0:  # Show expression for best kernel only
                    print(f"       Expression: {result.ir_expression[:80]}...")
        else:
            print("  No valid kernels found for this configuration")


def main():
    """Main function to run IR benchmarks."""
    OUTPUT_FILE = "./profile_result/benchmark.json"
    START_EXPRESSIONS = 0
    NUM_EXPRESSIONS = 10
    TOP_K = 5

    parser = argparse.ArgumentParser(description="Run IR benchmarks")
    parser.add_argument('--ir', type=str, help="Path to the IR expressions file")
    parser.add_argument('--output', type=str, default=OUTPUT_FILE, help="Path to save benchmark results")
    parser.add_argument('--start', type=int, default=START_EXPRESSIONS, help="Start from test case ID")
    parser.add_argument('--num', type=int, default=NUM_EXPRESSIONS, help="Number of expressions to benchmark")
    parser.add_argument('--end', action='store_true', help="Run from start ID to the last test case")
    parser.add_argument('--topk', type=int, default=TOP_K, help="Number of top kernels to report")
    parser.add_argument('--all', action='store_true', help="Run all configurations comprehensively")
    parser.add_argument('--shapes', type=str, required=True, help="Path to shapes.json for tensor shapes")
    parser.add_argument('--optimized', action='store_true', help="Benchmark only representative IRs and predict nearby candidates")
    parser.add_argument('--max-benchmarks', type=int, default=64, help="Maximum number of representative IRs to benchmark")
    parser.add_argument('--seed-samples', type=int, default=16, help="Number of diverse seed representatives")
    parser.add_argument('--local-refine-budget', type=int, default=16, help="Extra benchmarks for nearest neighbors of the best representatives")
    parser.add_argument('--local-refine-top-k', type=int, default=1, help="How many top representatives to use as local refinement anchors")
    parser.add_argument('--warmup-runs', type=int, default=10, help="Number of warmup runs before measurement")
    parser.add_argument('--benchmark-runs', type=int, default=100, help="Number of timed graph replays per IR")

    args = parser.parse_args()
    
    # Validate conflicting arguments
    if args.end and args.num != NUM_EXPRESSIONS:
        print("Error: Cannot use --end and --num together. Use either --num or --end, not both.")
        return
    
    total_expressions = 0
    if args.all:
        with open(args.ir, 'r') as f:
            total_expressions = len(f.readlines())
    elif args.end:
        total_expressions = None  # None means no limit
    else:
        total_expressions = args.num

    shapes_path = args.shapes

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA device not available. Triton requires CUDA.")
        return
    
    # Run benchmarks
    print("Starting IR benchmarks...")
    all_results, _ = run_comprehensive_benchmark(
        shapes_path,
        args.ir,
        args.start,
        total_expressions,
        args.output,
        optimized=args.optimized,
        max_benchmarks=args.max_benchmarks,
        seed_samples=args.seed_samples,
        local_refine_budget=args.local_refine_budget,
        local_refine_top_k=args.local_refine_top_k,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    
    print(f"\nAll results saved to: {args.output}")
    print_comprehensive_report(all_results, args.topk)


if __name__ == "__main__":
    main()
