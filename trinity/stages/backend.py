"""Backend wrapper that keeps the original backend code untouched."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..artifacts import BackendArtifacts, FrontendArtifacts, OptimizerArtifacts
from ..config import OptimizeConfig
from ..runtime import BACKEND_DIR, build_cuda_visible, load_kernel_from_file, load_shapes, log, prepend_sys_path
from ..workspace import PipelineWorkspace


class BackendStage:
    def run(
        self,
        basename: str,
        frontend: FrontendArtifacts,
        optimizer: OptimizerArtifacts,
        config: OptimizeConfig,
        workspace: PipelineWorkspace,
    ) -> BackendArtifacts:
        log("Backend: profiling kernels ...", config.verbose)

        benchmark_path = (workspace.backend_dir / f"{optimizer.expressions_path.stem}_benchmark.json").resolve()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = build_cuda_visible(config.device)

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "profile.benchmark",
                "--ir",
                str(optimizer.expressions_path.resolve()),
                "--shapes",
                str(frontend.shapes_path.resolve()),
                "--all",
                "--e2e",
                "--output",
                str(benchmark_path),
            ]
            + (
                [
                    "--optimized",
                    "--max-benchmarks",
                    str(config.backend_max_benchmarks),
                    "--seed-samples",
                    str(config.backend_seed_samples),
                    "--local-refine-budget",
                    str(config.backend_local_refine_budget),
                    "--local-refine-top-k",
                    str(config.backend_local_refine_top_k),
                    "--warmup-runs",
                    str(config.backend_warmup_runs),
                    "--benchmark-runs",
                    str(config.backend_benchmark_runs),
                ]
                if config.backend_optimized_benchmark
                else [
                    "--warmup-runs",
                    str(config.backend_warmup_runs),
                    "--benchmark-runs",
                    str(config.backend_benchmark_runs),
                ]
            )
            + (
                ["--cuda-graph"] if config.backend_cuda_graph else []
            ),
            cwd=str(BACKEND_DIR),
            env=env,
            capture_output=not config.verbose,
            text=True,
            timeout=config.backend_timeout_s,
        )

        workspace.write_text(workspace.logs_dir, "backend.stdout.log", result.stdout or "")
        workspace.write_text(workspace.logs_dir, "backend.stderr.log", result.stderr or "")

        if result.returncode != 0 and not benchmark_path.exists():
            err_msg = result.stderr or result.stdout if not config.verbose else ""
            raise RuntimeError(f"Backend profiling failed:\n{err_msg}")

        with benchmark_path.open("r") as handle:
            all_results = json.load(handle)

        return self._finalize(
            basename=basename,
            frontend=frontend,
            benchmark_path=benchmark_path,
            source_benchmark_path=benchmark_path,
            all_results=all_results,
            workspace=workspace,
            verbose=config.verbose,
        )

    def load_existing(
        self,
        basename: str,
        frontend: FrontendArtifacts,
        optimizer: OptimizerArtifacts,
        workspace: PipelineWorkspace,
        verbose: bool,
    ) -> BackendArtifacts:
        benchmark_path = workspace.backend_dir / f"{optimizer.expressions_path.stem}_benchmark.json"
        source_benchmark_path = (
            optimizer.source_expressions_path.parent
            / f"{optimizer.source_expressions_path.stem}_benchmark.json"
        )

        if not benchmark_path.exists():
            if source_benchmark_path.exists():
                benchmark_path = workspace.mirror_file(source_benchmark_path, workspace.backend_dir)
            else:
                return BackendArtifacts(
                    benchmark_path=None,
                    source_benchmark_path=None,
                    all_results=[],
                )

        with benchmark_path.open("r") as handle:
            all_results = json.load(handle)

        return self._finalize(
            basename=basename,
            frontend=frontend,
            benchmark_path=benchmark_path,
            source_benchmark_path=source_benchmark_path if source_benchmark_path.exists() else benchmark_path,
            all_results=all_results,
            workspace=workspace,
            verbose=verbose,
        )

    def _finalize(
        self,
        *,
        basename: str,
        frontend: FrontendArtifacts,
        benchmark_path: Path,
        source_benchmark_path: Path,
        all_results: list[dict],
        workspace: PipelineWorkspace,
        verbose: bool,
    ) -> BackendArtifacts:
        valid = [
            result
            for result in all_results
            if result.get("error") is None
            and result.get("execution_time_ms") is not None
            and result["execution_time_ms"] != float("inf")
        ]

        benchmarked_valid = [result for result in valid if result.get("benchmarked", True)]

        if not benchmarked_valid and not valid:
            log("No valid kernels found.", verbose)
            return BackendArtifacts(
                benchmark_path=benchmark_path,
                source_benchmark_path=source_benchmark_path,
                all_results=all_results,
            )

        best_pool = benchmarked_valid or valid
        best = min(best_pool, key=lambda result: result["execution_time_ms"])

        with prepend_sys_path(BACKEND_DIR):
            from codegen.convert_module import convert_ir_to_triton  # type: ignore

            triton_code = convert_ir_to_triton(
                best["ir_expression"],
                load_shapes(frontend.shapes_path),
            )

        kernel_path = workspace.kernels_dir / "best_kernel.py"
        kernel_path.write_text(triton_code)
        kernel_fn = load_kernel_from_file(kernel_path)

        log(
            f"Best: IR #{best['ir_id']}, {best['execution_time_ms']:.4f} ms -> {kernel_path}",
            verbose,
        )

        return BackendArtifacts(
            benchmark_path=benchmark_path,
            source_benchmark_path=source_benchmark_path,
            all_results=all_results,
            best_kernel_path=kernel_path,
            best_kernel=kernel_fn,
            best_execution_time_ms=best["execution_time_ms"],
            best_ir_id=best["ir_id"],
            best_ir_expression=best["ir_expression"],
        )
