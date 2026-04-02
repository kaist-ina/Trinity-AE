import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

import torch

import trinity
from scripts.gqa_vanilla import GQAVanilla
from scripts.keyformer import KeyformerAttn
from scripts.prenorm import PreNormAttn
from scripts.qknorm import QKNormAttn
from scripts.roco import build_model_and_inputs as build_roco_model_and_inputs
from scripts.vanilla import Vanilla


def ensure_rust_on_path() -> None:
    cargo_bin = Path.home() / ".cargo" / "bin"
    path = os.environ.get("PATH", "")
    parts = path.split(os.pathsep) if path else []
    cargo_str = str(cargo_bin)
    if cargo_str not in parts:
        os.environ["PATH"] = cargo_str if not path else f"{cargo_str}{os.pathsep}{path}"


def build_vanilla() -> tuple[Any, Any, str, dict[str, Any]]:
    m, n, d, h, p = 16, 4096, 128, 32, 1008
    x = torch.randn((m, n))
    k_cache = torch.randn((h, p + m, d))
    v_cache = torch.randn((h, p + m, d))
    model = Vanilla(m, n, d, p, k_cache, v_cache)
    return model, x, "vanilla", {}


def build_gqa_vanilla() -> tuple[Any, Any, str, dict[str, Any]]:
    m, qh, d, p = 16, 32, 128, 1008
    n = qh * d
    x = torch.randn((m, n))
    k_cache = torch.randn((qh, p + m, d))
    v_cache = torch.randn((qh, p + m, d))
    model = GQAVanilla(m, qh, d, p, k_cache, v_cache)
    return model, x, "gqa_vanilla", {}


def build_keyformer() -> tuple[Any, Any, str, dict[str, Any]]:
    m, h, d, p = 16, 32, 128, 1024
    n = h * d
    x = torch.randn((m, n))
    k_cache = torch.randn((h, p, d))
    v_cache = torch.randn((h, p, d))
    tau = torch.tensor(1.0)
    noise = torch.randn((h, m, p + m))
    model = KeyformerAttn(m, h, d, p, k_cache, v_cache, tau, noise)
    return model, x, "keyformer", {}


def build_prenorm() -> tuple[Any, Any, str, dict[str, Any]]:
    m, h, d, p = 16, 32, 128, 1024
    n = h * d
    x = torch.randn((m, n))
    k_cache = torch.randn((h, p, d))
    v_cache = torch.randn((h, p, d))
    model = PreNormAttn(m, h, d, p, k_cache, v_cache)
    return model, x, "prenorm", {}


def build_qknorm() -> tuple[Any, Any, str, dict[str, Any]]:
    m, h, d, p = 16, 32, 128, 1024
    n = h * d
    x = torch.randn((m, n))
    k_cache = torch.randn((h, p, d))
    v_cache = torch.randn((h, p, d))
    model = QKNormAttn(m, h, d, p, k_cache, v_cache)
    return model, x, "qknorm", {}


def build_roco() -> tuple[Any, Any, str, dict[str, Any]]:
    cfg = build_roco_model_and_inputs()
    optimize_kwargs = {
        "inline_shape_op": cfg["inline_shape_op"],
        "inline_elementwise_op": cfg["inline_elementwise_op"],
        "remove_short_loop_threshold": cfg["remove_short_loop_threshold"],
        "decompose_nested_op_ratio": cfg["decompose_nested_op_ratio"],
    }
    return cfg["model"], cfg["example_inputs"], "roco", optimize_kwargs


MODEL_BUILDERS = {
    "gqa_vanilla": build_gqa_vanilla,
    "keyformer": build_keyformer,
    "pernorm": build_prenorm,
    "qknorm": build_qknorm,
    "roco": build_roco,
    "vanilla": build_vanilla,
}
MODEL_ORDER = tuple(MODEL_BUILDERS.keys())


def build_workloads() -> list[dict[str, Any]]:
    workloads: list[dict[str, Any]] = []
    for index, model_name in enumerate(MODEL_ORDER):
        false_gpu = index % 2
        true_gpu = 1 - false_gpu
        workloads.append(
            {
                "model_name": model_name,
                "skip_frontend": False,
                "gpu": false_gpu,
                "seed": 1000 + index * 2,
            }
        )
        workloads.append(
            {
                "model_name": model_name,
                "skip_frontend": True,
                "gpu": true_gpu,
                "seed": 1001 + index * 2,
            }
        )
    return workloads


def serialize_result(result: Any) -> dict[str, Any]:
    frontend = None
    if result.frontend is not None:
        frontend = {
            "ir_path": str(result.frontend.ir_path),
            "shapes_path": str(result.frontend.shapes_path),
            "source_ir_path": str(result.frontend.source_ir_path),
            "source_shapes_path": str(result.frontend.source_shapes_path),
            "tensor_shapes": {key: list(value) for key, value in result.frontend.tensor_shapes.items()},
            "errors": list(result.frontend.errors),
        }

    optimizer = None
    if result.optimizer is not None:
        optimizer = {
            "expressions_path": str(result.optimizer.expressions_path),
            "source_expressions_path": str(result.optimizer.source_expressions_path),
            "semi_expressions_path": (
                str(result.optimizer.semi_expressions_path)
                if result.optimizer.semi_expressions_path is not None
                else None
            ),
            "source_semi_expressions_path": (
                str(result.optimizer.source_semi_expressions_path)
                if result.optimizer.source_semi_expressions_path is not None
                else None
            ),
            "expression_count": result.optimizer.expression_count,
            "stdout_log_path": (
                str(result.optimizer.stdout_log_path)
                if result.optimizer.stdout_log_path is not None
                else None
            ),
            "stderr_log_path": (
                str(result.optimizer.stderr_log_path)
                if result.optimizer.stderr_log_path is not None
                else None
            ),
        }

    backend = None
    if result.backend is not None:
        backend = {
            "benchmark_path": (
                str(result.backend.benchmark_path) if result.backend.benchmark_path is not None else None
            ),
            "source_benchmark_path": (
                str(result.backend.source_benchmark_path)
                if result.backend.source_benchmark_path is not None
                else None
            ),
            "all_results": result.backend.all_results,
            "best_kernel_path": (
                str(result.backend.best_kernel_path) if result.backend.best_kernel_path is not None else None
            ),
            "best_execution_time_ms": result.backend.best_execution_time_ms,
            "best_ir_id": result.backend.best_ir_id,
            "best_ir_expression": result.backend.best_ir_expression,
        }

    return {
        "execution_time_ms": result.execution_time_ms,
        "ir_id": result.ir_id,
        "ir_expression": result.ir_expression,
        "kernel_path": str(result.kernel_path) if result.kernel_path is not None else None,
        "workspace": str(result.workspace) if result.workspace is not None else None,
        "frontend": frontend,
        "optimizer": optimizer,
        "backend": backend,
    }


def run_single_workload(spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    model_name = spec["model_name"]
    builder = MODEL_BUILDERS[model_name]

    torch.manual_seed(spec["seed"])
    model, example_inputs, basename, extra_optimize_kwargs = builder()

    optimize_kwargs = {
        "basename": basename,
        "skip_frontend": spec["skip_frontend"],
        "device": spec["gpu"],
        "verbose": True,
        "output_dir": output_dir,
    }
    optimize_kwargs.update(extra_optimize_kwargs)

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        result = trinity.optimize(model, example_inputs, **optimize_kwargs)
        finished_at = datetime.now(timezone.utc).isoformat()
        return {
            "model_name": model_name,
            "basename": basename,
            "skip_frontend": spec["skip_frontend"],
            "gpu": spec["gpu"],
            "seed": spec["seed"],
            "status": "ok",
            "started_at": started_at,
            "finished_at": finished_at,
            "result": serialize_result(result),
        }
    except Exception as exc:
        finished_at = datetime.now(timezone.utc).isoformat()
        return {
            "model_name": model_name,
            "basename": basename,
            "skip_frontend": spec["skip_frontend"],
            "gpu": spec["gpu"],
            "seed": spec["seed"],
            "status": "error",
            "started_at": started_at,
            "finished_at": finished_at,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_worker(gpu_id: int, assignment_path: Path, results_path: Path, output_dir: Path) -> int:
    ensure_rust_on_path()
    specs = json.loads(assignment_path.read_text())
    results = []
    for index, spec in enumerate(specs, start=1):
        print(
            f"[worker gpu={gpu_id}] starting {index}/{len(specs)}: "
            f"{spec['model_name']} skip_frontend={spec['skip_frontend']}",
            flush=True,
        )
        result = run_single_workload(spec, output_dir)
        results.append(result)
        print(
            f"[worker gpu={gpu_id}] finished {index}/{len(specs)}: "
            f"{spec['model_name']} skip_frontend={spec['skip_frontend']} status={result['status']}",
            flush=True,
        )

    payload = {
        "gpu": gpu_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(item["status"] == "ok" for item in results) else 1


def aggregate_results(run_dir: Path, worker_results: list[Path], workloads: list[dict[str, Any]]) -> Path:
    results = []
    for path in worker_results:
        if path.exists():
            payload = json.loads(path.read_text())
            results.extend(payload["results"])

    ordered = []
    for workload in workloads:
        matched = None
        for result in results:
            if (
                result["model_name"] == workload["model_name"]
                and result["skip_frontend"] == workload["skip_frontend"]
                and result["gpu"] == workload["gpu"]
                and result["seed"] == workload["seed"]
            ):
                matched = result
                break
        if matched is None:
            matched = {
                "model_name": workload["model_name"],
                "skip_frontend": workload["skip_frontend"],
                "gpu": workload["gpu"],
                "seed": workload["seed"],
                "status": "missing",
                "error_type": "MissingResult",
                "error_message": "Worker did not produce a result entry for this workload.",
            }
        ordered.append(matched)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "total_workloads": len(workloads),
        "successful_workloads": sum(1 for item in ordered if item["status"] == "ok"),
        "failed_workloads": sum(1 for item in ordered if item["status"] != "ok"),
        "results": ordered,
    }
    summary_path = run_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary_path


def run_coordinator(output_root: Path, output_dir: Path) -> int:
    ensure_rust_on_path()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    workloads = build_workloads()
    assignments = {
        0: [spec for spec in workloads if spec["gpu"] == 0],
        1: [spec for spec in workloads if spec["gpu"] == 1],
    }

    (run_dir / "manifest.json").write_text(json.dumps({"workloads": workloads}, indent=2, sort_keys=True))

    worker_assignment_paths = {}
    worker_result_paths = {}
    for gpu_id, specs in assignments.items():
        assignment_path = run_dir / f"gpu{gpu_id}_assignments.json"
        assignment_path.write_text(json.dumps(specs, indent=2, sort_keys=True))
        worker_assignment_paths[gpu_id] = assignment_path
        worker_result_paths[gpu_id] = run_dir / f"gpu{gpu_id}_results.json"

    processes = []
    for gpu_id in (0, 1):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--gpu-id",
            str(gpu_id),
            "--assignment-file",
            str(worker_assignment_paths[gpu_id]),
            "--results-file",
            str(worker_result_paths[gpu_id]),
            "--output-dir",
            str(output_dir),
        ]
        processes.append(
            subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=os.environ.copy(),
            )
        )

    exit_codes = [process.wait() for process in processes]
    summary_path = aggregate_results(run_dir, list(worker_result_paths.values()), workloads)
    print(f"Saved aggregate results to {summary_path}", flush=True)
    return 0 if all(code == 0 for code in exit_codes) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Trinity attention workloads across GPU 0 and 1 and save JSON results."
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "trinity_output" / "batch_runs"),
        help="Directory used for batch-run metadata and result JSON files.",
    )
    parser.add_argument("--worker", action="store_true", help="Internal worker mode.")
    parser.add_argument("--gpu-id", type=int, help="GPU assigned to the worker.")
    parser.add_argument("--assignment-file", type=str, help="Path to the JSON workload assignment file.")
    parser.add_argument("--results-file", type=str, help="Path to the JSON worker result file.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "trinity_output"),
        help="Directory passed through to trinity.optimize output_dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if args.worker:
        if args.gpu_id is None or args.assignment_file is None or args.results_file is None:
            raise ValueError("Worker mode requires --gpu-id, --assignment-file, and --results-file.")
        return run_worker(
            gpu_id=args.gpu_id,
            assignment_path=Path(args.assignment_file).resolve(),
            results_path=Path(args.results_file).resolve(),
            output_dir=output_dir,
        )

    return run_coordinator(output_root, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
