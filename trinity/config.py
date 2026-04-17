"""Public configuration objects for the Trinity package facade."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

# Project root for default output directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class OptimizeConfig:
    """Configuration for a Trinity optimization/compilation run."""

    basename: Optional[str] = None
    cost: int = 6
    kern: int = 1
    output_dir: Union[str, Path] = _PROJECT_ROOT / "trinity_output"
    device: int = 0
    inline_shape_op: bool = True
    inline_elementwise_op: bool = True
    remove_short_loop_threshold: int = 24
    decompose_nested_op_ratio: float = 0.0
    skip_frontend: bool = False
    skip_optimizer: bool = False
    skip_backend: bool = False
    verbose: bool = True
    fail_on_frontend_errors: bool = False
    optimizer_iter_limit: int = 8
    optimizer_timeout_s: int = 3600
    backend_timeout_s: int = 9600
    backend_optimized_benchmark: bool = True
    backend_max_benchmarks: int = 64
    backend_seed_samples: int = 16
    backend_local_refine_budget: int = 16
    backend_local_refine_top_k: int = 1
    backend_warmup_runs: int = 10
    backend_benchmark_runs: int = 100
    backend_cuda_graph: bool = False
    shape_vars: Optional[dict] = None  # {pytorch_name: [sym, sym, ...]} for axis naming

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
