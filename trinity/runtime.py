"""Runtime helpers for the Trinity package facade."""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
OPTIMIZER_DIR = PROJECT_ROOT / "optimizer"
BACKEND_DIR = PROJECT_ROOT / "backend"
STANDARD_IR_DIR = PROJECT_ROOT / "standard_ir"


def log(message: str, verbose: bool) -> None:
    if verbose:
        print(f"[Trinity] {message}")


@contextmanager
def prepend_sys_path(path: Path) -> Iterator[None]:
    path_str = str(path)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


def load_shapes(shapes_path: Path) -> Dict[str, Tuple[int, ...]]:
    with shapes_path.open("r") as handle:
        data = json.load(handle)

    shapes: Dict[str, Tuple[int, ...]] = {}
    for name, value in data.items():
        dims = value.get("shape") if isinstance(value, dict) else value
        if isinstance(dims, list) and all(isinstance(dim, int) for dim in dims):
            shapes[name] = tuple(dims)
    return shapes


def count_expressions(expressions_path: Path) -> int:
    count = 0
    with expressions_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            prefix, _ = line.split(":", 1)
            try:
                int(prefix)
            except ValueError:
                continue
            count += 1
    return count


def load_kernel_from_file(kernel_path: Path) -> Optional[callable]:
    spec = importlib.util.spec_from_file_location("trinity_best_kernel", str(kernel_path))
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(kernel_path)
    sys.modules["trinity_best_kernel"] = module
    spec.loader.exec_module(module)
    return getattr(module, "forward", None)


def build_cuda_visible(device: int) -> str:
    return ",".join(str(device + offset) for offset in range(3))
