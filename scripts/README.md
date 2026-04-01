## Quick Start

### Setup

Run the repository setup script from the project root:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
conda activate trinity
```

What it does:
- creates the `trinity` conda environment from `frontend/environment.yml` if it does not already exist
- installs the vendored TVM wheel from `third_party/wheels/`
- installs backend Python dependencies from `backend/requirements.txt`
- installs optimizer system dependencies with `apt`

The script uses `sudo` for the optimizer packages, so it may prompt for your password.

### Vendored TVM Wheel

Place the vendored TVM wheel set in `third_party/wheels/` before running setup.
For the current repository pin, that means:
- `mlc_ai_nightly_cpu-0.20.dev908-py3-none-manylinux_2_28_x86_64.whl`
- `apache_tvm_ffi-0.1.10rc1-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`

```bash
ls third_party/wheels/
# example: mlc_ai_nightly_cpu-0.20.dev908-py3-none-manylinux_2_28_x86_64.whl
# example: apache_tvm_ffi-0.1.10rc1-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

The installer is pinned to the vendored filenames above.

### Usage

Define a PyTorch model, then call `trinity.optimize()`:

```python
import torch
import torch.nn as nn
import trinity

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4096, 4096, bias=False)

    def forward(self, X):
        return self.fc(X)

model = MyModel()
X = torch.randn(16, 4096)

result = trinity.optimize(model, X, basename="my_model")

# Access results
print(result.execution_time_ms)   # Best kernel execution time
kernel = result.kernel             # Compiled Triton kernel
```

### Running Existing Models

```bash
# Run from project root
python scripts/vanilla.py
python scripts/prenorm.py
python scripts/ffn.py
```

> **Note:** The frontend is currently under development and does not yet produce correct IR for all models. To work around this, pre-generated IR and shapes (`ir.txt`, `shapes.json`) from `optimizer/tests/` are placed in `trinity_output/{basename}/frontend/`. All scripts run with `skip_frontend=True` to use these pre-generated files instead of the frontend stage.

### Config Options

All options are passed as keyword arguments to `trinity.optimize()`:

| Option | Default | Description |
|--------|---------|-------------|
| `basename` | `model.__class__.__name__` | Model name for output directory |
| `cost` | `6` | Cost threshold for equality saturation |
| `kern` | `1` | Kernel partition parameter |
| `device` | `0` | CUDA device ID |
| `skip_frontend` | `False` | Skip IR generation (reuse existing) |
| `skip_optimizer` | `False` | Skip equality saturation (reuse existing) |
| `skip_backend` | `False` | Skip kernel profiling (reuse existing) |
| `verbose` | `True` | Enable logging |
| `optimizer_iter_limit` | `8` | Max equality saturation iterations |
| `optimizer_timeout_s` | `3600` | Optimizer timeout (seconds) |
| `backend_timeout_s` | `7200` | Backend profiling timeout (seconds) |

### Pipeline Stages

`trinity.optimize()` runs three stages sequentially:

1. **Frontend** — Traces the PyTorch model and converts it to Trinity IR (`ir.txt`, `shapes.json`)
2. **Optimizer** — Applies tile-level equality saturation to explore equivalent IR expressions (Rust)
3. **Backend** — Generates Triton kernels from IR expressions, profiles them, and selects the best

Each stage can be skipped via `skip_frontend`, `skip_optimizer`, `skip_backend` to reuse previous results.

### Output Structure

Results are saved under `trinity_output/{basename}/`:

```
{basename}/
├── config.json       # Pipeline configuration
├── frontend/         # IR and tensor shapes
├── optimizer/        # Optimized IR expressions
├── backend/          # Benchmark results
├── kernels/          # Best Triton kernel
└── logs/             # Process logs
```
