# Trinity Project \[ASPLOS 2026\]

> Three-Dimensional Tensor Program Optimization via Tile-level Equality Saturation

Trinity is the first tensor program optimizer that achieves scalable joint optimization through tile-level equality saturation. Trinity's IR can capture the essence of all three optimization axes (algebraic equivalence, memory I/O, compute orchestration). By leveraging equality saturation, Trinity enables scalable joint optimization across the entire graph.

## Project Structure

```
Trinity-AE/
├── frontend/       # PyTorch model → Trinity IR conversion
├── optimizer/      # Tile-level equality saturation (Rust)
├── backend/        # IR → Triton kernel generation & profiling
├── trinity/        # End-to-end pipeline automation
├── scripts/        # Model definitions & run example pipeline
├── standard_ir/    # Pre-generated Trinity IR (ir.txt, shapes.json) per benchmark
└── third_party/    # Vendored TVM wheels (third_party/wheels/)
```

For detailed documentation, see the README in each directory:
- [`scripts/`](scripts/README.md) — Setup, usage guide, API reference, and config options
- [`frontend/`](frontend/README.md) — IR conversion details
- [`optimizer/`](optimizer/README.md) — Equality saturation engine
- [`backend/`](backend/README.md) — Triton code generation and profiling

## Setup Note

The frontend now expects a vendored CPU TVM wheel under `third_party/wheels/`.
Run `./scripts/setup.sh` after placing the wheel there.


## Known Issues and Limitations

The current artifact has the following known issues and limitations. We are working to address these issues and will update this repository with fixes soon. Please review the notes below before running experiments.

### 1. Attention support is limited to decoding without masking

The current implementation does not support attention masking, including the causal masking required for autoregressive prefill. Attention support in this artifact is therefore limited to decoding configurations that do not require masking. Prefill and other configurations that require an attention mask are currently unsupported.

### 2. Limited support for GEMV and small matrix dimensions

Our current Triton backend relies on `tl.dot()` for matrix multiplication and requires tile dimensions of at least 16 along the matrix multiplication axes. Consequently, workloads that do not map well to this execution path, such as unbatched matrix–vector multiplication (GEMV), are not well supported. This is a limitation of the current Trinity backend.

### 3. Frontend correctness issues

The frontend currently has known correctness issues and may not generate correct Trinity IR for all models. We recommend setting `skip_frontend=True` and providing Trinity IR directly rather than relying on automatic frontend conversion.