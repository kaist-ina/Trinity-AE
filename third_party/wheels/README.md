# Vendored TVM Wheels

Place the Trinity frontend TVM wheel set in this directory.

Guidelines:
- Prefer a CPU-only wheel to avoid CUDA and driver coupling in the frontend.
- Keep the repo-pinned main TVM wheel and the matching `apache-tvm-ffi` wheel.
- Use a filename that encodes Python and platform compatibility.
- Optionally commit a matching `.sha256` file beside the wheel.

Examples:
- `mlc_ai_nightly_cpu-0.20.dev908-py3-none-manylinux_2_28_x86_64.whl`
- `apache_tvm_ffi-0.1.9-cp311-cp311-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl`
- `mlc_ai_nightly_cpu-0.20.dev908-py3-none-manylinux_2_28_x86_64.whl.sha256`

Install manually:

```bash
./scripts/install_tvm_wheel.sh
```
