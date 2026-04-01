#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TVM_WHEEL_DIR="$ROOT_DIR/third_party/wheels"
py_tag="cp$(python - <<'PY'
import sys
print(f"{sys.version_info.major}{sys.version_info.minor}")
PY
)"

main_wheel="$TVM_WHEEL_DIR/mlc_ai_nightly_cpu-0.20.dev908-py3-none-manylinux_2_28_x86_64.whl"
ffi_wheel="$TVM_WHEEL_DIR/apache_tvm_ffi-0.1.9-${py_tag}-${py_tag}-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"

installable_wheels=("$ffi_wheel" "$main_wheel")

for wheel_path in "${installable_wheels[@]}"; do
  if [[ ! -f "$wheel_path" ]]; then
    echo "Error: missing required vendored wheel: $wheel_path" >&2
    exit 1
  fi
done

printf 'Installing vendored wheels for %s:\n' "$py_tag"
printf '  %s\n' "${installable_wheels[@]}"
python -m pip install --force-reinstall "${installable_wheels[@]}"
