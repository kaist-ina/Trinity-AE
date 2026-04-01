#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV_NAME="trinity"

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' is not available." >&2
    exit 1
  fi
}

setup_conda() {
  require_command conda

  # Load conda shell functions in non-interactive shells.
  eval "$(conda shell.bash hook)"

  if conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    echo "[frontend] Reusing existing conda environment: $CONDA_ENV_NAME"
  else
    echo "[frontend] Creating conda environment: $CONDA_ENV_NAME"
    conda env create -n "$CONDA_ENV_NAME" -f "$ROOT_DIR/frontend/environment.yml"
  fi

  echo "[frontend] Activating conda environment: $CONDA_ENV_NAME"
  conda activate "$CONDA_ENV_NAME"
}

setup_backend() {
  echo "[backend] Installing Python dependencies"
  python -m pip install -r "$ROOT_DIR/backend/requirements.txt"
}

setup_optimizer() {
  require_command sudo
  require_command apt

  echo "[optimizer] Installing system packages via apt"
  sudo apt update
  sudo apt install -y \
    build-essential \
    clang \
    libclang-dev \
    llvm-dev \
    libz3-dev \
    pkg-config
}

main() {
  setup_conda
  echo "[frontend] Installing vendored TVM wheel"
  "$ROOT_DIR/scripts/install_tvm_wheel.sh"
  setup_backend
  setup_optimizer
  echo "Setup complete."
  echo "Run 'conda activate $CONDA_ENV_NAME' in your shell before using Trinity."
}

main "$@"
