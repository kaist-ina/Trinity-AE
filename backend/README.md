# Trinity Backend

> Triton Code Generation and Evaluation

## Overview

Trinity Backend converts optimized IR expressions from the optimizer into executable Triton GPU kernels. It also provides evaluation scripts for benchmarking generated kernels against baselines.

## Directory Structure

```
backend/
├── codegen/                # Triton code generation
│   ├── TritonGen.py        # Main Triton code generator
│   ├── IrParser.py         # IR expression parser
│   ├── AstNode.py          # AST node definitions
│   └── NodeType.py         # Node type enums
├── evaluation/             # IR list & profiled results
├── figure67/               # Data for figure6 & 7
├── scripts/                # Evaluation scripts
│   ├── evaluate_all.sh     # Run all benchmarks
│   └── evaluate67.sh
├── profile/                # Profile IR list & find optimal
├── results/                # Generated benchmark results
├── run_eval.py             # Main evaluation entry point
├── baselines.py            # Baseline implementations
└── format.py               # Output formatting utilities
```

## Requirements

- Python 3.12+
- PyTorch 2.8+
- Triton 3.4+
- CUDA 12.x+
- [uv](https://docs.astral.sh/uv/) (Python package manager)

## Installation

### First-time Setup

Run the install script to build TVM Relax (with CUDA/cuBLAS support) and Mirage:

```bash
python install_all.py
```

This will:
1. Initialize git submodules for TVM
2. Build TVM Relax with CUDA, cuBLAS, and CUTLASS support
3. Build Mirage
4. Sync all Python dependencies
5. Set up TVM environment (install `.pth` file)

### After Initial Setup

For subsequent dependency updates, simply run:

```bash
uv sync
```

### Verify Installation

```bash
python -c "import tvm; print(tvm.__file__)"
python -c "import mirage; print(mirage.__file__)"
```

### Troubleshooting

If TVM import fails after reactivating the virtual environment:

```bash
# Re-install the .pth file
uv run setup-tvm

# Or rebuild everything
python install_all.py
```

## Usage

### Run for figure 4

```bash
# Run all benchmarks
# GPU: 5090, A100, H100
./scripts/evaluate_all.sh {GPU}
```

### Run for figure6&7
```bash
# Run all benchmarks
./scripts/evaluate67.sh
# Run single test
python run_figures67.py [options]

# Options:
#   --m        : Model type (llama, falcon)
#   --t        : Architecture (vanilla, prenorm, qknorm, keyformer, roco, ffn)
#   --n        : Case number for IR
#   --d        : CUDA device number (default: 0)
```

## IR List Profiling

Profile multiple IR expressions from a file to find the best performing kernels. Pre-generated IR expression files from the optimizer are available in `evaluation/{architecture}/` for each architecture and model.

### Usage
```bash
python profile/{method}_{model}_benchmark.py [options]

# Options:
#   --start   : Start from specific test case ID (default: 0)
#   --num     : Number of expressions to benchmark (default: 10)
#   --end     : Run from start ID to the last test case
#   --device  : CUDA device number (default: 0)
#   --topk    : Number of top kernels to report (default: 5)
#   --all     : Profiling all cases in the IR expressions file
```

### Examples
```bash
# Profile 512 IR expressions starting from ID 0
python profile/vanilla_llama_benchmark.py --num 512 --device 0

# Profile from ID 500 to the end
python profile/ffn_falcon_benchmark.py --start 500 --end --device 1

# Profile all IR expressions
# Please use this command to find the best IR
python profile/prenorm_llama_benchmark.py --all
```

- You can check top-k kernel result in `{method}_{model}_{topk}.json` in `evaluation` directory.

### Convert IR to Triton and Run Benchmark

```bash
python run_eval.py [options]

# Options:
#   --o        : 0 = Convert IR to Triton only
#                1 = Run benchmark only (requires pre-generated Triton code)
#                2 = Convert and run benchmark
#   --m        : Model type (llama, falcon)
#   --t        : Architecture (vanilla, prenorm, qknorm, keyformer, roco, ffn)
#   --n        : Case number for IR
#   --baseline : Baselines to compare (trinity, tensorrt, pytorch, inductor, flashinfer, flashtensor)
#                If not specified, all baselines are run by default
#   --d        : CUDA device number (default: 0)
#   --print_output : Print kernel output values (default: off)
```

### Examples

```bash
# Convert and benchmark llama vanilla attention with all baselines
python run_eval.py --o 2 --m llama --t vanilla --n 946

# Convert and benchmark falcon ffn (case 2248)
python run_eval.py --o 2 --m falcon --t ffn --n 2248

# Only convert IR to Triton code
python run_eval.py --o 0 --m llama --t prenorm --n 579

# Run benchmark with baseline comparison (torch inductor)
python run_eval.py --o 2 --m llama --t vanilla --n 946 --baseline inductor
```
### Input & Output
- Input IR expression file: `results/{method}/{method}_{model}_case{n}.txt`
- Output generated Triton code: `results/{method}/{method}_{model}_benchmark{n}.py`
- Benchmark results will print to stdout

## Model Configurations

| Model  | sequence length (M) | head dimension (D)  | hidden dimension (N)   | head number (H) |
|--------|----|----|------|-----|
| LLaMA  | 16 | 128 | 4096 | 32  |
| Falcon | 16 | 64  | 4544 | 71  |

You can modify tensor sizes by editing `model_configs.json`.

## Appendix

### IR Formatting Helper

`format.py` is a helper script that formats LISP-style IR expressions for better readability.

```bash
python format.py [options]

# Options:
#   --n : Case number to format
#   --m : Model type (llama, falcon)
#   --t : Architecture type (vanilla, prenorm, qknorm, keyformer, roco, ffn)
```

#### Example
```bash
# Format llama vanilla IR case 946
python format.py --n 946 --m llama --t vanilla
```

The formatted output will overwrite the original file at `results/{t}/{t}_{m}_case{n}.txt`.
