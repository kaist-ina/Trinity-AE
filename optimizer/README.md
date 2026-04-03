# Trinity Optimizer

> Tile-level Equality Saturation for Tensor Program Optimization

## Overview

Trinity Optimizer generates optimized IR candidates for tensor programs using e-graph based equality saturation(egg library). It explores various optimization transformations by applying tile-level equivalence rules.

## Directory Structure

```
optimizer/
├── src/                    # Core library code
│   ├── lib.rs              # Library entry point
│   ├── language.rs         # TileLang DSL definition
│   ├── rules.rs            # Equality saturation rules
│   ├── cost.rs             # Cost model
│   ├── extract.rs          # Expression extraction
│   ├── shape.rs            # Tensor shape tracking
│   └── ...
├── tests/                  # Model-specific test cases
│   ├── vanilla.rs          # Vanilla attention
│   ├── prenorm.rs          # Pre-normalization
│   ├── qknorm.rs           # QK normalization
│   ├── keyformer.rs        # Keyformer
│   ├── roco.rs             # RoCo
│   ├── ffn.rs              # Feed-forward network
│   └── ...
├── expressions/            # Generated expression outputs
│   └── semi/               # Intermediate results (JSON)
├── egg/                    # egg library for equality saturation(local fork)
└── benchmark/              # Benchmark related files
```

## Usage

### Install Dependency
```bash
sudo apt-get install libz3-dev
```

### Run All Tests

Run all combinations of models (falcon, llama) and methods (vanilla, prenorm, qknorm, keyformer, roco, ffn):

```bash
./run_all_optimizer.sh
```

### Run Single Test

Run a specific test:

```bash
# Basic format
cargo test --test {test_file} {test_function} -- --nocapture

# Example: llama vanilla attention
cargo test --test vanilla llama_extract_rmsnorm_qkv_attn_expressions -- --nocapture

# Example: falcon ffn
cargo test --test ffn falcon_extract_ffn_expressions -- --nocapture
```
### Suppress Warnings

```bash
RUSTFLAGS="-A warnings" cargo test --test vanilla -- --nocapture
```

## Test Cases

| Model  | Method    | Test Function                              |
|--------|-----------|-------------------------------------------|
| llama  | vanilla   | `llama_extract_rmsnorm_qkv_attn_expressions` |
| falcon | vanilla   | `falcon_extract_rmsnorm_qkv_attn_expressions` |
| llama  | prenorm   | `llama_extract_rmsnorm_qkv_attn_expressions` |
| falcon | prenorm   | `falcon_extract_rmsnorm_qkv_attn_expressions` |
| llama  | qknorm    | `llama_extract_rmsnorm_qkv_attn_expressions` |
| falcon | qknorm    | `falcon_extract_rmsnorm_qkv_attn_expressions` |
| llama  | keyformer | `llama_extract_rmsnorm_qkv_attn_expressions` |
| falcon | keyformer | `falcon_extract_rmsnorm_qkv_attn_expressions` |
| llama  | roco      | `llama_extract_rmsnorm_qkv_attn_expressions` |
| falcon | roco      | `falcon_extract_rmsnorm_qkv_attn_expressions` |
| llama  | ffn       | `llama_extract_ffn_expressions`            |
| falcon | ffn       | `falcon_extract_ffn_expressions`           |

## Output

Test results are saved in the `expressions/` directory:
- `expressions/semi/*.json`: Intermediate results (semi-extracted expressions)
- `expressions/*.txt`: Final processed expression list
- `backend/evaluation/{method}/*.txt`: Pre-generated expression lists for profile execution

Use `backend/profile/{method}_{model}_benchmark.py` to find the optimal IR from the generated expression list. See `backend/README.md` for details.
