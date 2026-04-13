# Triton Generator Architecture

## Goal

`triton_generator` is organized around the current implementation reality:

- It generates Triton code strings directly from the AST.
- It does not have a separate lowered IR.
- Components collaborate through a shared `CodeGenState` and explicit cross-component calls via `self.gen`.

Because of that, the package is split by responsibility, not by an idealized
compiler pipeline that does not exist in this codebase. Thin package
initializers assemble focused helper modules into public components.

## Top-Level Layout

- `state.py`
  - Generator state initialization, counters, caches, debug state.
  - Precision 관련: `cast_expression`, `promote_dot_operands` (fp16/fp32 결정).
  - Store context 추적: `current_store_tensor`.
- `shape_utils.py`
  - Shape resolution, constant resolution, padding helpers.
- `analysis/`
  - Read-only AST/tensor analysis and planning.
  - `tensor_usage.py` — 텐서 수집 및 사용 패턴 분석.
  - `dependencies.py` — cross-kernel, cross-sloop 의존성 추적.
  - `accumulators.py` — accumulator 식별, fp32 텐서 식별, accumulator 초기화/저장 코드 생성.
  - `allocations.py` — 중간 텐서 할당 계획 및 코드 생성.
  - `__init__.py` assembles the public `Analyzer`.
- `codegen/`
  - Direct Triton string generation from AST nodes.
  - `dispatch.py` — AST 노드 타입별 중앙 라우터. 다른 codegen 모듈에 위임.
  - `loops.py` — parallel/sequential loop 코드 생성.
  - `indexing.py` — 텐서 인덱스 계산 및 shape 추론.
  - `memory_ops.py` — LOAD/STORE 처리, precision cast 결정, accumulator fp16 변환.
  - `math_ops.py` — 산술, 단항, cast, broadcast, reduction 연산.
  - `matmul_ops.py` — matmul, block matmul (concat 패턴) 처리.
  - `transforms.py` — shape 변환 (permute, transpose, unsqueeze, squeeze).
  - `expressions.py` — 표현식 조합, staged load 처리.
  - `masking.py` — 경계 초과 접근 방지 mask 생성.
  - `__init__.py` assembles codegen components.
- `kernel.py`
  - Kernel signature and autotune helpers.
- `pipeline/`
  - High-level orchestration for single-kernel, seq-kernel, and wrapper generation.
  - `entrypoint.py` — 메인 진입점, kernel 분할 결정.
  - `single_kernel.py` — 단일 커널 분석→코드생성 파이프라인.
  - `seq_kernels.py` — 다중 커널(SEQ) 순차 생성.
  - `wrapper.py` — Python wrapper 함수 및 메타데이터 생성.
  - `__init__.py` assembles the public `Pipeline`.

## Boundary Rules

### `analysis/`

Put code here when it answers questions such as:

- Which tensors are used?
- Which tensors cross kernel or loop boundaries?
- Which tensors are accumulators?
- What allocations or memory behavior are required?

`analysis/` should ideally be read-only (no Triton code emission).

**현재 경계 위반:**
- `accumulators.py`의 `generate_*()` 메서드 3개가 `tl.zeros`/`tl.store` 코드를 직접 생성.
- `allocations.py`의 `generate_intermediate_allocations()`가 `tl.zeros` 할당 코드를 생성.

이 codegen 로직은 향후 `codegen/` 하위로 이동하여 analysis는 순수 분석만 담당하도록 개선 필요.

### `codegen/`

Put code here when it directly emits Triton code strings from AST nodes or
generator state.

`dispatch.py`가 중앙 라우터 역할을 하며, AST 노드 타입에 따라 적절한 codegen 모듈로 위임한다.
각 모듈은 자기 책임의 Triton 코드만 생성하고, 다른 모듈의 영역은 `dispatch`를 통해 위임해야 한다.

`codegen/` is intentionally a single stage because this codebase does not have
a real `lowering -> emission` split today.

### `kernel.py`

Keep only kernel-level Triton concerns here:

- kernel signature/header
- autotune decorator generation

Do not place wrapper orchestration here.

### `pipeline/`

Put code here when it coordinates larger generation flows:

- entrypoint `generate()`
- single-kernel generation
- seq-kernel generation
- wrapper generation

`pipeline/` owns orchestration, not low-level AST emission details.

## Design Principles

1. Prefer direct names over compiler-theory names when the implementation is direct string emission.
2. Keep helper files small enough that a maintainer can understand one file in one sitting.
3. Shared state and shape helpers should stay out of `analysis/` and `codegen/` when possible.
4. New helpers should be placed by responsibility, not by caller convenience.
5. If a helper is not referenced internally and has no intended external use, remove it.
6. Keep the generated Triton code stable for representative benchmark cases.
7. Keep package `__init__.py` files thin: exports and component assembly only.

## Known Issues

### analysis/ 경계 위반

`accumulators.py`와 `allocations.py`에 codegen 로직이 포함되어 있음.
향후 `codegen/accumulator_ops.py`, `codegen/allocation_ops.py`로 분리하여
analysis는 "무엇을 할지 판단", codegen은 "코드 생성"으로 책임을 분리해야 함.

### 해결된 이슈

- `memory_ops.py`에 transform 로직(PERMUTE3, UNSQUEEZE 등) 290행이 inline으로 중복 구현되어 있었음.
  → `dispatch`를 통해 `transforms.py`에 위임하도록 수정 완료.

## Validation Rule

After structural changes, at minimum:

- import `backend.codegen.convert_module.convert_ir_to_triton`
- generate Triton for representative IR cases from `backend/results/`
- run at least one end-to-end caller such as `backend/run_eval.py`
