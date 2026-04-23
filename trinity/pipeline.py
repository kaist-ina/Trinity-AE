"""Public Trinity pipeline facade with a torch.compile-style entrypoint."""

from __future__ import annotations

import time
from typing import Any, Optional

from .artifacts import BackendArtifacts, OptimizeResult
from .config import OptimizeConfig
from .runtime import log
from .stages import BackendStage, FrontendStage, OptimizerStage
from .workspace import PipelineWorkspace


class TrinityPipeline:
    """Facade pipeline that orchestrates the original frontend/optimizer/backend engines."""

    def __init__(
        self,
        *,
        frontend_stage: Optional[FrontendStage] = None,
        optimizer_stage: Optional[OptimizerStage] = None,
        backend_stage: Optional[BackendStage] = None,
    ):
        self.frontend_stage = frontend_stage or FrontendStage()
        self.optimizer_stage = optimizer_stage or OptimizerStage()
        self.backend_stage = backend_stage or BackendStage()

    def optimize(
        self,
        model: Any,
        example_inputs: Any,
        *,
        config: OptimizeConfig,
    ) -> OptimizeResult:
        source_basename = config.basename or model.__class__.__name__
        workspace_basename = (
            f"{source_basename}_skip_fe" if config.skip_frontend else source_basename
        )
        workspace = PipelineWorkspace.create(config.output_path, workspace_basename)
        workspace.write_config(config)

        log(f"Model: {source_basename}", config.verbose)
        log(f"Config: cost={config.cost}, kern={config.kern}, device={config.device}", config.verbose)

        t0 = time.time()

        if config.skip_frontend:
            frontend = self.frontend_stage.load_existing(
                source_basename,
                workspace,
                use_standard_ir=True,
            )
            log("Frontend   skipped (using standard_ir input)", config.verbose)
        else:
            frontend = self.frontend_stage.run(model, example_inputs, source_basename, config, workspace)
            log(
                f"Frontend   complete ({len(frontend.tensor_shapes)} tensors, {len(frontend.errors)} validation errors)",
                config.verbose,
            )

        if config.skip_optimizer:
            optimizer = self.optimizer_stage.load_existing(workspace_basename, config, workspace)
            log("Optimizer  skipped (reusing existing expressions)", config.verbose)
        else:
            optimizer = self.optimizer_stage.run(workspace_basename, frontend, config, workspace)
            log(
                f"Optimizer  complete ({optimizer.expression_count} expressions)",
                config.verbose,
            )

        if not optimizer.expressions_path.exists():
            backend = BackendArtifacts(
                benchmark_path=None,
                source_benchmark_path=None,
                all_results=[],
            )
            log("Backend    skipped (no expressions to profile)", config.verbose)
        elif config.skip_backend:
            backend = BackendArtifacts(
                benchmark_path=None,
                source_benchmark_path=None,
                all_results=[],
            )
            log("Backend    skipped (skip_backend=True)", config.verbose)
        elif config.reuse_benchmark:
            backend = self.backend_stage.load_existing(
                workspace_basename,
                frontend,
                optimizer,
                workspace,
                verbose=config.verbose,
            )
            log("Backend    reused (existing benchmark loaded)", config.verbose)
        else:
            backend = self.backend_stage.run(
                workspace_basename,
                frontend,
                optimizer,
                config,
                workspace,
            )
            log(
                f"Backend    complete ({len([r for r in backend.all_results if r.get('error') is None])}/{len(backend.all_results)} valid kernels)",
                config.verbose,
            )

        total_dt = time.time() - t0
        log(f"Total      {total_dt:6.1f}s", config.verbose)

        result = OptimizeResult(
            kernel=backend.best_kernel,
            kernel_path=backend.best_kernel_path,
            execution_time_ms=backend.best_execution_time_ms,
            ir_id=backend.best_ir_id,
            ir_expression=backend.best_ir_expression,
            all_results=backend.all_results,
            workspace=workspace.root,
            frontend=frontend,
            optimizer=optimizer,
            backend=backend,
        )

        if config.verbose:
            result.print_summary()

        return result


def optimize(model: Any, example_inputs: Any, **kwargs: Any) -> OptimizeResult:
    """Run the full Trinity optimization pipeline and return the result."""
    config = OptimizeConfig(**kwargs)
    pipeline = TrinityPipeline()
    return pipeline.optimize(model, example_inputs, config=config)
