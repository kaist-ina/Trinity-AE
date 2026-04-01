"""Frontend wrapper that keeps the original frontend code untouched."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..artifacts import FrontendArtifacts
from ..config import OptimizeConfig
from ..runtime import FRONTEND_DIR, STANDARD_IR_DIR, load_shapes, log, prepend_sys_path
from ..workspace import PipelineWorkspace


class FrontendStage:
    _STANDARD_IR_ALIASES = {
        "vanila": "vanilla",
    }

    def run(
        self,
        model: Any,
        example_inputs: Any,
        basename: str,
        config: OptimizeConfig,
        workspace: PipelineWorkspace,
    ) -> FrontendArtifacts:
        log("Frontend: generating IR ...", config.verbose)

        with prepend_sys_path(FRONTEND_DIR):
            from utils.pipeline import export_model_ir  # type: ignore

            _main_func_ir, errors = export_model_ir(
                model,
                example_inputs,
                basename=basename,
                output_dir=str(workspace.frontend_dir),
                inline_shape_op=config.inline_shape_op,
                inline_elementwise_op=config.inline_elementwise_op,
                remove_short_loop_threshold=config.remove_short_loop_threshold,
                decompose_nested_op_ratio=config.decompose_nested_op_ratio,
                flat_output=True,
            )

        artifacts = self._load_from_workspace(workspace)
        artifacts.errors = list(errors)
        workspace.write_json(workspace.frontend_dir, "validation_errors.json", artifacts.errors)

        if config.fail_on_frontend_errors and artifacts.errors:
            joined = "\n".join(artifacts.errors)
            raise RuntimeError(f"Frontend validation failed:\n{joined}")

        return artifacts

    def _load_from_workspace(self, workspace: PipelineWorkspace) -> FrontendArtifacts:
        """Load artifacts directly from workspace (used after run() generates files there)."""
        ir_path = workspace.frontend_dir / "ir.txt"
        shapes_path = workspace.frontend_dir / "shapes.json"

        if not ir_path.exists() or not shapes_path.exists():
            raise FileNotFoundError(
                f"Expected frontend artifacts at {workspace.frontend_dir}, but they were not found."
            )

        errors_path = workspace.frontend_dir / "validation_errors.json"
        errors = json.loads(errors_path.read_text()) if errors_path.exists() else []

        return FrontendArtifacts(
            ir_path=ir_path,
            shapes_path=shapes_path,
            source_ir_path=ir_path,
            source_shapes_path=shapes_path,
            tensor_shapes=load_shapes(shapes_path),
            errors=errors,
        )

    def _resolve_standard_ir_dir(self, basename: str) -> Path:
        candidates = [
            STANDARD_IR_DIR / basename,
            STANDARD_IR_DIR / self._STANDARD_IR_ALIASES.get(basename, basename),
        ]
        for candidate in candidates:
            if (candidate / "ir.txt").exists() and (candidate / "shapes.json").exists():
                return candidate
        raise FileNotFoundError(
            f"Expected standard IR artifacts at one of {[str(path) for path in candidates]}, but they were not found."
        )

    def load_existing(
        self,
        basename: str,
        workspace: PipelineWorkspace,
        *,
        use_standard_ir: bool = False,
    ) -> FrontendArtifacts:
        workspace_ir_path = workspace.frontend_dir / "ir.txt"
        workspace_shapes_path = workspace.frontend_dir / "shapes.json"

        if use_standard_ir:
            standard_ir_dir = self._resolve_standard_ir_dir(basename)
            source_ir_path = standard_ir_dir / "ir.txt"
            source_shapes_path = standard_ir_dir / "shapes.json"
        else:
            source_ir_path = FRONTEND_DIR / "outputs" / "trinity" / basename / "ir.txt"
            source_shapes_path = FRONTEND_DIR / "outputs" / "trinity" / basename / "shapes.json"

        if workspace_ir_path.exists() and workspace_shapes_path.exists():
            source_ir_path = workspace_ir_path
            source_shapes_path = workspace_shapes_path

        if not source_ir_path.exists() or not source_shapes_path.exists():
            raise FileNotFoundError(
                f"Expected frontend artifacts at {source_ir_path.parent}, but they were not found."
            )

        if source_ir_path.parent == workspace.frontend_dir:
            ir_path = source_ir_path
            shapes_path = source_shapes_path
        else:
            ir_path = workspace.mirror_file(source_ir_path, workspace.frontend_dir)
            shapes_path = workspace.mirror_file(source_shapes_path, workspace.frontend_dir)

        errors_path = workspace.frontend_dir / "validation_errors.json"
        errors = json.loads(errors_path.read_text()) if errors_path.exists() else []

        return FrontendArtifacts(
            ir_path=ir_path,
            shapes_path=shapes_path,
            source_ir_path=source_ir_path,
            source_shapes_path=source_shapes_path,
            tensor_shapes=load_shapes(shapes_path),
            errors=errors,
        )
