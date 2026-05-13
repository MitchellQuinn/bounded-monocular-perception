"""File-based trace recording for single-frame inference diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

import interfaces.contracts as contracts
from interfaces import (
    FrameHash,
    InferenceRequest,
    InferenceResult,
    PreparedInferenceInputs,
    WorkerError,
)


APP_PROJECT_ROOT = Path(__file__).resolve().parents[3]
APP_PROJECT_NAME = APP_PROJECT_ROOT.name
LIVE_INFERENCE_VERSION = "v0.2"
DEFAULT_TRACE_OUTPUT_DIR = APP_PROJECT_ROOT / "live_traces"
_HASH_PREFIX_LENGTH = 8
_MAX_CHECKPOINT_HASH_BYTES = 256 * 1024 * 1024


def default_trace_output_dir(app_root_path: Path | str | None = None) -> Path:
    """Return a trace root derived from the resolved live app root."""
    root = Path(app_root_path) if app_root_path is not None else APP_PROJECT_ROOT
    return root.expanduser().resolve(strict=False) / "live_traces"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class InferenceTraceRecorder:
    """Write a per-run trace bundle for one single-frame inference."""

    def __init__(
        self,
        *,
        output_dir: Path | str | None = None,
        context_metadata: Mapping[str, Any] | None = None,
        now_utc_fn: Callable[[], str] | None = None,
    ) -> None:
        raw_context = dict(context_metadata or {})
        self.app_root_path = Path(
            raw_context.get("app_root_path") or APP_PROJECT_ROOT
        ).expanduser().resolve(strict=False)
        default_output_dir = default_trace_output_dir(self.app_root_path)
        self.output_dir = (
            Path(output_dir).expanduser().resolve(strict=False)
            if output_dir is not None
            else default_output_dir
        )
        trace_root_overridden = bool(
            raw_context.get(
                "trace_root_overridden",
                output_dir is not None
                and self.output_dir.resolve(strict=False)
                != default_output_dir.resolve(strict=False),
            )
        )
        self._context_metadata = {
            "app_project_name": raw_context.get("app_project_name") or APP_PROJECT_NAME,
            "app_version": raw_context.get("app_version") or LIVE_INFERENCE_VERSION,
            "live_inference_version": (
                raw_context.get("live_inference_version") or LIVE_INFERENCE_VERSION
            ),
            "app_root_path": str(self.app_root_path),
            "trace_root_path": str(self.output_dir),
            "trace_root_overridden": trace_root_overridden,
            "git_commit_sha": raw_context.get("git_commit_sha")
            if "git_commit_sha" in raw_context
            else _git_commit_sha(self.app_root_path),
            "git_dirty": raw_context.get("git_dirty")
            if "git_dirty" in raw_context
            else _git_dirty(self.app_root_path),
            **raw_context,
        }
        self._now_utc_fn = now_utc_fn or _utc_now_iso

    def create_trace_directory(
        self,
        *,
        request_id: str,
        frame_hash: FrameHash,
        created_at_utc: str | None = None,
    ) -> Path:
        """Create a unique trace directory without overwriting a prior bundle."""
        created = created_at_utc or self._now_utc_fn()
        name = (
            f"{_safe_timestamp(created)}__"
            f"{_safe_filename(request_id)}__"
            f"{_safe_filename(frame_hash.value[:_HASH_PREFIX_LENGTH])}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for index in range(0, 1000):
            suffix = "" if index == 0 else f"__{index + 1}"
            trace_dir = self.output_dir / f"{name}{suffix}"
            try:
                trace_dir.mkdir(parents=False, exist_ok=False)
                return trace_dir
            except FileExistsError:
                continue
        raise FileExistsError(f"Could not allocate a unique trace directory under {self.output_dir}")

    def record_trace(
        self,
        *,
        trace_dir: Path,
        image_bytes: bytes,
        request: InferenceRequest,
        prepared_inputs: PreparedInferenceInputs | None,
        result: InferenceResult | None,
        source_path: Path | None = None,
        created_at_utc: str | None = None,
        error: WorkerError | None = None,
    ) -> Path:
        """Write trace artifacts and a manifest, tolerating missing optional files."""
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        created = created_at_utc or self._now_utc_fn()
        frame_hash = request.frame.frame_hash or FrameHash("")
        preprocessing_metadata = (
            dict(prepared_inputs.preprocessing_metadata)
            if prepared_inputs is not None
            else _preprocessing_metadata_from_error(error)
        )

        artifacts: list[dict[str, Any]] = []
        missing_optional: list[dict[str, str]] = []

        raw_path = trace_dir / "accepted_raw_frame.png"
        raw_path.write_bytes(bytes(image_bytes))
        artifacts.append(_artifact("accepted_raw_frame", raw_path, required=True))

        if result is not None:
            result_path = trace_dir / "inference_result.json"
            _write_json(result_path, result.to_dict())
            artifacts.append(_artifact("inference_result", result_path, required=True))

            model_outputs_path = trace_dir / "model_outputs.json"
            _write_json(model_outputs_path, _model_outputs_payload(result))
            artifacts.append(_artifact("model_outputs", model_outputs_path, required=False))

        if preprocessing_metadata:
            metadata_path = trace_dir / "preprocessing_metadata.json"
            _write_json(metadata_path, preprocessing_metadata)
            artifacts.append(
                _artifact("preprocessing_metadata", metadata_path, required=False)
            )

            roi_fcn_metadata = _roi_fcn_metadata(preprocessing_metadata)
            if roi_fcn_metadata:
                roi_metadata_path = trace_dir / "roi_fcn_metadata.json"
                _write_json(roi_metadata_path, roi_fcn_metadata)
                artifacts.append(
                    _artifact("roi_fcn_metadata", roi_metadata_path, required=False)
                )

            mask_background = _mask_background_metadata(preprocessing_metadata)
            if mask_background:
                mask_path = trace_dir / "mask_background_metadata.json"
                _write_json(mask_path, mask_background)
                artifacts.append(
                    _artifact("mask_background_metadata", mask_path, required=False)
                )

        if prepared_inputs is not None:
            geometry = prepared_inputs.model_inputs.get(contracts.TRI_STREAM_GEOMETRY_KEY)
            if geometry is not None:
                geometry_path = trace_dir / "x_geometry.json"
                _write_json(geometry_path, {"x_geometry": _json_safe(geometry)})
                artifacts.append(_artifact("x_geometry", geometry_path, required=False))

        debug_paths = _debug_paths(preprocessing_metadata, result, error)
        for kind, path in sorted(debug_paths.items()):
            if str(kind) == contracts.DISPLAY_ARTIFACT_ACCEPTED_RAW_FRAME:
                continue
            copied = _copy_optional_debug_artifact(
                kind=kind,
                source=Path(path),
                trace_dir=trace_dir,
            )
            if copied is None:
                missing_optional.append({"kind": str(kind), "path": str(path)})
                continue
            artifacts.append(_artifact(str(kind), copied, required=False))

        if error is not None:
            failure_path = trace_dir / "failure_result.json"
            _write_json(failure_path, _failure_result_payload(error))
            artifacts.append(_artifact("failure_result", failure_path, required=True))

            error_path = trace_dir / "worker_error.json"
            _write_json(error_path, error.to_dict())
            artifacts.append(_artifact("worker_error", error_path, required=False))

        manifest_path = trace_dir / "trace_manifest.json"
        manifest = self._manifest_payload(
            trace_dir=trace_dir,
            request=request,
            result=result,
            frame_hash=frame_hash,
            created_at_utc=created,
            source_path=source_path,
            preprocessing_metadata=preprocessing_metadata,
            artifacts=artifacts,
            missing_optional_artifacts=missing_optional,
            error=error,
        )
        artifacts.append(_artifact("trace_manifest", manifest_path, required=True))
        manifest["artifacts"] = artifacts
        _write_json(manifest_path, manifest)
        return trace_dir

    def _manifest_payload(
        self,
        *,
        trace_dir: Path,
        request: InferenceRequest,
        result: InferenceResult | None,
        frame_hash: FrameHash,
        created_at_utc: str,
        source_path: Path | None,
        preprocessing_metadata: Mapping[str, Any],
        artifacts: list[dict[str, Any]],
        missing_optional_artifacts: list[dict[str, str]],
        error: WorkerError | None,
    ) -> dict[str, Any]:
        result_extras = dict(result.extras) if result is not None else {}
        context = self._manifest_context()
        preprocessing_revision = _first_present(
            getattr(result, "preprocessing_parameter_revision", None),
            preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_RUNTIME_PARAMETER_REVISION
            ),
            preprocessing_metadata.get("preprocessing_parameter_revision"),
        )
        checkpoint_paths = _checkpoint_paths(context, result_extras)
        stage_policy_snapshot = _stage_policy_snapshot(preprocessing_metadata)
        ui_state_snapshot = _ui_state_snapshot(preprocessing_metadata)
        distance_orientation_reached = _distance_orientation_regressor_reached(
            preprocessing_metadata,
            result=result,
        )
        return {
            "trace_kind": "single_frame_inference",
            "app_project_name": context.get("app_project_name"),
            "app_version": context.get("app_version"),
            "live_inference_version": context.get("live_inference_version"),
            "app_root_path": context.get("app_root_path"),
            "trace_root_path": context.get("trace_root_path"),
            "trace_root_overridden": context.get("trace_root_overridden"),
            "process_cwd": context.get("process_cwd"),
            "argv": context.get("argv"),
            "command_line": context.get("command_line"),
            "python_executable": context.get("python_executable"),
            "git_commit_sha": context.get("git_commit_sha"),
            "git_dirty": context.get("git_dirty"),
            "request_id": request.request_id,
            "frame_hash": frame_hash.value,
            "input_image_hash": frame_hash.to_dict(),
            "input_image_hash_value": frame_hash.value,
            "source_path": str(source_path) if source_path is not None else None,
            "request_frame_path": str(request.frame.image_path),
            "created_at_utc": created_at_utc,
            "trace_dir": str(trace_dir),
            "model_selection_path": context.get("model_selection_path"),
            "model_root": result_extras.get("model_root") or context.get("model_root"),
            "distance_orientation_root": (
                context.get("distance_orientation_root")
                or result_extras.get("model_root")
            ),
            "distance_orientation_artifact_root": (
                context.get("distance_orientation_root")
                or result_extras.get("model_root")
            ),
            "roi_fcn_root": context.get("roi_fcn_root"),
            "roi_fcn_artifact_root": context.get("roi_fcn_root"),
            "distance_orientation_checkpoint_path": checkpoint_paths.get(
                "distance_orientation"
            ),
            "roi_fcn_checkpoint_path": checkpoint_paths.get("roi_fcn"),
            "checkpoint_paths": checkpoint_paths,
            "checkpoint_file_hashes": _checkpoint_file_hashes(checkpoint_paths),
            "preprocessing_contract_name": (
                preprocessing_metadata.get("preprocessing_contract_name")
                or contracts.PREPROCESSING_CONTRACT_NAME
            ),
            "orientation_source_mode": preprocessing_metadata.get(
                "orientation_source_mode"
            ),
            "locator_input_mode": preprocessing_metadata.get(
                "roi_locator_input_mode"
            ),
            "locator_input_parameters": {
                "sheet_min_gray": preprocessing_metadata.get(
                    "roi_locator_sheet_min_gray"
                ),
                "target_max_gray": preprocessing_metadata.get(
                    "roi_locator_target_max_gray"
                ),
                "min_component_area_px": preprocessing_metadata.get(
                    "roi_locator_min_component_area_px"
                ),
                "morphology_close_kernel_px": preprocessing_metadata.get(
                    "roi_locator_morphology_close_kernel_px"
                ),
                "dilate_kernel_px": preprocessing_metadata.get(
                    "roi_locator_dilate_kernel_px"
                ),
                "restrict_to_lower_frame_fraction": preprocessing_metadata.get(
                    "roi_locator_restrict_to_lower_frame_fraction"
                ),
            },
            "roi_locator_input_polarity": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY
            ),
            "preprocessing_parameter_revision": preprocessing_revision,
            "mask_revision": preprocessing_metadata.get("frame_mask_revision"),
            "background_revision": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_BACKGROUND_REVISION
            ),
            "background_threshold": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_BACKGROUND_THRESHOLD
            ),
            "fill_value": preprocessing_metadata.get("frame_mask_fill_value"),
            "stage_policy_revision": preprocessing_metadata.get(
                "stage_policy_revision"
            ),
            "diagnostic_profile_name": preprocessing_metadata.get(
                "diagnostic_profile_name"
            ),
            "stage_policy_snapshot": stage_policy_snapshot,
            "ui_state_snapshot": ui_state_snapshot,
            "apply_manual_mask_to_roi_locator": preprocessing_metadata.get(
                "apply_manual_mask_to_roi_locator"
            ),
            "apply_background_removal_to_roi_locator": preprocessing_metadata.get(
                "apply_background_removal_to_roi_locator"
            ),
            "apply_manual_mask_to_regressor_preprocessing": (
                preprocessing_metadata.get(
                    "apply_manual_mask_to_regressor_preprocessing"
                )
            ),
            "apply_background_removal_to_regressor_preprocessing": (
                preprocessing_metadata.get(
                    "apply_background_removal_to_regressor_preprocessing"
                )
            ),
            "manual_mask_applied_to_roi_locator": preprocessing_metadata.get(
                "manual_mask_applied_to_roi_locator"
            ),
            "background_removal_applied_to_roi_locator": (
                preprocessing_metadata.get(
                    "background_removal_applied_to_roi_locator"
                )
            ),
            "roi_confidence": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_CONFIDENCE
            ),
            "roi_locator_confidence": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_LOCATOR_CONFIDENCE
            ),
            "roi_center_canvas_xy_px": preprocessing_metadata.get(
                "roi_center_canvas_xy_px"
            ),
            "roi_center_source_xy_px": preprocessing_metadata.get(
                "roi_center_source_xy_px"
            ),
            "roi_locator_center_source_xy_px": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_LOCATOR_CENTER_SOURCE_XY_PX
            ),
            "roi_requested_bounds_xyxy_px": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_REQUEST_XYXY_PX
            ),
            "roi_requested_xyxy_px": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_REQUESTED_XYXY_PX
            ),
            "roi_pre_clip_bounds_xyxy_px": preprocessing_metadata.get(
                "roi_pre_clip_bounds_xyxy_px"
            )
            or preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_REQUEST_XYXY_PX
            ),
            "roi_source_bounds_xyxy_px": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX
            ),
            "roi_clipped_bounds_xyxy_px": preprocessing_metadata.get(
                "roi_clipped_bounds_xyxy_px"
            )
            or preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX
            ),
            "roi_clipped": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_CLIPPED
            ),
            "roi_clip_max_px": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX
            ),
            "roi_clip_tolerance_px": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX
            ),
            "roi_clip_tolerated": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED
            ),
            "roi_accepted": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_ACCEPTED
            ),
            "roi_rejected": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_REJECTED
            ),
            "roi_rejection_reason": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASON
            ),
            "foreground_mask_empty": preprocessing_metadata.get("foreground_mask_empty"),
            "foreground_pixel_count": preprocessing_metadata.get(
                "foreground_pixel_count"
            ),
            "silhouette_diagnostics": preprocessing_metadata.get(
                "silhouette_diagnostics"
            ),
            "final_locator_input_stats": preprocessing_metadata.get(
                "final_locator_input_stats"
            ),
            "final_locator_input_min": preprocessing_metadata.get(
                "final_locator_input_min"
            ),
            "final_locator_input_max": preprocessing_metadata.get(
                "final_locator_input_max"
            ),
            "final_locator_input_mean": preprocessing_metadata.get(
                "final_locator_input_mean"
            ),
            "final_locator_input_median": preprocessing_metadata.get(
                "final_locator_input_median"
            ),
            "final_locator_input_nonzero_pixel_count": preprocessing_metadata.get(
                "final_locator_input_nonzero_pixel_count"
            ),
            "final_locator_input_non_whiteish_pixel_count": (
                preprocessing_metadata.get(
                    "final_locator_input_non_whiteish_pixel_count"
                )
            ),
            "distance_orientation_regressor_reached": distance_orientation_reached,
            "failure_stage": (
                error.failure_stage.value
                if error is not None and error.failure_stage is not None
                else None
            ),
            "error_type": error.error_type if error is not None else None,
            "device": result_extras.get("device") or context.get("device"),
            "context": context,
            "missing_optional_artifacts": missing_optional_artifacts,
            "artifacts": artifacts,
        }

    def _manifest_context(self) -> dict[str, Any]:
        context = dict(self._context_metadata)
        argv = list(sys.argv)
        context.update(
            {
                "app_root_path": str(self.app_root_path),
                "trace_root_path": str(self.output_dir),
                "process_cwd": str(Path.cwd()),
                "argv": argv,
                "command_line": shlex.join(argv),
                "python_executable": sys.executable,
            }
        )
        return context


def _copy_optional_debug_artifact(
    *,
    kind: str,
    source: Path,
    trace_dir: Path,
) -> Path | None:
    if not source.is_file():
        return None
    target = trace_dir / _canonical_artifact_name(kind, source)
    if source.resolve(strict=False) == target.resolve(strict=False):
        return target
    shutil.copy2(source, target)
    return target


def _canonical_artifact_name(kind: str, source: Path) -> str:
    mapping = {
        contracts.TRI_STREAM_DISTANCE_IMAGE_KEY: "x_distance_image.png",
        contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY: "x_orientation_image.png",
        contracts.DISPLAY_ARTIFACT_ROI_CROP: "roi_crop.png",
        contracts.DISPLAY_ARTIFACT_LOCATOR_INPUT: "locator_input.png",
        contracts.DISPLAY_ARTIFACT_ROI_OVERLAY_METADATA: "roi_overlay_metadata.json",
        contracts.DISPLAY_ARTIFACT_ROI_OVERLAY: "roi_overlay.png",
        contracts.DISPLAY_ARTIFACT_LOCATOR_INPUT_BEFORE_POLARITY: (
            "locator_input_before_polarity.png"
        ),
        contracts.DISPLAY_ARTIFACT_LOCATOR_INPUT_AFTER_POLARITY: (
            "locator_input_after_polarity.png"
        ),
        contracts.DISPLAY_ARTIFACT_FINAL_LOCATOR_INPUT: "final_locator_input.png",
        "locator_input_raw_or_pretransform": "locator_input_raw_or_pretransform.png",
        "locator_input_after_manual_mask": "locator_input_after_manual_mask.png",
        "locator_input_after_background_removal": (
            "locator_input_after_background_removal.png"
        ),
        "locator_input_as_is": "locator_input_as_is.png",
        "locator_input_inverted": "locator_input_inverted.png",
        "locator_input_sheet_dark_foreground": (
            "locator_input_sheet_dark_foreground.png"
        ),
        "roi_fcn_heatmap": "roi_fcn_heatmap.png",
        "roi_fcn_heatmap_pre_exclusion": "roi_fcn_heatmap_pre_exclusion.png",
        "roi_fcn_heatmap_post_exclusion": "roi_fcn_heatmap_post_exclusion.png",
        "preprocessor_source_before_regressor_masks": (
            "preprocessor_source_before_regressor_masks.png"
        ),
        "preprocessor_source_after_regressor_masks": (
            "preprocessor_source_after_regressor_masks.png"
        ),
        "manual_mask": "manual_mask.png",
        "background_snapshot": "background_snapshot.png",
        "background_removal_mask": "background_removal_mask.png",
        "combined_ignore_mask": "combined_ignore_mask.png",
    }
    if kind in mapping:
        return mapping[kind]
    suffix = source.suffix if source.suffix else ".artifact"
    return f"{_safe_filename(kind)}{suffix}"


def _debug_paths(
    preprocessing_metadata: Mapping[str, Any],
    result: InferenceResult | None,
    error: WorkerError | None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for raw in (
        preprocessing_metadata.get(contracts.PREPROCESSING_METADATA_DEBUG_PATHS),
        preprocessing_metadata.get(contracts.PREPROCESSING_METADATA_DEBUG_IMAGE_PATHS),
        result.debug_paths if result is not None else None,
        error.details.get(contracts.PREPROCESSING_METADATA_DEBUG_PATHS)
        if error is not None
        else None,
        error.details.get("debug_paths") if error is not None else None,
    ):
        if not isinstance(raw, Mapping):
            continue
        for key, value in raw.items():
            paths[str(key)] = Path(value)
    return paths


def _preprocessing_metadata_from_error(error: WorkerError | None) -> dict[str, Any]:
    if error is None:
        return {}
    raw = error.details.get("preprocessing_metadata")
    return dict(raw) if isinstance(raw, Mapping) else {}


def _stage_policy_snapshot(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "stage_policy_revision",
        "diagnostic_profile_name",
        "roi_locator_input_mode",
        contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY,
        "apply_manual_mask_to_roi_locator",
        "apply_manual_mask_to_regressor_preprocessing",
        "apply_background_removal_to_roi_locator",
        "apply_background_removal_to_regressor_preprocessing",
        "roi_min_confidence",
        "reject_clipped_roi",
        contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX,
        "roi_min_content_fraction",
        "roi_locator_sheet_min_gray",
        "roi_locator_target_max_gray",
        "roi_locator_min_component_area_px",
        "roi_locator_morphology_close_kernel_px",
        "roi_locator_dilate_kernel_px",
        "roi_locator_restrict_to_lower_frame_fraction",
    )
    return {str(key): metadata.get(key) for key in keys if key in metadata}


def _ui_state_snapshot(metadata: Mapping[str, Any]) -> dict[str, Any]:
    mode = metadata.get("roi_locator_input_mode")
    invert_checked: bool | None
    if mode in {"as_is", "inverted"}:
        invert_checked = mode == "inverted"
    else:
        invert_checked = None
    return {
        "diagnostic_profile_name": metadata.get("diagnostic_profile_name"),
        "roi_locator_input_mode_dropdown": mode,
        "invert_roi_locator_input_checkbox_checked": invert_checked,
        "invert_roi_locator_input_checkbox_enabled": mode in {"as_is", "inverted"},
        "apply_manual_mask_to_roi_locator_checkbox": metadata.get(
            "apply_manual_mask_to_roi_locator"
        ),
        "apply_manual_mask_to_model_preprocessing_checkbox": metadata.get(
            "apply_manual_mask_to_regressor_preprocessing"
        ),
        "apply_background_removal_to_roi_locator_checkbox": metadata.get(
            "apply_background_removal_to_roi_locator"
        ),
        "apply_background_removal_to_model_preprocessing_checkbox": metadata.get(
            "apply_background_removal_to_regressor_preprocessing"
        ),
        "mask_fill_value": metadata.get("frame_mask_fill_value"),
        "mask_revision": metadata.get("frame_mask_revision"),
        "mask_pixel_count": metadata.get("frame_mask_pixel_count"),
        "background_captured": metadata.get(
            contracts.PREPROCESSING_METADATA_BACKGROUND_CAPTURED
        ),
        "background_enabled": metadata.get(
            contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_ENABLED
        ),
        "background_revision": metadata.get(
            contracts.PREPROCESSING_METADATA_BACKGROUND_REVISION
        ),
        "background_threshold": metadata.get(
            contracts.PREPROCESSING_METADATA_BACKGROUND_THRESHOLD
        ),
    }


def _distance_orientation_regressor_reached(
    metadata: Mapping[str, Any],
    *,
    result: InferenceResult | None,
) -> bool:
    value = metadata.get("distance_orientation_regressor_reached")
    if isinstance(value, bool):
        return value
    if value is not None:
        return bool(value)
    return result is not None


def _checkpoint_paths(
    context: Mapping[str, Any],
    result_extras: Mapping[str, Any],
) -> dict[str, str]:
    paths: dict[str, str] = {}
    distance = (
        context.get("distance_orientation_checkpoint_path")
        or result_extras.get("distance_orientation_checkpoint_path")
    )
    roi = context.get("roi_fcn_checkpoint_path") or result_extras.get(
        "roi_fcn_checkpoint_path"
    )
    if distance:
        paths["distance_orientation"] = str(distance)
    if roi:
        paths["roi_fcn"] = str(roi)
    return paths


def _checkpoint_file_hashes(
    checkpoint_paths: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    return {
        str(name): _hash_checkpoint_file(Path(path))
        for name, path in checkpoint_paths.items()
    }


def _hash_checkpoint_file(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": str(path),
        "algorithm": "sha256",
        "value": None,
        "size_bytes": None,
        "skipped": None,
    }
    try:
        stat = path.stat()
    except OSError as exc:
        payload["skipped"] = f"unavailable: {exc}"
        return payload
    payload["size_bytes"] = int(stat.st_size)
    if stat.st_size > _MAX_CHECKPOINT_HASH_BYTES:
        payload["skipped"] = (
            f"file larger than {_MAX_CHECKPOINT_HASH_BYTES} bytes"
        )
        return payload
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        payload["skipped"] = f"read failed: {exc}"
        return payload
    payload["value"] = digest.hexdigest()
    return payload


def _git_commit_sha(app_root: Path) -> str | None:
    return _git_output(app_root, "rev-parse", "HEAD")


def _git_dirty(app_root: Path) -> bool | None:
    output = _git_output(app_root, "status", "--porcelain")
    if output is None:
        return None
    return bool(output.strip())


def _git_output(app_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=app_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _roi_fcn_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    raw = metadata.get(contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA)
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    for key in (
        "roi_locator_input_mode",
        contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY,
        "roi_locator_sheet_min_gray",
        "roi_locator_target_max_gray",
        "roi_locator_min_component_area_px",
        "roi_locator_morphology_close_kernel_px",
        "roi_locator_dilate_kernel_px",
        "roi_locator_restrict_to_lower_frame_fraction",
        contracts.PREPROCESSING_METADATA_ROI_CONFIDENCE,
        contracts.PREPROCESSING_METADATA_ROI_LOCATOR_CONFIDENCE,
        "roi_center_canvas_xy_px",
        "roi_center_source_xy_px",
        contracts.PREPROCESSING_METADATA_ROI_LOCATOR_CENTER_SOURCE_XY_PX,
        contracts.PREPROCESSING_METADATA_ROI_CLIPPED,
        contracts.PREPROCESSING_METADATA_ROI_CLIP_LEFT_PX,
        contracts.PREPROCESSING_METADATA_ROI_CLIP_RIGHT_PX,
        contracts.PREPROCESSING_METADATA_ROI_CLIP_TOP_PX,
        contracts.PREPROCESSING_METADATA_ROI_CLIP_BOTTOM_PX,
        contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX,
        contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX,
        contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED,
        contracts.PREPROCESSING_METADATA_ROI_ACCEPTED,
        contracts.PREPROCESSING_METADATA_ROI_REJECTED,
        contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASON,
        contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASONS,
        contracts.PREPROCESSING_METADATA_ROI_REQUEST_XYXY_PX,
        contracts.PREPROCESSING_METADATA_ROI_REQUESTED_XYXY_PX,
        contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX,
        "roi_pre_clip_bounds_xyxy_px",
        "roi_clipped_bounds_xyxy_px",
        contracts.PREPROCESSING_METADATA_ROI_CANVAS_INSERT_XYXY_PX,
        "foreground_mask_empty",
        "foreground_pixel_count",
        "silhouette_diagnostics",
        "final_locator_input_stats",
        "final_locator_input_min",
        "final_locator_input_max",
        "final_locator_input_mean",
        "final_locator_input_median",
        "final_locator_input_nonzero_pixel_count",
        "final_locator_input_non_whiteish_pixel_count",
        "preprocessing_failure_type",
        "preprocessing_failure_message",
        "distance_orientation_regressor_reached",
    ):
        if key in metadata:
            payload[str(key)] = metadata[key]
    return payload


def _failure_result_payload(error: WorkerError) -> dict[str, Any]:
    return {
        "failure_stage": (
            error.failure_stage.value if error.failure_stage is not None else None
        ),
        "error_type": error.error_type,
        "message": error.message,
        "recoverable": bool(error.recoverable),
        "timestamp_utc": error.timestamp_utc,
        "details": error.details,
    }


def _model_outputs_payload(result: InferenceResult) -> dict[str, Any]:
    return {
        contracts.MODEL_OUTPUT_DISTANCE_KEY: result.predicted_distance_m,
        contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY: [
            result.predicted_yaw_sin,
            result.predicted_yaw_cos,
        ],
        contracts.PREDICTED_DISTANCE_FIELD: result.predicted_distance_m,
        contracts.PREDICTED_YAW_SIN_FIELD: result.predicted_yaw_sin,
        contracts.PREDICTED_YAW_COS_FIELD: result.predicted_yaw_cos,
        contracts.PREDICTED_YAW_DEG_FIELD: result.predicted_yaw_deg,
    }


def _mask_background_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key)
        if (
            "mask" in key_text
            or "background" in key_text
            or key_text.startswith("combined_ignore")
        ):
            payload[key_text] = value
    return payload


def _artifact(kind: str, path: Path, *, required: bool) -> dict[str, Any]:
    return {
        "kind": str(kind),
        "path": str(path),
        "filename": path.name,
        "required": bool(required),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError):
            return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_timestamp(value: str) -> str:
    text = str(value).strip()
    if text.endswith("+00:00"):
        text = text[:-6] + "Z"
    text = text.replace("-", "").replace(":", "")
    text = text.replace("+0000", "Z")
    text = re.sub(r"\.\d+", "", text)
    text = text.replace("T", "T")
    return _safe_filename(text or _utc_now_iso())


def _safe_filename(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip(".-") or "artifact"


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


__all__ = [
    "APP_PROJECT_NAME",
    "APP_PROJECT_ROOT",
    "DEFAULT_TRACE_OUTPUT_DIR",
    "InferenceTraceRecorder",
    "LIVE_INFERENCE_VERSION",
    "default_trace_output_dir",
]
