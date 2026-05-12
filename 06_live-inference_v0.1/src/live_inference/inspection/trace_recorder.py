"""File-based trace recording for single-frame inference diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import interfaces.contracts as contracts
from interfaces import (
    FrameHash,
    InferenceRequest,
    InferenceResult,
    PreparedInferenceInputs,
    WorkerError,
)


DEFAULT_TRACE_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "live_traces"
_HASH_PREFIX_LENGTH = 8


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
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_TRACE_OUTPUT_DIR
        self._context_metadata = dict(context_metadata or {})
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
        context = dict(self._context_metadata)
        preprocessing_revision = _first_present(
            getattr(result, "preprocessing_parameter_revision", None),
            preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_RUNTIME_PARAMETER_REVISION
            ),
            preprocessing_metadata.get("preprocessing_parameter_revision"),
        )
        return {
            "trace_kind": "single_frame_inference",
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
            "distance_orientation_artifact_root": (
                context.get("distance_orientation_root")
                or result_extras.get("model_root")
            ),
            "roi_fcn_artifact_root": context.get("roi_fcn_root"),
            "preprocessing_contract_name": (
                preprocessing_metadata.get("preprocessing_contract_name")
                or contracts.PREPROCESSING_CONTRACT_NAME
            ),
            "orientation_source_mode": preprocessing_metadata.get(
                "orientation_source_mode"
            ),
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
            "roi_source_bounds_xyxy_px": preprocessing_metadata.get(
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
            "roi_rejection_reason": preprocessing_metadata.get(
                contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASON
            ),
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


def _roi_fcn_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    raw = metadata.get(contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA)
    payload = dict(raw) if isinstance(raw, Mapping) else {}
    for key in (
        contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY,
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
        contracts.PREPROCESSING_METADATA_ROI_CANVAS_INSERT_XYXY_PX,
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
    "DEFAULT_TRACE_OUTPUT_DIR",
    "InferenceTraceRecorder",
]
