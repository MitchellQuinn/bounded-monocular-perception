"""Single-sample raw-image inference pipeline for v0.2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd
import torch

from .discovery import RawCorpus, load_corpus_samples
from .external import ensure_external_paths
from .paths import results_root, sanitize_identifier, timestamp_slug, to_repo_relative

ensure_external_paths()

from rb_pipeline_v4.config import SilhouetteStageConfigV4
from rb_pipeline_v4.image_io import read_grayscale_uint8, write_grayscale_png
from rb_pipeline_v4.pack_dual_stream_stage import (
    _place_image_on_canvas,
    _reconstruct_roi_canvas_from_source,
    _render_inverted_vehicle_detail_on_white,
    _silhouette_to_background_mask,
    _yaw_targets_from_row,
)
from rb_pipeline_v4.silhouette_algorithms import (
    ContourSilhouetteGeneratorV2,
    ConvexHullFallbackV1,
    FilledArtifactWriterV1,
    OutlineArtifactWriterV1,
)
from roi_fcn_training_v0_1.geometry import decode_heatmap_argmax, derive_roi_bounds
from roi_fcn_training_v0_1.topologies import build_model_from_spec, resolve_topology_spec
from src.data import Batch
from src.evaluate import _load_model_from_run
from src.task_runtime import (
    batch_targets_to_tensor,
    batch_to_model_inputs,
    extract_prediction_heads,
    extract_target_heads,
    summarize_task_metrics,
)
from src.utils import read_json, utc_now_iso, write_json


@dataclass(frozen=True)
class ModelContext:
    """Loaded distance-orientation model metadata for one inference run."""

    label: str
    run_dir: Path
    checkpoint_path: Path
    device: str
    run_config: dict[str, Any]
    run_manifest: dict[str, Any]
    task_contract: dict[str, Any]
    preprocessing_contract: dict[str, Any]


@dataclass(frozen=True)
class RoiFcnModelContext:
    """Loaded ROI-FCN metadata for one inference run."""

    label: str
    run_dir: Path
    checkpoint_path: Path
    device: str
    run_config: dict[str, Any]
    dataset_contract: dict[str, Any]
    canvas_width_px: int
    canvas_height_px: int
    roi_width_px: int
    roi_height_px: int


@dataclass(frozen=True)
class RoiFcnLocatorInput:
    """Prepared ROI-FCN locator canvas plus traceability geometry."""

    locator_image: np.ndarray
    source_image_wh_px: np.ndarray
    resized_image_wh_px: np.ndarray
    padding_ltrb_px: np.ndarray
    resize_scale: float


@dataclass(frozen=True)
class PreprocessedSample:
    """In-memory model input plus associated truth and source metadata."""

    sample_row: dict[str, Any]
    source_image_path: Path
    source_run_json_path: Path
    source_samples_csv_path: Path
    roi_image: np.ndarray
    model_image: np.ndarray
    bbox_features: np.ndarray
    actual_distance_m: float
    actual_orientation_deg: float
    actual_yaw_sin: float
    actual_yaw_cos: float
    predicted_crop_center_x_px: float
    predicted_crop_center_y_px: float
    predicted_roi_request_xyxy_px: np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    """Notebook-facing inference output payload."""

    selected_model_label: str
    selected_roi_model_label: str
    selected_corpus_name: str
    selected_image_name: str
    sample_id: str
    roi_image: np.ndarray
    predicted_crop_center_x_px: float
    predicted_crop_center_y_px: float
    predicted_roi_request_xyxy_px: np.ndarray
    predicted_distance_m: float
    actual_distance_m: float
    distance_delta_m: float
    absolute_distance_error_m: float
    predicted_orientation_deg: float
    actual_orientation_deg: float
    orientation_delta_deg: float
    absolute_orientation_error_deg: float
    device: str
    roi_device: str
    run_dir: Path
    checkpoint_path: Path
    roi_run_dir: Path
    roi_checkpoint_path: Path
    source_image_path: Path
    source_run_json_path: Path
    source_samples_csv_path: Path
    preprocessing_contract_version: str
    saved_json_path: Path | None = None
    saved_roi_path: Path | None = None

    def to_json_payload(self) -> dict[str, Any]:
        """Serialize result details for the JSON save artifact."""
        payload: dict[str, Any] = {
            "created_utc": utc_now_iso(),
            "selected_model": {
                "label": self.selected_model_label,
                "run_dir": to_repo_relative(self.run_dir),
                "checkpoint_path": to_repo_relative(self.checkpoint_path),
            },
            "selected_models": {
                "distance_orientation": {
                    "label": self.selected_model_label,
                    "run_dir": to_repo_relative(self.run_dir),
                    "checkpoint_path": to_repo_relative(self.checkpoint_path),
                },
                "roi_fcn": {
                    "label": self.selected_roi_model_label,
                    "run_dir": to_repo_relative(self.roi_run_dir),
                    "checkpoint_path": to_repo_relative(self.roi_checkpoint_path),
                },
            },
            "selected_corpus": {
                "name": self.selected_corpus_name,
                "source_run_json_path": to_repo_relative(self.source_run_json_path),
                "source_samples_csv_path": to_repo_relative(self.source_samples_csv_path),
            },
            "selected_image": {
                "image_filename": self.selected_image_name,
                "sample_id": self.sample_id,
                "source_image_path": to_repo_relative(self.source_image_path),
            },
            "roi_prediction": {
                "center_original_xy_px": [
                    float(self.predicted_crop_center_x_px),
                    float(self.predicted_crop_center_y_px),
                ],
                "request_xyxy_px": [
                    float(value) for value in self.predicted_roi_request_xyxy_px.tolist()
                ],
            },
            "prediction": {
                "distance_m": float(self.predicted_distance_m),
                "orientation_deg": float(self.predicted_orientation_deg),
            },
            "actual": {
                "distance_m": float(self.actual_distance_m),
                "orientation_deg": float(self.actual_orientation_deg),
            },
            "deltas": {
                "distance_signed_m": float(self.distance_delta_m),
                "distance_absolute_m": float(self.absolute_distance_error_m),
                "orientation_signed_deg": float(self.orientation_delta_deg),
                "orientation_absolute_deg": float(self.absolute_orientation_error_deg),
            },
            "model_input": {
                "shape": list(self.roi_image.shape),
                "dtype": str(self.roi_image.dtype),
                "preprocessing_contract_version": self.preprocessing_contract_version,
            },
            "runtime": {
                "distance_orientation_device": self.device,
                "roi_fcn_device": self.roi_device,
            },
        }
        if self.saved_roi_path is not None:
            payload["artifacts"] = {
                "roi_image_path": to_repo_relative(self.saved_roi_path),
            }
        return payload


def _resolve_checkpoint_path(
    run_dir: Path,
    *,
    candidates: tuple[str, ...] = ("best.pt", "best_model.pt", "latest.pt"),
) -> Path:
    for candidate_name in candidates:
        candidate = run_dir / candidate_name
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find a checkpoint under {run_dir}; checked {list(candidates)}"
    )


def _label_from_run_dir(run_dir: Path, *, family: str) -> str:
    return f"{family} / {run_dir.parent.parent.name} / {run_dir.name}"


def _resolve_preprocessing_contract(
    run_manifest: Mapping[str, Any],
    run_config: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = [
        run_manifest.get("dataset_summary", {}).get("preprocessing_contract"),
        run_manifest.get("preprocessing_contract"),
        run_config.get("dataset_summary", {}).get("preprocessing_contract"),
        run_config.get("preprocessing_contract"),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def _stage_parameters(
    preprocessing_contract: Mapping[str, Any],
    stage_name: str,
) -> dict[str, Any]:
    stages = preprocessing_contract.get("Stages")
    if not isinstance(stages, Mapping):
        return {}
    stage = stages.get(stage_name)
    return dict(stage) if isinstance(stage, Mapping) else {}


def _silhouette_config_from_contract(preprocessing_contract: Mapping[str, Any]) -> SilhouetteStageConfigV4:
    stage = _stage_parameters(preprocessing_contract, "silhouette")
    return SilhouetteStageConfigV4(
        representation_mode=str(stage.get("RepresentationMode", "filled")).strip().lower() or "filled",
        generator_id=str(stage.get("GeneratorId", "silhouette.contour_v2")).strip() or "silhouette.contour_v2",
        fallback_id=str(stage.get("FallbackId", "fallback.convex_hull_v1")).strip() or "fallback.convex_hull_v1",
        roi_padding_px=int(stage.get("ROIPaddingPx", 0)),
        roi_canvas_width_px=int(stage.get("ROICanvasWidthPx", 300)),
        roi_canvas_height_px=int(stage.get("ROICanvasHeightPx", 300)),
        blur_kernel_size=int(stage.get("BlurKernelSize", 5)),
        canny_low_threshold=int(stage.get("CannyLowThreshold", 50)),
        canny_high_threshold=int(stage.get("CannyHighThreshold", 150)),
        close_kernel_size=int(stage.get("CloseKernelSize", 1)),
        dilate_kernel_size=int(stage.get("DilateKernelSize", 1)),
        min_component_area_px=int(stage.get("MinComponentAreaPx", 50)),
        outline_thickness=int(stage.get("OutlineThicknessPx", 1)),
        fill_holes=bool(stage.get("FillHoles", True)),
        use_convex_hull_fallback=bool(stage.get("UseConvexHullFallback", True)),
        overwrite=True,
        dry_run=False,
        continue_on_error=False,
        persist_debug=bool(stage.get("PersistDebug", False)),
        sample_offset=0,
        sample_limit=1,
    )


def _pack_settings_from_contract(preprocessing_contract: Mapping[str, Any]) -> dict[str, Any]:
    current_representation = preprocessing_contract.get("CurrentRepresentation")
    current = dict(current_representation) if isinstance(current_representation, Mapping) else {}
    stage = _stage_parameters(preprocessing_contract, "pack_dual_stream")
    return {
        "canvas_width_px": int(stage.get("CanvasWidth", current.get("CanvasWidth", 300))),
        "canvas_height_px": int(stage.get("CanvasHeight", current.get("CanvasHeight", 300))),
        "clip_policy": str(stage.get("ClipPolicy", "fail")).strip().lower() or "fail",
    }


def load_model_context(
    model_run_dir: str | Path,
    *,
    device: str | None = None,
) -> tuple[torch.nn.Module, ModelContext]:
    """Load one trained distance-orientation model plus the metadata it expects."""
    run_dir = Path(model_run_dir).expanduser().resolve()
    run_config = read_json(run_dir / "config.json")
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = read_json(run_manifest_path) if run_manifest_path.exists() else {}

    device_text = str(device).strip() if device is not None else ""
    if not device_text:
        device_text = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device_text)

    model, topology_spec = _load_model_from_run(run_dir, run_config, device_obj)
    model.eval()

    context = ModelContext(
        label=_label_from_run_dir(run_dir, family="distance-orientation"),
        run_dir=run_dir,
        checkpoint_path=_resolve_checkpoint_path(run_dir, candidates=("best.pt", "best_model.pt", "latest.pt")),
        device=str(device_obj),
        run_config=run_config,
        run_manifest=run_manifest,
        task_contract=dict(topology_spec.task_contract),
        preprocessing_contract=_resolve_preprocessing_contract(run_manifest, run_config),
    )
    return model, context


def _dataset_contract_split(dataset_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("train_split", "validation_split"):
        candidate = dataset_contract.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def load_roi_fcn_model_context(
    model_run_dir: str | Path,
    *,
    device: str | None = None,
) -> tuple[torch.nn.Module, RoiFcnModelContext]:
    """Load one trained ROI-FCN model plus the dataset geometry it expects."""
    run_dir = Path(model_run_dir).expanduser().resolve()
    run_config = read_json(run_dir / "run_config.json")
    dataset_contract_path = run_dir / "dataset_contract.json"
    dataset_contract = read_json(dataset_contract_path) if dataset_contract_path.exists() else {}

    device_text = str(device).strip() if device is not None else ""
    if not device_text:
        device_text = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device_text)

    topology_params = run_config.get("topology_params")
    topology_params = topology_params if isinstance(topology_params, Mapping) else {}
    spec = resolve_topology_spec(
        topology_id=str(run_config.get("topology_id", "")).strip(),
        topology_variant=str(run_config.get("topology_variant", "")).strip(),
        topology_params=topology_params,
    )
    model = build_model_from_spec(spec).to(device_obj)
    checkpoint_path = _resolve_checkpoint_path(run_dir, candidates=("best.pt", "latest.pt"))
    state = torch.load(checkpoint_path, map_location=device_obj)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    split_contract = _dataset_contract_split(dataset_contract)
    geometry = split_contract.get("geometry") if isinstance(split_contract.get("geometry"), Mapping) else {}
    canvas_width_px = int(geometry.get("canvas_width_px", 0) or run_config.get("locator_canvas_width_px", 0) or 480)
    canvas_height_px = int(geometry.get("canvas_height_px", 0) or run_config.get("locator_canvas_height_px", 0) or 300)
    roi_width_px = int(run_config.get("roi_width_px", 0) or split_contract.get("fixed_roi_width_px", 0) or 300)
    roi_height_px = int(run_config.get("roi_height_px", 0) or split_contract.get("fixed_roi_height_px", 0) or 300)
    if canvas_width_px <= 0 or canvas_height_px <= 0:
        raise ValueError(
            f"Invalid ROI-FCN locator canvas size: {canvas_width_px}x{canvas_height_px}"
        )
    if roi_width_px <= 0 or roi_height_px <= 0:
        raise ValueError(f"Invalid ROI-FCN crop size: {roi_width_px}x{roi_height_px}")

    context = RoiFcnModelContext(
        label=_label_from_run_dir(run_dir, family="roi-fcn"),
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        device=str(device_obj),
        run_config=run_config,
        dataset_contract=dataset_contract,
        canvas_width_px=canvas_width_px,
        canvas_height_px=canvas_height_px,
        roi_width_px=roi_width_px,
        roi_height_px=roi_height_px,
    )
    return model, context


def _build_roi_fcn_locator_input(
    source_gray: np.ndarray,
    *,
    canvas_width_px: int,
    canvas_height_px: int,
) -> RoiFcnLocatorInput:
    if source_gray.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got {source_gray.shape}")

    src_h, src_w = int(source_gray.shape[0]), int(source_gray.shape[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source image shape: {source_gray.shape}")

    scale = min(float(canvas_width_px) / float(src_w), float(canvas_height_px) / float(src_h))
    resized_w = int(round(float(src_w) * scale))
    resized_h = int(round(float(src_h) * scale))
    if resized_w <= 0 or resized_h <= 0:
        raise ValueError(
            "Resized image dimensions must stay positive after aspect-preserving scale: "
            f"src={src_w}x{src_h}, scale={scale}"
        )

    pad_left = int((canvas_width_px - resized_w) // 2)
    pad_right = int(canvas_width_px - resized_w - pad_left)
    pad_top = int((canvas_height_px - resized_h) // 2)
    pad_bottom = int(canvas_height_px - resized_h - pad_top)
    if min(pad_left, pad_right, pad_top, pad_bottom) < 0:
        raise ValueError(
            "Computed negative padding; source image does not fit locator canvas: "
            f"src={src_w}x{src_h}, resized={resized_w}x{resized_h}, "
            f"canvas={canvas_width_px}x{canvas_height_px}"
        )

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_gray = cv2.resize(source_gray, (resized_w, resized_h), interpolation=interpolation)
    locator_canvas = np.zeros((canvas_height_px, canvas_width_px), dtype=np.float32)
    locator_canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = (
        resized_gray.astype(np.float32) / 255.0
    )

    return RoiFcnLocatorInput(
        locator_image=locator_canvas[None, ...].astype(np.float32),
        source_image_wh_px=np.asarray([src_w, src_h], dtype=np.int32),
        resized_image_wh_px=np.asarray([resized_w, resized_h], dtype=np.int32),
        padding_ltrb_px=np.asarray([pad_left, pad_top, pad_right, pad_bottom], dtype=np.int32),
        resize_scale=float(scale),
    )


def _predict_roi_center(
    model: torch.nn.Module,
    *,
    locator_input: RoiFcnLocatorInput,
    roi_context: RoiFcnModelContext,
) -> tuple[float, float, np.ndarray]:
    device_obj = torch.device(roi_context.device)
    input_tensor = torch.from_numpy(locator_input.locator_image[None, ...]).to(
        device=device_obj,
        dtype=torch.float32,
    )
    with torch.no_grad():
        heatmap_tensor = model(input_tensor)
    if heatmap_tensor.ndim != 4 or int(heatmap_tensor.shape[0]) != 1 or int(heatmap_tensor.shape[1]) != 1:
        raise ValueError(
            "ROI-FCN output must have shape (1, 1, H, W); "
            f"got {tuple(heatmap_tensor.shape)}"
        )

    heatmap = heatmap_tensor[0, 0].detach().cpu().numpy()
    decoded = decode_heatmap_argmax(
        heatmap,
        canvas_hw=(roi_context.canvas_height_px, roi_context.canvas_width_px),
        resize_scale=float(locator_input.resize_scale),
        pad_left_px=float(locator_input.padding_ltrb_px[0]),
        pad_top_px=float(locator_input.padding_ltrb_px[1]),
        source_wh_px=locator_input.source_image_wh_px,
    )
    center_x = float(decoded.original_x)
    center_y = float(decoded.original_y)
    request_xyxy = derive_roi_bounds(
        np.asarray([center_x, center_y], dtype=np.float32),
        roi_width_px=float(roi_context.roi_width_px),
        roi_height_px=float(roi_context.roi_height_px),
    )
    return center_x, center_y, np.asarray(request_xyxy, dtype=np.float32)


def _extract_centered_canvas(
    source_gray: np.ndarray,
    *,
    center_x_px: float,
    center_y_px: float,
    canvas_width_px: int,
    canvas_height_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract one fixed-size ROI canvas centred on the predicted ROI-FCN point."""
    if source_gray.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got {source_gray.shape}")

    frame_height = int(source_gray.shape[0])
    frame_width = int(source_gray.shape[1])
    canvas_w = max(1, int(canvas_width_px))
    canvas_h = max(1, int(canvas_height_px))

    req_x1 = int(round(float(center_x_px) - (canvas_w / 2.0)))
    req_y1 = int(round(float(center_y_px) - (canvas_h / 2.0)))
    req_x2 = req_x1 + canvas_w
    req_y2 = req_y1 + canvas_h

    src_x1 = max(0, req_x1)
    src_y1 = max(0, req_y1)
    src_x2 = min(frame_width, req_x2)
    src_y2 = min(frame_height, req_y2)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        raise ValueError("empty ROI after centered canvas extraction")

    dst_x1 = src_x1 - req_x1
    dst_y1 = src_y1 - req_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = source_gray[src_y1:src_y2, src_x1:src_x2]

    return (
        canvas,
        np.asarray([src_x1, src_y1, src_x2, src_y2], dtype=np.float32),
        np.asarray([dst_x1, dst_y1, dst_x2, dst_y2], dtype=np.float32),
        np.asarray([req_x1, req_y1, req_x2, req_y2], dtype=np.float32),
    )


def _select_silhouette_components(
    silhouette_config: SilhouetteStageConfigV4,
) -> tuple[ContourSilhouetteGeneratorV2, ConvexHullFallbackV1, FilledArtifactWriterV1 | OutlineArtifactWriterV1]:
    if silhouette_config.normalized_generator_id() != "silhouette.contour_v2":
        raise ValueError("Only generator_id='silhouette.contour_v2' is supported in v0.2 inference")
    if silhouette_config.normalized_fallback_id() != "fallback.convex_hull_v1":
        raise ValueError("Only fallback_id='fallback.convex_hull_v1' is supported in v0.2 inference")
    mode = silhouette_config.normalized_representation_mode()
    writer = FilledArtifactWriterV1() if mode == "filled" else OutlineArtifactWriterV1()
    return ContourSilhouetteGeneratorV2(), ConvexHullFallbackV1(), writer


def _contour_break_reason(contour: np.ndarray | None) -> str:
    if contour is None:
        return "no_contour"
    if contour.ndim != 3 or contour.shape[0] < 3:
        return "degenerate_contour"
    area = float(abs(cv2.contourArea(contour)))
    if area <= 0.0:
        return "degenerate_contour_area"
    return ""


def _render_is_empty(gray_image: np.ndarray) -> bool:
    return gray_image.ndim != 2 or not bool(np.any(gray_image < 255))


def _mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, (0, 0, 0, 0)
    return int(xs.size), (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _bbox_features_from_xyxy(
    bbox_xyxy: np.ndarray,
    *,
    image_width_px: int,
    image_height_px: int,
) -> np.ndarray:
    x1 = float(bbox_xyxy[0])
    y1 = float(bbox_xyxy[1])
    x2 = float(bbox_xyxy[2])
    y2 = float(bbox_xyxy[3])

    frame_w = max(1.0, float(image_width_px))
    frame_h = max(1.0, float(image_height_px))
    w_px = max(1e-6, x2 - x1)
    h_px = max(1e-6, y2 - y1)
    cx_px = x1 + (0.5 * w_px)
    cy_px = y1 + (0.5 * h_px)

    values = np.asarray(
        [
            cx_px,
            cy_px,
            w_px,
            h_px,
            cx_px / frame_w,
            cy_px / frame_h,
            w_px / frame_w,
            h_px / frame_h,
            w_px / h_px,
            (w_px * h_px) / (frame_w * frame_h),
        ],
        dtype=np.float32,
    )
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("bbox feature vector contains NaN or Inf")
    return values


def _validate_model_compatibility(
    *,
    model_context: ModelContext,
    roi_context: RoiFcnModelContext,
) -> tuple[SilhouetteStageConfigV4, dict[str, Any]]:
    silhouette_config = _silhouette_config_from_contract(model_context.preprocessing_contract)
    pack_settings = _pack_settings_from_contract(model_context.preprocessing_contract)
    if (
        int(roi_context.roi_width_px) != int(silhouette_config.normalized_roi_canvas_width_px())
        or int(roi_context.roi_height_px) != int(silhouette_config.normalized_roi_canvas_height_px())
    ):
        raise ValueError(
            "Selected ROI-FCN crop size is incompatible with the distance model silhouette canvas: "
            f"roi_fcn={roi_context.roi_width_px}x{roi_context.roi_height_px}, "
            f"distance_model={silhouette_config.normalized_roi_canvas_width_px()}x"
            f"{silhouette_config.normalized_roi_canvas_height_px()}"
        )
    return silhouette_config, pack_settings


def preprocess_single_sample(
    *,
    corpus: RawCorpus,
    sample_row: pd.Series,
    model_context: ModelContext,
    roi_model: torch.nn.Module,
    roi_model_context: RoiFcnModelContext,
) -> PreprocessedSample:
    """Use ROI-FCN for crop placement, then render the dual-stream ROI input."""
    silhouette_config, pack_settings = _validate_model_compatibility(
        model_context=model_context,
        roi_context=roi_model_context,
    )
    source_image_path = Path(str(sample_row["__image_path__"])).resolve()
    source_gray = read_grayscale_uint8(source_image_path)

    locator_input = _build_roi_fcn_locator_input(
        source_gray,
        canvas_width_px=int(roi_model_context.canvas_width_px),
        canvas_height_px=int(roi_model_context.canvas_height_px),
    )
    predicted_center_x_px, predicted_center_y_px, _ = _predict_roi_center(
        roi_model,
        locator_input=locator_input,
        roi_context=roi_model_context,
    )

    roi_gray, source_bounds, roi_bounds, request_bounds = _extract_centered_canvas(
        source_gray,
        center_x_px=predicted_center_x_px,
        center_y_px=predicted_center_y_px,
        canvas_width_px=int(silhouette_config.normalized_roi_canvas_width_px()),
        canvas_height_px=int(silhouette_config.normalized_roi_canvas_height_px()),
    )

    generator, fallback, writer = _select_silhouette_components(silhouette_config)
    generated = generator.generate(
        roi_gray,
        blur_kernel_size=silhouette_config.normalized_blur_kernel_size(),
        canny_low_threshold=int(silhouette_config.canny_low_threshold),
        canny_high_threshold=int(silhouette_config.canny_high_threshold),
        close_kernel_size=silhouette_config.normalized_close_kernel_size(),
        dilate_kernel_size=silhouette_config.normalized_dilate_kernel_size(),
        min_component_area_px=silhouette_config.normalized_min_component_area_px(),
        fill_holes=bool(silhouette_config.fill_holes),
    )

    contour = generated.contour
    primary_break_reason = _contour_break_reason(contour)
    fallback_used = False
    if primary_break_reason:
        if not bool(silhouette_config.use_convex_hull_fallback):
            break_reason = generated.primary_reason or primary_break_reason
            raise ValueError(f"Primary contour failed ({break_reason}) and fallback is disabled")
        contour, recovery_reason = fallback.recover(generated.fallback_mask)
        fallback_used = True
        if contour is None:
            raise ValueError(f"Fallback failed: {recovery_reason}")

    roi_silhouette = writer.render(
        roi_gray.shape,
        contour,
        line_thickness=silhouette_config.normalized_outline_thickness(),
    )
    if _render_is_empty(roi_silhouette):
        if not fallback_used and bool(silhouette_config.use_convex_hull_fallback):
            contour, recovery_reason = fallback.recover(generated.fallback_mask)
            fallback_used = True
            if contour is None:
                raise ValueError(f"Fallback failed: {recovery_reason}")
            roi_silhouette = writer.render(
                roi_gray.shape,
                contour,
                line_thickness=silhouette_config.normalized_outline_thickness(),
            )
        if _render_is_empty(roi_silhouette):
            raise ValueError("Rendered silhouette is empty after fallback")

    src_x1, src_y1, src_x2, src_y2 = [int(value) for value in source_bounds.tolist()]
    roi_x1, roi_y1, roi_x2, roi_y2 = [int(value) for value in roi_bounds.tolist()]
    full_silhouette = np.full(source_gray.shape, 255, dtype=np.uint8)
    roi_target = full_silhouette[src_y1:src_y2, src_x1:src_x2]
    roi_source_aligned = roi_silhouette[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_target[roi_source_aligned < 255] = 0
    full_silhouette[src_y1:src_y2, src_x1:src_x2] = roi_target

    area_px, bbox = _mask_geometry(full_silhouette < 255)
    if area_px > 0:
        feature_bbox_xyxy = np.asarray(
            [
                float(bbox[0]),
                float(bbox[1]),
                float(min(int(source_gray.shape[1]), bbox[2] + 1)),
                float(min(int(source_gray.shape[0]), bbox[3] + 1)),
            ],
            dtype=np.float32,
        )
    else:
        feature_bbox_xyxy = np.asarray(source_bounds, dtype=np.float32)

    background_mask = _silhouette_to_background_mask(roi_silhouette)
    roi_source_gray = _reconstruct_roi_canvas_from_source(
        source_gray,
        source_xyxy=np.asarray(source_bounds, dtype=np.float32),
        canvas_insert_xyxy=np.asarray(roi_bounds, dtype=np.float32),
        canvas_width=int(silhouette_config.normalized_roi_canvas_width_px()),
        canvas_height=int(silhouette_config.normalized_roi_canvas_height_px()),
    )
    roi_repr = _render_inverted_vehicle_detail_on_white(
        roi_source_gray,
        background_mask,
    )
    canvas, _ = _place_image_on_canvas(
        roi_repr,
        canvas_height=int(pack_settings["canvas_height_px"]),
        canvas_width=int(pack_settings["canvas_width_px"]),
        clip_policy=str(pack_settings["clip_policy"]),
    )

    bbox_features = _bbox_features_from_xyxy(
        feature_bbox_xyxy,
        image_width_px=int(source_gray.shape[1]),
        image_height_px=int(source_gray.shape[0]),
    )
    yaw_deg, yaw_sin, yaw_cos = _yaw_targets_from_row(sample_row)
    row_payload = dict(sample_row.to_dict())
    row_payload.update(
        {
            "yaw_deg": float(yaw_deg),
            "yaw_sin": float(yaw_sin),
            "yaw_cos": float(yaw_cos),
            "image_width_px": int(source_gray.shape[1]),
            "image_height_px": int(source_gray.shape[0]),
            "detect_bbox_x1": float(feature_bbox_xyxy[0]),
            "detect_bbox_y1": float(feature_bbox_xyxy[1]),
            "detect_bbox_x2": float(feature_bbox_xyxy[2]),
            "detect_bbox_y2": float(feature_bbox_xyxy[3]),
            "detect_center_x_px": float(predicted_center_x_px),
            "detect_center_y_px": float(predicted_center_y_px),
            "silhouette_bbox_x1": int(bbox[0]) if area_px > 0 else "",
            "silhouette_bbox_y1": int(bbox[1]) if area_px > 0 else "",
            "silhouette_bbox_x2": int(bbox[2]) if area_px > 0 else "",
            "silhouette_bbox_y2": int(bbox[3]) if area_px > 0 else "",
            "silhouette_roi_request_x1_px": float(request_bounds[0]),
            "silhouette_roi_request_y1_px": float(request_bounds[1]),
            "silhouette_roi_request_x2_px": float(request_bounds[2]),
            "silhouette_roi_request_y2_px": float(request_bounds[3]),
            "silhouette_roi_source_x1_px": float(source_bounds[0]),
            "silhouette_roi_source_y1_px": float(source_bounds[1]),
            "silhouette_roi_source_x2_px": float(source_bounds[2]),
            "silhouette_roi_source_y2_px": float(source_bounds[3]),
            "silhouette_roi_canvas_x1_px": float(roi_bounds[0]),
            "silhouette_roi_canvas_y1_px": float(roi_bounds[1]),
            "silhouette_roi_canvas_x2_px": float(roi_bounds[2]),
            "silhouette_roi_canvas_y2_px": float(roi_bounds[3]),
        }
    )

    return PreprocessedSample(
        sample_row=row_payload,
        source_image_path=source_image_path,
        source_run_json_path=corpus.run_json_path,
        source_samples_csv_path=corpus.samples_csv_path,
        roi_image=canvas.astype(np.float32),
        model_image=canvas[None, ...].astype(np.float32),
        bbox_features=bbox_features.astype(np.float32),
        actual_distance_m=float(sample_row["distance_m"]),
        actual_orientation_deg=float(yaw_deg),
        actual_yaw_sin=float(yaw_sin),
        actual_yaw_cos=float(yaw_cos),
        predicted_crop_center_x_px=float(predicted_center_x_px),
        predicted_crop_center_y_px=float(predicted_center_y_px),
        predicted_roi_request_xyxy_px=np.asarray(request_bounds, dtype=np.float32),
    )


def _signed_orientation_delta_deg(predicted_deg: float, actual_deg: float) -> float:
    return float(((float(predicted_deg) - float(actual_deg) + 180.0) % 360.0) - 180.0)


def _build_single_sample_batch(preprocessed: PreprocessedSample) -> Batch:
    targets = np.asarray(
        [
            [
                preprocessed.actual_distance_m,
                preprocessed.actual_yaw_sin,
                preprocessed.actual_yaw_cos,
            ]
        ],
        dtype=np.float32,
    )
    return Batch(
        images=np.expand_dims(preprocessed.model_image, axis=0).astype(np.float32),
        targets=targets,
        rows=[dict(preprocessed.sample_row)],
        bbox_features=np.expand_dims(preprocessed.bbox_features, axis=0).astype(np.float32),
    )


def _build_multi_sample_batch(preprocessed_samples: list[PreprocessedSample]) -> Batch:
    if not preprocessed_samples:
        raise ValueError("At least one preprocessed sample is required.")

    return Batch(
        images=np.stack(
            [sample.model_image for sample in preprocessed_samples],
            axis=0,
        ).astype(np.float32),
        targets=np.asarray(
            [
                [
                    sample.actual_distance_m,
                    sample.actual_yaw_sin,
                    sample.actual_yaw_cos,
                ]
                for sample in preprocessed_samples
            ],
            dtype=np.float32,
        ),
        rows=[dict(sample.sample_row) for sample in preprocessed_samples],
        bbox_features=np.stack(
            [sample.bbox_features for sample in preprocessed_samples],
            axis=0,
        ).astype(np.float32),
    )


def _resolve_raw_corpus(corpus_dir: str | Path) -> RawCorpus:
    corpus_root = Path(corpus_dir).expanduser().resolve()
    return RawCorpus(
        name=corpus_root.name,
        root=corpus_root,
        images_dir=(corpus_root / "images").resolve(),
        run_json_path=(corpus_root / "manifests" / "run.json").resolve(),
        samples_csv_path=(corpus_root / "manifests" / "samples.csv").resolve(),
    )


def _run_prediction_batch(
    *,
    model: torch.nn.Module,
    batch: Batch,
    model_context: ModelContext,
) -> pd.DataFrame:
    device_obj = torch.device(model_context.device)
    with torch.no_grad():
        model_inputs = batch_to_model_inputs(batch, model_context.task_contract, device=device_obj)
        targets = batch_targets_to_tensor(batch, device=device_obj)
        outputs = model(model_inputs)
        prediction_heads = extract_prediction_heads(outputs, model_context.task_contract)
        target_heads = extract_target_heads(targets, model_context.task_contract)

    prediction_arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in prediction_heads.items()
    }
    target_arrays = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in target_heads.items()
    }
    metrics = summarize_task_metrics(
        prediction_heads=prediction_arrays,
        target_heads=target_arrays,
        task_contract=model_context.task_contract,
        tolerance_values=(0.10,),
        primary_tolerance=0.10,
        rows=batch.rows,
        collect_predictions=True,
    )
    if metrics.predictions is None or metrics.predictions.empty:
        raise RuntimeError("Inference did not produce any prediction rows.")
    return metrics.predictions.reset_index(drop=True)


def _build_inference_result(
    *,
    preprocessed: PreprocessedSample,
    prediction_row: pd.Series,
    model_context: ModelContext,
    roi_model_context: RoiFcnModelContext,
) -> InferenceResult:
    predicted_distance_m = float(prediction_row["prediction_distance_m"])
    actual_distance_m = float(prediction_row["truth_distance_m"])
    predicted_orientation_deg = float(prediction_row["prediction_yaw_deg"])
    actual_orientation_deg = float(prediction_row["truth_yaw_deg"])

    return InferenceResult(
        selected_model_label=model_context.label,
        selected_roi_model_label=roi_model_context.label,
        selected_corpus_name=str(preprocessed.source_run_json_path.parent.parent.name),
        selected_image_name=str(preprocessed.sample_row["image_filename"]),
        sample_id=str(preprocessed.sample_row["sample_id"]),
        roi_image=preprocessed.roi_image,
        predicted_crop_center_x_px=preprocessed.predicted_crop_center_x_px,
        predicted_crop_center_y_px=preprocessed.predicted_crop_center_y_px,
        predicted_roi_request_xyxy_px=preprocessed.predicted_roi_request_xyxy_px,
        predicted_distance_m=predicted_distance_m,
        actual_distance_m=actual_distance_m,
        distance_delta_m=float(predicted_distance_m - actual_distance_m),
        absolute_distance_error_m=float(abs(predicted_distance_m - actual_distance_m)),
        predicted_orientation_deg=predicted_orientation_deg,
        actual_orientation_deg=actual_orientation_deg,
        orientation_delta_deg=_signed_orientation_delta_deg(
            predicted_orientation_deg,
            actual_orientation_deg,
        ),
        absolute_orientation_error_deg=float(prediction_row["angular_error_deg"]),
        device=model_context.device,
        roi_device=roi_model_context.device,
        run_dir=model_context.run_dir,
        checkpoint_path=model_context.checkpoint_path,
        roi_run_dir=roi_model_context.run_dir,
        roi_checkpoint_path=roi_model_context.checkpoint_path,
        source_image_path=preprocessed.source_image_path,
        source_run_json_path=preprocessed.source_run_json_path,
        source_samples_csv_path=preprocessed.source_samples_csv_path,
        preprocessing_contract_version=str(
            model_context.preprocessing_contract.get("ContractVersion", "")
        ).strip(),
    )


def _model_output_name(run_dir: str | Path) -> str:
    """Derive a stable filesystem-safe name for aggregated inference artifacts."""
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    if (
        resolved_run_dir.name.startswith("run_")
        and resolved_run_dir.parent.name == "runs"
        and resolved_run_dir.parent.parent.name
    ):
        return sanitize_identifier(resolved_run_dir.parent.parent.name)
    return sanitize_identifier(resolved_run_dir.name)


def save_inference_result(
    result: InferenceResult,
    *,
    root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write the optional JSON result artifact and ROI image."""
    target_root = Path(root).expanduser().resolve() if root is not None else results_root()
    target_root.mkdir(parents=True, exist_ok=True)

    model_output_name = _model_output_name(result.run_dir)
    stem = "__".join(
        [
            timestamp_slug(),
            sanitize_identifier(result.selected_corpus_name),
            sanitize_identifier(result.sample_id),
        ]
    )
    roi_path = target_root / f"{stem}.roi.png"
    json_path = target_root / f"inference-output_{model_output_name}.json"

    write_grayscale_png(roi_path, np.clip(result.roi_image * 255.0, 0, 255).astype(np.uint8))
    payload = result.to_json_payload()
    payload["artifacts"] = {
        "json_path": to_repo_relative(json_path),
        "roi_image_path": to_repo_relative(roi_path),
    }
    existing_payloads: list[dict[str, Any]] = []
    if json_path.exists():
        existing_content = read_json(json_path)
        if isinstance(existing_content, list):
            existing_payloads = existing_content
        elif isinstance(existing_content, dict):
            existing_payloads = [existing_content]
        else:
            raise ValueError(f"Unsupported JSON payload in existing inference output: {json_path}")
    existing_payloads.append(payload)
    write_json(json_path, existing_payloads)
    return json_path, roi_path


def run_single_sample_inference(
    model_run_dir: str | Path,
    corpus_dir: str | Path,
    image_name: str,
    *,
    roi_model_run_dir: str | Path | None = None,
    save_result: bool = False,
    results_root_path: str | Path | None = None,
    device: str | None = None,
) -> InferenceResult:
    """Run one end-to-end raw-image inference pass using ROI-FCN crop placement."""
    if roi_model_run_dir is None:
        raise ValueError("roi_model_run_dir is required for v0.2 inference.")

    corpus = _resolve_raw_corpus(corpus_dir)

    from .discovery import select_sample_row

    sample_row = select_sample_row(corpus, image_name)
    model, model_context = load_model_context(model_run_dir, device=device)
    roi_model, roi_model_context = load_roi_fcn_model_context(roi_model_run_dir, device=device)
    preprocessed = preprocess_single_sample(
        corpus=corpus,
        sample_row=sample_row,
        model_context=model_context,
        roi_model=roi_model,
        roi_model_context=roi_model_context,
    )
    prediction_rows = _run_prediction_batch(
        model=model,
        batch=_build_single_sample_batch(preprocessed),
        model_context=model_context,
    )
    result = _build_inference_result(
        preprocessed=preprocessed,
        prediction_row=prediction_rows.iloc[0],
        model_context=model_context,
        roi_model_context=roi_model_context,
    )

    if save_result:
        json_path, roi_path = save_inference_result(
            result,
            root=results_root_path,
        )
        result = replace(
            result,
            saved_json_path=json_path,
            saved_roi_path=roi_path,
        )
    return result


def run_multi_sample_inference(
    model_run_dir: str | Path,
    corpus_dir: str | Path,
    *,
    roi_model_run_dir: str | Path | None = None,
    offset: int = 0,
    num_samples: int = 1,
    save_result: bool = False,
    results_root_path: str | Path | None = None,
    device: str | None = None,
) -> list[InferenceResult]:
    """Run one corpus slice without reloading the models for each sample."""
    if roi_model_run_dir is None:
        raise ValueError("roi_model_run_dir is required for v0.2 inference.")

    offset_value = int(offset)
    num_samples_value = int(num_samples)
    if offset_value < 0:
        raise ValueError(f"offset must be >= 0, got {offset_value}")
    if num_samples_value <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples_value}")

    corpus = _resolve_raw_corpus(corpus_dir)
    samples_df = load_corpus_samples(corpus)
    total_samples = int(len(samples_df))
    if total_samples <= 0:
        raise ValueError(f"Corpus {corpus.name} has no selectable samples.")
    if offset_value >= total_samples:
        raise ValueError(
            f"offset {offset_value} is out of range for corpus {corpus.name} "
            f"with {total_samples} selectable samples"
        )

    selected_df = samples_df.iloc[offset_value : offset_value + num_samples_value].copy()
    if selected_df.empty:
        raise RuntimeError("Resolved an empty sample slice for multi-sample inference.")

    model, model_context = load_model_context(model_run_dir, device=device)
    roi_model, roi_model_context = load_roi_fcn_model_context(roi_model_run_dir, device=device)

    preprocessed_samples = [
        preprocess_single_sample(
            corpus=corpus,
            sample_row=sample_row,
            model_context=model_context,
            roi_model=roi_model,
            roi_model_context=roi_model_context,
        )
        for _, sample_row in selected_df.iterrows()
    ]
    prediction_rows = _run_prediction_batch(
        model=model,
        batch=_build_multi_sample_batch(preprocessed_samples),
        model_context=model_context,
    )
    if len(prediction_rows) != len(preprocessed_samples):
        raise RuntimeError(
            "Prediction row count did not match the requested sample slice: "
            f"predictions={len(prediction_rows)}, samples={len(preprocessed_samples)}"
        )

    results: list[InferenceResult] = []
    for index, preprocessed in enumerate(preprocessed_samples):
        result = _build_inference_result(
            preprocessed=preprocessed,
            prediction_row=prediction_rows.iloc[index],
            model_context=model_context,
            roi_model_context=roi_model_context,
        )
        if save_result:
            json_path, roi_path = save_inference_result(
                result,
                root=results_root_path,
            )
            result = replace(
                result,
                saved_json_path=json_path,
                saved_roi_path=roi_path,
            )
        results.append(result)
    return results
