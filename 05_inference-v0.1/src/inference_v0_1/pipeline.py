"""Single-sample raw-image inference pipeline for v0.1."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .discovery import RawCorpus
from .external import ensure_external_paths
from .paths import results_root, sanitize_identifier, timestamp_slug, to_repo_relative

ensure_external_paths()

from rb_pipeline_v4.config import DetectStageConfigV4, SilhouetteStageConfigV4
from rb_pipeline_v4.detect_stage import run_detect_stage_v4
from rb_pipeline_v4.image_io import read_grayscale_uint8, write_grayscale_png
from rb_pipeline_v4.pack_dual_stream_stage import (
    _bbox_features_from_row,
    _place_image_on_canvas,
    _reconstruct_roi_canvas_from_source,
    _render_inverted_vehicle_detail_on_white,
    _roi_geometry_from_row,
    _silhouette_to_background_mask,
    _yaw_targets_from_row,
)
from rb_pipeline_v4.paths import input_run_paths, resolve_manifest_path, silhouette_run_paths
from rb_pipeline_v4.silhouette_stage import run_silhouette_stage_v4
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
    """Loaded model metadata required for one inference run."""

    label: str
    run_dir: Path
    checkpoint_path: Path
    device: str
    run_config: dict[str, Any]
    run_manifest: dict[str, Any]
    task_contract: dict[str, Any]
    preprocessing_contract: dict[str, Any]


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


@dataclass(frozen=True)
class InferenceResult:
    """Notebook-facing inference output payload."""

    selected_model_label: str
    selected_corpus_name: str
    selected_image_name: str
    sample_id: str
    roi_image: np.ndarray
    predicted_distance_m: float
    actual_distance_m: float
    distance_delta_m: float
    absolute_distance_error_m: float
    predicted_orientation_deg: float
    actual_orientation_deg: float
    orientation_delta_deg: float
    absolute_orientation_error_deg: float
    device: str
    run_dir: Path
    checkpoint_path: Path
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
                "device": self.device,
            },
        }
        if self.saved_roi_path is not None:
            payload["artifacts"] = {
                "roi_image_path": to_repo_relative(self.saved_roi_path),
            }
        return payload


def _resolve_checkpoint_path(run_dir: Path) -> Path:
    for candidate in (run_dir / "best.pt", run_dir / "best_model.pt"):
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find a best checkpoint under {run_dir}")


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


def _detect_config_from_contract(preprocessing_contract: Mapping[str, Any]) -> DetectStageConfigV4:
    stage = _stage_parameters(preprocessing_contract, "detect")
    return DetectStageConfigV4(
        detector_backend=str(stage.get("DetectorBackend", "edge")).strip().lower() or "edge",
        model_path=str(stage.get("ModelPath", "")).strip(),
        default_model_ref=str(stage.get("DefaultModelRef", "yolov8n.pt")).strip(),
        defender_class_ids=tuple(int(value) for value in stage.get("DefenderClassIds", ()) or ()),
        defender_class_names=tuple(str(value) for value in stage.get("DefenderClassNames", ()) or ()),
        conf_threshold=float(stage.get("ConfidenceThreshold", 0.10)),
        iou_threshold=float(stage.get("IoUThreshold", 0.70)),
        max_det=int(stage.get("MaxDetections", 32)),
        imgsz=int(stage.get("ImageSize", 1280)),
        device=str(stage.get("Device", "")).strip(),
        edge_blur_kernel_size=int(stage.get("EdgeBlurKernelSize", 5)),
        edge_canny_low_threshold=int(stage.get("EdgeCannyLowThreshold", 50)),
        edge_canny_high_threshold=int(stage.get("EdgeCannyHighThreshold", 150)),
        edge_foreground_threshold=int(stage.get("EdgeForegroundThreshold", 250)),
        edge_padding_px=int(stage.get("EdgePaddingPx", 0)),
        edge_min_foreground_px=int(stage.get("EdgeMinForegroundPx", 16)),
        edge_close_kernel_size=int(stage.get("EdgeCloseKernelSize", 1)),
        overwrite=True,
        dry_run=False,
        continue_on_error=False,
        persist_debug=bool(stage.get("PersistDebug", False)),
        sample_offset=0,
        sample_limit=1,
    )


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
    """Load one trained model plus the preprocessing/task metadata it expects."""
    run_dir = Path(model_run_dir).expanduser().resolve()
    run_config = read_json(run_dir / "config.json")
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = read_json(run_manifest_path) if run_manifest_path.exists() else {}

    device_text = str(device).strip() if device is not None else ""
    if not device_text:
        device_text = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device_text)

    model, topology_spec = _load_model_from_run(run_dir, run_config, device_obj)

    context = ModelContext(
        label=f"{run_dir.parent.parent.name} / {run_dir.name}",
        run_dir=run_dir,
        checkpoint_path=_resolve_checkpoint_path(run_dir),
        device=str(device_obj),
        run_config=run_config,
        run_manifest=run_manifest,
        task_contract=dict(topology_spec.task_contract),
        preprocessing_contract=_resolve_preprocessing_contract(run_manifest, run_config),
    )
    return model, context


def _link_or_copy_file(source_path: Path, target_path: Path) -> None:
    try:
        os.symlink(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _write_single_sample_input_run(
    *,
    corpus: RawCorpus,
    sample_row: pd.Series,
    project_root: Path,
    run_name: str,
) -> Path:
    input_run_root = project_root / "input-images" / run_name
    images_dir = input_run_root / "images"
    manifests_dir = input_run_root / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    image_name = str(sample_row["__image_name__"]).strip()
    source_image_path = Path(str(sample_row["__image_path__"])).resolve()
    target_image_path = images_dir / image_name
    _link_or_copy_file(source_image_path, target_image_path)

    payload = read_json(corpus.run_json_path)
    payload["RunId"] = run_name
    payload["RunRootPath"] = str(input_run_root.resolve())
    payload["ImagesDirectoryPath"] = str(images_dir.resolve())
    payload["ManifestFilePath"] = str((manifests_dir / "samples.csv").resolve())
    payload["RunMetadataFilePath"] = str((manifests_dir / "run.json").resolve())

    single_row = pd.DataFrame(
        [
            {
                column: value
                for column, value in sample_row.to_dict().items()
                if not str(column).startswith("__")
            }
        ]
    )
    single_row.to_csv(manifests_dir / "samples.csv", index=False)
    with (manifests_dir / "run.json").open("w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=4)
        handle.write("\n")

    return input_run_root


def _stage_result_row(samples_csv_path: Path) -> pd.Series:
    samples_df = pd.read_csv(samples_csv_path, low_memory=False)
    if samples_df.empty:
        raise ValueError(f"Stage output manifest is empty: {samples_csv_path}")
    return samples_df.iloc[0].copy()


def preprocess_single_sample(
    *,
    corpus: RawCorpus,
    sample_row: pd.Series,
    model_context: ModelContext,
) -> PreprocessedSample:
    """Run the existing raw-image preprocessing path for one selected sample."""
    run_name = f"inference_{sanitize_identifier(sample_row.get('sample_id', sample_row['__image_name__']))}"
    detect_config = _detect_config_from_contract(model_context.preprocessing_contract)
    silhouette_config = _silhouette_config_from_contract(model_context.preprocessing_contract)
    pack_settings = _pack_settings_from_contract(model_context.preprocessing_contract)
    quiet_sink = lambda _message: None

    with TemporaryDirectory(prefix="rb-inference-single-sample-") as tmp_dir:
        project_root = Path(tmp_dir)
        _write_single_sample_input_run(
            corpus=corpus,
            sample_row=sample_row,
            project_root=project_root,
            run_name=run_name,
        )

        run_detect_stage_v4(project_root, run_name, detect_config, log_sink=quiet_sink)
        run_silhouette_stage_v4(project_root, run_name, silhouette_config, log_sink=quiet_sink)

        input_paths = input_run_paths(project_root, run_name)
        silhouette_paths = silhouette_run_paths(project_root, run_name)
        stage_row = _stage_result_row(silhouette_paths.manifests_dir / "samples.csv")

        detect_status = str(stage_row.get("detect_stage_status", "")).strip().lower()
        silhouette_status = str(stage_row.get("silhouette_stage_status", "")).strip().lower()
        if detect_status != "success":
            raise RuntimeError(
                f"Detect stage failed for {sample_row['__image_name__']}: "
                f"{stage_row.get('detect_stage_error', '')}"
            )
        if silhouette_status != "success":
            raise RuntimeError(
                f"Silhouette stage failed for {sample_row['__image_name__']}: "
                f"{stage_row.get('silhouette_stage_error', '')}"
            )

        roi_rel = str(stage_row["silhouette_roi_image_filename"]).strip()
        if not roi_rel:
            raise ValueError("silhouette_roi_image_filename is empty after silhouette stage.")

        roi_path = resolve_manifest_path(silhouette_paths.root, "images", roi_rel)
        roi_gray = read_grayscale_uint8(roi_path)
        background_mask = _silhouette_to_background_mask(roi_gray)

        source_image_path = resolve_manifest_path(
            input_paths.root,
            "images",
            stage_row["image_filename"],
        )
        source_gray = read_grayscale_uint8(source_image_path)

        roi_request_xyxy, roi_source_xyxy, roi_canvas_insert_xyxy, _ = _roi_geometry_from_row(
            stage_row,
            canvas_width=int(pack_settings["canvas_width_px"]),
            canvas_height=int(pack_settings["canvas_height_px"]),
        )
        roi_source_gray = _reconstruct_roi_canvas_from_source(
            source_gray,
            source_xyxy=roi_source_xyxy,
            canvas_insert_xyxy=roi_canvas_insert_xyxy,
            canvas_width=int(pack_settings["canvas_width_px"]),
            canvas_height=int(pack_settings["canvas_height_px"]),
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

        bbox_features = _bbox_features_from_row(stage_row)
        yaw_deg, yaw_sin, yaw_cos = _yaw_targets_from_row(stage_row)
        row_payload = dict(stage_row.to_dict())
        row_payload["yaw_deg"] = float(yaw_deg)
        row_payload["yaw_sin"] = float(yaw_sin)
        row_payload["yaw_cos"] = float(yaw_cos)

        return PreprocessedSample(
            sample_row=row_payload,
            source_image_path=Path(str(sample_row["__image_path__"])).resolve(),
            source_run_json_path=corpus.run_json_path,
            source_samples_csv_path=corpus.samples_csv_path,
            roi_image=canvas.astype(np.float32),
            model_image=canvas[None, ...].astype(np.float32),
            bbox_features=bbox_features.astype(np.float32),
            actual_distance_m=float(stage_row["distance_m"]),
            actual_orientation_deg=float(yaw_deg),
            actual_yaw_sin=float(yaw_sin),
            actual_yaw_cos=float(yaw_cos),
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


def save_inference_result(
    result: InferenceResult,
    *,
    root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write the optional JSON result artifact and ROI image."""
    target_root = Path(root).expanduser().resolve() if root is not None else results_root()
    target_root.mkdir(parents=True, exist_ok=True)

    stem = "__".join(
        [
            timestamp_slug(),
            sanitize_identifier(result.selected_corpus_name),
            sanitize_identifier(result.sample_id),
        ]
    )
    roi_path = target_root / f"{stem}.roi.png"
    json_path = target_root / f"{stem}.json"

    write_grayscale_png(roi_path, np.clip(result.roi_image * 255.0, 0, 255).astype(np.uint8))
    payload = result.to_json_payload()
    payload["artifacts"] = {
        "json_path": to_repo_relative(json_path),
        "roi_image_path": to_repo_relative(roi_path),
    }
    write_json(json_path, payload)
    return json_path, roi_path


def run_single_sample_inference(
    model_run_dir: str | Path,
    corpus_dir: str | Path,
    image_name: str,
    *,
    save_result: bool = False,
    results_root_path: str | Path | None = None,
    device: str | None = None,
) -> InferenceResult:
    """Run one end-to-end raw-image inference pass."""
    corpus_root = Path(corpus_dir).expanduser().resolve()
    corpus = RawCorpus(
        name=corpus_root.name,
        root=corpus_root,
        images_dir=(corpus_root / "images").resolve(),
        run_json_path=(corpus_root / "manifests" / "run.json").resolve(),
        samples_csv_path=(corpus_root / "manifests" / "samples.csv").resolve(),
    )

    from .discovery import select_sample_row

    sample_row = select_sample_row(corpus, image_name)
    model, model_context = load_model_context(model_run_dir, device=device)
    preprocessed = preprocess_single_sample(
        corpus=corpus,
        sample_row=sample_row,
        model_context=model_context,
    )
    batch = _build_single_sample_batch(preprocessed)

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
        raise RuntimeError("Single-sample inference did not produce a prediction row.")

    prediction_row = metrics.predictions.iloc[0]
    predicted_distance_m = float(prediction_row["prediction_distance_m"])
    actual_distance_m = float(prediction_row["truth_distance_m"])
    predicted_orientation_deg = float(prediction_row["prediction_yaw_deg"])
    actual_orientation_deg = float(prediction_row["truth_yaw_deg"])

    result = InferenceResult(
        selected_model_label=model_context.label,
        selected_corpus_name=corpus.name,
        selected_image_name=str(preprocessed.sample_row["image_filename"]),
        sample_id=str(preprocessed.sample_row["sample_id"]),
        roi_image=preprocessed.roi_image,
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
        run_dir=model_context.run_dir,
        checkpoint_path=model_context.checkpoint_path,
        source_image_path=preprocessed.source_image_path,
        source_run_json_path=preprocessed.source_run_json_path,
        source_samples_csv_path=preprocessed.source_samples_csv_path,
        preprocessing_contract_version=str(
            model_context.preprocessing_contract.get("ContractVersion", "")
        ).strip(),
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
