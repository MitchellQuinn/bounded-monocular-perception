"""Evaluation entrypoints for saved ROI-FCN runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import EvalConfig
from .contracts import (
    CENTER_ERROR_PLOT_FILENAME,
    CENTER_EXAMPLES_FILENAME,
    HEATMAP_EXAMPLES_FILENAME,
    TRAIN_METRICS_FILENAME,
    TRAIN_PREDICTIONS_FILENAME,
    VALIDATION_METRICS_FILENAME,
    VALIDATION_PREDICTIONS_FILENAME,
)
from .data import (
    LoadedSplitDataset,
    RoiFcnDatasetValidationError,
    iter_split_batches,
    load_and_validate_split_dataset,
    validate_run_compatibility,
)
from .geometry import (
    canvas_point_to_output_space,
    decode_heatmap_argmax,
    derive_roi_bounds,
    original_point_to_canvas_space,
    roi_fully_contains_bbox,
)
from .paths import find_training_root
from .plots import (
    save_center_error_histogram,
    save_center_examples,
    save_heatmap_examples,
    save_prediction_scatter,
)
from .topologies import build_model_from_spec, resolve_topology_spec
from .utils import read_json, utc_now_iso, write_json
from .targets import build_gaussian_heatmaps, compute_heatmap_loss


@dataclass
class SplitEvaluation:
    """Evaluation outputs for one split."""

    split_name: str
    sample_count: int
    loss_mse: float
    mean_center_error_px: float
    median_center_error_px: float
    p95_center_error_px: float
    roi_full_containment_success_rate: float | None
    roi_full_containment_evaluable_count: int
    roi_full_containment_missing_bbox_count: int
    predictions_df: pd.DataFrame
    example_records: list[dict[str, Any]]

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "sample_count": self.sample_count,
            "loss_mse": self.loss_mse,
            "mean_center_error_px": self.mean_center_error_px,
            "median_center_error_px": self.median_center_error_px,
            "p95_center_error_px": self.p95_center_error_px,
            "roi_full_containment_success_rate": self.roi_full_containment_success_rate,
            "roi_full_containment_evaluable_count": self.roi_full_containment_evaluable_count,
            "roi_full_containment_missing_bbox_count": self.roi_full_containment_missing_bbox_count,
        }


def _cuda_required_message(requested_device: str | None = None) -> str:
    """Return the canonical CUDA-required error message for training."""
    requested_text = str(requested_device).strip() if requested_device is not None else ""
    prefix = (
        f"Requested device {requested_text!r} cannot be used for ROI-FCN training."
        if requested_text
        else "ROI-FCN training requires CUDA."
    )
    return (
        f"{prefix} CPU fallback is disabled. Activate the CUDA-enabled .venv and ensure "
        "torch.cuda.is_available() is true before launching training."
    )



def resolve_device(raw_device: str | None, *, require_cuda: bool = False) -> torch.device:
    """Resolve a runtime device, optionally enforcing CUDA availability."""
    requested = str(raw_device).strip() if raw_device is not None else ""
    if not requested:
        if require_cuda:
            if not torch.cuda.is_available():
                raise ValueError(_cuda_required_message())
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested)
    if require_cuda and device.type != "cuda":
        raise ValueError(_cuda_required_message(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(_cuda_required_message(requested))
    return device


def infer_output_hw(model: nn.Module, *, canvas_hw: tuple[int, int], device: torch.device) -> tuple[int, int]:
    """Infer the model output geometry from one dummy forward pass."""
    with torch.no_grad():
        dummy = torch.zeros((1, 1, int(canvas_hw[0]), int(canvas_hw[1])), dtype=torch.float32, device=device)
        output = model(dummy)
    if output.ndim != 4 or output.shape[1] != 1:
        raise ValueError(f"Model output must have shape (N,1,H,W); got {tuple(output.shape)}")
    return int(output.shape[2]), int(output.shape[3])


def _roi_xyxy_original_to_canvas(
    roi_xyxy: np.ndarray,
    *,
    resize_scale: float,
    pad_left_px: float,
    pad_top_px: float,
) -> np.ndarray:
    top_left = original_point_to_canvas_space(
        roi_xyxy[:2],
        resize_scale=resize_scale,
        pad_left_px=pad_left_px,
        pad_top_px=pad_top_px,
    )
    bottom_right = original_point_to_canvas_space(
        roi_xyxy[2:],
        resize_scale=resize_scale,
        pad_left_px=pad_left_px,
        pad_top_px=pad_top_px,
    )
    return np.asarray([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], dtype=np.float32)


def evaluate_split(
    model: nn.Module,
    split_dataset: LoadedSplitDataset,
    *,
    batch_size: int,
    device: torch.device,
    output_hw: tuple[int, int],
    gaussian_sigma_px: float,
    heatmap_loss_name: str,
    heatmap_positive_threshold: float,
    roi_width_px: int,
    roi_height_px: int,
    max_visual_examples: int,
) -> SplitEvaluation:
    """Evaluate a model on one validated split."""
    canvas_hw = (
        split_dataset.contract.geometry.canvas_height_px,
        split_dataset.contract.geometry.canvas_width_px,
    )
    model.eval()

    total_loss = 0.0
    total_count = 0
    prediction_rows: list[dict[str, Any]] = []
    example_records: list[dict[str, Any]] = []
    containment_hits = 0
    containment_evaluable = 0

    with torch.no_grad():
        for batch in iter_split_batches(
            split_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            seed=0,
        ):
            images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
            target_center_canvas = torch.from_numpy(batch.target_center_canvas_px).to(device=device, dtype=torch.float32)
            target_heatmaps = build_gaussian_heatmaps(
                target_center_canvas,
                canvas_hw=canvas_hw,
                output_hw=output_hw,
                sigma_px=float(gaussian_sigma_px),
            )
            predicted_heatmaps = model(images)
            batch_loss = compute_heatmap_loss(
                predicted_heatmaps,
                target_heatmaps,
                loss_name=heatmap_loss_name,
                positive_threshold=float(heatmap_positive_threshold),
            )
            batch_n = int(images.shape[0])
            total_loss += float(batch_loss.item()) * batch_n
            total_count += batch_n

            predicted_np = predicted_heatmaps.squeeze(1).cpu().numpy()
            target_np = target_heatmaps.squeeze(1).cpu().numpy()
            for index in range(batch_n):
                resize_scale = float(batch.resize_scale[index])
                pad_left_px = float(batch.padding_ltrb_px[index, 0])
                pad_top_px = float(batch.padding_ltrb_px[index, 1])
                source_wh = np.asarray(batch.source_image_wh_px[index], dtype=np.int32)
                predicted_point = decode_heatmap_argmax(
                    predicted_np[index],
                    canvas_hw=canvas_hw,
                    resize_scale=resize_scale,
                    pad_left_px=pad_left_px,
                    pad_top_px=pad_top_px,
                    source_wh_px=source_wh,
                )
                target_original_xy = np.asarray(batch.target_center_original_px[index], dtype=np.float32)
                target_canvas_xy = np.asarray(batch.target_center_canvas_px[index], dtype=np.float32)
                predicted_original_xy = np.asarray(
                    [predicted_point.original_x, predicted_point.original_y],
                    dtype=np.float32,
                )
                center_error = float(np.linalg.norm(predicted_original_xy - target_original_xy))
                predicted_roi_xyxy = derive_roi_bounds(
                    predicted_original_xy,
                    roi_width_px=float(roi_width_px),
                    roi_height_px=float(roi_height_px),
                )
                target_roi_xyxy = derive_roi_bounds(
                    target_original_xy,
                    roi_width_px=float(roi_width_px),
                    roi_height_px=float(roi_height_px),
                )

                containment_success: bool | None = None
                bbox_xyxy: np.ndarray | None = None
                if batch.bootstrap_bbox_xyxy_px is not None:
                    bbox_xyxy = np.asarray(batch.bootstrap_bbox_xyxy_px[index], dtype=np.float32)
                    if np.isfinite(bbox_xyxy).all():
                        containment_success = roi_fully_contains_bbox(predicted_roi_xyxy, bbox_xyxy)
                        containment_evaluable += 1
                        containment_hits += int(bool(containment_success))

                prediction_rows.append(
                    {
                        "sample_id": str(batch.sample_id[index]),
                        "image_filename": str(batch.image_filename[index]),
                        "npz_filename": str(batch.npz_filename[index]),
                        "npz_row_index": int(batch.npz_row_index[index]),
                        "target_center_x_px": float(target_original_xy[0]),
                        "target_center_y_px": float(target_original_xy[1]),
                        "predicted_center_x_px": float(predicted_original_xy[0]),
                        "predicted_center_y_px": float(predicted_original_xy[1]),
                        "target_canvas_x_px": float(target_canvas_xy[0]),
                        "target_canvas_y_px": float(target_canvas_xy[1]),
                        "predicted_canvas_x_px": float(predicted_point.canvas_x),
                        "predicted_canvas_y_px": float(predicted_point.canvas_y),
                        "center_error_px": center_error,
                        "confidence": float(predicted_point.confidence),
                        "source_image_width_px": int(source_wh[0]),
                        "source_image_height_px": int(source_wh[1]),
                        "resize_scale": resize_scale,
                        "pad_left_px": pad_left_px,
                        "pad_top_px": pad_top_px,
                        "pad_right_px": float(batch.padding_ltrb_px[index, 2]),
                        "pad_bottom_px": float(batch.padding_ltrb_px[index, 3]),
                        "predicted_roi_x1_px": float(predicted_roi_xyxy[0]),
                        "predicted_roi_y1_px": float(predicted_roi_xyxy[1]),
                        "predicted_roi_x2_px": float(predicted_roi_xyxy[2]),
                        "predicted_roi_y2_px": float(predicted_roi_xyxy[3]),
                        "containment_success": containment_success,
                        "bootstrap_bbox_x1": float(bbox_xyxy[0]) if bbox_xyxy is not None else np.nan,
                        "bootstrap_bbox_y1": float(bbox_xyxy[1]) if bbox_xyxy is not None else np.nan,
                        "bootstrap_bbox_x2": float(bbox_xyxy[2]) if bbox_xyxy is not None else np.nan,
                        "bootstrap_bbox_y2": float(bbox_xyxy[3]) if bbox_xyxy is not None else np.nan,
                    }
                )

                if len(example_records) < int(max_visual_examples):
                    target_output_xy = canvas_point_to_output_space(
                        target_canvas_xy,
                        canvas_hw=canvas_hw,
                        output_hw=output_hw,
                    )
                    example_records.append(
                        {
                            "sample_id": str(batch.sample_id[index]),
                            "input_image": np.asarray(batch.images[index, 0], dtype=np.float32),
                            "target_heatmap": np.asarray(target_np[index], dtype=np.float32),
                            "predicted_heatmap": np.asarray(predicted_np[index], dtype=np.float32),
                            "target_output_xy": np.asarray(target_output_xy, dtype=np.float32),
                            "predicted_output_xy": np.asarray([predicted_point.output_x, predicted_point.output_y], dtype=np.float32),
                            "target_canvas_xy": target_canvas_xy,
                            "predicted_canvas_xy": np.asarray([predicted_point.canvas_x, predicted_point.canvas_y], dtype=np.float32),
                            "target_original_xy": target_original_xy,
                            "predicted_original_xy": predicted_original_xy,
                            "target_roi_canvas_xyxy": _roi_xyxy_original_to_canvas(
                                target_roi_xyxy,
                                resize_scale=resize_scale,
                                pad_left_px=pad_left_px,
                                pad_top_px=pad_top_px,
                            ),
                            "predicted_roi_canvas_xyxy": _roi_xyxy_original_to_canvas(
                                predicted_roi_xyxy,
                                resize_scale=resize_scale,
                                pad_left_px=pad_left_px,
                                pad_top_px=pad_top_px,
                            ),
                            "confidence": float(predicted_point.confidence),
                        }
                    )

    if total_count <= 0:
        raise RoiFcnDatasetValidationError(f"No evaluation samples were available for split {split_dataset.contract.split_name}.")

    predictions_df = pd.DataFrame(prediction_rows)
    errors = predictions_df["center_error_px"].to_numpy(dtype=np.float32)
    success_rate = (
        float(containment_hits) / float(containment_evaluable)
        if containment_evaluable > 0
        else None
    )
    return SplitEvaluation(
        split_name=split_dataset.contract.split_name,
        sample_count=int(total_count),
        loss_mse=float(total_loss / float(total_count)),
        mean_center_error_px=float(np.mean(errors)),
        median_center_error_px=float(np.median(errors)),
        p95_center_error_px=float(np.percentile(errors, 95.0)),
        roi_full_containment_success_rate=success_rate,
        roi_full_containment_evaluable_count=int(containment_evaluable),
        roi_full_containment_missing_bbox_count=int(total_count - containment_evaluable),
        predictions_df=predictions_df,
        example_records=example_records,
    )


def write_split_artifacts(run_dir: Path, evaluation: SplitEvaluation) -> None:
    """Write metrics, exports, and visual artifacts for one split evaluation."""
    if evaluation.split_name == "train":
        metrics_path = run_dir / TRAIN_METRICS_FILENAME
        predictions_path = run_dir / TRAIN_PREDICTIONS_FILENAME
        scatter_path = None
        error_hist_path = None
        heatmap_examples_path = None
        center_examples_path = None
    elif evaluation.split_name == "validate":
        metrics_path = run_dir / VALIDATION_METRICS_FILENAME
        predictions_path = run_dir / VALIDATION_PREDICTIONS_FILENAME
        scatter_path = run_dir / "validation_prediction_scatter.png"
        error_hist_path = run_dir / CENTER_ERROR_PLOT_FILENAME
        heatmap_examples_path = run_dir / HEATMAP_EXAMPLES_FILENAME
        center_examples_path = run_dir / CENTER_EXAMPLES_FILENAME
    else:
        raise ValueError(f"Unsupported split_name {evaluation.split_name!r}")

    write_json(metrics_path, evaluation.metrics_dict())
    evaluation.predictions_df.to_csv(predictions_path, index=False)

    if scatter_path is not None:
        save_prediction_scatter(evaluation.predictions_df, scatter_path)
    if error_hist_path is not None:
        save_center_error_histogram(evaluation.predictions_df, error_hist_path)
    if heatmap_examples_path is not None and evaluation.example_records:
        save_heatmap_examples(evaluation.example_records, heatmap_examples_path)
    if center_examples_path is not None and evaluation.example_records:
        save_center_examples(evaluation.example_records, center_examples_path)


def evaluate_saved_run(config: EvalConfig | dict[str, Any]) -> dict[str, Any]:
    """Evaluate a saved run directory using the stored run contract."""
    eval_config = config if isinstance(config, EvalConfig) else EvalConfig.from_mapping(config)
    run_dir = Path(eval_config.model_run_directory).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    run_config = read_json(run_dir / "run_config.json")
    training_root = find_training_root(run_dir)
    training_dataset = str(eval_config.training_dataset or run_config.get("training_dataset", "")).strip()
    validation_dataset = str(eval_config.validation_dataset or run_config.get("validation_dataset", "")).strip()
    if not training_dataset or not validation_dataset:
        raise ValueError("Saved run is missing training/validation dataset references.")

    train_split = load_and_validate_split_dataset(
        training_root,
        training_dataset,
        "train",
        datasets_root_override=eval_config.datasets_root,
    )
    validation_split = load_and_validate_split_dataset(
        training_root,
        validation_dataset,
        "validate",
        datasets_root_override=eval_config.datasets_root,
    )
    validate_run_compatibility(train_split, validation_split)

    spec = resolve_topology_spec(
        topology_id=str(run_config.get("topology_id", "")).strip(),
        topology_variant=str(run_config.get("topology_variant", "")).strip(),
        topology_params=run_config.get("topology_params") if isinstance(run_config.get("topology_params"), dict) else {},
    )
    device = resolve_device(eval_config.device or str(run_config.get("device", "")).strip() or None)
    model = build_model_from_spec(spec).to(device)
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = run_dir / "latest.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    state = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        raise ValueError(f"Checkpoint is missing model_state_dict: {checkpoint_path}")
    model.load_state_dict(state["model_state_dict"])

    canvas_hw = (
        train_split.contract.geometry.canvas_height_px,
        train_split.contract.geometry.canvas_width_px,
    )
    output_hw = infer_output_hw(model, canvas_hw=canvas_hw, device=device)
    gaussian_sigma_px = float(run_config.get("gaussian_sigma_px", 2.5))
    heatmap_loss_name = str(run_config.get("heatmap_loss_name", "mse_heatmap")).strip() or "mse_heatmap"
    heatmap_positive_threshold = float(run_config.get("heatmap_positive_threshold", 0.05))
    roi_width_px = int(eval_config.roi_width_px or run_config.get("roi_width_px", 300))
    roi_height_px = int(eval_config.roi_height_px or run_config.get("roi_height_px", 300))
    max_examples = int(eval_config.evaluation_max_visual_examples)

    train_evaluation = evaluate_split(
        model,
        train_split,
        batch_size=int(eval_config.batch_size),
        device=device,
        output_hw=output_hw,
        gaussian_sigma_px=gaussian_sigma_px,
        heatmap_loss_name=heatmap_loss_name,
        heatmap_positive_threshold=heatmap_positive_threshold,
        roi_width_px=roi_width_px,
        roi_height_px=roi_height_px,
        max_visual_examples=max_examples,
    )
    validation_evaluation = evaluate_split(
        model,
        validation_split,
        batch_size=int(eval_config.batch_size),
        device=device,
        output_hw=output_hw,
        gaussian_sigma_px=gaussian_sigma_px,
        heatmap_loss_name=heatmap_loss_name,
        heatmap_positive_threshold=heatmap_positive_threshold,
        roi_width_px=roi_width_px,
        roi_height_px=roi_height_px,
        max_visual_examples=max_examples,
    )

    write_split_artifacts(run_dir, train_evaluation)
    write_split_artifacts(run_dir, validation_evaluation)

    summary = {
        "evaluation_contract_version": "rb-roi-fcn-eval-v0_1",
        "evaluated_at_utc": utc_now_iso(),
        "checkpoint_path": str(checkpoint_path),
        "train_metrics": train_evaluation.metrics_dict(),
        "validation_metrics": validation_evaluation.metrics_dict(),
    }
    write_json(run_dir / "evaluation_summary.json", summary)
    return summary
