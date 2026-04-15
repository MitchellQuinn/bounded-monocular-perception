"""Contract-aware batch preparation, loss composition, and metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import Batch


@dataclass
class LossResult:
    """Weighted loss result with per-component breakdown."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]


@dataclass
class MetricResult:
    """Task-aware evaluation metrics and optional prediction export rows."""

    sample_count: int
    accuracy_within_tolerance: float
    accuracy_by_tolerance: dict[float, float]
    mae: float
    rmse: float
    task_metrics: dict[str, Any]
    primary_truth: np.ndarray
    primary_prediction: np.ndarray
    predictions: pd.DataFrame | None


def batch_to_model_inputs(
    batch: Batch,
    task_contract: Mapping[str, Any],
    *,
    device: torch.device,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Convert a loader batch into the model input contract declared by the topology."""
    images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
    input_mode = str(task_contract.get("input_mode", "image_tensor")).strip() or "image_tensor"
    if input_mode == "image_tensor":
        return images
    if input_mode == "dual_stream_image_bbox_features":
        if batch.bbox_features is None:
            raise ValueError(
                "Topology input_mode='dual_stream_image_bbox_features' requires bbox_features in the batch."
            )
        bbox_features = torch.from_numpy(batch.bbox_features).to(device=device, dtype=torch.float32)
        return {
            "silhouette_crop": images,
            "bbox_features": bbox_features,
        }
    raise ValueError(f"Unsupported task_contract input_mode={input_mode!r}")


def batch_targets_to_tensor(batch: Batch, *, device: torch.device) -> torch.Tensor:
    """Convert the batch target array into a float tensor on the target device."""
    return torch.from_numpy(batch.targets).to(device=device, dtype=torch.float32)


def _ordered_heads(task_contract: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    raw_heads = task_contract.get("heads")
    if not isinstance(raw_heads, Mapping) or not raw_heads:
        raise ValueError("task_contract must define a non-empty heads mapping.")
    ordered: list[tuple[str, Mapping[str, Any]]] = []
    for name, spec in raw_heads.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"task_contract head {name!r} must be a mapping; got {type(spec)}")
        ordered.append((str(name), spec))
    return ordered


def _head_target_columns(head_spec: Mapping[str, Any]) -> tuple[str, ...]:
    columns = tuple(str(column).strip() for column in head_spec.get("target_columns", []))
    if not columns:
        raise ValueError(f"Head spec is missing target_columns: {head_spec}")
    return columns


def _ensure_2d_head_tensor(tensor: torch.Tensor, width: int, *, label: str) -> torch.Tensor:
    if tensor.ndim == 1:
        if width != 1:
            raise ValueError(f"{label} produced shape {tuple(tensor.shape)} for width={width}.")
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim != 2:
        raise ValueError(f"{label} must be 1D or 2D per batch; got shape={tuple(tensor.shape)}")
    if int(tensor.shape[1]) != int(width):
        raise ValueError(
            f"{label} width mismatch; expected {width}, got {int(tensor.shape[1])}."
        )
    return tensor


def extract_prediction_heads(
    model_outputs: torch.Tensor | Mapping[str, torch.Tensor],
    task_contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Split model outputs into normalized 2D tensors by named prediction head."""
    output_kind = str(task_contract.get("output_kind", "tensor")).strip() or "tensor"
    ordered_heads = _ordered_heads(task_contract)

    if output_kind == "mapping":
        if not isinstance(model_outputs, Mapping):
            raise TypeError(
                f"task_contract output_kind='mapping' expects a mapping output; got {type(model_outputs)}"
            )
        result: dict[str, torch.Tensor] = {}
        for head_name, head_spec in ordered_heads:
            output_key = str(head_spec.get("output_key", "")).strip()
            if not output_key:
                raise ValueError(f"Head {head_name!r} is missing output_key for mapping output.")
            if output_key not in model_outputs:
                raise KeyError(f"Model output is missing mapping key {output_key!r}.")
            tensor = model_outputs[output_key]
            if not torch.is_tensor(tensor):
                raise TypeError(
                    f"Model output key {output_key!r} must be a tensor; got {type(tensor)}"
                )
            result[head_name] = _ensure_2d_head_tensor(
                tensor,
                len(_head_target_columns(head_spec)),
                label=f"output[{output_key}]",
            )
        return result

    if output_kind != "tensor":
        raise ValueError(f"Unsupported task_contract output_kind={output_kind!r}")
    if not torch.is_tensor(model_outputs):
        raise TypeError(f"task_contract output_kind='tensor' expects a tensor output; got {type(model_outputs)}")

    output_tensor = model_outputs
    if output_tensor.ndim == 1:
        output_tensor = output_tensor.unsqueeze(-1)
    elif output_tensor.ndim != 2:
        raise ValueError(
            f"Tensor output must be rank-1 or rank-2 per batch; got shape={tuple(output_tensor.shape)}"
        )

    cursor = 0
    result: dict[str, torch.Tensor] = {}
    for head_name, head_spec in ordered_heads:
        head_width = len(_head_target_columns(head_spec))
        raw_slice = head_spec.get("tensor_slice")
        if isinstance(raw_slice, (list, tuple)) and len(raw_slice) == 2:
            start = int(raw_slice[0])
            end = int(raw_slice[1])
        else:
            start = cursor
            end = cursor + head_width
        head_tensor = output_tensor[:, start:end]
        if int(head_tensor.shape[1]) != int(head_width):
            raise ValueError(
                f"Head {head_name!r} expected width {head_width} but received slice "
                f"{(start, end)} from output shape {tuple(output_tensor.shape)}."
            )
        result[head_name] = head_tensor
        cursor = end

    if cursor != int(output_tensor.shape[1]):
        raise ValueError(
            f"Tensor output width mismatch; consumed {cursor} columns from output shape {tuple(output_tensor.shape)}."
        )
    return result


def extract_target_heads(
    target_tensor: torch.Tensor,
    task_contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Split batch targets into normalized 2D tensors by named prediction head."""
    target_columns = tuple(str(column).strip() for column in task_contract.get("target_columns", []))
    if not target_columns:
        raise ValueError("task_contract must define target_columns.")

    if target_tensor.ndim == 1:
        target_tensor = target_tensor.unsqueeze(-1)
    elif target_tensor.ndim != 2:
        raise ValueError(
            f"Target tensor must be rank-1 or rank-2 per batch; got shape={tuple(target_tensor.shape)}"
        )
    if int(target_tensor.shape[1]) != len(target_columns):
        raise ValueError(
            f"Target width mismatch; expected {len(target_columns)} columns for {target_columns}, "
            f"got shape={tuple(target_tensor.shape)}."
        )

    column_to_index = {column: idx for idx, column in enumerate(target_columns)}
    result: dict[str, torch.Tensor] = {}
    for head_name, head_spec in _ordered_heads(task_contract):
        indices = [column_to_index[column] for column in _head_target_columns(head_spec)]
        head_tensor = target_tensor[:, indices]
        result[head_name] = head_tensor
    return result


def compute_task_loss(
    prediction_heads: Mapping[str, torch.Tensor],
    target_heads: Mapping[str, torch.Tensor],
    task_contract: Mapping[str, Any],
    *,
    huber_delta: float,
    loss_weights: Mapping[str, float] | None = None,
) -> LossResult:
    """Compute a weighted multitask Huber loss from named prediction heads."""
    weights = {str(key): float(value) for key, value in (loss_weights or {}).items()}
    total: torch.Tensor | None = None
    components: dict[str, torch.Tensor] = {}

    for head_name, head_spec in _ordered_heads(task_contract):
        if head_name not in prediction_heads or head_name not in target_heads:
            raise KeyError(f"Missing prediction/target tensors for head {head_name!r}.")
        loss_role = str(head_spec.get("loss_role", head_name)).strip() or head_name
        component_name = f"{loss_role}_loss"
        component = F.huber_loss(
            prediction_heads[head_name],
            target_heads[head_name],
            delta=float(huber_delta),
            reduction="mean",
        )
        components[component_name] = component
        weighted = component * float(weights.get(loss_role, 1.0))
        total = weighted if total is None else total + weighted

    if total is None:
        raise ValueError("Could not compute task loss because no heads were resolved.")
    components["total_loss"] = total
    return LossResult(total=total, components=components)


def primary_sample_error(
    prediction_heads: Mapping[str, torch.Tensor],
    target_heads: Mapping[str, torch.Tensor],
    task_contract: Mapping[str, Any],
) -> torch.Tensor:
    """Return one scalar error value per sample for the contract's primary metrics."""
    for head_name, head_spec in _ordered_heads(task_contract):
        metrics_role = str(head_spec.get("metrics_role", "")).strip()
        if metrics_role == "distance":
            pred = prediction_heads[head_name].reshape(-1)
            true = target_heads[head_name].reshape(-1)
            return torch.abs(pred - true)
        if metrics_role == "position":
            pred = prediction_heads[head_name]
            true = target_heads[head_name]
            if pred.ndim != 2 or true.ndim != 2 or int(pred.shape[1]) != 3 or int(true.shape[1]) != 3:
                raise ValueError(
                    f"Position metrics require (B, 3) tensors; got pred={tuple(pred.shape)} true={tuple(true.shape)}"
                )
            return torch.linalg.vector_norm(pred - true, ord=2, dim=1)
    raise ValueError("task_contract does not define a supported primary metrics head.")


def _decode_yaw_deg(sin_component: np.ndarray, cos_component: np.ndarray) -> np.ndarray:
    yaw = np.degrees(np.arctan2(sin_component, cos_component))
    yaw = np.mod(yaw, 360.0)
    yaw[yaw < 0.0] += 360.0
    return yaw.astype(np.float32)


def _angular_error_deg(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    wrapped = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return np.abs(wrapped).astype(np.float32)


def _base_prediction_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": row.get("dataset_id"),
        "run_id": row.get("run_id"),
        "sample_id": row.get("sample_id"),
        "frame_index": row.get("frame_index"),
        "npz_filename": row.get("npz_filename"),
        "npz_row_index": row.get("npz_row_index"),
        "source_root": row.get("source_root"),
    }


def summarize_task_metrics(
    prediction_heads: Mapping[str, np.ndarray],
    target_heads: Mapping[str, np.ndarray],
    task_contract: Mapping[str, Any],
    *,
    tolerance_values: tuple[float, ...],
    primary_tolerance: float,
    rows: list[dict[str, Any]] | None = None,
    collect_predictions: bool = False,
) -> MetricResult:
    """Compute contract-aware scalar and auxiliary metrics for one split."""
    if not tolerance_values:
        raise ValueError("At least one tolerance value is required.")
    if not prediction_heads:
        raise ValueError("prediction_heads cannot be empty.")

    ordered_heads = _ordered_heads(task_contract)
    prediction_mode = str(task_contract.get("prediction_mode", "")).strip()
    head_specs = {head_name: head_spec for head_name, head_spec in ordered_heads}

    distance_head = next(
        (head_name for head_name, spec in ordered_heads if str(spec.get("metrics_role", "")).strip() == "distance"),
        None,
    )
    position_head = next(
        (head_name for head_name, spec in ordered_heads if str(spec.get("metrics_role", "")).strip() == "position"),
        None,
    )
    orientation_head = next(
        (head_name for head_name, spec in ordered_heads if str(spec.get("metrics_role", "")).strip() == "orientation"),
        None,
    )

    task_metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    sample_count = 0

    primary_truth = np.array([], dtype=np.float32)
    primary_prediction = np.array([], dtype=np.float32)

    if distance_head is not None:
        true_distance = np.asarray(target_heads[distance_head], dtype=np.float32).reshape(-1)
        pred_distance = np.asarray(prediction_heads[distance_head], dtype=np.float32).reshape(-1)
        sample_count = int(true_distance.shape[0])
        residual = pred_distance - true_distance
        abs_error = np.abs(residual)
        accuracy_by_tolerance = {
            float(tolerance): float(np.mean(abs_error <= float(tolerance)))
            for tolerance in tolerance_values
        }
        mae = float(np.mean(abs_error))
        rmse = float(np.sqrt(np.mean(np.square(residual))))
        primary_truth = true_distance
        primary_prediction = pred_distance
    elif position_head is not None:
        true_position = np.asarray(target_heads[position_head], dtype=np.float32)
        pred_position = np.asarray(prediction_heads[position_head], dtype=np.float32)
        if true_position.ndim != 2 or pred_position.ndim != 2 or true_position.shape[1] != 3 or pred_position.shape[1] != 3:
            raise ValueError(
                f"position head requires (N, 3) arrays; got true={true_position.shape} pred={pred_position.shape}"
            )
        sample_count = int(true_position.shape[0])
        diff = pred_position - true_position
        position_error = np.linalg.norm(diff, axis=1).astype(np.float32)
        accuracy_by_tolerance = {
            float(tolerance): float(np.mean(position_error <= float(tolerance)))
            for tolerance in tolerance_values
        }
        mae = float(np.mean(position_error))
        rmse = float(np.sqrt(np.mean(np.square(position_error))))
        task_metrics["position"] = {
            "mean_position_error_m": float(mae),
            "median_position_error_m": float(np.median(position_error)),
            "p95_position_error_m": float(np.percentile(position_error, 95)),
            "component_mae_m": {
                "x": float(np.mean(np.abs(diff[:, 0]))),
                "y": float(np.mean(np.abs(diff[:, 1]))),
                "z": float(np.mean(np.abs(diff[:, 2]))),
            },
            "component_rmse_m": {
                "x": float(math.sqrt(np.mean(np.square(diff[:, 0])))),
                "y": float(math.sqrt(np.mean(np.square(diff[:, 1])))),
                "z": float(math.sqrt(np.mean(np.square(diff[:, 2])))),
            },
        }
    else:
        raise ValueError(
            f"Unsupported task contract {prediction_mode!r}: no distance or position metrics head found."
        )

    if orientation_head is not None:
        true_orientation = np.asarray(target_heads[orientation_head], dtype=np.float32)
        pred_orientation = np.asarray(prediction_heads[orientation_head], dtype=np.float32)
        if true_orientation.ndim != 2 or pred_orientation.ndim != 2 or true_orientation.shape[1] != 2 or pred_orientation.shape[1] != 2:
            raise ValueError(
                "orientation head requires (N, 2) sin/cos arrays; "
                f"got true={true_orientation.shape} pred={pred_orientation.shape}"
            )
        true_yaw_deg = _decode_yaw_deg(true_orientation[:, 0], true_orientation[:, 1])
        if rows is not None and len(rows) == int(true_orientation.shape[0]) and "yaw_deg" in rows[0]:
            true_yaw_deg = np.asarray([float(row["yaw_deg"]) for row in rows], dtype=np.float32)
        pred_yaw_deg = _decode_yaw_deg(pred_orientation[:, 0], pred_orientation[:, 1])
        angular_error = _angular_error_deg(pred_yaw_deg, true_yaw_deg)
        task_metrics["orientation"] = {
            "mean_angular_error_deg": float(np.mean(angular_error)),
            "median_angular_error_deg": float(np.median(angular_error)),
            "p95_angular_error_deg": float(np.percentile(angular_error, 95)),
        }
    else:
        true_yaw_deg = None
        pred_yaw_deg = None
        angular_error = None

    if collect_predictions:
        if rows is None:
            raise ValueError("rows are required when collect_predictions=True")
        if len(rows) != sample_count:
            raise ValueError(f"Prediction row count mismatch: rows={len(rows)} sample_count={sample_count}")
        for idx, row in enumerate(rows):
            out_row = _base_prediction_row(row)
            if distance_head is not None:
                out_row.update(
                    {
                        "truth_distance_m": float(primary_truth[idx]),
                        "prediction_distance_m": float(primary_prediction[idx]),
                        "residual_m": float(primary_prediction[idx] - primary_truth[idx]),
                    }
                )
            if position_head is not None:
                true_position = np.asarray(target_heads[position_head], dtype=np.float32)
                pred_position = np.asarray(prediction_heads[position_head], dtype=np.float32)
                diff = pred_position - true_position
                out_row.update(
                    {
                        "truth_pos_x_m": float(true_position[idx, 0]),
                        "truth_pos_y_m": float(true_position[idx, 1]),
                        "truth_pos_z_m": float(true_position[idx, 2]),
                        "prediction_pos_x_m": float(pred_position[idx, 0]),
                        "prediction_pos_y_m": float(pred_position[idx, 1]),
                        "prediction_pos_z_m": float(pred_position[idx, 2]),
                        "position_error_m": float(np.linalg.norm(diff[idx])),
                    }
                )
            if orientation_head is not None and true_yaw_deg is not None and pred_yaw_deg is not None and angular_error is not None:
                true_orientation = np.asarray(target_heads[orientation_head], dtype=np.float32)
                pred_orientation = np.asarray(prediction_heads[orientation_head], dtype=np.float32)
                out_row.update(
                    {
                        "truth_yaw_deg": float(true_yaw_deg[idx]),
                        "prediction_yaw_deg": float(pred_yaw_deg[idx]),
                        "angular_error_deg": float(angular_error[idx]),
                        "truth_yaw_sin": float(true_orientation[idx, 0]),
                        "truth_yaw_cos": float(true_orientation[idx, 1]),
                        "prediction_yaw_sin": float(pred_orientation[idx, 0]),
                        "prediction_yaw_cos": float(pred_orientation[idx, 1]),
                    }
                )
            prediction_rows.append(out_row)

    predictions_df = pd.DataFrame(prediction_rows) if collect_predictions else None

    return MetricResult(
        sample_count=sample_count,
        accuracy_within_tolerance=float(accuracy_by_tolerance[float(primary_tolerance)]),
        accuracy_by_tolerance=accuracy_by_tolerance,
        mae=float(mae),
        rmse=float(rmse),
        task_metrics=task_metrics,
        primary_truth=primary_truth,
        primary_prediction=primary_prediction,
        predictions=predictions_df,
    )
