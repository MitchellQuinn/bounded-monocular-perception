"""Training entrypoint for the ROI-FCN centre-localiser."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from .config import TrainConfig
from .contracts import (
    BEST_CHECKPOINT_FILENAME,
    DATASET_CONTRACT_FILENAME,
    HISTORY_FILENAME,
    LATEST_CHECKPOINT_FILENAME,
    LOSS_HISTORY_PLOT_FILENAME,
    MODEL_ARCHITECTURE_FILENAME,
    RESUME_STATE_FILENAME,
    RUN_CONFIG_FILENAME,
    SUMMARY_FILENAME,
    TRAINING_CONTRACT_VERSION,
)
from .data import iter_split_batches, load_and_validate_split_dataset, validate_run_compatibility
from .evaluate import evaluate_split, infer_output_hw, resolve_device, write_split_artifacts
from .geometry import decode_heatmap_argmax, derive_roi_bounds, roi_fully_contains_bbox
from .paths import find_training_root, make_model_run_dir, resolve_models_root, to_repo_relative
from .plots import save_history_plot
from .resume_state import build_resume_state_payload, load_resume_state, save_resume_state
from .targets import build_gaussian_heatmaps, compute_heatmap_loss
from .topologies import (
    architecture_text_from_spec,
    build_model_from_spec,
    resolve_topology_spec,
    topology_contract_signature,
    topology_spec_signature,
)
from .utils import environment_summary, git_metadata, read_json, set_random_seeds, utc_now_iso, write_json


HEATMAP_LOSS_NAME = "balanced_mse_heatmap"


@dataclass(frozen=True)
class BatchLocalizationMetrics:
    """Decoded localisation metrics for one training batch."""

    sample_count: int
    center_error_sum_px: float
    mean_center_error_px: float
    roi_full_containment_hits: int
    roi_full_containment_evaluable_count: int
    mean_peak_confidence: float
    mean_heatmap_activation: float

    @property
    def roi_full_containment_accuracy(self) -> float | None:
        if self.roi_full_containment_evaluable_count <= 0:
            return None
        return float(self.roi_full_containment_hits) / float(self.roi_full_containment_evaluable_count)


@dataclass(frozen=True)
class TrainEpochSummary:
    """Training metrics for one completed epoch."""

    loss_mse: float
    mean_center_error_px: float
    roi_full_containment_success_rate: float | None
    roi_full_containment_evaluable_count: int


def _log_message(log_sink: Callable[[str], None] | None, message: str) -> None:
    if log_sink is None:
        print(message)
    else:
        log_sink(message)


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    best_epoch: int,
    best_validation_loss: float,
    best_validation_mean_center_error_px: float,
    epochs_without_improvement: int,
    history_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_mean_center_error_px": float(best_validation_mean_center_error_px),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history_rows": list(history_rows),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def _format_roi_accuracy(metric_name: str, value: float | None, evaluable_count: int) -> str:
    if value is None:
        return f"{metric_name}=na(eval={int(evaluable_count)})"
    return f"{metric_name}={float(value):.4f}(eval={int(evaluable_count)})"


def _is_better_validation_result(
    candidate_mean_center_error_px: float,
    candidate_loss_mse: float,
    *,
    best_mean_center_error_px: float,
    best_loss_mse: float,
) -> bool:
    if not np.isfinite(best_mean_center_error_px):
        return True
    candidate_center_error = float(candidate_mean_center_error_px)
    if candidate_center_error < float(best_mean_center_error_px) - 1e-6:
        return True
    if np.isclose(candidate_center_error, float(best_mean_center_error_px), atol=1e-6):
        return float(candidate_loss_mse) < float(best_loss_mse) - 1e-9
    return False


def _can_reuse_precreated_run_dir(run_dir: Path) -> bool:
    if not run_dir.exists() or not run_dir.is_dir():
        return False
    entries = sorted(run_dir.iterdir())
    if not entries:
        return True
    return len(entries) == 1 and entries[0].is_file() and entries[0].suffix == ".log"


def _normalize_history_rows(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("history_rows in resume state must be a list.")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"history_rows[{index}] in resume state must be an object; got {type(row)}")
        rows.append(dict(row))
    return rows


def _normalize_split_contract_for_resume(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Split contract must be a mapping; got {type(raw)}")
    payload = dict(raw)
    for key in ("split_root", "run_json_path", "samples_csv_path"):
        payload.pop(key, None)
    geometry_raw = payload.get("geometry")
    if not isinstance(geometry_raw, dict):
        raise ValueError("Split contract is missing a geometry mapping.")
    geometry = dict(geometry_raw)
    if isinstance(geometry.get("geometry_schema"), (list, tuple)):
        geometry["geometry_schema"] = list(geometry["geometry_schema"])
    payload["geometry"] = geometry
    array_keys = payload.get("representation_array_keys")
    if isinstance(array_keys, (list, tuple)):
        payload["representation_array_keys"] = list(array_keys)
    return payload


def _current_split_contract_for_resume(split_dataset) -> dict[str, Any]:
    return _normalize_split_contract_for_resume(split_dataset.contract.to_dict())


def _run_index_from_name(name: str) -> int:
    text = str(name).strip()
    if not text.startswith("run_"):
        return -1
    try:
        return int(text.split("_", maxsplit=1)[1])
    except ValueError:
        return -1


def _resolve_resume_source_run_dir(training_root: Path, source_raw: str | Path) -> Path:
    source = Path(source_raw).expanduser()
    repo_root = training_root.parents[1]
    candidates: list[Path] = []
    if source.is_absolute():
        candidates.append(source.resolve())
    else:
        for base in (Path.cwd(), training_root, repo_root):
            candidate = (base / source).resolve()
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / RUN_CONFIG_FILENAME).exists() and (candidate / RESUME_STATE_FILENAME).exists():
            return candidate
    child_candidates: list[Path] = []
    for candidate in candidates:
        for base in (candidate, candidate / "runs"):
            if not base.exists() or not base.is_dir():
                continue
            for child in base.glob("run_*"):
                if not child.is_dir():
                    continue
                if (child / RUN_CONFIG_FILENAME).exists() and (child / RESUME_STATE_FILENAME).exists():
                    child_candidates.append(child.resolve())
    if child_candidates:
        child_candidates.sort(key=lambda item: (_run_index_from_name(item.name), item.stat().st_mtime))
        return child_candidates[-1]
    raise FileNotFoundError(f"Could not resolve a resumable source run directory from {source_raw!r}")


def _assert_resume_config_matches_source(
    current_config: TrainConfig,
    *,
    source_config: dict[str, Any],
    validation_dataset: str,
) -> None:
    def _expect_equal(label: str, current: Any, source: Any) -> None:
        if current != source:
            raise ValueError(f"Resume config mismatch for {label}: source={source!r} current={current!r}")

    def _expect_close(label: str, current: float, source: float) -> None:
        if not np.isclose(float(current), float(source), atol=1e-12, rtol=0.0):
            raise ValueError(f"Resume config mismatch for {label}: source={source!r} current={current!r}")

    _expect_equal(
        "training_dataset",
        str(current_config.training_dataset).strip(),
        str(source_config.get("training_dataset", "")).strip(),
    )
    _expect_equal(
        "validation_dataset",
        validation_dataset,
        str(source_config.get("validation_dataset") or source_config.get("training_dataset") or "").strip(),
    )
    _expect_equal("epochs", int(current_config.epochs), int(source_config.get("epochs", current_config.epochs)))
    _expect_equal("batch_size", int(current_config.batch_size), int(source_config.get("batch_size", current_config.batch_size)))
    _expect_close(
        "learning_rate",
        float(current_config.learning_rate),
        float(source_config.get("learning_rate", current_config.learning_rate)),
    )
    _expect_close(
        "weight_decay",
        float(current_config.weight_decay),
        float(source_config.get("weight_decay", current_config.weight_decay)),
    )
    _expect_close(
        "gaussian_sigma_px",
        float(current_config.gaussian_sigma_px),
        float(source_config.get("gaussian_sigma_px", current_config.gaussian_sigma_px)),
    )
    _expect_close(
        "heatmap_positive_threshold",
        float(current_config.heatmap_positive_threshold),
        float(source_config.get("heatmap_positive_threshold", current_config.heatmap_positive_threshold)),
    )
    _expect_equal(
        "early_stopping_patience",
        int(current_config.early_stopping_patience),
        int(source_config.get("early_stopping_patience", current_config.early_stopping_patience)),
    )
    _expect_equal(
        "progress_log_interval_steps",
        int(current_config.progress_log_interval_steps),
        int(source_config.get("progress_log_interval_steps", current_config.progress_log_interval_steps)),
    )
    _expect_equal(
        "roi_width_px",
        int(current_config.roi_width_px),
        int(source_config.get("roi_width_px", current_config.roi_width_px)),
    )
    _expect_equal(
        "roi_height_px",
        int(current_config.roi_height_px),
        int(source_config.get("roi_height_px", current_config.roi_height_px)),
    )
    _expect_equal(
        "topology_id",
        str(current_config.topology_id).strip(),
        str(source_config.get("topology_id", "")).strip(),
    )
    _expect_equal(
        "topology_variant",
        str(current_config.topology_variant).strip(),
        str(source_config.get("topology_variant", "")).strip(),
    )
    source_topology_params = source_config.get("topology_params")
    if not isinstance(source_topology_params, dict):
        source_topology_params = {}
    if dict(current_config.topology_params) != dict(source_topology_params):
        raise ValueError(
            "Resume config mismatch for topology_params: "
            f"source={source_topology_params!r} current={current_config.topology_params!r}"
        )


def _load_resume_context(
    *,
    training_root: Path,
    config: TrainConfig,
    validation_dataset: str,
    spec,
    output_hw: tuple[int, int],
    train_split,
    validation_split,
) -> dict[str, Any] | None:
    source_raw = config.resume_from_run_dir
    if source_raw is None:
        return None

    source_run_dir = _resolve_resume_source_run_dir(training_root, source_raw)
    source_config = read_json(source_run_dir / RUN_CONFIG_FILENAME)
    _assert_resume_config_matches_source(
        config,
        source_config=source_config,
        validation_dataset=validation_dataset,
    )
    resume_state = load_resume_state(source_run_dir / RESUME_STATE_FILENAME, map_location="cpu")

    state_training_dataset = str(resume_state.get("training_dataset", "")).strip()
    if state_training_dataset != str(config.training_dataset).strip():
        raise ValueError(
            "Resume source training dataset mismatch: "
            f"source={state_training_dataset!r} current={str(config.training_dataset).strip()!r}"
        )
    state_validation_dataset = str(resume_state.get("validation_dataset", "")).strip()
    if state_validation_dataset != validation_dataset:
        raise ValueError(
            "Resume source validation dataset mismatch: "
            f"source={state_validation_dataset!r} current={validation_dataset!r}"
        )

    source_topology_signature = str(resume_state.get("topology_spec_signature", "")).strip()
    current_topology_signature = topology_spec_signature(spec)
    if source_topology_signature != current_topology_signature:
        raise ValueError(
            "Resume source topology signature mismatch: "
            f"source={source_topology_signature} current={current_topology_signature}"
        )
    source_contract_signature = str(resume_state.get("topology_contract_signature", "")).strip()
    current_contract_signature = topology_contract_signature(spec)
    if source_contract_signature != current_contract_signature:
        raise ValueError(
            "Resume source topology contract mismatch: "
            f"source={source_contract_signature} current={current_contract_signature}"
        )

    state_output_hw = resume_state.get("output_hw")
    if not isinstance(state_output_hw, (list, tuple)) or len(state_output_hw) != 2:
        raise ValueError("Resume state output_hw must be a two-element list or tuple.")
    source_output_hw = (int(state_output_hw[0]), int(state_output_hw[1]))
    if source_output_hw != (int(output_hw[0]), int(output_hw[1])):
        raise ValueError(
            "Resume source output_hw mismatch: "
            f"source={source_output_hw} current={(int(output_hw[0]), int(output_hw[1]))}"
        )

    source_train_contract = _normalize_split_contract_for_resume(resume_state.get("train_split_contract"))
    current_train_contract = _current_split_contract_for_resume(train_split)
    if source_train_contract != current_train_contract:
        raise ValueError(
            "Resume source training split contract mismatch: "
            f"source={source_train_contract} current={current_train_contract}"
        )
    source_validation_contract = _normalize_split_contract_for_resume(resume_state.get("validation_split_contract"))
    current_validation_contract = _current_split_contract_for_resume(validation_split)
    if source_validation_contract != current_validation_contract:
        raise ValueError(
            "Resume source validation split contract mismatch: "
            f"source={source_validation_contract} current={current_validation_contract}"
        )

    additional_epochs = config.additional_epochs
    if additional_epochs is None:
        raise ValueError("additional_epochs must be provided when resume_from_run_dir is set.")
    if int(additional_epochs) <= 0:
        raise ValueError(f"additional_epochs must be positive for resume runs; got {additional_epochs}")

    return {
        "source_run_dir": source_run_dir,
        "source_run_id": str(resume_state.get("run_id") or source_config.get("run_id") or source_run_dir.name),
        "source_config": source_config,
        "resume_state": resume_state,
        "additional_epochs": int(additional_epochs),
    }


def _summarize_training_batch_localisation(
    batch,
    predicted_heatmaps: torch.Tensor,
    *,
    canvas_hw: tuple[int, int],
    roi_width_px: int,
    roi_height_px: int,
) -> BatchLocalizationMetrics:
    predicted_np = predicted_heatmaps.detach().squeeze(1).cpu().numpy()
    batch_n = int(predicted_np.shape[0])
    flattened = predicted_np.reshape(batch_n, -1) if batch_n > 0 else np.zeros((0, 0), dtype=np.float32)
    mean_peak_confidence = float(flattened.max(axis=1).mean()) if batch_n > 0 else float("nan")
    mean_heatmap_activation = float(predicted_np.mean()) if batch_n > 0 else float("nan")
    center_error_sum = 0.0
    containment_hits = 0
    containment_evaluable = 0

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
        predicted_original_xy = np.asarray(
            [predicted_point.original_x, predicted_point.original_y],
            dtype=np.float32,
        )
        center_error_sum += float(np.linalg.norm(predicted_original_xy - target_original_xy))

        if batch.bootstrap_bbox_xyxy_px is not None:
            bbox_xyxy = np.asarray(batch.bootstrap_bbox_xyxy_px[index], dtype=np.float32)
            if np.isfinite(bbox_xyxy).all():
                predicted_roi_xyxy = derive_roi_bounds(
                    predicted_original_xy,
                    roi_width_px=float(roi_width_px),
                    roi_height_px=float(roi_height_px),
                )
                containment_hits += int(roi_fully_contains_bbox(predicted_roi_xyxy, bbox_xyxy))
                containment_evaluable += 1

    mean_center_error_px = center_error_sum / float(batch_n) if batch_n > 0 else float("nan")
    return BatchLocalizationMetrics(
        sample_count=batch_n,
        center_error_sum_px=float(center_error_sum),
        mean_center_error_px=float(mean_center_error_px),
        roi_full_containment_hits=int(containment_hits),
        roi_full_containment_evaluable_count=int(containment_evaluable),
        mean_peak_confidence=float(mean_peak_confidence),
        mean_heatmap_activation=float(mean_heatmap_activation),
    )


def _train_one_epoch(
    model: nn.Module,
    train_split,
    *,
    optimizer: Adam,
    device: torch.device,
    batch_size: int,
    output_hw: tuple[int, int],
    gaussian_sigma_px: float,
    heatmap_positive_threshold: float,
    epoch_index: int,
    progress_log_interval_steps: int,
    roi_width_px: int,
    roi_height_px: int,
    log_sink: Callable[[str], None] | None,
) -> TrainEpochSummary:
    canvas_hw = (
        train_split.contract.geometry.canvas_height_px,
        train_split.contract.geometry.canvas_width_px,
    )
    model.train()
    total_loss = 0.0
    total_count = 0
    total_center_error_sum = 0.0
    total_containment_hits = 0
    total_containment_evaluable = 0
    step_count = 0

    for batch in iter_split_batches(
        train_split,
        batch_size=int(batch_size),
        shuffle=True,
        seed=epoch_index,
    ):
        images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
        target_center_canvas = torch.from_numpy(batch.target_center_canvas_px).to(device=device, dtype=torch.float32)
        target_heatmaps = build_gaussian_heatmaps(
            target_center_canvas,
            canvas_hw=canvas_hw,
            output_hw=output_hw,
            sigma_px=float(gaussian_sigma_px),
        )
        optimizer.zero_grad(set_to_none=True)
        predicted_heatmaps = model(images)
        loss = compute_heatmap_loss(
            predicted_heatmaps,
            target_heatmaps,
            loss_name=HEATMAP_LOSS_NAME,
            positive_threshold=float(heatmap_positive_threshold),
        )
        loss.backward()
        optimizer.step()

        batch_n = int(images.shape[0])
        batch_metrics = _summarize_training_batch_localisation(
            batch,
            predicted_heatmaps,
            canvas_hw=canvas_hw,
            roi_width_px=int(roi_width_px),
            roi_height_px=int(roi_height_px),
        )
        total_loss += float(loss.item()) * batch_n
        total_count += batch_n
        total_center_error_sum += float(batch_metrics.center_error_sum_px)
        total_containment_hits += int(batch_metrics.roi_full_containment_hits)
        total_containment_evaluable += int(batch_metrics.roi_full_containment_evaluable_count)
        step_count += 1

        if progress_log_interval_steps > 0 and (
            step_count == 1 or step_count % int(progress_log_interval_steps) == 0
        ):
            running_loss = total_loss / float(total_count)
            running_mean_center_error_px = total_center_error_sum / float(total_count)
            running_roi_accuracy = (
                float(total_containment_hits) / float(total_containment_evaluable)
                if total_containment_evaluable > 0
                else None
            )
            _log_message(
                log_sink,
                "[train] "
                f"epoch={epoch_index} step={step_count} seen={total_count} loss={running_loss:.6f} "
                f"bch_mn_con_err_px={batch_metrics.mean_center_error_px:.3f} "
                f"bch_mn_pk_con={batch_metrics.mean_peak_confidence:.4f} "
                f"bch_mn_heatmap_act={batch_metrics.mean_heatmap_activation:.6f} "
                f"rn_mn_centre_err={running_mean_center_error_px:.3f} "
                f"{_format_roi_accuracy('bch_roi_acc', batch_metrics.roi_full_containment_accuracy, batch_metrics.roi_full_containment_evaluable_count)} "
                f"{_format_roi_accuracy('running_roi_acc', running_roi_accuracy, total_containment_evaluable)}",
            )

    if total_count <= 0:
        raise ValueError("Training split yielded zero samples.")
    return TrainEpochSummary(
        loss_mse=float(total_loss / float(total_count)),
        mean_center_error_px=float(total_center_error_sum / float(total_count)),
        roi_full_containment_success_rate=(
            float(total_containment_hits) / float(total_containment_evaluable)
            if total_containment_evaluable > 0
            else None
        ),
        roi_full_containment_evaluable_count=int(total_containment_evaluable),
    )


def train_roi_fcn(
    config: TrainConfig | dict[str, Any],
    *,
    log_sink: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train one ROI-FCN run end-to-end."""
    train_config = config if isinstance(config, TrainConfig) else TrainConfig.from_mapping(config)
    if train_config.resume_from_run_dir is None and train_config.additional_epochs is not None:
        raise ValueError("additional_epochs can only be used when resume_from_run_dir is set.")
    if not str(train_config.training_dataset).strip():
        raise ValueError("training_dataset cannot be blank.")
    validation_dataset = str(train_config.validation_dataset or train_config.training_dataset).strip()
    if not validation_dataset:
        raise ValueError("validation_dataset cannot be blank.")

    training_root = find_training_root()
    models_root = resolve_models_root(training_root, train_config.models_root)
    model_name = str(train_config.model_name).strip() or "roi-fcn-tiny"
    run_dir = make_model_run_dir(
        models_root,
        model_name=model_name,
        model_directory=train_config.model_directory,
        run_id=train_config.run_id,
        run_name_suffix=train_config.run_name_suffix,
    )
    model_directory = run_dir.parent.parent.name
    run_id = run_dir.name

    train_split = load_and_validate_split_dataset(
        training_root,
        str(train_config.training_dataset).strip(),
        "train",
        datasets_root_override=train_config.datasets_root,
    )
    validation_split = load_and_validate_split_dataset(
        training_root,
        validation_dataset,
        "validate",
        datasets_root_override=train_config.datasets_root,
    )
    validate_run_compatibility(train_split, validation_split)

    set_random_seeds(int(train_config.seed))
    device = resolve_device(train_config.device, require_cuda=True)
    spec = resolve_topology_spec(
        topology_id=str(train_config.topology_id).strip(),
        topology_variant=str(train_config.topology_variant).strip(),
        topology_params=train_config.topology_params,
    )
    topology_spec_signature_value = topology_spec_signature(spec)
    topology_contract_signature_value = topology_contract_signature(spec)
    model = build_model_from_spec(spec).to(device)
    output_hw = infer_output_hw(
        model,
        canvas_hw=(
            train_split.contract.geometry.canvas_height_px,
            train_split.contract.geometry.canvas_width_px,
        ),
        device=device,
    )
    optimizer = Adam(
        model.parameters(),
        lr=float(train_config.learning_rate),
        weight_decay=float(train_config.weight_decay),
    )

    resume_context = _load_resume_context(
        training_root=training_root,
        config=train_config,
        validation_dataset=validation_dataset,
        spec=spec,
        output_hw=output_hw,
        train_split=train_split,
        validation_split=validation_split,
    )

    history_rows: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_validation_mean_center_error_px = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    total_epochs_target = int(train_config.epochs)
    resume_mode = False
    resume_source_run_id: str | None = None
    resume_source_run_dir: str | None = None
    if resume_context is not None:
        resume_state = dict(resume_context["resume_state"])
        source_run_dir = Path(resume_context["source_run_dir"]).resolve()
        resume_source_run_dir = str(source_run_dir)
        resume_source_run_id = str(resume_context["source_run_id"])
        additional_epochs = int(resume_context["additional_epochs"])
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        history_rows = _normalize_history_rows(resume_state.get("history_rows"))
        best_validation_loss = float(resume_state.get("best_validation_loss", float("inf")))
        best_validation_mean_center_error_px = float(
            resume_state.get("best_validation_mean_center_error_px", float("inf"))
        )
        best_epoch = int(resume_state.get("best_epoch", 0))
        epochs_without_improvement = int(resume_state.get("epochs_without_improvement", 0))
        last_completed_epoch = int(resume_state["epoch"])
        start_epoch = last_completed_epoch + 1
        total_epochs_target = last_completed_epoch + additional_epochs
        if total_epochs_target < start_epoch:
            raise ValueError(
                "Resolved total epoch budget is before resume start: "
                f"start_epoch={start_epoch} total_epochs_target={total_epochs_target}"
            )
        checkpoint_to_copy = None
        for candidate_name in (BEST_CHECKPOINT_FILENAME, LATEST_CHECKPOINT_FILENAME):
            candidate = source_run_dir / candidate_name
            if candidate.exists():
                checkpoint_to_copy = candidate
                break
        if checkpoint_to_copy is None:
            raise FileNotFoundError(f"Resume source is missing both best.pt and latest.pt: {source_run_dir}")
        shutil.copy2(checkpoint_to_copy, run_dir / BEST_CHECKPOINT_FILENAME)
        resume_mode = True

    (run_dir / MODEL_ARCHITECTURE_FILENAME).write_text(
        architecture_text_from_spec(spec) + "\n",
        encoding="utf-8",
    )
    write_json(
        run_dir / DATASET_CONTRACT_FILENAME,
        {
            "training_contract_version": TRAINING_CONTRACT_VERSION,
            "train_split": train_split.contract.to_dict(),
            "validation_split": validation_split.contract.to_dict(),
        },
    )

    repo_root = training_root.parents[1]
    run_config_payload = train_config.to_dict()
    run_config_payload.update(
        {
            "training_contract_version": TRAINING_CONTRACT_VERSION,
            "training_dataset": str(train_config.training_dataset).strip(),
            "validation_dataset": validation_dataset,
            "device": str(device),
            "topology_id": spec.topology_id,
            "topology_variant": spec.topology_variant,
            "topology_params": dict(spec.topology_params),
            "topology_contract": dict(spec.topology_contract),
            "topology_contract_signature": topology_contract_signature_value,
            "topology_spec_signature": topology_spec_signature_value,
            "heatmap_loss_name": HEATMAP_LOSS_NAME,
            "model_directory": model_directory,
            "run_id": run_id,
            "train_run_dir": str(run_dir),
            "train_run_dir_relative": to_repo_relative(repo_root, run_dir),
            "output_hw": {"height": int(output_hw[0]), "width": int(output_hw[1])},
            "started_at_utc": utc_now_iso(),
            "environment": environment_summary(str(device)),
            "git": git_metadata(repo_root),
            "resume_from_run_dir_resolved": resume_source_run_dir,
            "resume_from_run_id": resume_source_run_id,
            "resume": {
                "enabled": bool(resume_mode),
                "source_run_id": resume_source_run_id if resume_mode else None,
                "source_run_dir": resume_source_run_dir if resume_mode else None,
                "start_epoch": int(start_epoch),
                "total_target_epochs": int(total_epochs_target),
                "additional_epochs": (
                    int(resume_context["additional_epochs"]) if resume_mode and resume_context is not None else None
                ),
            },
        }
    )
    write_json(run_dir / RUN_CONFIG_FILENAME, run_config_payload)

    mode_label = "resume" if resume_mode else "fresh"
    _log_message(log_sink, f"[train] model_directory={model_directory} run_id={run_id} run_dir={run_dir} mode={mode_label}")
    _log_message(
        log_sink,
        f"[train] train_dataset={train_split.contract.dataset_reference} val_dataset={validation_split.contract.dataset_reference} device={device}",
    )
    _log_message(
        log_sink,
        f"[train] canvas={train_split.contract.geometry.canvas_width_px}x{train_split.contract.geometry.canvas_height_px} output={output_hw[1]}x{output_hw[0]}",
    )
    _log_message(
        log_sink,
        f"[train] heatmap_loss={HEATMAP_LOSS_NAME} positive_threshold={float(train_config.heatmap_positive_threshold):.3f}",
    )
    if resume_mode and resume_source_run_id is not None:
        _log_message(
            log_sink,
            f"[train] resuming from run_id={resume_source_run_id} start_epoch={start_epoch} total_target_epochs={total_epochs_target}",
        )

    for epoch in range(start_epoch, total_epochs_target + 1):
        train_summary = _train_one_epoch(
            model,
            train_split,
            optimizer=optimizer,
            device=device,
            batch_size=int(train_config.batch_size),
            output_hw=output_hw,
            gaussian_sigma_px=float(train_config.gaussian_sigma_px),
            heatmap_positive_threshold=float(train_config.heatmap_positive_threshold),
            epoch_index=epoch,
            progress_log_interval_steps=int(train_config.progress_log_interval_steps),
            roi_width_px=int(train_config.roi_width_px),
            roi_height_px=int(train_config.roi_height_px),
            log_sink=log_sink,
        )
        validation_eval = evaluate_split(
            model,
            validation_split,
            batch_size=int(train_config.batch_size),
            device=device,
            output_hw=output_hw,
            gaussian_sigma_px=float(train_config.gaussian_sigma_px),
            heatmap_loss_name=HEATMAP_LOSS_NAME,
            heatmap_positive_threshold=float(train_config.heatmap_positive_threshold),
            roi_width_px=int(train_config.roi_width_px),
            roi_height_px=int(train_config.roi_height_px),
            max_visual_examples=0,
        )
        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_summary.loss_mse),
                "train_mean_center_error_px": float(train_summary.mean_center_error_px),
                "train_roi_full_containment_success_rate": train_summary.roi_full_containment_success_rate,
                "validation_loss": float(validation_eval.loss_mse),
                "validation_mean_center_error_px": float(validation_eval.mean_center_error_px),
                "validation_p95_center_error_px": float(validation_eval.p95_center_error_px),
                "validation_roi_full_containment_success_rate": validation_eval.roi_full_containment_success_rate,
            }
        )

        improved = _is_better_validation_result(
            float(validation_eval.mean_center_error_px),
            float(validation_eval.loss_mse),
            best_mean_center_error_px=float(best_validation_mean_center_error_px),
            best_loss_mse=float(best_validation_loss),
        )
        if improved:
            best_validation_loss = float(validation_eval.loss_mse)
            best_validation_mean_center_error_px = float(validation_eval.mean_center_error_px)
            best_epoch = int(epoch)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_df = pd.DataFrame(history_rows)
        history_df.to_json(run_dir / HISTORY_FILENAME, orient="records", indent=2)
        save_history_plot(history_df, run_dir / LOSS_HISTORY_PLOT_FILENAME)
        checkpoint_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_epoch=best_epoch,
            best_validation_loss=best_validation_loss,
            best_validation_mean_center_error_px=best_validation_mean_center_error_px,
            epochs_without_improvement=epochs_without_improvement,
            history_rows=history_rows,
        )
        torch.save(checkpoint_payload, run_dir / LATEST_CHECKPOINT_FILENAME)
        if improved:
            torch.save(checkpoint_payload, run_dir / BEST_CHECKPOINT_FILENAME)
        save_resume_state(
            run_dir / RESUME_STATE_FILENAME,
            build_resume_state_payload(
                epoch=epoch,
                run_id=run_id,
                training_dataset=str(train_config.training_dataset).strip(),
                validation_dataset=validation_dataset,
                topology_id=spec.topology_id,
                topology_variant=spec.topology_variant,
                topology_params=dict(spec.topology_params),
                topology_spec_signature=topology_spec_signature_value,
                topology_contract_signature=topology_contract_signature_value,
                output_hw=output_hw,
                train_split_contract=_current_split_contract_for_resume(train_split),
                validation_split_contract=_current_split_contract_for_resume(validation_split),
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                best_validation_mean_center_error_px=best_validation_mean_center_error_px,
                epochs_without_improvement=epochs_without_improvement,
                history_rows=history_rows,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            ),
        )

        _log_message(
            log_sink,
            "[train] "
            f"epoch={epoch} train_loss={train_summary.loss_mse:.6f} "
            f"train_mean_center_error_px={train_summary.mean_center_error_px:.3f} "
            f"{_format_roi_accuracy('train_roi_acc', train_summary.roi_full_containment_success_rate, train_summary.roi_full_containment_evaluable_count)} "
            f"val_loss={validation_eval.loss_mse:.6f} "
            f"val_mean_center_error_px={validation_eval.mean_center_error_px:.3f} "
            f"best_val_center_error_px={best_validation_mean_center_error_px:.3f} "
            f"{_format_roi_accuracy('val_roi_acc', validation_eval.roi_full_containment_success_rate, validation_eval.roi_full_containment_evaluable_count)}",
        )
        if epochs_without_improvement >= int(train_config.early_stopping_patience):
            _log_message(log_sink, f"[train] early stopping at epoch {epoch}")
            break

    best_checkpoint = torch.load(run_dir / BEST_CHECKPOINT_FILENAME, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    train_eval = evaluate_split(
        model,
        train_split,
        batch_size=int(train_config.batch_size),
        device=device,
        output_hw=output_hw,
        gaussian_sigma_px=float(train_config.gaussian_sigma_px),
        heatmap_loss_name=HEATMAP_LOSS_NAME,
        heatmap_positive_threshold=float(train_config.heatmap_positive_threshold),
        roi_width_px=int(train_config.roi_width_px),
        roi_height_px=int(train_config.roi_height_px),
        max_visual_examples=0,
    )
    validation_eval = evaluate_split(
        model,
        validation_split,
        batch_size=int(train_config.batch_size),
        device=device,
        output_hw=output_hw,
        gaussian_sigma_px=float(train_config.gaussian_sigma_px),
        heatmap_loss_name=HEATMAP_LOSS_NAME,
        heatmap_positive_threshold=float(train_config.heatmap_positive_threshold),
        roi_width_px=int(train_config.roi_width_px),
        roi_height_px=int(train_config.roi_height_px),
        max_visual_examples=int(train_config.evaluation_max_visual_examples),
    )
    write_split_artifacts(run_dir, train_eval)
    write_split_artifacts(run_dir, validation_eval)

    summary = {
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "completed_at_utc": utc_now_iso(),
        "model_directory": model_directory,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_mean_center_error_px": float(best_validation_mean_center_error_px),
        "epochs_completed": int(len(history_rows)),
        "resume_state_path": str(run_dir / RESUME_STATE_FILENAME),
        "resume": {
            "enabled": bool(resume_mode),
            "source_run_id": resume_source_run_id if resume_mode else None,
            "source_run_dir": resume_source_run_dir if resume_mode else None,
            "start_epoch": int(start_epoch),
            "total_target_epochs": int(total_epochs_target),
            "additional_epochs": (
                int(resume_context["additional_epochs"]) if resume_mode and resume_context is not None else None
            ),
        },
        "train_metrics": train_eval.metrics_dict(),
        "validation_metrics": validation_eval.metrics_dict(),
    }
    write_json(run_dir / SUMMARY_FILENAME, summary)
    return summary

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the ROI-FCN centre localiser.")
    parser.add_argument("--training-dataset", required=True)
    parser.add_argument("--validation-dataset")
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument("--models-root", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gaussian-sigma-px", type=float, default=2.5)
    parser.add_argument("--heatmap-positive-threshold", type=float, default=0.05)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--topology-id", default="roi_fcn_tiny")
    parser.add_argument("--topology-variant", default="tiny_v1")
    parser.add_argument("--model-name", default="roi-fcn-tiny")
    parser.add_argument("--model-directory")
    parser.add_argument("--run-id")
    parser.add_argument("--run-name-suffix")
    parser.add_argument("--device")
    parser.add_argument("--progress-log-interval-steps", type=int, default=50)
    parser.add_argument("--roi-width-px", type=int, default=300)
    parser.add_argument("--roi-height-px", type=int, default=300)
    parser.add_argument("--evaluation-max-visual-examples", type=int, default=12)
    parser.add_argument("--resume-from-run-dir", default=None)
    parser.add_argument("--additional-epochs", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    payload = vars(args)
    summary = train_roi_fcn(payload)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
