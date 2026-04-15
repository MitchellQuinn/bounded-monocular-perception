"""Training entrypoints for distance regression across multiple topologies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .config import TrainConfig
from .data import (
    ALLOWED_TRAIN_SHUFFLE_MODES,
    ShardArrayCache,
    detect_overlap_warnings,
    determine_target_hw,
    iter_batches,
    load_root_metadata,
    summarize_metadata,
    validate_task_contract_schema,
    validate_root_schema,
)
from .evaluate import evaluate_split
from .paths import (
    DEFAULT_TRAINING_ROOT,
    DEFAULT_VALIDATION_ROOT,
    find_repo_root,
    make_model_run_dir,
    resolve_data_root,
    resolve_output_root,
    to_repo_relative,
)
from .plots import save_history_plot, save_prediction_scatter, save_residual_plot
from .task_runtime import (
    batch_targets_to_tensor,
    batch_to_model_inputs,
    compute_task_loss,
    extract_prediction_heads,
    extract_target_heads,
    primary_sample_error,
)
from .resume.state import (
    RESUME_STATE_FILENAME,
    build_resume_state_payload,
    load_resume_state,
    save_resume_state,
)
from .topologies import (
    ResolvedTopologySpec,
    architecture_text_from_spec,
    build_model_from_spec,
    resolve_topology_spec,
    task_contract_signature,
    topology_spec_signature,
)
from .utils import (
    environment_summary,
    git_metadata,
    read_json,
    set_random_seeds,
    sha256_file,
    utc_now_iso,
    write_json,
)


def _schema_to_records(schema_df: pd.DataFrame) -> list[dict[str, Any]]:
    serializable = schema_df.copy()
    for col in ("x_shape", "y_shape", "sample_id_shape", "npz_row_index_shape", "keys", "missing_required_keys"):
        if col in serializable.columns:
            serializable[col] = serializable[col].map(
                lambda value: list(value) if isinstance(value, (tuple, list, np.ndarray)) else value
            )
    return serializable.to_dict(orient="records")


@dataclass
class TrainEpochSummary:
    """Aggregated training metrics for one epoch."""

    loss: float
    accuracy_within_tolerance: float
    accuracy_by_tolerance: dict[float, float]
    loss_components: dict[str, float]


def _loss_weights_from_config(config: TrainConfig) -> dict[str, float]:
    weights = {
        "distance": float(config.distance_loss_weight),
        "orientation": float(config.orientation_loss_weight),
        "position": float(config.position_loss_weight),
    }
    for name, value in weights.items():
        if value < 0.0:
            raise ValueError(f"{name}_loss_weight must be >= 0; got {value}")
    return weights


def _split_training_metadata(
    training_metadata: pd.DataFrame,
    enable_internal_test_split: bool,
    internal_test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not enable_internal_test_split:
        return training_metadata.copy(), pd.DataFrame(columns=training_metadata.columns)

    if not (0.0 < internal_test_fraction < 1.0):
        raise ValueError(
            f"internal_test_fraction must be in (0, 1); got {internal_test_fraction}"
        )
    rng = np.random.default_rng(seed)
    indices = np.arange(len(training_metadata))
    rng.shuffle(indices)
    test_count = max(1, int(round(len(indices) * internal_test_fraction)))
    test_idx = np.sort(indices[:test_count])
    train_idx = np.sort(indices[test_count:])
    train_split = training_metadata.iloc[train_idx].reset_index(drop=True)
    test_split = training_metadata.iloc[test_idx].reset_index(drop=True)
    return train_split, test_split


def _column_or_na(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")


def _parse_tolerance_values(raw: Any) -> tuple[float, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return ()
        parts = [part.strip() for part in text.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        parts = list(raw)
    else:
        parts = [raw]

    values: list[float] = []
    for part in parts:
        if part is None:
            continue
        value = float(part)
        if value <= 0.0:
            raise ValueError(f"Accuracy tolerance values must be positive; got {value}")
        values.append(value)
    if not values:
        return ()
    unique_sorted = sorted({round(value, 8) for value in values})
    return tuple(float(value) for value in unique_sorted)


def _accuracy_by_tolerance_json(accuracy_by_tolerance: dict[float, float]) -> dict[str, float]:
    return {
        f"{float(tolerance):.2f}m": float(accuracy)
        for tolerance, accuracy in sorted(accuracy_by_tolerance.items())
    }


def _format_accuracy_metrics(
    accuracy_by_tolerance: dict[float, float],
    *,
    prefix: str = "",
) -> str:
    return " ".join(
        f"{prefix}acc@{float(tolerance):.2f}m={float(accuracy):.4f}"
        for tolerance, accuracy in sorted(accuracy_by_tolerance.items())
    )


def _tolerance_suffix(tolerance: float) -> str:
    return f"{float(tolerance):.2f}".replace(".", "p")


def _resolve_topology_from_train_config(config: TrainConfig) -> ResolvedTopologySpec:
    raw_topology_params = config.topology_params
    topology_params = (
        raw_topology_params
        if isinstance(raw_topology_params, dict)
        else None
    )
    if raw_topology_params is not None and topology_params is None:
        raise ValueError(
            f"topology_params must be a JSON object/mapping; got {type(raw_topology_params)}"
        )
    return resolve_topology_spec(
        topology_id=str(config.topology_id).strip() if config.topology_id is not None else None,
        topology_variant=(
            str(config.topology_variant).strip()
            if config.topology_variant is not None
            else None
        ),
        topology_params=topology_params,
        legacy_model_architecture_variant=(
            str(config.model_architecture_variant).strip()
            if config.model_architecture_variant is not None
            else None
        ),
    )


def _build_split_membership(
    train_split: pd.DataFrame,
    validation_split: pd.DataFrame,
    internal_test_split: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_name, split_df in (
        ("train", train_split),
        ("validation", validation_split),
        ("test_internal", internal_test_split),
    ):
        if split_df.empty:
            continue
        frame = pd.DataFrame(
            {
                "dataset_id": _column_or_na(split_df, "dataset_id"),
                "run_id": _column_or_na(split_df, "run_id"),
                "sample_id": _column_or_na(split_df, "sample_id"),
                "frame_index": _column_or_na(split_df, "frame_index"),
                "distance_m": _column_or_na(split_df, "distance_m"),
                "split": split_name,
                "relative_source_samples_csv_path": _column_or_na(
                    split_df, "relative_source_samples_csv_path"
                ),
                "npz_filename": _column_or_na(split_df, "npz_filename"),
                "npz_row_index": _column_or_na(split_df, "npz_row_index"),
                "source_root": _column_or_na(split_df, "source_root"),
            }
        )
        frames.append(frame)
    if not frames:
        raise ValueError("No rows available to build split membership.")
    return pd.concat(frames, ignore_index=True)


def _train_one_epoch(
    model: nn.Module,
    optimizer: Adam,
    task_contract: dict[str, Any],
    huber_delta: float,
    loss_weights: dict[str, float],
    split_df: pd.DataFrame,
    batch_size: int,
    target_hw: tuple[int, int],
    padding_mode: str,
    device: torch.device,
    epoch_seed: int,
    accuracy_tolerance_m: float = 0.10,
    additional_accuracy_tolerances_m: Any = None,
    shard_cache: ShardArrayCache | None = None,
    shuffle_mode: str = "shard",
    active_shard_count: int = 3,
    epoch: int | None = None,
    total_epochs: int | None = None,
    progress_log_interval_batches: int = 250,
) -> TrainEpochSummary:
    model.train()
    tolerance_values = _parse_tolerance_values(
        [accuracy_tolerance_m, *_parse_tolerance_values(additional_accuracy_tolerances_m)]
    )
    if not tolerance_values:
        raise ValueError("No valid accuracy tolerances were configured.")
    primary_tolerance = float(min(tolerance_values, key=lambda value: abs(value - float(accuracy_tolerance_m))))
    total_loss = 0.0
    total_loss_components: dict[str, float] = {}
    total_count = 0
    within_tolerance_counts: dict[float, int] = {float(tolerance): 0 for tolerance in tolerance_values}
    total_batches_est = max(1, int(math.ceil(len(split_df) / float(batch_size))))
    started = perf_counter()
    epoch_label = (
        f"{epoch}/{total_epochs}"
        if epoch is not None and total_epochs is not None
        else str(epoch) if epoch is not None else "?"
    )

    for batch_index, batch in enumerate(
        iter_batches(
            metadata_df=split_df,
            batch_size=batch_size,
            target_hw=target_hw,
            padding_mode=padding_mode,
            shuffle_shards=True,
            seed=epoch_seed,
            shard_cache=shard_cache,
            shuffle_mode=shuffle_mode,
            active_shard_count=active_shard_count,
            target_columns=tuple(task_contract.get("target_columns", [])),
            include_bbox_features=(
                str(task_contract.get("input_mode", "")).strip()
                == "dual_stream_image_bbox_features"
            ),
        ),
        start=1,
    ):
        model_inputs = batch_to_model_inputs(batch, task_contract, device=device)
        targets = batch_targets_to_tensor(batch, device=device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(model_inputs)
        pred_heads = extract_prediction_heads(outputs, task_contract)
        target_heads = extract_target_heads(targets, task_contract)
        loss_result = compute_task_loss(
            pred_heads,
            target_heads,
            task_contract,
            huber_delta=float(huber_delta),
            loss_weights=loss_weights,
        )
        loss_result.total.backward()
        optimizer.step()

        batch_n = int(targets.shape[0])
        total_loss += float(loss_result.total.item()) * batch_n
        for name, tensor_value in loss_result.components.items():
            total_loss_components[name] = total_loss_components.get(name, 0.0) + (
                float(tensor_value.item()) * batch_n
            )
        total_count += batch_n
        batch_primary_error = primary_sample_error(pred_heads, target_heads, task_contract)
        for tolerance in tolerance_values:
            within_tolerance_counts[float(tolerance)] += int(
                torch.count_nonzero(batch_primary_error <= float(tolerance)).item()
            )
        running_loss = total_loss / float(total_count)
        running_accuracy_by_tolerance = {
            float(tolerance): (within_tolerance_counts[float(tolerance)] / float(total_count))
            for tolerance in tolerance_values
        }
        running_component_text = " ".join(
            f"{name}={float(total_loss_components[name]) / float(total_count):.6f}"
            for name in sorted(total_loss_components)
            if name != "total_loss"
        )

        if progress_log_interval_batches > 0:
            should_log = (
                batch_index == 1
                or (batch_index % progress_log_interval_batches == 0)
                or batch_index == total_batches_est
            )
            if should_log:
                elapsed = perf_counter() - started
                pct = 100.0 * (batch_index / float(total_batches_est))
                print(
                    f"[train][epoch {epoch_label}] "
                    f"batch {batch_index}/{total_batches_est} ({pct:5.1f}%) "
                    f"seen={total_count} {_format_accuracy_metrics(running_accuracy_by_tolerance)} "
                    f"running_loss={running_loss:.6f} "
                    f"{running_component_text} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if total_count == 0:
        raise ValueError("Training epoch produced zero samples.")
    accuracy_by_tolerance = {
        float(tolerance): (within_tolerance_counts[float(tolerance)] / float(total_count))
        for tolerance in tolerance_values
    }
    avg_components = {
        name: float(total / float(total_count))
        for name, total in sorted(total_loss_components.items())
    }
    return TrainEpochSummary(
        loss=(total_loss / total_count),
        accuracy_within_tolerance=accuracy_by_tolerance[primary_tolerance],
        accuracy_by_tolerance=accuracy_by_tolerance,
        loss_components=avg_components,
    )


def _resolve_entrypoint(repo_root: Path, entrypoint_path_raw: str) -> tuple[str, str | None]:
    entrypoint_candidate = Path(entrypoint_path_raw)
    if not entrypoint_candidate.is_absolute():
        entrypoint_candidate = (repo_root / entrypoint_candidate).resolve()
    if entrypoint_candidate.exists():
        return to_repo_relative(repo_root, entrypoint_candidate), sha256_file(entrypoint_candidate)
    return entrypoint_path_raw, None


def _load_preprocessing_contract_records(
    repo_root: Path,
    infos: list[Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for info in infos:
        run_payload = read_json(info.run_json_path)
        preprocessing_contract = run_payload.get("PreprocessingContract")
        records.append(
            {
                "source_root": str(info.source_root),
                "dataset_id": str(info.dataset_id),
                "relative_run_json_path": to_repo_relative(repo_root, info.run_json_path),
                "has_preprocessing_contract": isinstance(preprocessing_contract, dict),
                "preprocessing_contract": (
                    preprocessing_contract if isinstance(preprocessing_contract, dict) else None
                ),
            }
        )
    return records


def _resolve_preprocessing_contract(
    contract_records: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if not contract_records:
        return None, ["No source corpuses were available to resolve a preprocessing contract."]

    present = [record for record in contract_records if record["preprocessing_contract"] is not None]
    missing = [record for record in contract_records if record["preprocessing_contract"] is None]

    if missing and present:
        missing_paths = ", ".join(record["relative_run_json_path"] for record in missing)
        raise ValueError(
            "Selected corpuses mix run.json files with and without PreprocessingContract metadata. "
            f"Missing contract in: {missing_paths}"
        )

    if not present:
        return None, [
            "No source run.json files contain PreprocessingContract metadata; "
            "saved model artifacts cannot fully reconstruct preprocessing for inference."
        ]

    def _canonicalize_contract_for_compatibility(contract: dict[str, Any]) -> dict[str, Any]:
        canonical = json.loads(json.dumps(contract))
        stages = canonical.get("Stages")
        if isinstance(stages, dict):
            for stage_payload in stages.values():
                if isinstance(stage_payload, dict):
                    # These fields only control subset selection during preprocessing runs.
                    stage_payload.pop("SampleOffset", None)
                    stage_payload.pop("SampleLimit", None)
        return canonical

    warnings: list[str] = []
    raw_signatures: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in present:
        raw_signatures.add(
            json.dumps(
                record["preprocessing_contract"],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        canonical_contract = _canonicalize_contract_for_compatibility(
            dict(record["preprocessing_contract"])
        )
        signature = json.dumps(
            canonical_contract,
            sort_keys=True,
            separators=(",", ":"),
        )
        grouped.setdefault(signature, []).append(record)

    if len(grouped) > 1:
        mismatched_paths = ", ".join(
            record["relative_run_json_path"]
            for records in grouped.values()
            for record in records
        )
        raise ValueError(
            "Selected corpuses contain conflicting PreprocessingContract values. "
            f"Conflicting run.json files: {mismatched_paths}"
        )

    if len(raw_signatures) > 1:
        warnings.append(
            "PreprocessingContract values differ only by sample-selection fields "
            "(SampleOffset/SampleLimit); treating them as compatible."
        )

    resolved_contract = _canonicalize_contract_for_compatibility(
        dict(present[0]["preprocessing_contract"])
    )
    return dict(resolved_contract), warnings


def _describe_input_representation(preprocessing_contract: dict[str, Any] | None) -> str:
    fallback = "NPZ shard key X; full-frame bbox image, grayscale, normalized [0, 1]"
    if not isinstance(preprocessing_contract, dict):
        return fallback

    representation = preprocessing_contract.get("CurrentRepresentation")
    if not isinstance(representation, dict):
        return fallback

    storage = representation.get("StorageFormat")
    array_key = representation.get("ArrayKey")
    array_keys = representation.get("ArrayKeys")
    kind = representation.get("Kind")
    color = representation.get("ColorSpace")
    geometry = representation.get("Geometry")
    dtype = representation.get("ArrayDType")

    parts: list[str] = []
    if storage and array_key:
        parts.append(f"{storage} key {array_key}")
    elif storage and isinstance(array_keys, list) and array_keys:
        joined_keys = ",".join(str(value) for value in array_keys)
        parts.append(f"{storage} keys {joined_keys}")
    elif storage:
        parts.append(str(storage))
    if kind:
        parts.append(str(kind))
    if color:
        parts.append(str(color))
    if geometry:
        parts.append(str(geometry))
    if dtype:
        parts.append(f"dtype={dtype}")
    if "Normalize" in representation:
        parts.append(f"normalize={bool(representation['Normalize'])}")
    if "Invert" in representation:
        parts.append(f"invert={bool(representation['Invert'])}")

    return "; ".join(parts) if parts else fallback


def _write_model_card(
    run_dir: Path,
    run_id: str,
    config: TrainConfig,
    topology_spec: ResolvedTopologySpec,
    dataset_summary: dict[str, Any],
    split_summary: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    tolerance_values = _parse_tolerance_values(
        [config.accuracy_tolerance_m, *_parse_tolerance_values(config.extra_accuracy_tolerances_m)]
    )
    tolerance_text = ", ".join(f"+/-{float(tolerance):.2f} m" for tolerance in tolerance_values)
    scheduler_text = "disabled"
    if bool(config.enable_lr_scheduler):
        scheduler_text = (
            "ReduceLROnPlateau("
            f"factor={float(config.lr_scheduler_factor)}, "
            f"patience={int(config.lr_scheduler_patience)}, "
            f"min_lr={float(config.lr_scheduler_min_lr)})"
        )

    validation_metrics = metrics.get("validation", {})
    validation_accuracy_by_tolerance = validation_metrics.get("accuracy_by_tolerance")
    target_columns = tuple(
        str(column).strip() for column in topology_spec.task_contract.get("target_columns", [])
    )
    input_mode = str(topology_spec.task_contract.get("input_mode", "")).strip()
    prediction_mode = str(topology_spec.task_contract.get("prediction_mode", "")).strip()
    if input_mode == "dual_stream_image_bbox_features":
        input_text = "grayscale crop tensor plus bbox feature vector"
    else:
        input_text = "grayscale full-frame tensor normalized to [0, 1]"
    if str(topology_spec.task_contract.get("task_family", "")).strip() == "multitask_regression":
        loss_text = (
            "Weighted multitask Huber "
            f"(delta={config.huber_delta}, distance={config.distance_loss_weight}, "
            f"orientation={config.orientation_loss_weight})"
        )
    elif prediction_mode == "position_3d":
        loss_text = (
            f"Huber over position_3d targets (delta={config.huber_delta}, "
            f"weight={config.position_loss_weight})"
        )
    else:
        loss_text = f"Huber (delta={config.huber_delta})"
    validation_accuracy_lines: list[str] = []
    if isinstance(validation_accuracy_by_tolerance, dict) and validation_accuracy_by_tolerance:
        def _acc_sort_key(item: tuple[str, Any]) -> float:
            label = str(item[0]).rstrip("m")
            try:
                return float(label)
            except ValueError:
                return float("inf")

        for tolerance_label, accuracy_value in sorted(
            validation_accuracy_by_tolerance.items(), key=_acc_sort_key
        ):
            validation_accuracy_lines.append(
                f"- Validation accuracy (+/-{str(tolerance_label)}): {float(accuracy_value):.6f}"
            )
    else:
        validation_accuracy_lines.append(
            (
                f"- Validation accuracy (+/-{float(config.accuracy_tolerance_m):.2f} m): "
                f"{float(validation_metrics.get('accuracy_within_tolerance', float('nan'))):.6f}"
            )
        )

    lines = [
        f"# Model Card - {run_id}",
        "",
        "## Purpose",
        "Bounded falsification test over the topology's declared regression targets ",
        "without geometry-altering transforms.",
        "",
        "## Data",
        f"- Training source root: `{dataset_summary['training']['source_root']}`",
        f"- Validation source root: `{dataset_summary['validation']['source_root']}`",
        f"- Training samples: {split_summary['train']['num_samples']}",
        f"- Validation samples: {split_summary['validation']['num_samples']}",
        f"- Internal test enabled: {config.enable_internal_test_split}",
        "",
        "## Model",
        f"- Topology id: {topology_spec.topology_id}",
        f"- Topology variant: {topology_spec.topology_variant}",
        f"- Model class: {topology_spec.model_class_name}",
        f"- Topology params: {json.dumps(topology_spec.topology_params, sort_keys=True)}",
        f"- Prediction mode: `{prediction_mode}`",
        f"- Targets: {', '.join(f'`{column}`' for column in target_columns)}",
        f"- Input: {input_text}",
        "",
        "## Training",
        f"- Loss: {loss_text}",
        "- Optimizer: Adam",
        f"- Learning rate: {config.learning_rate}",
        f"- Weight decay: {config.weight_decay}",
        f"- LR scheduler: {scheduler_text}",
        f"- Early stopping patience: {config.early_stopping_patience}",
        f"- Accuracy metrics: fraction within {tolerance_text}",
        f"- Train shuffle mode: {config.train_shuffle_mode}",
        f"- Active shard count (reservoir mode): {config.train_active_shard_count}",
        f"- Training shard RAM cache budget: {config.train_cache_budget_gb:.1f} GiB",
        (
            f"- Validation RAM cache enabled: {config.cache_validation_in_ram} "
            f"(budget {config.validation_cache_budget_gb:.1f} GiB)"
        ),
        "",
        "## Results (This Run)",
        *validation_accuracy_lines,
        f"- Best validation MAE: {metrics['validation']['mae']:.6f}",
        f"- Best validation RMSE: {metrics['validation']['rmse']:.6f}",
        f"- Best validation loss: {metrics['validation']['loss']:.6f}",
        "",
        "## Artifact Tracking",
        "- `best.pt` and `latest.pt` are in this run directory.",
        "- `history.csv`, `metrics.json`, plots, and split membership files are co-located.",
    ]
    orientation_metrics = validation_metrics.get("orientation")
    if isinstance(orientation_metrics, dict):
        lines.insert(-3, f"- Validation mean angular error: {float(orientation_metrics.get('mean_angular_error_deg', float('nan'))):.6f} deg")
        lines.insert(-3, f"- Validation median angular error: {float(orientation_metrics.get('median_angular_error_deg', float('nan'))):.6f} deg")
        lines.insert(-3, f"- Validation p95 angular error: {float(orientation_metrics.get('p95_angular_error_deg', float('nan'))):.6f} deg")
    path = run_dir / "model_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_index_from_name(name: str) -> int:
    text = str(name).strip()
    if not text.startswith("run_"):
        return -1
    try:
        return int(text.split("_", maxsplit=1)[1])
    except ValueError:
        return -1


def _normalize_history_records(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("history_records in resume state must be a list.")
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(
                f"history_records[{idx}] in resume state must be an object; got {type(row)}"
            )
        records.append(dict(row))
    return records


def _resolve_resume_source_run_dir(repo_root: Path, source_raw: str | Path) -> Path:
    source = Path(source_raw).expanduser()
    if not source.is_absolute():
        source = (repo_root / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"resume_from_run_dir does not exist: {source}")

    if source.is_dir() and (source / "config.json").exists():
        return source

    candidates: list[Path] = []
    for base in (source, source / "runs"):
        if not base.exists() or not base.is_dir():
            continue
        for child in base.glob("run_*"):
            if not child.is_dir():
                continue
            if (child / RESUME_STATE_FILENAME).exists() and (child / "config.json").exists():
                candidates.append(child.resolve())
    if candidates:
        candidates.sort(
            key=lambda path: (
                _run_index_from_name(path.name),
                path.stat().st_mtime,
            )
        )
        return candidates[-1]

    raise FileNotFoundError(
        "Could not resolve a resume source run directory with both config.json and "
        f"{RESUME_STATE_FILENAME}: {source}"
    )


def _load_resume_context(
    *,
    repo_root: Path,
    config: TrainConfig,
    topology_spec: ResolvedTopologySpec,
    training_root: Path,
    validation_root: Path,
    target_hw: tuple[int, int],
) -> dict[str, Any] | None:
    source_raw = config.resume_from_run_dir
    if source_raw is None:
        return None

    source_run_dir = _resolve_resume_source_run_dir(repo_root, source_raw)
    source_config = read_json(source_run_dir / "config.json")
    resume_state_path = source_run_dir / RESUME_STATE_FILENAME
    resume_state = load_resume_state(resume_state_path, map_location="cpu")

    source_topology_params_raw = (
        resume_state.get("topology_params")
        if isinstance(resume_state.get("topology_params"), dict)
        else source_config.get("topology_params")
        if isinstance(source_config.get("topology_params"), dict)
        else None
    )
    source_topology_spec = resolve_topology_spec(
        topology_id=(
            str(
                resume_state.get("topology_id")
                or source_config.get("topology_id")
                or topology_spec.topology_id
            ).strip()
        ),
        topology_variant=(
            str(
                resume_state.get("topology_variant")
                or source_config.get("topology_variant")
                or ""
            ).strip()
            or None
        ),
        topology_params=source_topology_params_raw,
        legacy_model_architecture_variant=(
            str(
                resume_state.get("model_architecture_variant")
                or source_config.get("model_architecture_variant")
                or ""
            ).strip()
            or None
        ),
    )
    source_topology_signature = (
        str(resume_state.get("topology_signature", "")).strip()
        or topology_spec_signature(source_topology_spec)
    )
    current_topology_signature = topology_spec_signature(topology_spec)
    if source_topology_signature != current_topology_signature:
        raise ValueError(
            "Resume source topology mismatch: "
            f"source={source_topology_spec.to_dict()} current={topology_spec.to_dict()}"
        )
    source_task_contract_signature = (
        str(resume_state.get("task_contract_signature", "")).strip()
        or str(source_config.get("task_contract_signature", "")).strip()
    )
    if source_task_contract_signature:
        current_task_contract_signature = task_contract_signature(topology_spec)
        if source_task_contract_signature != current_task_contract_signature:
            raise ValueError(
                "Resume source task contract mismatch: "
                f"source={source_task_contract_signature} current={current_task_contract_signature}"
            )

    state_train_root = str(resume_state.get("training_data_root_resolved", "")).strip()
    state_val_root = str(resume_state.get("validation_data_root_resolved", "")).strip()
    if state_train_root and Path(state_train_root).resolve() != training_root.resolve():
        raise ValueError(
            "Resume source training root mismatch: "
            f"source={state_train_root} current={training_root}"
        )
    if state_val_root and Path(state_val_root).resolve() != validation_root.resolve():
        raise ValueError(
            "Resume source validation root mismatch: "
            f"source={state_val_root} current={validation_root}"
        )

    state_target_hw = resume_state.get("target_hw")
    if isinstance(state_target_hw, (list, tuple)) and len(state_target_hw) == 2:
        source_target_hw = (int(state_target_hw[0]), int(state_target_hw[1]))
        if source_target_hw != (int(target_hw[0]), int(target_hw[1])):
            raise ValueError(
                "Resume source target_hw mismatch: "
                f"source={source_target_hw} current={(int(target_hw[0]), int(target_hw[1]))}"
            )

    additional_epochs = config.additional_epochs
    if additional_epochs is None:
        raise ValueError(
            "additional_epochs must be provided when resume_from_run_dir is set."
        )
    if int(additional_epochs) <= 0:
        raise ValueError(
            f"additional_epochs must be positive for resume runs; got {additional_epochs}"
        )

    return {
        "source_run_dir": source_run_dir,
        "source_config": source_config,
        "source_run_id": str(source_config.get("run_id", source_run_dir.name)),
        "resume_state": resume_state,
        "additional_epochs": int(additional_epochs),
    }


def train_distance_regressor(config: TrainConfig | dict[str, Any]) -> dict[str, Any]:
    """Train distance regressor and write canonical run artifacts."""
    if isinstance(config, dict):
        config = TrainConfig(**{**asdict(TrainConfig()), **config})  # type: ignore[arg-type]
    if config.resume_from_run_dir is None and config.additional_epochs is not None:
        raise ValueError(
            "additional_epochs can only be used when resume_from_run_dir is set."
        )
    topology_spec = _resolve_topology_from_train_config(config)
    task_contract_signature_value = task_contract_signature(topology_spec)
    loss_weights = _loss_weights_from_config(config)
    active_loss_roles = {
        str(spec.get("loss_role", head_name)).strip() or str(head_name)
        for head_name, spec in dict(topology_spec.task_contract.get("heads", {})).items()
        if isinstance(spec, dict)
    }
    if active_loss_roles and sum(float(loss_weights.get(role, 1.0)) for role in active_loss_roles) <= 0.0:
        raise ValueError(
            f"Resolved loss weights disable every active head {sorted(active_loss_roles)}."
        )

    repo_root = find_repo_root()
    training_root = resolve_data_root(
        repo_root, config.training_data_root, default_rel=DEFAULT_TRAINING_ROOT
    )
    validation_root = resolve_data_root(
        repo_root, config.validation_data_root, default_rel=DEFAULT_VALIDATION_ROOT
    )
    models_root = resolve_output_root(repo_root, config.output_root)

    set_random_seeds(config.seed)

    training_metadata, training_infos = load_root_metadata(
        training_root, source_root="training", repo_root=repo_root
    )
    validation_metadata, validation_infos = load_root_metadata(
        validation_root, source_root="validation", repo_root=repo_root
    )
    preprocessing_contract_sources = _load_preprocessing_contract_records(
        repo_root,
        [*training_infos, *validation_infos],
    )
    preprocessing_contract, preprocessing_contract_warnings = _resolve_preprocessing_contract(
        preprocessing_contract_sources
    )

    training_schema = validate_root_schema(training_metadata, root_name="training")
    validation_schema = validate_root_schema(validation_metadata, root_name="validation")
    validate_task_contract_schema(
        training_metadata,
        training_schema,
        topology_spec.task_contract,
        root_name="training",
    )
    validate_task_contract_schema(
        validation_metadata,
        validation_schema,
        topology_spec.task_contract,
        root_name="validation",
    )
    combined_schema = pd.concat([training_schema, validation_schema], ignore_index=True)
    target_hw = determine_target_hw(combined_schema, padding_mode=config.padding_mode)
    resume_context = _load_resume_context(
        repo_root=repo_root,
        config=config,
        topology_spec=topology_spec,
        training_root=training_root,
        validation_root=validation_root,
        target_hw=target_hw,
    )

    overlap_warnings, overlap_details = detect_overlap_warnings(
        training_metadata,
        validation_metadata,
        check_shard_hashes=True,
    )

    train_split, internal_test_split = _split_training_metadata(
        training_metadata,
        enable_internal_test_split=config.enable_internal_test_split,
        internal_test_fraction=config.internal_test_fraction,
        seed=config.seed,
    )
    split_membership = _build_split_membership(
        train_split=train_split,
        validation_split=validation_metadata,
        internal_test_split=internal_test_split,
    )
    train_shard_paths = tuple(
        sorted(Path(path) for path in train_split["npz_path"].astype(str).unique())
    )
    validation_shard_paths = tuple(
        sorted(Path(path) for path in validation_metadata["npz_path"].astype(str).unique())
    )

    run_dir = make_model_run_dir(
        models_root=models_root,
        model_name=config.model_name,
        run_id=config.run_id,
        run_name_suffix=config.run_name_suffix,
    )
    run_id = run_dir.name
    topology_signature = topology_spec_signature(topology_spec)

    config_payload = asdict(config)
    resume_source_run_dir: str | None = None
    resume_source_run_id: str | None = None
    if resume_context is not None:
        resume_source_run_dir = str(resume_context["source_run_dir"])
        resume_source_run_id = str(resume_context["source_run_id"])
    config_payload.update(
        {
            "repo_root": str(repo_root),
            "training_data_root_resolved": str(training_root),
            "validation_data_root_resolved": str(validation_root),
            "output_root_resolved": str(models_root),
            "target_height": int(target_hw[0]),
            "target_width": int(target_hw[1]),
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            "resume_from_run_dir_resolved": resume_source_run_dir,
            "resume_from_run_id": resume_source_run_id,
            "topology_id": topology_spec.topology_id,
            "topology_variant": topology_spec.topology_variant,
            "topology_params": dict(topology_spec.topology_params),
            "topology_signature": topology_signature,
            "task_contract_signature": task_contract_signature_value,
            "model_architecture_variant": topology_spec.topology_variant,
            "model_topology": topology_spec.to_dict(),
            "task_contract": dict(topology_spec.task_contract),
        }
    )
    write_json(run_dir / "config.json", config_payload)

    dataset_summary = {
        "training": summarize_metadata(training_metadata),
        "validation": summarize_metadata(validation_metadata),
        "training_shard_schema": _schema_to_records(training_schema),
        "validation_shard_schema": _schema_to_records(validation_schema),
        "overlap_warnings": overlap_warnings,
        "overlap_details": overlap_details,
        "preprocessing_contract": preprocessing_contract,
        "preprocessing_contract_sources": [
            {
                "source_root": record["source_root"],
                "dataset_id": record["dataset_id"],
                "relative_run_json_path": record["relative_run_json_path"],
                "has_preprocessing_contract": bool(record["has_preprocessing_contract"]),
            }
            for record in preprocessing_contract_sources
        ],
        "preprocessing_contract_warnings": preprocessing_contract_warnings,
    }
    write_json(run_dir / "dataset_summary.json", dataset_summary)

    split_summary: dict[str, Any] = {
        "train": {"num_samples": int(len(train_split)), "source_root": "training"},
        "validation": {"num_samples": int(len(validation_metadata)), "source_root": "validation"},
        "internal_test_enabled": bool(config.enable_internal_test_split),
    }
    if config.enable_internal_test_split:
        split_summary["test_internal"] = {
            "num_samples": int(len(internal_test_split)),
            "source_root": "training",
        }
    write_json(run_dir / "split_summary.json", split_summary)

    split_membership.to_csv(run_dir / "split_membership.csv", index=False)
    split_membership.to_json(run_dir / "split_membership.json", orient="records", indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_spec(topology_spec).to(device)
    architecture_summary = architecture_text_from_spec(model, topology_spec)
    (run_dir / "model_architecture.txt").write_text(architecture_summary + "\n", encoding="utf-8")
    write_json(
        run_dir / "model_architecture.json",
        {
            "model_name": topology_spec.model_class_name,
            "topology_id": topology_spec.topology_id,
            "topology_variant": topology_spec.topology_variant,
            "topology_params": dict(topology_spec.topology_params),
            "topology_signature": topology_signature,
            "task_contract_signature": task_contract_signature_value,
            "model_architecture_variant": topology_spec.topology_variant,
            "architecture_text": architecture_summary,
            "model_topology": topology_spec.to_dict(),
            "task_contract": dict(topology_spec.task_contract),
        },
    )

    optimizer = Adam(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    lr_scheduler: ReduceLROnPlateau | None = None

    train_cache_budget_gb = float(config.train_cache_budget_gb)
    validation_cache_budget_gb = float(config.validation_cache_budget_gb)
    if train_cache_budget_gb < 0.0:
        raise ValueError(f"train_cache_budget_gb must be >= 0; got {train_cache_budget_gb}")
    if validation_cache_budget_gb < 0.0:
        raise ValueError(
            f"validation_cache_budget_gb must be >= 0; got {validation_cache_budget_gb}"
        )

    gib = float(1024**3)
    train_cache_budget_bytes = int(train_cache_budget_gb * gib)
    validation_cache_budget_bytes = int(validation_cache_budget_gb * gib)

    train_shard_cache: ShardArrayCache | None = None
    if train_cache_budget_bytes > 0:
        train_shard_cache = ShardArrayCache(
            max_bytes=train_cache_budget_bytes,
            name="train_shard_cache",
        )

    validation_shard_cache: ShardArrayCache | None = None
    if bool(config.cache_validation_in_ram) and validation_cache_budget_bytes > 0:
        print(
            f"[cache][validation] preloading {len(validation_shard_paths)} shard(s) "
            f"with budget={validation_cache_budget_gb:.1f}GiB",
            flush=True,
        )
        preload_started = perf_counter()
        validation_shard_cache = ShardArrayCache(
            max_bytes=validation_cache_budget_bytes,
            name="validation_shard_cache",
        )
        validation_shard_cache.preload(validation_shard_paths)
        preload_elapsed = perf_counter() - preload_started
        preload_stats = validation_shard_cache.stats()
        print(
            f"[cache][validation] ready cached_shards={preload_stats['cached_shards']} "
            f"used_gib={float(preload_stats['bytes_used']) / gib:.1f} "
            f"hit_rate={preload_stats['hit_rate']:.3f} elapsed={preload_elapsed:.1f}s",
            flush=True,
        )

    history_records: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    no_improvement_epochs = 0
    start_epoch = 1
    total_epochs_target = int(config.epochs)
    progress_log_interval_batches = max(0, int(config.progress_log_interval_batches))
    accuracy_tolerance_m = float(config.accuracy_tolerance_m)
    if accuracy_tolerance_m <= 0:
        raise ValueError(
            f"accuracy_tolerance_m must be positive; got {accuracy_tolerance_m}"
        )
    extra_accuracy_tolerances = _parse_tolerance_values(config.extra_accuracy_tolerances_m)
    accuracy_tolerance_values = _parse_tolerance_values(
        [accuracy_tolerance_m, *extra_accuracy_tolerances]
    )
    if not accuracy_tolerance_values:
        raise ValueError("No valid accuracy tolerances were configured.")

    enable_lr_scheduler = bool(config.enable_lr_scheduler)
    lr_scheduler_factor = float(config.lr_scheduler_factor)
    lr_scheduler_patience = int(config.lr_scheduler_patience)
    lr_scheduler_min_lr = float(config.lr_scheduler_min_lr)
    if lr_scheduler_factor <= 0.0 or lr_scheduler_factor >= 1.0:
        raise ValueError(
            f"lr_scheduler_factor must be in (0, 1); got {lr_scheduler_factor}"
        )
    if lr_scheduler_patience < 0:
        raise ValueError(
            f"lr_scheduler_patience must be >= 0; got {lr_scheduler_patience}"
        )
    if lr_scheduler_min_lr < 0.0:
        raise ValueError(
            f"lr_scheduler_min_lr must be >= 0; got {lr_scheduler_min_lr}"
        )
    if enable_lr_scheduler:
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
            min_lr=lr_scheduler_min_lr,
        )
    resume_mode = False
    resume_source_run_id = ""
    if resume_context is not None:
        resume_state = dict(resume_context["resume_state"])
        source_run_dir = Path(resume_context["source_run_dir"]).resolve()
        resume_source_run_id = str(resume_context["source_run_id"])
        additional_epochs = int(resume_context["additional_epochs"])
        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler_state = resume_state.get("lr_scheduler_state_dict")
        if lr_scheduler is not None and isinstance(scheduler_state, dict):
            lr_scheduler.load_state_dict(scheduler_state)
        history_records = _normalize_history_records(resume_state.get("history_records"))
        best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
        best_epoch = int(resume_state.get("best_epoch", -1))
        no_improvement_epochs = int(resume_state.get("no_improvement_epochs", 0))
        last_completed_epoch = int(resume_state["epoch"])
        start_epoch = last_completed_epoch + 1
        total_epochs_target = last_completed_epoch + additional_epochs
        if total_epochs_target < start_epoch:
            raise ValueError(
                "Resolved total epoch budget is before resume start: "
                f"start_epoch={start_epoch} total_epochs_target={total_epochs_target}"
            )
        source_best_path = source_run_dir / "best.pt"
        if source_best_path.exists():
            shutil.copy2(source_best_path, run_dir / "best.pt")
        resume_mode = True

    train_shuffle_mode = str(config.train_shuffle_mode)
    if train_shuffle_mode not in ALLOWED_TRAIN_SHUFFLE_MODES:
        raise ValueError(
            f"train_shuffle_mode must be one of {sorted(ALLOWED_TRAIN_SHUFFLE_MODES)}; "
            f"got {train_shuffle_mode}"
        )
    train_active_shard_count = int(config.train_active_shard_count)
    if train_active_shard_count <= 0:
        raise ValueError(
            f"train_active_shard_count must be positive; got {train_active_shard_count}"
        )
    accuracy_metrics_label = ",".join(f"{float(tolerance):.2f}m" for tolerance in accuracy_tolerance_values)
    scheduler_label = (
        (
            "ReduceLROnPlateau("
            f"factor={lr_scheduler_factor},patience={lr_scheduler_patience},min_lr={lr_scheduler_min_lr})"
        )
        if enable_lr_scheduler
        else "disabled"
    )

    print(
        f"[train] run_id={run_id} mode={'resume' if resume_mode else 'fresh'} "
        f"device={device} topology_id={topology_spec.topology_id} "
        f"topology_variant={topology_spec.topology_variant} "
        f"train_samples={len(train_split)} val_samples={len(validation_metadata)} "
        f"batch_size={int(config.batch_size)} "
        f"acc_metrics={accuracy_metrics_label} "
        f"lr_scheduler={scheduler_label} "
        f"shuffle_mode={train_shuffle_mode} "
        f"active_shards={train_active_shard_count} "
        f"train_cache_budget={train_cache_budget_gb:.1f}GiB "
        f"val_cache_enabled={bool(config.cache_validation_in_ram)} "
        f"val_cache_budget={validation_cache_budget_gb:.1f}GiB",
        flush=True,
    )
    if resume_mode:
        print(
            f"[train] resuming from run_id={resume_source_run_id} "
            f"start_epoch={start_epoch} total_target_epochs={total_epochs_target}",
            flush=True,
        )

    for epoch in range(start_epoch, total_epochs_target + 1):
        epoch_started = perf_counter()
        print(f"[train][epoch {epoch}/{total_epochs_target}] starting", flush=True)
        epoch_lr_before = float(optimizer.param_groups[0]["lr"])
        train_summary = _train_one_epoch(
            model=model,
            optimizer=optimizer,
            task_contract=dict(topology_spec.task_contract),
            huber_delta=float(config.huber_delta),
            loss_weights=loss_weights,
            split_df=train_split,
            batch_size=int(config.batch_size),
            target_hw=target_hw,
            padding_mode=config.padding_mode,
            device=device,
            epoch_seed=int(config.seed) + epoch,
            accuracy_tolerance_m=accuracy_tolerance_m,
            additional_accuracy_tolerances_m=extra_accuracy_tolerances,
            shard_cache=train_shard_cache,
            shuffle_mode=train_shuffle_mode,
            active_shard_count=train_active_shard_count,
            epoch=epoch,
            total_epochs=total_epochs_target,
            progress_log_interval_batches=progress_log_interval_batches,
        )
        val_eval = evaluate_split(
            model=model,
            split_df=validation_metadata,
            split_name="validation",
            batch_size=int(config.batch_size),
            target_hw=target_hw,
            padding_mode=config.padding_mode,
            device=device,
            task_contract=dict(topology_spec.task_contract),
            huber_delta=float(config.huber_delta),
            loss_weights=loss_weights,
            collect_predictions=False,
            accuracy_tolerance_m=accuracy_tolerance_m,
            additional_accuracy_tolerances_m=extra_accuracy_tolerances,
            progress_log_interval_batches=progress_log_interval_batches,
            progress_log_prefix=f"[validation][epoch {epoch}/{total_epochs_target}]",
            shard_cache=validation_shard_cache,
        )
        if val_eval.loss is None:
            raise RuntimeError("Validation loss was expected but got None.")
        if lr_scheduler is not None:
            lr_scheduler.step(float(val_eval.loss))
        epoch_lr_after = float(optimizer.param_groups[0]["lr"])

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": epoch_lr_before,
            "next_learning_rate": epoch_lr_after,
            "train_loss": float(train_summary.loss),
            "train_accuracy": float(train_summary.accuracy_within_tolerance),
            "val_loss": float(val_eval.loss),
            "val_accuracy": float(val_eval.accuracy_within_tolerance),
            "val_mae": float(val_eval.mae),
            "val_rmse": float(val_eval.rmse),
        }
        for tolerance in accuracy_tolerance_values:
            suffix = _tolerance_suffix(float(tolerance))
            epoch_record[f"train_acc_at_{suffix}m"] = float(
                train_summary.accuracy_by_tolerance.get(float(tolerance), float("nan"))
            )
            epoch_record[f"val_acc_at_{suffix}m"] = float(
                val_eval.accuracy_by_tolerance.get(float(tolerance), float("nan"))
            )
        if str(topology_spec.task_contract.get("task_family", "")).strip() == "multitask_regression":
            for name, value in sorted(train_summary.loss_components.items()):
                if name != "total_loss":
                    epoch_record[f"train_{name}"] = float(value)
            for name, value in sorted(val_eval.loss_components.items()):
                if name != "total_loss":
                    epoch_record[f"val_{name}"] = float(value)
        if val_eval.task_metrics:
            orientation_metrics = val_eval.task_metrics.get("orientation")
            if isinstance(orientation_metrics, dict):
                for key, value in orientation_metrics.items():
                    epoch_record[f"val_{key}"] = float(value)
            position_metrics = val_eval.task_metrics.get("position")
            if isinstance(position_metrics, dict):
                for key, value in position_metrics.items():
                    if isinstance(value, dict):
                        continue
                    epoch_record[f"val_{key}"] = float(value)
        history_records.append(epoch_record)

        if val_eval.loss < best_val_loss:
            best_val_loss = float(val_eval.loss)
            best_epoch = epoch
            no_improvement_epochs = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            no_improvement_epochs += 1

        epoch_elapsed = perf_counter() - epoch_started
        lr_text = f"lr={epoch_lr_before:.2e}"
        if not math.isclose(epoch_lr_before, epoch_lr_after):
            lr_text = f"lr={epoch_lr_before:.2e}->{epoch_lr_after:.2e}"
        print(
            f"[train][epoch {epoch}/{total_epochs_target}] complete "
            f"train_loss={train_summary.loss:.6f} {_format_accuracy_metrics(train_summary.accuracy_by_tolerance, prefix='train_')} "
            f"val_loss={float(val_eval.loss):.6f} {_format_accuracy_metrics(val_eval.accuracy_by_tolerance, prefix='val_')} "
            f"val_mae={val_eval.mae:.6f} val_rmse={val_eval.rmse:.6f} "
            f"{lr_text} "
            f"elapsed={epoch_elapsed:.1f}s",
            flush=True,
        )
        history_df_epoch = pd.DataFrame(history_records)
        history_df_epoch.to_csv(run_dir / "history.csv", index=False)
        torch.save(model.state_dict(), run_dir / "latest.pt")
        save_resume_state(
            run_dir / RESUME_STATE_FILENAME,
            build_resume_state_payload(
                epoch=epoch,
                run_id=run_id,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                lr_scheduler_state_dict=(
                    lr_scheduler.state_dict() if lr_scheduler is not None else None
                ),
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                no_improvement_epochs=no_improvement_epochs,
                history_records=history_records,
                topology_id=topology_spec.topology_id,
                topology_variant=topology_spec.topology_variant,
                topology_params=dict(topology_spec.topology_params),
                topology_signature=topology_signature,
                task_contract_signature=task_contract_signature_value,
                model_architecture_variant=topology_spec.topology_variant,
                training_data_root_resolved=str(training_root),
                validation_data_root_resolved=str(validation_root),
                target_hw=target_hw,
            ),
        )

        if no_improvement_epochs >= int(config.early_stopping_patience):
            print(
                f"[train] early stopping triggered after epoch {epoch} "
                f"(patience={int(config.early_stopping_patience)})",
                flush=True,
            )
            break

    torch.save(model.state_dict(), run_dir / "latest.pt")

    history_df = pd.DataFrame(history_records)
    if history_df.empty:
        raise RuntimeError("Training completed without any history records.")
    history_df.to_csv(run_dir / "history.csv", index=False)
    save_history_plot(history_df, run_dir / "history_plot.png")

    best_path = run_dir / "best.pt"
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)
        best_epoch = int(history_df["epoch"].iloc[-1])
        best_val_loss = float(history_df["val_loss"].iloc[-1])

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()

    final_val = evaluate_split(
        model=model,
        split_df=validation_metadata,
        split_name="validation",
        batch_size=int(config.batch_size),
        target_hw=target_hw,
        padding_mode=config.padding_mode,
        device=device,
        task_contract=dict(topology_spec.task_contract),
        huber_delta=float(config.huber_delta),
        loss_weights=loss_weights,
        collect_predictions=True,
        accuracy_tolerance_m=accuracy_tolerance_m,
        additional_accuracy_tolerances_m=extra_accuracy_tolerances,
        progress_log_interval_batches=progress_log_interval_batches,
        progress_log_prefix="[validation][final]",
        shard_cache=validation_shard_cache,
    )
    if final_val.predictions is None:
        raise RuntimeError("Expected validation prediction rows.")
    final_val.predictions.to_csv(run_dir / "sample_predictions.csv", index=False)
    if final_val.y_true.size > 0 and final_val.y_pred.size > 0:
        save_prediction_scatter(final_val.y_true, final_val.y_pred, run_dir / "prediction_scatter.png")
        save_residual_plot(final_val.y_true, final_val.y_pred, run_dir / "residual_plot.png")

    metrics: dict[str, Any] = {
        "best_epoch": int(best_epoch),
        "best_val_loss_from_training": float(best_val_loss),
        "stopped_after_epoch": int(history_df["epoch"].iloc[-1]),
        "validation": {
            "sample_count": int(final_val.sample_count),
            "loss": float(final_val.loss) if final_val.loss is not None else None,
            "accuracy_within_tolerance": float(final_val.accuracy_within_tolerance),
            "accuracy_by_tolerance": _accuracy_by_tolerance_json(final_val.accuracy_by_tolerance),
            "mae": float(final_val.mae),
            "rmse": float(final_val.rmse),
        },
    }
    if str(topology_spec.task_contract.get("task_family", "")).strip() == "multitask_regression":
        metrics["validation"]["loss_components"] = dict(final_val.loss_components)
    if final_val.task_metrics:
        metrics["validation"].update(dict(final_val.task_metrics))

    if not internal_test_split.empty:
        test_eval = evaluate_split(
            model=model,
            split_df=internal_test_split,
            split_name="test_internal",
            batch_size=int(config.batch_size),
            target_hw=target_hw,
            padding_mode=config.padding_mode,
            device=device,
            task_contract=dict(topology_spec.task_contract),
            huber_delta=float(config.huber_delta),
            loss_weights=loss_weights,
            collect_predictions=False,
            accuracy_tolerance_m=accuracy_tolerance_m,
            additional_accuracy_tolerances_m=extra_accuracy_tolerances,
            progress_log_interval_batches=progress_log_interval_batches,
            progress_log_prefix="[test_internal]",
            shard_cache=train_shard_cache,
        )
        metrics["test_internal"] = {
            "sample_count": int(test_eval.sample_count),
            "loss": float(test_eval.loss) if test_eval.loss is not None else None,
            "accuracy_within_tolerance": float(test_eval.accuracy_within_tolerance),
            "accuracy_by_tolerance": _accuracy_by_tolerance_json(test_eval.accuracy_by_tolerance),
            "mae": float(test_eval.mae),
            "rmse": float(test_eval.rmse),
        }
        if str(topology_spec.task_contract.get("task_family", "")).strip() == "multitask_regression":
            metrics["test_internal"]["loss_components"] = dict(test_eval.loss_components)
        if test_eval.task_metrics:
            metrics["test_internal"].update(dict(test_eval.task_metrics))

    if train_shard_cache is not None:
        train_cache_final = train_shard_cache.stats()
        print(
            f"[cache][train] cached_shards={train_cache_final['cached_shards']} "
            f"used_gib={float(train_cache_final['bytes_used']) / gib:.1f} "
            f"hits={train_cache_final['hits']} misses={train_cache_final['misses']} "
            f"hit_rate={train_cache_final['hit_rate']:.3f}",
            flush=True,
        )
    if validation_shard_cache is not None:
        val_cache_final = validation_shard_cache.stats()
        print(
            f"[cache][validation] cached_shards={val_cache_final['cached_shards']} "
            f"used_gib={float(val_cache_final['bytes_used']) / gib:.1f} "
            f"hits={val_cache_final['hits']} misses={val_cache_final['misses']} "
            f"hit_rate={val_cache_final['hit_rate']:.3f}",
            flush=True,
        )

    write_json(run_dir / "metrics.json", metrics)

    source_run_json_paths = sorted(
        to_repo_relative(repo_root, info.run_json_path)
        for info in [*training_infos, *validation_infos]
    )
    source_samples_csv_paths = sorted(
        to_repo_relative(repo_root, info.samples_csv_path)
        for info in [*training_infos, *validation_infos]
    )
    entrypoint_path_rel, entrypoint_sha = _resolve_entrypoint(repo_root, config.entrypoint_path)
    git_info = git_metadata(repo_root)
    env_summary = environment_summary(str(device))
    train_cache_stats = train_shard_cache.stats() if train_shard_cache is not None else None
    validation_cache_stats = (
        validation_shard_cache.stats() if validation_shard_cache is not None else None
    )
    target_columns = tuple(
        str(column).strip() for column in topology_spec.task_contract.get("target_columns", [])
    )
    prediction_mode = str(topology_spec.task_contract.get("prediction_mode", "")).strip()
    if prediction_mode == "position_3d":
        target_name = "position_3d"
    elif len(target_columns) == 1:
        target_name = target_columns[0]
    else:
        target_name = ",".join(target_columns)
    if str(topology_spec.task_contract.get("task_family", "")).strip() == "multitask_regression":
        loss_function_text = (
            f"WeightedMultiTaskHuber(delta={config.huber_delta},"
            f"distance={loss_weights['distance']},orientation={loss_weights['orientation']})"
        )
    elif prediction_mode == "position_3d":
        loss_function_text = (
            f"WeightedHuber(delta={config.huber_delta},position={loss_weights['position']})"
        )
    else:
        loss_function_text = f"HuberLoss(delta={config.huber_delta})"

    run_manifest = {
        "run_id": run_id,
        "created_utc": utc_now_iso(),
        "git_commit": git_info["git_commit"],
        "git_branch": git_info["git_branch"],
        "dirty_worktree": git_info["dirty_worktree"],
        "entrypoint_type": config.entrypoint_type,
        "entrypoint_path": entrypoint_path_rel,
        "entrypoint_sha256": entrypoint_sha,
        "source_run_json_paths": source_run_json_paths,
        "source_samples_csv_paths": source_samples_csv_paths,
        "dataset_summary": dataset_summary,
        "preprocessing_contract": preprocessing_contract,
        "preprocessing_contract_sources": dataset_summary["preprocessing_contract_sources"],
        "preprocessing_contract_warnings": preprocessing_contract_warnings,
        "model_name": config.model_name,
        "model_class_name": topology_spec.model_class_name,
        "topology_id": topology_spec.topology_id,
        "topology_variant": topology_spec.topology_variant,
        "topology_params": dict(topology_spec.topology_params),
        "topology_signature": topology_signature,
        "task_contract_signature": task_contract_signature_value,
        "model_architecture_variant": topology_spec.topology_variant,
        "model_architecture_summary": architecture_summary,
        "model_topology": topology_spec.to_dict(),
        "task_contract": dict(topology_spec.task_contract),
        "input_representation": _describe_input_representation(preprocessing_contract),
        "input_shape": [1, int(target_hw[0]), int(target_hw[1])],
        "target_name": target_name,
        "target_names": list(target_columns),
        "optimizer": "Adam",
        "loss_function": loss_function_text,
        "resume": {
            "enabled": bool(resume_mode),
            "source_run_id": resume_source_run_id if resume_mode else None,
            "source_run_dir": (
                str(resume_context["source_run_dir"]) if resume_mode and resume_context is not None else None
            ),
            "start_epoch": int(start_epoch),
            "total_target_epochs": int(total_epochs_target),
            "additional_epochs": (
                int(resume_context["additional_epochs"]) if resume_mode and resume_context is not None else None
            ),
        },
        "hyperparameters": {
            "batch_size": int(config.batch_size),
            "epochs": int(config.epochs),
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
            "early_stopping_patience": int(config.early_stopping_patience),
            "model_name": config.model_name,
            "run_id": run_id,
            "padding_mode": config.padding_mode,
            "topology_id": topology_spec.topology_id,
            "topology_variant": topology_spec.topology_variant,
            "topology_params": dict(topology_spec.topology_params),
            "topology_signature": topology_signature,
            "task_contract_signature": task_contract_signature_value,
            "model_architecture_variant": topology_spec.topology_variant,
            "progress_log_interval_batches": progress_log_interval_batches,
            "accuracy_tolerance_m": accuracy_tolerance_m,
            "extra_accuracy_tolerances_m": [float(value) for value in extra_accuracy_tolerances],
            "distance_loss_weight": float(config.distance_loss_weight),
            "orientation_loss_weight": float(config.orientation_loss_weight),
            "position_loss_weight": float(config.position_loss_weight),
            "enable_lr_scheduler": enable_lr_scheduler,
            "lr_scheduler_factor": lr_scheduler_factor,
            "lr_scheduler_patience": lr_scheduler_patience,
            "lr_scheduler_min_lr": lr_scheduler_min_lr,
            "train_cache_budget_gb": float(config.train_cache_budget_gb),
            "train_shuffle_mode": train_shuffle_mode,
            "train_active_shard_count": train_active_shard_count,
            "cache_validation_in_ram": bool(config.cache_validation_in_ram),
            "validation_cache_budget_gb": float(config.validation_cache_budget_gb),
            "enable_internal_test_split": bool(config.enable_internal_test_split),
            "internal_test_fraction": float(config.internal_test_fraction),
            "resume_from_run_dir": (
                str(config.resume_from_run_dir) if config.resume_from_run_dir is not None else None
            ),
            "additional_epochs": (
                int(config.additional_epochs) if config.additional_epochs is not None else None
            ),
            "start_epoch": int(start_epoch),
            "total_target_epochs": int(total_epochs_target),
        },
        "cache_runtime_stats": {
            "train_shard_cache": train_cache_stats,
            "validation_shard_cache": validation_cache_stats,
        },
        "random_seeds": {
            "global_seed": int(config.seed),
            "python_random_seed": int(config.seed),
            "numpy_seed": int(config.seed),
            "torch_seed": int(config.seed),
        },
        "environment": env_summary,
        "artifact_paths": {
            "config_json": to_repo_relative(repo_root, run_dir / "config.json"),
            "dataset_summary_json": to_repo_relative(repo_root, run_dir / "dataset_summary.json"),
            "split_summary_json": to_repo_relative(repo_root, run_dir / "split_summary.json"),
            "split_membership_csv": to_repo_relative(repo_root, run_dir / "split_membership.csv"),
            "split_membership_json": to_repo_relative(repo_root, run_dir / "split_membership.json"),
            "model_architecture_txt": to_repo_relative(repo_root, run_dir / "model_architecture.txt"),
            "model_architecture_json": to_repo_relative(repo_root, run_dir / "model_architecture.json"),
            "model_card_md": to_repo_relative(repo_root, run_dir / "model_card.md"),
            "best_pt": to_repo_relative(repo_root, run_dir / "best.pt"),
            "latest_pt": to_repo_relative(repo_root, run_dir / "latest.pt"),
            "resume_state_pt": to_repo_relative(repo_root, run_dir / RESUME_STATE_FILENAME),
            "history_csv": to_repo_relative(repo_root, run_dir / "history.csv"),
            "metrics_json": to_repo_relative(repo_root, run_dir / "metrics.json"),
            "prediction_scatter_png": to_repo_relative(repo_root, run_dir / "prediction_scatter.png"),
            "residual_plot_png": to_repo_relative(repo_root, run_dir / "residual_plot.png"),
            "sample_predictions_csv": to_repo_relative(repo_root, run_dir / "sample_predictions.csv"),
        },
        "change_note": config.change_note,
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

    _write_model_card(
        run_dir=run_dir,
        run_id=run_id,
        config=config,
        topology_spec=topology_spec,
        dataset_summary=dataset_summary,
        split_summary=split_summary,
        metrics=metrics,
    )

    return {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "best_model_path": str(run_dir / "best.pt"),
        "last_model_path": str(run_dir / "latest.pt"),
        "resume_state_path": str(run_dir / RESUME_STATE_FILENAME),
        "metrics": metrics,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train distance regressor with pluggable topology families."
    )
    parser.add_argument("--training-data-root", default="training-data")
    parser.add_argument("--validation-data-root", default="validation-data")
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--distance-loss-weight", type=float, default=1.0)
    parser.add_argument("--orientation-loss-weight", type=float, default=1.0)
    parser.add_argument("--position-loss-weight", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--model-name", default="2d-cnn")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name-suffix", default=None)
    parser.add_argument("--padding-mode", default="disabled")
    parser.add_argument("--topology-id", default="distance_regressor_2d_cnn")
    parser.add_argument("--topology-variant", default=None)
    parser.add_argument("--topology-params-json", default="{}")
    parser.add_argument(
        "--model-architecture-variant",
        default=None,
        help="Deprecated alias for --topology-variant.",
    )
    parser.add_argument("--progress-log-interval-batches", type=int, default=250)
    parser.add_argument("--accuracy-tolerance-m", type=float, default=0.10)
    parser.add_argument("--extra-accuracy-tolerances-m", default="0.25,0.50")
    parser.add_argument(
        "--enable-lr-scheduler",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=1)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--train-cache-budget-gb", type=float, default=48.0)
    parser.add_argument("--train-shuffle-mode", default="shard")
    parser.add_argument("--train-active-shard-count", type=int, default=3)
    parser.add_argument(
        "--cache-validation-in-ram",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validation-cache-budget-gb", type=float, default=40.0)
    parser.add_argument("--resume-from-run-dir", default=None)
    parser.add_argument("--additional-epochs", type=int, default=None)
    parser.add_argument("--change-note", default="CLI training run.")
    parser.add_argument("--enable-internal-test-split", action="store_true")
    parser.add_argument("--internal-test-fraction", type=float, default=0.1)
    return parser


def _parse_topology_params_json(raw: Any) -> dict[str, Any]:
    text = str(raw).strip() if raw is not None else "{}"
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(
            "topology-params-json must decode to a JSON object, "
            f"got {type(payload)}"
        )
    return payload


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = TrainConfig(
        training_data_root=args.training_data_root,
        validation_data_root=args.validation_data_root,
        output_root=args.output_root,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        huber_delta=args.huber_delta,
        distance_loss_weight=args.distance_loss_weight,
        orientation_loss_weight=args.orientation_loss_weight,
        position_loss_weight=args.position_loss_weight,
        early_stopping_patience=args.early_stopping_patience,
        model_name=args.model_name,
        run_id=args.run_id,
        run_name_suffix=args.run_name_suffix,
        enable_internal_test_split=bool(args.enable_internal_test_split),
        internal_test_fraction=args.internal_test_fraction,
        padding_mode=args.padding_mode,
        topology_id=args.topology_id,
        topology_variant=args.topology_variant,
        topology_params=_parse_topology_params_json(args.topology_params_json),
        model_architecture_variant=args.model_architecture_variant,
        progress_log_interval_batches=args.progress_log_interval_batches,
        accuracy_tolerance_m=args.accuracy_tolerance_m,
        extra_accuracy_tolerances_m=_parse_tolerance_values(args.extra_accuracy_tolerances_m),
        enable_lr_scheduler=bool(args.enable_lr_scheduler),
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        train_cache_budget_gb=args.train_cache_budget_gb,
        train_shuffle_mode=args.train_shuffle_mode,
        train_active_shard_count=args.train_active_shard_count,
        cache_validation_in_ram=bool(args.cache_validation_in_ram),
        validation_cache_budget_gb=args.validation_cache_budget_gb,
        resume_from_run_dir=args.resume_from_run_dir,
        additional_epochs=args.additional_epochs,
        change_note=args.change_note,
        entrypoint_type="cli",
        entrypoint_path="src/train.py",
    )
    result = train_distance_regressor(config)
    print(f"Training complete. Run directory: {result['run_dir']}")


if __name__ == "__main__":
    main()
