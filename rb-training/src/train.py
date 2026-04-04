"""Training entrypoints for the first-pass 2D CNN distance regressor."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
from pathlib import Path
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
    validate_root_schema,
)
from .evaluate import evaluate_split
from .model_2d_cnn import DistanceRegressor2DCNN, architecture_text
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
from .utils import environment_summary, git_metadata, set_random_seeds, sha256_file, utc_now_iso, write_json


def _schema_to_records(schema_df: pd.DataFrame) -> list[dict[str, Any]]:
    serializable = schema_df.copy()
    for col in ("x_shape", "y_shape", "sample_id_shape", "npz_row_index_shape", "keys", "missing_required_keys"):
        if col in serializable.columns:
            serializable[col] = serializable[col].map(
                lambda value: list(value) if isinstance(value, (tuple, list, np.ndarray)) else value
            )
    return serializable.to_dict(orient="records")


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
    loss_fn: nn.Module,
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
) -> tuple[float, float, dict[float, float]]:
    model.train()
    tolerance_values = _parse_tolerance_values(
        [accuracy_tolerance_m, *_parse_tolerance_values(additional_accuracy_tolerances_m)]
    )
    if not tolerance_values:
        raise ValueError("No valid accuracy tolerances were configured.")
    primary_tolerance = float(min(tolerance_values, key=lambda value: abs(value - float(accuracy_tolerance_m))))
    total_loss = 0.0
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
        ),
        start=1,
    ):
        images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
        targets = torch.from_numpy(batch.targets).to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images).reshape(-1)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_n = int(targets.shape[0])
        total_loss += float(loss.item()) * batch_n
        total_count += batch_n
        abs_error = torch.abs(outputs - targets)
        for tolerance in tolerance_values:
            within_tolerance_counts[float(tolerance)] += int(
                torch.count_nonzero(abs_error <= float(tolerance)).item()
            )
        running_loss = total_loss / float(total_count)
        running_accuracy_by_tolerance = {
            float(tolerance): (within_tolerance_counts[float(tolerance)] / float(total_count))
            for tolerance in tolerance_values
        }

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
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if total_count == 0:
        raise ValueError("Training epoch produced zero samples.")
    accuracy_by_tolerance = {
        float(tolerance): (within_tolerance_counts[float(tolerance)] / float(total_count))
        for tolerance in tolerance_values
    }
    return total_loss / total_count, accuracy_by_tolerance[primary_tolerance], accuracy_by_tolerance


def _resolve_entrypoint(repo_root: Path, entrypoint_path_raw: str) -> tuple[str, str | None]:
    entrypoint_candidate = Path(entrypoint_path_raw)
    if not entrypoint_candidate.is_absolute():
        entrypoint_candidate = (repo_root / entrypoint_candidate).resolve()
    if entrypoint_candidate.exists():
        return to_repo_relative(repo_root, entrypoint_candidate), sha256_file(entrypoint_candidate)
    return entrypoint_path_raw, None


def _write_model_card(
    run_dir: Path,
    run_id: str,
    config: TrainConfig,
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
        "First-pass bounded falsification test: can a 2D CNN regress `distance_m` ",
        "from full-frame sparse bounding-box imagery without geometry-altering transforms.",
        "",
        "## Data",
        f"- Training source root: `{dataset_summary['training']['source_root']}`",
        f"- Validation source root: `{dataset_summary['validation']['source_root']}`",
        f"- Training samples: {split_summary['train']['num_samples']}",
        f"- Validation samples: {split_summary['validation']['num_samples']}",
        f"- Internal test enabled: {config.enable_internal_test_split}",
        "",
        "## Model",
        f"- Architecture variant: {config.model_architecture_variant}",
        "- Architecture: 2D CNN distance regressor",
        "- Target: `distance_m`",
        "- Input: grayscale full-frame tensor normalized to [0, 1]",
        "",
        "## Training",
        f"- Loss: Huber (delta={config.huber_delta})",
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
    path = run_dir / "model_card.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_distance_regressor(config: TrainConfig | dict[str, Any]) -> dict[str, Any]:
    """Train the first-pass 2D CNN regressor and write canonical run artifacts."""
    if isinstance(config, dict):
        config = TrainConfig(**{**asdict(TrainConfig()), **config})  # type: ignore[arg-type]

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

    training_schema = validate_root_schema(training_metadata, root_name="training")
    validation_schema = validate_root_schema(validation_metadata, root_name="validation")
    combined_schema = pd.concat([training_schema, validation_schema], ignore_index=True)
    target_hw = determine_target_hw(combined_schema, padding_mode=config.padding_mode)

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

    config_payload = asdict(config)
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
    model = DistanceRegressor2DCNN(
        input_channels=1,
        architecture_variant=config.model_architecture_variant,
    ).to(device)
    architecture_summary = architecture_text(model)
    (run_dir / "model_architecture.txt").write_text(architecture_summary + "\n", encoding="utf-8")
    write_json(
        run_dir / "model_architecture.json",
        {
            "model_name": "DistanceRegressor2DCNN",
            "model_architecture_variant": config.model_architecture_variant,
            "architecture_text": architecture_summary,
        },
    )

    loss_fn = nn.HuberLoss(delta=float(config.huber_delta))
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
    total_epochs = int(config.epochs)
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
        f"[train] run_id={run_id} device={device} model_variant={config.model_architecture_variant} "
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

    for epoch in range(1, total_epochs + 1):
        epoch_started = perf_counter()
        print(f"[train][epoch {epoch}/{total_epochs}] starting", flush=True)
        epoch_lr_before = float(optimizer.param_groups[0]["lr"])
        train_loss, train_accuracy, train_accuracy_by_tolerance = _train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
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
            total_epochs=total_epochs,
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
            loss_fn=loss_fn,
            collect_predictions=False,
            accuracy_tolerance_m=accuracy_tolerance_m,
            additional_accuracy_tolerances_m=extra_accuracy_tolerances,
            progress_log_interval_batches=progress_log_interval_batches,
            progress_log_prefix=f"[validation][epoch {epoch}/{total_epochs}]",
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
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_eval.loss),
            "val_accuracy": float(val_eval.accuracy_within_tolerance),
            "val_mae": float(val_eval.mae),
            "val_rmse": float(val_eval.rmse),
        }
        for tolerance in accuracy_tolerance_values:
            suffix = _tolerance_suffix(float(tolerance))
            epoch_record[f"train_acc_at_{suffix}m"] = float(
                train_accuracy_by_tolerance.get(float(tolerance), float("nan"))
            )
            epoch_record[f"val_acc_at_{suffix}m"] = float(
                val_eval.accuracy_by_tolerance.get(float(tolerance), float("nan"))
            )
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
            f"[train][epoch {epoch}/{total_epochs}] complete "
            f"train_loss={train_loss:.6f} {_format_accuracy_metrics(train_accuracy_by_tolerance, prefix='train_')} "
            f"val_loss={float(val_eval.loss):.6f} {_format_accuracy_metrics(val_eval.accuracy_by_tolerance, prefix='val_')} "
            f"val_mae={val_eval.mae:.6f} val_rmse={val_eval.rmse:.6f} "
            f"{lr_text} "
            f"elapsed={epoch_elapsed:.1f}s",
            flush=True,
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
        loss_fn=loss_fn,
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

    if not internal_test_split.empty:
        test_eval = evaluate_split(
            model=model,
            split_df=internal_test_split,
            split_name="test_internal",
            batch_size=int(config.batch_size),
            target_hw=target_hw,
            padding_mode=config.padding_mode,
            device=device,
            loss_fn=loss_fn,
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
        "model_name": config.model_name,
        "model_class_name": "DistanceRegressor2DCNN",
        "model_architecture_variant": config.model_architecture_variant,
        "model_architecture_summary": architecture_text(model),
        "input_representation": "NPZ shard key X; full-frame bbox image, grayscale, normalized [0, 1]",
        "input_shape": [1, int(target_hw[0]), int(target_hw[1])],
        "target_name": "distance_m",
        "optimizer": "Adam",
        "loss_function": f"HuberLoss(delta={config.huber_delta})",
        "hyperparameters": {
            "batch_size": int(config.batch_size),
            "epochs": int(config.epochs),
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
            "early_stopping_patience": int(config.early_stopping_patience),
            "model_name": config.model_name,
            "run_id": run_id,
            "padding_mode": config.padding_mode,
            "model_architecture_variant": config.model_architecture_variant,
            "progress_log_interval_batches": progress_log_interval_batches,
            "accuracy_tolerance_m": accuracy_tolerance_m,
            "extra_accuracy_tolerances_m": [float(value) for value in extra_accuracy_tolerances],
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
        dataset_summary=dataset_summary,
        split_summary=split_summary,
        metrics=metrics,
    )

    return {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "best_model_path": str(run_dir / "best.pt"),
        "last_model_path": str(run_dir / "latest.pt"),
        "metrics": metrics,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train first-pass 2D CNN distance regressor.")
    parser.add_argument("--training-data-root", default="training-data")
    parser.add_argument("--validation-data-root", default="validation-data")
    parser.add_argument("--output-root", default="models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--model-name", default="2d-cnn")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-name-suffix", default=None)
    parser.add_argument("--padding-mode", default="disabled")
    parser.add_argument("--model-architecture-variant", default="fast_v0_2")
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
    parser.add_argument("--change-note", default="CLI training run.")
    parser.add_argument("--enable-internal-test-split", action="store_true")
    parser.add_argument("--internal-test-fraction", type=float, default=0.1)
    return parser


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
        early_stopping_patience=args.early_stopping_patience,
        model_name=args.model_name,
        run_id=args.run_id,
        run_name_suffix=args.run_name_suffix,
        enable_internal_test_split=bool(args.enable_internal_test_split),
        internal_test_fraction=args.internal_test_fraction,
        padding_mode=args.padding_mode,
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
        change_note=args.change_note,
        entrypoint_type="cli",
        entrypoint_path="src/train.py",
    )
    result = train_distance_regressor(config)
    print(f"Training complete. Run directory: {result['run_dir']}")


if __name__ == "__main__":
    main()
