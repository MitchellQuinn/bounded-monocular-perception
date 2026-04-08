"""Evaluation entrypoints for saved runs across topology families."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import EvalConfig
from .data import (
    ShardArrayCache,
    determine_target_hw,
    iter_batches,
    load_root_metadata,
    validate_root_schema,
)
from .paths import DEFAULT_TRAINING_ROOT, DEFAULT_VALIDATION_ROOT, find_repo_root, resolve_data_root
from .plots import save_prediction_scatter, save_residual_plot
from .topologies import (
    ResolvedTopologySpec,
    build_model_from_spec,
    resolve_topology_spec_from_mapping,
    topology_spec_signature,
)
from .utils import read_json, utc_now_iso, write_json


@dataclass
class SplitEvaluation:
    """Evaluation outputs for one split."""

    split_name: str
    sample_count: int
    accuracy_within_tolerance: float
    accuracy_by_tolerance: dict[float, float]
    mae: float
    rmse: float
    loss: float | None
    y_true: np.ndarray
    y_pred: np.ndarray
    predictions: pd.DataFrame | None


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


def evaluate_split(
    model: nn.Module,
    split_df: pd.DataFrame,
    split_name: str,
    batch_size: int,
    target_hw: tuple[int, int],
    padding_mode: str,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    collect_predictions: bool = False,
    accuracy_tolerance_m: float = 0.10,
    additional_accuracy_tolerances_m: Any = None,
    progress_log_interval_batches: int | None = None,
    progress_log_prefix: str | None = None,
    shard_cache: ShardArrayCache | None = None,
) -> SplitEvaluation:
    """Evaluate a model on one metadata split."""
    if split_df.empty:
        raise ValueError(f"{split_name}: split dataframe is empty.")
    if accuracy_tolerance_m <= 0:
        raise ValueError(
            f"accuracy_tolerance_m must be positive; got {accuracy_tolerance_m}"
        )
    tolerance_values = _parse_tolerance_values(
        [accuracy_tolerance_m, *_parse_tolerance_values(additional_accuracy_tolerances_m)]
    )
    if not tolerance_values:
        raise ValueError("No valid accuracy tolerances were configured.")
    primary_tolerance = float(min(tolerance_values, key=lambda value: abs(value - float(accuracy_tolerance_m))))

    model.eval()
    total_loss = 0.0
    total_count = 0
    within_tolerance_counts: dict[float, int] = {float(tolerance): 0 for tolerance in tolerance_values}
    pred_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    prediction_rows: list[dict[str, Any]] = []
    total_batches_est = max(1, int(math.ceil(len(split_df) / float(batch_size))))
    log_every = max(0, int(progress_log_interval_batches or 0))
    log_prefix = progress_log_prefix or f"[{split_name}]"
    started = perf_counter()

    with torch.no_grad():
        for batch_index, batch in enumerate(
            iter_batches(
                metadata_df=split_df,
                batch_size=batch_size,
                target_hw=target_hw,
                padding_mode=padding_mode,
                shuffle_shards=False,
                shard_cache=shard_cache,
            ),
            start=1,
        ):
            images = torch.from_numpy(batch.images).to(device=device, dtype=torch.float32)
            targets = torch.from_numpy(batch.targets).to(device=device, dtype=torch.float32)

            outputs = model(images).reshape(-1)
            if loss_fn is not None:
                loss_value = loss_fn(outputs, targets)
                total_loss += float(loss_value.item()) * int(targets.shape[0])
            total_count += int(targets.shape[0])
            abs_error = torch.abs(outputs - targets)
            for tolerance in tolerance_values:
                within_tolerance_counts[float(tolerance)] += int(
                    torch.count_nonzero(abs_error <= float(tolerance)).item()
                )

            y_pred = outputs.detach().cpu().numpy().astype(np.float32)
            y_true = targets.detach().cpu().numpy().astype(np.float32)
            pred_chunks.append(y_pred)
            true_chunks.append(y_true)

            if collect_predictions:
                for row, pred_val, true_val in zip(batch.rows, y_pred, y_true):
                    prediction_rows.append(
                        {
                            "dataset_id": row.get("dataset_id"),
                            "run_id": row.get("run_id"),
                            "sample_id": row.get("sample_id"),
                            "frame_index": row.get("frame_index"),
                            "npz_filename": row.get("npz_filename"),
                            "npz_row_index": row.get("npz_row_index"),
                            "truth_distance_m": float(true_val),
                            "prediction_distance_m": float(pred_val),
                            "residual_m": float(pred_val - true_val),
                            "source_root": row.get("source_root"),
                        }
                    )

            if log_every > 0:
                should_log = (
                    batch_index == 1
                    or (batch_index % log_every == 0)
                    or batch_index == total_batches_est
                )
                if should_log:
                    elapsed = perf_counter() - started
                    pct = 100.0 * (batch_index / float(total_batches_est))
                    accuracy_text = " ".join(
                        (
                            f"acc@{float(tolerance):.2f}m="
                            f"{within_tolerance_counts[float(tolerance)] / float(total_count):.4f}"
                        )
                        for tolerance in tolerance_values
                    )
                    loss_text = ""
                    if loss_fn is not None and total_count > 0:
                        loss_text = f" running_loss={total_loss / float(total_count):.6f}"
                    print(
                        f"{log_prefix} batch {batch_index}/{total_batches_est} "
                        f"({pct:5.1f}%) seen={total_count}"
                        f" {accuracy_text}"
                        f"{loss_text} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    y_true_all = np.concatenate(true_chunks) if true_chunks else np.array([], dtype=np.float32)
    y_pred_all = np.concatenate(pred_chunks) if pred_chunks else np.array([], dtype=np.float32)
    if y_true_all.size == 0:
        raise ValueError(f"{split_name}: evaluation produced zero samples.")

    residual = y_pred_all - y_true_all
    accuracy_by_tolerance = {
        float(tolerance): (within_tolerance_counts[float(tolerance)] / float(total_count))
        for tolerance in tolerance_values
    }
    accuracy_within_tolerance = float(accuracy_by_tolerance[primary_tolerance])
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    avg_loss = (total_loss / total_count) if (loss_fn is not None and total_count > 0) else None
    pred_df = pd.DataFrame(prediction_rows) if collect_predictions else None

    return SplitEvaluation(
        split_name=split_name,
        sample_count=int(y_true_all.size),
        accuracy_within_tolerance=float(accuracy_within_tolerance),
        accuracy_by_tolerance=accuracy_by_tolerance,
        mae=mae,
        rmse=rmse,
        loss=avg_loss,
        y_true=y_true_all,
        y_pred=y_pred_all,
        predictions=pred_df,
    )


def _load_model_from_run(
    run_dir: Path,
    run_config: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, ResolvedTopologySpec]:
    topology_spec = resolve_topology_spec_from_mapping(run_config)
    expected_signature = str(run_config.get("topology_signature", "")).strip()
    if expected_signature:
        actual_signature = topology_spec_signature(topology_spec)
        if actual_signature != expected_signature:
            raise ValueError(
                "Run config topology_signature does not match resolved topology fields: "
                f"expected={expected_signature} actual={actual_signature}"
            )

    model = build_model_from_spec(topology_spec)
    checkpoint_candidates = [run_dir / "best.pt", run_dir / "best_model.pt"]
    best_model_path = next((path for path in checkpoint_candidates if path.exists()), None)
    if best_model_path is None:
        candidate_text = ", ".join(str(path) for path in checkpoint_candidates)
        raise FileNotFoundError(f"Missing best model checkpoint. Checked: {candidate_text}")
    state = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, topology_spec


def _load_internal_test_split(
    split_membership_path: Path,
    training_metadata: pd.DataFrame,
) -> pd.DataFrame:
    if split_membership_path.suffix.lower() == ".json":
        membership = pd.read_json(split_membership_path)
    else:
        membership = pd.read_csv(split_membership_path)
    test_membership = membership[membership["split"] == "test_internal"].copy()
    if test_membership.empty:
        return pd.DataFrame()

    merge_keys = ["dataset_id", "npz_filename", "npz_row_index"]
    missing_keys = [key for key in merge_keys if key not in test_membership.columns]
    if missing_keys:
        raise ValueError(
            f"split_membership missing merge keys for internal test reconstruction: {missing_keys}"
        )
    reconstructed = training_metadata.merge(
        test_membership[merge_keys].drop_duplicates(),
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    if reconstructed.shape[0] != test_membership.shape[0]:
        raise ValueError(
            "Could not reconstruct full internal test split membership from training-data."
        )
    return reconstructed


def _resolve_run_directory(
    *,
    repo_root: Path,
    model_run_directory: str | Path,
) -> Path:
    """Resolve a concrete run directory that contains config.json.

    Supports either:
    - direct run directory path (contains config.json), or
    - model parent directory (contains runs/run_*/config.json), or
    - runs directory itself (contains run_*/config.json).
    """
    run_dir = Path(model_run_directory)
    if not run_dir.is_absolute():
        run_dir = (repo_root / run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    run_config_path = run_dir / "config.json"
    if run_config_path.exists():
        return run_dir

    candidates: list[Path] = []
    for base in (run_dir, run_dir / "runs"):
        if not base.exists() or not base.is_dir():
            continue
        for child in sorted(base.glob("run_*")):
            if child.is_dir():
                candidates.append(child.resolve())

    run_register_path = run_dir / "run_register.json"
    if run_register_path.exists():
        payload = read_json(run_register_path)
        runs = payload.get("runs", [])
        if isinstance(runs, list):
            for row in runs:
                if not isinstance(row, dict):
                    continue
                raw_run_path = row.get("run_dir")
                if not raw_run_path:
                    continue
                candidate = Path(str(raw_run_path))
                if not candidate.is_absolute():
                    candidate = (repo_root / candidate).resolve()
                if candidate.is_dir():
                    candidates.append(candidate)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)

    valid = [path for path in deduped if (path / "config.json").exists()]
    if valid:
        return valid[-1]

    raise FileNotFoundError(
        "Could not resolve a run directory with config.json from "
        f"{run_dir}. Checked direct path and run_* descendants."
    )


def evaluate_saved_run(config: EvalConfig | dict[str, Any]) -> dict[str, Any]:
    """Evaluate a saved run primarily against validation-data."""
    if isinstance(config, dict):
        config = EvalConfig.from_mapping(config)

    repo_root = find_repo_root()
    run_dir = _resolve_run_directory(
        repo_root=repo_root,
        model_run_directory=config.model_run_directory,
    )

    run_config_path = run_dir / "config.json"
    run_config = read_json(run_config_path)

    validation_root = resolve_data_root(
        repo_root,
        config.validation_data_root,
        default_rel=DEFAULT_VALIDATION_ROOT,
    )
    val_metadata, _ = load_root_metadata(validation_root, source_root="validation", repo_root=repo_root)
    val_schema = validate_root_schema(val_metadata, root_name="validation")

    padding_mode = config.padding_mode_override or str(run_config.get("padding_mode", "disabled"))
    if "target_height" in run_config and "target_width" in run_config:
        target_hw = (int(run_config["target_height"]), int(run_config["target_width"]))
    else:
        target_hw = determine_target_hw(val_schema, padding_mode=padding_mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, topology_spec = _load_model_from_run(run_dir, run_config=run_config, device=device)
    loss_fn = nn.HuberLoss(delta=float(run_config.get("huber_delta", 1.0)))
    progress_log_interval_batches = max(0, int(run_config.get("progress_log_interval_batches", 0)))
    accuracy_tolerance_m = float(run_config.get("accuracy_tolerance_m", 0.10))
    gib = float(1024**3)

    validation_shard_cache: ShardArrayCache | None = None
    if bool(run_config.get("cache_validation_in_ram", False)):
        val_cache_budget_gb = float(run_config.get("validation_cache_budget_gb", 0.0))
        val_cache_budget_bytes = int(max(0.0, val_cache_budget_gb) * gib)
        if val_cache_budget_bytes > 0:
            val_shard_paths = tuple(
                sorted(Path(path) for path in val_metadata["npz_path"].astype(str).unique())
            )
            print(
                f"[cache][validation][evaluation] preloading {len(val_shard_paths)} shard(s) "
                f"with budget={val_cache_budget_gb:.1f}GiB",
                flush=True,
            )
            preload_started = perf_counter()
            validation_shard_cache = ShardArrayCache(
                max_bytes=val_cache_budget_bytes,
                name="validation_eval_shard_cache",
            )
            validation_shard_cache.preload(val_shard_paths)
            preload_elapsed = perf_counter() - preload_started
            preload_stats = validation_shard_cache.stats()
            print(
                f"[cache][validation][evaluation] ready cached_shards={preload_stats['cached_shards']} "
                f"used_gib={float(preload_stats['bytes_used']) / gib:.1f} "
                f"hit_rate={preload_stats['hit_rate']:.3f} elapsed={preload_elapsed:.1f}s",
                flush=True,
            )

    val_eval = evaluate_split(
        model=model,
        split_df=val_metadata,
        split_name="validation",
        batch_size=int(config.batch_size),
        target_hw=target_hw,
        padding_mode=padding_mode,
        device=device,
        loss_fn=loss_fn,
        collect_predictions=True,
        accuracy_tolerance_m=accuracy_tolerance_m,
        additional_accuracy_tolerances_m=run_config.get("extra_accuracy_tolerances_m"),
        progress_log_interval_batches=progress_log_interval_batches,
        progress_log_prefix="[validation][evaluation]",
        shard_cache=validation_shard_cache,
    )

    summary: dict[str, Any] = {
        "evaluated_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "device": str(device),
        "topology_id": topology_spec.topology_id,
        "topology_variant": topology_spec.topology_variant,
        "topology_params": dict(topology_spec.topology_params),
        "topology_signature": topology_spec_signature(topology_spec),
        "model_class_name": topology_spec.model_class_name,
        "model_architecture_variant": topology_spec.topology_variant,
        "model_topology": topology_spec.to_dict(),
        "padding_mode": padding_mode,
        "target_hw": {"height": int(target_hw[0]), "width": int(target_hw[1])},
        "accuracy_tolerance_m": accuracy_tolerance_m,
        "cache_runtime_stats": {
            "validation_shard_cache": (
                validation_shard_cache.stats()
                if validation_shard_cache is not None
                else None
            )
        },
        "validation": {
            "sample_count": val_eval.sample_count,
            "loss": val_eval.loss,
            "accuracy_within_tolerance": val_eval.accuracy_within_tolerance,
            "accuracy_by_tolerance": _accuracy_by_tolerance_json(val_eval.accuracy_by_tolerance),
            "mae": val_eval.mae,
            "rmse": val_eval.rmse,
        },
    }

    split_membership_csv_path = run_dir / "split_membership.csv"
    split_membership_json_path = run_dir / "split_membership.json"
    split_membership_path = (
        split_membership_csv_path
        if split_membership_csv_path.exists()
        else split_membership_json_path
    )
    if (
        config.evaluate_internal_test_if_present
        and config.training_data_root is not None
        and split_membership_path.exists()
    ):
        training_root = resolve_data_root(
            repo_root,
            config.training_data_root,
            default_rel=DEFAULT_TRAINING_ROOT,
        )
        training_metadata, _ = load_root_metadata(
            training_root,
            source_root="training",
            repo_root=repo_root,
        )
        training_schema = validate_root_schema(training_metadata, root_name="training")
        _ = training_schema  # schema check side-effect
        internal_test_df = _load_internal_test_split(split_membership_path, training_metadata)
        if not internal_test_df.empty:
            train_shard_cache: ShardArrayCache | None = None
            train_cache_budget_gb = float(run_config.get("train_cache_budget_gb", 0.0))
            train_cache_budget_bytes = int(max(0.0, train_cache_budget_gb) * gib)
            if train_cache_budget_bytes > 0:
                train_shard_cache = ShardArrayCache(
                    max_bytes=train_cache_budget_bytes,
                    name="test_internal_eval_shard_cache",
                )
            test_eval = evaluate_split(
                model=model,
                split_df=internal_test_df,
                split_name="test_internal",
                batch_size=int(config.batch_size),
                target_hw=target_hw,
                padding_mode=padding_mode,
                device=device,
                loss_fn=loss_fn,
                collect_predictions=False,
                accuracy_tolerance_m=accuracy_tolerance_m,
                additional_accuracy_tolerances_m=run_config.get("extra_accuracy_tolerances_m"),
                progress_log_interval_batches=progress_log_interval_batches,
                progress_log_prefix="[test_internal][evaluation]",
                shard_cache=train_shard_cache,
            )
            summary["test_internal"] = {
                "sample_count": test_eval.sample_count,
                "loss": test_eval.loss,
                "accuracy_within_tolerance": test_eval.accuracy_within_tolerance,
                "accuracy_by_tolerance": _accuracy_by_tolerance_json(test_eval.accuracy_by_tolerance),
                "mae": test_eval.mae,
                "rmse": test_eval.rmse,
            }
            if train_shard_cache is not None:
                summary["cache_runtime_stats"][
                    "test_internal_shard_cache"
                ] = train_shard_cache.stats()

    # Save canonical evaluation outputs in the run folder.
    if val_eval.predictions is None:
        raise RuntimeError("Expected prediction rows when collect_predictions=True.")
    val_eval.predictions.to_csv(run_dir / "sample_predictions.csv", index=False)
    save_prediction_scatter(val_eval.y_true, val_eval.y_pred, run_dir / "prediction_scatter.png")
    save_residual_plot(val_eval.y_true, val_eval.y_pred, run_dir / "residual_plot.png")
    write_json(run_dir / "evaluation_summary.json", summary)

    return {
        "summary": summary,
        "sample_predictions_path": str(run_dir / "sample_predictions.csv"),
        "evaluation_summary_path": str(run_dir / "evaluation_summary.json"),
        "prediction_scatter_path": str(run_dir / "prediction_scatter.png"),
        "residual_plot_path": str(run_dir / "residual_plot.png"),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved run.")
    parser.add_argument("--model-run-directory", required=True)
    parser.add_argument("--validation-data-root", default="validation-data")
    parser.add_argument("--training-data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--padding-mode-override", default=None)
    parser.add_argument(
        "--evaluate-internal-test-if-present",
        action="store_true",
        help="If split_membership contains test_internal and training root is provided, evaluate it too.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = EvalConfig(
        model_run_directory=args.model_run_directory,
        validation_data_root=args.validation_data_root,
        training_data_root=args.training_data_root,
        batch_size=args.batch_size,
        padding_mode_override=args.padding_mode_override,
        evaluate_internal_test_if_present=bool(args.evaluate_internal_test_if_present),
    )
    result = evaluate_saved_run(cfg)
    print("Evaluation complete.")
    print(pd.Series(result["summary"]).to_string())


if __name__ == "__main__":
    main()
