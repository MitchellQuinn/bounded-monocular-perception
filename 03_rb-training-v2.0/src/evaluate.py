"""Evaluation entrypoints for saved runs across topology families."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

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
    validate_task_contract_schema,
    validate_root_schema,
)
from .paths import DEFAULT_TRAINING_ROOT, DEFAULT_VALIDATION_ROOT, find_repo_root, resolve_data_root
from .plots import save_prediction_scatter, save_residual_plot
from .task_runtime import (
    append_head_chunks,
    batch_targets_to_tensor,
    batch_to_model_inputs,
    compute_task_loss,
    extract_prediction_heads,
    extract_target_heads,
    format_orientation_metrics_log_line,
    primary_sample_error,
    summarize_task_metrics_from_chunks,
)
from .topologies import (
    ResolvedTopologySpec,
    build_model_from_spec,
    resolve_topology_spec_from_mapping,
    task_contract_signature,
    topology_contract_signature,
    topology_spec_signature,
)
from .utils import read_json, utc_now_iso, write_json
from .topologies.contracts import reporting_train_losses


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
    loss_components: dict[str, float]
    task_metrics: dict[str, Any]
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


def _loss_weights_from_payload(payload: dict[str, Any]) -> dict[str, float]:
    return {
        "distance": float(payload.get("distance_loss_weight", 1.0)),
        "orientation": float(payload.get("orientation_loss_weight", 1.0)),
        "position": float(payload.get("position_loss_weight", 1.0)),
    }


def _declared_component_loss_names(task_contract: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        name
        for name in reporting_train_losses(task_contract)
        if str(name).strip() and str(name).strip() != "total_loss"
    )


def _selected_loss_components(
    loss_components: Mapping[str, float],
    task_contract: Mapping[str, Any],
) -> dict[str, float]:
    selected: dict[str, float] = {}
    for name in _declared_component_loss_names(task_contract):
        if name in loss_components:
            selected[str(name)] = float(loss_components[name])
    return selected


def evaluate_split(
    model: nn.Module,
    split_df: pd.DataFrame,
    split_name: str,
    batch_size: int,
    target_hw: tuple[int, int],
    padding_mode: str,
    device: torch.device,
    task_contract: dict[str, Any],
    huber_delta: float,
    loss_weights: dict[str, float] | None = None,
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
    loss_component_totals: dict[str, float] = {}
    within_tolerance_counts: dict[float, int] = {float(tolerance): 0 for tolerance in tolerance_values}
    pred_head_chunks: dict[str, list[np.ndarray]] = {}
    true_head_chunks: dict[str, list[np.ndarray]] = {}
    collected_rows: list[dict[str, Any]] = []
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
                target_columns=tuple(task_contract.get("target_columns", [])),
                include_bbox_features=(
                    str(task_contract.get("input_mode", "")).strip()
                    == "dual_stream_image_bbox_features"
                ),
                include_geometry=(
                    str(task_contract.get("input_mode", "")).strip()
                    == "tri_stream_distance_orientation_geometry"
                ),
                extra_input_array_keys=(
                    ("x_orientation_image",)
                    if str(task_contract.get("input_mode", "")).strip()
                    == "tri_stream_distance_orientation_geometry"
                    else ()
                ),
            ),
            start=1,
        ):
            model_inputs = batch_to_model_inputs(batch, task_contract, device=device)
            targets = batch_targets_to_tensor(batch, device=device)
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
            batch_n = int(targets.shape[0])
            total_loss += float(loss_result.total.item()) * batch_n
            for name, tensor_value in loss_result.components.items():
                loss_component_totals[name] = loss_component_totals.get(name, 0.0) + (
                    float(tensor_value.item()) * batch_n
                )
            total_count += batch_n

            primary_error = primary_sample_error(pred_heads, target_heads, task_contract)
            for tolerance in tolerance_values:
                within_tolerance_counts[float(tolerance)] += int(
                    torch.count_nonzero(primary_error <= float(tolerance)).item()
                )

            append_head_chunks(pred_head_chunks, pred_heads)
            append_head_chunks(true_head_chunks, target_heads)
            if collect_predictions:
                collected_rows.extend(batch.rows)

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
                    loss_text = f" running_loss={total_loss / float(total_count):.6f}"
                    summary_line = (
                        f"{log_prefix} batch {batch_index}/{total_batches_est} "
                        f"({pct:5.1f}%) seen={total_count}"
                        f" {accuracy_text}"
                        f"{loss_text} elapsed={elapsed:.1f}s"
                    )
                    running_task_metrics = summarize_task_metrics_from_chunks(
                        pred_head_chunks,
                        true_head_chunks,
                        task_contract,
                        tolerance_values=tolerance_values,
                        primary_tolerance=primary_tolerance,
                    ).task_metrics
                    orientation_log_line = format_orientation_metrics_log_line(
                        running_task_metrics,
                        contract=task_contract,
                        indent="        ",
                    )
                    if orientation_log_line:
                        summary_line = f"{summary_line}\n{orientation_log_line}"
                    print(summary_line, flush=True)

    if total_count == 0:
        raise ValueError(f"{split_name}: evaluation produced zero samples.")

    metrics_result = summarize_task_metrics_from_chunks(
        pred_head_chunks,
        true_head_chunks,
        task_contract,
        tolerance_values=tolerance_values,
        primary_tolerance=primary_tolerance,
        rows=collected_rows if collect_predictions else None,
        collect_predictions=collect_predictions,
    )
    accuracy_by_tolerance = {
        float(tolerance): (within_tolerance_counts[float(tolerance)] / float(total_count))
        for tolerance in tolerance_values
    }
    avg_loss = total_loss / total_count
    avg_loss_components = {
        name: float(total / float(total_count))
        for name, total in sorted(loss_component_totals.items())
    }

    return SplitEvaluation(
        split_name=split_name,
        sample_count=int(metrics_result.sample_count),
        accuracy_within_tolerance=float(accuracy_by_tolerance[primary_tolerance]),
        accuracy_by_tolerance=accuracy_by_tolerance,
        mae=float(metrics_result.mae),
        rmse=float(metrics_result.rmse),
        loss=avg_loss,
        loss_components=avg_loss_components,
        task_metrics=dict(metrics_result.task_metrics),
        y_true=metrics_result.primary_truth,
        y_pred=metrics_result.primary_prediction,
        predictions=metrics_result.predictions,
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
    expected_task_signature = str(run_config.get("task_contract_signature", "")).strip()
    if expected_task_signature:
        actual_task_signature = task_contract_signature(topology_spec)
        if actual_task_signature != expected_task_signature:
            raise ValueError(
                "Run config task_contract_signature does not match resolved task contract: "
                f"expected={expected_task_signature} actual={actual_task_signature}"
            )
    expected_topology_contract_signature = str(
        run_config.get("topology_contract_signature", "")
    ).strip()
    if expected_topology_contract_signature:
        actual_topology_contract_signature = topology_contract_signature(topology_spec)
        if actual_topology_contract_signature != expected_topology_contract_signature:
            raise ValueError(
                "Run config topology_contract_signature does not match resolved topology "
                "contract: "
                f"expected={expected_topology_contract_signature} "
                f"actual={actual_topology_contract_signature}"
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
    validate_task_contract_schema(
        val_metadata,
        val_schema,
        topology_spec.task_contract,
        root_name="validation",
    )
    huber_delta = float(run_config.get("huber_delta", 1.0))
    loss_weights = _loss_weights_from_payload(run_config)
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
        task_contract=dict(topology_spec.task_contract),
        huber_delta=huber_delta,
        loss_weights=loss_weights,
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
        "topology_contract_signature": topology_contract_signature(topology_spec),
        "task_contract_signature": task_contract_signature(topology_spec),
        "model_class_name": topology_spec.model_class_name,
        "model_architecture_variant": topology_spec.topology_variant,
        "model_topology": topology_spec.to_dict(),
        "topology_contract": dict(topology_spec.topology_contract),
        "task_contract": dict(topology_spec.task_contract),
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
    val_loss_components = _selected_loss_components(
        val_eval.loss_components,
        topology_spec.task_contract,
    )
    if val_loss_components:
        summary["validation"]["loss_components"] = val_loss_components
    if val_eval.task_metrics:
        summary["validation"].update(dict(val_eval.task_metrics))

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
        validate_task_contract_schema(
            training_metadata,
            training_schema,
            topology_spec.task_contract,
            root_name="training",
        )
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
                task_contract=dict(topology_spec.task_contract),
                huber_delta=huber_delta,
                loss_weights=loss_weights,
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
            test_loss_components = _selected_loss_components(
                test_eval.loss_components,
                topology_spec.task_contract,
            )
            if test_loss_components:
                summary["test_internal"]["loss_components"] = test_loss_components
            if test_eval.task_metrics:
                summary["test_internal"].update(dict(test_eval.task_metrics))
            if train_shard_cache is not None:
                summary["cache_runtime_stats"][
                    "test_internal_shard_cache"
                ] = train_shard_cache.stats()

    # Save canonical evaluation outputs in the run folder.
    if val_eval.predictions is None:
        raise RuntimeError("Expected prediction rows when collect_predictions=True.")
    val_eval.predictions.to_csv(run_dir / "sample_predictions.csv", index=False)
    scatter_path = run_dir / "prediction_scatter.png"
    residual_path = run_dir / "residual_plot.png"
    if val_eval.y_true.size > 0 and val_eval.y_pred.size > 0:
        save_prediction_scatter(val_eval.y_true, val_eval.y_pred, scatter_path)
        save_residual_plot(val_eval.y_true, val_eval.y_pred, residual_path)
    write_json(run_dir / "evaluation_summary.json", summary)

    result = {
        "summary": summary,
        "sample_predictions_path": str(run_dir / "sample_predictions.csv"),
        "evaluation_summary_path": str(run_dir / "evaluation_summary.json"),
    }
    if scatter_path.exists():
        result["prediction_scatter_path"] = str(scatter_path)
    if residual_path.exists():
        result["residual_plot_path"] = str(residual_path)
    return result


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
