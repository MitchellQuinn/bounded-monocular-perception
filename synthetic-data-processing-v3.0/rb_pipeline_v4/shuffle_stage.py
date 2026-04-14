"""Optional corpus shuffling for packed v4 dual-stream datasets."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import ShuffleStageConfigV4, StageSummaryV4
from .logging_utils import StageLogger
from .manifest import load_samples_csv, samples_csv_path, write_samples_csv
from .paths import training_run_paths
from .validation import PipelineValidationError, validate_run_structure


_REQUIRED_SOURCE_COLUMNS = ["sample_id", "npz_filename", "npz_row_index"]



def run_shuffle_stage_v4(
    project_root: Path,
    source_run_name: str,
    output_run_name: str,
    config: ShuffleStageConfigV4 | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV4:
    """Shuffle one packed v4 corpus into a new packed run."""

    stage_config = config or ShuffleStageConfigV4()
    source_run = str(source_run_name).strip()
    output_run = str(output_run_name).strip()

    if not source_run:
        raise ValueError("source_run_name cannot be blank")
    if not output_run:
        raise ValueError("output_run_name cannot be blank")

    output_root_name = stage_config.normalized_output_root_name()
    ledger_filename = stage_config.normalized_ledger_filename()

    source_paths = training_run_paths(project_root, source_run)
    validation_errors = validate_run_structure(source_paths, require_arrays=False)
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    output_run_root = project_root / output_root_name / output_run
    output_manifest_dir = output_run_root / "manifests"
    log_path = output_manifest_dir / "shuffle_stage_log.txt"
    logger = StageLogger(
        stage_name="shuffle",
        run_name=output_run,
        log_path=log_path,
        dry_run=stage_config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running v4 corpus shuffle from '{source_run}' to '{output_root_name}/{output_run}'")
    logger.log_parameters(stage_config.to_log_dict())

    source_samples = load_samples_csv(samples_csv_path(source_paths.manifests_dir))
    selected = _select_rows_for_shuffle(source_samples)

    if stage_config.strict_unique_sample_ids:
        duplicate_ids = selected["sample_id"].astype(str).duplicated(keep=False)
        if duplicate_ids.any():
            raise PipelineValidationError(
                f"Source manifest has duplicate sample_id values ({int(duplicate_ids.sum())} rows)."
            )

    rng = np.random.default_rng(int(stage_config.random_seed))
    order = rng.permutation(len(selected))
    shuffled = selected.iloc[order].copy().reset_index(drop=True)

    if output_run_root.exists():
        if not stage_config.overwrite:
            raise FileExistsError(f"Output run directory already exists: {output_run_root}")
        logger.log(f"Overwrite enabled; replacing output directory: {output_run_root}")
        if not stage_config.dry_run:
            shutil.rmtree(output_run_root)

    if not stage_config.dry_run:
        (output_run_root / "manifests").mkdir(parents=True, exist_ok=True)

    payload = _build_shuffled_payload(
        source_run_root=source_paths.root,
        shuffled_df=shuffled,
    )

    output_npz_name = f"{output_run}.npz"
    output_npz_path = output_run_root / output_npz_name
    if not stage_config.dry_run:
        save_fn = np.savez_compressed if stage_config.compress else np.savez
        save_fn(output_npz_path, **payload)

    output_samples = source_samples.loc[shuffled["_source_manifest_index"].tolist()].copy().reset_index(drop=True)
    if "run_id" in output_samples.columns:
        output_samples["run_id"] = output_run
    output_samples["npz_filename"] = output_npz_name
    output_samples["npz_row_index"] = np.arange(len(output_samples), dtype=np.int64)
    if "pack_dual_stream_stage_status" in output_samples.columns:
        output_samples["pack_dual_stream_stage_status"] = "success"
    if "pack_dual_stream_stage_error" in output_samples.columns:
        output_samples["pack_dual_stream_stage_error"] = ""

    ledger = pd.DataFrame(
        {
            "selection_index": np.arange(len(shuffled), dtype=np.int64),
            "source_run_name": source_run,
            "source_npz_filename": shuffled["npz_filename"].astype(str).tolist(),
            "source_npz_row_index": shuffled["npz_row_index"].astype(int).tolist(),
            "source_sample_id": shuffled["sample_id"].astype(str).tolist(),
            "target_run_name": output_run,
            "target_npz_filename": output_npz_name,
            "target_npz_row_index": np.arange(len(shuffled), dtype=np.int64),
        }
    )

    _copy_source_manifests(source_paths.manifests_dir, output_manifest_dir, dry_run=stage_config.dry_run)
    write_samples_csv(output_samples, output_manifest_dir / "samples.csv", dry_run=stage_config.dry_run)
    if not stage_config.dry_run:
        ledger.to_csv(output_manifest_dir / ledger_filename, index=False)

    _rewrite_run_json(
        output_manifest_dir / "run.json",
        source_run_name=source_run,
        output_run_name=output_run,
        dry_run=stage_config.dry_run,
        logger=logger,
    )

    total_rows = int(len(output_samples))
    logger.log_summary(
        total_rows=total_rows,
        successful_rows=total_rows,
        failed_rows=0,
        skipped_rows=0,
        output_path=output_run_root,
    )
    logger.write()

    return StageSummaryV4(
        run_name=output_run,
        stage_name="shuffle",
        total_rows=total_rows,
        successful_rows=total_rows,
        failed_rows=0,
        skipped_rows=0,
        output_path=str(output_run_root),
        log_path=str(log_path),
        dry_run=stage_config.dry_run,
    )



def _select_rows_for_shuffle(samples_df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _REQUIRED_SOURCE_COLUMNS if column not in samples_df.columns]
    if missing:
        raise PipelineValidationError(f"Missing required source columns: {', '.join(missing)}")

    npz_filename = samples_df["npz_filename"].astype("string").fillna("").str.strip()
    npz_row_index = pd.to_numeric(samples_df["npz_row_index"], errors="coerce")

    include_mask = npz_filename.ne("") & npz_row_index.notna()
    if "pack_dual_stream_stage_status" in samples_df.columns:
        status = samples_df["pack_dual_stream_stage_status"].astype("string").fillna("").str.strip().str.lower()
        include_mask &= status.eq("success")

    out = samples_df.loc[include_mask].copy()
    out["npz_filename"] = npz_filename.loc[include_mask].astype(str)
    out["npz_row_index"] = npz_row_index.loc[include_mask].astype(int)
    out["_source_manifest_index"] = out.index.astype(int)

    if out.empty:
        raise PipelineValidationError("No pack-success rows found in source samples.csv")

    duplicated = out.duplicated(subset=["npz_filename", "npz_row_index"], keep=False)
    if duplicated.any():
        raise PipelineValidationError(
            f"Source manifest has duplicated (npz_filename, npz_row_index) rows: {int(duplicated.sum())}"
        )

    return out



def _build_shuffled_payload(source_run_root: Path, shuffled_df: pd.DataFrame) -> dict[str, np.ndarray]:
    first_npz = source_run_root / str(shuffled_df.iloc[0]["npz_filename"])
    with np.load(first_npz, allow_pickle=False) as data:
        keys = [key for key in data.files if key != "npz_row_index"]

    cache: dict[str, dict[str, np.ndarray]] = {}
    buckets: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    for _, row in shuffled_df.iterrows():
        npz_name = str(row["npz_filename"])
        row_idx = int(row["npz_row_index"])

        if npz_name not in cache:
            with np.load(source_run_root / npz_name, allow_pickle=False) as data:
                cache[npz_name] = {key: np.asarray(data[key]) for key in data.files}

        arrays = cache[npz_name]
        for key in keys:
            buckets[key].append(np.asarray(arrays[key][row_idx]))

    payload: dict[str, np.ndarray] = {}
    for key, values in buckets.items():
        sample = values[0]
        if sample.ndim == 0:
            payload[key] = np.asarray(values)
        else:
            payload[key] = np.stack(values, axis=0)

        if payload[key].dtype.kind in {"U", "S", "O"}:
            payload[key] = np.asarray(payload[key], dtype=str)

    payload["npz_row_index"] = np.arange(len(shuffled_df), dtype=np.int64)
    return payload



def _copy_source_manifests(source_manifest_dir: Path, target_manifest_dir: Path, *, dry_run: bool) -> None:
    if dry_run:
        return

    target_manifest_dir.mkdir(parents=True, exist_ok=True)
    for source_file in source_manifest_dir.iterdir():
        if source_file.is_file():
            shutil.copy2(source_file, target_manifest_dir / source_file.name)



def _rewrite_run_json(
    run_json_path: Path,
    *,
    source_run_name: str,
    output_run_name: str,
    dry_run: bool,
    logger: StageLogger,
) -> None:
    if dry_run or not run_json_path.is_file():
        return

    try:
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload["RunId"] = output_run_name
            payload["ShuffledFromRunId"] = source_run_name
            run_json_path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    except Exception as exc:
        logger.log(f"Could not update run.json metadata ({run_json_path.name}): {exc}")
