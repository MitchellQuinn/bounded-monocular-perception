"""Standalone corpus shuffling for v3 training corpuses."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from rb_pipeline.logging_utils import StageLogger
from rb_pipeline.manifest import load_samples_csv, samples_csv_path, write_samples_csv
from rb_pipeline.shuffle_stage import (
    _build_assignments,
    _build_target_npz_plan,
    _collect_source_npz_metadata,
    _copy_source_manifests,
    _included_source_rows,
    _prepare_output_run_dir,
    _rewrite_run_json,
    _sanitize_ledger_filename,
    _validate_manifest_row_coverage,
    _write_output_npz_files,
)
from rb_pipeline.validation import PipelineValidationError, validate_run_structure

from .config import ShuffleStageConfigV3, StageSummaryV3
from .paths import training_v3_run_paths



def run_shuffle_stage_v3(
    project_root: Path,
    source_run_name: str,
    output_run_name: str,
    config: ShuffleStageConfigV3 | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV3:
    """Shuffle one packed v3 training corpus into a new duplicate-free corpus."""

    stage_config = config or ShuffleStageConfigV3()
    source_run = str(source_run_name).strip()
    output_run = str(output_run_name).strip()

    if not source_run:
        raise ValueError("source_run_name cannot be blank")
    if not output_run:
        raise ValueError("output_run_name cannot be blank")

    output_root_name = str(stage_config.output_root_name).strip()
    if not output_root_name:
        raise ValueError("output_root_name cannot be blank")

    ledger_filename = _sanitize_ledger_filename(stage_config.ledger_filename)

    source_paths = training_v3_run_paths(project_root, source_run)
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
    logger.log(f"Running v3 corpus shuffle from '{source_run}' to '{output_root_name}/{output_run}'")
    logger.log_parameters(stage_config.to_log_dict())

    source_samples_path = samples_csv_path(source_paths.manifests_dir)
    source_samples_df = load_samples_csv(source_samples_path)
    included_df = _included_source_rows(source_samples_df)

    if stage_config.strict_unique_sample_ids:
        duplicate_ids = included_df["sample_id"].astype(str).duplicated(keep=False)
        if duplicate_ids.any():
            duplicated_count = int(duplicate_ids.sum())
            raise PipelineValidationError(
                f"Source manifest has duplicate sample_id values across pack-success rows ({duplicated_count} rows). "
                "Set strict_unique_sample_ids=False to allow this."
            )

    source_npz_names, shard_sizes, source_keys, array_specs = _collect_source_npz_metadata(
        source_paths.root,
        included_df,
    )
    _validate_manifest_row_coverage(included_df, source_npz_names, shard_sizes)

    target_plan = _build_target_npz_plan(source_run, output_run, source_npz_names, shard_sizes)
    total_samples = int(sum(size for _, size in target_plan))
    logger.log(f"Source NPZ files: {len(source_npz_names)}")
    logger.log(f"Total included samples: {total_samples}")

    _prepare_output_run_dir(
        output_run_root,
        overwrite=stage_config.overwrite,
        dry_run=stage_config.dry_run,
        logger=logger,
    )

    assignments = _build_assignments(
        included_df,
        target_plan,
        random_seed=stage_config.random_seed,
        strict_unique_sample_ids=stage_config.strict_unique_sample_ids,
    )

    _write_output_npz_files(
        source_paths.root,
        output_run_root,
        assignments,
        target_plan,
        source_keys,
        array_specs,
        compress=stage_config.compress,
        dry_run=stage_config.dry_run,
        logger=logger,
    )

    output_samples_df = source_samples_df.loc[
        [item.source_manifest_row_idx for item in assignments]
    ].copy().reset_index(drop=True)
    if "run_id" in output_samples_df.columns:
        output_samples_df["run_id"] = output_run
    output_samples_df["npz_filename"] = [item.target_npz_filename for item in assignments]
    output_samples_df["npz_row_index"] = [item.target_npz_row_index for item in assignments]
    if "pack_stage_status" in output_samples_df.columns:
        output_samples_df["pack_stage_status"] = "success"
    if "pack_stage_error" in output_samples_df.columns:
        output_samples_df["pack_stage_error"] = ""

    ledger_df = pd.DataFrame(
        {
            "selection_index": np.arange(len(assignments), dtype=np.int64),
            "source_run_name": source_run,
            "source_npz_filename": [item.source_npz_filename for item in assignments],
            "source_npz_row_index": [item.source_npz_row_index for item in assignments],
            "source_sample_id": [item.source_sample_id for item in assignments],
            "target_run_name": output_run,
            "target_npz_filename": [item.target_npz_filename for item in assignments],
            "target_npz_row_index": [item.target_npz_row_index for item in assignments],
        }
    )

    _copy_source_manifests(source_paths.manifests_dir, output_manifest_dir, stage_config.dry_run)
    write_samples_csv(output_samples_df, output_manifest_dir / "samples.csv", dry_run=stage_config.dry_run)
    if not stage_config.dry_run:
        ledger_df.to_csv(output_manifest_dir / ledger_filename, index=False)

    _rewrite_run_json(
        output_manifest_dir / "run.json",
        source_run_name=source_run,
        output_run_name=output_run,
        dry_run=stage_config.dry_run,
        logger=logger,
    )

    if stage_config.dry_run:
        logger.log(f"Dry run: ledger would be written to {output_manifest_dir / ledger_filename}")
    else:
        logger.log(f"Wrote ledger: {output_manifest_dir / ledger_filename}")
    logger.log_summary(
        total_rows=total_samples,
        successful_rows=total_samples,
        failed_rows=0,
        skipped_rows=0,
        output_path=output_run_root,
    )
    logger.write()

    return StageSummaryV3(
        run_name=output_run,
        stage_name="shuffle",
        total_rows=total_samples,
        successful_rows=total_samples,
        failed_rows=0,
        skipped_rows=0,
        output_path=str(output_run_root),
        log_path=str(log_path),
        dry_run=stage_config.dry_run,
    )
