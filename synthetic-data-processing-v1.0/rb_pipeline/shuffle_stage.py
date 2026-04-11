"""Standalone corpus shuffling: training-data run -> shuffled NPZ run."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import ShuffleStageConfig, StageSummary
from .logging_utils import StageLogger
from .manifest import load_samples_csv, samples_csv_path, write_samples_csv
from .paths import training_run_paths
from .validation import PipelineValidationError, validate_npz_file, validate_run_structure


_REQUIRED_SOURCE_COLUMNS = ["sample_id", "npz_filename", "npz_row_index"]


@dataclass(frozen=True)
class _ArraySpec:
    dtype: np.dtype
    tail_shape: tuple[int, ...]
    string_like: bool


@dataclass(frozen=True)
class _ShuffleAssignment:
    source_manifest_row_idx: int
    source_npz_filename: str
    source_npz_row_index: int
    source_sample_id: str
    target_npz_filename: str
    target_npz_row_index: int


def _sanitize_ledger_filename(ledger_filename: str) -> str:
    name = str(ledger_filename).strip()
    if not name:
        raise ValueError("ledger_filename cannot be blank")

    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("ledger_filename must be a simple relative filename")

    return candidate.as_posix()


def _prepare_output_run_dir(
    output_run_root: Path,
    *,
    overwrite: bool,
    dry_run: bool,
    logger: StageLogger,
) -> None:
    if output_run_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output run directory already exists: {output_run_root}")
        logger.log(f"Overwrite enabled; existing output directory will be replaced: {output_run_root}")
        if not dry_run:
            shutil.rmtree(output_run_root)

    if not dry_run:
        (output_run_root / "manifests").mkdir(parents=True, exist_ok=True)


def _included_source_rows(samples_df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _REQUIRED_SOURCE_COLUMNS if column not in samples_df.columns]
    if missing:
        raise PipelineValidationError(f"Missing required source columns: {', '.join(missing)}")

    npz_filename = samples_df["npz_filename"].astype("string").fillna("").str.strip()
    npz_row_index = pd.to_numeric(samples_df["npz_row_index"], errors="coerce")

    include_mask = npz_filename.ne("") & npz_row_index.notna()
    if "pack_stage_status" in samples_df.columns:
        pack_status = samples_df["pack_stage_status"].astype("string").fillna("").str.strip().str.lower()
        include_mask &= pack_status.eq("success")

    included_df = samples_df.loc[include_mask].copy()
    included_df["npz_filename"] = npz_filename.loc[include_mask].astype(str)
    included_df["npz_row_index"] = npz_row_index.loc[include_mask].astype(int)

    if included_df.empty:
        raise PipelineValidationError(
            "No pack-success rows were found in samples.csv. "
            "Expected non-empty npz_filename/npz_row_index values."
        )

    duplicated_selection = included_df.duplicated(subset=["npz_filename", "npz_row_index"], keep=False)
    if duplicated_selection.any():
        duplicated_count = int(duplicated_selection.sum())
        raise PipelineValidationError(
            f"Source manifest has duplicated (npz_filename, npz_row_index) references: {duplicated_count} rows."
        )

    return included_df


def _collect_source_npz_metadata(
    source_run_root: Path,
    included_df: pd.DataFrame,
) -> tuple[list[str], dict[str, int], list[str], dict[str, _ArraySpec]]:
    source_npz_names = sorted(included_df["npz_filename"].astype(str).unique().tolist())
    if not source_npz_names:
        raise PipelineValidationError("No source NPZ filenames were found in samples.csv.")

    shard_sizes: dict[str, int] = {}
    source_keys: list[str] | None = None
    array_specs: dict[str, _ArraySpec] = {}

    for npz_name in source_npz_names:
        npz_path = source_run_root / npz_name
        if not npz_path.is_file():
            raise PipelineValidationError(f"Missing source NPZ file referenced by manifest: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as data:
            current_keys = list(data.files)
            if source_keys is None:
                source_keys = current_keys
            elif current_keys != source_keys:
                raise PipelineValidationError(
                    f"Inconsistent NPZ schema. Expected keys {source_keys}, found {current_keys} in {npz_name}."
                )

            if "sample_id" not in current_keys:
                raise PipelineValidationError(f"NPZ file is missing required key 'sample_id': {npz_name}")
            if "X" not in current_keys:
                raise PipelineValidationError(f"NPZ file is missing required key 'X': {npz_name}")

            shard_size = int(len(data["sample_id"]))
            shard_sizes[npz_name] = shard_size

            for key in current_keys:
                array = data[key]
                if array.shape and int(array.shape[0]) != shard_size:
                    raise PipelineValidationError(
                        f"NPZ key '{key}' in {npz_name} has invalid first dimension {array.shape[0]} "
                        f"(expected {shard_size})."
                    )

                spec = _ArraySpec(
                    dtype=np.dtype(array.dtype),
                    tail_shape=tuple(int(value) for value in array.shape[1:]),
                    string_like=array.dtype.kind in {"U", "S", "O"},
                )

                if key not in array_specs:
                    array_specs[key] = spec
                    continue

                previous = array_specs[key]
                if previous.tail_shape != spec.tail_shape:
                    raise PipelineValidationError(
                        f"NPZ key '{key}' has inconsistent sample shape between files: "
                        f"{previous.tail_shape} vs {spec.tail_shape}."
                    )
                if not previous.string_like and previous.dtype != spec.dtype:
                    raise PipelineValidationError(
                        f"NPZ key '{key}' has inconsistent dtype between files: {previous.dtype} vs {spec.dtype}."
                    )

    assert source_keys is not None
    return source_npz_names, shard_sizes, source_keys, array_specs


def _validate_manifest_row_coverage(
    included_df: pd.DataFrame,
    source_npz_names: list[str],
    shard_sizes: dict[str, int],
) -> None:
    for npz_name in source_npz_names:
        shard_size = shard_sizes[npz_name]
        row_indices = included_df.loc[included_df["npz_filename"] == npz_name, "npz_row_index"].astype(int).tolist()
        if len(row_indices) != shard_size:
            raise PipelineValidationError(
                f"Manifest row count for '{npz_name}' is {len(row_indices)}, but NPZ contains {shard_size} rows."
            )

        index_set = set(row_indices)
        expected_set = set(range(shard_size))
        if index_set != expected_set:
            raise PipelineValidationError(
                f"Manifest row indices for '{npz_name}' must cover 0..{shard_size - 1} exactly."
            )


def _build_target_npz_plan(
    source_run_name: str,
    output_run_name: str,
    source_npz_names: list[str],
    shard_sizes: dict[str, int],
) -> list[tuple[str, int]]:
    if len(source_npz_names) == 1 and source_npz_names[0] == f"{source_run_name}.npz":
        return [(f"{output_run_name}.npz", shard_sizes[source_npz_names[0]])]

    plan: list[tuple[str, int]] = []
    for idx, source_name in enumerate(source_npz_names):
        plan.append((f"{output_run_name}_shard_{idx:05d}.npz", shard_sizes[source_name]))
    return plan


def _copy_source_manifests(source_manifest_dir: Path, target_manifest_dir: Path, dry_run: bool) -> None:
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
        content = json.loads(run_json_path.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            content["RunId"] = output_run_name
            content["ShuffledFromRunId"] = source_run_name
            run_json_path.write_text(json.dumps(content, indent=4) + "\n", encoding="utf-8")
    except Exception as exc:
        logger.log(f"Could not update run.json metadata ({run_json_path.name}): {exc}")


def _build_assignments(
    included_df: pd.DataFrame,
    target_plan: list[tuple[str, int]],
    *,
    random_seed: int,
    strict_unique_sample_ids: bool,
) -> list[_ShuffleAssignment]:
    source_refs: list[tuple[int, str, int, str]] = []
    for row_idx in included_df.index:
        source_refs.append(
            (
                int(row_idx),
                str(included_df.at[row_idx, "npz_filename"]),
                int(included_df.at[row_idx, "npz_row_index"]),
                str(included_df.at[row_idx, "sample_id"]),
            )
        )

    rng = np.random.default_rng(int(random_seed))
    order = rng.permutation(len(source_refs))

    shuffled_refs = [source_refs[int(i)] for i in order]

    seen_selection_keys: set[tuple[str, int]] = set()
    seen_sample_ids: set[str] = set()
    assignments: list[_ShuffleAssignment] = []

    cursor = 0
    for target_npz_filename, target_size in target_plan:
        for target_row_index in range(target_size):
            source_manifest_row_idx, source_npz_filename, source_npz_row_index, source_sample_id = shuffled_refs[cursor]
            cursor += 1

            selection_key = (source_npz_filename, source_npz_row_index)
            if selection_key in seen_selection_keys:
                raise PipelineValidationError(
                    "Duplicate source selection detected during assignment. "
                    "This indicates an invalid selection state."
                )
            seen_selection_keys.add(selection_key)

            if strict_unique_sample_ids and source_sample_id in seen_sample_ids:
                raise PipelineValidationError(
                    f"Duplicate sample_id detected in shuffled corpus assignment: {source_sample_id}"
                )
            seen_sample_ids.add(source_sample_id)

            assignments.append(
                _ShuffleAssignment(
                    source_manifest_row_idx=source_manifest_row_idx,
                    source_npz_filename=source_npz_filename,
                    source_npz_row_index=source_npz_row_index,
                    source_sample_id=source_sample_id,
                    target_npz_filename=target_npz_filename,
                    target_npz_row_index=target_row_index,
                )
            )

    return assignments


def _write_output_npz_files(
    source_run_root: Path,
    output_run_root: Path,
    assignments: list[_ShuffleAssignment],
    target_plan: list[tuple[str, int]],
    source_keys: list[str],
    array_specs: dict[str, _ArraySpec],
    *,
    compress: bool,
    dry_run: bool,
    logger: StageLogger,
) -> None:
    if dry_run:
        logger.log("Dry run enabled: NPZ files will not be written.")
        return

    keys_without_row_index = [key for key in source_keys if key != "npz_row_index"]
    assignment_df = pd.DataFrame(
        {
            "source_npz_filename": [a.source_npz_filename for a in assignments],
            "source_npz_row_index": [a.source_npz_row_index for a in assignments],
            "target_npz_filename": [a.target_npz_filename for a in assignments],
            "target_npz_row_index": [a.target_npz_row_index for a in assignments],
        }
    )

    for target_npz_filename, target_size in target_plan:
        target_rows_df = assignment_df.loc[assignment_df["target_npz_filename"] == target_npz_filename]
        if len(target_rows_df) != target_size:
            raise PipelineValidationError(
                f"Internal assignment mismatch for {target_npz_filename}: "
                f"expected {target_size} rows, got {len(target_rows_df)}."
            )

        target_arrays: dict[str, np.ndarray] = {}
        for key in keys_without_row_index:
            spec = array_specs[key]
            full_shape = (target_size, *spec.tail_shape)
            target_arrays[key] = np.empty(full_shape, dtype=(object if spec.string_like else spec.dtype))

        fill_mask = np.zeros(target_size, dtype=bool)

        for source_npz_filename in sorted(target_rows_df["source_npz_filename"].unique().tolist()):
            source_rows_df = target_rows_df.loc[target_rows_df["source_npz_filename"] == source_npz_filename]
            source_idx = source_rows_df["source_npz_row_index"].to_numpy(dtype=np.int64)
            target_idx = source_rows_df["target_npz_row_index"].to_numpy(dtype=np.int64)

            source_path = source_run_root / source_npz_filename
            with np.load(source_path, allow_pickle=False) as data:
                for key in keys_without_row_index:
                    target_arrays[key][target_idx] = data[key][source_idx]
            fill_mask[target_idx] = True

        if not bool(np.all(fill_mask)):
            missing = int((~fill_mask).sum())
            raise PipelineValidationError(
                f"Internal assignment error: {missing} rows were not filled for {target_npz_filename}."
            )

        payload: dict[str, np.ndarray] = {}
        for key in keys_without_row_index:
            spec = array_specs[key]
            value = target_arrays[key]
            payload[key] = np.asarray(value, dtype=str) if spec.string_like else value
        payload["npz_row_index"] = np.arange(target_size, dtype=np.int64)

        target_npz_path = output_run_root / target_npz_filename
        save_fn = np.savez_compressed if compress else np.savez
        save_fn(target_npz_path, **payload)

        if "X" in payload:
            validate_npz_file(target_npz_path, allowed_x_dtypes={np.dtype(payload["X"].dtype)})
        else:
            validate_npz_file(target_npz_path)

        logger.log(
            f"Wrote shuffled shard '{target_npz_filename}' rows={target_size}, "
            f"dtype={payload['X'].dtype}, compression={'on' if compress else 'off'}"
        )


def run_shuffle_stage(
    project_root: Path,
    source_run_name: str,
    output_run_name: str,
    config: ShuffleStageConfig | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummary:
    """Shuffle one packed training corpus into a new duplicate-free corpus."""

    stage_config = config or ShuffleStageConfig()
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
    logger.log(f"Running corpus shuffle from '{source_run}' to '{output_root_name}/{output_run}'")
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

    output_samples_df = source_samples_df.loc[[item.source_manifest_row_idx for item in assignments]].copy().reset_index(drop=True)
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

    return StageSummary(
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
