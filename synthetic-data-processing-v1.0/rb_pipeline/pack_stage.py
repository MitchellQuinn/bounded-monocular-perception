"""Stage 4: NPY arrays -> one or more NPZ shards per run."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import PackStageConfig, StageSummary
from .logging_utils import StageLogger
from .manifest import (
    PREPROCESSING_CONTRACT_KEY,
    NPY_STAGE_COLUMNS,
    PACK_STAGE_COLUMNS,
    append_columns,
    load_run_json,
    load_samples_csv,
    samples_csv_path,
    upsert_preprocessing_contract,
    write_samples_csv,
)
from .paths import normalize_relative_filename, resolve_manifest_path, training_run_paths
from .validation import (
    PipelineValidationError,
    validate_npz_file,
    validate_pack_array,
    validate_required_columns,
    validate_run_structure,
)


_REQUIRED_PACK_COLUMNS = ["sample_id", "image_filename", "distance_m"] + NPY_STAGE_COLUMNS
_OPTIONAL_FILENAME_COLUMNS = ["edge_image_filename", "bbox_image_filename", "npy_filename"]
_VALID_PACK_OUTPUT_DTYPES = {"preserve", "float32", "float16", "uint8"}


def _normalize_pack_output_dtype(output_dtype: str) -> str:
    dtype_name = str(output_dtype).strip().lower()
    if dtype_name not in _VALID_PACK_OUTPUT_DTYPES:
        allowed = ", ".join(sorted(_VALID_PACK_OUTPUT_DTYPES))
        raise ValueError(f"Unsupported pack output_dtype '{output_dtype}'. Allowed: {allowed}.")
    return dtype_name


def _normalize_shard_size(shard_size: int) -> int:
    size = int(shard_size)
    if size < 0:
        raise ValueError("shard_size must be >= 0")
    return size


def _existing_npz_paths(run_root: Path, run_name: str) -> list[Path]:
    """Return existing NPZ outputs for this run (single-file and sharded variants)."""

    candidates = [run_root / f"{run_name}.npz", *sorted(run_root.glob(f"{run_name}_shard_*.npz"))]
    return [path for path in candidates if path.is_file()]


def _shard_filename(run_name: str, shard_idx: int, use_shards: bool) -> str:
    if use_shards:
        return f"{run_name}_shard_{shard_idx:05d}.npz"
    return f"{run_name}.npz"


def _coerce_pack_array_dtype(array: np.ndarray, output_dtype: str) -> np.ndarray:
    """Convert one array to configured output dtype."""

    if output_dtype == "preserve":
        return array

    if output_dtype == "float32":
        return array.astype(np.float32, copy=False)

    if output_dtype == "float16":
        return array.astype(np.float16, copy=False)

    if output_dtype == "uint8":
        if np.issubdtype(array.dtype, np.floating):
            if np.isnan(array).any() or np.isinf(array).any():
                raise ValueError("Array contains NaN or infinite values")
            min_value = float(np.min(array))
            max_value = float(np.max(array))
            # If values are normalized [0, 1], quantize to 0..255.
            if min_value >= -1e-6 and max_value <= 1.0 + 1e-6:
                scaled = array * 255.0
            else:
                scaled = array
            return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)

        if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
            return np.clip(array, 0, 255).astype(np.uint8)

        raise ValueError(f"Cannot convert dtype {array.dtype} to uint8")

    # Guardrail: keep explicit error even though normalize function already validates.
    raise ValueError(f"Unsupported pack output dtype: {output_dtype}")



def run_pack_stage(
    project_root: Path,
    run_name: str,
    config: PackStageConfig | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummary:
    """Pack successful NPY arrays into NPZ outputs for one run."""

    stage_config = config or PackStageConfig()
    output_dtype = _normalize_pack_output_dtype(stage_config.output_dtype)
    shard_size = _normalize_shard_size(stage_config.shard_size)
    use_shards = shard_size > 0

    run_paths = training_run_paths(project_root, run_name)

    validation_errors = validate_run_structure(run_paths, require_arrays=True)
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    samples_path = samples_csv_path(run_paths.manifests_dir)
    samples_df = load_samples_csv(samples_path)

    validation_errors.extend(validate_required_columns(samples_df, _REQUIRED_PACK_COLUMNS))
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    append_columns(samples_df, PACK_STAGE_COLUMNS)

    run_payload = load_run_json(run_paths.manifests_dir)
    prior_contract = run_payload.get(PREPROCESSING_CONTRACT_KEY)
    prior_representation = (
        prior_contract.get("CurrentRepresentation", {})
        if isinstance(prior_contract, dict)
        else {}
    )
    effective_array_dtype = (
        output_dtype
        if output_dtype != "preserve"
        else str(prior_representation.get("ArrayDType", "preserve"))
    )
    current_representation = {
        "Kind": "full_frame_bbox_array",
        "StorageFormat": "npz",
        "ArrayKey": "X",
        "ColorSpace": "grayscale",
        "Geometry": "full_frame_bbox_outline",
        "ArrayLayout": "N,H,W",
        "ArrayDType": effective_array_dtype,
    }
    if "Normalize" in prior_representation:
        current_representation["Normalize"] = bool(prior_representation["Normalize"])
    if "Invert" in prior_representation:
        current_representation["Invert"] = bool(prior_representation["Invert"])
    upsert_preprocessing_contract(
        run_paths.manifests_dir,
        stage_name="pack",
        stage_parameters={
            "OutputDType": output_dtype,
            "EffectiveArrayDType": effective_array_dtype,
            "Compress": bool(stage_config.compress),
            "ShardSize": int(shard_size),
        },
        current_representation=current_representation,
        dry_run=stage_config.dry_run,
    )

    samples_df["npz_filename"] = samples_df["npz_filename"].astype("string")
    samples_df["npz_row_index"] = pd.to_numeric(samples_df["npz_row_index"], errors="coerce").astype("Int64")
    samples_df["pack_stage_status"] = samples_df["pack_stage_status"].astype("string")
    samples_df["pack_stage_error"] = samples_df["pack_stage_error"].astype("string")

    log_path = run_paths.manifests_dir / "pack_stage_log.txt"
    logger = StageLogger(
        stage_name="pack",
        run_name=run_name,
        log_path=log_path,
        dry_run=stage_config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running pack stage for run '{run_name}'")
    logger.log_parameters(
        stage_config.to_log_dict()
        | {
            "npz_pattern": (
                f"{run_name}_shard_00000.npz, {run_name}_shard_00001.npz, ..."
                if use_shards
                else f"{run_name}.npz"
            )
        }
    )

    existing_npz_paths = _existing_npz_paths(run_paths.root, run_name)
    overwrite_blocked = bool(existing_npz_paths) and not stage_config.overwrite
    if overwrite_blocked:
        existing_preview = ", ".join(str(path.name) for path in existing_npz_paths[:4])
        more_suffix = " ..." if len(existing_npz_paths) > 4 else ""
        logger.log(f"NPZ already exists and overwrite is false: {existing_preview}{more_suffix}")
    elif existing_npz_paths and stage_config.overwrite and not stage_config.dry_run:
        for existing in existing_npz_paths:
            existing.unlink()
            logger.log(f"Removed existing NPZ: {existing}")

    expected_shape: tuple[int, int] | None = None
    written_npz_paths: list[Path] = []
    current_shard_idx = 0
    total_rows = len(samples_df)
    progress_step = max(1, total_rows // 100) if total_rows > 0 else 1
    logger.log(f"Progress updates: every ~1% ({progress_step} rows)")
    processed_rows = 0

    arrays_buffer: list[np.ndarray] = []
    labels_buffer: list[np.float32] = []
    sample_ids_buffer: list[str] = []
    image_filenames_buffer: list[str] = []
    optional_values_buffer: dict[str, list[str]] = {
        column: [] for column in _OPTIONAL_FILENAME_COLUMNS if column in samples_df.columns
    }
    row_indices_buffer: list[int] = []
    npy_paths_buffer: list[Path] = []

    aborted = False

    def _reset_buffers() -> None:
        arrays_buffer.clear()
        labels_buffer.clear()
        sample_ids_buffer.clear()
        image_filenames_buffer.clear()
        row_indices_buffer.clear()
        npy_paths_buffer.clear()
        for values in optional_values_buffer.values():
            values.clear()

    def _flush_shard() -> bool:
        nonlocal current_shard_idx

        if not arrays_buffer:
            return True

        npz_name = _shard_filename(run_name, current_shard_idx, use_shards)
        npz_path = run_paths.root / npz_name
        shard_count = len(arrays_buffer)

        try:
            x = np.stack(arrays_buffer, axis=0)
            y = np.asarray(labels_buffer, dtype=np.float32)
            sample_id_arr = np.asarray(sample_ids_buffer, dtype=str)
            image_filename_arr = np.asarray(image_filenames_buffer, dtype=str)
            npz_row_index_arr = np.arange(shard_count, dtype=np.int64)

            payload = {
                "X": x,
                "y": y,
                "sample_id": sample_id_arr,
                "image_filename": image_filename_arr,
                "npz_row_index": npz_row_index_arr,
            }

            if stage_config.include_optional_filename_arrays:
                for column, values in optional_values_buffer.items():
                    payload[column] = np.asarray(values, dtype=str)

            if not stage_config.dry_run:
                save_fn = np.savez_compressed if stage_config.compress else np.savez
                save_fn(npz_path, **payload)
                validate_npz_file(npz_path, allowed_x_dtypes={np.dtype(x.dtype)})

            written_npz_paths.append(npz_path)

            for pos, row_idx in enumerate(row_indices_buffer):
                samples_df.at[row_idx, "npz_filename"] = npz_name
                samples_df.at[row_idx, "npz_row_index"] = int(pos)
                samples_df.at[row_idx, "pack_stage_status"] = "success"
                samples_df.at[row_idx, "pack_stage_error"] = ""

            if stage_config.delete_source_npy_after_pack and not stage_config.dry_run:
                for npy_path in npy_paths_buffer:
                    try:
                        npy_path.unlink()
                    except FileNotFoundError:
                        continue
                    except Exception as exc:
                        logger.log(f"Could not delete source npy '{npy_path}': {exc}")

            logger.log(
                f"Wrote NPZ shard '{npz_name}' rows={shard_count}, dtype={x.dtype}, "
                f"compression={'on' if stage_config.compress else 'off'}"
            )

            current_shard_idx += 1
            return True
        except Exception as exc:
            for row_idx in row_indices_buffer:
                samples_df.at[row_idx, "pack_stage_status"] = "failed"
                samples_df.at[row_idx, "pack_stage_error"] = f"NPZ shard write/validate failed: {exc}"
                samples_df.at[row_idx, "npz_filename"] = ""
                samples_df.at[row_idx, "npz_row_index"] = pd.NA
            logger.log(f"Pack stage failed while writing shard '{npz_name}': {exc}")
            return False
        finally:
            _reset_buffers()

    for row_idx in samples_df.index:
        try:
            npy_status = str(samples_df.at[row_idx, "npy_stage_status"]).strip().lower()

            samples_df.at[row_idx, "npz_filename"] = ""
            samples_df.at[row_idx, "npz_row_index"] = pd.NA

            if npy_status != "success":
                samples_df.at[row_idx, "pack_stage_status"] = "skipped"
                samples_df.at[row_idx, "pack_stage_error"] = "npy_stage_status is not success"
                continue

            if overwrite_blocked:
                samples_df.at[row_idx, "pack_stage_status"] = "skipped"
                samples_df.at[row_idx, "pack_stage_error"] = "npz exists and overwrite is false"
                continue

            npy_filename = samples_df.at[row_idx, "npy_filename"]

            try:
                npy_rel = normalize_relative_filename(npy_filename, new_suffix=".npy")
                npy_path = resolve_manifest_path(run_paths.root, "arrays", npy_rel)

                if not npy_path.is_file():
                    raise FileNotFoundError(f"Missing npy file: {npy_path}")

                array = np.load(npy_path, allow_pickle=False)
                expected_shape = validate_pack_array(array, expected_shape)
                array = _coerce_pack_array_dtype(array, output_dtype)

                label = np.float32(samples_df.at[row_idx, "distance_m"])
                if np.isnan(label) or np.isinf(label):
                    raise ValueError("distance_m is NaN or infinite")

                arrays_buffer.append(array)
                labels_buffer.append(label)
                sample_ids_buffer.append(str(samples_df.at[row_idx, "sample_id"]))
                image_filenames_buffer.append(str(samples_df.at[row_idx, "image_filename"]))
                row_indices_buffer.append(int(row_idx))
                npy_paths_buffer.append(npy_path)

                for column, values in optional_values_buffer.items():
                    values.append(str(samples_df.at[row_idx, column]))

                samples_df.at[row_idx, "pack_stage_status"] = "pending"
                samples_df.at[row_idx, "pack_stage_error"] = ""

                if shard_size > 0 and len(arrays_buffer) >= shard_size:
                    shard_ok = _flush_shard()
                    if not shard_ok and not stage_config.continue_on_error:
                        aborted = True
                        break
            except Exception as exc:
                samples_df.at[row_idx, "pack_stage_status"] = "failed"
                samples_df.at[row_idx, "pack_stage_error"] = str(exc)
                logger.log(f"Row {row_idx} failed: {exc}")
                if not stage_config.continue_on_error:
                    aborted = True
                    break
        finally:
            processed_rows += 1
            if processed_rows % progress_step == 0 or processed_rows == total_rows:
                percent = (100.0 * processed_rows / total_rows) if total_rows else 100.0
                logger.log(f"Progress: {processed_rows}/{total_rows} processed ({percent:.1f}%)")

    if not overwrite_blocked and arrays_buffer:
        shard_ok = _flush_shard()
        if not shard_ok and not stage_config.continue_on_error:
            aborted = True

    output_samples_path = samples_csv_path(run_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=stage_config.dry_run)

    status_series = samples_df["pack_stage_status"].fillna("")
    successful_rows = int((status_series == "success").sum())
    failed_rows = int((status_series == "failed").sum())
    skipped_rows = int((status_series == "skipped").sum())

    if written_npz_paths:
        summary_output = written_npz_paths[0] if len(written_npz_paths) == 1 else run_paths.root
    elif use_shards:
        summary_output = run_paths.root / f"{run_name}_shard_*.npz"
    else:
        summary_output = run_paths.root / f"{run_name}.npz"

    logger.log_summary(
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=summary_output,
    )
    logger.write()

    if aborted:
        raise RuntimeError("Pack stage stopped after failure (continue_on_error=False).")

    return StageSummary(
        run_name=run_name,
        stage_name="pack",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(summary_output),
        log_path=str(log_path),
        dry_run=stage_config.dry_run,
    )
