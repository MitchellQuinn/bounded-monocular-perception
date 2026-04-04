"""Combined stage: bbox PNG -> NPY -> NPZ shards with shard-wise NPY cleanup."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import NpyStageConfig, PackStageConfig, StageSummary
from .image_io import bbox_png_to_training_array, read_grayscale_uint8
from .logging_utils import StageLogger
from .manifest import (
    BBOX_STAGE_COLUMNS,
    EDGE_STAGE_COLUMNS,
    NPY_STAGE_COLUMNS,
    PACK_STAGE_COLUMNS,
    UNITY_REQUIRED_COLUMNS,
    append_columns,
    copy_run_json,
    load_samples_csv,
    samples_csv_path,
    write_samples_csv,
)
from .npy_stage import _coerce_training_array_dtype, _normalize_output_dtype
from .pack_stage import (
    _coerce_pack_array_dtype,
    _existing_npz_paths,
    _normalize_pack_output_dtype,
    _normalize_shard_size,
    _shard_filename,
)
from .paths import (
    bbox_run_paths,
    ensure_run_dirs,
    normalize_relative_filename,
    resolve_manifest_path,
    to_posix_path,
    training_run_paths,
)
from .validation import (
    PipelineValidationError,
    validate_npz_file,
    validate_pack_array,
    validate_required_columns,
    validate_run_structure,
)


def run_npy_pack_stage(
    project_root: Path,
    run_name: str,
    npy_config: NpyStageConfig | None = None,
    pack_config: PackStageConfig | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummary:
    """Run interleaved npy+pack with bounded disk usage (one shard at a time)."""

    npy_stage_config = npy_config or NpyStageConfig()
    pack_stage_config = pack_config or PackStageConfig()

    npy_output_dtype = _normalize_output_dtype(npy_stage_config.output_dtype)
    pack_output_dtype = _normalize_pack_output_dtype(pack_stage_config.output_dtype)
    shard_size = _normalize_shard_size(pack_stage_config.shard_size)
    use_shards = shard_size > 0

    dry_run = bool(npy_stage_config.dry_run or pack_stage_config.dry_run)
    continue_on_error = bool(npy_stage_config.continue_on_error and pack_stage_config.continue_on_error)

    source_paths = bbox_run_paths(project_root, run_name)
    output_paths = training_run_paths(project_root, run_name)

    validation_errors = validate_run_structure(source_paths, require_images=True)
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    source_samples_path = samples_csv_path(source_paths.manifests_dir)
    samples_df = load_samples_csv(source_samples_path)

    required_columns = UNITY_REQUIRED_COLUMNS + EDGE_STAGE_COLUMNS + BBOX_STAGE_COLUMNS
    validation_errors.extend(validate_required_columns(samples_df, required_columns))
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    append_columns(samples_df, NPY_STAGE_COLUMNS)
    append_columns(samples_df, PACK_STAGE_COLUMNS)

    samples_df["npz_filename"] = samples_df["npz_filename"].astype("string")
    samples_df["npz_row_index"] = pd.to_numeric(samples_df["npz_row_index"], errors="coerce").astype("Int64")
    samples_df["pack_stage_status"] = samples_df["pack_stage_status"].astype("string")
    samples_df["pack_stage_error"] = samples_df["pack_stage_error"].astype("string")

    ensure_run_dirs(output_paths, dry_run=dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=dry_run)

    npy_log_path = output_paths.manifests_dir / "npy_stage_log.txt"
    pack_log_path = output_paths.manifests_dir / "pack_stage_log.txt"
    logger = StageLogger(
        stage_name="npy",
        run_name=run_name,
        log_path=npy_log_path,
        dry_run=dry_run,
        sink=log_sink,
    )
    logger.log(f"Running interleaved npy+pack stage for run '{run_name}'")

    combined_params: dict[str, object] = {}
    for key, value in npy_stage_config.to_log_dict().items():
        combined_params[f"npy_{key}"] = value
    for key, value in pack_stage_config.to_log_dict().items():
        combined_params[f"pack_{key}"] = value
    combined_params["pack_npz_pattern"] = (
        f"{run_name}_shard_00000.npz, {run_name}_shard_00001.npz, ..."
        if use_shards
        else f"{run_name}.npz"
    )
    combined_params["effective_dry_run"] = dry_run
    combined_params["effective_continue_on_error"] = continue_on_error
    logger.log_parameters(combined_params)

    existing_npz_paths = _existing_npz_paths(output_paths.root, run_name)
    overwrite_blocked = bool(existing_npz_paths) and not pack_stage_config.overwrite
    if overwrite_blocked:
        existing_preview = ", ".join(str(path.name) for path in existing_npz_paths[:4])
        more_suffix = " ..." if len(existing_npz_paths) > 4 else ""
        logger.log(f"NPZ already exists and pack overwrite is false: {existing_preview}{more_suffix}")
    elif existing_npz_paths and pack_stage_config.overwrite and not dry_run:
        for existing in existing_npz_paths:
            existing.unlink()
            logger.log(f"Removed existing NPZ: {existing}")

    total_rows = len(samples_df)
    progress_step = max(1, total_rows // 100) if total_rows > 0 else 1
    logger.log(f"Progress updates: every ~1% ({progress_step} rows)")
    processed_rows = 0

    expected_shape: tuple[int, int] | None = None
    written_npz_paths: list[Path] = []
    current_shard_idx = 0

    arrays_buffer: list[np.ndarray] = []
    labels_buffer: list[np.float32] = []
    sample_ids_buffer: list[str] = []
    image_filenames_buffer: list[str] = []
    optional_values_buffer: dict[str, list[str]] = {
        column: [] for column in ["edge_image_filename", "bbox_image_filename", "npy_filename"] if column in samples_df.columns
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

    def _mark_buffer_rows_failed(error_message: str) -> None:
        for buffered_row_idx in row_indices_buffer:
            samples_df.at[buffered_row_idx, "pack_stage_status"] = "failed"
            samples_df.at[buffered_row_idx, "pack_stage_error"] = error_message
            samples_df.at[buffered_row_idx, "npz_filename"] = ""
            samples_df.at[buffered_row_idx, "npz_row_index"] = pd.NA

    def _flush_shard() -> bool:
        nonlocal current_shard_idx

        if not arrays_buffer:
            return True

        npz_name = _shard_filename(run_name, current_shard_idx, use_shards)
        npz_path = output_paths.root / npz_name
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

            if pack_stage_config.include_optional_filename_arrays:
                for column, values in optional_values_buffer.items():
                    payload[column] = np.asarray(values, dtype=str)

            if not dry_run:
                save_fn = np.savez_compressed if pack_stage_config.compress else np.savez
                save_fn(npz_path, **payload)
                validate_npz_file(npz_path, allowed_x_dtypes={np.dtype(x.dtype)})

            written_npz_paths.append(npz_path)

            for pos, buffered_row_idx in enumerate(row_indices_buffer):
                samples_df.at[buffered_row_idx, "npz_filename"] = npz_name
                samples_df.at[buffered_row_idx, "npz_row_index"] = int(pos)
                samples_df.at[buffered_row_idx, "pack_stage_status"] = "success"
                samples_df.at[buffered_row_idx, "pack_stage_error"] = ""

            if pack_stage_config.delete_source_npy_after_pack and not dry_run:
                for npy_path in npy_paths_buffer:
                    try:
                        npy_path.unlink()
                    except FileNotFoundError:
                        continue
                    except Exception as exc:
                        logger.log(f"Could not delete source npy '{npy_path}': {exc}")

            logger.log(
                f"Wrote NPZ shard '{npz_name}' rows={shard_count}, dtype={x.dtype}, "
                f"compression={'on' if pack_stage_config.compress else 'off'}"
            )
            current_shard_idx += 1
            return True
        except Exception as exc:
            _mark_buffer_rows_failed(f"NPZ shard write/validate failed: {exc}")
            logger.log(f"NPZ shard write/validate failed for '{npz_name}': {exc}")
            return False
        finally:
            _reset_buffers()

    for row_idx in samples_df.index:
        try:
            samples_df.at[row_idx, "npz_filename"] = ""
            samples_df.at[row_idx, "npz_row_index"] = pd.NA
            samples_df.at[row_idx, "pack_stage_status"] = "skipped"
            samples_df.at[row_idx, "pack_stage_error"] = "npy_stage_status is not success"

            bbox_status = str(samples_df.at[row_idx, "bbox_stage_status"]).strip().lower()
            bbox_filename = samples_df.at[row_idx, "bbox_image_filename"]

            try:
                output_rel = normalize_relative_filename(bbox_filename, new_suffix=".npy")
            except Exception as exc:
                samples_df.at[row_idx, "npy_filename"] = ""
                samples_df.at[row_idx, "npy_stage_status"] = "failed"
                samples_df.at[row_idx, "npy_stage_error"] = f"Invalid bbox filename: {exc}"
                samples_df.at[row_idx, "pack_stage_status"] = "failed"
                samples_df.at[row_idx, "pack_stage_error"] = f"Invalid bbox filename: {exc}"
                logger.log(f"Row {row_idx} failed: invalid bbox filename {bbox_filename}")
                if not continue_on_error:
                    aborted = True
                    break
                continue

            samples_df.at[row_idx, "npy_filename"] = to_posix_path(output_rel)

            if bbox_status != "success":
                samples_df.at[row_idx, "npy_stage_status"] = "skipped"
                samples_df.at[row_idx, "npy_stage_error"] = "bbox_stage_status is not success"
                samples_df.at[row_idx, "pack_stage_status"] = "skipped"
                samples_df.at[row_idx, "pack_stage_error"] = "bbox_stage_status is not success"
                continue

            bbox_path = resolve_manifest_path(source_paths.root, "images", bbox_filename)
            output_path = output_paths.arrays_dir / output_rel

            if output_path.exists() and not npy_stage_config.overwrite:
                samples_df.at[row_idx, "npy_stage_status"] = "skipped"
                samples_df.at[row_idx, "npy_stage_error"] = "output exists and overwrite is false"
                samples_df.at[row_idx, "pack_stage_status"] = "skipped"
                samples_df.at[row_idx, "pack_stage_error"] = "npy output exists and overwrite is false"
                continue

            try:
                bbox_gray = read_grayscale_uint8(bbox_path)
                training_array = bbox_png_to_training_array(
                    bbox_gray,
                    normalize=npy_stage_config.normalize,
                    invert=npy_stage_config.invert,
                )
                if np.isnan(training_array).any() or np.isinf(training_array).any():
                    raise ValueError("Generated array contains NaN or infinite values")
                training_array = _coerce_training_array_dtype(
                    training_array,
                    output_dtype=npy_output_dtype,
                    normalize=npy_stage_config.normalize,
                )
                if np.issubdtype(training_array.dtype, np.floating):
                    if np.isnan(training_array).any() or np.isinf(training_array).any():
                        raise ValueError("Converted array contains NaN or infinite values")

                if not dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_path, training_array)

                samples_df.at[row_idx, "npy_stage_status"] = "success"
                samples_df.at[row_idx, "npy_stage_error"] = ""
            except Exception as exc:
                samples_df.at[row_idx, "npy_stage_status"] = "failed"
                samples_df.at[row_idx, "npy_stage_error"] = str(exc)
                samples_df.at[row_idx, "pack_stage_status"] = "failed"
                samples_df.at[row_idx, "pack_stage_error"] = f"npy generation failed: {exc}"
                logger.log(f"Row {row_idx} failed during npy generation: {exc}")
                if not continue_on_error:
                    aborted = True
                    break
                continue

            if overwrite_blocked:
                samples_df.at[row_idx, "pack_stage_status"] = "skipped"
                samples_df.at[row_idx, "pack_stage_error"] = "npz exists and overwrite is false"
                continue

            try:
                pack_array = _coerce_pack_array_dtype(training_array, pack_output_dtype)
                expected_shape = validate_pack_array(pack_array, expected_shape)

                label = np.float32(samples_df.at[row_idx, "distance_m"])
                if np.isnan(label) or np.isinf(label):
                    raise ValueError("distance_m is NaN or infinite")

                arrays_buffer.append(pack_array)
                labels_buffer.append(label)
                sample_ids_buffer.append(str(samples_df.at[row_idx, "sample_id"]))
                image_filenames_buffer.append(str(samples_df.at[row_idx, "image_filename"]))
                row_indices_buffer.append(int(row_idx))
                npy_paths_buffer.append(output_path)

                for column, values in optional_values_buffer.items():
                    values.append(str(samples_df.at[row_idx, column]))

                samples_df.at[row_idx, "pack_stage_status"] = "pending"
                samples_df.at[row_idx, "pack_stage_error"] = ""

                if shard_size > 0 and len(arrays_buffer) >= shard_size:
                    shard_ok = _flush_shard()
                    if not shard_ok and not continue_on_error:
                        aborted = True
                        break
            except Exception as exc:
                samples_df.at[row_idx, "pack_stage_status"] = "failed"
                samples_df.at[row_idx, "pack_stage_error"] = str(exc)
                logger.log(f"Row {row_idx} failed during shard buffering: {exc}")
                if not continue_on_error:
                    aborted = True
                    break
        finally:
            processed_rows += 1
            if processed_rows % progress_step == 0 or processed_rows == total_rows:
                percent = (100.0 * processed_rows / total_rows) if total_rows else 100.0
                logger.log(f"Progress: {processed_rows}/{total_rows} processed ({percent:.1f}%)")

    if not overwrite_blocked and arrays_buffer:
        shard_ok = _flush_shard()
        if not shard_ok and not continue_on_error:
            aborted = True

    output_samples_path = samples_csv_path(output_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=dry_run)

    status_series = samples_df["pack_stage_status"].fillna("")
    successful_rows = int((status_series == "success").sum())
    failed_rows = int((status_series == "failed").sum())
    skipped_rows = int((status_series == "skipped").sum())

    if written_npz_paths:
        summary_output = written_npz_paths[0] if len(written_npz_paths) == 1 else output_paths.root
    elif use_shards:
        summary_output = output_paths.root / f"{run_name}_shard_*.npz"
    else:
        summary_output = output_paths.root / f"{run_name}.npz"

    logger.log_summary(
        total_rows=total_rows,
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=summary_output,
    )
    logger.write()

    # Keep a pack-stage log artifact for notebook/debug parity.
    if not dry_run:
        pack_log_path.write_text("\n".join(logger.lines) + "\n", encoding="utf-8")

    if aborted:
        raise RuntimeError("Interleaved npy+pack stage stopped after failure (continue_on_error=False).")

    return StageSummary(
        run_name=run_name,
        stage_name="npy",
        total_rows=total_rows,
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(summary_output),
        log_path=str(npy_log_path),
        dry_run=dry_run,
    )
