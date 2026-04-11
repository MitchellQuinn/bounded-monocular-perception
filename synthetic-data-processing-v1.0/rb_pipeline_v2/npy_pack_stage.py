"""Stage 2 (v2): silhouette PNG -> NPY -> NPZ shards."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from rb_pipeline.image_io import read_grayscale_uint8
from rb_pipeline.logging_utils import StageLogger
from rb_pipeline.validation import (
    PipelineValidationError,
    validate_npz_file,
    validate_pack_array,
    validate_required_columns,
    validate_run_structure,
)

from .algorithms import register_default_components
from .config import NpyPackStageConfigV2, StageSummaryV2
from .manifest import (
    NPY_STAGE_COLUMNS,
    PACK_STAGE_COLUMNS,
    SILHOUETTE_STAGE_COLUMNS,
    UNITY_REQUIRED_COLUMNS,
    append_columns,
    copy_run_json,
    load_run_json,
    load_samples_csv,
    samples_csv_path,
    upsert_preprocessing_contract_v2,
    write_samples_csv,
)
from .paths import (
    ensure_run_dirs_v2,
    normalize_relative_filename,
    resolve_manifest_path,
    silhouette_run_paths,
    to_posix_path,
    training_v2_run_paths,
)
from .registry import get_array_exporter


_VALID_PACK_OUTPUT_DTYPES = {"preserve", "float32", "float16", "uint8"}


def run_npy_pack_stage_v2(
    project_root: Path,
    run_name: str,
    config: NpyPackStageConfigV2,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV2:
    """Run interleaved npy+pack for v2 silhouette outputs."""

    register_default_components()

    mode = config.normalized_representation_mode()
    exporter_id = config.normalized_array_exporter_id()
    npy_output_dtype = config.normalized_npy_output_dtype()
    pack_output_dtype = config.normalized_pack_output_dtype()
    shard_size = config.normalized_shard_size()
    source_image_column = config.normalized_training_image_source_column()
    use_shards = shard_size > 0

    exporter = get_array_exporter(exporter_id)

    source_paths = silhouette_run_paths(project_root, run_name)
    output_paths = training_v2_run_paths(project_root, run_name)

    validation_errors = validate_run_structure(source_paths, require_images=True)
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    source_samples_path = samples_csv_path(source_paths.manifests_dir)
    samples_df = load_samples_csv(source_samples_path)

    required_columns = UNITY_REQUIRED_COLUMNS + SILHOUETTE_STAGE_COLUMNS
    if source_image_column not in required_columns:
        required_columns = required_columns + [source_image_column]
    validation_errors.extend(validate_required_columns(samples_df, required_columns))
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    append_columns(samples_df, NPY_STAGE_COLUMNS)
    append_columns(samples_df, PACK_STAGE_COLUMNS)

    samples_df["npz_filename"] = samples_df["npz_filename"].astype("string")
    samples_df["npz_row_index"] = pd.to_numeric(samples_df["npz_row_index"], errors="coerce").astype("Int64")
    samples_df["pack_stage_status"] = samples_df["pack_stage_status"].astype("string")
    samples_df["pack_stage_error"] = samples_df["pack_stage_error"].astype("string")

    ensure_run_dirs_v2(output_paths, dry_run=config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=config.dry_run)

    run_payload = load_run_json(output_paths.manifests_dir)
    prior_contract = run_payload.get("PreprocessingContract")
    prior_representation = (
        prior_contract.get("CurrentRepresentation", {}) if isinstance(prior_contract, dict) else {}
    )

    upsert_preprocessing_contract_v2(
        output_paths.manifests_dir,
        stage_name="npy",
        stage_parameters={
            "RepresentationMode": mode,
            "ArrayExporterId": exporter_id,
            "SourceImageColumn": source_image_column,
            "Normalize": bool(config.normalize),
            "Invert": bool(config.invert),
            "OutputDType": npy_output_dtype,
        },
        current_representation={
            "Kind": _representation_kind_for_source(source_image_column, mode),
            "RepresentationMode": mode,
            "SourceImageColumn": source_image_column,
            "StorageFormat": "npy",
            "ColorSpace": "grayscale",
            "Geometry": "full_frame_mask",
            "ArrayLayout": "H,W",
            "ArrayDType": npy_output_dtype,
            "Normalize": bool(config.normalize),
            "Invert": bool(config.invert),
        },
        dry_run=config.dry_run,
    )

    effective_pack_array_dtype = (
        pack_output_dtype if pack_output_dtype != "preserve" else npy_output_dtype
    )
    upsert_preprocessing_contract_v2(
        output_paths.manifests_dir,
        stage_name="pack",
        stage_parameters={
            "RepresentationMode": mode,
            "SourceImageColumn": source_image_column,
            "OutputDType": pack_output_dtype,
            "EffectiveArrayDType": effective_pack_array_dtype,
            "Compress": bool(config.compress),
            "ShardSize": int(shard_size),
        },
        current_representation={
            "Kind": _representation_kind_for_source(source_image_column, mode),
            "RepresentationMode": mode,
            "SourceImageColumn": source_image_column,
            "StorageFormat": "npz",
            "ArrayKey": "X",
            "ColorSpace": "grayscale",
            "Geometry": "full_frame_mask",
            "ArrayLayout": "N,H,W",
            "ArrayDType": effective_pack_array_dtype,
            "Normalize": bool(prior_representation.get("Normalize", config.normalize)),
            "Invert": bool(prior_representation.get("Invert", config.invert)),
        },
        dry_run=config.dry_run,
    )

    npy_log_path = output_paths.manifests_dir / "npy_stage_log.txt"
    pack_log_path = output_paths.manifests_dir / "pack_stage_log.txt"
    logger = StageLogger(
        stage_name="npy",
        run_name=run_name,
        log_path=npy_log_path,
        dry_run=config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running v2 npy+pack stage for run '{run_name}'")
    logger.log_parameters(
        config.to_log_dict()
        | {
            "representation_mode_used": mode,
            "array_exporter_id_used": exporter_id,
            "npy_output_dtype_used": npy_output_dtype,
            "pack_output_dtype_used": pack_output_dtype,
            "source_image_column_used": source_image_column,
            "pack_npz_pattern": (
                f"{run_name}_shard_00000.npz, {run_name}_shard_00001.npz, ..."
                if use_shards
                else f"{run_name}.npz"
            ),
        }
    )

    existing_npz_paths = _existing_npz_paths(output_paths.root, run_name)
    overwrite_blocked = bool(existing_npz_paths) and not config.overwrite
    if overwrite_blocked:
        existing_preview = ", ".join(str(path.name) for path in existing_npz_paths[:4])
        more_suffix = " ..." if len(existing_npz_paths) > 4 else ""
        logger.log(f"NPZ already exists and overwrite is false: {existing_preview}{more_suffix}")
    elif existing_npz_paths and config.overwrite and not config.dry_run:
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
    optional_filename_columns: list[str] = []
    for column in [
        "silhouette_image_filename",
        "silhouette_edge_debug_filename",
        source_image_column,
        "npy_filename",
        "npy_source_image_filename",
    ]:
        if column in samples_df.columns and column not in optional_filename_columns:
            optional_filename_columns.append(column)

    optional_values_buffer: dict[str, list[str]] = {
        column: [] for column in optional_filename_columns
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

            if config.include_optional_filename_arrays:
                for column, values in optional_values_buffer.items():
                    payload[column] = np.asarray(values, dtype=str)

            if not config.dry_run:
                save_fn = np.savez_compressed if config.compress else np.savez
                save_fn(npz_path, **payload)
                validate_npz_file(npz_path, allowed_x_dtypes={np.dtype(x.dtype)})

            written_npz_paths.append(npz_path)

            for pos, buffered_row_idx in enumerate(row_indices_buffer):
                samples_df.at[buffered_row_idx, "npz_filename"] = npz_name
                samples_df.at[buffered_row_idx, "npz_row_index"] = int(pos)
                samples_df.at[buffered_row_idx, "pack_stage_status"] = "success"
                samples_df.at[buffered_row_idx, "pack_stage_error"] = ""

            if config.delete_source_npy_after_pack and not config.dry_run:
                for npy_path in npy_paths_buffer:
                    try:
                        npy_path.unlink()
                    except FileNotFoundError:
                        continue
                    except Exception as exc:
                        logger.log(f"Could not delete source npy '{npy_path}': {exc}")

            logger.log(
                f"Wrote NPZ shard '{npz_name}' rows={shard_count}, dtype={x.dtype}, "
                f"compression={'on' if config.compress else 'off'}"
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
            samples_df.at[row_idx, "npy_source_image_column"] = source_image_column
            samples_df.at[row_idx, "npy_source_image_filename"] = ""
            samples_df.at[row_idx, "npz_filename"] = ""
            samples_df.at[row_idx, "npz_row_index"] = pd.NA
            samples_df.at[row_idx, "pack_stage_status"] = "skipped"
            samples_df.at[row_idx, "pack_stage_error"] = "npy_stage_status is not success"

            silhouette_status = str(samples_df.at[row_idx, "silhouette_stage_status"]).strip().lower()
            silhouette_mode = str(samples_df.at[row_idx, "silhouette_mode"]).strip().lower()

            if silhouette_status != "success":
                samples_df.at[row_idx, "npy_filename"] = ""
                samples_df.at[row_idx, "npy_stage_status"] = "skipped"
                samples_df.at[row_idx, "npy_stage_error"] = "silhouette_stage_status is not success"
                samples_df.at[row_idx, "pack_stage_status"] = "skipped"
                samples_df.at[row_idx, "pack_stage_error"] = "silhouette_stage_status is not success"
                continue

            if silhouette_mode and silhouette_mode != mode:
                samples_df.at[row_idx, "npy_filename"] = ""
                samples_df.at[row_idx, "npy_stage_status"] = "skipped"
                samples_df.at[row_idx, "npy_stage_error"] = (
                    f"silhouette_mode '{silhouette_mode}' does not match configured mode '{mode}'"
                )
                samples_df.at[row_idx, "pack_stage_status"] = "skipped"
                samples_df.at[row_idx, "pack_stage_error"] = "silhouette mode mismatch"
                continue

            source_image_value = samples_df.at[row_idx, source_image_column]
            source_image_filename = "" if pd.isna(source_image_value) else str(source_image_value).strip()
            samples_df.at[row_idx, "npy_source_image_filename"] = source_image_filename

            if not source_image_filename:
                error_message = (
                    f"Source image column '{source_image_column}' is empty; "
                    "enable matching debug persistence or choose another source."
                )
                samples_df.at[row_idx, "npy_filename"] = ""
                samples_df.at[row_idx, "npy_stage_status"] = "failed"
                samples_df.at[row_idx, "npy_stage_error"] = error_message
                samples_df.at[row_idx, "pack_stage_status"] = "failed"
                samples_df.at[row_idx, "pack_stage_error"] = error_message
                logger.log(f"Row {row_idx} failed: {error_message}")
                if not config.continue_on_error:
                    aborted = True
                    break
                continue

            try:
                output_rel = normalize_relative_filename(source_image_filename, new_suffix=".npy")
            except Exception as exc:
                samples_df.at[row_idx, "npy_filename"] = ""
                samples_df.at[row_idx, "npy_stage_status"] = "failed"
                samples_df.at[row_idx, "npy_stage_error"] = f"Invalid source image filename: {exc}"
                samples_df.at[row_idx, "pack_stage_status"] = "failed"
                samples_df.at[row_idx, "pack_stage_error"] = f"Invalid source image filename: {exc}"
                logger.log(f"Row {row_idx} failed: invalid source image filename {source_image_filename}")
                if not config.continue_on_error:
                    aborted = True
                    break
                continue

            samples_df.at[row_idx, "npy_filename"] = to_posix_path(output_rel)

            source_image_path = resolve_manifest_path(source_paths.root, "images", source_image_filename)
            npy_path = output_paths.arrays_dir / output_rel

            if npy_path.exists() and not config.overwrite:
                samples_df.at[row_idx, "npy_stage_status"] = "skipped"
                samples_df.at[row_idx, "npy_stage_error"] = "output exists and overwrite is false"
                samples_df.at[row_idx, "pack_stage_status"] = "skipped"
                samples_df.at[row_idx, "pack_stage_error"] = "npy output exists and overwrite is false"
                continue

            try:
                source_gray = read_grayscale_uint8(source_image_path)
                training_array = exporter.export(
                    source_gray,
                    normalize=config.normalize,
                    invert=config.invert,
                    output_dtype=npy_output_dtype,
                )
                expected_shape = validate_pack_array(
                    training_array,
                    expected_shape,
                    allowed_dtypes={np.dtype(npy_output_dtype)},
                )

                if not config.dry_run:
                    npy_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(npy_path, training_array)

                samples_df.at[row_idx, "npy_stage_status"] = "success"
                samples_df.at[row_idx, "npy_stage_error"] = ""
            except Exception as exc:
                samples_df.at[row_idx, "npy_stage_status"] = "failed"
                samples_df.at[row_idx, "npy_stage_error"] = str(exc)
                samples_df.at[row_idx, "pack_stage_status"] = "failed"
                samples_df.at[row_idx, "pack_stage_error"] = f"npy generation failed: {exc}"
                logger.log(f"Row {row_idx} failed during npy generation: {exc}")
                if not config.continue_on_error:
                    aborted = True
                    break
                continue

            if overwrite_blocked:
                samples_df.at[row_idx, "pack_stage_status"] = "skipped"
                samples_df.at[row_idx, "pack_stage_error"] = "npz exists and overwrite is false"
                continue

            try:
                pack_array = _coerce_pack_array_dtype(training_array, output_dtype=pack_output_dtype)
                validate_pack_array(pack_array, expected_shape)

                arrays_buffer.append(pack_array)
                labels_buffer.append(np.float32(samples_df.at[row_idx, "distance_m"]))
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
                    if not shard_ok and not config.continue_on_error:
                        aborted = True
                        break
            except Exception as exc:
                samples_df.at[row_idx, "pack_stage_status"] = "failed"
                samples_df.at[row_idx, "pack_stage_error"] = str(exc)
                logger.log(f"Row {row_idx} failed during shard buffering: {exc}")
                if not config.continue_on_error:
                    aborted = True
                    break
        finally:
            processed_rows += 1
            if processed_rows % progress_step == 0 or processed_rows == total_rows:
                percent = (100.0 * processed_rows / total_rows) if total_rows else 100.0
                logger.log(f"Progress: {processed_rows}/{total_rows} processed ({percent:.1f}%)")

    if not overwrite_blocked and arrays_buffer:
        shard_ok = _flush_shard()
        if not shard_ok and not config.continue_on_error:
            aborted = True

    output_samples_path = samples_csv_path(output_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=config.dry_run)

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

    if not config.dry_run:
        pack_log_path.write_text("\n".join(logger.lines) + "\n", encoding="utf-8")

    if aborted:
        raise RuntimeError("Interleaved v2 npy+pack stage stopped after failure (continue_on_error=False).")

    return StageSummaryV2(
        run_name=run_name,
        stage_name="npy",
        total_rows=total_rows,
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(summary_output),
        log_path=str(npy_log_path),
        dry_run=config.dry_run,
    )



def _normalize_pack_output_dtype(output_dtype: str) -> str:
    dtype_name = str(output_dtype).strip().lower()
    if dtype_name not in _VALID_PACK_OUTPUT_DTYPES:
        allowed = ", ".join(sorted(_VALID_PACK_OUTPUT_DTYPES))
        raise ValueError(f"Unsupported pack output_dtype '{output_dtype}'. Allowed: {allowed}.")
    return dtype_name



def _existing_npz_paths(run_root: Path, run_name: str) -> list[Path]:
    candidates = [run_root / f"{run_name}.npz", *sorted(run_root.glob(f"{run_name}_shard_*.npz"))]
    return [path for path in candidates if path.is_file()]



def _shard_filename(run_name: str, shard_idx: int, use_shards: bool) -> str:
    if use_shards:
        return f"{run_name}_shard_{shard_idx:05d}.npz"
    return f"{run_name}.npz"



def _coerce_pack_array_dtype(array: np.ndarray, output_dtype: str) -> np.ndarray:
    dtype_name = _normalize_pack_output_dtype(output_dtype)

    if dtype_name == "preserve":
        return array

    if dtype_name == "float32":
        return array.astype(np.float32, copy=False)

    if dtype_name == "float16":
        return array.astype(np.float16, copy=False)

    if dtype_name == "uint8":
        if np.issubdtype(array.dtype, np.floating):
            if np.isnan(array).any() or np.isinf(array).any():
                raise ValueError("Array contains NaN or infinite values")
            min_value = float(np.min(array))
            max_value = float(np.max(array))
            if min_value >= -1e-6 and max_value <= 1.0 + 1e-6:
                scaled = array * 255.0
            else:
                scaled = array
            return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8)

        if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
            return np.clip(array, 0, 255).astype(np.uint8)

        raise ValueError(f"Cannot convert dtype {array.dtype} to uint8")

    raise ValueError(f"Unsupported pack output dtype: {output_dtype}")


def _representation_kind_for_source(source_image_column: str, mode: str) -> str:
    if source_image_column == "silhouette_image_filename":
        return f"silhouette_{mode}_array"
    stem = str(source_image_column).strip()
    if stem.endswith("_filename"):
        stem = stem[: -len("_filename")]
    stem = stem.replace(" ", "_")
    if not stem:
        stem = "source_image"
    return f"{stem}_array"
