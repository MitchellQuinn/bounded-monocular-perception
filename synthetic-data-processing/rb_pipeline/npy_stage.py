"""Stage 3: bbox PNG -> NPY array."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .config import NpyStageConfig, StageSummary
from .image_io import bbox_png_to_training_array, read_grayscale_uint8
from .logging_utils import StageLogger
from .manifest import (
    BBOX_STAGE_COLUMNS,
    EDGE_STAGE_COLUMNS,
    NPY_STAGE_COLUMNS,
    UNITY_REQUIRED_COLUMNS,
    append_columns,
    copy_run_json,
    load_samples_csv,
    samples_csv_path,
    upsert_preprocessing_contract,
    write_samples_csv,
)
from .paths import (
    bbox_run_paths,
    ensure_run_dirs,
    normalize_relative_filename,
    resolve_manifest_path,
    to_posix_path,
    training_run_paths,
)
from .validation import PipelineValidationError, validate_required_columns, validate_run_structure

_VALID_NPY_OUTPUT_DTYPES = {"float32", "float16", "uint8"}


def _normalize_output_dtype(output_dtype: str) -> str:
    dtype_name = str(output_dtype).strip().lower()
    if dtype_name not in _VALID_NPY_OUTPUT_DTYPES:
        allowed = ", ".join(sorted(_VALID_NPY_OUTPUT_DTYPES))
        raise ValueError(f"Unsupported output_dtype '{output_dtype}'. Allowed: {allowed}.")
    return dtype_name


def _coerce_training_array_dtype(array: np.ndarray, *, output_dtype: str, normalize: bool) -> np.ndarray:
    """Convert a float training array to configured output dtype."""

    if output_dtype == "float32":
        return array.astype(np.float32, copy=False)

    if output_dtype == "float16":
        return array.astype(np.float16, copy=False)

    if output_dtype == "uint8":
        if normalize:
            scaled = array * 255.0
        else:
            scaled = array
        quantized = np.clip(np.rint(scaled), 0.0, 255.0)
        return quantized.astype(np.uint8, copy=False)

    # Guardrail: keep explicit error even though normalize function already validates.
    raise ValueError(f"Unsupported output dtype: {output_dtype}")



def run_npy_stage(
    project_root: Path,
    run_name: str,
    config: NpyStageConfig | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummary:
    """Run NPY array generation for one run."""

    stage_config = config or NpyStageConfig()

    source_paths = bbox_run_paths(project_root, run_name)
    output_paths = training_run_paths(project_root, run_name)
    output_dtype = _normalize_output_dtype(stage_config.output_dtype)

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

    ensure_run_dirs(output_paths, dry_run=stage_config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=stage_config.dry_run)
    upsert_preprocessing_contract(
        output_paths.manifests_dir,
        stage_name="npy",
        stage_parameters={
            "Normalize": bool(stage_config.normalize),
            "Invert": bool(stage_config.invert),
            "OutputDType": output_dtype,
        },
        current_representation={
            "Kind": "full_frame_bbox_array",
            "StorageFormat": "npy",
            "ColorSpace": "grayscale",
            "Geometry": "full_frame_bbox_outline",
            "ArrayLayout": "H,W",
            "ArrayDType": output_dtype,
            "Normalize": bool(stage_config.normalize),
            "Invert": bool(stage_config.invert),
        },
        dry_run=stage_config.dry_run,
    )

    log_path = output_paths.manifests_dir / "npy_stage_log.txt"
    logger = StageLogger(
        stage_name="npy",
        run_name=run_name,
        log_path=log_path,
        dry_run=stage_config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running npy stage for run '{run_name}'")
    logger.log_parameters(stage_config.to_log_dict())

    total_rows = len(samples_df)
    progress_step = max(1, total_rows // 100) if total_rows > 0 else 1
    logger.log(f"Progress updates: every ~1% ({progress_step} rows)")
    processed_rows = 0

    aborted = False

    for row_idx in samples_df.index:
        try:
            bbox_status = str(samples_df.at[row_idx, "bbox_stage_status"]).strip().lower()
            bbox_filename = samples_df.at[row_idx, "bbox_image_filename"]

            try:
                output_rel = normalize_relative_filename(bbox_filename, new_suffix=".npy")
            except Exception as exc:
                samples_df.at[row_idx, "npy_filename"] = ""
                samples_df.at[row_idx, "npy_stage_status"] = "failed"
                samples_df.at[row_idx, "npy_stage_error"] = f"Invalid bbox filename: {exc}"
                logger.log(f"Row {row_idx} failed: invalid bbox filename {bbox_filename}")
                if not stage_config.continue_on_error:
                    aborted = True
                    break
                continue

            samples_df.at[row_idx, "npy_filename"] = to_posix_path(output_rel)

            if bbox_status != "success":
                samples_df.at[row_idx, "npy_stage_status"] = "skipped"
                samples_df.at[row_idx, "npy_stage_error"] = "bbox_stage_status is not success"
                continue

            bbox_path = resolve_manifest_path(source_paths.root, "images", bbox_filename)
            output_path = output_paths.arrays_dir / output_rel

            if output_path.exists() and not stage_config.overwrite:
                samples_df.at[row_idx, "npy_stage_status"] = "skipped"
                samples_df.at[row_idx, "npy_stage_error"] = "output exists and overwrite is false"
                continue

            try:
                bbox_gray = read_grayscale_uint8(bbox_path)
                training_array = bbox_png_to_training_array(
                    bbox_gray,
                    normalize=stage_config.normalize,
                    invert=stage_config.invert,
                )
                if np.isnan(training_array).any() or np.isinf(training_array).any():
                    raise ValueError("Generated array contains NaN or infinite values")
                training_array = _coerce_training_array_dtype(
                    training_array,
                    output_dtype=output_dtype,
                    normalize=stage_config.normalize,
                )
                if np.issubdtype(training_array.dtype, np.floating):
                    if np.isnan(training_array).any() or np.isinf(training_array).any():
                        raise ValueError("Converted array contains NaN or infinite values")

                if not stage_config.dry_run:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(output_path, training_array)

                samples_df.at[row_idx, "npy_stage_status"] = "success"
                samples_df.at[row_idx, "npy_stage_error"] = ""
            except Exception as exc:
                samples_df.at[row_idx, "npy_stage_status"] = "failed"
                samples_df.at[row_idx, "npy_stage_error"] = str(exc)
                logger.log(f"Row {row_idx} failed: {exc}")
                if not stage_config.continue_on_error:
                    aborted = True
                    break
        finally:
            processed_rows += 1
            if processed_rows % progress_step == 0 or processed_rows == total_rows:
                percent = (100.0 * processed_rows / total_rows) if total_rows else 100.0
                logger.log(f"Progress: {processed_rows}/{total_rows} processed ({percent:.1f}%)")

    output_samples_path = samples_csv_path(output_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=stage_config.dry_run)

    status_series = samples_df["npy_stage_status"].fillna("")
    successful_rows = int((status_series == "success").sum())
    failed_rows = int((status_series == "failed").sum())
    skipped_rows = int((status_series == "skipped").sum())

    logger.log_summary(
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=output_paths.root,
    )
    logger.write()

    if aborted:
        raise RuntimeError("NPY stage stopped after first row failure (continue_on_error=False).")

    return StageSummary(
        run_name=run_name,
        stage_name="npy",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(output_paths.root),
        log_path=str(log_path),
        dry_run=stage_config.dry_run,
    )
