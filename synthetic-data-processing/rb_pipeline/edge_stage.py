"""Stage 1: source PNG -> edge PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from .config import EdgeStageConfig, StageSummary
from .image_io import edge_image_black_on_white, read_grayscale_uint8, write_grayscale_png
from .logging_utils import StageLogger
from .manifest import (
    EDGE_STAGE_COLUMNS,
    UNITY_REQUIRED_COLUMNS,
    append_columns,
    copy_run_json,
    load_samples_csv,
    samples_csv_path,
    upsert_preprocessing_contract,
    write_samples_csv,
)
from .paths import (
    edge_run_paths,
    ensure_run_dirs,
    input_run_paths,
    normalize_relative_filename,
    resolve_manifest_path,
    to_posix_path,
)
from .validation import (
    PipelineValidationError,
    capture_success_mask,
    validate_capture_success_images,
    validate_required_columns,
    validate_run_structure,
)



def run_edge_stage(
    project_root: Path,
    run_name: str,
    config: EdgeStageConfig | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummary:
    """Run edge image generation for one input run."""

    stage_config = config or EdgeStageConfig()

    source_paths = input_run_paths(project_root, run_name)
    output_paths = edge_run_paths(project_root, run_name)

    validation_errors = validate_run_structure(source_paths, require_images=True)
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    source_samples_path = samples_csv_path(source_paths.manifests_dir)
    samples_df = load_samples_csv(source_samples_path)

    validation_errors.extend(validate_required_columns(samples_df, UNITY_REQUIRED_COLUMNS))
    validation_errors.extend(
        validate_capture_success_images(
            samples_df,
            source_paths.root,
            image_column="image_filename",
            default_subdir="images",
            capture_column="capture_success",
        )
    )
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    append_columns(samples_df, EDGE_STAGE_COLUMNS)
    capture_mask = capture_success_mask(samples_df)

    ensure_run_dirs(output_paths, dry_run=stage_config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=stage_config.dry_run)
    upsert_preprocessing_contract(
        output_paths.manifests_dir,
        stage_name="edge",
        stage_parameters={
            "BlurKernelSize": int(stage_config.blur_kernel_size),
            "BlurKernelSizeUsed": int(stage_config.normalized_blur_kernel_size()),
            "CannyLowThreshold": int(stage_config.canny_low_threshold),
            "CannyHighThreshold": int(stage_config.canny_high_threshold),
        },
        current_representation={
            "Kind": "edge_png",
            "StorageFormat": "png",
            "ColorSpace": "grayscale",
            "ForegroundEncoding": "black_on_white",
        },
        dry_run=stage_config.dry_run,
    )

    log_path = output_paths.manifests_dir / "edge_stage_log.txt"
    logger = StageLogger(
        stage_name="edge",
        run_name=run_name,
        log_path=log_path,
        dry_run=stage_config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running edge stage for run '{run_name}'")
    logger.log_parameters(stage_config.to_log_dict() | {"blur_kernel_size_used": stage_config.normalized_blur_kernel_size()})

    aborted = False

    for row_idx in samples_df.index:
        row_capture_success = bool(capture_mask.loc[row_idx])
        source_filename = samples_df.at[row_idx, "image_filename"]

        try:
            output_rel = normalize_relative_filename(source_filename, new_suffix=".png")
        except Exception as exc:
            samples_df.at[row_idx, "edge_image_filename"] = ""
            samples_df.at[row_idx, "edge_stage_status"] = "failed"
            samples_df.at[row_idx, "edge_stage_error"] = f"Invalid source filename: {exc}"
            logger.log(f"Row {row_idx} failed: invalid source filename {source_filename}")
            if not stage_config.continue_on_error:
                aborted = True
                break
            continue

        samples_df.at[row_idx, "edge_image_filename"] = to_posix_path(output_rel)

        if not row_capture_success:
            samples_df.at[row_idx, "edge_stage_status"] = "skipped"
            samples_df.at[row_idx, "edge_stage_error"] = "capture_success is false"
            continue

        source_path = resolve_manifest_path(source_paths.root, "images", source_filename)
        output_path = output_paths.images_dir / output_rel

        if output_path.exists() and not stage_config.overwrite:
            samples_df.at[row_idx, "edge_stage_status"] = "skipped"
            samples_df.at[row_idx, "edge_stage_error"] = "output exists and overwrite is false"
            continue

        try:
            source_gray = read_grayscale_uint8(source_path)
            output_edge = edge_image_black_on_white(
                source_gray,
                blur_kernel_size=stage_config.normalized_blur_kernel_size(),
                canny_low_threshold=stage_config.canny_low_threshold,
                canny_high_threshold=stage_config.canny_high_threshold,
            )
            write_grayscale_png(output_path, output_edge, dry_run=stage_config.dry_run)

            samples_df.at[row_idx, "edge_stage_status"] = "success"
            samples_df.at[row_idx, "edge_stage_error"] = ""
        except Exception as exc:
            samples_df.at[row_idx, "edge_stage_status"] = "failed"
            samples_df.at[row_idx, "edge_stage_error"] = str(exc)
            logger.log(f"Row {row_idx} failed: {exc}")
            if not stage_config.continue_on_error:
                aborted = True
                break

    output_samples_path = samples_csv_path(output_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=stage_config.dry_run)

    status_series = samples_df["edge_stage_status"].fillna("")
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
        raise RuntimeError("Edge stage stopped after first row failure (continue_on_error=False).")

    return StageSummary(
        run_name=run_name,
        stage_name="edge",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(output_paths.root),
        log_path=str(log_path),
        dry_run=stage_config.dry_run,
    )
