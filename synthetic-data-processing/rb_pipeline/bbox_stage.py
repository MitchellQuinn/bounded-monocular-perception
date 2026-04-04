"""Stage 2: edge PNG -> full-frame bbox PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import BBoxStageConfig, StageSummary
from .image_io import read_grayscale_uint8, write_grayscale_png
from .logging_utils import StageLogger
from .manifest import (
    BBOX_STAGE_COLUMNS,
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
    bbox_run_paths,
    edge_run_paths,
    ensure_run_dirs,
    normalize_relative_filename,
    resolve_manifest_path,
    to_posix_path,
)
from .validation import PipelineValidationError, validate_required_columns, validate_run_structure



def run_bbox_stage(
    project_root: Path,
    run_name: str,
    config: BBoxStageConfig | None = None,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummary:
    """Run bbox image generation for one run."""

    stage_config = config or BBoxStageConfig()

    source_paths = edge_run_paths(project_root, run_name)
    output_paths = bbox_run_paths(project_root, run_name)

    validation_errors = validate_run_structure(source_paths, require_images=True)
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    source_samples_path = samples_csv_path(source_paths.manifests_dir)
    samples_df = load_samples_csv(source_samples_path)

    required_columns = UNITY_REQUIRED_COLUMNS + EDGE_STAGE_COLUMNS
    validation_errors.extend(validate_required_columns(samples_df, required_columns))
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    append_columns(samples_df, BBOX_STAGE_COLUMNS)

    ensure_run_dirs(output_paths, dry_run=stage_config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=stage_config.dry_run)
    upsert_preprocessing_contract(
        output_paths.manifests_dir,
        stage_name="bbox",
        stage_parameters={
            "ForegroundThreshold": int(stage_config.foreground_threshold),
            "LineThicknessPx": int(stage_config.line_thickness),
            "LineThicknessPxUsed": max(1, int(stage_config.line_thickness)),
            "PaddingPx": int(stage_config.padding_px),
            "PaddingPxUsed": max(0, int(stage_config.padding_px)),
            "PostDrawBlur": bool(stage_config.post_draw_blur),
            "PostDrawBlurKernelSize": int(stage_config.post_draw_blur_kernel_size),
            "PostDrawBlurKernelSizeUsed": int(stage_config.normalized_blur_kernel_size()),
        },
        current_representation={
            "Kind": "full_frame_bbox_png",
            "StorageFormat": "png",
            "ColorSpace": "grayscale",
            "ForegroundEncoding": "black_on_white",
            "Geometry": "full_frame_bbox_outline",
        },
        dry_run=stage_config.dry_run,
    )

    log_path = output_paths.manifests_dir / "bbox_stage_log.txt"
    logger = StageLogger(
        stage_name="bbox",
        run_name=run_name,
        log_path=log_path,
        dry_run=stage_config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running bbox stage for run '{run_name}'")
    logger.log_parameters(
        stage_config.to_log_dict()
        | {
            "line_thickness_used": max(1, int(stage_config.line_thickness)),
            "padding_px_used": max(0, int(stage_config.padding_px)),
            "post_draw_blur_kernel_size_used": stage_config.normalized_blur_kernel_size(),
        }
    )

    thickness = max(1, int(stage_config.line_thickness))
    padding = max(0, int(stage_config.padding_px))

    aborted = False

    for row_idx in samples_df.index:
        edge_status = str(samples_df.at[row_idx, "edge_stage_status"]).strip().lower()
        edge_filename = samples_df.at[row_idx, "edge_image_filename"]

        try:
            output_rel = normalize_relative_filename(edge_filename, new_suffix=".png")
        except Exception as exc:
            samples_df.at[row_idx, "bbox_image_filename"] = ""
            samples_df.at[row_idx, "bbox_stage_status"] = "failed"
            samples_df.at[row_idx, "bbox_stage_error"] = f"Invalid edge filename: {exc}"
            logger.log(f"Row {row_idx} failed: invalid edge filename {edge_filename}")
            if not stage_config.continue_on_error:
                aborted = True
                break
            continue

        samples_df.at[row_idx, "bbox_image_filename"] = to_posix_path(output_rel)

        if edge_status != "success":
            samples_df.at[row_idx, "bbox_stage_status"] = "skipped"
            samples_df.at[row_idx, "bbox_stage_error"] = "edge_stage_status is not success"
            continue

        edge_path = resolve_manifest_path(source_paths.root, "images", edge_filename)
        output_path = output_paths.images_dir / output_rel

        if output_path.exists() and not stage_config.overwrite:
            samples_df.at[row_idx, "bbox_stage_status"] = "skipped"
            samples_df.at[row_idx, "bbox_stage_error"] = "output exists and overwrite is false"
            continue

        try:
            edge_gray = read_grayscale_uint8(edge_path)
            foreground_mask = edge_gray < int(stage_config.foreground_threshold)

            if not np.any(foreground_mask):
                raise ValueError("No foreground pixels found in edge image")

            ys, xs = np.where(foreground_mask)

            x1 = max(0, int(xs.min()) - padding)
            y1 = max(0, int(ys.min()) - padding)
            x2 = min(edge_gray.shape[1] - 1, int(xs.max()) + padding)
            y2 = min(edge_gray.shape[0] - 1, int(ys.max()) + padding)

            canvas = np.full(edge_gray.shape, 255, dtype=np.uint8)
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color=0,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            if stage_config.post_draw_blur:
                kernel = stage_config.normalized_blur_kernel_size()
                canvas = cv2.GaussianBlur(canvas, (kernel, kernel), 0)

            write_grayscale_png(output_path, canvas, dry_run=stage_config.dry_run)

            samples_df.at[row_idx, "bbox_stage_status"] = "success"
            samples_df.at[row_idx, "bbox_stage_error"] = ""
        except Exception as exc:
            samples_df.at[row_idx, "bbox_stage_status"] = "failed"
            samples_df.at[row_idx, "bbox_stage_error"] = str(exc)
            logger.log(f"Row {row_idx} failed: {exc}")
            if not stage_config.continue_on_error:
                aborted = True
                break

    output_samples_path = samples_csv_path(output_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=stage_config.dry_run)

    status_series = samples_df["bbox_stage_status"].fillna("")
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
        raise RuntimeError("BBox stage stopped after first row failure (continue_on_error=False).")

    return StageSummary(
        run_name=run_name,
        stage_name="bbox",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(output_paths.root),
        log_path=str(log_path),
        dry_run=stage_config.dry_run,
    )
