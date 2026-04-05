"""Stage 1 (v2): source PNG -> silhouette PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from rb_pipeline.image_io import read_grayscale_uint8, write_grayscale_png
from rb_pipeline.logging_utils import StageLogger
from rb_pipeline.validation import (
    PipelineValidationError,
    capture_success_mask,
    validate_capture_success_images,
    validate_required_columns,
    validate_run_structure,
)

from .algorithms import register_default_components
from .config import SilhouetteStageConfigV2, StageSummaryV2
from .manifest import (
    SILHOUETTE_STAGE_COLUMNS,
    UNITY_REQUIRED_COLUMNS,
    append_columns,
    copy_run_json,
    load_samples_csv,
    samples_csv_path,
    upsert_preprocessing_contract_v2,
    write_samples_csv,
)
from .paths import (
    ensure_run_dirs_v2,
    input_run_paths,
    normalize_relative_filename,
    resolve_manifest_path,
    silhouette_run_paths,
    to_posix_path,
)
from .registry import (
    get_artifact_writer_by_mode,
    get_fallback_strategy,
    get_representation_generator,
)



def run_silhouette_stage_v2(
    project_root: Path,
    run_name: str,
    config: SilhouetteStageConfigV2,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV2:
    """Run v2 silhouette generation for one input run."""

    register_default_components()

    mode = config.normalized_representation_mode()
    generator_id = config.normalized_generator_id()
    fallback_id = config.normalized_fallback_id()

    generator = get_representation_generator(generator_id)
    fallback = get_fallback_strategy(fallback_id)
    writer = get_artifact_writer_by_mode(mode)

    source_paths = input_run_paths(project_root, run_name)
    output_paths = silhouette_run_paths(project_root, run_name)

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

    append_columns(samples_df, SILHOUETTE_STAGE_COLUMNS)
    capture_mask = capture_success_mask(samples_df)

    selected_rows = _selected_row_indices(
        samples_df,
        offset=config.normalized_sample_offset(),
        limit=config.normalized_sample_limit(),
    )

    ensure_run_dirs_v2(output_paths, dry_run=config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=config.dry_run)
    upsert_preprocessing_contract_v2(
        output_paths.manifests_dir,
        stage_name="silhouette",
        stage_parameters={
            "GeneratorId": generator_id,
            "FallbackId": fallback_id,
            "RepresentationMode": mode,
            "BlurKernelSize": int(config.blur_kernel_size),
            "BlurKernelSizeUsed": int(config.normalized_blur_kernel_size()),
            "CannyLowThreshold": int(config.canny_low_threshold),
            "CannyHighThreshold": int(config.canny_high_threshold),
            "CloseKernelSize": int(config.close_kernel_size),
            "CloseKernelSizeUsed": int(config.normalized_close_kernel_size()),
            "DilateKernelSize": int(config.dilate_kernel_size),
            "DilateKernelSizeUsed": int(config.normalized_dilate_kernel_size()),
            "MinComponentAreaPx": int(config.min_component_area_px),
            "MinComponentAreaPxUsed": int(config.normalized_min_component_area_px()),
            "OutlineThicknessPx": int(config.outline_thickness),
            "OutlineThicknessPxUsed": int(config.normalized_outline_thickness()),
            "PersistEdgeDebug": bool(config.persist_edge_debug),
            "SampleOffset": int(config.normalized_sample_offset()),
            "SampleLimit": int(config.normalized_sample_limit()),
        },
        current_representation={
            "Kind": f"silhouette_{mode}_png",
            "RepresentationMode": mode,
            "StorageFormat": "png",
            "ColorSpace": "grayscale",
            "ForegroundEncoding": "black_on_white",
            "Geometry": f"full_frame_silhouette_{mode}",
        },
        dry_run=config.dry_run,
    )

    log_path = output_paths.manifests_dir / "silhouette_stage_log.txt"
    logger = StageLogger(
        stage_name="silhouette",
        run_name=run_name,
        log_path=log_path,
        dry_run=config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running v2 silhouette stage for run '{run_name}'")
    logger.log_parameters(
        config.to_log_dict()
        | {
            "representation_mode_used": mode,
            "generator_id_used": generator_id,
            "fallback_id_used": fallback_id,
            "blur_kernel_size_used": config.normalized_blur_kernel_size(),
            "close_kernel_size_used": config.normalized_close_kernel_size(),
            "dilate_kernel_size_used": config.normalized_dilate_kernel_size(),
            "outline_thickness_used": config.normalized_outline_thickness(),
            "min_component_area_px_used": config.normalized_min_component_area_px(),
        }
    )

    aborted = False

    for row_idx in samples_df.index:
        samples_df.at[row_idx, "silhouette_mode"] = mode
        samples_df.at[row_idx, "silhouette_fallback_used"] = "false"
        samples_df.at[row_idx, "silhouette_fallback_reason"] = ""
        samples_df.at[row_idx, "silhouette_area_px"] = ""
        samples_df.at[row_idx, "silhouette_bbox_x1"] = ""
        samples_df.at[row_idx, "silhouette_bbox_y1"] = ""
        samples_df.at[row_idx, "silhouette_bbox_x2"] = ""
        samples_df.at[row_idx, "silhouette_bbox_y2"] = ""

        row_capture_success = bool(capture_mask.loc[row_idx])
        source_filename = samples_df.at[row_idx, "image_filename"]

        try:
            silhouette_rel = normalize_relative_filename(source_filename, new_suffix=".png")
        except Exception as exc:
            samples_df.at[row_idx, "silhouette_image_filename"] = ""
            samples_df.at[row_idx, "silhouette_edge_debug_filename"] = ""
            samples_df.at[row_idx, "silhouette_stage_status"] = "failed"
            samples_df.at[row_idx, "silhouette_stage_error"] = f"Invalid source filename: {exc}"
            logger.log(f"Row {row_idx} failed: invalid source filename {source_filename}")
            if not config.continue_on_error:
                aborted = True
                break
            continue

        edge_debug_rel = silhouette_rel.with_name(f"{silhouette_rel.stem}.edge.png")

        samples_df.at[row_idx, "silhouette_image_filename"] = to_posix_path(silhouette_rel)
        samples_df.at[row_idx, "silhouette_edge_debug_filename"] = (
            to_posix_path(edge_debug_rel) if config.persist_edge_debug else ""
        )

        if row_idx not in selected_rows:
            samples_df.at[row_idx, "silhouette_stage_status"] = "skipped"
            samples_df.at[row_idx, "silhouette_stage_error"] = "outside selected subset"
            continue

        if not row_capture_success:
            samples_df.at[row_idx, "silhouette_stage_status"] = "skipped"
            samples_df.at[row_idx, "silhouette_stage_error"] = "capture_success is false"
            continue

        source_path = resolve_manifest_path(source_paths.root, "images", source_filename)
        silhouette_path = output_paths.images_dir / silhouette_rel
        edge_debug_path = output_paths.images_dir / edge_debug_rel

        if silhouette_path.exists() and not config.overwrite:
            samples_df.at[row_idx, "silhouette_stage_status"] = "skipped"
            samples_df.at[row_idx, "silhouette_stage_error"] = "output exists and overwrite is false"
            continue

        try:
            source_gray = read_grayscale_uint8(source_path)

            generated = generator.generate(
                source_gray,
                blur_kernel_size=config.normalized_blur_kernel_size(),
                canny_low_threshold=config.canny_low_threshold,
                canny_high_threshold=config.canny_high_threshold,
                close_kernel_size=config.normalized_close_kernel_size(),
                dilate_kernel_size=config.normalized_dilate_kernel_size(),
                min_component_area_px=config.normalized_min_component_area_px(),
            )

            contour = generated.contour
            fallback_used = False
            fallback_reason = ""

            primary_break_reason = _contour_break_reason(contour)
            if primary_break_reason:
                contour, recovery_reason = fallback.recover(generated.fallback_mask)
                fallback_used = True
                fallback_reason = f"primary_{generated.primary_reason or primary_break_reason}"
                if contour is None:
                    raise ValueError(f"Fallback failed: {recovery_reason}")

            silhouette_img = writer.render(
                source_gray.shape,
                contour,
                line_thickness=config.normalized_outline_thickness(),
            )

            if _render_is_empty(silhouette_img):
                if not fallback_used:
                    contour, recovery_reason = fallback.recover(generated.fallback_mask)
                    fallback_used = True
                    fallback_reason = "primary_render_empty"
                    if contour is None:
                        raise ValueError(f"Fallback failed: {recovery_reason}")
                    silhouette_img = writer.render(
                        source_gray.shape,
                        contour,
                        line_thickness=config.normalized_outline_thickness(),
                    )

                if _render_is_empty(silhouette_img):
                    raise ValueError("Rendered silhouette is empty after fallback")

            write_grayscale_png(silhouette_path, silhouette_img, dry_run=config.dry_run)

            if config.persist_edge_debug:
                edge_debug = np.full(generated.edge_binary.shape, 255, dtype=np.uint8)
                edge_debug[generated.edge_binary > 0] = 0
                write_grayscale_png(edge_debug_path, edge_debug, dry_run=config.dry_run)

            area_px, bbox = _contour_geometry(contour)

            samples_df.at[row_idx, "silhouette_fallback_used"] = "true" if fallback_used else "false"
            samples_df.at[row_idx, "silhouette_fallback_reason"] = fallback_reason
            samples_df.at[row_idx, "silhouette_area_px"] = str(area_px)
            samples_df.at[row_idx, "silhouette_bbox_x1"] = str(bbox[0])
            samples_df.at[row_idx, "silhouette_bbox_y1"] = str(bbox[1])
            samples_df.at[row_idx, "silhouette_bbox_x2"] = str(bbox[2])
            samples_df.at[row_idx, "silhouette_bbox_y2"] = str(bbox[3])

            samples_df.at[row_idx, "silhouette_stage_status"] = "success"
            samples_df.at[row_idx, "silhouette_stage_error"] = ""
        except Exception as exc:
            samples_df.at[row_idx, "silhouette_stage_status"] = "failed"
            samples_df.at[row_idx, "silhouette_stage_error"] = str(exc)
            logger.log(f"Row {row_idx} failed: {exc}")
            if not config.continue_on_error:
                aborted = True
                break

    output_samples_path = samples_csv_path(output_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=config.dry_run)

    status_series = samples_df["silhouette_stage_status"].fillna("")
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
        raise RuntimeError("Silhouette stage stopped after first row failure (continue_on_error=False).")

    return StageSummaryV2(
        run_name=run_name,
        stage_name="silhouette",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(output_paths.root),
        log_path=str(log_path),
        dry_run=config.dry_run,
    )



def _selected_row_indices(samples_df, *, offset: int, limit: int) -> set[int]:
    rows = list(samples_df.index)
    if not rows:
        return set()

    start = max(0, int(offset))
    if start >= len(rows):
        return set()

    if limit <= 0:
        return set(rows[start:])

    end = start + int(limit)
    return set(rows[start:end])



def _contour_break_reason(contour: np.ndarray | None) -> str:
    if contour is None:
        return "no_contour"
    if contour.ndim != 3 or contour.shape[0] < 3:
        return "degenerate_contour"

    area = float(abs(cv2.contourArea(contour)))
    if area < 1.0:
        return "degenerate_contour_area"

    return ""



def _render_is_empty(gray_image: np.ndarray) -> bool:
    if gray_image.ndim != 2:
        return True
    return not bool(np.any(gray_image < 255))



def _contour_geometry(contour: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    area_px = int(round(abs(float(cv2.contourArea(contour)))))
    x, y, w, h = cv2.boundingRect(contour)
    x2 = int(x + max(0, w - 1))
    y2 = int(y + max(0, h - 1))
    return area_px, (int(x), int(y), x2, y2)
