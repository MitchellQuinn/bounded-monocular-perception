"""Stage 1 (v3): source PNG -> threshold PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from rb_pipeline.image_io import read_image_unchanged, to_grayscale_uint8, write_grayscale_png
from rb_pipeline.logging_utils import StageLogger
from rb_pipeline.validation import (
    PipelineValidationError,
    capture_success_mask,
    validate_capture_success_images,
    validate_required_columns,
    validate_run_structure,
)

from .config import StageSummaryV3, ThresholdStageConfigV3
from .manifest import (
    THRESHOLD_STAGE_COLUMNS,
    UNITY_REQUIRED_COLUMNS,
    append_columns,
    copy_run_json,
    load_samples_csv,
    samples_csv_path,
    upsert_preprocessing_contract_v3,
    write_samples_csv,
)
from .paths import (
    ensure_run_dirs_v3,
    input_run_paths,
    normalize_relative_filename,
    resolve_manifest_path,
    threshold_run_paths,
    to_posix_path,
)



def run_threshold_stage_v3(
    project_root: Path,
    run_name: str,
    config: ThresholdStageConfigV3,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV3:
    """Run v3 threshold generation for one input run."""

    mode = config.normalized_representation_mode()
    low_value, high_value = config.normalized_threshold_bounds()
    min_component_area_px = config.normalized_min_component_area_px()
    hole_close_kernel_size = config.normalized_hole_close_kernel_size()
    outline_thickness = config.normalized_outline_thickness()
    persist_debug = bool(config.persist_debug)
    keep_individual_debug = bool(config.keep_individual_debug_outputs)
    amalgamate_debug = bool(config.amalgamate_debug_outputs)

    source_paths = input_run_paths(project_root, run_name)
    output_paths = threshold_run_paths(project_root, run_name)

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

    append_columns(samples_df, THRESHOLD_STAGE_COLUMNS)
    capture_mask = capture_success_mask(samples_df)

    selected_rows = _selected_row_indices(
        samples_df,
        offset=config.normalized_sample_offset(),
        limit=config.normalized_sample_limit(),
    )

    ensure_run_dirs_v3(output_paths, dry_run=config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=config.dry_run)
    upsert_preprocessing_contract_v3(
        output_paths.manifests_dir,
        stage_name="threshold",
        stage_parameters={
            "RepresentationMode": mode,
            "ThresholdLowValue": int(config.threshold_low_value),
            "ThresholdHighValue": int(config.threshold_high_value),
            "ThresholdLowValueUsed": int(low_value),
            "ThresholdHighValueUsed": int(high_value),
            "InvertSelection": bool(config.invert_selection),
            "MinComponentAreaPx": int(config.min_component_area_px),
            "MinComponentAreaPxUsed": int(min_component_area_px),
            "FillInternalHoles": bool(config.fill_internal_holes),
            "HoleCloseKernelSize": int(config.hole_close_kernel_size),
            "HoleCloseKernelSizeUsed": int(hole_close_kernel_size),
            "OutlineThicknessPx": int(config.outline_thickness),
            "OutlineThicknessPxUsed": int(outline_thickness),
            "PersistDebug": persist_debug,
            "AmalgamateDebugOutputs": amalgamate_debug,
            "KeepIndividualDebugOutputs": keep_individual_debug,
            "SampleOffset": int(config.normalized_sample_offset()),
            "SampleLimit": int(config.normalized_sample_limit()),
        },
        current_representation={
            "Kind": f"threshold_{mode}_png",
            "RepresentationMode": mode,
            "StorageFormat": "png",
            "ColorSpace": "grayscale",
            "ForegroundEncoding": "white_on_black",
            "ThresholdSemantics": "gimp_inclusive_range",
            "Geometry": f"full_frame_threshold_{mode}",
        },
        dry_run=config.dry_run,
    )

    log_path = output_paths.manifests_dir / "threshold_stage_log.txt"
    logger = StageLogger(
        stage_name="threshold",
        run_name=run_name,
        log_path=log_path,
        dry_run=config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running v3 threshold stage for run '{run_name}'")
    logger.log_parameters(
        config.to_log_dict()
        | {
            "representation_mode_used": mode,
            "threshold_low_value_used": low_value,
            "threshold_high_value_used": high_value,
            "min_component_area_px_used": min_component_area_px,
            "fill_internal_holes_used": bool(config.fill_internal_holes),
            "hole_close_kernel_size_used": hole_close_kernel_size,
            "outline_thickness_used": outline_thickness,
        }
    )

    aborted = False

    for row_idx in samples_df.index:
        samples_df.at[row_idx, "threshold_mode"] = mode
        samples_df.at[row_idx, "threshold_low_value"] = str(low_value)
        samples_df.at[row_idx, "threshold_high_value"] = str(high_value)
        samples_df.at[row_idx, "threshold_invert_selection"] = "true" if config.invert_selection else "false"
        samples_df.at[row_idx, "threshold_area_px"] = ""
        samples_df.at[row_idx, "threshold_bbox_x1"] = ""
        samples_df.at[row_idx, "threshold_bbox_y1"] = ""
        samples_df.at[row_idx, "threshold_bbox_x2"] = ""
        samples_df.at[row_idx, "threshold_bbox_y2"] = ""
        samples_df.at[row_idx, "threshold_num_components_total"] = ""
        samples_df.at[row_idx, "threshold_num_components_after_filter"] = ""
        samples_df.at[row_idx, "threshold_quality_flags"] = ""
        samples_df.at[row_idx, "threshold_debug_binary_filename"] = ""
        samples_df.at[row_idx, "threshold_debug_selected_component_filename"] = ""
        samples_df.at[row_idx, "threshold_debug_amalgamated_filename"] = ""

        row_capture_success = bool(capture_mask.loc[row_idx])
        source_filename = samples_df.at[row_idx, "image_filename"]

        try:
            threshold_rel = normalize_relative_filename(source_filename, new_suffix=".png")
        except Exception as exc:
            samples_df.at[row_idx, "threshold_image_filename"] = ""
            samples_df.at[row_idx, "threshold_stage_status"] = "failed"
            samples_df.at[row_idx, "threshold_stage_error"] = f"Invalid source filename: {exc}"
            logger.log(f"Row {row_idx} failed: invalid source filename {source_filename}")
            if not config.continue_on_error:
                aborted = True
                break
            continue

        binary_debug_rel = threshold_rel.with_name(f"{threshold_rel.stem}.debug.binary.png")
        selected_debug_rel = threshold_rel.with_name(f"{threshold_rel.stem}.debug.selected_component.png")
        amalgamated_rel = threshold_rel.with_name(f"{threshold_rel.stem}.debug.amalgamated.png")

        samples_df.at[row_idx, "threshold_image_filename"] = to_posix_path(threshold_rel)
        if persist_debug:
            if keep_individual_debug:
                samples_df.at[row_idx, "threshold_debug_binary_filename"] = to_posix_path(binary_debug_rel)
                samples_df.at[row_idx, "threshold_debug_selected_component_filename"] = to_posix_path(selected_debug_rel)
            if amalgamate_debug:
                samples_df.at[row_idx, "threshold_debug_amalgamated_filename"] = to_posix_path(amalgamated_rel)

        if row_idx not in selected_rows:
            samples_df.at[row_idx, "threshold_stage_status"] = "skipped"
            samples_df.at[row_idx, "threshold_stage_error"] = "outside selected subset"
            continue

        if not row_capture_success:
            samples_df.at[row_idx, "threshold_stage_status"] = "skipped"
            samples_df.at[row_idx, "threshold_stage_error"] = "capture_success is false"
            continue

        source_path = resolve_manifest_path(source_paths.root, "images", source_filename)
        threshold_path = output_paths.images_dir / threshold_rel
        binary_debug_path = output_paths.images_dir / binary_debug_rel
        selected_debug_path = output_paths.images_dir / selected_debug_rel
        amalgamated_path = output_paths.images_dir / amalgamated_rel

        if threshold_path.exists() and not config.overwrite:
            samples_df.at[row_idx, "threshold_stage_status"] = "skipped"
            samples_df.at[row_idx, "threshold_stage_error"] = "output exists and overwrite is false"
            continue

        try:
            source_image = read_image_unchanged(source_path)
            source_gray = to_grayscale_uint8(source_image)

            gimp_binary = _gimp_threshold_binary(
                source_gray,
                low_value=low_value,
                high_value=high_value,
                invert_selection=bool(config.invert_selection),
            )
            selected_mask = _filter_components(gimp_binary, min_component_area_px=min_component_area_px)
            selected_mask = _postprocess_selected_mask(
                selected_mask,
                fill_internal_holes=bool(config.fill_internal_holes),
                hole_close_kernel_size=hole_close_kernel_size,
            )

            num_components_total, num_components_after_filter = _component_counts(
                gimp_binary,
                min_component_area_px=min_component_area_px,
            )

            if not np.any(selected_mask > 0):
                raise ValueError("Threshold output has no components after area filtering")

            output_image = _render_threshold_artifact(
                selected_mask,
                representation_mode=mode,
                outline_thickness=outline_thickness,
            )

            write_grayscale_png(threshold_path, output_image, dry_run=config.dry_run)

            if persist_debug:
                if keep_individual_debug:
                    write_grayscale_png(binary_debug_path, gimp_binary, dry_run=config.dry_run)
                    write_grayscale_png(selected_debug_path, selected_mask, dry_run=config.dry_run)
                if amalgamate_debug:
                    amalgamated = _assemble_debug_overlay(gimp_binary=gimp_binary, selected_mask=selected_mask)
                    write_grayscale_png(amalgamated_path, amalgamated, dry_run=config.dry_run)

            area_px, bbox = _mask_geometry(selected_mask)
            quality_flags: list[str] = []
            if bool(config.invert_selection):
                quality_flags.append("inverted_selection")
            if mode == "outline":
                quality_flags.append("outline_render")

            samples_df.at[row_idx, "threshold_area_px"] = str(area_px)
            samples_df.at[row_idx, "threshold_bbox_x1"] = str(bbox[0])
            samples_df.at[row_idx, "threshold_bbox_y1"] = str(bbox[1])
            samples_df.at[row_idx, "threshold_bbox_x2"] = str(bbox[2])
            samples_df.at[row_idx, "threshold_bbox_y2"] = str(bbox[3])
            samples_df.at[row_idx, "threshold_num_components_total"] = str(num_components_total)
            samples_df.at[row_idx, "threshold_num_components_after_filter"] = str(num_components_after_filter)
            samples_df.at[row_idx, "threshold_quality_flags"] = ";".join(quality_flags)
            if not keep_individual_debug:
                samples_df.at[row_idx, "threshold_debug_binary_filename"] = ""
                samples_df.at[row_idx, "threshold_debug_selected_component_filename"] = ""
            if not amalgamate_debug:
                samples_df.at[row_idx, "threshold_debug_amalgamated_filename"] = ""

            samples_df.at[row_idx, "threshold_stage_status"] = "success"
            samples_df.at[row_idx, "threshold_stage_error"] = ""
        except Exception as exc:
            samples_df.at[row_idx, "threshold_stage_status"] = "failed"
            samples_df.at[row_idx, "threshold_stage_error"] = str(exc)
            logger.log(f"Row {row_idx} failed: {exc}")
            if not config.continue_on_error:
                aborted = True
                break

    output_samples_path = samples_csv_path(output_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=config.dry_run)

    status_series = samples_df["threshold_stage_status"].fillna("")
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
        raise RuntimeError("Threshold stage stopped after first row failure (continue_on_error=False).")

    return StageSummaryV3(
        run_name=run_name,
        stage_name="threshold",
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



def _gimp_threshold_binary(
    source_gray: np.ndarray,
    *,
    low_value: int,
    high_value: int,
    invert_selection: bool,
) -> np.ndarray:
    if source_gray.ndim != 2:
        raise ValueError("source_gray must be 2D grayscale")

    binary = np.where((source_gray >= low_value) & (source_gray <= high_value), 255, 0).astype(np.uint8)
    if invert_selection:
        binary = 255 - binary
    return binary



def _filter_components(gimp_binary: np.ndarray, *, min_component_area_px: int) -> np.ndarray:
    if gimp_binary.ndim != 2:
        raise ValueError("gimp_binary must be 2D")

    foreground = (gimp_binary > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)

    selected = np.zeros_like(gimp_binary, dtype=np.uint8)
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_component_area_px):
            selected[labels == label] = 255

    return selected



def _component_counts(gimp_binary: np.ndarray, *, min_component_area_px: int) -> tuple[int, int]:
    foreground = (gimp_binary > 0).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)

    total = max(0, int(component_count - 1))
    passing = 0
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_component_area_px):
            passing += 1

    return total, passing


def _postprocess_selected_mask(
    selected_mask: np.ndarray,
    *,
    fill_internal_holes: bool,
    hole_close_kernel_size: int,
) -> np.ndarray:
    processed = selected_mask.copy()

    kernel_size = max(1, int(hole_close_kernel_size))
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

    if fill_internal_holes:
        processed = _fill_internal_holes(processed)

    return processed


def _fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")

    flood = mask.copy()
    h, w = flood.shape
    floodfill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, floodfill_mask, seedPoint=(0, 0), newVal=255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def _render_threshold_artifact(
    selected_mask: np.ndarray,
    *,
    representation_mode: str,
    outline_thickness: int,
) -> np.ndarray:
    canvas = np.zeros(selected_mask.shape, dtype=np.uint8)

    if representation_mode == "filled":
        canvas[selected_mask > 0] = 255
        return canvas

    contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found for outline rendering")

    cv2.drawContours(
        canvas,
        contours,
        contourIdx=-1,
        color=255,
        thickness=max(1, int(outline_thickness)),
        lineType=cv2.LINE_AA,
    )
    return canvas



def _mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return 0, (0, 0, 0, 0)

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    area_px = int(mask[mask > 0].size)
    return area_px, (x1, y1, x2, y2)



def _assemble_debug_overlay(*, gimp_binary: np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros(gimp_binary.shape, dtype=np.uint8)
    overlay[gimp_binary > 0] = 127
    overlay[selected_mask > 0] = 255
    return overlay
