"""Stage 2 (v4): YOLO ROI -> silhouette artifacts (ROI + full-frame)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import SilhouetteStageConfigV4, StageSummaryV4
from .constants import DETECT_STAGE_COLUMNS, SILHOUETTE_STAGE_COLUMNS, UNITY_REQUIRED_COLUMNS
from .image_io import read_image_unchanged, to_grayscale_uint8, write_grayscale_png
from .logging_utils import StageLogger
from .manifest import (
    append_columns,
    copy_run_json,
    load_samples_csv,
    samples_csv_path,
    upsert_preprocessing_contract_v4,
    write_samples_csv,
)
from .paths import (
    detect_run_paths,
    ensure_run_dirs,
    input_run_paths,
    normalize_relative_filename,
    resolve_manifest_path,
    silhouette_run_paths,
    to_posix_path,
)
from .silhouette_algorithms import (
    ContourSilhouetteGeneratorV2,
    ConvexHullFallbackV1,
    FilledArtifactWriterV1,
    OutlineArtifactWriterV1,
)
from .utils import selected_row_indices
from .validation import (
    PipelineValidationError,
    capture_success_mask,
    validate_capture_success_images,
    validate_required_columns,
    validate_run_structure,
)

ROI_GEOMETRY_COLUMNS = [
    "silhouette_roi_request_x1_px",
    "silhouette_roi_request_y1_px",
    "silhouette_roi_request_x2_px",
    "silhouette_roi_request_y2_px",
    "silhouette_roi_source_x1_px",
    "silhouette_roi_source_y1_px",
    "silhouette_roi_source_x2_px",
    "silhouette_roi_source_y2_px",
    "silhouette_roi_canvas_x1_px",
    "silhouette_roi_canvas_y1_px",
    "silhouette_roi_canvas_x2_px",
    "silhouette_roi_canvas_y2_px",
    "silhouette_roi_canvas_width_px",
    "silhouette_roi_canvas_height_px",
    "silhouette_roi_padding_px",
]



def run_silhouette_stage_v4(
    project_root: Path,
    run_name: str,
    config: SilhouetteStageConfigV4,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV4:
    """Run silhouette stage using detection metadata and ROI extraction."""

    source_paths = detect_run_paths(project_root, run_name)
    input_paths = input_run_paths(project_root, run_name)
    output_paths = silhouette_run_paths(project_root, run_name)

    validation_errors = validate_run_structure(source_paths, require_images=False)
    validation_errors.extend(validate_run_structure(input_paths, require_images=True))
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    source_samples_path = samples_csv_path(source_paths.manifests_dir)
    samples_df = load_samples_csv(source_samples_path)

    required_columns = UNITY_REQUIRED_COLUMNS + DETECT_STAGE_COLUMNS
    validation_errors.extend(validate_required_columns(samples_df, required_columns))
    validation_errors.extend(
        validate_capture_success_images(
            samples_df,
            input_paths.root,
            image_column="image_filename",
            default_subdir="images",
            capture_column="capture_success",
        )
    )
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    append_columns(samples_df, SILHOUETTE_STAGE_COLUMNS)
    for column in SILHOUETTE_STAGE_COLUMNS:
        samples_df[column] = samples_df[column].astype("object")
    append_columns(samples_df, ROI_GEOMETRY_COLUMNS)
    for column in ROI_GEOMETRY_COLUMNS:
        samples_df[column] = samples_df[column].astype("object")
    capture_mask = capture_success_mask(samples_df)

    selected_rows = selected_row_indices(
        len(samples_df),
        offset=config.normalized_sample_offset(),
        limit=config.normalized_sample_limit(),
    )

    ensure_run_dirs(output_paths, dry_run=config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=config.dry_run)

    upsert_preprocessing_contract_v4(
        output_paths.manifests_dir,
        stage_name="silhouette",
        stage_parameters={
            "RepresentationMode": config.normalized_representation_mode(),
            "GeneratorId": config.normalized_generator_id(),
            "FallbackId": config.normalized_fallback_id(),
            "ROIPaddingPx": config.normalized_roi_padding_px(),
            "ROICanvasWidthPx": config.normalized_roi_canvas_width_px(),
            "ROICanvasHeightPx": config.normalized_roi_canvas_height_px(),
            "BlurKernelSize": config.normalized_blur_kernel_size(),
            "CannyLowThreshold": int(config.canny_low_threshold),
            "CannyHighThreshold": int(config.canny_high_threshold),
            "CloseKernelSize": config.normalized_close_kernel_size(),
            "DilateKernelSize": config.normalized_dilate_kernel_size(),
            "MinComponentAreaPx": config.normalized_min_component_area_px(),
            "FillHoles": bool(config.fill_holes),
            "OutlineThicknessPx": config.normalized_outline_thickness(),
            "UseConvexHullFallback": bool(config.use_convex_hull_fallback),
            "PersistDebug": bool(config.persist_debug),
            "SampleOffset": config.normalized_sample_offset(),
            "SampleLimit": config.normalized_sample_limit(),
        },
        current_representation={
            "Kind": f"silhouette_{config.normalized_representation_mode()}_png",
            "StorageFormat": "png",
            "ColorSpace": "grayscale",
            "ForegroundEncoding": "black_on_white",
            "Geometry": "full_frame_and_roi",
            "ROICanvas": [
                config.normalized_roi_canvas_height_px(),
                config.normalized_roi_canvas_width_px(),
            ],
            "SilhouetteScaling": "disabled",
        },
        dry_run=config.dry_run,
    )

    generator, fallback, writer, filled_writer = _resolve_components(config)

    log_path = output_paths.manifests_dir / "silhouette_stage_log.txt"
    logger = StageLogger(
        stage_name="silhouette",
        run_name=run_name,
        log_path=log_path,
        dry_run=config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running v4 silhouette stage for run '{run_name}'")
    logger.log_parameters(config.to_log_dict())

    aborted = False
    selected_total = int(len(selected_rows))
    progress_interval = 500
    processed_selected = 0
    progress_success = 0
    progress_failed = 0
    progress_skipped = 0
    logger.log(f"Selected rows for silhouette processing: {selected_total} / {len(samples_df)}")

    def _log_progress_if_needed() -> None:
        if (
            selected_total > 0
            and (
                processed_selected % progress_interval == 0
                or processed_selected == selected_total
            )
        ):
            logger.log(
                "Progress silhouette: "
                f"{processed_selected}/{selected_total} selected rows "
                f"(success={progress_success}, failed={progress_failed}, skipped={progress_skipped})"
            )

    for row_idx in samples_df.index:
        samples_df.at[row_idx, "silhouette_stage_status"] = ""
        samples_df.at[row_idx, "silhouette_stage_error"] = ""
        samples_df.at[row_idx, "silhouette_mode"] = config.normalized_representation_mode()
        samples_df.at[row_idx, "silhouette_image_filename"] = ""
        samples_df.at[row_idx, "silhouette_roi_image_filename"] = ""
        samples_df.at[row_idx, "silhouette_fallback_used"] = "false"
        samples_df.at[row_idx, "silhouette_fallback_reason"] = ""
        samples_df.at[row_idx, "silhouette_area_px"] = ""
        samples_df.at[row_idx, "silhouette_bbox_x1"] = ""
        samples_df.at[row_idx, "silhouette_bbox_y1"] = ""
        samples_df.at[row_idx, "silhouette_bbox_x2"] = ""
        samples_df.at[row_idx, "silhouette_bbox_y2"] = ""
        samples_df.at[row_idx, "silhouette_quality_flags"] = ""
        samples_df.at[row_idx, "silhouette_debug_roi_filename"] = ""
        samples_df.at[row_idx, "silhouette_debug_amalgamated_filename"] = ""
        for column in ROI_GEOMETRY_COLUMNS:
            samples_df.at[row_idx, column] = ""

        if row_idx not in selected_rows:
            samples_df.at[row_idx, "silhouette_stage_status"] = "skipped"
            samples_df.at[row_idx, "silhouette_stage_error"] = "outside selected subset"
            continue

        if not bool(capture_mask.loc[row_idx]):
            samples_df.at[row_idx, "silhouette_stage_status"] = "skipped"
            samples_df.at[row_idx, "silhouette_stage_error"] = "capture_success is false"
            processed_selected += 1
            progress_skipped += 1
            _log_progress_if_needed()
            continue

        detect_status = str(samples_df.at[row_idx, "detect_stage_status"]).strip().lower()
        if detect_status != "success":
            samples_df.at[row_idx, "silhouette_stage_status"] = "skipped"
            samples_df.at[row_idx, "silhouette_stage_error"] = "detect_stage_status is not success"
            processed_selected += 1
            progress_skipped += 1
            _log_progress_if_needed()
            continue

        image_filename = samples_df.at[row_idx, "image_filename"]

        try:
            source_path = resolve_manifest_path(input_paths.root, "images", image_filename)
            source_image = read_image_unchanged(source_path)
            source_gray = to_grayscale_uint8(source_image)

            roi_gray, source_bounds, roi_bounds, request_bounds = _extract_centered_canvas_from_row(
                source_gray,
                samples_df.loc[row_idx],
                canvas_width=config.normalized_roi_canvas_width_px(),
                canvas_height=config.normalized_roi_canvas_height_px(),
                padding_px=config.normalized_roi_padding_px(),
            )
            if roi_gray.size == 0:
                raise ValueError("empty ROI after bbox clamping")

            generated = generator.generate(
                roi_gray,
                blur_kernel_size=config.normalized_blur_kernel_size(),
                canny_low_threshold=config.canny_low_threshold,
                canny_high_threshold=config.canny_high_threshold,
                close_kernel_size=config.normalized_close_kernel_size(),
                dilate_kernel_size=config.normalized_dilate_kernel_size(),
                min_component_area_px=config.normalized_min_component_area_px(),
                fill_holes=bool(config.fill_holes),
            )

            contour = generated.contour
            fallback_used = False
            fallback_reason = ""
            quality_flags = [str(v).strip() for v in generated.quality_flags if str(v).strip()]

            primary_break_reason = _contour_break_reason(contour)
            if primary_break_reason:
                if not bool(config.use_convex_hull_fallback):
                    break_reason = generated.primary_reason or primary_break_reason
                    raise ValueError(f"Primary contour failed ({break_reason}) and fallback is disabled")

                contour, recovery_reason = fallback.recover(generated.fallback_mask)
                fallback_used = True
                fallback_reason = generated.primary_reason or primary_break_reason
                if contour is None:
                    raise ValueError(f"Fallback failed: {recovery_reason}")

            roi_silhouette = writer.render(
                roi_gray.shape,
                contour,
                line_thickness=config.normalized_outline_thickness(),
            )
            if _render_is_empty(roi_silhouette):
                if not fallback_used and bool(config.use_convex_hull_fallback):
                    contour, recovery_reason = fallback.recover(generated.fallback_mask)
                    fallback_used = True
                    fallback_reason = "primary_render_empty"
                    if contour is None:
                        raise ValueError(f"Fallback failed: {recovery_reason}")
                    roi_silhouette = writer.render(
                        roi_gray.shape,
                        contour,
                        line_thickness=config.normalized_outline_thickness(),
                    )

                if _render_is_empty(roi_silhouette):
                    raise ValueError("Rendered silhouette is empty after fallback")

            src_x1, src_y1, src_x2, src_y2 = source_bounds
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds

            full_silhouette = np.full(source_gray.shape, 255, dtype=np.uint8)
            roi_target = full_silhouette[src_y1:src_y2, src_x1:src_x2]
            roi_source_aligned = roi_silhouette[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_target[roi_source_aligned < 255] = 0
            full_silhouette[src_y1:src_y2, src_x1:src_x2] = roi_target

            silhouette_rel = normalize_relative_filename(image_filename, new_suffix=".png")
            roi_rel = silhouette_rel.with_name(f"{silhouette_rel.stem}.roi.png")
            silhouette_path = output_paths.images_dir / silhouette_rel
            roi_path = output_paths.images_dir / roi_rel
            write_grayscale_png(silhouette_path, full_silhouette, dry_run=config.dry_run)
            write_grayscale_png(roi_path, roi_silhouette, dry_run=config.dry_run)

            debug_roi_rel = silhouette_rel.with_name(f"{silhouette_rel.stem}.debug.roi.png")
            debug_amalg_rel = silhouette_rel.with_name(f"{silhouette_rel.stem}.debug.amalgamated.png")
            if config.persist_debug:
                debug_roi_path = output_paths.images_dir / debug_roi_rel
                edge_debug = np.full(generated.edge_binary.shape, 255, dtype=np.uint8)
                edge_debug[generated.edge_binary > 0] = 0
                write_grayscale_png(debug_roi_path, edge_debug, dry_run=config.dry_run)

                filled_roi = filled_writer.render(roi_gray.shape, contour, line_thickness=1)
                amalgamated = _assemble_debug_overlay(
                    roi_shape=roi_gray.shape,
                    edge_debug=edge_debug,
                    final_filled=filled_roi,
                )
                write_grayscale_png(output_paths.images_dir / debug_amalg_rel, amalgamated, dry_run=config.dry_run)

                samples_df.at[row_idx, "silhouette_debug_roi_filename"] = to_posix_path(debug_roi_rel)
                samples_df.at[row_idx, "silhouette_debug_amalgamated_filename"] = to_posix_path(debug_amalg_rel)

            area_px, bbox = _mask_geometry(full_silhouette < 255)
            if fallback_used:
                quality_flags.append("used_fallback")

            samples_df.at[row_idx, "silhouette_image_filename"] = to_posix_path(silhouette_rel)
            samples_df.at[row_idx, "silhouette_roi_image_filename"] = to_posix_path(roi_rel)
            samples_df.at[row_idx, "silhouette_fallback_used"] = "true" if fallback_used else "false"
            samples_df.at[row_idx, "silhouette_fallback_reason"] = fallback_reason
            samples_df.at[row_idx, "silhouette_area_px"] = int(area_px)
            samples_df.at[row_idx, "silhouette_bbox_x1"] = int(bbox[0])
            samples_df.at[row_idx, "silhouette_bbox_y1"] = int(bbox[1])
            samples_df.at[row_idx, "silhouette_bbox_x2"] = int(bbox[2])
            samples_df.at[row_idx, "silhouette_bbox_y2"] = int(bbox[3])
            samples_df.at[row_idx, "silhouette_quality_flags"] = ";".join(sorted(set(quality_flags)))
            samples_df.at[row_idx, "silhouette_stage_status"] = "success"
            samples_df.at[row_idx, "silhouette_stage_error"] = ""

            req_x1, req_y1, req_x2, req_y2 = request_bounds
            src_x1, src_y1, src_x2, src_y2 = source_bounds
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
            samples_df.at[row_idx, "silhouette_roi_request_x1_px"] = int(req_x1)
            samples_df.at[row_idx, "silhouette_roi_request_y1_px"] = int(req_y1)
            samples_df.at[row_idx, "silhouette_roi_request_x2_px"] = int(req_x2)
            samples_df.at[row_idx, "silhouette_roi_request_y2_px"] = int(req_y2)
            samples_df.at[row_idx, "silhouette_roi_source_x1_px"] = int(src_x1)
            samples_df.at[row_idx, "silhouette_roi_source_y1_px"] = int(src_y1)
            samples_df.at[row_idx, "silhouette_roi_source_x2_px"] = int(src_x2)
            samples_df.at[row_idx, "silhouette_roi_source_y2_px"] = int(src_y2)
            samples_df.at[row_idx, "silhouette_roi_canvas_x1_px"] = int(roi_x1)
            samples_df.at[row_idx, "silhouette_roi_canvas_y1_px"] = int(roi_y1)
            samples_df.at[row_idx, "silhouette_roi_canvas_x2_px"] = int(roi_x2)
            samples_df.at[row_idx, "silhouette_roi_canvas_y2_px"] = int(roi_y2)
            samples_df.at[row_idx, "silhouette_roi_canvas_width_px"] = int(config.normalized_roi_canvas_width_px())
            samples_df.at[row_idx, "silhouette_roi_canvas_height_px"] = int(config.normalized_roi_canvas_height_px())
            samples_df.at[row_idx, "silhouette_roi_padding_px"] = int(config.normalized_roi_padding_px())

            processed_selected += 1
            progress_success += 1
            _log_progress_if_needed()
        except Exception as exc:
            samples_df.at[row_idx, "silhouette_stage_status"] = "failed"
            samples_df.at[row_idx, "silhouette_stage_error"] = str(exc)
            logger.log(f"Row {row_idx} failed: {exc}")
            processed_selected += 1
            progress_failed += 1
            _log_progress_if_needed()
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

    return StageSummaryV4(
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



def _resolve_components(config: SilhouetteStageConfigV4):
    if config.normalized_generator_id() != "silhouette.contour_v2":
        raise ValueError("Only generator_id='silhouette.contour_v2' is supported in v4")
    if config.normalized_fallback_id() != "fallback.convex_hull_v1":
        raise ValueError("Only fallback_id='fallback.convex_hull_v1' is supported in v4")

    mode = config.normalized_representation_mode()
    writer = FilledArtifactWriterV1() if mode == "filled" else OutlineArtifactWriterV1()
    return ContourSilhouetteGeneratorV2(), ConvexHullFallbackV1(), writer, FilledArtifactWriterV1()



def _extract_centered_canvas_from_row(
    source_gray: np.ndarray,
    row,
    *,
    canvas_width: int,
    canvas_height: int,
    padding_px: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Extract fixed-size ROI canvas centered on detection center without scaling."""
    if source_gray.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got {source_gray.shape}")

    frame_height = int(source_gray.shape[0])
    frame_width = int(source_gray.shape[1])
    canvas_w = max(1, int(canvas_width))
    canvas_h = max(1, int(canvas_height))

    x1 = float(row["detect_bbox_x1"]) - float(padding_px)
    y1 = float(row["detect_bbox_y1"]) - float(padding_px)
    x2 = float(row["detect_bbox_x2"]) + float(padding_px)
    y2 = float(row["detect_bbox_y2"]) + float(padding_px)

    center_x = _safe_row_float(row.get("detect_center_x_px"))
    center_y = _safe_row_float(row.get("detect_center_y_px"))
    if center_x is None:
        center_x = (x1 + x2) * 0.5
    if center_y is None:
        center_y = (y1 + y2) * 0.5

    req_x1 = int(round(float(center_x) - (canvas_w / 2.0)))
    req_y1 = int(round(float(center_y) - (canvas_h / 2.0)))
    req_x2 = req_x1 + canvas_w
    req_y2 = req_y1 + canvas_h

    src_x1 = max(0, req_x1)
    src_y1 = max(0, req_y1)
    src_x2 = min(frame_width, req_x2)
    src_y2 = min(frame_height, req_y2)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        raise ValueError("empty ROI after centered canvas extraction")

    dst_x1 = src_x1 - req_x1
    dst_y1 = src_y1 - req_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = source_gray[src_y1:src_y2, src_x1:src_x2]

    return (
        canvas,
        (src_x1, src_y1, src_x2, src_y2),
        (dst_x1, dst_y1, dst_x2, dst_y2),
        (req_x1, req_y1, req_x2, req_y2),
    )


def _safe_row_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return number



def _contour_break_reason(contour: np.ndarray | None) -> str:
    if contour is None:
        return "no_contour"
    if contour.ndim != 3 or contour.shape[0] < 3:
        return "degenerate_contour"
    area = float(abs(cv2.contourArea(contour)))
    if area <= 0.0:
        return "degenerate_contour_area"
    return ""



def _render_is_empty(gray_image: np.ndarray) -> bool:
    return gray_image.ndim != 2 or not bool(np.any(gray_image < 255))



def _mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, (0, 0, 0, 0)
    return int(xs.size), (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))



def _assemble_debug_overlay(
    *,
    roi_shape: tuple[int, int],
    edge_debug: np.ndarray | None,
    final_filled: np.ndarray | None,
) -> np.ndarray:
    h, w = int(roi_shape[0]), int(roi_shape[1])
    blank = np.full((h, w), 255, dtype=np.uint8)

    layers = [
        edge_debug if edge_debug is not None else blank,
        final_filled if final_filled is not None else blank,
    ]

    out = blank.copy()
    for layer in layers:
        norm = np.clip(layer, 0, 255).astype(np.uint8)
        out[norm < 255] = 0
    return out
