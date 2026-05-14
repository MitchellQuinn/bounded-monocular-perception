"""Stage 1 (v4): source image -> defender ROI detection metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2

from .config import DetectStageConfigV4, StageSummaryV4
from .constants import DETECT_STAGE_COLUMNS, UNITY_REQUIRED_COLUMNS
from .contracts import Detection, ObjectDetector
from .detector import EdgeRoiDetector, UltralyticsYoloDetector
from .image_io import read_image_unchanged, to_bgr_uint8, write_bgr_png
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
    to_posix_path,
)
from .utils import selected_row_indices, sha256_file
from .validation import (
    PipelineValidationError,
    capture_success_mask,
    validate_capture_success_images,
    validate_required_columns,
    validate_run_structure,
)



def run_detect_stage_v4(
    project_root: Path,
    run_name: str,
    config: DetectStageConfigV4,
    *,
    detector: ObjectDetector | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV4:
    """Run detector stage (YOLO or edge ROI backend) for one run."""

    source_paths = input_run_paths(project_root, run_name)
    output_paths = detect_run_paths(project_root, run_name)

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

    append_columns(samples_df, DETECT_STAGE_COLUMNS)
    for column in DETECT_STAGE_COLUMNS:
        samples_df[column] = samples_df[column].astype("object")
    capture_mask = capture_success_mask(samples_df)

    selected_rows = selected_row_indices(
        len(samples_df),
        offset=config.normalized_sample_offset(),
        limit=config.normalized_sample_limit(),
    )

    ensure_run_dirs(output_paths, dry_run=config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=config.dry_run)

    detector_backend = config.normalized_detector_backend()

    model_path = ""
    default_model_ref = ""
    model_ref = ""
    model_sha = ""
    detector_name = "edge_roi_v1" if detector_backend == "edge" else "ultralytics_yolo"

    if detector_backend == "yolo":
        model_path = config.normalized_model_path()
        default_model_ref = config.normalized_default_model_ref()
        model_ref = _resolve_model_ref(
            project_root=project_root,
            explicit_model_path=model_path,
            default_model_ref=default_model_ref,
        )
        if not model_ref:
            raise ValueError(
                "No YOLO model configured. Set DetectStageConfigV4.model_path or "
                "DetectStageConfigV4.default_model_ref."
            )

        if Path(model_ref).is_file():
            model_sha = sha256_file(Path(model_ref))
    else:
        model_ref = "edge_roi_detector_v1"

    upsert_preprocessing_contract_v4(
        output_paths.manifests_dir,
        stage_name="detect",
        stage_parameters={
            "DetectorBackend": detector_backend,
            "ModelPath": model_path,
            "ModelRef": model_ref,
            "DefaultModelRef": default_model_ref,
            "ModelSHA256": model_sha,
            "DefenderClassIds": list(config.normalized_defender_class_ids()),
            "DefenderClassNames": list(config.normalized_defender_class_names()),
            "ConfidenceThreshold": config.normalized_conf_threshold(),
            "IoUThreshold": config.normalized_iou_threshold(),
            "ImageSize": config.normalized_imgsz(),
            "MaxDetections": config.normalized_max_det(),
            "EdgeBlurKernelSize": config.normalized_edge_blur_kernel_size(),
            "EdgeCannyLowThreshold": config.normalized_edge_canny_low_threshold(),
            "EdgeCannyHighThreshold": config.normalized_edge_canny_high_threshold(),
            "EdgeForegroundThreshold": config.normalized_edge_foreground_threshold(),
            "EdgePaddingPx": config.normalized_edge_padding_px(),
            "EdgeMinForegroundPx": config.normalized_edge_min_foreground_px(),
            "EdgeCloseKernelSize": config.normalized_edge_close_kernel_size(),
            "EdgeIgnoreBorderPx": config.normalized_edge_ignore_border_px(),
            "PersistDebug": bool(config.persist_debug),
            "SampleOffset": config.normalized_sample_offset(),
            "SampleLimit": config.normalized_sample_limit(),
        },
        current_representation={
            "Kind": "detected_bbox_metadata",
            "StorageFormat": "csv",
            "Detector": detector_name,
            "BBoxFormat": "xyxy_px",
            "ClassSelection": "configured_allowlist" if detector_backend == "yolo" else "single_candidate",
        },
        dry_run=config.dry_run,
    )

    if detector is None:
        if detector_backend == "yolo":
            detector = UltralyticsYoloDetector(
                model_path=model_ref,
                conf_threshold=config.normalized_conf_threshold(),
                iou_threshold=config.normalized_iou_threshold(),
                imgsz=config.normalized_imgsz(),
                max_det=config.normalized_max_det(),
                device=config.normalized_device(),
            )
        else:
            edge_names = config.normalized_defender_class_names()
            edge_ids = config.normalized_defender_class_ids()
            detector = EdgeRoiDetector(
                blur_kernel_size=config.normalized_edge_blur_kernel_size(),
                canny_low_threshold=config.normalized_edge_canny_low_threshold(),
                canny_high_threshold=config.normalized_edge_canny_high_threshold(),
                foreground_threshold=config.normalized_edge_foreground_threshold(),
                padding_px=config.normalized_edge_padding_px(),
                min_foreground_px=config.normalized_edge_min_foreground_px(),
                close_kernel_size=config.normalized_edge_close_kernel_size(),
                ignore_border_px=config.normalized_edge_ignore_border_px(),
                class_id=int(edge_ids[0]) if edge_ids else 0,
                class_name=str(edge_names[0]) if edge_names else "defender",
            )

    log_path = output_paths.manifests_dir / "detect_stage_log.txt"
    logger = StageLogger(
        stage_name="detect",
        run_name=run_name,
        log_path=log_path,
        dry_run=config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running v4 detect stage for run '{run_name}'")
    logger.log_parameters(config.to_log_dict())
    if detector_backend == "yolo" and not model_path:
        logger.log(f"No explicit model_path set; using fallback model ref '{model_ref}'.")
    if detector_backend == "edge":
        logger.log("Using edge detector backend (v1-style Canny + foreground bbox with centroid-centered ROI).")

    if detector_backend == "edge":
        allowed_ids: set[int] = set()
        allowed_names: set[str] = set()
    else:
        allowed_ids = set(config.normalized_defender_class_ids())
        allowed_names = {value.lower() for value in config.normalized_defender_class_names()}

    aborted = False
    selected_total = int(len(selected_rows))
    progress_interval = 500
    processed_selected = 0
    progress_success = 0
    progress_failed = 0
    progress_skipped = 0
    logger.log(f"Selected rows for detect processing: {selected_total} / {len(samples_df)}")

    def _log_progress_if_needed() -> None:
        if (
            selected_total > 0
            and (
                processed_selected % progress_interval == 0
                or processed_selected == selected_total
            )
        ):
            logger.log(
                "Progress detect: "
                f"{processed_selected}/{selected_total} selected rows "
                f"(success={progress_success}, failed={progress_failed}, skipped={progress_skipped})"
            )

    for row_idx in samples_df.index:
        samples_df.at[row_idx, "detect_stage_status"] = ""
        samples_df.at[row_idx, "detect_stage_error"] = ""
        samples_df.at[row_idx, "detect_model_path"] = model_ref
        samples_df.at[row_idx, "detect_model_sha256"] = model_sha
        samples_df.at[row_idx, "detect_class_id"] = ""
        samples_df.at[row_idx, "detect_class_name"] = ""
        samples_df.at[row_idx, "detect_confidence"] = ""
        samples_df.at[row_idx, "detect_bbox_x1"] = ""
        samples_df.at[row_idx, "detect_bbox_y1"] = ""
        samples_df.at[row_idx, "detect_bbox_x2"] = ""
        samples_df.at[row_idx, "detect_bbox_y2"] = ""
        samples_df.at[row_idx, "detect_bbox_w_px"] = ""
        samples_df.at[row_idx, "detect_bbox_h_px"] = ""
        samples_df.at[row_idx, "detect_center_x_px"] = ""
        samples_df.at[row_idx, "detect_center_y_px"] = ""
        samples_df.at[row_idx, "detect_candidates_total"] = ""
        samples_df.at[row_idx, "detect_debug_image_filename"] = ""

        if row_idx not in selected_rows:
            samples_df.at[row_idx, "detect_stage_status"] = "skipped"
            samples_df.at[row_idx, "detect_stage_error"] = "outside selected subset"
            continue

        if not bool(capture_mask.loc[row_idx]):
            samples_df.at[row_idx, "detect_stage_status"] = "skipped"
            samples_df.at[row_idx, "detect_stage_error"] = "capture_success is false"
            processed_selected += 1
            progress_skipped += 1
            _log_progress_if_needed()
            continue

        image_filename = samples_df.at[row_idx, "image_filename"]
        try:
            source_path = resolve_manifest_path(source_paths.root, "images", image_filename)
            source_img = read_image_unchanged(source_path)
            source_bgr = to_bgr_uint8(source_img)

            detections = detector.detect(source_bgr)
            samples_df.at[row_idx, "detect_candidates_total"] = int(len(detections))

            selected = _select_detection(detections, allowed_ids=allowed_ids, allowed_names=allowed_names)
            if selected is None:
                samples_df.at[row_idx, "detect_stage_status"] = "failed"
                samples_df.at[row_idx, "detect_stage_error"] = "no matching defender detection"
                processed_selected += 1
                progress_failed += 1
                _log_progress_if_needed()
                continue

            clamped = _clamp_detection(selected, frame_width=source_bgr.shape[1], frame_height=source_bgr.shape[0])
            x1, y1, x2, y2 = clamped.x1, clamped.y1, clamped.x2, clamped.y2
            w_px = max(1e-6, x2 - x1)
            h_px = max(1e-6, y2 - y1)
            cx_px = float(clamped.center_x_px) if clamped.center_x_px is not None else float(x1 + (w_px * 0.5))
            cy_px = float(clamped.center_y_px) if clamped.center_y_px is not None else float(y1 + (h_px * 0.5))

            samples_df.at[row_idx, "detect_class_id"] = int(clamped.class_id)
            samples_df.at[row_idx, "detect_class_name"] = str(clamped.class_name)
            samples_df.at[row_idx, "detect_confidence"] = float(clamped.confidence)
            samples_df.at[row_idx, "detect_bbox_x1"] = float(x1)
            samples_df.at[row_idx, "detect_bbox_y1"] = float(y1)
            samples_df.at[row_idx, "detect_bbox_x2"] = float(x2)
            samples_df.at[row_idx, "detect_bbox_y2"] = float(y2)
            samples_df.at[row_idx, "detect_bbox_w_px"] = float(w_px)
            samples_df.at[row_idx, "detect_bbox_h_px"] = float(h_px)
            samples_df.at[row_idx, "detect_center_x_px"] = float(cx_px)
            samples_df.at[row_idx, "detect_center_y_px"] = float(cy_px)

            if config.persist_debug:
                debug_rel = normalize_relative_filename(image_filename, new_suffix=".detect.png")
                debug_path = output_paths.images_dir / debug_rel
                debug_img = _draw_detection_debug(source_bgr, clamped)
                write_bgr_png(debug_path, debug_img, dry_run=config.dry_run)
                samples_df.at[row_idx, "detect_debug_image_filename"] = to_posix_path(debug_rel)

            samples_df.at[row_idx, "detect_stage_status"] = "success"
            samples_df.at[row_idx, "detect_stage_error"] = ""
            processed_selected += 1
            progress_success += 1
            _log_progress_if_needed()
        except Exception as exc:
            samples_df.at[row_idx, "detect_stage_status"] = "failed"
            samples_df.at[row_idx, "detect_stage_error"] = str(exc)
            logger.log(f"Row {row_idx} failed: {exc}")
            processed_selected += 1
            progress_failed += 1
            _log_progress_if_needed()
            if not config.continue_on_error:
                aborted = True
                break

    output_samples_path = samples_csv_path(output_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=config.dry_run)

    status_series = samples_df["detect_stage_status"].fillna("")
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
        raise RuntimeError("Detect stage stopped after first row failure (continue_on_error=False).")

    return StageSummaryV4(
        run_name=run_name,
        stage_name="detect",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(output_paths.root),
        log_path=str(log_path),
        dry_run=config.dry_run,
    )



def _select_detection(
    detections: list[Detection],
    *,
    allowed_ids: set[int],
    allowed_names: set[str],
) -> Detection | None:
    if not detections:
        return None

    filtered: list[Detection] = []
    for detection in detections:
        class_name = str(detection.class_name).strip().lower()

        if allowed_ids or allowed_names:
            if detection.class_id in allowed_ids or class_name in allowed_names:
                filtered.append(detection)
        else:
            filtered.append(detection)

    if not filtered:
        return None
    return max(filtered, key=lambda item: float(item.confidence))



def _clamp_detection(detection: Detection, *, frame_width: int, frame_height: int) -> Detection:
    width = max(1, int(frame_width))
    height = max(1, int(frame_height))

    x1 = max(0.0, min(float(width - 1), float(detection.x1)))
    y1 = max(0.0, min(float(height - 1), float(detection.y1)))
    x2 = max(x1 + 1e-6, min(float(width), float(detection.x2)))
    y2 = max(y1 + 1e-6, min(float(height), float(detection.y2)))

    center_x = detection.center_x_px
    center_y = detection.center_y_px
    if center_x is not None:
        center_x = max(0.0, min(float(width - 1), float(center_x)))
    if center_y is not None:
        center_y = max(0.0, min(float(height - 1), float(center_y)))

    return Detection(
        class_id=int(detection.class_id),
        class_name=str(detection.class_name),
        confidence=float(detection.confidence),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x_px=center_x,
        center_y_px=center_y,
    )



def _draw_detection_debug(image_bgr, detection: Detection):
    out = image_bgr.copy()
    x1 = int(max(0, round(detection.x1)))
    y1 = int(max(0, round(detection.y1)))
    x2 = int(max(x1 + 1, round(detection.x2)))
    y2 = int(max(y1 + 1, round(detection.y2)))

    cv2.rectangle(out, (x1, y1), (x2, y2), color=(40, 220, 40), thickness=2)
    label = f"{detection.class_name} {float(detection.confidence):.3f}"
    cv2.putText(
        out,
        label,
        (x1, max(12, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 220, 40),
        1,
        cv2.LINE_AA,
    )

    center_x = detection.center_x_px
    center_y = detection.center_y_px
    if center_x is None or center_y is None:
        center_x = (float(detection.x1) + float(detection.x2)) * 0.5
        center_y = (float(detection.y1) + float(detection.y2)) * 0.5
    cx = int(round(float(center_x)))
    cy = int(round(float(center_y)))
    cv2.circle(out, (cx, cy), radius=4, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
    return out


def _resolve_model_ref(
    *,
    project_root: Path,
    explicit_model_path: str,
    default_model_ref: str,
) -> str:
    if explicit_model_path:
        return explicit_model_path

    ref = str(default_model_ref).strip()
    if not ref:
        return ""

    ref_path = Path(ref)
    if ref_path.is_file():
        return str(ref_path)

    candidates = [
        project_root / ref,
        project_root / "rb_ui_v4" / ref,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return ref
