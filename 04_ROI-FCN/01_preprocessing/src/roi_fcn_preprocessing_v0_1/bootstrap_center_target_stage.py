"""Stage 1: authoritative crop-center bootstrapping via edge ROI v1."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

import cv2
import numpy as np

from .config import BootstrapCenterTargetConfig
from .contracts import BOOTSTRAP_STAGE_COLUMNS, StageSummaryV01
from .edge_roi_adapter import build_edge_roi_detector
from .external import ensure_external_paths
from .manifest import (
    append_columns,
    copy_run_json,
    load_samples_csv,
    upsert_preprocessing_contract,
    write_samples_csv,
)
from .paths import SplitPaths, ensure_split_output_dirs, resolve_input_image_path, split_relative_path
from .validation import (
    RoiFcnPreprocessingValidationError,
    capture_success_mask,
    ensure_bootstrap_prerequisites,
    stage_summary_counts_from_samples,
    validate_capture_success_images,
    validate_input_split_structure,
)

ensure_external_paths()

from rb_pipeline_v4.image_io import read_image_unchanged, to_bgr_uint8, write_bgr_png
from rb_pipeline_v4.logging_utils import StageLogger


_STATUS_COLUMN = "bootstrap_center_target_stage_status"
_ERROR_COLUMN = "bootstrap_center_target_stage_error"
_ALGORITHM_NAME = "edge_roi_v1"


def run_bootstrap_center_target_stage(
    split_paths: SplitPaths,
    config: BootstrapCenterTargetConfig,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV01:
    """Run stage 1 for one split."""

    validation_errors = validate_input_split_structure(split_paths)
    if validation_errors:
        raise RoiFcnPreprocessingValidationError("\n".join(validation_errors))

    samples_df = load_samples_csv(split_paths.input_samples_csv_path)
    ensure_bootstrap_prerequisites(samples_df)
    image_errors = validate_capture_success_images(samples_df, split_paths)
    if image_errors:
        raise RoiFcnPreprocessingValidationError("\n".join(image_errors))

    append_columns(samples_df, BOOTSTRAP_STAGE_COLUMNS)
    for column in BOOTSTRAP_STAGE_COLUMNS:
        samples_df[column] = samples_df[column].astype("object")

    _prepare_split_output_root(split_paths, overwrite=config.overwrite, dry_run=config.dry_run)
    ensure_split_output_dirs(split_paths, dry_run=config.dry_run)
    copy_run_json(
        split_paths.input_manifests_dir,
        split_paths.output_manifests_dir,
        dry_run=config.dry_run,
    )

    detector = build_edge_roi_detector(config)

    log_path = split_paths.output_manifests_dir / "bootstrap_center_target_stage_log.txt"
    logger = StageLogger(
        stage_name="bootstrap_center_target",
        run_name=f"{split_paths.dataset_reference}/{split_paths.split_name}",
        log_path=log_path,
        dry_run=config.dry_run,
        sink=log_sink,
    )
    logger.log(
        "Running bootstrap_center_target for "
        f"dataset='{split_paths.dataset_reference}' split='{split_paths.split_name}'"
    )
    logger.log_parameters(config.to_log_dict())

    capture_mask = capture_success_mask(samples_df)
    aborted = False
    for row_idx, row in samples_df.iterrows():
        if not bool(capture_mask.at[row_idx]):
            _mark_row_skipped(samples_df, row_idx)
            continue

        try:
            image_path = resolve_input_image_path(split_paths, row.get("image_filename"))
            image = read_image_unchanged(image_path)
            image_bgr = to_bgr_uint8(image)
            detections = detector.detect(image_bgr)
            if not detections:
                raise ValueError("edge_roi_v1 returned no detection")
            if len(detections) != 1:
                raise ValueError(f"edge_roi_v1 returned {len(detections)} detections; expected exactly 1")

            detection = detections[0]
            bbox_w = max(0.0, float(detection.x2) - float(detection.x1))
            bbox_h = max(0.0, float(detection.y2) - float(detection.y1))
            center_x = (
                float(detection.center_x_px)
                if detection.center_x_px is not None
                else float(detection.x1 + detection.x2) / 2.0
            )
            center_y = (
                float(detection.center_y_px)
                if detection.center_y_px is not None
                else float(detection.y1 + detection.y2) / 2.0
            )

            samples_df.at[row_idx, _STATUS_COLUMN] = "success"
            samples_df.at[row_idx, _ERROR_COLUMN] = ""
            samples_df.at[row_idx, "bootstrap_target_algorithm"] = _ALGORITHM_NAME
            samples_df.at[row_idx, "bootstrap_confidence"] = float(detection.confidence)
            samples_df.at[row_idx, "bootstrap_bbox_x1"] = float(detection.x1)
            samples_df.at[row_idx, "bootstrap_bbox_y1"] = float(detection.y1)
            samples_df.at[row_idx, "bootstrap_bbox_x2"] = float(detection.x2)
            samples_df.at[row_idx, "bootstrap_bbox_y2"] = float(detection.y2)
            samples_df.at[row_idx, "bootstrap_bbox_w_px"] = float(bbox_w)
            samples_df.at[row_idx, "bootstrap_bbox_h_px"] = float(bbox_h)
            samples_df.at[row_idx, "bootstrap_center_x_px"] = float(center_x)
            samples_df.at[row_idx, "bootstrap_center_y_px"] = float(center_y)
            samples_df.at[row_idx, "bootstrap_debug_image_filename"] = ""

            if config.persist_debug_images:
                debug_rel = _write_debug_image(
                    split_paths=split_paths,
                    row=row,
                    image_bgr=image_bgr,
                    x1=float(detection.x1),
                    y1=float(detection.y1),
                    x2=float(detection.x2),
                    y2=float(detection.y2),
                    center_x=float(center_x),
                    center_y=float(center_y),
                    dry_run=config.dry_run,
                )
                samples_df.at[row_idx, "bootstrap_debug_image_filename"] = debug_rel
        except Exception as exc:
            _mark_row_failed(samples_df, row_idx, str(exc))
            if not config.continue_on_error:
                aborted = True
                logger.log(f"Stopping after row {row_idx} failure: {exc}")
                break

    write_samples_csv(samples_df, split_paths.output_samples_csv_path, dry_run=config.dry_run)

    successful_rows, failed_rows, skipped_rows = stage_summary_counts_from_samples(samples_df, _STATUS_COLUMN)
    summary = StageSummaryV01(
        dataset_reference=split_paths.dataset_reference,
        split_name=split_paths.split_name,
        stage_name="bootstrap_center_target",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(split_paths.output_samples_csv_path),
        log_path=str(log_path),
        dry_run=config.dry_run,
    )

    upsert_preprocessing_contract(
        split_paths.output_manifests_dir,
        stage_name="bootstrap_center_target",
        stage_parameters={
            "DetectorBackend": config.normalized_detector_backend(),
            "EdgeBlurKernelSize": config.normalized_edge_blur_k(),
            "EdgeCannyLowThreshold": config.normalized_edge_low(),
            "EdgeCannyHighThreshold": config.normalized_edge_high(),
            "EdgeForegroundThreshold": config.normalized_fg_threshold(),
            "EdgePaddingPx": config.normalized_edge_pad(),
            "EdgeMinForegroundPx": config.normalized_min_edge_pixels(),
            "EdgeCloseKernelSize": config.normalized_edge_close_kernel_size(),
        },
        current_representation={
            "Kind": "bootstrap_center_target_metadata",
            "StorageFormat": "csv",
            "TargetType": "crop_center_point",
            "TargetSource": "edge_roi_v1_bootstrap",
        },
        stage_summary=summary,
        dry_run=config.dry_run,
    )

    logger.log_summary(
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=split_paths.output_samples_csv_path,
    )
    logger.write()

    if aborted:
        raise RuntimeError("bootstrap_center_target stopped after first row failure (continue_on_error=False).")

    return summary


def _prepare_split_output_root(split_paths: SplitPaths, *, overwrite: bool, dry_run: bool) -> None:
    if not split_paths.output_root.exists():
        return
    existing_entries = list(split_paths.output_root.iterdir())
    if not existing_entries:
        return
    if not overwrite:
        raise FileExistsError(
            "Output split already exists. Enable overwrite=True to replace it: "
            f"{split_paths.output_root}"
        )
    if not dry_run:
        shutil.rmtree(split_paths.output_root)


def _mark_row_skipped(samples_df, row_idx: int) -> None:
    samples_df.at[row_idx, _STATUS_COLUMN] = "skipped"
    samples_df.at[row_idx, _ERROR_COLUMN] = ""
    samples_df.at[row_idx, "bootstrap_target_algorithm"] = ""
    samples_df.at[row_idx, "bootstrap_confidence"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_x1"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_y1"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_x2"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_y2"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_w_px"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_h_px"] = ""
    samples_df.at[row_idx, "bootstrap_center_x_px"] = ""
    samples_df.at[row_idx, "bootstrap_center_y_px"] = ""
    samples_df.at[row_idx, "bootstrap_debug_image_filename"] = ""


def _mark_row_failed(samples_df, row_idx: int, error_message: str) -> None:
    samples_df.at[row_idx, _STATUS_COLUMN] = "failed"
    samples_df.at[row_idx, _ERROR_COLUMN] = str(error_message)
    samples_df.at[row_idx, "bootstrap_target_algorithm"] = _ALGORITHM_NAME
    samples_df.at[row_idx, "bootstrap_confidence"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_x1"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_y1"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_x2"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_y2"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_w_px"] = ""
    samples_df.at[row_idx, "bootstrap_bbox_h_px"] = ""
    samples_df.at[row_idx, "bootstrap_center_x_px"] = ""
    samples_df.at[row_idx, "bootstrap_center_y_px"] = ""
    samples_df.at[row_idx, "bootstrap_debug_image_filename"] = ""


def _write_debug_image(
    *,
    split_paths: SplitPaths,
    row,
    image_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    center_x: float,
    center_y: float,
    dry_run: bool,
) -> str:
    canvas = np.asarray(image_bgr, dtype=np.uint8).copy()
    cv2.rectangle(
        canvas,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        color=(0, 255, 0),
        thickness=2,
    )
    cv2.circle(
        canvas,
        (int(round(center_x)), int(round(center_y))),
        radius=4,
        color=(0, 0, 255),
        thickness=-1,
    )

    sample_id = str(row.get("sample_id", "sample")).strip() or "sample"
    image_stem = Path(str(row.get("image_filename", sample_id))).stem
    debug_path = split_paths.output_root / "debug" / "bootstrap_center_target" / f"{sample_id}__{image_stem}.png"
    write_bgr_png(debug_path, canvas, dry_run=dry_run)
    return split_relative_path(split_paths.output_root, debug_path)
