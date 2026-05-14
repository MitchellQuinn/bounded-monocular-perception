"""Central adapter around the authoritative rb_pipeline_v4 edge ROI path."""

from __future__ import annotations

from .config import BootstrapCenterTargetConfig
from .external import ensure_external_paths

ensure_external_paths()

from rb_pipeline_v4.config import DetectStageConfigV4
from rb_pipeline_v4.contracts import Detection
from rb_pipeline_v4.detector import EdgeRoiDetector


_PUBLIC_BACKEND = "edge_roi_v1"
_RB_PIPELINE_BACKEND = "edge"


def normalize_public_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend != _PUBLIC_BACKEND:
        raise ValueError(
            f"Unsupported detector_backend '{value}'. Only '{_PUBLIC_BACKEND}' is allowed in v0.1."
        )
    return backend


def to_rb_pipeline_backend(value: str) -> str:
    normalize_public_backend(value)
    return _RB_PIPELINE_BACKEND


def build_detect_stage_config(config: BootstrapCenterTargetConfig) -> DetectStageConfigV4:
    return DetectStageConfigV4(
        detector_backend=to_rb_pipeline_backend(config.normalized_detector_backend()),
        edge_blur_kernel_size=config.normalized_edge_blur_k(),
        edge_canny_low_threshold=config.normalized_edge_low(),
        edge_canny_high_threshold=config.normalized_edge_high(),
        edge_foreground_threshold=config.normalized_fg_threshold(),
        edge_padding_px=config.normalized_edge_pad(),
        edge_ignore_border_px=config.normalized_edge_ignore_border_px(),
        edge_min_foreground_px=config.normalized_min_edge_pixels(),
        edge_close_kernel_size=config.normalized_edge_close_kernel_size(),
        defender_class_names=("defender",),
        overwrite=bool(config.overwrite),
        dry_run=bool(config.dry_run),
        continue_on_error=bool(config.continue_on_error),
        persist_debug=bool(config.persist_debug_images),
    )


def build_edge_roi_detector(config: BootstrapCenterTargetConfig) -> EdgeRoiDetector:
    detect_config = build_detect_stage_config(config)
    return EdgeRoiDetector(
        blur_kernel_size=detect_config.normalized_edge_blur_kernel_size(),
        canny_low_threshold=detect_config.normalized_edge_canny_low_threshold(),
        canny_high_threshold=detect_config.normalized_edge_canny_high_threshold(),
        foreground_threshold=detect_config.normalized_edge_foreground_threshold(),
        padding_px=detect_config.normalized_edge_padding_px(),
        ignore_border_px=detect_config.normalized_edge_ignore_border_px(),
        min_foreground_px=detect_config.normalized_edge_min_foreground_px(),
        close_kernel_size=detect_config.normalized_edge_close_kernel_size(),
        class_id=0,
        class_name="defender",
    )


def detect_single_roi(image_bgr, config: BootstrapCenterTargetConfig) -> Detection | None:
    detector = build_edge_roi_detector(config)
    detections = detector.detect(image_bgr)
    if not detections:
        return None
    if len(detections) != 1:
        raise ValueError(f"Expected exactly one edge ROI detection, got {len(detections)}.")
    return detections[0]
