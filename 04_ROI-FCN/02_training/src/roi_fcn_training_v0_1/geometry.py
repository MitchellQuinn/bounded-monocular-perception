"""Geometry mapping helpers for ROI-FCN targets, decode, and ROI evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .contracts import DecodedHeatmapPoint


def canvas_point_to_output_space(
    canvas_xy: np.ndarray | Iterable[float],
    *,
    canvas_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Map one canvas-space point into model output space."""
    canvas_h, canvas_w = int(canvas_hw[0]), int(canvas_hw[1])
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    point = np.asarray(canvas_xy, dtype=np.float32)
    if point.shape != (2,):
        raise ValueError(f"canvas_xy must have shape (2,), got {point.shape}")
    scale_x = float(output_w) / float(canvas_w)
    scale_y = float(output_h) / float(canvas_h)
    return np.asarray([point[0] * scale_x, point[1] * scale_y], dtype=np.float32)


def output_point_to_canvas_space(
    output_xy: np.ndarray | Iterable[float],
    *,
    canvas_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Map one output-space point back into locator canvas space."""
    canvas_h, canvas_w = int(canvas_hw[0]), int(canvas_hw[1])
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    point = np.asarray(output_xy, dtype=np.float32)
    if point.shape != (2,):
        raise ValueError(f"output_xy must have shape (2,), got {point.shape}")
    scale_x = float(canvas_w) / float(output_w)
    scale_y = float(canvas_h) / float(output_h)
    return np.asarray([point[0] * scale_x, point[1] * scale_y], dtype=np.float32)


def original_point_to_canvas_space(
    original_xy: np.ndarray | Iterable[float],
    *,
    resize_scale: float,
    pad_left_px: float,
    pad_top_px: float,
) -> np.ndarray:
    """Map an original-image point into locator canvas space."""
    point = np.asarray(original_xy, dtype=np.float32)
    if point.shape != (2,):
        raise ValueError(f"original_xy must have shape (2,), got {point.shape}")
    return np.asarray(
        [
            (float(point[0]) * float(resize_scale)) + float(pad_left_px),
            (float(point[1]) * float(resize_scale)) + float(pad_top_px),
        ],
        dtype=np.float32,
    )


def canvas_point_to_original_space(
    canvas_xy: np.ndarray | Iterable[float],
    *,
    resize_scale: float,
    pad_left_px: float,
    pad_top_px: float,
    source_wh_px: np.ndarray | Iterable[int],
) -> np.ndarray:
    """Map a locator-canvas point back into original image space."""
    if float(resize_scale) <= 0.0:
        raise ValueError(f"resize_scale must be > 0; got {resize_scale}")
    point = np.asarray(canvas_xy, dtype=np.float32)
    if point.shape != (2,):
        raise ValueError(f"canvas_xy must have shape (2,), got {point.shape}")
    source_wh = np.asarray(source_wh_px, dtype=np.float32)
    if source_wh.shape != (2,):
        raise ValueError(f"source_wh_px must have shape (2,), got {source_wh.shape}")
    original = np.asarray(
        [
            (float(point[0]) - float(pad_left_px)) / float(resize_scale),
            (float(point[1]) - float(pad_top_px)) / float(resize_scale),
        ],
        dtype=np.float32,
    )
    original[0] = np.clip(original[0], 0.0, max(0.0, float(source_wh[0]) - 1.0))
    original[1] = np.clip(original[1], 0.0, max(0.0, float(source_wh[1]) - 1.0))
    return original


def derive_roi_bounds(center_xy: np.ndarray | Iterable[float], *, roi_width_px: float, roi_height_px: float) -> np.ndarray:
    """Derive raw request ROI bounds from a predicted centre."""
    if float(roi_width_px) <= 0.0 or float(roi_height_px) <= 0.0:
        raise ValueError(
            f"roi_width_px and roi_height_px must be > 0; got {roi_width_px}, {roi_height_px}"
        )
    center = np.asarray(center_xy, dtype=np.float32)
    if center.shape != (2,):
        raise ValueError(f"center_xy must have shape (2,), got {center.shape}")
    half_w = float(roi_width_px) / 2.0
    half_h = float(roi_height_px) / 2.0
    return np.asarray(
        [
            float(center[0]) - half_w,
            float(center[1]) - half_h,
            float(center[0]) + half_w,
            float(center[1]) + half_h,
        ],
        dtype=np.float32,
    )


def roi_fully_contains_bbox(roi_xyxy: np.ndarray | Iterable[float], bbox_xyxy: np.ndarray | Iterable[float]) -> bool:
    """Return True when the ROI fully contains the bbox."""
    roi = np.asarray(roi_xyxy, dtype=np.float32)
    bbox = np.asarray(bbox_xyxy, dtype=np.float32)
    if roi.shape != (4,):
        raise ValueError(f"roi_xyxy must have shape (4,), got {roi.shape}")
    if bbox.shape != (4,):
        raise ValueError(f"bbox_xyxy must have shape (4,), got {bbox.shape}")
    return bool(
        float(roi[0]) <= float(bbox[0])
        and float(roi[1]) <= float(bbox[1])
        and float(roi[2]) >= float(bbox[2])
        and float(roi[3]) >= float(bbox[3])
    )


def decode_heatmap_argmax(
    heatmap: np.ndarray,
    *,
    canvas_hw: tuple[int, int],
    resize_scale: float,
    pad_left_px: float,
    pad_top_px: float,
    source_wh_px: np.ndarray | Iterable[int],
) -> DecodedHeatmapPoint:
    """Decode one predicted heatmap into output/canvas/original coordinates."""
    if heatmap.ndim != 2:
        raise ValueError(f"heatmap must have shape (H, W), got {heatmap.shape}")
    flat_index = int(np.argmax(heatmap))
    output_h, output_w = int(heatmap.shape[0]), int(heatmap.shape[1])
    output_y = int(flat_index // output_w)
    output_x = int(flat_index % output_w)
    canvas_xy = output_point_to_canvas_space(
        np.asarray([float(output_x), float(output_y)], dtype=np.float32),
        canvas_hw=canvas_hw,
        output_hw=(output_h, output_w),
    )
    original_xy = canvas_point_to_original_space(
        canvas_xy,
        resize_scale=resize_scale,
        pad_left_px=pad_left_px,
        pad_top_px=pad_top_px,
        source_wh_px=source_wh_px,
    )
    return DecodedHeatmapPoint(
        output_x=float(output_x),
        output_y=float(output_y),
        canvas_x=float(canvas_xy[0]),
        canvas_y=float(canvas_xy[1]),
        original_x=float(original_xy[0]),
        original_y=float(original_xy[1]),
        confidence=float(heatmap[output_y, output_x]),
    )
