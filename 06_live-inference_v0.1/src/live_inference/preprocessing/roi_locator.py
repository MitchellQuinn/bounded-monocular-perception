"""Implementation-local ROI locator seam for live preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, runtime_checkable


def _ensure_preprocessing_paths() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    for path in (repo_root / "02_synthetic-data-processing-v4.0",):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


_ensure_preprocessing_paths()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


@dataclass(frozen=True)
class RoiLocation:
    """Predicted source-image ROI center plus optional trace metadata."""

    center_xy_px: tuple[float, float]
    roi_bounds_xyxy_px: tuple[float, float, float, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class RoiLocator(Protocol):
    """Locates the source-image crop center for live preprocessing."""

    def locate(
        self,
        source_gray_image: Any,
        *,
        excluded_source_mask: np.ndarray | None = None,
    ) -> RoiLocation:
        ...


@dataclass(frozen=True)
class RoiFcnLocatorInput:
    """Prepared ROI-FCN locator canvas plus traceability geometry."""

    locator_image: np.ndarray
    source_image_wh_px: np.ndarray
    resized_image_wh_px: np.ndarray
    padding_ltrb_px: np.ndarray
    resize_scale: float


def build_roi_fcn_locator_input(
    source_gray: np.ndarray,
    *,
    canvas_width_px: int,
    canvas_height_px: int,
) -> RoiFcnLocatorInput:
    """Build the v0.4 ROI-FCN locator input canvas without loading a model."""
    if source_gray.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got {source_gray.shape}")

    src_h, src_w = int(source_gray.shape[0]), int(source_gray.shape[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source image shape: {source_gray.shape}")

    scale = min(float(canvas_width_px) / float(src_w), float(canvas_height_px) / float(src_h))
    resized_w = int(round(float(src_w) * scale))
    resized_h = int(round(float(src_h) * scale))
    if resized_w <= 0 or resized_h <= 0:
        raise ValueError(
            "Resized image dimensions must stay positive after aspect-preserving scale: "
            f"src={src_w}x{src_h}, scale={scale}"
        )

    pad_left = int((canvas_width_px - resized_w) // 2)
    pad_right = int(canvas_width_px - resized_w - pad_left)
    pad_top = int((canvas_height_px - resized_h) // 2)
    pad_bottom = int(canvas_height_px - resized_h - pad_top)
    if min(pad_left, pad_right, pad_top, pad_bottom) < 0:
        raise ValueError(
            "Computed negative padding; source image does not fit locator canvas: "
            f"src={src_w}x{src_h}, resized={resized_w}x{resized_h}, "
            f"canvas={canvas_width_px}x{canvas_height_px}"
        )

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_gray = cv2.resize(source_gray, (resized_w, resized_h), interpolation=interpolation)
    locator_canvas = np.zeros((canvas_height_px, canvas_width_px), dtype=np.float32)
    locator_canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = (
        resized_gray.astype(np.float32) / 255.0
    )

    return RoiFcnLocatorInput(
        locator_image=locator_canvas[None, ...].astype(np.float32),
        source_image_wh_px=np.asarray([src_w, src_h], dtype=np.int32),
        resized_image_wh_px=np.asarray([resized_w, resized_h], dtype=np.int32),
        padding_ltrb_px=np.asarray([pad_left, pad_top, pad_right, pad_bottom], dtype=np.int32),
        resize_scale=float(scale),
    )


def build_roi_fcn_exclusion_mask(
    excluded_source_mask: np.ndarray,
    *,
    locator_input: RoiFcnLocatorInput,
    output_hw: tuple[int, int],
) -> np.ndarray:
    """Map a source-space exclusion mask into ROI-FCN output heatmap space."""
    source_mask = np.asarray(excluded_source_mask, dtype=bool)
    src_w, src_h = (int(value) for value in locator_input.source_image_wh_px.tolist())
    if source_mask.shape != (src_h, src_w):
        raise ValueError(
            "Excluded source mask shape must match the source image: "
            f"shape={source_mask.shape}, expected={(src_h, src_w)}."
        )

    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    if output_h <= 0 or output_w <= 0:
        raise ValueError(f"Invalid ROI-FCN output shape: {(output_h, output_w)}.")

    canvas_h, canvas_w = (int(value) for value in locator_input.locator_image.shape[-2:])
    resized_w, resized_h = (
        int(value) for value in locator_input.resized_image_wh_px.tolist()
    )
    pad_left, pad_top, _, _ = (
        int(value) for value in locator_input.padding_ltrb_px.tolist()
    )

    resized_mask = cv2.resize(
        source_mask.astype(np.uint8),
        (resized_w, resized_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_mask[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = (
        resized_mask.astype(np.uint8)
    )
    if (output_h, output_w) == (canvas_h, canvas_w):
        return canvas_mask.astype(bool)
    return cv2.resize(
        canvas_mask,
        (output_w, output_h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


__all__ = [
    "RoiFcnLocatorInput",
    "RoiLocation",
    "RoiLocator",
    "build_roi_fcn_exclusion_mask",
    "build_roi_fcn_locator_input",
]
