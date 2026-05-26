"""Display-only smoothing for live preview ROI overlays."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .frame_preview_widget import FramePreviewOverlay


DEFAULT_OVERLAY_SMOOTHING_WINDOW_SECONDS = 0.5


@dataclass(frozen=True)
class _OverlaySample:
    timestamp_seconds: float
    source_image_wh_px: tuple[int, int] | None
    bbox_xyxy_px: tuple[float, float, float, float] | None
    center_xy_px: tuple[float, float] | None
    roi_bounds_xyxy_px: tuple[float, float, float, float] | None
    label: str


class FramePreviewOverlaySmoother:
    """Moving-average smoother for the GUI preview overlay only."""

    def __init__(
        self,
        *,
        window_seconds: float = DEFAULT_OVERLAY_SMOOTHING_WINDOW_SECONDS,
        smooth_bbox: bool = True,
    ) -> None:
        if not math.isfinite(float(window_seconds)) or float(window_seconds) <= 0.0:
            raise ValueError("window_seconds must be a positive finite number.")
        self.window_seconds = float(window_seconds)
        self.smooth_bbox = bool(smooth_bbox)
        self._samples: deque[_OverlaySample] = deque()
        self._source_image_wh_px: tuple[int, int] | None = None
        self._last_timestamp_seconds: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._source_image_wh_px = None
        self._last_timestamp_seconds = None

    def smooth_overlay(
        self,
        overlay: FramePreviewOverlay | None,
        *,
        now_seconds: float,
    ) -> FramePreviewOverlay | None:
        """Return a smoothed overlay, resetting on missing or incompatible input."""
        if overlay is None:
            self.reset()
            return None

        source_image_wh_px = _source_size(overlay.source_image_wh_px)
        timestamp = float(now_seconds)
        if not math.isfinite(timestamp):
            timestamp = (
                0.0
                if self._last_timestamp_seconds is None
                else self._last_timestamp_seconds
            )

        if self._should_reset(source_image_wh_px, timestamp):
            self.reset()

        sample = _OverlaySample(
            timestamp_seconds=timestamp,
            source_image_wh_px=source_image_wh_px,
            bbox_xyxy_px=_rect(overlay.bbox_xyxy_px),
            center_xy_px=_point(overlay.center_xy_px),
            roi_bounds_xyxy_px=_rect(overlay.roi_bounds_xyxy_px),
            label=overlay.label,
        )
        self._samples.append(sample)
        self._source_image_wh_px = source_image_wh_px
        self._last_timestamp_seconds = timestamp
        self._trim_window(timestamp)
        return _smoothed_overlay(
            tuple(self._samples),
            latest=sample,
            smooth_bbox=self.smooth_bbox,
        )

    def _should_reset(
        self,
        source_image_wh_px: tuple[int, int] | None,
        timestamp_seconds: float,
    ) -> bool:
        if self._source_image_wh_px != source_image_wh_px:
            return True
        return (
            self._last_timestamp_seconds is not None
            and timestamp_seconds < self._last_timestamp_seconds
        )

    def _trim_window(self, timestamp_seconds: float) -> None:
        cutoff = timestamp_seconds - self.window_seconds
        while self._samples and self._samples[0].timestamp_seconds < cutoff:
            self._samples.popleft()


def _smoothed_overlay(
    samples: tuple[_OverlaySample, ...],
    *,
    latest: _OverlaySample,
    smooth_bbox: bool,
) -> FramePreviewOverlay:
    return FramePreviewOverlay(
        source_image_wh_px=latest.source_image_wh_px,
        bbox_xyxy_px=(
            _average_rect(samples, "bbox_xyxy_px", keep_latest_size=False)
            if smooth_bbox
            else latest.bbox_xyxy_px
        ),
        center_xy_px=_average_point(samples, "center_xy_px"),
        roi_bounds_xyxy_px=_average_rect(
            samples,
            "roi_bounds_xyxy_px",
            keep_latest_size=True,
        ),
        label=latest.label,
    )


def _average_rect(
    samples: tuple[_OverlaySample, ...],
    field_name: str,
    *,
    keep_latest_size: bool,
) -> tuple[float, float, float, float] | None:
    rects = tuple(
        rect
        for sample in samples
        if (rect := getattr(sample, field_name)) is not None
    )
    if not rects:
        return None

    centers = tuple(_rect_center(rect) for rect in rects)
    center_x = sum(point[0] for point in centers) / len(centers)
    center_y = sum(point[1] for point in centers) / len(centers)
    if keep_latest_size:
        width, height = _rect_size(rects[-1])
    else:
        sizes = tuple(_rect_size(rect) for rect in rects)
        width = sum(size[0] for size in sizes) / len(sizes)
        height = sum(size[1] for size in sizes) / len(sizes)
    return _rect_from_center(center_x, center_y, width, height)


def _average_point(
    samples: tuple[_OverlaySample, ...],
    field_name: str,
) -> tuple[float, float] | None:
    points = tuple(
        point
        for sample in samples
        if (point := getattr(sample, field_name)) is not None
    )
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _rect_center(rect: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = rect
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _rect_size(rect: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = rect
    return (max(0.0, x2 - x1), max(0.0, y2 - y1))


def _rect_from_center(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    half_width = width / 2.0
    half_height = height / 2.0
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def _source_size(value: tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        width, height = value
        return int(width), int(height)
    except (TypeError, ValueError):
        return None


def _point(value: tuple[float, float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    try:
        x, y = value
        point = (float(x), float(y))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in point):
        return None
    return point


def _rect(
    value: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    try:
        x1, y1, x2, y2 = value
        rect = (float(x1), float(y1), float(x2), float(y2))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in rect):
        return None
    return rect


__all__ = [
    "DEFAULT_OVERLAY_SMOOTHING_WINDOW_SECONDS",
    "FramePreviewOverlaySmoother",
]
