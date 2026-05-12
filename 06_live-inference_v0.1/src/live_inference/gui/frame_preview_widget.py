"""Aspect-preserving live frame preview with mask editing and ROI overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget
import numpy as np

from live_inference.masking import (
    BackgroundSnapshot,
    FrameMaskSnapshot,
    apply_fill_to_mask,
    combine_ignore_masks,
    compute_background_removal_mask,
)


@dataclass(frozen=True)
class FramePreviewHeatmapOverlay:
    """ROI-FCN heatmap plus locator geometry needed to project it to source space."""

    heatmap_u8: np.ndarray
    canvas_wh_px: tuple[int, int]
    resized_image_wh_px: tuple[int, int]
    padding_ltrb_px: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        heatmap = np.asarray(self.heatmap_u8, dtype=np.uint8)
        if heatmap.ndim != 2:
            raise ValueError(f"Heatmap overlay must be 2D; got shape {heatmap.shape}.")
        copied = np.ascontiguousarray(heatmap)
        copied.setflags(write=False)
        object.__setattr__(self, "heatmap_u8", copied)


@dataclass(frozen=True)
class FramePreviewOverlay:
    """Source-image-space geometry to draw over the preview."""

    source_image_wh_px: tuple[int, int] | None = None
    bbox_xyxy_px: tuple[float, float, float, float] | None = None
    center_xy_px: tuple[float, float] | None = None
    roi_bounds_xyxy_px: tuple[float, float, float, float] | None = None
    roi_fcn_heatmap: FramePreviewHeatmapOverlay | None = None
    label: str = "Pipeline ROI / bbox"


@dataclass(frozen=True)
class FrameMaskEditResult:
    """Source-pixel draft mask returned when an edit session is applied."""

    width_px: int
    height_px: int
    mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        expected_shape = (int(self.height_px), int(self.width_px))
        if mask.shape != expected_shape:
            raise ValueError(
                f"Mask edit result shape {mask.shape} does not match {expected_shape}."
            )
        copied = np.array(mask, dtype=bool, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "mask", copied)


class FramePreviewWidget(QWidget):
    """Display one QPixmap and draw/edit source-pixel mask geometry over it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._raw_image: QImage | None = None
        self._raw_pixmap: QPixmap | None = None
        self._raw_rgb: np.ndarray | None = None
        self._raw_gray: np.ndarray | None = None
        self._placeholder_text = "No frame yet"
        self._overlay: FramePreviewOverlay | None = None
        self._heatmap_overlay_enabled = True
        self._committed_mask: np.ndarray | None = None
        self._committed_mask_size: tuple[int, int] | None = None
        self._background_snapshot: BackgroundSnapshot | None = None
        self._mask_fill_value = 255
        self._draft_mask: np.ndarray | None = None
        self._edit_mode: str | None = None
        self._is_painting = False
        self._brush_diameter_px = 100
        self._cursor_source_xy: tuple[int, int] | None = None
        self._committed_overlay_pixmap: QPixmap | None = None
        self._draft_overlay_pixmap: QPixmap | None = None
        self._heatmap_overlay_pixmap_cache: tuple[tuple[int, ...], QPixmap] | None = None
        self._background_preview_snapshot_cache: (
            tuple[tuple[int, bool, int, int, int, int, int], BackgroundSnapshot] | None
        ) = None
        self.setMouseTracking(False)

    def placeholder_text(self) -> str:
        return self._placeholder_text

    def set_placeholder_text(self, text: str) -> None:
        self._placeholder_text = str(text)
        self.update()

    def load_image(self, path: Path | str) -> bool:
        image = QImage(str(path))
        if image.isNull():
            return False
        self.set_image(image)
        return True

    def load_background_preview(
        self,
        path: Path | str,
        snapshot: BackgroundSnapshot | None,
        *,
        max_width_px: int = 480,
        max_height_px: int = 300,
        fill_value: int = 255,
    ) -> bool:
        image = QImage(str(path))
        if image.isNull():
            return False
        self.set_background_preview_image(
            image,
            snapshot,
            max_width_px=max_width_px,
            max_height_px=max_height_px,
            fill_value=fill_value,
        )
        return True

    def set_pixmap(self, pixmap: QPixmap) -> None:
        if pixmap.isNull():
            self._pixmap = None
            self._raw_image = None
            self._raw_pixmap = None
            self._raw_rgb = None
            self._raw_gray = None
            self.update()
            return
        self.set_image(pixmap.toImage())

    def set_image(self, image: QImage) -> None:
        if image.isNull():
            self._pixmap = None
            self._raw_image = None
            self._raw_pixmap = None
            self._raw_rgb = None
            self._raw_gray = None
            self.update()
            return
        safe_image = image.copy()
        self._raw_image = safe_image
        self._raw_rgb = None
        self._raw_gray = None
        self._raw_pixmap = QPixmap.fromImage(safe_image)
        self._refresh_effective_pixmap()
        if self._edit_mode is not None:
            self._ensure_draft_mask_for_current_source()
        self.update()

    def set_background_preview_image(
        self,
        image: QImage,
        snapshot: BackgroundSnapshot | None,
        *,
        max_width_px: int = 480,
        max_height_px: int = 300,
        fill_value: int = 255,
    ) -> None:
        if image.isNull():
            self.set_background_snapshot(None)
            self.set_image(image)
            return

        original_size = (int(image.width()), int(image.height()))
        preview_image = _scaled_image_to_fit(
            image,
            max_width_px=max_width_px,
            max_height_px=max_height_px,
        )
        preview_size = (int(preview_image.width()), int(preview_image.height()))
        preview_snapshot = self._scaled_background_snapshot_for_preview(
            snapshot,
            original_size=original_size,
            preview_size=preview_size,
        )
        self.set_background_snapshot(None)
        self.set_mask_fill_value(fill_value)
        self.set_image(preview_image)
        self.set_background_snapshot(preview_snapshot)

    def pixmap(self) -> QPixmap | None:
        return self._pixmap

    def raw_image(self) -> QImage | None:
        if self._raw_image is None or self._raw_image.isNull():
            return None
        return self._raw_image.copy()

    def _scaled_background_snapshot_for_preview(
        self,
        snapshot: BackgroundSnapshot | None,
        *,
        original_size: tuple[int, int],
        preview_size: tuple[int, int],
    ) -> BackgroundSnapshot | None:
        if snapshot is None or not snapshot.captured or not snapshot.enabled:
            return None
        original_w, original_h = (int(original_size[0]), int(original_size[1]))
        preview_w, preview_h = (int(preview_size[0]), int(preview_size[1]))
        if preview_w <= 0 or preview_h <= 0:
            return None
        if not snapshot.dimensions_match(original_w, original_h):
            return None
        key = (
            int(snapshot.revision),
            bool(snapshot.enabled),
            int(snapshot.threshold),
            original_w,
            original_h,
            preview_w,
            preview_h,
        )
        cached = self._background_preview_snapshot_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        preview_snapshot = _scaled_background_snapshot(
            snapshot,
            original_size=original_size,
            preview_size=preview_size,
        )
        if preview_snapshot is not None:
            self._background_preview_snapshot_cache = (key, preview_snapshot)
        return preview_snapshot

    def raw_pixmap(self) -> QPixmap | None:
        if self._raw_pixmap is None:
            return None
        return QPixmap(self._raw_pixmap)

    def raw_source_gray(self) -> np.ndarray | None:
        raw_gray = self._raw_gray_array()
        if raw_gray is None:
            return None
        return np.array(raw_gray, dtype=np.uint8, copy=True)

    def effective_preview_image(self) -> np.ndarray | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        return _qimage_to_rgb_array(self._pixmap.toImage())

    def source_image_size(self) -> tuple[int, int] | None:
        if self._raw_image is not None and not self._raw_image.isNull():
            return int(self._raw_image.width()), int(self._raw_image.height())
        if self._raw_gray is not None:
            return int(self._raw_gray.shape[1]), int(self._raw_gray.shape[0])
        if self._pixmap is None or self._pixmap.isNull():
            return None
        return int(self._pixmap.width()), int(self._pixmap.height())

    def set_overlay(self, overlay: FramePreviewOverlay | None) -> None:
        self._overlay = overlay
        self._heatmap_overlay_pixmap_cache = None
        self.update()

    def overlay(self) -> FramePreviewOverlay | None:
        return self._overlay

    def set_heatmap_overlay_enabled(self, enabled: bool) -> None:
        self._heatmap_overlay_enabled = bool(enabled)
        self.update()

    def heatmap_overlay_enabled(self) -> bool:
        return bool(self._heatmap_overlay_enabled)

    def set_committed_mask_snapshot(self, snapshot: FrameMaskSnapshot | None) -> None:
        if snapshot is not None:
            self._mask_fill_value = int(snapshot.fill_value)
        if snapshot is None or not snapshot.enabled or not snapshot.has_geometry:
            self._committed_mask = None
            self._committed_mask_size = None
        else:
            self._committed_mask = np.array(snapshot.mask, dtype=bool, copy=True)
            self._committed_mask_size = (int(snapshot.width_px), int(snapshot.height_px))
        self._committed_overlay_pixmap = None
        if self._edit_mode is not None:
            self._ensure_draft_mask_for_current_source()
        self._refresh_effective_pixmap()
        self.update()

    def set_background_snapshot(self, snapshot: BackgroundSnapshot | None) -> None:
        previous_key = _background_snapshot_key(self._background_snapshot)
        next_key = _background_snapshot_key(snapshot)
        self._background_snapshot = snapshot
        if previous_key == next_key:
            return
        self._refresh_effective_pixmap()
        self.update()

    def background_snapshot(self) -> BackgroundSnapshot | None:
        return self._background_snapshot

    def set_mask_fill_value(self, fill_value: int) -> None:
        value = int(fill_value)
        if value not in {0, 255}:
            raise ValueError(f"Mask fill value must be 0 or 255; got {fill_value!r}.")
        self._mask_fill_value = value
        self._refresh_effective_pixmap()
        self.update()

    def clear_masks(self) -> None:
        self._committed_mask = None
        self._committed_mask_size = None
        self._draft_mask = None
        self._edit_mode = None
        self._cursor_source_xy = None
        self._is_painting = False
        self._committed_overlay_pixmap = None
        self._draft_overlay_pixmap = None
        self.setMouseTracking(False)
        self._refresh_effective_pixmap()
        self.update()

    def begin_mask_edit(self, mode: str) -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"draw", "erase"}:
            raise ValueError(f"Unsupported mask edit mode: {mode!r}.")
        self._edit_mode = normalized
        self._is_painting = False
        self._cursor_source_xy = None
        self._ensure_draft_mask_for_current_source()
        self.setMouseTracking(True)
        self.update()

    def finish_mask_edit(self, *, commit: bool = True) -> FrameMaskEditResult | None:
        if self._edit_mode is None:
            return None
        result: FrameMaskEditResult | None = None
        source_size = self.source_image_size()
        if commit and source_size is not None and self._draft_mask is not None:
            width, height = source_size
            if self._draft_mask.shape == (height, width):
                result = FrameMaskEditResult(
                    width_px=width,
                    height_px=height,
                    mask=self._draft_mask,
                )
                self._committed_mask = np.array(self._draft_mask, dtype=bool, copy=True)
                self._committed_mask_size = (width, height)
                self._committed_overlay_pixmap = None
                self._refresh_effective_pixmap()
        self._edit_mode = None
        self._draft_mask = None
        self._draft_overlay_pixmap = None
        self._cursor_source_xy = None
        self._is_painting = False
        self.setMouseTracking(False)
        self.update()
        return result

    def cancel_mask_edit(self) -> None:
        self.finish_mask_edit(commit=False)

    def mask_edit_mode(self) -> str | None:
        return self._edit_mode

    def is_mask_editing(self) -> bool:
        return self._edit_mode is not None

    def set_brush_diameter_px(self, diameter_px: int) -> None:
        self._brush_diameter_px = min(1000, max(1, int(diameter_px)))
        self.update()

    def brush_diameter_px(self) -> int:
        return int(self._brush_diameter_px)

    def committed_mask(self) -> np.ndarray | None:
        if self._committed_mask is None:
            return None
        return np.array(self._committed_mask, dtype=bool, copy=True)

    def draft_mask(self) -> np.ndarray | None:
        if self._draft_mask is None:
            return None
        return np.array(self._draft_mask, dtype=bool, copy=True)

    def committed_mask_pixel_count(self) -> int:
        if self._committed_mask is None:
            return 0
        return int(np.count_nonzero(self._committed_mask))

    def draft_mask_pixel_count(self) -> int:
        if self._draft_mask is None:
            return 0
        return int(np.count_nonzero(self._draft_mask))

    def image_target_rect(self) -> QRectF:
        if self._pixmap is None or self._pixmap.isNull():
            return QRectF()
        widget_w = max(0.0, float(self.width()))
        widget_h = max(0.0, float(self.height()))
        image_w = float(self._pixmap.width())
        image_h = float(self._pixmap.height())
        if widget_w <= 0.0 or widget_h <= 0.0 or image_w <= 0.0 or image_h <= 0.0:
            return QRectF()
        scale = min(widget_w / image_w, widget_h / image_h)
        target_w = image_w * scale
        target_h = image_h * scale
        return QRectF(
            (widget_w - target_w) / 2.0,
            (widget_h - target_h) / 2.0,
            target_w,
            target_h,
        )

    def map_widget_point_to_source(
        self,
        point: QPointF | tuple[float, float],
    ) -> tuple[int, int] | None:
        rect = self.image_target_rect()
        source_size = self.source_image_size()
        if rect.isNull() or source_size is None:
            return None
        source_w, source_h = source_size
        if source_w <= 0 or source_h <= 0:
            return None
        widget_x, widget_y = _point_xy(point)
        if (
            widget_x < rect.left()
            or widget_y < rect.top()
            or widget_x >= rect.left() + rect.width()
            or widget_y >= rect.top() + rect.height()
        ):
            return None
        source_x = int((widget_x - rect.left()) * (float(source_w) / rect.width()))
        source_y = int((widget_y - rect.top()) * (float(source_h) / rect.height()))
        return (
            min(source_w - 1, max(0, source_x)),
            min(source_h - 1, max(0, source_y)),
        )

    def map_source_point_to_widget(
        self,
        point_xy_px: tuple[float, float],
        source_image_wh_px: tuple[int, int] | None,
    ) -> QPointF | None:
        rect = self.image_target_rect()
        source_w, source_h = _valid_source_size(source_image_wh_px)
        if rect.isNull() or source_w <= 0 or source_h <= 0:
            return None
        if self.source_image_size() != (source_w, source_h):
            return None
        x, y = point_xy_px
        return QPointF(
            rect.left() + (float(x) / float(source_w)) * rect.width(),
            rect.top() + (float(y) / float(source_h)) * rect.height(),
        )

    def map_source_rect_to_widget(
        self,
        xyxy_px: tuple[float, float, float, float],
        source_image_wh_px: tuple[int, int] | None,
    ) -> QRectF | None:
        source_w, source_h = _valid_source_size(source_image_wh_px)
        if source_w <= 0 or source_h <= 0:
            return None
        top_left = self.map_source_point_to_widget(
            (xyxy_px[0], xyxy_px[1]),
            (source_w, source_h),
        )
        bottom_right = self.map_source_point_to_widget(
            (xyxy_px[2], xyxy_px[3]),
            (source_w, source_h),
        )
        if top_left is None or bottom_right is None:
            return None
        return QRectF(top_left, bottom_right).normalized()

    def overlay_source_size_matches(
        self,
        source_image_wh_px: tuple[int, int] | None,
    ) -> bool:
        current = self.source_image_size()
        if current is None:
            return True
        source_w, source_h = _valid_source_size(source_image_wh_px)
        return source_w > 0 and source_h > 0 and current == (source_w, source_h)

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(18, 22, 27))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor(190, 196, 204))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder_text)
            return

        target_rect = self.image_target_rect()
        painter.drawPixmap(target_rect, self._pixmap, QRectF(self._pixmap.rect()))
        self._draw_heatmap_overlay(painter)
        self._draw_mask_overlay(painter)
        self._draw_overlay(painter)
        self._draw_brush_cursor(painter)

    def mousePressEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        if self._edit_mode is None or _event_button(event) != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        source_xy = self.map_widget_point_to_source(_event_position(event))
        if source_xy is None:
            self._is_painting = False
            return
        self._cursor_source_xy = source_xy
        self._is_painting = True
        self._apply_brush_at_source(source_xy)

    def mouseMoveEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        if self._edit_mode is None:
            super().mouseMoveEvent(event)
            return
        source_xy = self.map_widget_point_to_source(_event_position(event))
        self._cursor_source_xy = source_xy
        if source_xy is not None and self._is_painting and _event_left_button_down(event):
            self._apply_brush_at_source(source_xy)
            return
        self.update()

    def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        if self._edit_mode is None or _event_button(event) != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._is_painting = False
        self._cursor_source_xy = self.map_widget_point_to_source(_event_position(event))
        self.update()

    def leaveEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        self._cursor_source_xy = None
        self._is_painting = False
        self.update()
        super().leaveEvent(event)

    def _ensure_draft_mask_for_current_source(self) -> None:
        source_size = self.source_image_size()
        if source_size is None:
            self._draft_mask = None
            self._draft_overlay_pixmap = None
            return
        width, height = source_size
        expected_shape = (height, width)
        if self._draft_mask is not None and self._draft_mask.shape == expected_shape:
            return
        if (
            self._committed_mask is not None
            and self._committed_mask_size == source_size
            and self._committed_mask.shape == expected_shape
        ):
            self._draft_mask = np.array(self._committed_mask, dtype=bool, copy=True)
        else:
            self._draft_mask = np.zeros(expected_shape, dtype=bool)
        self._draft_overlay_pixmap = None

    def _apply_brush_at_source(self, source_xy: tuple[int, int]) -> None:
        if self._edit_mode is None:
            return
        self._ensure_draft_mask_for_current_source()
        if self._draft_mask is None:
            return
        source_x, source_y = source_xy
        height, width = self._draft_mask.shape
        radius = max(0.5, float(self._brush_diameter_px) / 2.0)
        x1 = max(0, int(np.floor(float(source_x) - radius)))
        y1 = max(0, int(np.floor(float(source_y) - radius)))
        x2 = min(width - 1, int(np.ceil(float(source_x) + radius)))
        y2 = min(height - 1, int(np.ceil(float(source_y) + radius)))
        if x2 < x1 or y2 < y1:
            return
        yy, xx = np.ogrid[y1 : y2 + 1, x1 : x2 + 1]
        circle = (xx - source_x) ** 2 + (yy - source_y) ** 2 <= radius**2
        self._draft_mask[y1 : y2 + 1, x1 : x2 + 1][circle] = self._edit_mode == "draw"
        self._draft_overlay_pixmap = None
        self.update()

    def _draw_mask_overlay(self, painter: QPainter) -> None:
        target_rect = self.image_target_rect()
        if target_rect.isNull():
            return
        if self._committed_mask_matches_current_source():
            if self._committed_overlay_pixmap is None and self._committed_mask is not None:
                self._committed_overlay_pixmap = _mask_overlay_pixmap(
                    self._committed_mask,
                    rgba=(255, 80, 80, 70),
                )
            if self._committed_overlay_pixmap is not None:
                painter.drawPixmap(
                    target_rect,
                    self._committed_overlay_pixmap,
                    QRectF(self._committed_overlay_pixmap.rect()),
                )
        if self._edit_mode is not None and self._draft_mask is not None:
            if self._draft_overlay_pixmap is None:
                self._draft_overlay_pixmap = _mask_overlay_pixmap(
                    self._draft_mask,
                    rgba=(255, 210, 55, 110),
                )
            painter.drawPixmap(
                target_rect,
                self._draft_overlay_pixmap,
                QRectF(self._draft_overlay_pixmap.rect()),
            )

    def _committed_mask_matches_current_source(self) -> bool:
        source_size = self.source_image_size()
        if self._committed_mask is None or self._committed_mask_size is None:
            return False
        if source_size != self._committed_mask_size:
            return False
        width, height = source_size
        return self._committed_mask.shape == (height, width)

    def _effective_manual_mask(self) -> np.ndarray | None:
        if not self._committed_mask_matches_current_source():
            return None
        if self._committed_mask is None:
            return None
        return self._committed_mask

    def _refresh_effective_pixmap(self) -> None:
        if self._raw_pixmap is None or self._raw_pixmap.isNull():
            self._pixmap = None
            return
        manual_mask = self._effective_manual_mask()
        if manual_mask is None and not self._background_removal_matches_current_source():
            self._pixmap = QPixmap(self._raw_pixmap)
            return

        raw_rgb = self._raw_rgb_array()
        if raw_rgb is None:
            self._pixmap = QPixmap(self._raw_pixmap)
            return
        source_h, source_w = int(raw_rgb.shape[0]), int(raw_rgb.shape[1])

        background_mask: np.ndarray | None = None
        raw_gray = self._raw_gray_array()
        if raw_gray is not None:
            background_result = compute_background_removal_mask(
                raw_gray,
                self._background_snapshot,
            )
            background_mask = background_result.mask

        if manual_mask is None and background_mask is None:
            self._pixmap = QPixmap(self._raw_pixmap)
            return

        combined_mask = combine_ignore_masks(
            shape=(source_h, source_w),
            manual_mask=manual_mask,
            background_mask=background_mask,
        )
        effective_rgb = apply_fill_to_mask(
            raw_rgb,
            combined_mask,
            fill_value=self._mask_fill_value,
        )
        self._pixmap = _rgb_array_to_pixmap(effective_rgb)

    def _raw_rgb_array(self) -> np.ndarray | None:
        if self._raw_rgb is None and self._raw_image is not None and not self._raw_image.isNull():
            self._raw_rgb = _qimage_to_rgb_array(self._raw_image)
        return self._raw_rgb

    def _raw_gray_array(self) -> np.ndarray | None:
        if self._raw_gray is None and self._raw_image is not None and not self._raw_image.isNull():
            self._raw_gray = _qimage_to_gray_array(self._raw_image)
        return self._raw_gray

    def _background_removal_matches_current_source(self) -> bool:
        snapshot = self._background_snapshot
        source_size = self.source_image_size()
        if snapshot is None or source_size is None:
            return False
        return (
            bool(snapshot.captured)
            and bool(snapshot.enabled)
            and snapshot.dimensions_match(source_size[0], source_size[1])
        )

    def _draw_brush_cursor(self, painter: QPainter) -> None:
        if self._edit_mode is None or self._cursor_source_xy is None:
            return
        center = self.map_source_point_to_widget(
            (float(self._cursor_source_xy[0]), float(self._cursor_source_xy[1])),
            self.source_image_size(),
        )
        rect = self.image_target_rect()
        source_size = self.source_image_size()
        if center is None or rect.isNull() or source_size is None:
            return
        source_w, _ = source_size
        if source_w <= 0:
            return
        scale = rect.width() / float(source_w)
        radius = max(0.5, float(self._brush_diameter_px) * scale / 2.0)
        pen = QPen(QColor(255, 255, 255), 1.5)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawEllipse(center, radius, radius)

    def _draw_heatmap_overlay(self, painter: QPainter) -> None:
        if not self._heatmap_overlay_enabled:
            return
        overlay = self._overlay
        if overlay is None or overlay.roi_fcn_heatmap is None:
            return
        source_size = overlay.source_image_wh_px
        if not self.overlay_source_size_matches(source_size):
            return
        target_rect = self.image_target_rect()
        if target_rect.isNull():
            return
        source_w, source_h = _valid_source_size(source_size)
        if source_w <= 0 or source_h <= 0:
            return
        heatmap_pixmap = self._heatmap_overlay_pixmap(
            overlay.roi_fcn_heatmap,
            source_wh_px=(source_w, source_h),
        )
        if heatmap_pixmap is None or heatmap_pixmap.isNull():
            return
        painter.drawPixmap(
            target_rect,
            heatmap_pixmap,
            QRectF(heatmap_pixmap.rect()),
        )

    def _heatmap_overlay_pixmap(
        self,
        heatmap: FramePreviewHeatmapOverlay,
        *,
        source_wh_px: tuple[int, int],
    ) -> QPixmap | None:
        source_w, source_h = _valid_source_size(source_wh_px)
        canvas_w, canvas_h = _valid_source_size(heatmap.canvas_wh_px)
        resized_w, resized_h = _valid_source_size(heatmap.resized_image_wh_px)
        padding = _valid_padding_ltrb(heatmap.padding_ltrb_px)
        if (
            source_w <= 0
            or source_h <= 0
            or canvas_w <= 0
            or canvas_h <= 0
            or resized_w <= 0
            or resized_h <= 0
            or padding is None
        ):
            return None
        cache_key = (
            source_w,
            source_h,
            canvas_w,
            canvas_h,
            resized_w,
            resized_h,
            *padding,
            int(heatmap.heatmap_u8.shape[1]),
            int(heatmap.heatmap_u8.shape[0]),
        )
        cached = self._heatmap_overlay_pixmap_cache
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        source_heatmap = _project_locator_heatmap_to_source(
            heatmap.heatmap_u8,
            source_wh_px=(source_w, source_h),
            canvas_wh_px=(canvas_w, canvas_h),
            resized_image_wh_px=(resized_w, resized_h),
            padding_ltrb_px=padding,
        )
        if source_heatmap is None:
            return None
        pixmap = _heatmap_overlay_pixmap(source_heatmap)
        self._heatmap_overlay_pixmap_cache = (cache_key, pixmap)
        return pixmap

    def _draw_overlay(self, painter: QPainter) -> None:
        overlay = self._overlay
        if overlay is None or not self.overlay_source_size_matches(overlay.source_image_wh_px):
            return

        if overlay.roi_bounds_xyxy_px is not None:
            roi_rect = self.map_source_rect_to_widget(
                overlay.roi_bounds_xyxy_px,
                overlay.source_image_wh_px,
            )
            if roi_rect is not None and not roi_rect.isNull():
                pen = QPen(QColor(255, 176, 62), 2.0)
                pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawRect(roi_rect)

        label_anchor: QPointF | None = None
        if overlay.bbox_xyxy_px is not None:
            bbox_rect = self.map_source_rect_to_widget(
                overlay.bbox_xyxy_px,
                overlay.source_image_wh_px,
            )
            if bbox_rect is not None and not bbox_rect.isNull():
                painter.setPen(QPen(QColor(74, 222, 128), 2.5))
                painter.drawRect(bbox_rect)
                label_anchor = bbox_rect.topLeft()

        if overlay.center_xy_px is not None:
            center = self.map_source_point_to_widget(
                overlay.center_xy_px,
                overlay.source_image_wh_px,
            )
            if center is not None:
                painter.setPen(QPen(QColor(248, 113, 113), 2.0))
                radius = 7.0
                painter.drawLine(
                    QPointF(center.x() - radius, center.y()),
                    QPointF(center.x() + radius, center.y()),
                )
                painter.drawLine(
                    QPointF(center.x(), center.y() - radius),
                    QPointF(center.x(), center.y() + radius),
                )
                if label_anchor is None:
                    label_anchor = center

        if label_anchor is not None and overlay.label:
            painter.setFont(QFont(self.font().family(), 10))
            painter.setPen(QColor(230, 255, 238))
            painter.drawText(
                QPointF(max(4.0, label_anchor.x()), max(16.0, label_anchor.y() - 6.0)),
                overlay.label,
            )


def _valid_source_size(source_image_wh_px: tuple[int, int] | None) -> tuple[int, int]:
    if source_image_wh_px is None:
        return 0, 0
    try:
        width, height = source_image_wh_px
    except (TypeError, ValueError):
        return 0, 0
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        return 0, 0


def _valid_padding_ltrb(
    padding_ltrb_px: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if padding_ltrb_px is None:
        return None
    try:
        left, top, right, bottom = padding_ltrb_px
        parsed = (int(left), int(top), int(right), int(bottom))
    except (TypeError, ValueError):
        return None
    if min(parsed) < 0:
        return None
    return parsed


def _point_xy(point: QPointF | tuple[float, float]) -> tuple[float, float]:
    if isinstance(point, QPointF):
        return float(point.x()), float(point.y())
    try:
        x, y = point
    except (TypeError, ValueError):
        return 0.0, 0.0
    return float(x), float(y)


def _event_position(event: object) -> QPointF:
    position = getattr(event, "position", None)
    if callable(position):
        return position()
    pos = getattr(event, "pos", None)
    if callable(pos):
        point = pos()
        return QPointF(float(point.x()), float(point.y()))
    return QPointF()


def _event_button(event: object) -> Qt.MouseButton:
    button = getattr(event, "button", None)
    if callable(button):
        return button()
    return Qt.MouseButton.NoButton


def _event_left_button_down(event: object) -> bool:
    buttons = getattr(event, "buttons", None)
    if not callable(buttons):
        return False
    return bool(buttons() & Qt.MouseButton.LeftButton)


def _mask_overlay_pixmap(mask: np.ndarray, *, rgba: tuple[int, int, int, int]) -> QPixmap:
    mask_array = np.asarray(mask, dtype=bool)
    height, width = mask_array.shape
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[mask_array] = np.asarray(rgba, dtype=np.uint8)
    image = QImage(
        overlay.data,
        width,
        height,
        int(overlay.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()
    return QPixmap.fromImage(image)


def _scaled_image_to_fit(
    image: QImage,
    *,
    max_width_px: int,
    max_height_px: int,
) -> QImage:
    max_width = max(1, int(max_width_px))
    max_height = max(1, int(max_height_px))
    width = int(image.width())
    height = int(image.height())
    if width <= 0 or height <= 0:
        return image.copy()
    if width <= max_width and height <= max_height:
        return image.copy()
    return image.scaled(
        max_width,
        max_height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )


def _scaled_background_snapshot(
    snapshot: BackgroundSnapshot | None,
    *,
    original_size: tuple[int, int],
    preview_size: tuple[int, int],
) -> BackgroundSnapshot | None:
    if snapshot is None or not snapshot.captured or not snapshot.enabled:
        return None
    original_w, original_h = (int(original_size[0]), int(original_size[1]))
    preview_w, preview_h = (int(preview_size[0]), int(preview_size[1]))
    if preview_w <= 0 or preview_h <= 0:
        return None
    if not snapshot.dimensions_match(original_w, original_h):
        return None

    background_image = _gray_array_to_qimage(snapshot.grayscale_background)
    scaled_background = background_image.scaled(
        preview_w,
        preview_h,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )
    return BackgroundSnapshot(
        revision=int(snapshot.revision),
        width_px=preview_w,
        height_px=preview_h,
        grayscale_background=_qimage_to_gray_array(scaled_background),
        enabled=bool(snapshot.enabled),
        threshold=int(snapshot.threshold),
        captured_at_utc=snapshot.captured_at_utc,
    )


def _project_locator_heatmap_to_source(
    heatmap_u8: np.ndarray,
    *,
    source_wh_px: tuple[int, int],
    canvas_wh_px: tuple[int, int],
    resized_image_wh_px: tuple[int, int],
    padding_ltrb_px: tuple[int, int, int, int],
) -> np.ndarray | None:
    source_w, source_h = source_wh_px
    canvas_w, canvas_h = canvas_wh_px
    resized_w, resized_h = resized_image_wh_px
    pad_left, pad_top, _, _ = padding_ltrb_px
    if min(source_w, source_h, canvas_w, canvas_h, resized_w, resized_h) <= 0:
        return None

    heatmap = np.asarray(heatmap_u8, dtype=np.uint8)
    if heatmap.ndim != 2 or heatmap.size <= 0:
        return None

    heatmap_image = _gray_array_to_qimage(heatmap)
    if (int(heatmap.shape[1]), int(heatmap.shape[0])) != (canvas_w, canvas_h):
        heatmap_image = heatmap_image.scaled(
            canvas_w,
            canvas_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    canvas_heatmap = _qimage_to_gray_array(heatmap_image)

    x1 = min(canvas_w, max(0, int(pad_left)))
    y1 = min(canvas_h, max(0, int(pad_top)))
    x2 = min(canvas_w, x1 + int(resized_w))
    y2 = min(canvas_h, y1 + int(resized_h))
    if x2 <= x1 or y2 <= y1:
        return None

    unpadded = np.ascontiguousarray(canvas_heatmap[y1:y2, x1:x2])
    unpadded_image = _gray_array_to_qimage(unpadded)
    source_image = unpadded_image.scaled(
        source_w,
        source_h,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    return _qimage_to_gray_array(source_image)


def _heatmap_overlay_pixmap(heatmap_u8: np.ndarray) -> QPixmap:
    heatmap = np.asarray(heatmap_u8, dtype=np.uint8)
    height, width = int(heatmap.shape[0]), int(heatmap.shape[1])
    normalized = heatmap.astype(np.float32) / 255.0
    alpha = np.rint(np.power(normalized, 0.65) * 160.0).astype(np.uint8)
    alpha[heatmap == 0] = 0

    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[..., 0] = 255
    overlay[..., 1] = np.rint(48.0 + normalized * 180.0).astype(np.uint8)
    overlay[..., 2] = 32
    overlay[..., 3] = alpha
    image = QImage(
        overlay.data,
        width,
        height,
        int(overlay.strides[0]),
        QImage.Format.Format_RGBA8888,
    ).copy()
    return QPixmap.fromImage(image)


def _background_snapshot_key(
    snapshot: BackgroundSnapshot | None,
) -> tuple[int, bool, bool, int, int, int] | None:
    if snapshot is None:
        return None
    return (
        int(snapshot.revision),
        bool(snapshot.captured),
        bool(snapshot.enabled),
        int(snapshot.threshold),
        int(snapshot.width_px),
        int(snapshot.height_px),
    )


def _qimage_to_rgb_array(image: QImage) -> np.ndarray:
    rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
    width = int(rgb_image.width())
    height = int(rgb_image.height())
    bytes_per_line = int(rgb_image.bytesPerLine())
    buffer = rgb_image.bits()
    array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, bytes_per_line)
    return np.array(array[:, : width * 3].reshape(height, width, 3), copy=True)


def _qimage_to_gray_array(image: QImage) -> np.ndarray:
    gray_image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    width = int(gray_image.width())
    height = int(gray_image.height())
    bytes_per_line = int(gray_image.bytesPerLine())
    buffer = gray_image.bits()
    array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, bytes_per_line)
    return np.array(array[:, :width], dtype=np.uint8, copy=True)


def _gray_array_to_qimage(image: np.ndarray) -> QImage:
    gray = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if gray.ndim != 2:
        raise ValueError(f"Expected grayscale image with shape (H, W), got {gray.shape}.")
    height, width = int(gray.shape[0]), int(gray.shape[1])
    qimage = QImage(
        gray.data,
        width,
        height,
        int(gray.strides[0]),
        QImage.Format.Format_Grayscale8,
    ).copy()
    return qimage


def _rgb_array_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb.shape}.")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    qimage = QImage(
        rgb.data,
        width,
        height,
        int(rgb.strides[0]),
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qimage)


__all__ = [
    "FrameMaskEditResult",
    "FramePreviewHeatmapOverlay",
    "FramePreviewOverlay",
    "FramePreviewWidget",
]
