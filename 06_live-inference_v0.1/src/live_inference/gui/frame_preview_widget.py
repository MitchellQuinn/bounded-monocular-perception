"""Aspect-preserving live frame preview with pipeline ROI overlay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget


@dataclass(frozen=True)
class FramePreviewOverlay:
    """Source-image-space geometry to draw over the preview."""

    source_image_wh_px: tuple[int, int] | None = None
    bbox_xyxy_px: tuple[float, float, float, float] | None = None
    center_xy_px: tuple[float, float] | None = None
    roi_bounds_xyxy_px: tuple[float, float, float, float] | None = None
    label: str = "Pipeline ROI / bbox"


class FramePreviewWidget(QWidget):
    """Display one QPixmap and draw source-pixel ROI geometry over it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder_text = "No frame yet"
        self._overlay: FramePreviewOverlay | None = None

    def placeholder_text(self) -> str:
        return self._placeholder_text

    def set_placeholder_text(self, text: str) -> None:
        self._placeholder_text = str(text)
        self.update()

    def load_image(self, path: Path | str) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return False
        self.set_pixmap(pixmap)
        return True

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = QPixmap(pixmap)
        self.update()

    def pixmap(self) -> QPixmap | None:
        return self._pixmap

    def source_image_size(self) -> tuple[int, int] | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        return int(self._pixmap.width()), int(self._pixmap.height())

    def set_overlay(self, overlay: FramePreviewOverlay | None) -> None:
        self._overlay = overlay
        self.update()

    def overlay(self) -> FramePreviewOverlay | None:
        return self._overlay

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
        self._draw_overlay(painter)

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


__all__ = ["FramePreviewOverlay", "FramePreviewWidget"]
