"""Live preview widget for camera frames and detection overlays."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy

from rb_camera_calibration.contracts import CameraFrame, CharucoDetection
from rb_camera_calibration.utils import opencv_compat as cvx


class PreviewWidget(QLabel):
    """QLabel-based preview that renders encoded frame payloads."""

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #101418; color: #d8dee9;")
        self.setText("Camera preview")
        self._latest_frame: CameraFrame | None = None
        self._latest_detection: CharucoDetection | None = None

    def set_frame(self, frame: CameraFrame) -> None:
        self._latest_frame = frame
        self._render()

    def set_detection(self, detection: CharucoDetection) -> None:
        self._latest_detection = detection
        self._render()

    def _render(self) -> None:
        if self._latest_frame is None:
            return
        cv2 = cvx.import_cv2()
        image = cvx.decode_image_bytes(self._latest_frame.image_bytes)
        if self._latest_detection is not None:
            image = cvx.draw_detection_overlay(image, self._latest_detection.extras)
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        bytes_per_line = int(rgb.strides[0])
        qimage = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._render()
