"""OpenCV ``VideoCapture`` camera source."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from rb_camera_calibration.contracts import CameraCaptureConfig, CameraFrame, FrameMetadata
from rb_camera_calibration.utils import opencv_compat as cvx
from rb_camera_calibration.utils.hashing import hash_bytes
from rb_camera_calibration.utils.timestamps import utc_now_iso

LOGGER = logging.getLogger(__name__)


class OpenCvCameraSource:
    """Read frames from an OpenCV camera device and emit contract payloads."""

    def __init__(self, config: CameraCaptureConfig) -> None:
        self.config = config
        self._capture: Any | None = None
        self._sequence_index = 0
        self.actual_properties: dict[str, Any] = {}

    def start(self) -> None:
        cv2 = cvx.import_cv2()
        backend = _backend_id(self.config.backend)
        device = _camera_device_value(self.config.camera_device)
        capture = cv2.VideoCapture(device, backend) if backend is not None else cv2.VideoCapture(device)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera device {self.config.camera_device!r}.")
        self._capture = capture
        self._apply_requested_properties(capture)
        self.actual_properties = self._read_actual_properties(capture)
        LOGGER.info("Opened camera %s with properties %s", self.config.camera_device, self.actual_properties)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read_frame(self) -> CameraFrame | None:
        if self._capture is None:
            raise RuntimeError("Camera source has not been started.")
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        self._sequence_index += 1
        encoded = cvx.encode_png(image)
        frame_hash = hash_bytes(encoded)
        frame_id = f"frame-{self._sequence_index:08d}-{uuid.uuid4().hex[:8]}"
        metadata = FrameMetadata(
            frame_id=frame_id,
            sequence_index=self._sequence_index,
            captured_at_utc=utc_now_iso(),
            width_px=int(image.shape[1]),
            height_px=int(image.shape[0]),
            pixel_format=self.config.pixel_format,
            source_name=str(self.config.camera_device),
            extras={"actual_camera_properties": self.actual_properties},
        )
        return CameraFrame(
            frame_id=frame_id,
            metadata=metadata,
            frame_hash=frame_hash,
            image_bytes=encoded,
        )

    def _apply_requested_properties(self, capture: Any) -> None:
        cv2 = cvx.import_cv2()
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width_px))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height_px))
        capture.set(cv2.CAP_PROP_FPS, float(self.config.fps))
        if self.config.pixel_format:
            fourcc = cv2.VideoWriter_fourcc(*self.config.pixel_format[:4])
            capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        if self.config.exposure is not None:
            capture.set(cv2.CAP_PROP_EXPOSURE, float(self.config.exposure))
        if self.config.gain is not None:
            capture.set(cv2.CAP_PROP_GAIN, float(self.config.gain))
        if self.config.focus is not None and hasattr(cv2, "CAP_PROP_FOCUS"):
            capture.set(cv2.CAP_PROP_FOCUS, float(self.config.focus))
        if self.config.auto_white_balance is not None and hasattr(cv2, "CAP_PROP_AUTO_WB"):
            capture.set(cv2.CAP_PROP_AUTO_WB, 1.0 if self.config.auto_white_balance else 0.0)
        if self.config.auto_exposure is not None and hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
            # V4L2 convention: 1 manual, 3 aperture priority/auto.
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3.0 if self.config.auto_exposure else 1.0)

    def _read_actual_properties(self, capture: Any) -> dict[str, Any]:
        cv2 = cvx.import_cv2()
        fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> 8 * i) & 0xFF) for i in range(4)).strip()
        properties = {
            "width_px": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height_px": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "fourcc": fourcc,
            "backend": self.config.backend,
        }
        for key, prop_name in {
            "exposure": "CAP_PROP_EXPOSURE",
            "gain": "CAP_PROP_GAIN",
            "focus": "CAP_PROP_FOCUS",
            "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
            "auto_white_balance": "CAP_PROP_AUTO_WB",
        }.items():
            if hasattr(cv2, prop_name):
                properties[key] = float(capture.get(getattr(cv2, prop_name)))
        return properties


def _backend_id(backend: str) -> int | None:
    cv2 = cvx.import_cv2()
    normalized = backend.strip().upper()
    if normalized in {"", "ANY", "GENERIC"}:
        return None
    if normalized == "V4L2":
        return int(cv2.CAP_V4L2)
    attr = f"CAP_{normalized}"
    return int(getattr(cv2, attr)) if hasattr(cv2, attr) else None


def _camera_device_value(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped
