"""OpenCV/V4L2 camera publisher for live inference frame handoff."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import logging
import shutil
import subprocess
from typing import Any

from interfaces import FrameMetadata, FrameReference
from live_inference.frame_handoff import AtomicFrameHandoffWriter


LOGGER = logging.getLogger(__name__)
DEFAULT_CAMERA_NAME = "Arducam B0495 AR0234"
DEFAULT_DEVICE = "/dev/video0"
DEFAULT_WIDTH_PX = 960
DEFAULT_HEIGHT_PX = 600
DEFAULT_FPS = 80
DEFAULT_PIXEL_FORMAT = "YUYV"
DEFAULT_ENCODING = "png"
BACKEND_LABEL = "opencv_v4l2"


class CameraOpenError(RuntimeError):
    """Raised when the OpenCV/V4L2 camera cannot be opened."""


class FrameCaptureError(RuntimeError):
    """Raised when the camera fails to return a frame."""


class FrameEncodingError(RuntimeError):
    """Raised when a captured frame cannot be encoded for handoff."""


class OpenCvV4L2CameraPublisher:
    """Capture OpenCV frames and publish encoded bytes via atomic handoff."""

    def __init__(
        self,
        *,
        handoff_writer: AtomicFrameHandoffWriter,
        device: str | int = DEFAULT_DEVICE,
        width_px: int = DEFAULT_WIDTH_PX,
        height_px: int = DEFAULT_HEIGHT_PX,
        fps: int = DEFAULT_FPS,
        pixel_format: str = DEFAULT_PIXEL_FORMAT,
        encoding: str = DEFAULT_ENCODING,
        camera_name: str = DEFAULT_CAMERA_NAME,
        capture_backend: int | None = None,
        auto_open: bool = True,
        now_utc_fn: Callable[[], str] | None = None,
        log_v4l2_controls_at_startup: bool = False,
        cv2_module: Any | None = None,
    ) -> None:
        self.writer = handoff_writer
        self.device = device
        self.width_px = _positive_int(width_px, "width_px")
        self.height_px = _positive_int(height_px, "height_px")
        self.fps = _positive_int(fps, "fps")
        self.pixel_format = _normalize_non_empty_text(pixel_format, "pixel_format").upper()
        self.encoding = _normalize_encoding(encoding)
        self.camera_name = _normalize_non_empty_text(camera_name, "camera_name")
        self.capture_backend = capture_backend
        self.now_utc_fn = now_utc_fn or _utc_now_iso
        self.log_v4l2_controls_at_startup = bool(log_v4l2_controls_at_startup)
        self._cv2 = cv2_module
        self._capture: Any | None = None
        self._frame_index = 0
        self._actual_properties: dict[str, Any] = {}
        self._v4l2_controls_snapshot: str | None = None

        if auto_open:
            self.open()

    def open(self) -> None:
        """Open and configure the V4L2 camera if it is not already open."""
        if self._capture is not None:
            return

        cv2 = self._cv2 or _import_cv2()
        self._cv2 = cv2
        backend = self.capture_backend if self.capture_backend is not None else cv2.CAP_V4L2
        try:
            capture = cv2.VideoCapture(self.device, backend)
        except Exception as exc:
            raise CameraOpenError(
                f"Could not open OpenCV V4L2 camera device {self.device!r}: {exc}"
            ) from exc
        if not _capture_is_opened(capture):
            _release_capture(capture)
            raise CameraOpenError(
                f"Could not open OpenCV V4L2 camera device {self.device!r}."
            )

        self._capture = capture
        self._configure_capture(capture, cv2)
        self._actual_properties = self._read_actual_properties(capture, cv2)
        self._log_startup_diagnostics()

    def close(self) -> None:
        """Release the OpenCV capture handle."""
        capture = self._capture
        self._capture = None
        if capture is not None:
            _release_capture(capture)

    def publish_next(self) -> FrameReference:
        """Capture, encode, and publish one frame."""
        if self._capture is None:
            self.open()
        if self._capture is None:
            raise CameraOpenError(
                f"OpenCV V4L2 camera device {self.device!r} is not open."
            )

        try:
            success, frame = self._capture.read()
        except Exception as exc:
            raise FrameCaptureError(
                f"Failed to capture frame from OpenCV V4L2 camera device {self.device!r}: {exc}"
            ) from exc
        if not success or frame is None:
            raise FrameCaptureError(
                f"Failed to capture frame from OpenCV V4L2 camera device {self.device!r}."
            )

        captured_at_utc = self.now_utc_fn()
        image_bytes = self._encode_frame(frame)
        width_px, height_px = _frame_dimensions(frame)
        metadata = FrameMetadata(
            frame_id=f"opencv-v4l2-{self._frame_index:012d}",
            source_name=self.camera_name,
            captured_at_utc=captured_at_utc,
            width_px=width_px,
            height_px=height_px,
            pixel_format=self.pixel_format,
            encoding=self.encoding,
            byte_size=len(image_bytes),
            extras=self._metadata_extras(),
        )
        frame_reference = self.writer.publish_frame(image_bytes, metadata)
        self._frame_index += 1
        return frame_reference

    def _configure_capture(self, capture: Any, cv2: Any) -> None:
        _set_capture_property(
            capture,
            getattr(cv2, "CAP_PROP_FRAME_WIDTH", None),
            self.width_px,
        )
        _set_capture_property(
            capture,
            getattr(cv2, "CAP_PROP_FRAME_HEIGHT", None),
            self.height_px,
        )
        _set_capture_property(capture, getattr(cv2, "CAP_PROP_FPS", None), self.fps)
        fourcc_property = getattr(cv2, "CAP_PROP_FOURCC", None)
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
        if (
            fourcc_property is not None
            and callable(fourcc_fn)
            and len(self.pixel_format) == 4
        ):
            fourcc = fourcc_fn(*self.pixel_format)
            _set_capture_property(capture, fourcc_property, fourcc)

    def _read_actual_properties(self, capture: Any, cv2: Any) -> dict[str, Any]:
        actual: dict[str, Any] = {}
        property_names = {
            "actual_width_px": "CAP_PROP_FRAME_WIDTH",
            "actual_height_px": "CAP_PROP_FRAME_HEIGHT",
            "actual_fps": "CAP_PROP_FPS",
            "actual_fourcc": "CAP_PROP_FOURCC",
        }
        for metadata_name, cv2_name in property_names.items():
            prop_id = getattr(cv2, cv2_name, None)
            value = _get_capture_property(capture, prop_id)
            if value is not None:
                actual[metadata_name] = _plain_number(value)

        fourcc = actual.get("actual_fourcc")
        if isinstance(fourcc, int):
            actual["actual_pixel_format"] = _decode_fourcc(fourcc)
        return actual

    def _encode_frame(self, frame: Any) -> bytes:
        cv2 = self._cv2 or _import_cv2()
        try:
            success, encoded = cv2.imencode(f".{self.encoding}", frame)
        except Exception as exc:
            raise FrameEncodingError(
                f"Failed to encode OpenCV V4L2 frame as {self.encoding!r}: {exc}"
            ) from exc
        if not success:
            raise FrameEncodingError(
                f"Failed to encode OpenCV V4L2 frame as {self.encoding!r}."
            )
        try:
            return bytes(encoded.tobytes())
        except AttributeError:
            return bytes(encoded)

    def _metadata_extras(self) -> dict[str, Any]:
        extras: dict[str, Any] = {
            "device": self.device,
            "requested_width_px": self.width_px,
            "requested_height_px": self.height_px,
            "requested_fps": self.fps,
            "requested_pixel_format": self.pixel_format,
            "backend": BACKEND_LABEL,
        }
        extras.update(self._actual_properties)
        return extras

    def _log_startup_diagnostics(self) -> None:
        LOGGER.info(
            "Opened %s on %r via %s: requested %sx%s @ %s fps %s; actual=%s",
            self.camera_name,
            self.device,
            BACKEND_LABEL,
            self.width_px,
            self.height_px,
            self.fps,
            self.pixel_format,
            self._actual_properties,
        )
        if not self.log_v4l2_controls_at_startup:
            return
        self._v4l2_controls_snapshot = _capture_v4l2_controls(self.device)
        if self._v4l2_controls_snapshot:
            LOGGER.info(
                "v4l2-ctl startup diagnostics for %r:\n%s",
                self.device,
                self._v4l2_controls_snapshot,
            )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _import_cv2() -> Any:
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:
        raise CameraOpenError("OpenCV cv2 is required for OpenCvV4L2CameraPublisher.") from exc
    return cv2


def _capture_is_opened(capture: Any) -> bool:
    is_opened = getattr(capture, "isOpened", None)
    return bool(is_opened()) if callable(is_opened) else False


def _release_capture(capture: Any) -> None:
    release = getattr(capture, "release", None)
    if callable(release):
        release()


def _set_capture_property(capture: Any, prop_id: Any, value: Any) -> None:
    if prop_id is None:
        return
    set_property = getattr(capture, "set", None)
    if callable(set_property):
        try:
            set_property(prop_id, value)
        except Exception as exc:
            LOGGER.warning(
                "OpenCV capture property set failed for %r=%r: %s",
                prop_id,
                value,
                exc,
            )


def _get_capture_property(capture: Any, prop_id: Any) -> Any | None:
    if prop_id is None:
        return None
    get_property = getattr(capture, "get", None)
    if not callable(get_property):
        return None
    try:
        value = get_property(prop_id)
    except Exception:
        return None
    if value is None:
        return None
    return value


def _plain_number(value: Any) -> int | float:
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _decode_fourcc(value: int) -> str:
    chars = []
    for shift in (0, 8, 16, 24):
        code = (value >> shift) & 0xFF
        if code:
            chars.append(chr(code))
    return "".join(chars)


def _frame_dimensions(frame: Any) -> tuple[int | None, int | None]:
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        return None, None
    return int(shape[1]), int(shape[0])


def _positive_int(value: int, label: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be > 0; got {value!r}.")
    return number


def _normalize_non_empty_text(value: str, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be empty.")
    return text


def _normalize_encoding(value: str) -> str:
    text = _normalize_non_empty_text(value, "encoding").lower().lstrip(".")
    if not text:
        raise ValueError("encoding must not be empty.")
    return text


def _capture_v4l2_controls(device: str | int) -> str | None:
    if isinstance(device, int):
        return None
    if shutil.which("v4l2-ctl") is None:
        LOGGER.info("v4l2-ctl is not available; skipping V4L2 startup diagnostics.")
        return None
    try:
        completed = subprocess.run(
            ["v4l2-ctl", "-d", str(device), "--all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.warning("Could not capture v4l2-ctl diagnostics for %r: %s", device, exc)
        return None
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        LOGGER.warning(
            "v4l2-ctl diagnostics failed for %r with exit code %s: %s",
            device,
            completed.returncode,
            (completed.stderr or "").strip(),
        )
    return output or None


__all__ = [
    "BACKEND_LABEL",
    "DEFAULT_CAMERA_NAME",
    "DEFAULT_DEVICE",
    "DEFAULT_ENCODING",
    "DEFAULT_FPS",
    "DEFAULT_HEIGHT_PX",
    "DEFAULT_PIXEL_FORMAT",
    "DEFAULT_WIDTH_PX",
    "CameraOpenError",
    "FrameCaptureError",
    "FrameEncodingError",
    "OpenCvV4L2CameraPublisher",
]
