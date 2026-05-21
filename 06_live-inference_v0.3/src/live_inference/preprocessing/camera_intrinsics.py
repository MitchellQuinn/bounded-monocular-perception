"""Runtime camera intrinsics transforms for live preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import json
from pathlib import Path
from threading import RLock
from typing import Any

import cv2
import numpy as np

import interfaces.contracts as contracts


CAMERA_INTRINSICS_MODE_DISABLED = contracts.CAMERA_INTRINSICS_MODE_DISABLED
CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP = (
    contracts.CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP
)
CAMERA_INTRINSICS_MODE_REAL_UNDISTORT_ONLY = (
    contracts.CAMERA_INTRINSICS_MODE_REAL_UNDISTORT_ONLY
)
SUPPORTED_CAMERA_INTRINSICS_MODES = contracts.SUPPORTED_CAMERA_INTRINSICS_MODES

CAMERA_INTRINSICS_MODE_LABELS = {
    CAMERA_INTRINSICS_MODE_DISABLED: "Disabled",
    CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP: (
        "Real -> Unity intrinsics"
    ),
    CAMERA_INTRINSICS_MODE_REAL_UNDISTORT_ONLY: "Real undistort only",
}

CAMERA_INTRINSICS_METADATA_MODE = (
    contracts.PREPROCESSING_METADATA_CAMERA_INTRINSICS_MODE
)
CAMERA_INTRINSICS_METADATA_REVISION = (
    contracts.PREPROCESSING_METADATA_CAMERA_INTRINSICS_REVISION
)
CAMERA_INTRINSICS_METADATA_APPLIED = (
    contracts.PREPROCESSING_METADATA_CAMERA_INTRINSICS_APPLIED
)


@dataclass(frozen=True)
class CameraCalibration:
    """OpenCV pinhole calibration loaded from repository calibration artifacts."""

    camera_name: str
    image_size_wh_px: tuple[int, int]
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    artifact_path: Path | None = None

    @classmethod
    def from_json(cls, path: Path | str) -> "CameraCalibration":
        """Load a calibration artifact from ``calibration_result.json``."""
        resolved = Path(path).expanduser().resolve(strict=False)
        with resolved.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_mapping(payload, artifact_path=resolved)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        artifact_path: Path | None = None,
    ) -> "CameraCalibration":
        """Build a calibration object from an artifact-shaped mapping."""
        size = _image_size(payload.get("image_size_wh_px"))
        return cls(
            camera_name=str(payload.get("camera_name") or "camera"),
            image_size_wh_px=size,
            camera_matrix=_camera_matrix(payload.get("camera_matrix")),
            distortion_coefficients=_distortion_coefficients(
                payload.get("distortion_coefficients")
            ),
            artifact_path=artifact_path,
        )

    def scaled_camera_matrix(self, image_size_wh_px: tuple[int, int]) -> np.ndarray:
        """Return camera matrix scaled from calibration resolution to image size."""
        width_px, height_px = _positive_size(image_size_wh_px)
        calibration_w, calibration_h = _positive_size(self.image_size_wh_px)
        sx = float(width_px) / float(calibration_w)
        sy = float(height_px) / float(calibration_h)
        matrix = np.array(self.camera_matrix, dtype=np.float64, copy=True)
        matrix[0, :] *= sx
        matrix[1, :] *= sy
        matrix[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return matrix

    def metadata(self) -> dict[str, Any]:
        """Return compact serializable calibration metadata."""
        return {
            "camera_name": self.camera_name,
            "image_size_wh_px": tuple(int(value) for value in self.image_size_wh_px),
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
        }


@dataclass(frozen=True)
class CameraIntrinsicsTransformSnapshot:
    """Immutable runtime mode for calibration-backed frame transforms."""

    revision: int = 0
    camera_intrinsics_mode: str = CAMERA_INTRINSICS_MODE_DISABLED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_intrinsics_mode",
            normalize_camera_intrinsics_mode(self.camera_intrinsics_mode),
        )
        object.__setattr__(self, "revision", int(self.revision))

    @property
    def mode(self) -> str:
        return self.camera_intrinsics_mode

    def to_metadata(self) -> dict[str, Any]:
        return {
            CAMERA_INTRINSICS_METADATA_MODE: self.camera_intrinsics_mode,
            CAMERA_INTRINSICS_METADATA_REVISION: int(self.revision),
        }


class CameraIntrinsicsTransformState:
    """Lock-protected holder for the selected camera intrinsics mode."""

    def __init__(
        self,
        initial: CameraIntrinsicsTransformSnapshot | None = None,
    ) -> None:
        self._lock = RLock()
        self._snapshot = initial or CameraIntrinsicsTransformSnapshot()

    def snapshot(self) -> CameraIntrinsicsTransformSnapshot:
        with self._lock:
            return self._snapshot

    def update(
        self,
        *,
        mode: str | None = None,
        camera_intrinsics_mode: str | None = None,
    ) -> tuple[CameraIntrinsicsTransformSnapshot, int]:
        next_mode = mode if mode is not None else camera_intrinsics_mode
        if next_mode is None:
            snapshot = self.snapshot()
            return snapshot, int(snapshot.revision)

        normalized = normalize_camera_intrinsics_mode(next_mode)
        with self._lock:
            if normalized == self._snapshot.camera_intrinsics_mode:
                return self._snapshot, int(self._snapshot.revision)
            self._snapshot = replace(
                self._snapshot,
                revision=int(self._snapshot.revision) + 1,
                camera_intrinsics_mode=normalized,
            )
            return self._snapshot, int(self._snapshot.revision)

    def revision(self) -> int:
        return int(self.snapshot().revision)


@dataclass(frozen=True)
class CameraIntrinsicsTransformResult:
    """Decoded/transformed image plus encoded bytes for downstream consumers."""

    image: np.ndarray
    image_bytes: bytes
    metadata: Mapping[str, Any]


class CameraIntrinsicsFrameTransformer:
    """Apply AR0234 undistortion or AR0234-to-Unity intrinsics remapping."""

    def __init__(
        self,
        state: CameraIntrinsicsTransformState | None = None,
        *,
        real_calibration: CameraCalibration | None = None,
        unity_calibration: CameraCalibration | None = None,
        real_calibration_path: Path | str | None = None,
        unity_calibration_path: Path | str | None = None,
        interpolation: int = cv2.INTER_LINEAR,
        border_value: int = 255,
    ) -> None:
        self._state = state or CameraIntrinsicsTransformState()
        self._real_calibration = real_calibration or CameraCalibration.from_json(
            real_calibration_path or default_real_calibration_path()
        )
        self._unity_calibration = unity_calibration or CameraCalibration.from_json(
            unity_calibration_path or default_unity_calibration_path()
        )
        self._interpolation = int(interpolation)
        self._border_value = int(border_value)
        self._map_lock = RLock()
        self._map_cache: dict[
            tuple[str, tuple[int, int]], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

    @property
    def state(self) -> CameraIntrinsicsTransformState:
        return self._state

    def transform_image_bytes(
        self,
        image_bytes: bytes,
        *,
        grayscale: bool = False,
    ) -> CameraIntrinsicsTransformResult:
        """Decode, optionally remap, and re-encode image bytes."""
        decoded = _decode_image_bytes(image_bytes)
        image = _to_grayscale_uint8(decoded) if grayscale else _ensure_uint8(decoded)
        transformed, metadata = self.transform_array(image)
        if not bool(metadata[CAMERA_INTRINSICS_METADATA_APPLIED]):
            return CameraIntrinsicsTransformResult(
                image=np.ascontiguousarray(transformed),
                image_bytes=image_bytes,
                metadata=metadata,
            )
        return CameraIntrinsicsTransformResult(
            image=np.ascontiguousarray(transformed),
            image_bytes=_encode_png(transformed),
            metadata=metadata,
        )

    def transform_array(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply the current transform mode to an already-decoded image."""
        snapshot = self._state.snapshot()
        mode = snapshot.camera_intrinsics_mode
        source_wh = _array_size_wh(image)
        metadata = self._metadata(
            snapshot=snapshot,
            mode=mode,
            input_wh=source_wh,
            output_wh=source_wh,
            applied=False,
            new_camera_matrix=None,
        )
        if mode == CAMERA_INTRINSICS_MODE_DISABLED:
            return np.ascontiguousarray(image), metadata

        map_x, map_y, new_camera_matrix = self._remap_for(mode, source_wh)
        remapped = cv2.remap(
            _ensure_uint8(image),
            map_x,
            map_y,
            interpolation=self._interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_border_value(image, self._border_value),
        )
        metadata = self._metadata(
            snapshot=snapshot,
            mode=mode,
            input_wh=source_wh,
            output_wh=source_wh,
            applied=True,
            new_camera_matrix=new_camera_matrix,
        )
        return np.ascontiguousarray(remapped), metadata

    def _remap_for(
        self,
        mode: str,
        image_size_wh_px: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (mode, _positive_size(image_size_wh_px))
        with self._map_lock:
            cached = self._map_cache.get(key)
            if cached is not None:
                return cached
            source_matrix = self._real_calibration.scaled_camera_matrix(key[1])
            if mode == CAMERA_INTRINSICS_MODE_REAL_UNDISTORT_ONLY:
                new_camera_matrix = np.array(source_matrix, dtype=np.float64, copy=True)
            elif mode == CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP:
                new_camera_matrix = self._unity_calibration.scaled_camera_matrix(key[1])
            else:
                raise ValueError(f"Unsupported camera intrinsics mode: {mode!r}.")
            map_x, map_y = cv2.initUndistortRectifyMap(
                source_matrix,
                self._real_calibration.distortion_coefficients,
                np.eye(3, dtype=np.float64),
                new_camera_matrix,
                key[1],
                cv2.CV_32FC1,
            )
            cached = (map_x, map_y, new_camera_matrix)
            self._map_cache[key] = cached
            return cached

    def _metadata(
        self,
        *,
        snapshot: CameraIntrinsicsTransformSnapshot,
        mode: str,
        input_wh: tuple[int, int],
        output_wh: tuple[int, int],
        applied: bool,
        new_camera_matrix: np.ndarray | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **snapshot.to_metadata(),
            CAMERA_INTRINSICS_METADATA_MODE: mode,
            CAMERA_INTRINSICS_METADATA_APPLIED: bool(applied),
            "camera_intrinsics_input_wh_px": tuple(int(value) for value in input_wh),
            "camera_intrinsics_output_wh_px": tuple(int(value) for value in output_wh),
            "camera_intrinsics_source_calibration": self._real_calibration.metadata(),
            "camera_intrinsics_target_calibration": (
                self._unity_calibration.metadata()
                if mode == CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP
                else None
            ),
        }
        if applied:
            payload["camera_intrinsics_source_camera_matrix"] = (
                self._real_calibration.scaled_camera_matrix(input_wh).tolist()
            )
            payload["camera_intrinsics_distortion_coefficients"] = (
                self._real_calibration.distortion_coefficients.tolist()
            )
            payload["camera_intrinsics_new_camera_matrix"] = (
                new_camera_matrix.tolist() if new_camera_matrix is not None else None
            )
        return payload


def normalize_camera_intrinsics_mode(value: Any) -> str:
    """Return a canonical camera intrinsics mode string."""
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "off": CAMERA_INTRINSICS_MODE_DISABLED,
        "none": CAMERA_INTRINSICS_MODE_DISABLED,
        "disable": CAMERA_INTRINSICS_MODE_DISABLED,
        "disabled": CAMERA_INTRINSICS_MODE_DISABLED,
        "real_to_unity": CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP,
        "ar0234_to_unity": CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP,
        "real_to_unity_intrinsics": (
            CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP
        ),
        "real_to_unity_intrinsics_remap": (
            CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP
        ),
        "undistort": CAMERA_INTRINSICS_MODE_REAL_UNDISTORT_ONLY,
        "undistort_only": CAMERA_INTRINSICS_MODE_REAL_UNDISTORT_ONLY,
        "real_undistort_only": CAMERA_INTRINSICS_MODE_REAL_UNDISTORT_ONLY,
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(
        "camera_intrinsics_mode must be one of "
        f"{SUPPORTED_CAMERA_INTRINSICS_MODES!r}; got {value!r}."
    )


def default_real_calibration_path() -> Path:
    return (
        _live_project_root()
        / "config/calibration/260519-1501_calibio_charuco_30mm_a4/calibration_result.json"
    )


def default_unity_calibration_path() -> Path:
    return (
        _live_project_root()
        / "config/calibration/260520-1130_unity_ar0234_pinhole_1920x1200/calibration_result.json"
    )


def _live_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _image_size(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Calibration image_size_wh_px must be [width, height]; got {value!r}.")
    return _positive_size((int(value[0]), int(value[1])))


def _positive_size(value: tuple[int, int]) -> tuple[int, int]:
    width_px, height_px = int(value[0]), int(value[1])
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"Image size must be positive; got {(width_px, height_px)!r}.")
    return width_px, height_px


def _camera_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3; got shape {matrix.shape!r}.")
    return np.ascontiguousarray(matrix)


def _distortion_coefficients(value: object) -> np.ndarray:
    coeffs = np.asarray(value, dtype=np.float64).reshape(-1, 1)
    if coeffs.size < 4:
        raise ValueError(
            "distortion_coefficients must include at least four OpenCV coefficients; "
            f"got {coeffs.size}."
        )
    return np.ascontiguousarray(coeffs)


def _decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Could not decode image bytes for camera intrinsics transform.")
    return decoded


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", _ensure_uint8(image))
    if not ok:
        raise ValueError("Could not encode transformed image bytes.")
    return encoded.tobytes()


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    array = _ensure_uint8(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and int(array.shape[2]) == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    if array.ndim == 3 and int(array.shape[2]) == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported decoded image shape: {array.shape!r}.")


def _array_size_wh(image: np.ndarray) -> tuple[int, int]:
    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise ValueError(f"Camera intrinsics transform expects 2D/3D image; got {array.shape!r}.")
    return int(array.shape[1]), int(array.shape[0])


def _border_value(image: np.ndarray, value: int) -> int | tuple[int, ...]:
    array = np.asarray(image)
    if array.ndim == 3:
        return tuple(int(value) for _ in range(int(array.shape[2])))
    return int(value)


__all__ = [
    "CAMERA_INTRINSICS_METADATA_APPLIED",
    "CAMERA_INTRINSICS_METADATA_MODE",
    "CAMERA_INTRINSICS_METADATA_REVISION",
    "CAMERA_INTRINSICS_MODE_DISABLED",
    "CAMERA_INTRINSICS_MODE_LABELS",
    "CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP",
    "CAMERA_INTRINSICS_MODE_REAL_UNDISTORT_ONLY",
    "SUPPORTED_CAMERA_INTRINSICS_MODES",
    "CameraCalibration",
    "CameraIntrinsicsFrameTransformer",
    "CameraIntrinsicsTransformResult",
    "CameraIntrinsicsTransformSnapshot",
    "CameraIntrinsicsTransformState",
    "default_real_calibration_path",
    "default_unity_calibration_path",
    "normalize_camera_intrinsics_mode",
]
