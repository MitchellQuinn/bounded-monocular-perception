"""Dependency-light contracts for the ChArUco calibration app.

This module is the boundary between the PySide6 GUI and the functional camera
calibration logic.  It intentionally avoids PySide6, OpenCV, NumPy, and direct
camera imports so workers can exchange stable, serialisable payloads.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


CAMERA_CALIBRATION_CONTRACT_VERSION = "rb-camera-calibration-v0_1"
CAMERA_CALIBRATION_ARTIFACT_VERSION = "rb-camera-calibration-artifact-v0_1"
DEFAULT_FRAME_HASH_ALGORITHM = "blake2b-128"
DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES = 16
PLACEHOLDER_ARUCO_DICTIONARY = "<ARUCO_DICTIONARY>"


class CalibrationPatternType(str, Enum):
    """Supported calibration board families."""

    CHARUCO = "charuco"


class CameraSourceType(str, Enum):
    """Supported camera capture backends."""

    OPENCV_V4L2 = "opencv_v4l2"
    OPENCV_GENERIC = "opencv_generic"


class CalibrationWorkerName(str, Enum):
    """Named workers used by the GUI and worker bridge."""

    CAMERA = "camera"
    DETECTION = "detection"
    CAPTURE = "capture"
    CALIBRATION = "calibration"


class WorkerState(str, Enum):
    """Lifecycle states emitted by long-running workers."""

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class CaptureDecisionType(str, Enum):
    """High-level capture decision vocabulary."""

    ACCEPT = "accept"
    REJECT = "reject"
    HOLD = "hold"


class CaptureRejectReason(str, Enum):
    """Visible reasons a frame may not be accepted for calibration."""

    NO_BOARD = "no_board"
    TOO_FEW_MARKERS = "too_few_markers"
    TOO_FEW_CHARUCO_CORNERS = "too_few_charuco_corners"
    BOARD_TOO_SMALL = "board_too_small"
    BOARD_TOO_LARGE = "board_too_large"
    TOO_CLOSE_TO_EDGE = "too_close_to_edge"
    IMAGE_TOO_BLURRY = "image_too_blurry"
    EXPOSURE_POOR = "exposure_poor"
    DUPLICATE_POSE = "duplicate_pose"
    COOLDOWN_ACTIVE = "cooldown_active"
    UNSTABLE_DETECTION = "unstable_detection"
    CONFIG_MISMATCH = "config_mismatch"


class DictionaryProbeConfidence(str, Enum):
    """Coarse confidence labels for dictionary probing diagnostics."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _to_plain(value: Any) -> Any:
    """Convert contract values to JSON-compatible containers."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "encoding": "base64",
            "byte_size": len(value),
            "data": base64.b64encode(value).decode("ascii"),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            return _to_plain(to_list())
        except (TypeError, ValueError):
            return value
    return value


class ContractMixin:
    """Small shared serializer for contract payloads."""

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


def _require_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}.")


@dataclass(frozen=True)
class CharucoBoardConfig(ContractMixin):
    """Measured ChArUco board definition."""

    pattern_type: CalibrationPatternType = CalibrationPatternType.CHARUCO
    squares_x: int = 0
    squares_y: int = 0
    square_length_m: float = 0.0
    marker_length_m: float = 0.0
    aruco_dictionary: str = PLACEHOLDER_ARUCO_DICTIONARY
    board_name: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_positive("squares_x", self.squares_x)
        _require_positive("squares_y", self.squares_y)
        _require_positive("square_length_m", self.square_length_m)
        _require_positive("marker_length_m", self.marker_length_m)
        if self.marker_length_m >= self.square_length_m:
            raise ValueError(
                "marker_length_m must be smaller than square_length_m "
                f"({self.marker_length_m!r} >= {self.square_length_m!r})."
            )
        if not self.aruco_dictionary:
            raise ValueError("aruco_dictionary is required; use an explicit OpenCV dictionary name.")


@dataclass(frozen=True)
class CameraCaptureConfig(ContractMixin):
    """OpenCV camera capture configuration."""

    camera_source_type: CameraSourceType = CameraSourceType.OPENCV_V4L2
    camera_device: str | int = "/dev/video0"
    width_px: int = 1920
    height_px: int = 1200
    fps: float = 80.0
    pixel_format: str = "YUYV"
    backend: str = "V4L2"
    exposure: float | None = None
    gain: float | None = None
    focus: float | None = None
    auto_exposure: bool | None = None
    auto_white_balance: bool | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_positive("width_px", self.width_px)
        _require_positive("height_px", self.height_px)
        _require_positive("fps", self.fps)


@dataclass(frozen=True)
class CapturePolicyConfig(ContractMixin):
    """Automatic capture gating thresholds."""

    target_accepted_frame_count: int = 45
    min_marker_count: int = 8
    min_charuco_corner_count: int = 24
    min_board_area_fraction: float = 0.03
    max_board_area_fraction: float = 0.85
    min_edge_margin_px: float = 12.0
    min_laplacian_variance: float = 80.0
    cooldown_seconds: float = 1.5
    require_pose_novelty: bool = True
    require_stability: bool = True
    stability_window_frames: int = 3
    pose_grid_cols: int = 3
    pose_grid_rows: int = 3
    scale_bin_count: int = 3
    tilt_bin_count: int = 2
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_positive("target_accepted_frame_count", self.target_accepted_frame_count)
        _require_positive("min_marker_count", self.min_marker_count)
        _require_positive("min_charuco_corner_count", self.min_charuco_corner_count)
        _require_positive("pose_grid_cols", self.pose_grid_cols)
        _require_positive("pose_grid_rows", self.pose_grid_rows)
        _require_positive("scale_bin_count", self.scale_bin_count)
        _require_positive("tilt_bin_count", self.tilt_bin_count)
        if not 0.0 < self.min_board_area_fraction < self.max_board_area_fraction <= 1.0:
            raise ValueError(
                "Board area thresholds must satisfy "
                "0 < min_board_area_fraction < max_board_area_fraction <= 1."
            )
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative.")
        if self.stability_window_frames < 1:
            raise ValueError("stability_window_frames must be at least 1.")


@dataclass(frozen=True)
class CalibrationSessionConfig(ContractMixin):
    """Top-level session configuration used by storage, GUI, and workers."""

    session_root: Path
    board_config: CharucoBoardConfig
    camera_config: CameraCaptureConfig
    capture_policy: CapturePolicyConfig = field(default_factory=CapturePolicyConfig)
    save_rejected_samples: bool = True
    save_debug_overlays: bool = True
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class FrameMetadata(ContractMixin):
    """Camera/source metadata attached to a frame."""

    frame_id: str
    sequence_index: int
    captured_at_utc: str
    width_px: int
    height_px: int
    pixel_format: str
    source_name: str
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class FrameHash(ContractMixin):
    """Hash of the exact image bytes used at a boundary."""

    value: str
    algorithm: str = DEFAULT_FRAME_HASH_ALGORITHM
    digest_size_bytes: int = DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class CameraFrame(ContractMixin):
    """A camera frame exchanged by workers."""

    frame_id: str
    metadata: FrameMetadata
    frame_hash: FrameHash
    image_bytes: bytes
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class CharucoDetection(ContractMixin):
    """Dependency-free ChArUco detection summary."""

    frame_id: str
    detected: bool
    marker_count: int
    charuco_corner_count: int
    marker_ids: tuple[int, ...] = ()
    charuco_ids: tuple[int, ...] = ()
    board_center_xy_px: tuple[float, float] | None = None
    board_bounds_xyxy_px: tuple[float, float, float, float] | None = None
    board_area_fraction: float = 0.0
    edge_margin_px: float | None = None
    detection_time_ms: float = 0.0
    warning_messages: tuple[str, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class FrameQualityMetrics(ContractMixin):
    """Simple image quality metrics used by capture gating."""

    laplacian_variance: float
    mean_luma: float
    luma_std: float
    clipped_black_fraction: float
    clipped_white_fraction: float
    contrast_score: float
    blur_score: float
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class PoseSignature(ContractMixin):
    """Binned pose signature for coverage tracking."""

    center_x_norm: float
    center_y_norm: float
    area_fraction: float
    roll_like_angle_deg: float
    perspective_skew_score: float
    grid_cell: tuple[int, int]
    scale_bin: int
    tilt_bin: int
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class PoseCoverageState(ContractMixin):
    """Coverage summary after evaluating accepted poses."""

    accepted_count: int
    occupied_center_cells: tuple[tuple[int, int], ...] = ()
    occupied_scale_bins: tuple[int, ...] = ()
    occupied_tilt_bins: tuple[int, ...] = ()
    coverage_score: float = 0.0
    suggested_next_pose: str = "start with a centered board view"
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class CaptureDecision(ContractMixin):
    """Result of automatic capture policy evaluation."""

    decision_type: CaptureDecisionType
    accepted: bool
    reason: CaptureRejectReason | None
    message: str
    detection: CharucoDetection
    quality: FrameQualityMetrics
    pose_signature: PoseSignature | None = None
    coverage_state: PoseCoverageState | None = None
    cooldown_remaining_s: float = 0.0
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class AcceptedCalibrationFrame(ContractMixin):
    """Stored accepted frame and its diagnostic payloads."""

    frame_id: str
    image_path: Path
    detection_json_path: Path
    overlay_path: Path | None
    frame_hash: FrameHash
    captured_at_utc: str
    charuco_corner_count: int
    marker_count: int
    pose_signature: PoseSignature
    quality: FrameQualityMetrics
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class CalibrationRequest(ContractMixin):
    """Request to solve camera intrinsics from accepted frames."""

    session_root: Path
    board_config: CharucoBoardConfig
    image_size_wh_px: tuple[int, int]
    accepted_frames: tuple[AcceptedCalibrationFrame, ...]
    calibration_flags: tuple[str, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class PerViewCalibrationError(ContractMixin):
    """Reprojection error diagnostics for one accepted frame."""

    frame_id: str
    rms_error_px: float
    mean_error_px: float
    max_error_px: float
    point_count: int
    include_in_final: bool = True
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class CalibrationResult(ContractMixin):
    """Machine-readable calibration result."""

    success: bool
    rms_reprojection_error_px: float
    camera_matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    distortion_coefficients: tuple[float, ...]
    image_size_wh_px: tuple[int, int]
    board_config: CharucoBoardConfig
    accepted_frame_count: int
    used_frame_count: int
    rejected_outlier_count: int
    per_view_errors: tuple[PerViewCalibrationError, ...]
    generated_at_utc: str
    opencv_version: str
    calibration_flags: tuple[str, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class CalibrationArtifactManifest(ContractMixin):
    """Manifest describing exported calibration artifacts."""

    artifact_version: str
    generated_at_utc: str
    session_root: Path
    board_config: CharucoBoardConfig
    camera_config: CameraCaptureConfig
    capture_policy: CapturePolicyConfig
    result_json_path: Path
    result_yaml_path: Path
    accepted_frame_dir: Path
    rejected_sample_dir: Path
    report_csv_path: Path
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class DictionaryProbeCandidate(ContractMixin):
    """One dictionary probe result."""

    dictionary_name: str
    marker_count: int
    marker_ids: tuple[int, ...]
    confidence: DictionaryProbeConfidence
    usefulness_score: float
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class DictionaryProbeReport(ContractMixin):
    """Summary produced by trying common predefined dictionaries."""

    frame_id: str | None
    candidates: tuple[DictionaryProbeCandidate, ...]
    best_candidate: DictionaryProbeCandidate | None
    image_size_wh_px: tuple[int, int] | None = None
    warning_messages: tuple[str, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@dataclass(frozen=True)
class BoardDimensionDebugReport(ContractMixin):
    """Diagnostics comparing configured and reversed ChArUco board dimensions."""

    configured_squares_xy: tuple[int, int]
    reversed_squares_xy: tuple[int, int]
    configured_marker_count: int
    configured_charuco_corner_count: int
    reversed_marker_count: int
    reversed_charuco_corner_count: int
    reversing_appears_more_plausible: bool
    message: str
    extras: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CAMERA_CALIBRATION_CONTRACT_VERSION


@runtime_checkable
class CameraSource(Protocol):
    """Camera frame source used by workers."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def read_frame(self) -> CameraFrame | None:
        ...


@runtime_checkable
class CharucoDetectorProtocol(Protocol):
    """Detector boundary used by GUI and capture workers."""

    def detect(self, frame: CameraFrame) -> CharucoDetection:
        ...


@runtime_checkable
class FrameQualityScorer(Protocol):
    """Image quality scoring boundary."""

    def score(self, frame: CameraFrame) -> FrameQualityMetrics:
        ...


@runtime_checkable
class PoseDiversityTracker(Protocol):
    """Pose coverage boundary for automatic capture."""

    def evaluate(
        self,
        detection: CharucoDetection,
        quality: FrameQualityMetrics,
    ) -> PoseSignature:
        ...

    def update_accepted(self, signature: PoseSignature) -> PoseCoverageState:
        ...


@runtime_checkable
class CaptureControllerProtocol(Protocol):
    """Automatic capture policy boundary."""

    def evaluate_frame(
        self,
        frame: CameraFrame,
        detection: CharucoDetection,
        quality: FrameQualityMetrics,
    ) -> CaptureDecision:
        ...


@runtime_checkable
class CalibrationSolverProtocol(Protocol):
    """Calibration solver boundary."""

    def solve(self, request: CalibrationRequest) -> CalibrationResult:
        ...


@runtime_checkable
class CalibrationArtifactExporterProtocol(Protocol):
    """Artifact exporter boundary."""

    def export(
        self,
        result: CalibrationResult,
        session_config: CalibrationSessionConfig,
    ) -> CalibrationArtifactManifest:
        ...


__all__ = [
    "CAMERA_CALIBRATION_ARTIFACT_VERSION",
    "CAMERA_CALIBRATION_CONTRACT_VERSION",
    "DEFAULT_FRAME_HASH_ALGORITHM",
    "DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES",
    "PLACEHOLDER_ARUCO_DICTIONARY",
    "AcceptedCalibrationFrame",
    "BoardDimensionDebugReport",
    "CalibrationArtifactExporterProtocol",
    "CalibrationArtifactManifest",
    "CalibrationPatternType",
    "CalibrationRequest",
    "CalibrationResult",
    "CalibrationSessionConfig",
    "CalibrationSolverProtocol",
    "CalibrationWorkerName",
    "CameraCaptureConfig",
    "CameraFrame",
    "CameraSource",
    "CameraSourceType",
    "CaptureControllerProtocol",
    "CaptureDecision",
    "CaptureDecisionType",
    "CapturePolicyConfig",
    "CaptureRejectReason",
    "CharucoBoardConfig",
    "CharucoDetection",
    "CharucoDetectorProtocol",
    "ContractMixin",
    "DictionaryProbeCandidate",
    "DictionaryProbeConfidence",
    "DictionaryProbeReport",
    "FrameHash",
    "FrameMetadata",
    "FrameQualityMetrics",
    "FrameQualityScorer",
    "PerViewCalibrationError",
    "PoseCoverageState",
    "PoseDiversityTracker",
    "PoseSignature",
    "WorkerState",
]
