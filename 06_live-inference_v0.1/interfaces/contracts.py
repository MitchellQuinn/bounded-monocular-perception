"""Contract and interface definitions for the live inference pipeline.

This module deliberately avoids PySide6, camera, image-processing, NumPy, and
model-runtime imports.  Qt signals can carry these payload objects later, but
the contract layer itself stays small, inspectable, and replaceable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


LIVE_INFERENCE_CONTRACT_VERSION = "rb-live-inference-v0_1"

DEFAULT_FRAME_DIR = Path("./live_frames")
DEFAULT_LATEST_FRAME_FILENAME = "latest_frame.png"
DEFAULT_TEMP_FRAME_FILENAME = "latest_frame.tmp.png"
DEFAULT_DEBUG_OUTPUT_DIR = Path("./live_debug")

DEFAULT_FRAME_HASH_ALGORITHM = "blake2b-128"
TRI_STREAM_DISTANCE_IMAGE_KEY = "x_distance_image"
TRI_STREAM_ORIENTATION_IMAGE_KEY = "x_orientation_image"
TRI_STREAM_GEOMETRY_KEY = "x_geometry"
TRI_STREAM_INPUT_KEYS = (
    TRI_STREAM_DISTANCE_IMAGE_KEY,
    TRI_STREAM_ORIENTATION_IMAGE_KEY,
    TRI_STREAM_GEOMETRY_KEY,
)


class WorkerName(str, Enum):
    """Known live pipeline workers."""

    CAMERA = "camera"
    INFERENCE = "inference"


class WorkerState(str, Enum):
    """Shared lifecycle state vocabulary for all live workers."""

    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


class WorkerEventType(str, Enum):
    """Lifecycle and boundary events workers may emit."""

    START_REQUESTED = "start_requested"
    STARTING = "starting"
    STARTED = "started"
    STOP_REQUESTED = "stop_requested"
    STOPPING = "stopping"
    STOPPED = "stopped"
    STATUS_CHANGED = "status_changed"
    FRAME_WRITTEN = "frame_written"
    FRAME_SKIPPED = "frame_skipped"
    RESULT_READY = "result_ready"
    DEBUG_IMAGE_READY = "debug_image_ready"
    WARNING_OCCURRED = "warning_occurred"
    ERROR_OCCURRED = "error_occurred"


class IssueSeverity(str, Enum):
    """Simple GUI/log severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class FrameProcessingPolicy(str, Enum):
    """How the inference worker should choose frames from the handoff."""

    NEWEST_COMPLETED_SKIP_STALE = "newest_completed_skip_stale"


class FrameSkipReason(str, Enum):
    """Reasons a candidate frame may be skipped without producing a result."""

    DUPLICATE_HASH = "duplicate_hash"
    MISSING_FILE = "missing_file"
    UNREADABLE_FILE = "unreadable_file"
    DECODE_FAILED = "decode_failed"
    PREPROCESS_FAILED = "preprocess_failed"
    INFERENCE_FAILED = "inference_failed"


class InferenceInputMode(str, Enum):
    """Input contract labels used across preprocessing and inference."""

    RAW_IMAGE = "raw_image"
    TRI_STREAM_V0_4 = "tri_stream_distance_orientation_geometry"


ALLOWED_WORKER_STATE_TRANSITIONS: Mapping[WorkerState, tuple[WorkerState, ...]] = {
    WorkerState.STOPPED: (WorkerState.STARTING,),
    WorkerState.STARTING: (WorkerState.RUNNING, WorkerState.ERROR, WorkerState.STOPPING),
    WorkerState.RUNNING: (WorkerState.STOPPING, WorkerState.ERROR),
    WorkerState.STOPPING: (WorkerState.STOPPED, WorkerState.ERROR),
    WorkerState.ERROR: (WorkerState.STOPPED,),
}


def _to_plain(value: Any) -> Any:
    """Convert common contract values to plain Python containers."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class FrameHandoffPaths:
    """File names and paths for atomic latest-frame handoff."""

    frame_dir: Path = DEFAULT_FRAME_DIR
    latest_frame_filename: str = DEFAULT_LATEST_FRAME_FILENAME
    temp_frame_filename: str = DEFAULT_TEMP_FRAME_FILENAME

    @property
    def latest_frame_path(self) -> Path:
        """Completed frame path consumed by inference."""
        return self.frame_dir / self.latest_frame_filename

    @property
    def temp_frame_path(self) -> Path:
        """Temporary frame path produced before atomic replace."""
        return self.frame_dir / self.temp_frame_filename

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class LiveInferenceConfig:
    """Central configuration shared by GUI, camera, and inference boundaries."""

    frame_dir: Path = DEFAULT_FRAME_DIR
    latest_frame_filename: str = DEFAULT_LATEST_FRAME_FILENAME
    temp_frame_filename: str = DEFAULT_TEMP_FRAME_FILENAME
    camera_index: int = 0
    camera_width_px: int | None = None
    camera_height_px: int | None = None
    camera_fps: int | None = None
    inference_poll_interval_ms: int = 10
    duplicate_hash_skip_enabled: bool = True
    model_path: Path | None = None
    device: str = "cuda"
    save_debug_images: bool = False
    debug_output_dir: Path | None = DEFAULT_DEBUG_OUTPUT_DIR
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def handoff_paths(self) -> FrameHandoffPaths:
        """Resolved file-handoff paths derived from this config."""
        return FrameHandoffPaths(
            frame_dir=self.frame_dir,
            latest_frame_filename=self.latest_frame_filename,
            temp_frame_filename=self.temp_frame_filename,
        )

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class FrameMetadata:
    """Optional camera/source metadata for a frame handoff."""

    frame_id: str | None = None
    camera_index: int | None = None
    source_name: str | None = None
    captured_at_utc: str | None = None
    written_at_utc: str | None = None
    width_px: int | None = None
    height_px: int | None = None
    pixel_format: str | None = None
    encoding: str = "png"
    byte_size: int | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class FrameReference:
    """Reference to a completed raw image frame visible to inference."""

    image_path: Path
    metadata: FrameMetadata = field(default_factory=FrameMetadata)
    completed_at_utc: str | None = None
    frame_hash: str | None = None
    hash_algorithm: str | None = None
    byte_size: int | None = None
    handoff_paths: FrameHandoffPaths | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class InferenceRequest:
    """Request for processing one raw completed frame."""

    request_id: str
    frame: FrameReference
    requested_at_utc: str
    source_input_mode: InferenceInputMode = InferenceInputMode.RAW_IMAGE
    processing_policy: FrameProcessingPolicy = FrameProcessingPolicy.NEWEST_COMPLETED_SKIP_STALE
    duplicate_hash_skip_enabled: bool = True
    model_path: Path | None = None
    device: str | None = None
    save_debug_images: bool = False
    debug_output_dir: Path | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class PreparedInferenceInputs:
    """Dependency-free reference to the model inputs prepared from a raw image."""

    request_id: str
    input_mode: InferenceInputMode = InferenceInputMode.TRI_STREAM_V0_4
    input_keys: tuple[str, ...] = TRI_STREAM_INPUT_KEYS
    source_frame: FrameReference | None = None
    preprocessing_metadata: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class RoiMetadata:
    """Optional ROI, crop, bbox, and geometry metadata attached to a result."""

    bbox_xyxy_px: tuple[float, float, float, float] | None = None
    center_xy_px: tuple[float, float] | None = None
    source_image_wh_px: tuple[int, int] | None = None
    distance_canvas_wh_px: tuple[int, int] | None = None
    orientation_canvas_wh_px: tuple[int, int] | None = None
    geometry_schema: tuple[str, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class InferenceResult:
    """Structured inference output emitted to the GUI."""

    request_id: str
    input_image_path: Path
    input_image_hash: str
    timestamp_utc: str
    predicted_distance_m: float
    predicted_yaw_sin: float
    predicted_yaw_cos: float
    predicted_yaw_deg: float
    inference_time_ms: float
    hash_algorithm: str = DEFAULT_FRAME_HASH_ALGORITHM
    preprocessing_time_ms: float | None = None
    total_time_ms: float | None = None
    model_input_mode: InferenceInputMode = InferenceInputMode.TRI_STREAM_V0_4
    roi_metadata: RoiMetadata | None = None
    debug_paths: Mapping[str, Path] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)
    debug: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class DebugImageReference:
    """Reference to one optional debug image generated during inference."""

    request_id: str
    image_kind: str
    path: Path
    created_at_utc: str
    source_frame_hash: str | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class CameraWorkerCounters:
    """Minimum observable camera-worker counters."""

    frames_captured: int = 0
    frames_written: int = 0
    frame_write_failures: int = 0
    last_frame_write_time_utc: str | None = None
    last_frame_path: Path | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class InferenceWorkerCounters:
    """Minimum observable inference-worker counters."""

    frames_seen: int = 0
    frames_processed: int = 0
    frames_skipped_duplicate: int = 0
    frames_failed_read: int = 0
    frames_failed_decode: int = 0
    frames_failed_preprocess: int = 0
    frames_failed_inference: int = 0
    last_input_hash: str | None = None
    last_inference_time_ms: float | None = None
    last_preprocessing_time_ms: float | None = None
    last_total_time_ms: float | None = None
    last_result_time_utc: str | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class WorkerStatus:
    """Structured worker status message for GUI state indicators."""

    worker_name: WorkerName
    state: WorkerState
    message: str
    timestamp_utc: str
    counters: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class WorkerLifecycleEvent:
    """Structured lifecycle event emitted across worker boundaries."""

    worker_name: WorkerName
    event_type: WorkerEventType
    state: WorkerState
    timestamp_utc: str
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class WorkerWarning:
    """Recoverable warning emitted by a worker."""

    worker_name: WorkerName
    warning_type: str
    message: str
    timestamp_utc: str
    recoverable: bool = True
    frame: FrameReference | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class WorkerError:
    """Structured worker error message for recoverable or fatal failures."""

    worker_name: WorkerName
    error_type: str
    message: str
    recoverable: bool
    timestamp_utc: str
    frame: FrameReference | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@runtime_checkable
class WorkerControl(Protocol):
    """Minimal control surface shared by camera and inference workers."""

    @property
    def worker_name(self) -> WorkerName:
        ...

    def start(self) -> None:
        ...

    def request_stop(self) -> None:
        ...

    def current_status(self) -> WorkerStatus:
        ...


@runtime_checkable
class CameraWorkerProtocol(WorkerControl, Protocol):
    """Control boundary for a replaceable camera worker."""

    def configure(self, config: LiveInferenceConfig) -> None:
        ...


@runtime_checkable
class InferenceWorkerProtocol(WorkerControl, Protocol):
    """Control boundary for a replaceable inference worker."""

    def configure(self, config: LiveInferenceConfig) -> None:
        ...


@runtime_checkable
class WorkerEventSink(Protocol):
    """Generic event sink usable by a GUI adapter or test harness."""

    def status_changed(self, status: WorkerStatus) -> None:
        ...

    def lifecycle_event(self, event: WorkerLifecycleEvent) -> None:
        ...

    def warning_occurred(self, warning: WorkerWarning) -> None:
        ...

    def error_occurred(self, error: WorkerError) -> None:
        ...


@runtime_checkable
class CameraWorkerEventSink(WorkerEventSink, Protocol):
    """Events emitted by the camera-worker boundary."""

    def frame_written(self, frame: FrameReference) -> None:
        ...


@runtime_checkable
class InferenceWorkerEventSink(WorkerEventSink, Protocol):
    """Events emitted by the inference-worker boundary."""

    def result_ready(self, result: InferenceResult) -> None:
        ...

    def debug_image_ready(self, image: DebugImageReference) -> None:
        ...


@runtime_checkable
class FrameHandoffWriter(Protocol):
    """Service boundary for publishing completed latest-frame files."""

    def publish_frame(self, image_bytes: bytes, metadata: FrameMetadata) -> FrameReference:
        ...


@runtime_checkable
class FrameHandoffReader(Protocol):
    """Service boundary for discovering and reading completed frame files."""

    def latest_completed_frame(self) -> FrameReference | None:
        ...

    def read_frame_bytes(self, frame: FrameReference) -> bytes:
        ...


@runtime_checkable
class RawImagePreprocessor(Protocol):
    """Prepares model inputs from raw image bytes, not NPZ training shards."""

    def prepare_model_inputs(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
    ) -> PreparedInferenceInputs:
        ...


@runtime_checkable
class InferenceEngine(Protocol):
    """Runs model inference against already prepared model inputs."""

    def run_inference(self, inputs: PreparedInferenceInputs) -> InferenceResult:
        ...


__all__ = [
    "ALLOWED_WORKER_STATE_TRANSITIONS",
    "DEFAULT_DEBUG_OUTPUT_DIR",
    "DEFAULT_FRAME_DIR",
    "DEFAULT_FRAME_HASH_ALGORITHM",
    "DEFAULT_LATEST_FRAME_FILENAME",
    "DEFAULT_TEMP_FRAME_FILENAME",
    "LIVE_INFERENCE_CONTRACT_VERSION",
    "TRI_STREAM_DISTANCE_IMAGE_KEY",
    "TRI_STREAM_GEOMETRY_KEY",
    "TRI_STREAM_INPUT_KEYS",
    "TRI_STREAM_ORIENTATION_IMAGE_KEY",
    "CameraWorkerCounters",
    "CameraWorkerEventSink",
    "CameraWorkerProtocol",
    "DebugImageReference",
    "FrameHandoffPaths",
    "FrameHandoffReader",
    "FrameHandoffWriter",
    "FrameMetadata",
    "FrameProcessingPolicy",
    "FrameReference",
    "FrameSkipReason",
    "InferenceEngine",
    "InferenceInputMode",
    "InferenceRequest",
    "InferenceResult",
    "InferenceWorkerCounters",
    "InferenceWorkerEventSink",
    "InferenceWorkerProtocol",
    "IssueSeverity",
    "LiveInferenceConfig",
    "PreparedInferenceInputs",
    "RawImagePreprocessor",
    "RoiMetadata",
    "WorkerControl",
    "WorkerError",
    "WorkerEventSink",
    "WorkerEventType",
    "WorkerLifecycleEvent",
    "WorkerName",
    "WorkerState",
    "WorkerStatus",
    "WorkerWarning",
]
