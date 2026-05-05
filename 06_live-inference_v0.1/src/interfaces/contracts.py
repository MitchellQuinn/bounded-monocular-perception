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
MODEL_TOPOLOGY_CONTRACT_VERSION = "rb-topology-output-reporting-v1"

DEFAULT_FRAME_DIR = Path("./live_frames")
DEFAULT_LATEST_FRAME_FILENAME = "latest_frame.png"
DEFAULT_TEMP_FRAME_FILENAME = "latest_frame.tmp.png"
DEFAULT_DEBUG_OUTPUT_DIR = Path("./live_debug")

DEFAULT_FRAME_HASH_ALGORITHM = "blake2b-128"
DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES = 16
RAW_IMAGE_INPUT_MODE = "raw_image"
TRI_STREAM_INPUT_MODE = "tri_stream_distance_orientation_geometry"
TRI_STREAM_DISTANCE_IMAGE_KEY = "x_distance_image"
TRI_STREAM_ORIENTATION_IMAGE_KEY = "x_orientation_image"
TRI_STREAM_GEOMETRY_KEY = "x_geometry"
DISPLAY_ARTIFACT_ACCEPTED_RAW_FRAME = "accepted_raw_frame"
DISPLAY_ARTIFACT_DISTANCE_IMAGE = TRI_STREAM_DISTANCE_IMAGE_KEY
DISPLAY_ARTIFACT_ORIENTATION_IMAGE = TRI_STREAM_ORIENTATION_IMAGE_KEY
DISPLAY_ARTIFACT_ROI_OVERLAY = "roi_overlay"
TRI_STREAM_INPUT_KEYS = (
    TRI_STREAM_DISTANCE_IMAGE_KEY,
    TRI_STREAM_ORIENTATION_IMAGE_KEY,
    TRI_STREAM_GEOMETRY_KEY,
)
PREPROCESSING_CONTRACT_KEY = "PreprocessingContract"
PREPROCESSING_CONTRACT_NAME = "rb-preprocess-v4-tri-stream-orientation-v1"
TRI_STREAM_PREPROCESSING_CONTRACT_VERSION = PREPROCESSING_CONTRACT_NAME
TRI_STREAM_REPRESENTATION_KIND = "tri_stream_npz"
TRI_STREAM_STORAGE_FORMAT = "npz"
DISTANCE_IMAGE_CONTRACT_NAME = "fixed_unscaled_roi_canvas"
ORIENTATION_IMAGE_CONTRACT_NAME = "target_centered_scaled_by_silhouette_extent"
GEOMETRY_SCHEMA_NAME = "x_geometry_schema"
TRI_STREAM_GEOMETRY_SCHEMA = (
    "cx_px",
    "cy_px",
    "w_px",
    "h_px",
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "aspect_ratio",
    "area_norm",
)
DISTANCE_TARGET_COLUMN = "distance_m"
YAW_DEG_TARGET_COLUMN = "yaw_deg"
YAW_SIN_TARGET_COLUMN = "yaw_sin"
YAW_COS_TARGET_COLUMN = "yaw_cos"
MODEL_OUTPUT_DISTANCE_KEY = "distance_m"
MODEL_OUTPUT_YAW_SIN_COS_KEY = "yaw_sin_cos"
PREDICTED_DISTANCE_FIELD = "predicted_distance_m"
PREDICTED_YAW_SIN_FIELD = "predicted_yaw_sin"
PREDICTED_YAW_COS_FIELD = "predicted_yaw_cos"
PREDICTED_YAW_DEG_FIELD = "predicted_yaw_deg"
LIVE_INFERENCE_OUTPUT_FIELDS = (
    PREDICTED_DISTANCE_FIELD,
    PREDICTED_YAW_SIN_FIELD,
    PREDICTED_YAW_COS_FIELD,
    PREDICTED_YAW_DEG_FIELD,
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
    STALE_FRAME = "stale_frame"


class FrameFailureStage(str, Enum):
    """Processing stages where an attempted frame can fail."""

    READ = "read"
    DECODE = "decode"
    PREPROCESS = "preprocess"
    INFERENCE = "inference"
    OUTPUT = "output"


class InferenceInputMode(str, Enum):
    """Input contract labels used across preprocessing and inference."""

    RAW_IMAGE = RAW_IMAGE_INPUT_MODE
    TRI_STREAM_V0_4 = TRI_STREAM_INPUT_MODE


class RuntimeParameterValueType(str, Enum):
    """Serializable value types for GUI-discoverable runtime parameters."""

    BOOL = "bool"
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    ENUM = "enum"


class RuntimeParameterWidgetHint(str, Enum):
    """Optional GUI widget hints for runtime parameter controls."""

    CHECKBOX = "checkbox"
    INT_INPUT = "int_input"
    FLOAT_INPUT = "float_input"
    SLIDER = "slider"
    DROPDOWN = "dropdown"
    TEXT_INPUT = "text_input"


ALLOWED_WORKER_STATE_TRANSITIONS: Mapping[WorkerState, tuple[WorkerState, ...]] = {
    WorkerState.STOPPED: (WorkerState.STARTING,),
    WorkerState.STARTING: (WorkerState.RUNNING, WorkerState.ERROR, WorkerState.STOPPING),
    WorkerState.RUNNING: (WorkerState.STOPPING, WorkerState.ERROR),
    WorkerState.STOPPING: (WorkerState.STOPPED, WorkerState.ERROR),
    WorkerState.ERROR: (WorkerState.STOPPED,),
}


def is_allowed_worker_state_transition(
    current: WorkerState,
    next_state: WorkerState,
    *,
    allow_idempotent: bool = True,
) -> bool:
    """Return whether a worker state transition is allowed by the shared contract.

    Idempotent status emissions are allowed by default. Repeated
    ``request_stop()`` calls while a worker is already ``STOPPING`` should be
    safe, ``request_stop()`` while ``STOPPED`` should be a no-op, and
    ``request_stop()`` while ``ERROR`` should still allow cleanup toward
    ``STOPPED``. This helper is contract vocabulary only; it is not a worker
    state machine implementation.
    """
    if allow_idempotent and current == next_state:
        return True
    return next_state in ALLOWED_WORKER_STATE_TRANSITIONS.get(current, ())


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


def get_contract_version(value: Any) -> str | None:
    """Extract a live contract version from a payload object or mapping."""
    if isinstance(value, Mapping):
        version = value.get("contract_version")
        return version if isinstance(version, str) else None
    version = getattr(value, "contract_version", None)
    return version if isinstance(version, str) else None


def contract_version_matches(
    value: Any,
    expected: str = LIVE_INFERENCE_CONTRACT_VERSION,
) -> bool:
    """Return whether a payload declares the expected live contract version."""
    return get_contract_version(value) == expected


def require_contract_version(
    value: Any,
    expected: str = LIVE_INFERENCE_CONTRACT_VERSION,
) -> None:
    """Raise if a payload is missing or mismatches the expected live contract version."""
    actual = get_contract_version(value)
    if actual == expected:
        return
    if actual is None:
        actual_text = "missing"
    else:
        actual_text = repr(actual)
    raise ValueError(
        "Live inference contract version mismatch: "
        f"expected {expected!r}, actual {actual_text}."
    )


@dataclass(frozen=True)
class InferenceOutputContract:
    """Stable GUI-facing names for live inference result fields."""

    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    distance_field: str = PREDICTED_DISTANCE_FIELD
    yaw_sin_field: str = PREDICTED_YAW_SIN_FIELD
    yaw_cos_field: str = PREDICTED_YAW_COS_FIELD
    yaw_deg_field: str = PREDICTED_YAW_DEG_FIELD
    model_distance_output_key: str = MODEL_OUTPUT_DISTANCE_KEY
    model_yaw_output_key: str = MODEL_OUTPUT_YAW_SIN_COS_KEY
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class ModelContractReference:
    """Reference metadata for the repository model/preprocessing contracts."""

    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    model_path: Path | None = None
    model_contract_version: str | None = MODEL_TOPOLOGY_CONTRACT_VERSION
    preprocessing_contract_name: str | None = PREPROCESSING_CONTRACT_NAME
    input_mode: InferenceInputMode = InferenceInputMode.TRI_STREAM_V0_4
    input_keys: tuple[str, ...] = TRI_STREAM_INPUT_KEYS
    representation_kind: str = TRI_STREAM_REPRESENTATION_KIND
    storage_format: str = TRI_STREAM_STORAGE_FORMAT
    geometry_schema_name: str = GEOMETRY_SCHEMA_NAME
    geometry_schema: tuple[str, ...] = TRI_STREAM_GEOMETRY_SCHEMA
    output_fields: tuple[str, ...] = LIVE_INFERENCE_OUTPUT_FIELDS
    model_output_keys: tuple[str, ...] = (
        MODEL_OUTPUT_DISTANCE_KEY,
        MODEL_OUTPUT_YAW_SIN_COS_KEY,
    )
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class FrameHandoffPaths:
    """File names and same-directory paths for atomic latest-frame handoff."""

    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
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

    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    frame_dir: Path = DEFAULT_FRAME_DIR
    latest_frame_filename: str = DEFAULT_LATEST_FRAME_FILENAME
    temp_frame_filename: str = DEFAULT_TEMP_FRAME_FILENAME
    frame_hash_algorithm: str = DEFAULT_FRAME_HASH_ALGORITHM
    frame_hash_digest_size_bytes: int = DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES
    camera_index: int = 0
    camera_width_px: int | None = None
    camera_height_px: int | None = None
    camera_fps: int | None = None
    inference_poll_interval_ms: int = 10
    duplicate_hash_skip_enabled: bool = True
    model_path: Path | None = None
    model_contract: ModelContractReference | None = None
    device: str = "auto"
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

    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
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
class FrameHash:
    """Hash of the exact image bytes accepted for frame processing."""

    value: str
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    algorithm: str = DEFAULT_FRAME_HASH_ALGORITHM
    digest_size_bytes: int = DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class FrameReference:
    """Reference to a completed raw image frame visible to inference.

    The referenced path must identify a completed frame, not a temporary or
    partially written file. With latest-frame handoff this is normally the
    configured latest path after an atomic replace operation.
    """

    image_path: Path
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    metadata: FrameMetadata = field(default_factory=FrameMetadata)
    completed_at_utc: str | None = None
    frame_hash: FrameHash | None = None
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
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    source_input_mode: InferenceInputMode = InferenceInputMode.RAW_IMAGE
    processing_policy: FrameProcessingPolicy = FrameProcessingPolicy.NEWEST_COMPLETED_SKIP_STALE
    duplicate_hash_skip_enabled: bool = True
    model_path: Path | None = None
    model_contract: ModelContractReference | None = None
    device: str | None = None
    save_debug_images: bool = False
    debug_output_dir: Path | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class PreparedInferenceInputs:
    """Dependency-free prepared model input payload for the inference engine.

    ``model_inputs`` may contain runtime arrays or tensors. This contract
    intentionally types those values as ``Any`` so this module does not import
    NumPy, PyTorch, or image/runtime implementation modules.
    """

    request_id: str
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    input_mode: InferenceInputMode = InferenceInputMode.TRI_STREAM_V0_4
    input_keys: tuple[str, ...] = TRI_STREAM_INPUT_KEYS
    model_inputs: Mapping[str, Any] = field(default_factory=dict)
    source_frame: FrameReference | None = None
    preprocessing_metadata: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(
            {
                "contract_version": self.contract_version,
                "request_id": self.request_id,
                "input_mode": self.input_mode,
                "input_keys": self.input_keys,
                "model_input_keys": tuple(str(key) for key in self.model_inputs.keys()),
                "source_frame": self.source_frame,
                "preprocessing_metadata": self.preprocessing_metadata,
                "extras": self.extras,
            }
        )


@dataclass(frozen=True)
class RoiMetadata:
    """Optional ROI, crop, bbox, and geometry metadata attached to a result."""

    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    bbox_xyxy_px: tuple[float, float, float, float] | None = None
    bbox_coordinate_space: str = "source_image_px"
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
    input_image_hash: FrameHash
    timestamp_utc: str
    predicted_distance_m: float
    predicted_yaw_sin: float
    predicted_yaw_cos: float
    predicted_yaw_deg: float
    inference_time_ms: float
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    preprocessing_time_ms: float | None = None
    preprocessing_parameter_revision: int | None = None
    total_time_ms: float | None = None
    model_input_mode: InferenceInputMode = InferenceInputMode.TRI_STREAM_V0_4
    output_contract: InferenceOutputContract = field(default_factory=InferenceOutputContract)
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
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    source_frame_hash: FrameHash | None = None
    model_input_key: str | None = None
    parameter_revision: int | None = None
    label: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class RuntimeParameterSpec:
    """GUI-discoverable metadata for one runtime-tunable parameter."""

    name: str
    label: str
    value_type: RuntimeParameterValueType
    default_value: Any
    current_value: Any
    widget_hint: RuntimeParameterWidgetHint
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    group: str = "default"
    description: str = ""
    minimum: float | int | None = None
    maximum: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] = ()
    read_only: bool = False
    requires_restart: bool = False
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class RuntimeParameterSetSpec:
    """Versioned set of runtime parameters exposed by a worker namespace."""

    owner: WorkerName
    namespace: str
    revision: int
    parameters: tuple[RuntimeParameterSpec, ...]
    timestamp_utc: str
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class RuntimeParameterUpdate:
    """Request to update one or more runtime-tunable parameters."""

    owner: WorkerName
    namespace: str
    updates: Mapping[str, Any]
    requested_at_utc: str
    base_revision: int | None = None
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class RuntimeParameterUpdateResult:
    """Worker response to a runtime parameter update request."""

    owner: WorkerName
    namespace: str
    accepted: bool
    revision: int
    timestamp_utc: str
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    applied_updates: Mapping[str, Any] = field(default_factory=dict)
    rejected_updates: Mapping[str, str] = field(default_factory=dict)
    message: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class CameraWorkerCounters:
    """Minimum observable camera-worker counters."""

    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
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

    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    frames_seen: int = 0
    frames_processed: int = 0
    frames_skipped_duplicate: int = 0
    frames_failed_read: int = 0
    frames_failed_decode: int = 0
    frames_failed_preprocess: int = 0
    frames_failed_inference: int = 0
    last_input_hash: FrameHash | None = None
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
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
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
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    message: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@dataclass(frozen=True)
class FrameSkipped:
    """Structured event for a candidate frame that was intentionally skipped."""

    worker_name: WorkerName
    reason: FrameSkipReason
    timestamp_utc: str
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    frame: FrameReference | None = None
    frame_hash: FrameHash | None = None
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
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    recoverable: bool = True
    frame: FrameReference | None = None
    failure_stage: FrameFailureStage | None = None
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
    contract_version: str = LIVE_INFERENCE_CONTRACT_VERSION
    frame: FrameReference | None = None
    failure_stage: FrameFailureStage | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


@runtime_checkable
class WorkerControl(Protocol):
    """Minimal control surface shared by camera and inference workers.

    ``start_work`` names the worker entrypoint without colliding with
    ``QThread.start``. Implementations should make repeated ``request_stop``
    calls safe, including while already stopping, stopped, or cleaning up after
    an error.
    """

    @property
    def worker_name(self) -> WorkerName:
        ...

    def start_work(self) -> None:
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

    def update_runtime_parameters(self, update: RuntimeParameterUpdate) -> None:
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

    def frame_skipped(self, skipped: FrameSkipped) -> None:
        ...

    def debug_image_ready(self, image: DebugImageReference) -> None:
        ...

    def runtime_parameters_available(self, spec: RuntimeParameterSetSpec) -> None:
        ...

    def runtime_parameter_update_result(self, result: RuntimeParameterUpdateResult) -> None:
        ...


@runtime_checkable
class FrameHandoffWriter(Protocol):
    """Service boundary for publishing completed latest-frame files.

    Implementations must write image bytes to the configured temporary path
    first, then atomically replace the configured latest-frame path. The
    temporary and latest paths must be on the same filesystem. Implementations
    must not delete the current valid latest-frame file before replacement.
    """

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
    "DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES",
    "DEFAULT_LATEST_FRAME_FILENAME",
    "DEFAULT_TEMP_FRAME_FILENAME",
    "DISPLAY_ARTIFACT_ACCEPTED_RAW_FRAME",
    "DISPLAY_ARTIFACT_DISTANCE_IMAGE",
    "DISPLAY_ARTIFACT_ORIENTATION_IMAGE",
    "DISPLAY_ARTIFACT_ROI_OVERLAY",
    "DISTANCE_IMAGE_CONTRACT_NAME",
    "DISTANCE_TARGET_COLUMN",
    "GEOMETRY_SCHEMA_NAME",
    "LIVE_INFERENCE_CONTRACT_VERSION",
    "LIVE_INFERENCE_OUTPUT_FIELDS",
    "MODEL_OUTPUT_DISTANCE_KEY",
    "MODEL_OUTPUT_YAW_SIN_COS_KEY",
    "MODEL_TOPOLOGY_CONTRACT_VERSION",
    "ORIENTATION_IMAGE_CONTRACT_NAME",
    "PREDICTED_DISTANCE_FIELD",
    "PREDICTED_YAW_COS_FIELD",
    "PREDICTED_YAW_DEG_FIELD",
    "PREDICTED_YAW_SIN_FIELD",
    "PREPROCESSING_CONTRACT_KEY",
    "PREPROCESSING_CONTRACT_NAME",
    "RAW_IMAGE_INPUT_MODE",
    "TRI_STREAM_DISTANCE_IMAGE_KEY",
    "TRI_STREAM_GEOMETRY_KEY",
    "TRI_STREAM_GEOMETRY_SCHEMA",
    "TRI_STREAM_INPUT_KEYS",
    "TRI_STREAM_INPUT_MODE",
    "TRI_STREAM_ORIENTATION_IMAGE_KEY",
    "TRI_STREAM_PREPROCESSING_CONTRACT_VERSION",
    "TRI_STREAM_REPRESENTATION_KIND",
    "TRI_STREAM_STORAGE_FORMAT",
    "YAW_COS_TARGET_COLUMN",
    "YAW_DEG_TARGET_COLUMN",
    "YAW_SIN_TARGET_COLUMN",
    "CameraWorkerCounters",
    "CameraWorkerEventSink",
    "CameraWorkerProtocol",
    "DebugImageReference",
    "FrameFailureStage",
    "FrameHash",
    "FrameHandoffPaths",
    "FrameHandoffReader",
    "FrameHandoffWriter",
    "FrameMetadata",
    "FrameProcessingPolicy",
    "FrameReference",
    "FrameSkipped",
    "FrameSkipReason",
    "InferenceEngine",
    "InferenceInputMode",
    "InferenceOutputContract",
    "InferenceRequest",
    "InferenceResult",
    "InferenceWorkerCounters",
    "InferenceWorkerEventSink",
    "InferenceWorkerProtocol",
    "IssueSeverity",
    "LiveInferenceConfig",
    "ModelContractReference",
    "PreparedInferenceInputs",
    "RawImagePreprocessor",
    "RoiMetadata",
    "RuntimeParameterSetSpec",
    "RuntimeParameterSpec",
    "RuntimeParameterUpdate",
    "RuntimeParameterUpdateResult",
    "RuntimeParameterValueType",
    "RuntimeParameterWidgetHint",
    "WorkerControl",
    "WorkerError",
    "WorkerEventSink",
    "WorkerEventType",
    "WorkerLifecycleEvent",
    "WorkerName",
    "WorkerState",
    "WorkerStatus",
    "WorkerWarning",
    "contract_version_matches",
    "get_contract_version",
    "is_allowed_worker_state_transition",
    "require_contract_version",
]
