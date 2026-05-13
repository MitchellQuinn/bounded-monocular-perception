"""Synthetic camera publisher for exercising the live inference handoff."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import time
import tomllib
from typing import Any

from interfaces import (
    LIVE_INFERENCE_CONTRACT_VERSION,
    FrameMetadata,
    FrameReference,
    LiveInferenceConfig,
)
from live_inference.frame_handoff import AtomicFrameHandoffWriter, compute_frame_hash


SOURCE_KIND = "synthetic_camera"
DEFAULT_ALLOWED_EXTENSIONS = ("png", "jpg", "jpeg")
DEFAULT_METADATA_FILENAME = "latest_frame.json"
DEFAULT_TEMP_METADATA_FILENAME = "latest_frame.tmp.json"


class SyntheticCameraSortOrder(str, Enum):
    """Available source image sort orders for synthetic camera playback."""

    NAME_ASCENDING = "name_ascending"
    NAME_DESCENDING = "name_descending"
    MODIFIED_TIME_ASCENDING = "modified_time_ascending"
    MODIFIED_TIME_DESCENDING = "modified_time_descending"


@dataclass(frozen=True)
class SyntheticCameraConfig:
    """Implementation-local config for synthetic camera playback."""

    source_dir: Path
    output_dir: Path = Path("./live_frames")
    allowed_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS
    sort_order: SyntheticCameraSortOrder = SyntheticCameraSortOrder.MODIFIED_TIME_ASCENDING
    frame_interval_ms: int = 100
    max_images: int = 2048
    loop: bool = True
    start_index: int = 0
    rescan_on_loop: bool = False
    latest_frame_filename: str = "latest_frame.png"
    temp_frame_filename: str = "latest_frame.tmp.png"
    metadata_filename: str = DEFAULT_METADATA_FILENAME
    temp_metadata_filename: str = DEFAULT_TEMP_METADATA_FILENAME

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_dir",
            _validate_relative_path(self.source_dir, label="source_dir", reject_current=True),
        )
        object.__setattr__(
            self,
            "output_dir",
            _validate_relative_path(self.output_dir, label="output_dir", reject_current=False),
        )
        object.__setattr__(
            self,
            "allowed_extensions",
            _normalize_extensions(self.allowed_extensions),
        )
        object.__setattr__(self, "sort_order", _coerce_sort_order(self.sort_order))
        object.__setattr__(self, "frame_interval_ms", _positive_int(self.frame_interval_ms, "frame_interval_ms"))
        object.__setattr__(self, "max_images", _positive_int(self.max_images, "max_images"))
        start_index = int(self.start_index)
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0; got {self.start_index!r}")
        object.__setattr__(self, "start_index", start_index)
        object.__setattr__(self, "loop", bool(self.loop))
        object.__setattr__(self, "rescan_on_loop", bool(self.rescan_on_loop))
        object.__setattr__(
            self,
            "latest_frame_filename",
            _non_empty_filename(self.latest_frame_filename, "latest_frame_filename"),
        )
        object.__setattr__(
            self,
            "temp_frame_filename",
            _non_empty_filename(self.temp_frame_filename, "temp_frame_filename"),
        )
        object.__setattr__(
            self,
            "metadata_filename",
            _non_empty_filename(self.metadata_filename, "metadata_filename"),
        )
        object.__setattr__(
            self,
            "temp_metadata_filename",
            _non_empty_filename(self.temp_metadata_filename, "temp_metadata_filename"),
        )

    @property
    def live_config(self) -> LiveInferenceConfig:
        """Build the generic live handoff config for this synthetic camera."""
        return LiveInferenceConfig(
            frame_dir=self.output_dir,
            latest_frame_filename=self.latest_frame_filename,
            temp_frame_filename=self.temp_frame_filename,
        )


def load_synthetic_camera_config(path: Path) -> SyntheticCameraConfig:
    """Load a synthetic camera config from a TOML file."""
    payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    raw_config = payload.get("synthetic_camera")
    if not isinstance(raw_config, Mapping):
        raise ValueError("TOML config must contain a [synthetic_camera] table.")
    if "source_dir" not in raw_config:
        raise ValueError("[synthetic_camera] source_dir is required.")

    return SyntheticCameraConfig(
        source_dir=Path(str(raw_config["source_dir"])),
        output_dir=Path(str(raw_config.get("output_dir", Path("./live_frames")))),
        allowed_extensions=tuple(raw_config.get("allowed_extensions", DEFAULT_ALLOWED_EXTENSIONS)),
        sort_order=_coerce_sort_order(
            raw_config.get(
                "sort_order",
                SyntheticCameraSortOrder.MODIFIED_TIME_ASCENDING,
            )
        ),
        frame_interval_ms=int(raw_config.get("frame_interval_ms", 100)),
        max_images=int(raw_config.get("max_images", 2048)),
        loop=bool(raw_config.get("loop", True)),
        start_index=int(raw_config.get("start_index", 0)),
        rescan_on_loop=bool(raw_config.get("rescan_on_loop", False)),
        latest_frame_filename=str(raw_config.get("latest_frame_filename", "latest_frame.png")),
        temp_frame_filename=str(raw_config.get("temp_frame_filename", "latest_frame.tmp.png")),
        metadata_filename=str(raw_config.get("metadata_filename", DEFAULT_METADATA_FILENAME)),
        temp_metadata_filename=str(
            raw_config.get("temp_metadata_filename", DEFAULT_TEMP_METADATA_FILENAME)
        ),
    )


def discover_source_images(
    config: SyntheticCameraConfig,
    base_dir: Path | None = None,
) -> list[Path]:
    """Discover and sort source images for synthetic camera playback."""
    source_root = _source_root(config, base_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"Synthetic camera source_dir does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Synthetic camera source_dir is not a directory: {source_root}")

    allowed = set(config.allowed_extensions)
    images = [
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in allowed
    ]
    images = _sort_source_images(images, source_root, config.sort_order)
    images = images[: config.max_images]
    if not images:
        raise ValueError(f"No supported source images found under {source_root}.")
    return images


def write_metadata_sidecar_atomic(
    payload: Mapping[str, Any],
    metadata_path: Path,
    temp_metadata_path: Path,
) -> None:
    """Write a latest-frame metadata sidecar via same-directory atomic replace."""
    metadata_path = Path(metadata_path)
    temp_metadata_path = Path(temp_metadata_path)
    if metadata_path.parent != temp_metadata_path.parent:
        raise ValueError(
            "Atomic metadata sidecar handoff requires temp and final paths to share a directory: "
            f"temp={temp_metadata_path} metadata={metadata_path}"
        )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with temp_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_metadata_path, metadata_path)


class SyntheticCameraPublisher:
    """Publish source image bytes through the atomic latest-frame handoff."""

    def __init__(
        self,
        config: SyntheticCameraConfig,
        *,
        base_dir: Path | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_utc_fn: Callable[[], str] | None = None,
    ) -> None:
        self.config = config
        self.base_dir = Path.cwd() if base_dir is None else Path(base_dir)
        self.sleep_fn = sleep_fn
        self.now_utc_fn = now_utc_fn or _utc_now_iso
        self.source_root = _source_root(config, self.base_dir)
        self.output_dir = self.base_dir / config.output_dir
        self.metadata_path = self.output_dir / config.metadata_filename
        self.temp_metadata_path = self.output_dir / config.temp_metadata_filename
        self.writer = AtomicFrameHandoffWriter(
            LiveInferenceConfig(
                frame_dir=self.output_dir,
                latest_frame_filename=config.latest_frame_filename,
                temp_frame_filename=config.temp_frame_filename,
            )
        )
        self._source_images = discover_source_images(config, self.base_dir)
        self._current_index = int(config.start_index)
        self._loop_index = 0

    def publish_next(self) -> FrameReference:
        """Publish the next synthetic camera frame and return its generic reference."""
        if not self._source_images:
            raise ValueError("Synthetic camera has no source images to publish.")
        self._prepare_next_index()

        source_path = self._source_images[self._current_index]
        sequence_index = self._current_index
        loop_index = self._loop_index
        image_bytes = source_path.read_bytes()
        byte_size = len(image_bytes)
        published_at_utc = self.now_utc_fn()
        frame_hash = compute_frame_hash(image_bytes)
        source_mtime_ns = source_path.stat().st_mtime_ns
        source_relative_path = _relative_source_path(source_path, self.source_root)
        encoding = source_path.suffix.lower().lstrip(".")
        frame_id = f"synthetic-{loop_index:06d}-{sequence_index:06d}"

        metadata = FrameMetadata(
            frame_id=frame_id,
            source_name=SOURCE_KIND,
            captured_at_utc=published_at_utc,
            written_at_utc=published_at_utc,
            encoding=encoding,
            byte_size=byte_size,
            extras={
                "source_relative_path": source_relative_path,
                "sequence_index": sequence_index,
                "loop_index": loop_index,
                "source_mtime_ns": source_mtime_ns,
                "synthetic_captured_at_utc": published_at_utc,
            },
        )
        frame_reference = self.writer.publish_frame(image_bytes, metadata)
        sidecar_payload = {
            "contract_version": LIVE_INFERENCE_CONTRACT_VERSION,
            "source_kind": SOURCE_KIND,
            "source_relative_path": source_relative_path,
            "published_at_utc": published_at_utc,
            "synthetic_captured_at_utc": published_at_utc,
            "sequence_index": sequence_index,
            "loop_index": loop_index,
            "frame_id": frame_id,
            "frame_hash": frame_hash.value,
            "frame_hash_algorithm": frame_hash.algorithm,
            "frame_hash_digest_size_bytes": frame_hash.digest_size_bytes,
            "byte_size": byte_size,
            "source_mtime_ns": source_mtime_ns,
            "output_image_path": str(frame_reference.image_path),
        }
        write_metadata_sidecar_atomic(
            sidecar_payload,
            self.metadata_path,
            self.temp_metadata_path,
        )
        self._current_index += 1
        return frame_reference

    def run_forever(self, stop_requested: Callable[[], bool] | None = None) -> None:
        """Publish frames until stopped or until a non-looping source is exhausted."""
        while True:
            if stop_requested is not None and stop_requested():
                return
            try:
                self.publish_next()
            except StopIteration:
                return
            if not self.config.loop and self._current_index >= len(self._source_images):
                return
            if stop_requested is not None and stop_requested():
                return
            self.sleep_fn(self.config.frame_interval_ms / 1000.0)

    def current_index(self) -> int:
        return self._current_index

    def source_count(self) -> int:
        return len(self._source_images)

    def _prepare_next_index(self) -> None:
        if self._current_index < len(self._source_images):
            return
        if not self.config.loop:
            raise StopIteration("Synthetic camera source list exhausted.")
        self._loop_index += 1
        if self.config.rescan_on_loop:
            self._source_images = discover_source_images(self.config, self.base_dir)
            self.source_root = _source_root(self.config, self.base_dir)
        self._current_index = 0


def _validate_relative_path(value: Path | str, *, label: str, reject_current: bool) -> Path:
    raw_text = str(value).strip()
    if not raw_text:
        raise ValueError(f"{label} must not be empty.")
    path = Path(raw_text)
    if reject_current and str(path) == ".":
        raise ValueError(f"{label} must name a relative source directory, not '.'.")
    if path.is_absolute():
        raise ValueError(f"{label} must be a relative path, got {value!r}.")
    if ".." in path.parts:
        raise ValueError(f"{label} must not contain '..': {value!r}.")
    return path


def _normalize_extensions(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        text = str(value).strip().lower().lstrip(".")
        if not text:
            raise ValueError("allowed_extensions must not contain empty values.")
        normalized.append(text)
    if not normalized:
        raise ValueError("allowed_extensions must not be empty.")
    return tuple(normalized)


def _coerce_sort_order(value: SyntheticCameraSortOrder | str) -> SyntheticCameraSortOrder:
    if isinstance(value, SyntheticCameraSortOrder):
        return value
    try:
        return SyntheticCameraSortOrder(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SyntheticCameraSortOrder)
        raise ValueError(f"Invalid synthetic camera sort_order={value!r}; expected one of {allowed}.") from exc


def _positive_int(value: int, label: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be > 0; got {value!r}.")
    return number


def _non_empty_filename(value: str, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be empty.")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"{label} must be a simple relative filename, got {value!r}.")
    return text


def _source_root(config: SyntheticCameraConfig, base_dir: Path | None) -> Path:
    root = Path.cwd() if base_dir is None else Path(base_dir)
    return root / config.source_dir


def _relative_source_path(path: Path, source_root: Path) -> str:
    return path.relative_to(source_root).as_posix()


def _sort_source_images(
    images: list[Path],
    source_root: Path,
    sort_order: SyntheticCameraSortOrder,
) -> list[Path]:
    def relative_key(path: Path) -> str:
        return _relative_source_path(path, source_root)

    if sort_order == SyntheticCameraSortOrder.NAME_ASCENDING:
        return sorted(images, key=relative_key)
    if sort_order == SyntheticCameraSortOrder.NAME_DESCENDING:
        return sorted(images, key=relative_key, reverse=True)
    if sort_order == SyntheticCameraSortOrder.MODIFIED_TIME_ASCENDING:
        return sorted(images, key=lambda path: (path.stat().st_mtime_ns, relative_key(path)))
    if sort_order == SyntheticCameraSortOrder.MODIFIED_TIME_DESCENDING:
        return sorted(images, key=lambda path: (-path.stat().st_mtime_ns, relative_key(path)))
    raise ValueError(f"Unsupported synthetic camera sort_order={sort_order!r}.")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "SOURCE_KIND",
    "SyntheticCameraConfig",
    "SyntheticCameraPublisher",
    "SyntheticCameraSortOrder",
    "discover_source_images",
    "load_synthetic_camera_config",
    "write_metadata_sidecar_atomic",
]
