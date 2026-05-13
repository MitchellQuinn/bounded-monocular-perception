"""Atomic latest-frame file handoff services for live inference."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path

from interfaces import (
    DEFAULT_FRAME_HASH_ALGORITHM,
    DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES,
    FrameHash,
    FrameHandoffPaths,
    FrameMetadata,
    FrameReference,
    LiveInferenceConfig,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_from_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_handoff_paths(config_or_paths: LiveInferenceConfig | FrameHandoffPaths) -> FrameHandoffPaths:
    if isinstance(config_or_paths, LiveInferenceConfig):
        paths = config_or_paths.handoff_paths
    elif isinstance(config_or_paths, FrameHandoffPaths):
        paths = config_or_paths
    else:
        raise TypeError(
            "Expected LiveInferenceConfig or FrameHandoffPaths; "
            f"got {type(config_or_paths).__name__}."
        )
    return FrameHandoffPaths(
        frame_dir=Path(paths.frame_dir),
        latest_frame_filename=str(paths.latest_frame_filename),
        temp_frame_filename=str(paths.temp_frame_filename),
    )


def _hash_settings(
    config_or_paths: LiveInferenceConfig | FrameHandoffPaths,
) -> tuple[str, int]:
    if isinstance(config_or_paths, LiveInferenceConfig):
        return (
            str(config_or_paths.frame_hash_algorithm),
            int(config_or_paths.frame_hash_digest_size_bytes),
        )
    return DEFAULT_FRAME_HASH_ALGORITHM, DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES


def compute_frame_hash(
    image_bytes: bytes,
    *,
    algorithm: str = DEFAULT_FRAME_HASH_ALGORITHM,
    digest_size_bytes: int = DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES,
) -> FrameHash:
    """Hash the exact supplied image bytes using BLAKE2b."""
    if not str(algorithm).startswith("blake2b"):
        raise ValueError(f"Unsupported frame hash algorithm: {algorithm!r}")
    if int(digest_size_bytes) <= 0:
        raise ValueError(f"digest_size_bytes must be positive; got {digest_size_bytes!r}")
    digest = hashlib.blake2b(
        image_bytes,
        digest_size=int(digest_size_bytes),
    ).hexdigest()
    return FrameHash(
        value=digest,
        algorithm=str(algorithm),
        digest_size_bytes=int(digest_size_bytes),
    )


class AtomicFrameHandoffWriter:
    """Publish image bytes using atomic latest-frame replacement."""

    def __init__(self, config_or_paths: LiveInferenceConfig | FrameHandoffPaths) -> None:
        self.handoff_paths = _resolve_handoff_paths(config_or_paths)
        self.frame_hash_algorithm, self.frame_hash_digest_size_bytes = _hash_settings(
            config_or_paths
        )
        self._validate_same_directory_paths()

    @property
    def latest_frame_path(self) -> Path:
        return self.handoff_paths.latest_frame_path

    @property
    def temp_frame_path(self) -> Path:
        return self.handoff_paths.temp_frame_path

    def publish_frame(self, image_bytes: bytes, metadata: FrameMetadata) -> FrameReference:
        """Write temp bytes, close them, then atomically replace the latest frame."""
        self.handoff_paths.frame_dir.mkdir(parents=True, exist_ok=True)
        completed_at_utc = _utc_now_iso()
        byte_size = len(image_bytes)

        with self.temp_frame_path.open("wb") as handle:
            handle.write(image_bytes)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(self.temp_frame_path, self.latest_frame_path)

        return FrameReference(
            image_path=self.latest_frame_path,
            metadata=_metadata_with_write_details(
                metadata,
                byte_size=byte_size,
                written_at_utc=completed_at_utc,
            ),
            completed_at_utc=completed_at_utc,
            frame_hash=compute_frame_hash(
                image_bytes,
                algorithm=self.frame_hash_algorithm,
                digest_size_bytes=self.frame_hash_digest_size_bytes,
            ),
            byte_size=byte_size,
            handoff_paths=self.handoff_paths,
        )

    def _validate_same_directory_paths(self) -> None:
        if self.temp_frame_path.parent != self.latest_frame_path.parent:
            raise ValueError(
                "Atomic frame handoff requires temp and latest paths to share a directory: "
                f"temp={self.temp_frame_path} latest={self.latest_frame_path}"
            )


class LatestFrameHandoffReader:
    """Read only the completed latest-frame file from a handoff directory."""

    def __init__(self, config_or_paths: LiveInferenceConfig | FrameHandoffPaths) -> None:
        self.handoff_paths = _resolve_handoff_paths(config_or_paths)
        self._validate_same_directory_paths()

    @property
    def latest_frame_path(self) -> Path:
        return self.handoff_paths.latest_frame_path

    @property
    def temp_frame_path(self) -> Path:
        return self.handoff_paths.temp_frame_path

    def latest_completed_frame(self) -> FrameReference | None:
        try:
            stat_result = self.latest_frame_path.stat()
        except FileNotFoundError:
            return None
        return FrameReference(
            image_path=self.latest_frame_path,
            completed_at_utc=_utc_from_timestamp(stat_result.st_mtime),
            byte_size=int(stat_result.st_size),
            handoff_paths=self.handoff_paths,
        )

    def read_frame_bytes(self, frame: FrameReference) -> bytes:
        return Path(frame.image_path).read_bytes()

    def _validate_same_directory_paths(self) -> None:
        if self.temp_frame_path.parent != self.latest_frame_path.parent:
            raise ValueError(
                "Atomic frame handoff requires temp and latest paths to share a directory: "
                f"temp={self.temp_frame_path} latest={self.latest_frame_path}"
            )


def _metadata_with_write_details(
    metadata: FrameMetadata,
    *,
    byte_size: int,
    written_at_utc: str,
) -> FrameMetadata:
    if metadata.byte_size == byte_size and metadata.written_at_utc is not None:
        return metadata
    return replace(
        metadata,
        byte_size=metadata.byte_size if metadata.byte_size is not None else byte_size,
        written_at_utc=(
            metadata.written_at_utc if metadata.written_at_utc is not None else written_at_utc
        ),
    )


__all__ = [
    "AtomicFrameHandoffWriter",
    "LatestFrameHandoffReader",
    "compute_frame_hash",
]
