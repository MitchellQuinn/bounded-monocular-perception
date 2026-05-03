"""Latest-frame selection core for live inference polling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable
import uuid

from interfaces import (
    FrameFailureStage,
    FrameHash,
    FrameHandoffReader,
    FrameReference,
    FrameSkipped,
    FrameSkipReason,
    InferenceRequest,
    WorkerName,
    WorkerWarning,
)
from live_inference.frame_handoff import compute_frame_hash


@dataclass(frozen=True)
class SelectedFrameForInference:
    """A selected request plus the exact bytes that produced its hash."""

    request: InferenceRequest
    image_bytes: bytes
    frame_hash: FrameHash


@dataclass(frozen=True)
class FrameSelectionResult:
    """Outcome from one latest-frame selection attempt."""

    selected: SelectedFrameForInference | None = None
    skipped: FrameSkipped | None = None
    warning: WorkerWarning | None = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_request_id() -> str:
    return str(uuid.uuid4())


class InferenceFrameSelector:
    """Select latest completed handoff frames for inference requests."""

    def __init__(
        self,
        reader: FrameHandoffReader,
        *,
        duplicate_hash_skip_enabled: bool = True,
        request_id_factory: Callable[[], str] | None = None,
        now_utc_fn: Callable[[], str] | None = None,
    ) -> None:
        self._reader = reader
        self._duplicate_hash_skip_enabled = duplicate_hash_skip_enabled
        self._request_id_factory = request_id_factory or _default_request_id
        self._now_utc_fn = now_utc_fn or _utc_now_iso
        self._last_processed_hash: FrameHash | None = None

    def select_latest(self) -> FrameSelectionResult:
        """Return an inference request for the newest completed non-duplicate frame."""
        timestamp_utc = self._now_utc_fn()
        frame = self._reader.latest_completed_frame()
        if frame is None:
            return FrameSelectionResult(
                skipped=FrameSkipped(
                    worker_name=WorkerName.INFERENCE,
                    reason=FrameSkipReason.MISSING_FILE,
                    timestamp_utc=timestamp_utc,
                    message="No completed frame is available.",
                )
            )

        try:
            image_bytes = self._reader.read_frame_bytes(frame)
        except Exception as exc:
            return FrameSelectionResult(
                warning=WorkerWarning(
                    worker_name=WorkerName.INFERENCE,
                    warning_type="frame_read_failed",
                    message=f"Failed to read completed frame: {exc}",
                    timestamp_utc=timestamp_utc,
                    frame=frame,
                    failure_stage=FrameFailureStage.READ,
                    details={"exception_type": type(exc).__name__},
                )
            )

        frame_hash = compute_frame_hash(image_bytes)
        updated_frame = _frame_with_hash_and_size(
            frame,
            frame_hash=frame_hash,
            byte_size=len(image_bytes),
        )

        if (
            self._duplicate_hash_skip_enabled
            and self._last_processed_hash is not None
            and frame_hash == self._last_processed_hash
        ):
            return FrameSelectionResult(
                skipped=FrameSkipped(
                    worker_name=WorkerName.INFERENCE,
                    reason=FrameSkipReason.DUPLICATE_HASH,
                    timestamp_utc=timestamp_utc,
                    frame=updated_frame,
                    frame_hash=frame_hash,
                    message="Frame hash matches the last successfully processed frame.",
                )
            )

        request = InferenceRequest(
            request_id=self._request_id_factory(),
            frame=updated_frame,
            requested_at_utc=timestamp_utc,
            duplicate_hash_skip_enabled=self._duplicate_hash_skip_enabled,
        )
        return FrameSelectionResult(
            selected=SelectedFrameForInference(
                request=request,
                image_bytes=image_bytes,
                frame_hash=frame_hash,
            )
        )

    def select(self) -> FrameSelectionResult:
        """Alias for callers that treat frame selection as one polling step."""
        return self.select_latest()

    def poll(self) -> FrameSelectionResult:
        """Alias for polling-loop callers."""
        return self.select_latest()

    def select_latest_frame(self) -> FrameSelectionResult:
        """Alias naming the selected resource explicitly."""
        return self.select_latest()

    def mark_processed(self, frame_hash: FrameHash) -> None:
        """Record a frame hash after downstream inference completes successfully."""
        self._last_processed_hash = frame_hash

    def last_processed_hash(self) -> FrameHash | None:
        return self._last_processed_hash

    def reset(self) -> None:
        self._last_processed_hash = None


def _frame_with_hash_and_size(
    frame: FrameReference,
    *,
    frame_hash: FrameHash,
    byte_size: int,
) -> FrameReference:
    return replace(
        frame,
        frame_hash=frame_hash,
        byte_size=byte_size,
    )


__all__ = [
    "FrameSelectionResult",
    "InferenceFrameSelector",
    "SelectedFrameForInference",
]
