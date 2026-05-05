"""GUI-framework-independent synthetic camera worker loop."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import time
from typing import Callable, Protocol

from interfaces import (
    CameraWorkerCounters,
    CameraWorkerEventSink,
    FrameReference,
    WorkerError,
    WorkerEventType,
    WorkerLifecycleEvent,
    WorkerName,
    WorkerState,
    WorkerStatus,
    WorkerWarning,
    is_allowed_worker_state_transition,
)


class _FramePublisher(Protocol):
    def publish_next(self) -> FrameReference:
        ...


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CameraWorker:
    """Blocking, stoppable work loop around a synthetic frame publisher.

    The worker owns no thread. Callers that need threading should run
    ``start_work()`` in their own adapter and call ``request_stop()`` from the
    controlling side.
    """

    def __init__(
        self,
        publisher: _FramePublisher,
        event_sink: CameraWorkerEventSink | None = None,
        *,
        now_utc_fn: Callable[[], str] | None = None,
    ) -> None:
        if not callable(getattr(publisher, "publish_next", None)):
            raise TypeError("publisher must provide a callable publish_next().")

        self._publisher = publisher
        self._event_sink = event_sink
        self._now_utc_fn = now_utc_fn or _utc_now_iso
        self._sleep_fn = _publisher_sleep_fn(publisher)
        self._frame_interval_seconds = _publisher_frame_interval_seconds(publisher)

        self._state = WorkerState.STOPPED
        self._status_message = "Stopped."
        self._last_status_timestamp_utc = self._now_utc_fn()
        self._stop_requested = False
        self._counters = CameraWorkerCounters()

    @property
    def worker_name(self) -> WorkerName:
        return WorkerName.CAMERA

    def start_work(self) -> None:
        """Publish frames until stopped, exhausted, or a fatal error occurs."""
        if self._state in {
            WorkerState.STARTING,
            WorkerState.RUNNING,
            WorkerState.STOPPING,
            WorkerState.ERROR,
        }:
            return

        self._stop_requested = False
        self._transition_to(
            WorkerState.STARTING,
            WorkerEventType.STARTING,
            "Camera worker is starting.",
        )

        if self._stop_requested or self._state == WorkerState.STOPPING:
            self._finish_stopping()
            return

        self._transition_to(
            WorkerState.RUNNING,
            WorkerEventType.STARTED,
            "Camera worker is running.",
        )

        while not self._stop_requested and self._state == WorkerState.RUNNING:
            try:
                frame = self._publisher.publish_next()
            except StopIteration:
                self._finish_stopping()
                return
            except Exception as exc:
                self._handle_unexpected_exception(exc)
                return

            self._record_successful_publish(frame)
            self._emit_frame_written(frame)

            if self._stop_requested or self._state != WorkerState.RUNNING:
                break

            if self._frame_interval_seconds > 0:
                self._sleep_fn(self._frame_interval_seconds)

        if self._state in {WorkerState.RUNNING, WorkerState.STOPPING}:
            self._finish_stopping()

    def request_stop(self) -> None:
        """Request a clean stop after the current publish/sleep point."""
        if self._state in {WorkerState.STARTING, WorkerState.RUNNING}:
            self._stop_requested = True
            self._transition_to(
                WorkerState.STOPPING,
                WorkerEventType.STOPPING,
                "Camera worker is stopping.",
            )
            return

        if self._state == WorkerState.STOPPING:
            self._stop_requested = True
            return

        if self._state == WorkerState.ERROR:
            self._stop_requested = True
            self._transition_to(
                WorkerState.STOPPED,
                WorkerEventType.STOPPED,
                "Camera worker stopped after error.",
            )

    def current_status(self) -> WorkerStatus:
        return WorkerStatus(
            worker_name=self.worker_name,
            state=self._state,
            message=self._status_message,
            timestamp_utc=self._last_status_timestamp_utc,
            counters=self._counters.to_dict(),
        )

    def current_counters(self) -> CameraWorkerCounters:
        return self._counters

    def _record_successful_publish(self, frame: FrameReference) -> None:
        write_time_utc = (
            frame.completed_at_utc
            or frame.metadata.written_at_utc
            or self._now_utc_fn()
        )
        self._counters = replace(
            self._counters,
            frames_captured=self._counters.frames_captured + 1,
            frames_written=self._counters.frames_written + 1,
            last_frame_write_time_utc=write_time_utc,
            last_frame_path=frame.image_path,
        )

    def _handle_unexpected_exception(self, exc: Exception) -> None:
        error = WorkerError(
            worker_name=self.worker_name,
            error_type="unexpected_exception",
            message=f"Unexpected camera worker exception: {exc}",
            recoverable=False,
            timestamp_utc=self._now_utc_fn(),
            details={"exception_type": type(exc).__name__},
        )
        self._counters = replace(
            self._counters,
            frame_write_failures=self._counters.frame_write_failures + 1,
            last_error=error.message,
        )
        self._transition_to(
            WorkerState.ERROR,
            WorkerEventType.ERROR_OCCURRED,
            error.message,
        )
        self._emit_error_occurred(error)

    def _finish_stopping(self) -> None:
        if self._state == WorkerState.RUNNING:
            self._transition_to(
                WorkerState.STOPPING,
                WorkerEventType.STOPPING,
                "Camera worker is stopping.",
            )

        if self._state == WorkerState.STOPPING:
            self._transition_to(
                WorkerState.STOPPED,
                WorkerEventType.STOPPED,
                "Camera worker stopped.",
            )

        self._stop_requested = False

    def _transition_to(
        self,
        state: WorkerState,
        event_type: WorkerEventType,
        message: str,
    ) -> None:
        if self._state == state:
            return

        if not is_allowed_worker_state_transition(self._state, state):
            raise RuntimeError(
                f"Invalid camera worker state transition: {self._state} -> {state}"
            )

        timestamp_utc = self._now_utc_fn()
        self._state = state
        self._status_message = message
        self._last_status_timestamp_utc = timestamp_utc

        self._emit_lifecycle_event(
            WorkerLifecycleEvent(
                worker_name=self.worker_name,
                event_type=event_type,
                state=state,
                timestamp_utc=timestamp_utc,
                message=message,
            )
        )
        self._emit_status_changed(self.current_status())

    def _emit_status_changed(self, status: WorkerStatus) -> None:
        if self._event_sink is not None:
            self._event_sink.status_changed(status)

    def _emit_lifecycle_event(self, event: WorkerLifecycleEvent) -> None:
        if self._event_sink is not None:
            self._event_sink.lifecycle_event(event)

    def _emit_warning_occurred(self, warning: WorkerWarning) -> None:
        if self._event_sink is not None:
            self._event_sink.warning_occurred(warning)

    def _emit_error_occurred(self, error: WorkerError) -> None:
        if self._event_sink is not None:
            self._event_sink.error_occurred(error)

    def _emit_frame_written(self, frame: FrameReference) -> None:
        if self._event_sink is not None:
            self._event_sink.frame_written(frame)


def _publisher_sleep_fn(publisher: object) -> Callable[[float], None]:
    sleep_fn = getattr(publisher, "sleep_fn", None)
    if sleep_fn is None:
        return time.sleep
    if not callable(sleep_fn):
        raise TypeError("publisher.sleep_fn must be callable when provided.")
    return sleep_fn


def _publisher_frame_interval_seconds(publisher: object) -> float:
    config = getattr(publisher, "config", None)
    if config is None:
        return 0.0

    frame_interval_ms = getattr(config, "frame_interval_ms", 0)
    try:
        seconds = int(frame_interval_ms) / 1000.0
    except (TypeError, ValueError) as exc:
        raise ValueError("publisher.config.frame_interval_ms must be an integer.") from exc

    if seconds < 0:
        raise ValueError("publisher.config.frame_interval_ms must be non-negative.")
    return seconds


__all__ = ["CameraWorker"]
