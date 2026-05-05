"""GUI-framework-independent inference worker loop."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import time
from typing import Callable

from interfaces import (
    DebugImageReference,
    FrameFailureStage,
    FrameSkipReason,
    FrameSkipped,
    InferenceResult,
    InferenceWorkerCounters,
    InferenceWorkerEventSink,
    WorkerError,
    WorkerEventType,
    WorkerLifecycleEvent,
    WorkerName,
    WorkerState,
    WorkerStatus,
    WorkerWarning,
    is_allowed_worker_state_transition,
)
from live_inference.inference_core import (
    InferenceProcessingCore,
    InferenceProcessingOutcome,
)


_FAILURE_COUNTER_BY_STAGE = {
    FrameFailureStage.READ: "frames_failed_read",
    FrameFailureStage.DECODE: "frames_failed_decode",
    FrameFailureStage.PREPROCESS: "frames_failed_preprocess",
    FrameFailureStage.INFERENCE: "frames_failed_inference",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class InferenceWorker:
    """Blocking, stoppable work loop around ``InferenceProcessingCore``.

    Policy: recoverable ``WorkerError`` outcomes are emitted and the loop keeps
    polling. Non-recoverable ``WorkerError`` outcomes and unexpected exceptions
    move the worker to ``ERROR`` and exit the loop. The worker checks for stop
    requests only between ``process_once()`` calls, so in-flight model inference
    is allowed to complete.
    """

    def __init__(
        self,
        core: InferenceProcessingCore,
        event_sink: InferenceWorkerEventSink | None = None,
        *,
        poll_interval_ms: int = 10,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_utc_fn: Callable[[], str] | None = None,
    ) -> None:
        if poll_interval_ms < 0:
            raise ValueError("poll_interval_ms must be non-negative.")

        self._core = core
        self._event_sink = event_sink
        self._poll_interval_seconds = poll_interval_ms / 1000.0
        self._sleep_fn = sleep_fn
        self._now_utc_fn = now_utc_fn or _utc_now_iso

        self._state = WorkerState.STOPPED
        self._status_message = "Stopped."
        self._last_status_timestamp_utc = self._now_utc_fn()
        self._stop_requested = False
        self._counters = InferenceWorkerCounters()

    @property
    def worker_name(self) -> WorkerName:
        return WorkerName.INFERENCE

    def start_work(self) -> None:
        """Run the worker loop until stop is requested or a fatal error occurs."""
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
            "Inference worker is starting.",
        )

        if self._stop_requested or self._state == WorkerState.STOPPING:
            self._finish_stopping()
            return

        self._transition_to(
            WorkerState.RUNNING,
            WorkerEventType.STARTED,
            "Inference worker is running.",
        )

        while not self._stop_requested and self._state == WorkerState.RUNNING:
            try:
                outcome = self._core.process_once()
            except Exception as exc:
                self._handle_unexpected_exception(exc)
                return

            should_continue = self._dispatch_outcome(outcome)
            if not should_continue:
                return

            if self._stop_requested or self._state != WorkerState.RUNNING:
                break

            if self._poll_interval_seconds > 0:
                self._sleep_fn(self._poll_interval_seconds)

        if self._state in {WorkerState.RUNNING, WorkerState.STOPPING}:
            self._finish_stopping()

    def request_stop(self) -> None:
        """Request a clean stop after the current ``process_once()`` completes."""
        if self._state in {WorkerState.STARTING, WorkerState.RUNNING}:
            self._stop_requested = True
            self._transition_to(
                WorkerState.STOPPING,
                WorkerEventType.STOPPING,
                "Inference worker is stopping.",
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
                "Inference worker stopped after error.",
            )

    def current_status(self) -> WorkerStatus:
        return WorkerStatus(
            worker_name=self.worker_name,
            state=self._state,
            message=self._status_message,
            timestamp_utc=self._last_status_timestamp_utc,
            counters=self._counters.to_dict(),
        )

    def current_counters(self) -> InferenceWorkerCounters:
        return self._counters

    def _dispatch_outcome(self, outcome: InferenceProcessingOutcome) -> bool:
        if outcome.result is not None:
            self._dispatch_result(outcome.result)

        for image in outcome.debug_images:
            self._emit_debug_image_ready(image)

        if outcome.skipped is not None:
            self._dispatch_skipped(outcome.skipped)

        if outcome.warning is not None:
            self._dispatch_warning(outcome.warning)

        if outcome.error is not None:
            return self._dispatch_error(outcome.error)

        return True

    def _dispatch_result(self, result: InferenceResult) -> None:
        self._counters = replace(
            self._counters,
            frames_seen=self._counters.frames_seen + 1,
            frames_processed=self._counters.frames_processed + 1,
            last_input_hash=result.input_image_hash,
            last_inference_time_ms=result.inference_time_ms,
            last_preprocessing_time_ms=result.preprocessing_time_ms,
            last_total_time_ms=result.total_time_ms,
            last_result_time_utc=result.timestamp_utc,
        )
        self._emit_result_ready(result)

    def _dispatch_skipped(self, skipped: FrameSkipped) -> None:
        changes: dict[str, object] = {}

        if skipped.frame is not None or skipped.frame_hash is not None:
            changes["frames_seen"] = self._counters.frames_seen + 1

        if skipped.frame_hash is not None:
            changes["last_input_hash"] = skipped.frame_hash

        if skipped.reason == FrameSkipReason.DUPLICATE_HASH:
            changes["frames_skipped_duplicate"] = (
                self._counters.frames_skipped_duplicate + 1
            )

        if changes:
            self._counters = replace(self._counters, **changes)

        self._emit_frame_skipped(skipped)

    def _dispatch_warning(self, warning: WorkerWarning) -> None:
        self._record_failure_stage(warning.failure_stage)
        self._record_observed_frame(warning.frame)
        self._counters = replace(self._counters, last_error=warning.message)
        self._emit_warning_occurred(warning)

    def _dispatch_error(self, error: WorkerError) -> bool:
        self._record_failure_stage(error.failure_stage)
        self._record_observed_frame(error.frame)
        self._counters = replace(self._counters, last_error=error.message)

        if error.recoverable:
            self._emit_error_occurred(error)
            return True

        self._transition_to(
            WorkerState.ERROR,
            WorkerEventType.ERROR_OCCURRED,
            error.message,
        )
        self._emit_error_occurred(error)
        return False

    def _record_failure_stage(self, failure_stage: FrameFailureStage | None) -> None:
        counter_name = _FAILURE_COUNTER_BY_STAGE.get(failure_stage)
        if counter_name is None:
            # TODO: contracts do not currently expose a counter for OUTPUT stage.
            return
        self._counters = replace(
            self._counters,
            **{counter_name: getattr(self._counters, counter_name) + 1},
        )

    def _record_observed_frame(self, frame: object | None) -> None:
        if frame is None:
            return

        changes: dict[str, object] = {"frames_seen": self._counters.frames_seen + 1}
        frame_hash = getattr(frame, "frame_hash", None)
        if frame_hash is not None:
            changes["last_input_hash"] = frame_hash
        self._counters = replace(self._counters, **changes)

    def _handle_unexpected_exception(self, exc: Exception) -> None:
        error = WorkerError(
            worker_name=self.worker_name,
            error_type="unexpected_exception",
            message=f"Unexpected inference worker exception: {exc}",
            recoverable=False,
            timestamp_utc=self._now_utc_fn(),
            details={"exception_type": type(exc).__name__},
        )
        self._counters = replace(self._counters, last_error=error.message)
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
                "Inference worker is stopping.",
            )

        if self._state == WorkerState.STOPPING:
            self._transition_to(
                WorkerState.STOPPED,
                WorkerEventType.STOPPED,
                "Inference worker stopped.",
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
                f"Invalid inference worker state transition: {self._state} -> {state}"
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

    def _emit_result_ready(self, result: InferenceResult) -> None:
        if self._event_sink is not None:
            self._event_sink.result_ready(result)

    def _emit_frame_skipped(self, skipped: FrameSkipped) -> None:
        if self._event_sink is not None:
            self._event_sink.frame_skipped(skipped)

    def _emit_debug_image_ready(self, image: DebugImageReference) -> None:
        if self._event_sink is not None:
            self._event_sink.debug_image_ready(image)


__all__ = ["InferenceWorker"]
