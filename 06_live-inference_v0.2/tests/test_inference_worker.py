"""Unit tests for the plain Python inference worker loop."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from typing import Callable
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from interfaces import (  # noqa: E402
    DebugImageReference,
    FrameFailureStage,
    FrameHash,
    FrameReference,
    FrameSkipped,
    FrameSkipReason,
    InferenceResult,
    WorkerError,
    WorkerLifecycleEvent,
    WorkerName,
    WorkerState,
    WorkerStatus,
    WorkerWarning,
)
from live_inference.inference_core import InferenceProcessingOutcome  # noqa: E402
from live_inference.workers import InferenceWorker  # noqa: E402


NOW = "2026-05-04T10:00:00Z"
RESULT_AT = "2026-05-04T10:00:01Z"


class InferenceWorkerTests(unittest.TestCase):
    def test_initial_status_is_stopped(self) -> None:
        worker = InferenceWorker(
            FakeCore(),  # type: ignore[arg-type]
            now_utc_fn=lambda: NOW,
        )

        self.assertEqual(worker.current_status().state, WorkerState.STOPPED)

    def test_worker_name_is_inference(self) -> None:
        worker = InferenceWorker(
            FakeCore(),  # type: ignore[arg-type]
            now_utc_fn=lambda: NOW,
        )

        self.assertEqual(worker.worker_name, WorkerName.INFERENCE)

    def test_start_work_transitions_starting_running_stopping_stopped(self) -> None:
        sink = RecordingSink()
        worker_ref: dict[str, InferenceWorker] = {}

        def stop_worker() -> None:
            worker_ref["worker"].request_stop()

        worker = InferenceWorker(
            FakeCore(
                outcomes=[InferenceProcessingOutcome(warning=_warning())],
                on_process=stop_worker,
            ),  # type: ignore[arg-type]
            sink,
            poll_interval_ms=0,
            now_utc_fn=lambda: NOW,
        )
        worker_ref["worker"] = worker

        worker.start_work()

        self.assertEqual(
            [status.state for status in sink.statuses],
            [
                WorkerState.STARTING,
                WorkerState.RUNNING,
                WorkerState.STOPPING,
                WorkerState.STOPPED,
            ],
        )
        self.assertEqual(worker.current_status().state, WorkerState.STOPPED)

    def test_request_stop_is_idempotent(self) -> None:
        worker_ref: dict[str, InferenceWorker] = {}

        def stop_twice() -> None:
            worker_ref["worker"].request_stop()
            worker_ref["worker"].request_stop()

        worker = InferenceWorker(
            FakeCore(
                outcomes=[InferenceProcessingOutcome(warning=_warning())],
                on_process=stop_twice,
            ),  # type: ignore[arg-type]
            poll_interval_ms=0,
            now_utc_fn=lambda: NOW,
        )
        worker_ref["worker"] = worker

        worker.request_stop()
        worker.request_stop()
        worker.start_work()
        worker.request_stop()
        worker.request_stop()

        self.assertEqual(worker.current_status().state, WorkerState.STOPPED)

    def test_result_outcome_emits_result_ready(self) -> None:
        result = _result()
        sink, worker = _worker_stopping_after_one_outcome(
            InferenceProcessingOutcome(result=result)
        )

        worker.start_work()

        self.assertEqual(sink.results, [result])

    def test_result_outcome_increments_frames_processed(self) -> None:
        _, worker = _worker_stopping_after_one_outcome(
            InferenceProcessingOutcome(result=_result())
        )

        worker.start_work()

        self.assertEqual(worker.current_counters().frames_processed, 1)

    def test_result_outcome_updates_timing_and_hash_counters(self) -> None:
        frame_hash = FrameHash("hash-with-timing")
        result = _result(frame_hash=frame_hash)
        _, worker = _worker_stopping_after_one_outcome(
            InferenceProcessingOutcome(result=result)
        )

        worker.start_work()

        counters = worker.current_counters()
        self.assertEqual(counters.last_input_hash, frame_hash)
        self.assertEqual(counters.last_inference_time_ms, 12.5)
        self.assertEqual(counters.last_preprocessing_time_ms, 3.25)
        self.assertEqual(counters.last_total_time_ms, 15.75)
        self.assertEqual(counters.last_result_time_utc, RESULT_AT)

    def test_skipped_duplicate_emits_frame_skipped_and_increments_counter(self) -> None:
        skipped = _duplicate_skipped()
        sink, worker = _worker_stopping_after_one_outcome(
            InferenceProcessingOutcome(skipped=skipped)
        )

        worker.start_work()

        self.assertEqual(sink.skipped, [skipped])
        self.assertEqual(worker.current_counters().frames_skipped_duplicate, 1)

    def test_warning_outcome_emits_warning_occurred(self) -> None:
        warning = _warning()
        sink, worker = _worker_stopping_after_one_outcome(
            InferenceProcessingOutcome(warning=warning)
        )

        worker.start_work()

        self.assertEqual(sink.warnings, [warning])

    def test_error_outcome_emits_error_and_increments_failure_counter(self) -> None:
        error = _error(failure_stage=FrameFailureStage.INFERENCE)
        sink, worker = _worker_stopping_after_one_outcome(
            InferenceProcessingOutcome(error=error)
        )

        worker.start_work()

        self.assertEqual(sink.errors, [error])
        self.assertEqual(worker.current_counters().frames_failed_inference, 1)

    def test_debug_images_are_emitted_via_debug_image_ready(self) -> None:
        image = _debug_image()
        sink, worker = _worker_stopping_after_one_outcome(
            InferenceProcessingOutcome(result=_result(), debug_images=(image,))
        )

        worker.start_work()

        self.assertEqual(sink.debug_images, [image])

    def test_poll_interval_uses_injected_sleep_fn(self) -> None:
        sink = RecordingSink()
        sleeps: list[float] = []
        worker_ref: dict[str, InferenceWorker] = {}

        def sleep_fn(seconds: float) -> None:
            sleeps.append(seconds)
            worker_ref["worker"].request_stop()

        worker = InferenceWorker(
            FakeCore(
                outcomes=[InferenceProcessingOutcome(warning=_warning())],
            ),  # type: ignore[arg-type]
            sink,
            poll_interval_ms=25,
            sleep_fn=sleep_fn,
            now_utc_fn=lambda: NOW,
        )
        worker_ref["worker"] = worker

        worker.start_work()

        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 0.025)

    def test_worker_exits_cleanly_when_stop_requested(self) -> None:
        _, worker = _worker_stopping_after_one_outcome(
            InferenceProcessingOutcome(warning=_warning())
        )

        worker.start_work()

        self.assertEqual(worker.current_status().state, WorkerState.STOPPED)

    def test_unexpected_core_exception_emits_worker_error_and_sets_error(self) -> None:
        sink = RecordingSink()
        worker = InferenceWorker(
            FakeCore(
                exception=RuntimeError("model loop failed"),
            ),  # type: ignore[arg-type]
            sink,
            poll_interval_ms=0,
            now_utc_fn=lambda: NOW,
        )

        worker.start_work()

        self.assertEqual(worker.current_status().state, WorkerState.ERROR)
        self.assertEqual(len(sink.errors), 1)
        self.assertEqual(sink.errors[0].error_type, "unexpected_exception")
        self.assertFalse(sink.errors[0].recoverable)

    def test_inference_worker_module_has_no_pyside6_import(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "workers" / "inference_worker.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        found_roots: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found_roots.update(
                    alias.name.split(".", maxsplit=1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                found_roots.add(node.module.split(".", maxsplit=1)[0])

        self.assertNotIn("PySide6", found_roots)
        self.assertNotIn("QThread", module_path.read_text(encoding="utf-8"))

    def test_worker_unit_path_uses_fake_core_without_model_artifacts(self) -> None:
        sink, worker = _worker_stopping_after_one_outcome(
            InferenceProcessingOutcome(result=_result())
        )

        worker.start_work()

        self.assertEqual(len(sink.results), 1)


class FakeCore:
    def __init__(
        self,
        *,
        outcomes: list[InferenceProcessingOutcome] | None = None,
        exception: Exception | None = None,
        on_process: Callable[[], None] | None = None,
    ) -> None:
        self.outcomes = list(outcomes or [])
        self.exception = exception
        self.on_process = on_process
        self.calls = 0

    def process_once(self) -> InferenceProcessingOutcome:
        self.calls += 1
        if self.on_process is not None:
            self.on_process()
        if self.exception is not None:
            raise self.exception
        if self.outcomes:
            return self.outcomes.pop(0)
        return InferenceProcessingOutcome(warning=_warning("idle"))


class RecordingSink:
    def __init__(self) -> None:
        self.statuses: list[WorkerStatus] = []
        self.lifecycle_events: list[WorkerLifecycleEvent] = []
        self.results: list[InferenceResult] = []
        self.skipped: list[FrameSkipped] = []
        self.debug_images: list[DebugImageReference] = []
        self.warnings: list[WorkerWarning] = []
        self.errors: list[WorkerError] = []

    def status_changed(self, status: WorkerStatus) -> None:
        self.statuses.append(status)

    def lifecycle_event(self, event: WorkerLifecycleEvent) -> None:
        self.lifecycle_events.append(event)

    def warning_occurred(self, warning: WorkerWarning) -> None:
        self.warnings.append(warning)

    def error_occurred(self, error: WorkerError) -> None:
        self.errors.append(error)

    def result_ready(self, result: InferenceResult) -> None:
        self.results.append(result)

    def frame_skipped(self, skipped: FrameSkipped) -> None:
        self.skipped.append(skipped)

    def debug_image_ready(self, image: DebugImageReference) -> None:
        self.debug_images.append(image)


def _worker_stopping_after_one_outcome(
    outcome: InferenceProcessingOutcome,
) -> tuple[RecordingSink, InferenceWorker]:
    sink = RecordingSink()
    worker_ref: dict[str, InferenceWorker] = {}

    def stop_worker() -> None:
        worker_ref["worker"].request_stop()

    worker = InferenceWorker(
        FakeCore(
            outcomes=[outcome],
            on_process=stop_worker,
        ),  # type: ignore[arg-type]
        sink,
        poll_interval_ms=0,
        now_utc_fn=lambda: NOW,
    )
    worker_ref["worker"] = worker
    return sink, worker


def _result(frame_hash: FrameHash | None = None) -> InferenceResult:
    return InferenceResult(
        request_id="request-1",
        input_image_path=Path("live_frames/latest_frame.png"),
        input_image_hash=frame_hash or FrameHash("hash-1"),
        timestamp_utc=RESULT_AT,
        predicted_distance_m=4.5,
        predicted_yaw_sin=0.0,
        predicted_yaw_cos=1.0,
        predicted_yaw_deg=0.0,
        inference_time_ms=12.5,
        preprocessing_time_ms=3.25,
        total_time_ms=15.75,
    )


def _duplicate_skipped() -> FrameSkipped:
    frame_hash = FrameHash("duplicate-hash")
    return FrameSkipped(
        worker_name=WorkerName.INFERENCE,
        reason=FrameSkipReason.DUPLICATE_HASH,
        timestamp_utc=NOW,
        frame=FrameReference(
            image_path=Path("live_frames/latest_frame.png"),
            frame_hash=frame_hash,
        ),
        frame_hash=frame_hash,
        message="Duplicate frame.",
    )


def _warning(message: str = "warning") -> WorkerWarning:
    return WorkerWarning(
        worker_name=WorkerName.INFERENCE,
        warning_type="test_warning",
        message=message,
        timestamp_utc=NOW,
    )


def _error(
    *,
    failure_stage: FrameFailureStage = FrameFailureStage.PREPROCESS,
    recoverable: bool = True,
) -> WorkerError:
    return WorkerError(
        worker_name=WorkerName.INFERENCE,
        error_type="test_error",
        message="error",
        recoverable=recoverable,
        timestamp_utc=NOW,
        failure_stage=failure_stage,
    )


def _debug_image() -> DebugImageReference:
    return DebugImageReference(
        request_id="request-1",
        image_kind="distance",
        path=Path("live_debug/distance.png"),
        created_at_utc=RESULT_AT,
        source_frame_hash=FrameHash("hash-1"),
    )


if __name__ == "__main__":
    unittest.main()
