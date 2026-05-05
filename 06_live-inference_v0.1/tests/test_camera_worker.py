"""Unit tests for the plain Python synthetic camera worker loop."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Callable
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cameras.synthetic_camera import (  # noqa: E402
    SyntheticCameraConfig,
    SyntheticCameraPublisher,
)
from interfaces import (  # noqa: E402
    FrameReference,
    WorkerError,
    WorkerLifecycleEvent,
    WorkerName,
    WorkerState,
    WorkerStatus,
    WorkerWarning,
)
from live_inference.workers import CameraWorker  # noqa: E402


NOW = "2026-05-04T10:00:00Z"
FRAME_WRITTEN_AT = "2026-05-04T10:00:01Z"


class CameraWorkerTests(unittest.TestCase):
    def test_initial_status_is_stopped(self) -> None:
        worker = CameraWorker(FakePublisher(), now_utc_fn=lambda: NOW)

        self.assertEqual(worker.current_status().state, WorkerState.STOPPED)

    def test_worker_name_is_camera(self) -> None:
        worker = CameraWorker(FakePublisher(), now_utc_fn=lambda: NOW)

        self.assertEqual(worker.worker_name, WorkerName.CAMERA)

    def test_start_work_transitions_starting_running_stopping_stopped(self) -> None:
        sink = RecordingSink()
        worker = CameraWorker(
            FakePublisher(frames=[_frame()]),
            sink,
            now_utc_fn=lambda: NOW,
        )

        worker.start_work()

        expected_states = [
            WorkerState.STARTING,
            WorkerState.RUNNING,
            WorkerState.STOPPING,
            WorkerState.STOPPED,
        ]
        self.assertEqual([status.state for status in sink.statuses], expected_states)
        self.assertEqual(
            [event.state for event in sink.lifecycle_events],
            expected_states,
        )
        self.assertEqual(worker.current_status().state, WorkerState.STOPPED)

    def test_request_stop_is_idempotent(self) -> None:
        worker_ref: dict[str, CameraWorker] = {}

        def stop_twice() -> None:
            worker_ref["worker"].request_stop()
            worker_ref["worker"].request_stop()

        worker = CameraWorker(
            FakePublisher(frames=[_frame()], on_publish=stop_twice),
            now_utc_fn=lambda: NOW,
        )
        worker_ref["worker"] = worker

        worker.request_stop()
        worker.request_stop()
        worker.start_work()
        worker.request_stop()
        worker.request_stop()

        self.assertEqual(worker.current_status().state, WorkerState.STOPPED)

    def test_successful_publish_emits_frame_written(self) -> None:
        frame = _frame()
        sink, worker = _worker_with_single_frame(frame)

        worker.start_work()

        self.assertEqual(sink.frames, [frame])

    def test_successful_publish_increments_frames_written(self) -> None:
        _, worker = _worker_with_single_frame(_frame())

        worker.start_work()

        self.assertEqual(worker.current_counters().frames_written, 1)

    def test_successful_publish_updates_last_frame_path(self) -> None:
        frame = _frame(Path("live_frames/camera_latest.png"))
        _, worker = _worker_with_single_frame(frame)

        worker.start_work()

        self.assertEqual(
            worker.current_counters().last_frame_path,
            Path("live_frames/camera_latest.png"),
        )

    def test_publisher_stop_iteration_stops_cleanly(self) -> None:
        sink = RecordingSink()
        worker = CameraWorker(FakePublisher(), sink, now_utc_fn=lambda: NOW)

        worker.start_work()

        self.assertEqual(worker.current_status().state, WorkerState.STOPPED)
        self.assertEqual(sink.errors, [])

    def test_publisher_exception_emits_worker_error(self) -> None:
        sink = RecordingSink()
        worker = CameraWorker(
            FakePublisher(exception=RuntimeError("camera publish failed")),
            sink,
            now_utc_fn=lambda: NOW,
        )

        worker.start_work()

        self.assertEqual(worker.current_status().state, WorkerState.ERROR)
        self.assertEqual(len(sink.errors), 1)
        self.assertEqual(sink.errors[0].worker_name, WorkerName.CAMERA)
        self.assertEqual(sink.errors[0].error_type, "unexpected_exception")
        self.assertFalse(sink.errors[0].recoverable)

    def test_publisher_exception_updates_failure_counters(self) -> None:
        worker = CameraWorker(
            FakePublisher(exception=RuntimeError("write failed")),
            now_utc_fn=lambda: NOW,
        )

        worker.start_work()

        counters = worker.current_counters()
        self.assertEqual(counters.frame_write_failures, 1)
        self.assertEqual(
            counters.last_error,
            "Unexpected camera worker exception: write failed",
        )

    def test_camera_worker_module_has_no_pyside6_import(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "workers" / "camera_worker.py"
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

    def test_worker_unit_path_uses_fake_publisher_without_real_camera(self) -> None:
        sink, worker = _worker_with_single_frame(_frame())

        worker.start_work()

        self.assertEqual(len(sink.frames), 1)

    def test_stop_closes_publisher_when_close_is_available(self) -> None:
        publisher = CloseablePublisher(frames=[_frame()])
        worker = CameraWorker(publisher, now_utc_fn=lambda: NOW)

        worker.start_work()

        self.assertEqual(publisher.close_calls, 1)

    def test_camera_worker_module_has_no_inference_model_preprocessor_imports(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "workers" / "camera_worker.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)

        self.assertNotIn("live_inference.inference_core", imported_modules)
        self.assertFalse(
            any(
                module.startswith("live_inference.preprocessing")
                for module in imported_modules
            )
        )
        self.assertFalse(
            any(module.startswith("live_inference.engines") for module in imported_modules)
        )
        self.assertNotIn("model", source.lower())

    def test_real_synthetic_camera_publisher_integration_is_fast_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            source_dir = base_dir / "source"
            source_dir.mkdir()
            (source_dir / "frame_001.png").write_bytes(b"synthetic-frame")
            config = SyntheticCameraConfig(
                source_dir=Path("source"),
                output_dir=Path("live_frames"),
                frame_interval_ms=1,
                loop=False,
            )
            publisher = SyntheticCameraPublisher(
                config,
                base_dir=base_dir,
                sleep_fn=lambda _seconds: None,
                now_utc_fn=lambda: FRAME_WRITTEN_AT,
            )
            sink = RecordingSink()
            worker = CameraWorker(publisher, sink, now_utc_fn=lambda: NOW)

            worker.start_work()

            self.assertEqual(worker.current_status().state, WorkerState.STOPPED)
            self.assertEqual(worker.current_counters().frames_written, 1)
            self.assertEqual(len(sink.frames), 1)
            self.assertTrue((base_dir / "live_frames" / "latest_frame.png").exists())


@dataclass
class FakePublisherConfig:
    frame_interval_ms: int = 0


class FakePublisher:
    def __init__(
        self,
        *,
        frames: list[FrameReference] | None = None,
        exception: Exception | None = None,
        on_publish: Callable[[], None] | None = None,
        frame_interval_ms: int = 0,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.frames = list(frames or [])
        self.exception = exception
        self.on_publish = on_publish
        self.calls = 0
        self.config = FakePublisherConfig(frame_interval_ms=frame_interval_ms)
        self.sleep_fn = sleep_fn or (lambda _seconds: None)

    def publish_next(self) -> FrameReference:
        self.calls += 1
        if self.on_publish is not None:
            self.on_publish()
        if self.exception is not None:
            raise self.exception
        if not self.frames:
            raise StopIteration("Synthetic camera source list exhausted.")
        return self.frames.pop(0)


class CloseablePublisher(FakePublisher):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class RecordingSink:
    def __init__(self) -> None:
        self.statuses: list[WorkerStatus] = []
        self.lifecycle_events: list[WorkerLifecycleEvent] = []
        self.frames: list[FrameReference] = []
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

    def frame_written(self, frame: FrameReference) -> None:
        self.frames.append(frame)


def _worker_with_single_frame(
    frame: FrameReference,
) -> tuple[RecordingSink, CameraWorker]:
    sink = RecordingSink()
    worker = CameraWorker(
        FakePublisher(frames=[frame]),
        sink,
        now_utc_fn=lambda: NOW,
    )
    return sink, worker


def _frame(path: Path | None = None) -> FrameReference:
    return FrameReference(
        image_path=path or Path("live_frames/latest_frame.png"),
        completed_at_utc=FRAME_WRITTEN_AT,
    )


if __name__ == "__main__":
    unittest.main()
