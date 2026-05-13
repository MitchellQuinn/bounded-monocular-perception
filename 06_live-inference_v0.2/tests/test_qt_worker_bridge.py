"""Tests for the PySide6 worker-thread bridge."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
import unittest
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class QtWorkerEventSinkTests(unittest.TestCase):
    def test_status_changed_forwards_to_qt_signal(self) -> None:
        _, QtWorkerEventSink, WorkerQtSignals, _ = _qt_bridge()
        signals = WorkerQtSignals()
        sink = QtWorkerEventSink(signals)
        payload = _Payload(state="RUNNING")
        received: list[object] = []
        signals.status_changed.connect(received.append)

        sink.status_changed(payload)

        self.assertEqual(received, [payload])

    def test_lifecycle_event_forwards_to_qt_signal(self) -> None:
        _, QtWorkerEventSink, WorkerQtSignals, _ = _qt_bridge()
        signals = WorkerQtSignals()
        sink = QtWorkerEventSink(signals)
        payload = _Payload(state="STOPPED", event_type="stopped")
        received: list[object] = []
        signals.lifecycle_event.connect(received.append)

        sink.lifecycle_event(payload)

        self.assertEqual(received, [payload])

    def test_error_occurred_forwards_to_qt_signal(self) -> None:
        _, QtWorkerEventSink, WorkerQtSignals, _ = _qt_bridge()
        signals = WorkerQtSignals()
        sink = QtWorkerEventSink(signals)
        payload = _Payload(message="worker failed")
        received: list[object] = []
        signals.error_occurred.connect(received.append)

        sink.error_occurred(payload)

        self.assertEqual(received, [payload])

    def test_camera_frame_written_forwards_to_qt_signal(self) -> None:
        _, QtWorkerEventSink, WorkerQtSignals, _ = _qt_bridge()
        signals = WorkerQtSignals()
        sink = QtWorkerEventSink(signals)
        payload = _Payload(message="frame")
        received: list[object] = []
        signals.frame_written.connect(received.append)

        sink.frame_written(payload)

        self.assertEqual(received, [payload])

    def test_inference_result_ready_forwards_to_qt_signal(self) -> None:
        _, QtWorkerEventSink, WorkerQtSignals, _ = _qt_bridge()
        signals = WorkerQtSignals()
        sink = QtWorkerEventSink(signals)
        payload = _Payload(message="result")
        received: list[object] = []
        signals.result_ready.connect(received.append)

        sink.result_ready(payload)

        self.assertEqual(received, [payload])


class WorkerThreadControllerTests(unittest.TestCase):
    def test_controller_can_run_fake_plain_worker_that_stops_immediately(self) -> None:
        _, _, _, WorkerThreadController = _qt_bridge()
        _core_application()
        worker = _ImmediateStopWorker()
        controller = WorkerThreadController(worker)
        lifecycle_events: list[object] = []
        controller.signals.lifecycle_event.connect(lifecycle_events.append)

        controller.start()

        self.assertTrue(controller.wait(1000))
        _process_events()
        self.assertTrue(worker.started.is_set())
        self.assertFalse(controller.is_running())
        self.assertIsNone(controller.last_exception)
        self.assertEqual(
            [event.state for event in lifecycle_events],
            ["RUNNING", "STOPPED"],
        )

    def test_request_stop_is_safe_and_idempotent(self) -> None:
        _, _, _, WorkerThreadController = _qt_bridge()
        _core_application()
        worker = _BlockingWorker()
        controller = WorkerThreadController(worker)

        controller.start()
        self.assertTrue(_wait_until(worker.started.is_set))

        controller.request_stop()
        controller.request_stop()

        self.assertTrue(controller.wait(1000))
        controller.request_stop()
        _process_events()

        self.assertFalse(controller.is_running())
        self.assertGreaterEqual(worker.stop_requests, 3)


class QtWorkerBridgeImportHygieneTests(unittest.TestCase):
    def test_worker_thread_controller_module_does_not_import_gui_widgets(self) -> None:
        module_path = (
            SRC_ROOT / "live_inference" / "gui" / "qt_worker_bridge.py"
        )
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        pyside_modules: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pyside_modules.update(
                    alias.name for alias in node.names if alias.name.startswith("PySide6")
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("PySide6"):
                    pyside_modules.add(node.module)

        self.assertEqual(pyside_modules, {"PySide6.QtCore"})
        self.assertNotIn("QtWidgets", source)
        self.assertNotIn("QApplication", source)

    def test_backend_modules_remain_pyside6_free(self) -> None:
        offenders: dict[str, set[str]] = {}

        for module_path in SRC_ROOT.rglob("*.py"):
            relative_parts = module_path.relative_to(SRC_ROOT).parts
            if relative_parts[:2] == ("live_inference", "gui"):
                continue

            imported_roots = _imported_roots(module_path)
            if "PySide6" in imported_roots:
                offenders[str(module_path)] = imported_roots

        self.assertEqual(offenders, {})


@dataclass(frozen=True)
class _Payload:
    state: object | None = None
    event_type: object | None = None
    message: str = ""


class _ImmediateStopWorker:
    def __init__(self) -> None:
        self._event_sink: object | None = None
        self.started = threading.Event()

    def start_work(self) -> None:
        self.started.set()
        assert self._event_sink is not None
        self._event_sink.lifecycle_event(_Payload(state="RUNNING", event_type="started"))
        self._event_sink.status_changed(_Payload(state="RUNNING"))
        self._event_sink.lifecycle_event(_Payload(state="STOPPED", event_type="stopped"))
        self._event_sink.status_changed(_Payload(state="STOPPED"))

    def request_stop(self) -> None:
        pass

    def current_status(self) -> _Payload:
        return _Payload(state="STOPPED")

    def current_counters(self) -> dict[str, object]:
        return {}


class _BlockingWorker:
    def __init__(self) -> None:
        self._event_sink: object | None = None
        self.started = threading.Event()
        self._lock = threading.Lock()
        self._stop_requested = False
        self.stop_requests = 0

    def start_work(self) -> None:
        self.started.set()
        while True:
            with self._lock:
                if self._stop_requested:
                    break
            time.sleep(0.001)

        assert self._event_sink is not None
        self._event_sink.lifecycle_event(_Payload(state="STOPPED", event_type="stopped"))
        self._event_sink.status_changed(_Payload(state="STOPPED"))

    def request_stop(self) -> None:
        with self._lock:
            self.stop_requests += 1
            self._stop_requested = True

    def current_status(self) -> _Payload:
        return _Payload(state="STOPPED" if self._stop_requested else "RUNNING")

    def current_counters(self) -> dict[str, object]:
        return {}


_QT_BRIDGE_IMPORTS: tuple[Any, Any, Any, Any] | None = None
_PYSIDE6_IMPORT_ERROR: ImportError | None = None


def _qt_bridge() -> tuple[Any, Any, Any, Any]:
    global _PYSIDE6_IMPORT_ERROR, _QT_BRIDGE_IMPORTS

    if _QT_BRIDGE_IMPORTS is not None:
        return _QT_BRIDGE_IMPORTS
    if _PYSIDE6_IMPORT_ERROR is not None:
        raise unittest.SkipTest(
            f"PySide6 unavailable; skipping Qt bridge tests: {_PYSIDE6_IMPORT_ERROR}"
        )

    try:
        from PySide6.QtCore import QCoreApplication
    except ImportError as exc:
        _PYSIDE6_IMPORT_ERROR = exc
        raise unittest.SkipTest(
            f"PySide6 unavailable; skipping Qt bridge tests: {exc}"
        ) from exc

    from live_inference.gui.qt_worker_bridge import (
        QtWorkerEventSink,
        WorkerQtSignals,
        WorkerThreadController,
    )

    _QT_BRIDGE_IMPORTS = (
        QCoreApplication,
        QtWorkerEventSink,
        WorkerQtSignals,
        WorkerThreadController,
    )
    return _QT_BRIDGE_IMPORTS


def _core_application() -> object:
    QCoreApplication, _, _, _ = _qt_bridge()
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    return app


def _process_events() -> None:
    _core_application().processEvents()


def _wait_until(predicate: Callable[[], bool], timeout_seconds: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _process_events()
        if predicate():
            return True
        time.sleep(0.005)
    _process_events()
    return bool(predicate())


def _imported_roots(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])

    return imported_roots


if __name__ == "__main__":
    unittest.main()
