"""Tests for the minimal PySide6 live inference main window."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LiveInferenceMainWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        QApplication, _, WorkerQtSignals, LiveInferenceMainWindow = _gui_imports()
        self.app = _application(QApplication)
        self.camera_controller = _FakeController(WorkerQtSignals())
        self.inference_controller = _FakeController(WorkerQtSignals())
        self.window = LiveInferenceMainWindow(
            camera_controller=self.camera_controller,
            inference_controller=self.inference_controller,
            stop_wait_ms=0,
        )

    def tearDown(self) -> None:
        self.window.hide()
        self.window.deleteLater()
        _process_events(self.app)

    def test_main_window_can_be_constructed_with_fake_controllers(self) -> None:
        self.assertEqual(self.window.windowTitle(), "Live Inference")

    def test_buttons_exist(self) -> None:
        _, QPushButton, _, _ = _gui_imports()
        for object_name in (
            "start_camera_button",
            "stop_camera_button",
            "start_inference_button",
            "stop_inference_button",
            "stop_all_button",
        ):
            self.assertIsNotNone(self.window.findChild(QPushButton, object_name))

    def test_main_window_has_image_preview_widget(self) -> None:
        preview = self._preview_label()

        self.assertGreaterEqual(preview.minimumWidth(), 320)
        self.assertGreaterEqual(preview.minimumHeight(), 240)

    def test_image_preview_shows_placeholder_before_first_frame(self) -> None:
        preview = self._preview_label()

        self.assertEqual(preview.text(), "No frame yet")
        self.assertFalse(_label_has_pixmap(preview))

    def test_frame_written_with_valid_image_path_loads_preview_pixmap(self) -> None:
        image_path = self._write_temp_frame_image()

        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        _process_events(self.app)

        preview = self._preview_label()
        self.assertTrue(_label_has_pixmap(preview))

    def test_frame_written_with_missing_image_path_logs_warning_without_crashing(self) -> None:
        missing_path = Path(tempfile.mkdtemp()) / "missing-frame.png"
        self.addCleanup(lambda: missing_path.parent.rmdir())

        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=missing_path))
        _process_events(self.app)

        self.assertEqual(self.window.frames_written_value.text(), "1")
        log_text = self.window.log_panel.toPlainText()
        self.assertIn("WARNING", log_text)
        self.assertIn("Frame preview unavailable", log_text)
        self.assertIn(str(missing_path), log_text)

    def test_frame_preview_does_not_affect_result_label_updates(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        self.inference_controller.signals.result_ready.emit(
            _ResultPayload(
                predicted_distance_m=2.5,
                predicted_yaw_deg=-15.25,
                inference_time_ms=8.0,
                preprocessing_time_ms=3.0,
                total_time_ms=11.0,
            )
        )
        _process_events(self.app)

        self.assertTrue(_label_has_pixmap(self._preview_label()))
        self.assertEqual(self.window.distance_value.text(), "2.500 m")
        self.assertEqual(self.window.yaw_value.text(), "-15.25 deg")
        self.assertEqual(self.window.inference_time_value.text(), "8.0 ms")
        self.assertEqual(self.window.preprocessing_time_value.text(), "3.0 ms")
        self.assertEqual(self.window.total_time_value.text(), "11.0 ms")

    def test_start_camera_button_calls_camera_controller_start(self) -> None:
        self.window.start_camera_button.click()

        self.assertEqual(self.camera_controller.start_calls, 1)

    def test_stop_camera_button_calls_camera_controller_request_stop(self) -> None:
        self.window.stop_camera_button.click()

        self.assertEqual(self.camera_controller.stop_calls, 1)

    def test_start_inference_button_calls_inference_controller_start(self) -> None:
        self.window.start_inference_button.click()

        self.assertEqual(self.inference_controller.start_calls, 1)

    def test_stop_inference_button_calls_inference_controller_request_stop(self) -> None:
        self.window.stop_inference_button.click()

        self.assertEqual(self.inference_controller.stop_calls, 1)

    def test_stop_all_calls_both_request_stop_methods(self) -> None:
        self.window.stop_all_button.click()

        self.assertEqual(self.camera_controller.stop_calls, 1)
        self.assertEqual(self.inference_controller.stop_calls, 1)

    def test_result_ready_updates_distance_and_yaw_labels(self) -> None:
        self.inference_controller.signals.result_ready.emit(
            _ResultPayload(
                predicted_distance_m=1.23456,
                predicted_yaw_deg=42.125,
                inference_time_ms=9.0,
                preprocessing_time_ms=4.5,
                total_time_ms=13.5,
            )
        )
        _process_events(self.app)

        self.assertEqual(self.window.distance_value.text(), "1.235 m")
        self.assertEqual(self.window.yaw_value.text(), "42.12 deg")
        self.assertEqual(self.window.inference_time_value.text(), "9.0 ms")
        self.assertEqual(self.window.preprocessing_time_value.text(), "4.5 ms")
        self.assertEqual(self.window.total_time_value.text(), "13.5 ms")

    def test_status_changed_updates_status_labels(self) -> None:
        self.camera_controller.signals.status_changed.emit(
            _StatusPayload(
                state="RUNNING",
                message="Camera active",
                counters={"frames_written": 7},
            )
        )
        self.inference_controller.signals.status_changed.emit(
            _StatusPayload(
                state="STOPPING",
                message="Inference stopping",
                counters={
                    "frames_processed": 3,
                    "frames_skipped_duplicate": 2,
                    "last_inference_time_ms": 8.25,
                    "last_preprocessing_time_ms": 2.0,
                    "last_total_time_ms": 10.25,
                },
            )
        )
        _process_events(self.app)

        self.assertIn("RUNNING", self.window.camera_status_value.text())
        self.assertIn("Camera active", self.window.camera_status_value.text())
        self.assertEqual(self.window.frames_written_value.text(), "7")
        self.assertIn("STOPPING", self.window.inference_status_value.text())
        self.assertIn("Inference stopping", self.window.inference_status_value.text())
        self.assertEqual(self.window.frames_processed_value.text(), "3")
        self.assertEqual(self.window.duplicate_skipped_value.text(), "2")

    def test_warning_and_error_events_append_to_log_panel(self) -> None:
        self.camera_controller.signals.warning_occurred.emit(
            _IssuePayload(warning_type="low_light", message="Camera warning")
        )
        self.inference_controller.signals.error_occurred.emit(
            _IssuePayload(error_type="model_error", message="Inference error")
        )
        _process_events(self.app)

        log_text = self.window.log_panel.toPlainText()
        self.assertIn("WARNING", log_text)
        self.assertIn("low_light", log_text)
        self.assertIn("Camera warning", log_text)
        self.assertIn("ERROR", log_text)
        self.assertIn("model_error", log_text)
        self.assertIn("Inference error", log_text)

    def test_close_event_requests_worker_stop(self) -> None:
        self.window.close()
        _process_events(self.app)

        self.assertEqual(self.camera_controller.stop_calls, 1)
        self.assertEqual(self.inference_controller.stop_calls, 1)

    def _preview_label(self) -> object:
        from PySide6.QtWidgets import QLabel

        preview = self.window.findChild(QLabel, "frame_preview_label")
        self.assertIsNotNone(preview)
        return preview

    def _write_temp_frame_image(self) -> Path:
        from PySide6.QtGui import QColor, QImage

        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        image_path = Path(temp_dir.name) / "frame.png"
        image = QImage(80, 48, QImage.Format.Format_RGB32)
        image.fill(QColor(20, 120, 200))
        self.assertTrue(image.save(str(image_path)))
        return image_path


class PySide6IsolationTests(unittest.TestCase):
    def test_pyside6_imports_remain_isolated_to_gui_modules(self) -> None:
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
class _StatusPayload:
    state: str
    message: str
    counters: dict[str, Any]


@dataclass(frozen=True)
class _ResultPayload:
    predicted_distance_m: float
    predicted_yaw_deg: float
    inference_time_ms: float
    preprocessing_time_ms: float | None
    total_time_ms: float | None


@dataclass(frozen=True)
class _IssuePayload:
    message: str
    warning_type: str | None = None
    error_type: str | None = None


@dataclass(frozen=True)
class _FramePayload:
    image_path: Path
    completed_at_utc: str | None = None
    frame_hash: object | None = None
    metadata: object | None = None


class _FakeController:
    def __init__(self, signals: object) -> None:
        self.signals = signals
        self.start_calls = 0
        self.stop_calls = 0
        self.wait_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def request_stop(self) -> None:
        self.stop_calls += 1

    def wait(self, timeout_ms: int | None = None) -> bool:
        self.wait_calls += 1
        return True


_GUI_IMPORTS: tuple[Any, Any, Any, Any] | None = None
_PYSIDE6_IMPORT_ERROR: ImportError | None = None


def _gui_imports() -> tuple[Any, Any, Any, Any]:
    global _GUI_IMPORTS, _PYSIDE6_IMPORT_ERROR

    if _GUI_IMPORTS is not None:
        return _GUI_IMPORTS
    if _PYSIDE6_IMPORT_ERROR is not None:
        raise unittest.SkipTest(
            f"PySide6 unavailable; skipping GUI tests: {_PYSIDE6_IMPORT_ERROR}"
        )

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication, QPushButton
    except ImportError as exc:
        _PYSIDE6_IMPORT_ERROR = exc
        raise unittest.SkipTest(
            f"PySide6 unavailable; skipping GUI tests: {exc}"
        ) from exc

    from live_inference.gui.main_window import LiveInferenceMainWindow
    from live_inference.gui.qt_worker_bridge import WorkerQtSignals

    _GUI_IMPORTS = (
        QApplication,
        QPushButton,
        WorkerQtSignals,
        LiveInferenceMainWindow,
    )
    return _GUI_IMPORTS


def _application(QApplication: Any) -> Any:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    if not isinstance(app, QApplication):
        raise unittest.SkipTest("A QCoreApplication already exists; QWidget tests need QApplication.")
    app.setQuitOnLastWindowClosed(False)
    return app


def _process_events(app: object) -> None:
    app.processEvents()


def _label_has_pixmap(label: object) -> bool:
    pixmap = label.pixmap()
    return pixmap is not None and not pixmap.isNull()


def _imported_roots(module_path: Path) -> set[str]:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", maxsplit=1)[0])

    return imported
