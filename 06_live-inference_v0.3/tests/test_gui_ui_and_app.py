from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2  # noqa: E402
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QGridLayout,
    QGroupBox,
    QSplitter,
    QVBoxLayout,
)

import interfaces.contracts as contracts  # noqa: E402
from live_inference.gui.app import build_live_inference_gui_context  # noqa: E402
from live_inference.gui.frame_preview_widget import FramePreviewWidget  # noqa: E402
from live_inference.gui.main_window import LiveInferenceMainWindow  # noqa: E402
from live_inference.preprocessing import (  # noqa: E402
    CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP,
    CameraIntrinsicsTransformState,
    ForegroundExtractionPolicyState,
    StageTransformPolicyState,
)


class _Controller:
    signals = None

    def start(self) -> None:
        pass

    def request_stop(self) -> None:
        pass

    def wait(self, _timeout_ms: int) -> bool:
        return True


class _StartCountingController(_Controller):
    def __init__(self) -> None:
        self.start_count = 0

    def start(self) -> None:
        self.start_count += 1


class _FrameReader:
    def __init__(self) -> None:
        self._latest_frame: object | None = None
        self._bytes_by_path: dict[Path, bytes] = {}

    def set_latest(self, frame: object, image_bytes: bytes) -> None:
        image_path = getattr(frame, "image_path")
        self._latest_frame = frame
        self._bytes_by_path[Path(image_path)] = bytes(image_bytes)

    def latest_completed_frame(self) -> object | None:
        return self._latest_frame

    def read_frame_bytes(self, frame: object) -> bytes:
        image_path = getattr(frame, "image_path")
        return self._bytes_by_path[Path(image_path)]


class _SingleFrameRunner:
    def __init__(self) -> None:
        self.run_count = 0

    def run_single_frame(
        self,
        _image_bytes: bytes,
        *,
        source_path: Path | None = None,
        frame_metadata: object | None = None,
        record_trace: bool = False,
    ) -> object:
        del frame_metadata, record_trace
        self.run_count += 1
        return SimpleNamespace(
            result=_inference_result(source_path or Path("captured.png")),
            error=None,
            trace_path=None,
        )


def _solid_png_bytes(value: int) -> bytes:
    image = np.full((6, 8, 3), int(value), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("Could not encode test image.")
    return encoded.tobytes()


def _frame_reference(path: Path) -> contracts.FrameReference:
    return contracts.FrameReference(
        image_path=path,
        metadata=contracts.FrameMetadata(width_px=8, height_px=6),
    )


def _inference_result(path: Path) -> contracts.InferenceResult:
    return contracts.InferenceResult(
        request_id="request",
        input_image_path=path,
        input_image_hash=contracts.FrameHash("hash"),
        timestamp_utc="2026-05-24T00:00:00Z",
        predicted_distance_m=1.0,
        predicted_yaw_sin=0.0,
        predicted_yaw_cos=1.0,
        predicted_yaw_deg=0.0,
        inference_time_ms=1.0,
    )


def _inference_result_with_roi(
    path: Path,
    *,
    center_xy_px: tuple[float, float],
    roi_bounds_xyxy_px: tuple[float, float, float, float],
) -> contracts.InferenceResult:
    return contracts.InferenceResult(
        request_id="request",
        input_image_path=path,
        input_image_hash=contracts.FrameHash("hash"),
        timestamp_utc="2026-05-24T00:00:00Z",
        predicted_distance_m=1.0,
        predicted_yaw_sin=0.0,
        predicted_yaw_cos=1.0,
        predicted_yaw_deg=0.0,
        inference_time_ms=1.0,
        roi_metadata=contracts.RoiMetadata(
            source_image_wh_px=(640, 480),
            center_xy_px=center_xy_px,
            roi_requested_xyxy_px=roi_bounds_xyxy_px,
            extras={
                contracts.PREPROCESSING_METADATA_ROI_REQUESTED_XYXY_PX: (
                    roi_bounds_xyxy_px
                ),
            },
        ),
    )


def _qimage_rgb_array(image: QImage) -> np.ndarray:
    rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
    width = int(rgb_image.width())
    height = int(rgb_image.height())
    bytes_per_line = int(rgb_image.bytesPerLine())
    buffer = rgb_image.bits()
    array = np.frombuffer(buffer, dtype=np.uint8).reshape(height, bytes_per_line)
    return np.array(array[:, : width * 3].reshape(height, width, 3), copy=True)


class GuiUiAndAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_ui_file_loads_with_custom_preview_widget(self) -> None:
        window = LiveInferenceMainWindow(
            camera_controller=_Controller(),
            inference_controller=_Controller(),
        )

        self.assertEqual(type(window.main_preview_widget).__name__, "FramePreviewWidget")
        self.assertEqual(window.start_camera_button.text(), "Start Camera")
        self.assertEqual(window.draw_mask_button.text(), "Draw")
        self.assertEqual(window.erase_mask_button.text(), "Erase")
        self.assertEqual(window.apply_mask_button.text(), "Apply")
        self.assertEqual(window.mask_brush_size_spinbox.value(), 100)
        self.assertEqual(
            window.use_silhouette_preprocessing_checkbox.text(),
            "Use silhouette preprocessing",
        )
        self.assertFalse(window.use_silhouette_preprocessing_checkbox.isChecked())
        self.assertEqual(window.camera_intrinsics_mode_combo.currentData(), "disabled")
        self.assertIn("mask:", window.mask_status_value.text())
        self.assertEqual(window.capture_background_button.text(), "Capture Background")
        self.assertEqual(window.clear_background_button.text(), "Clear Background")
        self.assertEqual(
            window.enable_background_removal_checkbox.text(),
            "Enable Background Removal",
        )
        self.assertEqual(
            window.apply_background_removal_to_locator_checkbox.text(),
            "Apply to locator",
        )
        self.assertEqual(
            window.apply_background_removal_to_model_preprocessing_checkbox.text(),
            "Apply to model preprocessing",
        )
        self.assertIn("background:", window.background_removal_status_value.text())
        self.assertTrue(window.background_removal_status_value.wordWrap())
        self.assertEqual(window.camera_fps_value.text(), "raw frame FPS: n/a")
        self.assertEqual(window.inference_fps_value.text(), "inference FPS: n/a")
        status_layout = window.findChild(QVBoxLayout, "statusLayout")
        self.assertIsNotNone(status_layout)
        self.assertEqual(status_layout.itemAt(0).widget().objectName(), "cameraFpsValue")
        self.assertEqual(
            status_layout.itemAt(1).widget().objectName(),
            "inferenceFpsValue",
        )
        self.assertEqual(
            window.capture_background_button.parent().objectName(),
            "backgroundRemovalGroup",
        )
        inference_layout = window.findChild(QGridLayout, "inferenceButtonLayout")
        self.assertIsNotNone(inference_layout)
        self.assertEqual(
            inference_layout.itemAtPosition(0, 0).widget().objectName(),
            "runSingleInferenceButton",
        )
        self.assertEqual(
            inference_layout.itemAtPosition(1, 0).widget().objectName(),
            "startContinuousButton",
        )
        self.assertEqual(
            inference_layout.itemAtPosition(2, 0).widget().objectName(),
            "stopContinuousButton",
        )
        self.assertIsNone(inference_layout.itemAtPosition(0, 1))
        draw_mask_layout = window.findChild(QGridLayout, "drawMaskButtonLayout")
        self.assertIsNotNone(draw_mask_layout)
        self.assertEqual(
            draw_mask_layout.itemAtPosition(1, 1).widget().objectName(),
            "cancelMaskButton",
        )
        background_layout = window.findChild(QGridLayout, "backgroundRemovalLayout")
        self.assertIsNotNone(background_layout)
        self.assertEqual(
            background_layout.itemAtPosition(0, 0).widget().objectName(),
            "captureBackgroundButton",
        )
        self.assertEqual(
            background_layout.itemAtPosition(1, 0).widget().objectName(),
            "clearBackgroundButton",
        )
        self.assertIsInstance(window.top_workspace_splitter, QSplitter)
        self.assertEqual(window.top_workspace_splitter.count(), 4)
        self.assertIsNotNone(window.findChild(QGroupBox, "additionalControlsGroup"))
        self.assertEqual(window.show_roi_checkbox.parent().objectName(), "overlayDebugGroup")

    def test_background_controls_capture_enable_and_update_stage_policy(self) -> None:
        stage_policy_state = StageTransformPolicyState()
        window = LiveInferenceMainWindow(
            camera_controller=_Controller(),
            inference_controller=_Controller(),
            stage_policy_state=stage_policy_state,
        )
        image = np.full((24, 32), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        window._captured_single_frame = SimpleNamespace(image_bytes=encoded.tobytes())

        window.capture_background()

        snapshot = window.background_state.get_snapshot()
        self.assertTrue(snapshot.captured)
        self.assertFalse(snapshot.enabled)
        self.assertFalse(window.enable_background_removal_checkbox.isChecked())
        self.assertIn("captured 32x24", window.background_removal_status_value.text())
        self.assertIn("\nrevision", window.background_removal_status_value.text())
        self.assertIsNone(window.main_preview_widget.background_snapshot())

        window.enable_background_removal_checkbox.setChecked(True)
        self.assertTrue(window.background_state.get_snapshot().enabled)
        self.assertIsNone(window.main_preview_widget.background_snapshot())
        window.apply_background_removal_to_locator_checkbox.setChecked(True)
        window.apply_background_removal_to_model_preprocessing_checkbox.setChecked(True)

        policy = stage_policy_state.get_snapshot()
        self.assertTrue(policy.apply_background_removal_to_roi_locator)
        self.assertTrue(policy.apply_background_removal_to_regressor_preprocessing)

    def test_fps_metrics_update_from_camera_and_live_inference_events(self) -> None:
        window = LiveInferenceMainWindow(
            camera_controller=_Controller(),
            inference_controller=_Controller(),
        )

        window._on_status_changed(
            contracts.WorkerStatus(
                worker_name=contracts.WorkerName.CAMERA,
                state=contracts.WorkerState.RUNNING,
                message="Camera worker is running.",
                timestamp_utc="2026-05-24T00:00:00Z",
            )
        )
        window._camera_frame_timestamps.extend([10.0, 10.5, 11.0])
        window._refresh_performance_metrics(now=11.0)

        self.assertEqual(window.camera_fps_value.text(), "raw frame FPS: 2.0")
        self.assertEqual(window.inference_fps_value.text(), "inference FPS: n/a")

        window._on_status_changed(
            contracts.WorkerStatus(
                worker_name=contracts.WorkerName.INFERENCE,
                state=contracts.WorkerState.RUNNING,
                message="Inference worker is running.",
                timestamp_utc="2026-05-24T00:00:01Z",
            )
        )
        window._inference_result_timestamps.extend([20.0, 20.25, 20.5])
        window._refresh_performance_metrics(now=20.5)

        self.assertEqual(window.inference_fps_value.text(), "inference FPS: 4.0")

        window._on_status_changed(
            contracts.WorkerStatus(
                worker_name=contracts.WorkerName.INFERENCE,
                state=contracts.WorkerState.STOPPED,
                message="Inference worker stopped.",
                timestamp_utc="2026-05-24T00:00:02Z",
            )
        )

        self.assertEqual(window.inference_fps_value.text(), "inference FPS: n/a")

    def test_inference_regression_labels_use_display_precision(self) -> None:
        window = LiveInferenceMainWindow(
            camera_controller=_Controller(),
            inference_controller=_Controller(),
        )
        result = contracts.InferenceResult(
            request_id="request",
            input_image_path=Path("frame.png"),
            input_image_hash=contracts.FrameHash("hash"),
            timestamp_utc="2026-05-24T00:00:00Z",
            predicted_distance_m=1.234,
            predicted_yaw_sin=0.0,
            predicted_yaw_cos=1.0,
            predicted_yaw_deg=12.6,
            inference_time_ms=1.0,
        )

        window._on_inference_result_ready(result)

        self.assertEqual(window.distance_value.text(), "distance: 1.23 m")
        self.assertEqual(window.yaw_value.text(), "yaw: 13 deg")

    def test_start_continuous_inference_resumes_live_preview_after_single_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            captured_path = tmp_path / "captured.png"
            live_path = tmp_path / "live.png"
            newer_path = tmp_path / "newer.png"
            captured_bytes = _solid_png_bytes(32)
            live_bytes = _solid_png_bytes(220)
            newer_bytes = _solid_png_bytes(96)
            captured_path.write_bytes(captured_bytes)
            live_path.write_bytes(live_bytes)
            newer_path.write_bytes(newer_bytes)

            frame_reader = _FrameReader()
            captured_frame = _frame_reference(captured_path)
            live_frame = _frame_reference(live_path)
            newer_frame = _frame_reference(newer_path)
            frame_reader.set_latest(captured_frame, captured_bytes)
            inference_controller = _StartCountingController()
            single_frame_runner = _SingleFrameRunner()
            window = LiveInferenceMainWindow(
                camera_controller=_Controller(),
                inference_controller=inference_controller,
                frame_reader=frame_reader,
                single_frame_runner=single_frame_runner,
            )

            window.capture_frame()
            window.run_single_inference()
            self.assertEqual(single_frame_runner.run_count, 1)
            self.assertIsNotNone(window._captured_single_frame)
            captured_preview = window.main_preview_widget.effective_preview_image()
            self.assertIsNotNone(captured_preview)
            assert captured_preview is not None
            self.assertEqual(int(captured_preview[0, 0, 0]), 32)

            frame_reader.set_latest(live_frame, live_bytes)
            window.start_continuous_inference()

            self.assertEqual(inference_controller.start_count, 1)
            self.assertIsNone(window._captured_single_frame)
            live_preview = window.main_preview_widget.effective_preview_image()
            self.assertIsNotNone(live_preview)
            assert live_preview is not None
            self.assertEqual(int(live_preview[0, 0, 0]), 220)

            frame_reader.set_latest(newer_frame, newer_bytes)
            window._on_frame_written(newer_frame)
            window._on_inference_result_ready(_inference_result(newer_path))

            newer_preview = window.main_preview_widget.effective_preview_image()
            self.assertIsNotNone(newer_preview)
            assert newer_preview is not None
            self.assertEqual(int(newer_preview[0, 0, 0]), 96)

    def test_live_inference_overlay_is_display_smoothed(self) -> None:
        window = LiveInferenceMainWindow(
            camera_controller=_Controller(),
            inference_controller=_Controller(),
        )

        window._on_inference_result_ready(
            _inference_result_with_roi(
                Path("frame-1.png"),
                center_xy_px=(50.0, 50.0),
                roi_bounds_xyxy_px=(0.0, 0.0, 100.0, 100.0),
            )
        )
        window._on_inference_result_ready(
            _inference_result_with_roi(
                Path("frame-2.png"),
                center_xy_px=(70.0, 70.0),
                roi_bounds_xyxy_px=(40.0, 40.0, 120.0, 120.0),
            )
        )

        overlay = window._last_overlay
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.center_xy_px, (60.0, 60.0))
        self.assertEqual(overlay.roi_bounds_xyxy_px, (25.0, 25.0, 105.0, 105.0))

    def test_single_frame_overlay_stays_raw(self) -> None:
        window = LiveInferenceMainWindow(
            camera_controller=_Controller(),
            inference_controller=_Controller(),
        )
        window._captured_single_frame = SimpleNamespace(image_bytes=b"captured")

        window._on_inference_result_ready(
            _inference_result_with_roi(
                Path("captured.png"),
                center_xy_px=(70.0, 70.0),
                roi_bounds_xyxy_px=(40.0, 40.0, 120.0, 120.0),
            )
        )

        overlay = window._last_overlay
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.center_xy_px, (70.0, 70.0))
        self.assertEqual(overlay.roi_bounds_xyxy_px, (40.0, 40.0, 120.0, 120.0))

    def test_silhouette_checkbox_updates_preprocessing_policy_state(self) -> None:
        policy_state = ForegroundExtractionPolicyState()
        window = LiveInferenceMainWindow(
            camera_controller=_Controller(),
            inference_controller=_Controller(),
            foreground_extraction_policy_state=policy_state,
        )

        window.use_silhouette_preprocessing_checkbox.setChecked(True)

        snapshot = policy_state.snapshot()
        self.assertEqual(
            snapshot.foreground_extraction_mode,
            contracts.ForegroundExtractionMode.SILHOUETTE_CONTOUR_V2.value,
        )
        self.assertEqual(snapshot.revision, 1)

    def test_camera_intrinsics_dropdown_updates_shared_state(self) -> None:
        intrinsics_state = CameraIntrinsicsTransformState()
        window = LiveInferenceMainWindow(
            camera_controller=_Controller(),
            inference_controller=_Controller(),
            camera_intrinsics_state=intrinsics_state,
        )

        index = window.camera_intrinsics_mode_combo.findData(
            CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP
        )
        self.assertGreaterEqual(index, 0)
        window.camera_intrinsics_mode_combo.setCurrentIndex(index)

        snapshot = intrinsics_state.snapshot()
        self.assertEqual(
            snapshot.camera_intrinsics_mode,
            CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP,
        )
        self.assertEqual(snapshot.revision, 1)

    def test_draw_mask_buttons_commit_to_shared_mask_state(self) -> None:
        window = LiveInferenceMainWindow(
            camera_controller=_Controller(),
            inference_controller=_Controller(),
        )
        image = QImage(40, 30, QImage.Format.Format_Grayscale8)
        image.fill(255)
        window.main_preview_widget.set_image(image)

        window.start_draw_mask()
        window.main_preview_widget._apply_brush_at_source((20, 15))
        window.apply_mask()

        snapshot = window.mask_state.get_snapshot()
        self.assertTrue(snapshot.enabled)
        self.assertEqual((snapshot.width_px, snapshot.height_px), (40, 30))
        self.assertGreater(snapshot.pixel_count, 0)
        preview = window.main_preview_widget.effective_preview_image()
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertTrue(np.all(preview == 255))

    def test_drawn_mask_is_visible_as_opaque_white_gui_overlay(self) -> None:
        widget = FramePreviewWidget()
        widget.resize(40, 30)
        image = QImage(40, 30, QImage.Format.Format_RGB888)
        image.fill(QColor(100, 100, 100))
        widget.set_image(image)

        widget.set_brush_diameter_px(5)
        widget.begin_mask_edit("draw")
        widget._apply_brush_at_source((20, 15))
        widget.show()
        self.app.processEvents()

        rendered = _qimage_rgb_array(widget.grab().toImage())
        base_preview = widget.effective_preview_image()
        self.assertIsNotNone(base_preview)
        assert base_preview is not None

        self.assertTrue(np.array_equal(base_preview[15, 20], [100, 100, 100]))
        self.assertTrue(np.array_equal(rendered[0, 0], [100, 100, 100]))
        self.assertTrue(np.array_equal(rendered[15, 20], [255, 255, 255]))

        widget.finish_mask_edit(commit=True)
        self.app.processEvents()
        committed_rendered = _qimage_rgb_array(widget.grab().toImage())

        self.assertTrue(np.array_equal(committed_rendered[0, 0], [100, 100, 100]))
        self.assertTrue(np.array_equal(committed_rendered[15, 20], [255, 255, 255]))

    def test_app_context_uses_background_edge_locator_by_default(self) -> None:
        context = build_live_inference_gui_context(device="cpu")
        try:
            self.assertEqual(
                context.locator_kind,
                contracts.LocatorKind.BACKGROUND_EDGE_V1,
            )
            self.assertEqual(
                type(context.locator_parameter_state).__name__,
                "LocatorRuntimeParameterState",
            )
            self.assertEqual(
                type(context.frame_mask_state).__name__,
                "FrameMaskState",
            )
            self.assertEqual(
                context.foreground_extraction_policy_state.snapshot().foreground_extraction_mode,
                contracts.ForegroundExtractionMode.THRESHOLD_FOREGROUND_V1.value,
            )
            self.assertEqual(
                context.camera_intrinsics_state.snapshot().camera_intrinsics_mode,
                "disabled",
            )
            self.assertFalse(
                context.stage_policy_state.get_snapshot().apply_background_removal_to_roi_locator
            )
            self.assertFalse(
                context.stage_policy_state.get_snapshot().apply_background_removal_to_regressor_preprocessing
            )
        finally:
            context.camera_controller.request_stop()
            context.inference_controller.request_stop()
            context.camera_controller.wait(10)
            context.inference_controller.wait(10)


if __name__ == "__main__":
    unittest.main()
