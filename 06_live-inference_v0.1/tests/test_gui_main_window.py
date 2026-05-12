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

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import interfaces.contracts as contracts  # noqa: E402
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402


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
        self.window.resize(900, 600)
        self.window.show()
        _process_events(self.app)

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

    def test_single_frame_controls_exist(self) -> None:
        _, QPushButton, _, _ = _gui_imports()
        from PySide6.QtWidgets import QCheckBox, QLabel

        self.assertIsNotNone(self.window.findChild(QPushButton, "capture_frame_button"))
        self.assertIsNotNone(
            self.window.findChild(QPushButton, "run_single_inference_button")
        )
        self.assertIsNotNone(
            self.window.findChild(QCheckBox, "record_trace_checkbox")
        )
        self.assertIsNotNone(
            self.window.findChild(QLabel, "last_captured_frame_hash_value")
        )
        self.assertIsNotNone(self.window.findChild(QLabel, "last_trace_path_value"))

    def test_right_side_control_panel_contains_camera_and_inference_buttons(self) -> None:
        _, QPushButton, _, _ = _gui_imports()
        from PySide6.QtWidgets import QWidget

        panel = self.window.findChild(QWidget, "right_control_panel")
        self.assertIsNotNone(panel)
        assert panel is not None
        for object_name in (
            "start_camera_button",
            "stop_camera_button",
            "start_inference_button",
            "stop_inference_button",
            "stop_all_button",
        ):
            button = self.window.findChild(QPushButton, object_name)
            self.assertIsNotNone(button)
            self.assertTrue(_is_descendant(button, panel))

    def test_mask_controls_exist(self) -> None:
        _, QPushButton, _, _ = _gui_imports()
        from PySide6.QtWidgets import QCheckBox, QLabel, QSpinBox

        for object_name in (
            "draw_mask_button",
            "apply_mask_button",
            "erase_mask_button",
            "apply_erase_button",
            "clear_mask_button",
        ):
            self.assertIsNotNone(self.window.findChild(QPushButton, object_name))
        self.assertIsNotNone(
            self.window.findChild(QSpinBox, "mask_brush_diameter_input")
        )
        self.assertIsNotNone(
            self.window.findChild(QCheckBox, "mask_fill_white_checkbox")
        )
        self.assertIsNotNone(
            self.window.findChild(
                QCheckBox,
                "apply_manual_mask_to_roi_locator_checkbox",
            )
        )
        self.assertIsNotNone(
            self.window.findChild(
                QCheckBox,
                "apply_manual_mask_to_model_preprocessing_checkbox",
            )
        )
        self.assertIsNotNone(self.window.findChild(QLabel, "mask_fill_value_label"))

    def test_background_controls_exist(self) -> None:
        _, QPushButton, _, _ = _gui_imports()
        from PySide6.QtWidgets import QCheckBox, QLabel, QSpinBox

        self.assertIsNotNone(
            self.window.findChild(QPushButton, "capture_background_button")
        )
        self.assertIsNotNone(
            self.window.findChild(QCheckBox, "enable_background_removal_checkbox")
        )
        self.assertIsNotNone(
            self.window.findChild(
                QCheckBox,
                "apply_background_removal_to_roi_locator_checkbox",
            )
        )
        self.assertIsNotNone(
            self.window.findChild(
                QCheckBox,
                "apply_background_removal_to_model_preprocessing_checkbox",
            )
        )
        self.assertIsNotNone(
            self.window.findChild(QPushButton, "clear_background_button")
        )
        self.assertIsNotNone(
            self.window.findChild(QSpinBox, "background_threshold_input")
        )
        self.assertIsNotNone(self.window.findChild(QLabel, "background_status_label"))
        self.assertIsNotNone(self._background_preview_widget())

    def test_roi_locator_controls_exist(self) -> None:
        from PySide6.QtWidgets import QCheckBox, QLabel, QSpinBox

        self.assertIsNotNone(
            self.window.findChild(QCheckBox, "invert_roi_locator_input_checkbox")
        )
        self.assertIsNotNone(
            self.window.findChild(QSpinBox, "roi_clip_tolerance_input")
        )
        self.assertIsNotNone(
            self.window.findChild(QLabel, "roi_locator_status_label")
        )
        self.assertIsNotNone(self.window.findChild(QLabel, "roi_clip_amount_value"))

    def test_stage_policy_checkboxes_update_shared_state(self) -> None:
        from PySide6.QtWidgets import QCheckBox

        manual_locator = self.window.findChild(
            QCheckBox,
            "apply_manual_mask_to_roi_locator_checkbox",
        )
        manual_model = self.window.findChild(
            QCheckBox,
            "apply_manual_mask_to_model_preprocessing_checkbox",
        )
        background_locator = self.window.findChild(
            QCheckBox,
            "apply_background_removal_to_roi_locator_checkbox",
        )
        background_model = self.window.findChild(
            QCheckBox,
            "apply_background_removal_to_model_preprocessing_checkbox",
        )
        assert manual_locator is not None
        assert manual_model is not None
        assert background_locator is not None
        assert background_model is not None

        manual_locator.setChecked(True)
        manual_model.setChecked(False)
        background_locator.setChecked(True)
        background_model.setChecked(True)
        _process_events(self.app)

        snapshot = self.window.stage_policy_state.get_snapshot()
        self.assertTrue(snapshot.apply_manual_mask_to_roi_locator)
        self.assertFalse(snapshot.apply_manual_mask_to_regressor_preprocessing)
        self.assertTrue(snapshot.apply_background_removal_to_roi_locator)
        self.assertTrue(snapshot.apply_background_removal_to_regressor_preprocessing)

    def test_roi_locator_controls_update_shared_state(self) -> None:
        from PySide6.QtWidgets import QCheckBox, QSpinBox

        invert = self.window.findChild(
            QCheckBox,
            "invert_roi_locator_input_checkbox",
        )
        tolerance = self.window.findChild(QSpinBox, "roi_clip_tolerance_input")
        assert invert is not None
        assert tolerance is not None

        invert.setChecked(True)
        tolerance.setValue(10)
        _process_events(self.app)

        snapshot = self.window.stage_policy_state.get_snapshot()
        self.assertEqual(snapshot.roi_locator_input_polarity, "inverted")
        self.assertEqual(snapshot.roi_clip_tolerance_px, 10)

    def test_debug_heatmap_overlay_checkbox_defaults_to_enabled(self) -> None:
        from PySide6.QtWidgets import QCheckBox

        checkbox = self.window.findChild(
            QCheckBox,
            "show_roi_fcn_heatmap_overlay_checkbox",
        )

        self.assertIsNotNone(checkbox)
        assert checkbox is not None
        self.assertTrue(checkbox.isChecked())
        self.assertTrue(self._preview_widget().heatmap_overlay_enabled())

    def test_debug_heatmap_overlay_checkbox_toggles_frame_preview(self) -> None:
        from PySide6.QtWidgets import QCheckBox

        checkbox = self.window.findChild(
            QCheckBox,
            "show_roi_fcn_heatmap_overlay_checkbox",
        )
        assert checkbox is not None

        checkbox.setChecked(False)
        _process_events(self.app)

        self.assertFalse(self._preview_widget().heatmap_overlay_enabled())

    def test_fill_checkbox_updates_label_and_mask_state_fill_value(self) -> None:
        from PySide6.QtWidgets import QCheckBox, QLabel

        checkbox = self.window.findChild(QCheckBox, "mask_fill_white_checkbox")
        label = self.window.findChild(QLabel, "mask_fill_value_label")
        assert checkbox is not None
        assert label is not None

        checkbox.setChecked(False)
        _process_events(self.app)

        self.assertEqual(label.text(), "Mask fill: Black (0)")
        self.assertEqual(self.window.mask_state.get_snapshot().fill_value, 0)

    def test_main_window_has_image_preview_widget(self) -> None:
        preview = self._preview_widget()

        self.assertGreaterEqual(preview.minimumWidth(), 320)
        self.assertGreaterEqual(preview.minimumHeight(), 240)

    def test_image_preview_shows_placeholder_before_first_frame(self) -> None:
        preview = self._preview_widget()

        self.assertEqual(preview.placeholder_text(), "No frame yet")
        self.assertFalse(_widget_has_pixmap(preview))

    def test_frame_written_with_valid_image_path_loads_preview_pixmap(self) -> None:
        image_path = self._write_temp_frame_image()

        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        _process_events(self.app)

        preview = self._preview_widget()
        self.assertTrue(_widget_has_pixmap(preview))

    def test_capture_background_requires_inference_stopped(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        _process_events(self.app)

        self.window.start_inference_button.click()
        self.window.capture_background_button.click()
        _process_events(self.app)

        self.assertFalse(self.window.background_state.get_snapshot().captured)
        self.assertIn(
            "Stop inference before capturing background",
            self.window.log_panel.toPlainText(),
        )

    def test_capture_background_updates_status_and_preview_state(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        _process_events(self.app)

        self.window.capture_background_button.click()
        _process_events(self.app)

        snapshot = self.window.background_state.get_snapshot()
        self.assertTrue(snapshot.captured)
        self.assertEqual(snapshot.width_px, 80)
        self.assertEqual(snapshot.height_px, 48)
        self.assertEqual(self.window.background_status_label.text(), "Background: captured")
        self.assertIsNone(self._preview_widget().background_snapshot())
        self.assertTrue(_widget_has_pixmap(self._background_preview_widget()))

    def test_enable_background_removal_toggles_state(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        self.window.capture_background_button.click()
        _process_events(self.app)

        self.window.enable_background_removal_checkbox.setChecked(True)
        _process_events(self.app)

        self.assertTrue(self.window.background_state.get_snapshot().enabled)
        self.assertEqual(self.window.background_status_label.text(), "Background: enabled")
        self.assertIsNone(self._preview_widget().background_snapshot())
        background_preview = self._background_preview_widget()
        self.assertIsNotNone(background_preview.background_snapshot())
        effective = background_preview.effective_preview_image()
        self.assertIsNotNone(effective)
        assert effective is not None
        self.assertEqual(tuple(int(value) for value in effective[24, 40]), (255, 255, 255))

    def test_clear_background_resets_state_and_status(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        self.window.capture_background_button.click()
        self.window.enable_background_removal_checkbox.setChecked(True)
        _process_events(self.app)

        self.window.clear_background_button.click()
        _process_events(self.app)

        snapshot = self.window.background_state.get_snapshot()
        self.assertFalse(snapshot.captured)
        self.assertFalse(snapshot.enabled)
        self.assertFalse(self.window.enable_background_removal_checkbox.isChecked())
        self.assertEqual(self.window.background_status_label.text(), "Background: not captured")

    def test_background_threshold_control_updates_state(self) -> None:
        from PySide6.QtWidgets import QSpinBox

        threshold = self.window.findChild(QSpinBox, "background_threshold_input")
        assert threshold is not None

        threshold.setValue(42)
        _process_events(self.app)

        self.assertEqual(self.window.background_state.get_snapshot().threshold, 42)

    def test_background_threshold_updates_background_preview_image(self) -> None:
        from PySide6.QtWidgets import QSpinBox

        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        _process_events(self.app)
        gray = self._preview_widget().raw_source_gray()
        assert gray is not None
        background = np.clip(gray.astype(np.int16) + 10, 0, 255).astype(np.uint8)

        threshold = self.window.findChild(QSpinBox, "background_threshold_input")
        assert threshold is not None
        threshold.setValue(5)
        self.window.background_state.capture_background(background)
        self.window.enable_background_removal_checkbox.setChecked(True)
        _process_events(self.app)
        low_threshold = self._background_preview_widget().effective_preview_image()

        threshold.setValue(15)
        _process_events(self.app)
        high_threshold = self._background_preview_widget().effective_preview_image()

        self.assertIsNotNone(low_threshold)
        self.assertIsNotNone(high_threshold)
        assert low_threshold is not None
        assert high_threshold is not None
        self.assertEqual(
            tuple(int(value) for value in low_threshold[24, 40]),
            (20, 120, 200),
        )
        self.assertEqual(
            tuple(int(value) for value in high_threshold[24, 40]),
            (255, 255, 255),
        )

    def test_frame_written_log_spam_is_suppressed(self) -> None:
        image_path = self._write_temp_frame_image()

        for _ in range(3):
            self.camera_controller.signals.frame_written.emit(
                _FramePayload(image_path=image_path)
            )
        _process_events(self.app)

        self.assertEqual(self.window.frames_written_value.text(), "3")
        self.assertNotIn("Frame written:", self.window.log_panel.toPlainText())

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

    def test_capture_frame_updates_last_captured_hash_label_from_reader(self) -> None:
        from PySide6.QtWidgets import QLabel

        image_path = self._write_temp_frame_image()
        frame_bytes = image_path.read_bytes()
        self.window.frame_reader = _FakeFrameReader(
            image_path=image_path,
            image_bytes=frame_bytes,
        )

        self.window.capture_frame_button.click()
        _process_events(self.app)

        label = self.window.findChild(QLabel, "last_captured_frame_hash_value")
        assert label is not None
        expected_hash = compute_frame_hash(frame_bytes).value
        self.assertEqual(label.text(), expected_hash)
        self.assertIn(
            f"Captured frame {expected_hash}",
            self.window.log_panel.toPlainText(),
        )
        self.assertTrue(_widget_has_pixmap(self._preview_widget()))

    def test_run_single_inference_updates_result_labels_using_fake_runner(self) -> None:
        image_path = self._write_temp_frame_image()
        frame_bytes = image_path.read_bytes()
        runner = _FakeSingleFrameRunner()
        self.window.frame_reader = _FakeFrameReader(
            image_path=image_path,
            image_bytes=frame_bytes,
        )
        self.window.single_frame_runner = runner
        self.window.capture_frame_button.click()

        self.window.run_single_inference_button.click()
        _process_events(self.app)

        self.assertEqual(runner.calls, [frame_bytes])
        self.assertEqual(self.window.distance_value.text(), "7.250 m")
        self.assertEqual(self.window.yaw_value.text(), "-12.50 deg")
        self.assertIn(
            "Single-frame inference completed",
            self.window.log_panel.toPlainText(),
        )

    def test_continuous_inference_running_refuses_single_frame_run(self) -> None:
        image_path = self._write_temp_frame_image()
        frame_bytes = image_path.read_bytes()
        runner = _FakeSingleFrameRunner()
        self.window.frame_reader = _FakeFrameReader(
            image_path=image_path,
            image_bytes=frame_bytes,
        )
        self.window.single_frame_runner = runner
        self.window.capture_frame_button.click()

        self.window.start_inference_button.click()
        self.window.run_single_inference_button.click()
        _process_events(self.app)

        self.assertEqual(runner.calls, [])
        self.assertIn(
            "Single-frame inference refused because continuous inference is running",
            self.window.log_panel.toPlainText(),
        )

    def test_trace_path_label_updates_when_single_frame_trace_is_written(self) -> None:
        from PySide6.QtWidgets import QLabel

        image_path = self._write_temp_frame_image()
        frame_bytes = image_path.read_bytes()
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        trace_path = Path(temp_dir.name) / "trace"
        runner = _FakeSingleFrameRunner(trace_path=trace_path)
        self.window.frame_reader = _FakeFrameReader(
            image_path=image_path,
            image_bytes=frame_bytes,
        )
        self.window.single_frame_runner = runner
        self.window.capture_frame_button.click()

        self.window.record_trace_checkbox.setChecked(True)
        self.window.run_single_inference_button.click()
        _process_events(self.app)

        label = self.window.findChild(QLabel, "last_trace_path_value")
        assert label is not None
        self.assertEqual(label.text(), str(trace_path))
        self.assertEqual(runner.record_trace_values, [True])

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

        self.assertTrue(_widget_has_pixmap(self._preview_widget()))
        self.assertEqual(self.window.distance_value.text(), "2.500 m")
        self.assertEqual(self.window.yaw_value.text(), "-15.25 deg")
        self.assertEqual(self.window.inference_time_value.text(), "8.0 ms")
        self.assertEqual(self.window.preprocessing_time_value.text(), "3.0 ms")
        self.assertEqual(self.window.total_time_value.text(), "11.0 ms")

    def test_clicking_draw_mask_enables_preview_draw_mode(self) -> None:
        self.window.draw_mask_button.click()
        _process_events(self.app)

        preview = self._preview_widget()
        self.assertTrue(preview.is_mask_editing())
        self.assertEqual(preview.mask_edit_mode(), "draw")

    def test_clicking_stop_apply_commits_mask(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        _process_events(self.app)

        self.window.mask_brush_diameter_input.setValue(20)
        self.window.draw_mask_button.click()
        _mouse_click_center(self._preview_widget())
        self.window.apply_mask_button.click()
        _process_events(self.app)

        snapshot = self.window.mask_state.get_snapshot()
        self.assertTrue(snapshot.enabled)
        self.assertGreater(snapshot.pixel_count, 0)
        self.assertIn("Mask committed", self.window.log_panel.toPlainText())

    def test_clicking_erase_and_apply_subtraction_commits_erased_mask(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        _process_events(self.app)
        self.window.mask_brush_diameter_input.setValue(40)
        self.window.draw_mask_button.click()
        _mouse_click_center(self._preview_widget())
        self.window.apply_mask_button.click()
        before = self.window.mask_state.get_snapshot().pixel_count

        self.window.erase_mask_button.click()
        _mouse_click_center(self._preview_widget())
        self.window.apply_erase_button.click()
        _process_events(self.app)

        after = self.window.mask_state.get_snapshot().pixel_count
        self.assertLess(after, before)

    def test_clear_mask_clears_committed_and_draft_display(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        _process_events(self.app)
        self.window.mask_brush_diameter_input.setValue(20)
        self.window.draw_mask_button.click()
        _mouse_click_center(self._preview_widget())
        self.window.apply_mask_button.click()

        self.window.clear_mask_button.click()
        _process_events(self.app)

        self.assertFalse(self.window.mask_state.get_snapshot().enabled)
        self.assertEqual(self._preview_widget().committed_mask_pixel_count(), 0)

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

    def test_result_ready_updates_preview_overlay_metadata(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        self.inference_controller.signals.result_ready.emit(
            _ResultPayload(
                predicted_distance_m=1.0,
                predicted_yaw_deg=2.0,
                inference_time_ms=3.0,
                preprocessing_time_ms=4.0,
                total_time_ms=7.0,
                roi_metadata=_RoiMetadataPayload(
                    bbox_xyxy_px=(10.0, 12.0, 40.0, 32.0),
                    center_xy_px=(25.0, 22.0),
                    source_image_wh_px=(80, 48),
                    extras={"roi_source_xyxy_px": (0.0, 0.0, 60.0, 48.0)},
                ),
            )
        )
        _process_events(self.app)

        overlay = self._preview_widget().overlay()
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertEqual(overlay.bbox_xyxy_px, (10.0, 12.0, 40.0, 32.0))
        self.assertEqual(overlay.center_xy_px, (25.0, 22.0))
        self.assertEqual(overlay.roi_bounds_xyxy_px, (0.0, 0.0, 60.0, 48.0))

    def test_result_ready_updates_preview_heatmap_overlay_metadata(self) -> None:
        image_path = self._write_temp_frame_image()
        self.camera_controller.signals.frame_written.emit(_FramePayload(image_path=image_path))
        heatmap = np.zeros((48, 80), dtype=np.uint8)
        heatmap[24, 40] = 255
        self.inference_controller.signals.result_ready.emit(
            _ResultPayload(
                predicted_distance_m=1.0,
                predicted_yaw_deg=2.0,
                inference_time_ms=3.0,
                preprocessing_time_ms=4.0,
                total_time_ms=7.0,
                roi_metadata=_RoiMetadataPayload(
                    source_image_wh_px=(80, 48),
                    extras={
                        contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA: {
                            contracts.PREPROCESSING_METADATA_ROI_FCN_HEATMAP_U8: heatmap,
                            contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_WIDTH_PX: 80,
                            contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_HEIGHT_PX: 48,
                            contracts.PREPROCESSING_METADATA_ROI_FCN_RESIZED_IMAGE_WH_PX: (
                                80,
                                48,
                            ),
                            contracts.PREPROCESSING_METADATA_ROI_FCN_PADDING_LTRB_PX: (
                                0,
                                0,
                                0,
                                0,
                            ),
                        }
                    },
                ),
            )
        )
        _process_events(self.app)

        overlay = self._preview_widget().overlay()
        self.assertIsNotNone(overlay)
        assert overlay is not None
        self.assertIsNotNone(overlay.roi_fcn_heatmap)
        assert overlay.roi_fcn_heatmap is not None
        self.assertEqual(overlay.roi_fcn_heatmap.heatmap_u8.shape, (48, 80))
        self.assertEqual(overlay.roi_fcn_heatmap.canvas_wh_px, (80, 48))

    def test_result_ready_updates_roi_confidence_and_acceptance_status(self) -> None:
        self.inference_controller.signals.result_ready.emit(
            _ResultPayload(
                predicted_distance_m=1.0,
                predicted_yaw_deg=2.0,
                inference_time_ms=3.0,
                preprocessing_time_ms=4.0,
                total_time_ms=7.0,
                roi_metadata=_RoiMetadataPayload(
                    source_image_wh_px=(80, 48),
                    extras={
                        contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY: "inverted",
                        contracts.PREPROCESSING_METADATA_ROI_CONFIDENCE: 0.876,
                        contracts.PREPROCESSING_METADATA_ROI_CLIPPED: True,
                        contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX: 6,
                        contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX: 10,
                        contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED: True,
                        contracts.PREPROCESSING_METADATA_ROI_ACCEPTED: True,
                        "apply_manual_mask_to_roi_locator": False,
                        "manual_mask_applied_to_roi_locator": False,
                        "apply_background_removal_to_roi_locator": True,
                        "background_removal_applied_to_roi_locator": True,
                    },
                ),
            )
        )
        _process_events(self.app)

        self.assertEqual(self.window.roi_confidence_value.text(), "0.876")
        self.assertEqual(self.window.roi_clipped_value.text(), "yes")
        self.assertEqual(self.window.roi_clip_amount_value.text(), "6 px / tol 10 px tolerated")
        self.assertEqual(self.window.roi_acceptance_value.text(), "accepted")
        self.assertIn("ROI confidence 0.876", self.window.roi_locator_status_label.text())
        self.assertIn("clip 6 px / tol 10 px", self.window.roi_locator_status_label.text())
        self.assertIn("polarity inverted", self.window.roi_locator_transforms_value.text())
        self.assertIn("background yes", self.window.roi_locator_transforms_value.text())

    def test_rejected_roi_error_clears_prediction_labels_and_updates_status(self) -> None:
        self.window.distance_value.setText("7.250 m")
        self.window.yaw_value.setText("-12.50 deg")

        self.inference_controller.signals.error_occurred.emit(
            _IssuePayload(
                error_type="roi_rejected",
                message="ROI rejected",
                details={
                    contracts.PREPROCESSING_METADATA_ROI_CONFIDENCE: 0.12,
                    contracts.PREPROCESSING_METADATA_ROI_CLIPPED: True,
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX: 126,
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX: 10,
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED: False,
                    contracts.PREPROCESSING_METADATA_ROI_ACCEPTED: False,
                    contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASON: "low_confidence",
                    "apply_manual_mask_to_roi_locator": False,
                    "manual_mask_applied_to_roi_locator": False,
                    "apply_background_removal_to_roi_locator": False,
                    "background_removal_applied_to_roi_locator": False,
                },
            )
        )
        _process_events(self.app)

        self.assertEqual(self.window.distance_value.text(), "n/a")
        self.assertEqual(self.window.yaw_value.text(), "n/a")
        self.assertEqual(self.window.roi_confidence_value.text(), "0.120")
        self.assertEqual(self.window.roi_clipped_value.text(), "yes")
        self.assertEqual(self.window.roi_clip_amount_value.text(), "126 px / tol 10 px")
        self.assertIn("rejected", self.window.roi_acceptance_value.text())
        self.assertIsNone(self._preview_widget().overlay())

    def test_result_ready_shows_debug_artifact_paths(self) -> None:
        self.inference_controller.signals.result_ready.emit(
            _ResultPayload(
                predicted_distance_m=1.0,
                predicted_yaw_deg=2.0,
                inference_time_ms=3.0,
                preprocessing_time_ms=4.0,
                total_time_ms=7.0,
                debug_paths={
                    "accepted_raw_frame": Path("live_debug/raw.png"),
                    "x_distance_image": Path("live_debug/distance.png"),
                },
            )
        )
        _process_events(self.app)

        self.assertIn("accepted_raw_frame", self.window.debug_artifacts_value.text())
        self.assertIn("live_debug/raw.png", self.window.debug_artifacts_value.text())
        self.assertIn("Debug artifacts", self.window.log_panel.toPlainText())

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

    def _preview_widget(self) -> object:
        from live_inference.gui.frame_preview_widget import FramePreviewWidget

        preview = self.window.findChild(FramePreviewWidget, "frame_preview_widget")
        self.assertIsNotNone(preview)
        return preview

    def _background_preview_widget(self) -> object:
        from live_inference.gui.frame_preview_widget import FramePreviewWidget

        preview = self.window.findChild(FramePreviewWidget, "background_preview_widget")
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
    roi_metadata: object | None = None
    debug_paths: dict[str, Path] | None = None


@dataclass(frozen=True)
class _RoiMetadataPayload:
    bbox_xyxy_px: tuple[float, float, float, float] | None = None
    center_xy_px: tuple[float, float] | None = None
    source_image_wh_px: tuple[int, int] | None = None
    extras: dict[str, Any] | None = None


@dataclass(frozen=True)
class _IssuePayload:
    message: str
    warning_type: str | None = None
    error_type: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class _FramePayload:
    image_path: Path
    completed_at_utc: str | None = None
    frame_hash: object | None = None
    metadata: object | None = None


@dataclass(frozen=True)
class _SingleFrameOutcomePayload:
    result: object | None
    error: object | None
    trace_path: Path | None
    frame_hash: object | None = None
    request_id: str = "single-frame-request"


class _FakeFrameReader:
    def __init__(self, *, image_path: Path, image_bytes: bytes) -> None:
        self.image_path = image_path
        self.image_bytes = image_bytes

    def latest_completed_frame(self) -> _FramePayload:
        return _FramePayload(
            image_path=self.image_path,
            metadata=contracts.FrameMetadata(frame_id="fake-frame"),
        )

    def read_frame_bytes(self, frame: object) -> bytes:
        return self.image_bytes


class _FakeSingleFrameRunner:
    def __init__(self, *, trace_path: Path | None = None) -> None:
        self.trace_path = trace_path
        self.calls: list[bytes] = []
        self.record_trace_values: list[bool] = []

    @property
    def trace_output_dir(self) -> Path:
        return Path("live_traces")

    def run_single_frame(
        self,
        image_bytes: bytes,
        *,
        source_path: Path | None = None,
        frame_metadata: object | None = None,
        record_trace: bool = False,
    ) -> _SingleFrameOutcomePayload:
        self.calls.append(bytes(image_bytes))
        self.record_trace_values.append(bool(record_trace))
        return _SingleFrameOutcomePayload(
            result=_ResultPayload(
                predicted_distance_m=7.25,
                predicted_yaw_deg=-12.5,
                inference_time_ms=5.0,
                preprocessing_time_ms=2.0,
                total_time_ms=7.0,
            ),
            error=None,
            trace_path=self.trace_path,
        )


class _FakeController:
    def __init__(self, signals: object) -> None:
        self.signals = signals
        self.start_calls = 0
        self.stop_calls = 0
        self.wait_calls = 0
        self.running = False

    def start(self) -> None:
        self.start_calls += 1
        self.running = True

    def request_stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def wait(self, timeout_ms: int | None = None) -> bool:
        self.wait_calls += 1
        return True

    def is_running(self) -> bool:
        return self.running


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


def _widget_has_pixmap(widget: object) -> bool:
    pixmap = widget.pixmap()
    return pixmap is not None and not pixmap.isNull()


def _is_descendant(child: object, ancestor: object) -> bool:
    current = child
    while current is not None:
        if current is ancestor:
            return True
        parent = getattr(current, "parentWidget", lambda: None)()
        current = parent
    return False


def _mouse_click(widget: Any, x: int, y: int) -> None:
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, QPoint(x, y))


def _mouse_click_center(widget: Any) -> None:
    _mouse_click(widget, max(1, widget.width() // 2), max(1, widget.height() // 2))


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
