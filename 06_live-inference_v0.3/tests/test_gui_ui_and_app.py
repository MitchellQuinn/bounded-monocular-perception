from __future__ import annotations

from pathlib import Path
import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QGroupBox, QSplitter  # noqa: E402

import interfaces.contracts as contracts  # noqa: E402
from live_inference.gui.app import build_live_inference_gui_context  # noqa: E402
from live_inference.gui.main_window import LiveInferenceMainWindow  # noqa: E402
from live_inference.preprocessing import (  # noqa: E402
    CAMERA_INTRINSICS_MODE_REAL_TO_UNITY_INTRINSICS_REMAP,
    CameraIntrinsicsTransformState,
    ForegroundExtractionPolicyState,
)


class _Controller:
    signals = None

    def start(self) -> None:
        pass

    def request_stop(self) -> None:
        pass

    def wait(self, _timeout_ms: int) -> bool:
        return True


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
        self.assertIsInstance(window.top_workspace_splitter, QSplitter)
        self.assertEqual(window.top_workspace_splitter.count(), 4)
        self.assertIsNotNone(window.findChild(QGroupBox, "additionalControlsGroup"))
        self.assertEqual(window.show_roi_checkbox.parent().objectName(), "overlayDebugGroup")

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
        finally:
            context.camera_controller.request_stop()
            context.inference_controller.request_stop()
            context.camera_controller.wait(10)
            context.inference_controller.wait(10)


if __name__ == "__main__":
    unittest.main()
