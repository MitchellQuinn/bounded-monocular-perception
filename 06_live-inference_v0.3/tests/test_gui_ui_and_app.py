from __future__ import annotations

from pathlib import Path
import os
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

import interfaces.contracts as contracts  # noqa: E402
from live_inference.gui.app import build_live_inference_gui_context  # noqa: E402
from live_inference.gui.main_window import LiveInferenceMainWindow  # noqa: E402


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
        finally:
            context.camera_controller.request_stop()
            context.inference_controller.request_stop()
            context.camera_controller.wait(10)
            context.inference_controller.wait(10)


if __name__ == "__main__":
    unittest.main()
