"""Tests for the live frame preview overlay widget."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class FramePreviewWidgetTests(unittest.TestCase):
    def setUp(self) -> None:
        QApplication, _, _, FramePreviewWidget = _gui_imports()
        self.app = _application(QApplication)
        self.widget = FramePreviewWidget()
        self.widget.resize(400, 400)

    def tearDown(self) -> None:
        self.widget.hide()
        self.widget.deleteLater()
        self.app.processEvents()

    def test_overlay_bbox_can_be_set_without_crashing(self) -> None:
        _, QPixmap, FramePreviewOverlay, _ = _gui_imports()
        self.widget.set_pixmap(QPixmap.fromImage(_image(200, 100)))

        self.widget.set_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(200, 100),
                bbox_xyxy_px=(50.0, 25.0, 150.0, 75.0),
                center_xy_px=(100.0, 50.0),
            )
        )

        self.assertIsNotNone(self.widget.grab())
        self.assertIsNotNone(self.widget.overlay())

    def test_source_to_widget_mapping_accounts_for_letterboxing(self) -> None:
        _, QPixmap, _, _ = _gui_imports()
        self.widget.set_pixmap(QPixmap.fromImage(_image(200, 100)))

        mapped = self.widget.map_source_rect_to_widget(
            (50.0, 25.0, 150.0, 75.0),
            (200, 100),
        )

        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertAlmostEqual(mapped.left(), 100.0)
        self.assertAlmostEqual(mapped.top(), 150.0)
        self.assertAlmostEqual(mapped.width(), 200.0)
        self.assertAlmostEqual(mapped.height(), 100.0)

    def test_invalid_or_missing_source_size_skips_overlay_safely(self) -> None:
        _, QPixmap, FramePreviewOverlay, _ = _gui_imports()
        self.widget.set_pixmap(QPixmap.fromImage(_image(200, 100)))

        self.widget.set_overlay(
            FramePreviewOverlay(
                source_image_wh_px=None,
                bbox_xyxy_px=(10.0, 10.0, 50.0, 50.0),
            )
        )

        self.assertIsNone(
            self.widget.map_source_rect_to_widget((10.0, 10.0, 50.0, 50.0), None)
        )
        self.assertFalse(self.widget.overlay_source_size_matches(None))
        self.assertIsNotNone(self.widget.grab())

    def test_preview_loads_normal_frame_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "frame.png"
            self.assertTrue(_image(80, 48).save(str(image_path)))

            loaded = self.widget.load_image(image_path)

            self.assertTrue(loaded)
            self.assertIsNotNone(self.widget.pixmap())
            self.assertEqual(self.widget.source_image_size(), (80, 48))


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
        from PySide6.QtGui import QColor, QImage, QPixmap
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        _PYSIDE6_IMPORT_ERROR = exc
        raise unittest.SkipTest(
            f"PySide6 unavailable; skipping GUI tests: {exc}"
        ) from exc

    from live_inference.gui.frame_preview_widget import (
        FramePreviewOverlay,
        FramePreviewWidget,
    )

    _GUI_IMPORTS = (QApplication, QPixmap, FramePreviewOverlay, FramePreviewWidget)
    return _GUI_IMPORTS


def _application(QApplication: Any) -> Any:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    if not isinstance(app, QApplication):
        raise unittest.SkipTest("A QCoreApplication already exists; QWidget tests need QApplication.")
    app.setQuitOnLastWindowClosed(False)
    return app


def _image(width: int, height: int) -> Any:
    from PySide6.QtGui import QColor, QImage

    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor(40, 120, 200))
    return image


if __name__ == "__main__":
    unittest.main()
