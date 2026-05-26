from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live_inference.gui.frame_preview_widget import FramePreviewOverlay  # noqa: E402
from live_inference.gui.overlay_smoothing import FramePreviewOverlaySmoother  # noqa: E402


class FramePreviewOverlaySmootherTests(unittest.TestCase):
    def test_smooths_roi_center_while_keeping_latest_roi_size(self) -> None:
        smoother = FramePreviewOverlaySmoother(window_seconds=1.0)

        smoother.smooth_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(640, 480),
                center_xy_px=(50.0, 50.0),
                roi_bounds_xyxy_px=(0.0, 0.0, 100.0, 100.0),
            ),
            now_seconds=0.0,
        )
        smoothed = smoother.smooth_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(640, 480),
                center_xy_px=(70.0, 70.0),
                roi_bounds_xyxy_px=(40.0, 40.0, 120.0, 120.0),
            ),
            now_seconds=0.5,
        )

        self.assertIsNotNone(smoothed)
        assert smoothed is not None
        self.assertEqual(smoothed.center_xy_px, (60.0, 60.0))
        self.assertEqual(smoothed.roi_bounds_xyxy_px, (25.0, 25.0, 105.0, 105.0))

    def test_smooths_bbox_when_enabled(self) -> None:
        smoother = FramePreviewOverlaySmoother(window_seconds=1.0, smooth_bbox=True)

        smoother.smooth_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(640, 480),
                bbox_xyxy_px=(0.0, 0.0, 20.0, 10.0),
            ),
            now_seconds=0.0,
        )
        smoothed = smoother.smooth_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(640, 480),
                bbox_xyxy_px=(10.0, 0.0, 30.0, 20.0),
            ),
            now_seconds=0.25,
        )

        self.assertIsNotNone(smoothed)
        assert smoothed is not None
        self.assertEqual(smoothed.bbox_xyxy_px, (5.0, 0.0, 25.0, 15.0))

    def test_source_size_change_resets_history(self) -> None:
        smoother = FramePreviewOverlaySmoother(window_seconds=1.0)

        smoother.smooth_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(640, 480),
                roi_bounds_xyxy_px=(0.0, 0.0, 100.0, 100.0),
            ),
            now_seconds=0.0,
        )
        smoothed = smoother.smooth_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(800, 600),
                roi_bounds_xyxy_px=(40.0, 40.0, 120.0, 120.0),
            ),
            now_seconds=0.25,
        )

        self.assertIsNotNone(smoothed)
        assert smoothed is not None
        self.assertEqual(smoothed.roi_bounds_xyxy_px, (40.0, 40.0, 120.0, 120.0))

    def test_window_trims_old_samples(self) -> None:
        smoother = FramePreviewOverlaySmoother(window_seconds=0.5)

        smoother.smooth_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(640, 480),
                center_xy_px=(0.0, 0.0),
            ),
            now_seconds=0.0,
        )
        smoother.smooth_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(640, 480),
                center_xy_px=(10.0, 10.0),
            ),
            now_seconds=0.25,
        )
        smoothed = smoother.smooth_overlay(
            FramePreviewOverlay(
                source_image_wh_px=(640, 480),
                center_xy_px=(30.0, 30.0),
            ),
            now_seconds=0.8,
        )

        self.assertIsNotNone(smoothed)
        assert smoothed is not None
        self.assertEqual(smoothed.center_xy_px, (30.0, 30.0))


if __name__ == "__main__":
    unittest.main()
