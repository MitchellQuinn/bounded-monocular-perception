"""Unit tests for v4 silhouette algorithm primitives."""

from __future__ import annotations

from pathlib import Path
import unittest

import cv2
import numpy as np

from rb_pipeline_v4.silhouette_algorithms import ContourSilhouetteGeneratorV2, ConvexHullFallbackV1


class V4SilhouetteAlgorithmsTests(unittest.TestCase):
    def test_generator_recovers_from_sparse_edges_via_intensity_threshold(self) -> None:
        # Low-contrast rectangle where high Canny thresholds produce sparse/failed edges.
        roi = np.full((96, 96), 24, dtype=np.uint8)
        cv2.rectangle(roi, (22, 20), (74, 76), color=52, thickness=-1)
        roi = cv2.GaussianBlur(roi, (5, 5), 0)

        output = ContourSilhouetteGeneratorV2().generate(
            roi,
            blur_kernel_size=5,
            canny_low_threshold=90,
            canny_high_threshold=180,
            close_kernel_size=1,
            dilate_kernel_size=1,
            min_component_area_px=50,
            fill_holes=True,
        )

        self.assertIsNotNone(output.contour)
        self.assertEqual(output.primary_reason, "")
        self.assertTrue((output.diagnostics or {}).get("recovered_via") == "intensity_otsu_v1")

        assert output.contour is not None
        rendered = np.full(roi.shape, 255, dtype=np.uint8)
        cv2.drawContours(rendered, [output.contour], contourIdx=-1, color=0, thickness=cv2.FILLED)
        self.assertGreater(int(np.count_nonzero(rendered < 128)), 1800)

    def test_intensity_recovery_selects_large_border_touching_vehicle_component(self) -> None:
        fixture = (
            Path(__file__).resolve().parents[2]
            / "failure-analysis"
            / "incident_1"
            / "20260518T082310Z__8ed41d13-9fbb-45ad-8083-dcdc385667e6__e0001d54"
            / "roi_crop.png"
        )
        roi = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(roi, str(fixture))
        assert roi is not None

        output = ContourSilhouetteGeneratorV2().generate(
            roi,
            blur_kernel_size=5,
            canny_low_threshold=50,
            canny_high_threshold=150,
            close_kernel_size=3,
            dilate_kernel_size=1,
            min_component_area_px=50,
            fill_holes=True,
        )

        self.assertIsNotNone(output.contour)
        self.assertEqual(output.primary_reason, "")
        diagnostics = output.diagnostics or {}
        self.assertEqual(diagnostics.get("recovered_via"), "intensity_otsu_v1")
        self.assertEqual(diagnostics.get("recovery_threshold_mode"), "binary_inv")
        self.assertTrue(diagnostics.get("recovery_component_touches_border"))

        assert output.contour is not None
        x, y, w, h = cv2.boundingRect(output.contour)
        self.assertLessEqual(x, 45)
        self.assertLessEqual(y, 20)
        self.assertGreaterEqual(w, 230)
        self.assertGreaterEqual(h, 290)
        self.assertGreaterEqual(int(diagnostics.get("selected_component_area", 0)), 40_000)

    def test_convex_hull_fallback_recovers_collinear_points(self) -> None:
        fallback_mask = np.zeros((32, 32), dtype=np.uint8)
        fallback_mask[16, 6:26] = 255

        contour, reason = ConvexHullFallbackV1().recover(fallback_mask)

        self.assertIsNotNone(contour)
        self.assertEqual(reason, "")
        assert contour is not None
        self.assertGreaterEqual(int(contour.shape[0]), 3)
        self.assertGreaterEqual(float(abs(cv2.contourArea(contour))), 1.0)

    def test_convex_hull_fallback_still_fails_for_insufficient_points(self) -> None:
        fallback_mask = np.zeros((32, 32), dtype=np.uint8)
        fallback_mask[10, 10] = 255
        fallback_mask[10, 11] = 255

        contour, reason = ConvexHullFallbackV1().recover(fallback_mask)

        self.assertIsNone(contour)
        self.assertEqual(reason, "insufficient_points_for_hull")


if __name__ == "__main__":
    unittest.main()
