"""Unit tests for v4 silhouette algorithm primitives."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from rb_pipeline_v4.silhouette_algorithms import ConvexHullFallbackV1


class V4SilhouetteAlgorithmsTests(unittest.TestCase):
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
