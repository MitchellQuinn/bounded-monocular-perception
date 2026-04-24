"""Tests for deterministic foreground-only brightness normalization."""

from __future__ import annotations

import unittest

import numpy as np

from rb_pipeline_v4.brightness_normalization import apply_brightness_normalization_v4
from rb_pipeline_v4.config import BrightnessNormalizationConfigV4


class BrightnessNormalizationTests(unittest.TestCase):
    def test_masked_median_darkness_gain_preserves_background_and_targets_median(self) -> None:
        image = np.asarray(
            [
                [1.0, 0.8, 0.4],
                [0.9, 0.7, 1.0],
            ],
            dtype=np.float32,
        )
        mask = np.asarray(
            [
                [False, True, True],
                [False, True, False],
            ],
            dtype=bool,
        )

        result = apply_brightness_normalization_v4(
            image,
            mask,
            BrightnessNormalizationConfigV4(
                enabled=True,
                method="masked_median_darkness_gain",
                target_median_darkness=0.5,
                min_gain=0.1,
                max_gain=10.0,
            ),
        )

        normalized_darkness = 1.0 - result.image[mask]
        self.assertEqual(result.status, "success")
        self.assertAlmostEqual(float(np.median(normalized_darkness)), 0.5, places=6)
        np.testing.assert_array_equal(result.image[~mask], image[~mask])

    def test_empty_mask_skip_returns_image_unchanged(self) -> None:
        image = np.asarray([[1.0, 0.7], [1.0, 0.6]], dtype=np.float32)
        mask = np.zeros(image.shape, dtype=bool)

        result = apply_brightness_normalization_v4(
            image,
            mask,
            BrightnessNormalizationConfigV4(
                enabled=True,
                method="masked_median_darkness_gain",
                empty_mask_policy="skip",
            ),
        )

        self.assertEqual(result.status, "skipped_empty_mask")
        self.assertEqual(result.foreground_pixel_count, 0)
        np.testing.assert_array_equal(result.image, image)

    def test_empty_mask_fail_raises(self) -> None:
        image = np.asarray([[1.0, 0.7], [1.0, 0.6]], dtype=np.float32)
        mask = np.zeros(image.shape, dtype=bool)

        with self.assertRaisesRegex(ValueError, "foreground mask is empty"):
            apply_brightness_normalization_v4(
                image,
                mask,
                BrightnessNormalizationConfigV4(
                    enabled=True,
                    method="masked_median_darkness_gain",
                    empty_mask_policy="fail",
                ),
            )

    def test_gain_is_clamped(self) -> None:
        image = np.asarray([[0.9, 0.9, 1.0]], dtype=np.float32)
        mask = np.asarray([[True, True, False]], dtype=bool)

        result = apply_brightness_normalization_v4(
            image,
            mask,
            BrightnessNormalizationConfigV4(
                enabled=True,
                method="masked_median_darkness_gain",
                target_median_darkness=0.8,
                min_gain=0.5,
                max_gain=2.0,
            ),
        )

        self.assertAlmostEqual(result.gain, 2.0, places=6)
        self.assertAlmostEqual(float(np.median(1.0 - result.image[mask])), 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
