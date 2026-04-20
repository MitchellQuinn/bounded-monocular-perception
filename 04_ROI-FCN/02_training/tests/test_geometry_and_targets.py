from __future__ import annotations

import unittest

import numpy as np
import torch

from roi_fcn_training_v0_1.geometry import canvas_point_to_output_space, decode_heatmap_argmax
from roi_fcn_training_v0_1.targets import (
    build_balanced_heatmap_weights,
    build_gaussian_heatmaps,
    compute_heatmap_loss,
)


class GeometryAndTargetsTests(unittest.TestCase):
    def test_gaussian_peak_decodes_back_near_target(self) -> None:
        target_canvas = torch.tensor([[24.0, 18.0]], dtype=torch.float32)
        heatmap = build_gaussian_heatmaps(
            target_canvas,
            canvas_hw=(64, 96),
            output_hw=(16, 24),
            sigma_px=1.5,
        )[0, 0].cpu().numpy()
        decoded = decode_heatmap_argmax(
            heatmap,
            canvas_hw=(64, 96),
            resize_scale=1.0,
            pad_left_px=0.0,
            pad_top_px=0.0,
            source_wh_px=np.asarray([96, 64], dtype=np.int32),
        )
        self.assertAlmostEqual(decoded.original_x, 24.0, delta=4.5)
        self.assertAlmostEqual(decoded.original_y, 18.0, delta=4.5)

    def test_canvas_to_output_mapping_is_scaled(self) -> None:
        mapped = canvas_point_to_output_space(
            np.asarray([48.0, 32.0], dtype=np.float32),
            canvas_hw=(64, 96),
            output_hw=(16, 24),
        )
        self.assertTrue(np.allclose(mapped, np.asarray([12.0, 8.0], dtype=np.float32)))

    def test_balanced_heatmap_loss_penalizes_blank_prediction_more_than_plain_mse(self) -> None:
        target_canvas = torch.tensor([[240.0, 150.0]], dtype=torch.float32)
        target_heatmaps = build_gaussian_heatmaps(
            target_canvas,
            canvas_hw=(300, 480),
            output_hw=(300, 480),
            sigma_px=2.5,
        )
        blank_prediction = torch.zeros_like(target_heatmaps)

        plain_loss = compute_heatmap_loss(
            blank_prediction,
            target_heatmaps,
            loss_name="mse_heatmap",
        )
        balanced_loss = compute_heatmap_loss(
            blank_prediction,
            target_heatmaps,
            loss_name="balanced_mse_heatmap",
            positive_threshold=0.05,
        )
        weights = build_balanced_heatmap_weights(target_heatmaps, positive_threshold=0.05)

        self.assertGreater(float(balanced_loss), float(plain_loss) * 10.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
