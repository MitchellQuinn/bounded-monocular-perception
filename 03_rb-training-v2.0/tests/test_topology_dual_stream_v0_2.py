"""Tests for the retired-v0.1 replacement dual-stream topology."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.topologies import build_model_from_spec, resolve_topology_spec


class DualStreamV02Tests(unittest.TestCase):
    def test_dual_stream_v0_2_uses_group_norm_and_no_dropout_by_default(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_dual_stream",
            topology_variant="dual_stream_v0_2",
            topology_params={},
        )
        model = build_model_from_spec(spec)

        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))
        self.assertGreaterEqual(
            sum(isinstance(module, nn.GroupNorm) for module in model.modules()),
            5,
        )
        self.assertFalse(any(isinstance(module, nn.Dropout) for module in model.modules()))

    def test_dual_stream_v0_2_forward_matches_scalar_distance_contract(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_dual_stream",
            topology_variant="dual_stream_v0_2",
            topology_params={},
        )
        model = build_model_from_spec(spec)

        batch = torch.rand(4, 1, 224, 224)
        outputs = model(batch)

        self.assertEqual(tuple(outputs.shape), (4,))

    def test_dual_stream_v0_1_is_no_longer_resolvable(self) -> None:
        with self.assertRaises(ValueError):
            resolve_topology_spec(
                topology_id="distance_regressor_dual_stream",
                topology_variant="dual_stream_v0_1",
                topology_params={},
            )


if __name__ == "__main__":
    unittest.main()
