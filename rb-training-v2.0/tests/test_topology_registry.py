"""Tests for topology registry and multi-topology resolution."""

from __future__ import annotations

import unittest

from torch import nn

from src.topologies import (
    DEFAULT_TOPOLOGY_ID,
    architecture_text_from_spec,
    build_model_from_spec,
    list_topology_ids,
    list_topology_variants,
    resolve_topology_spec,
    resolve_topology_spec_from_mapping,
    topology_spec_signature,
)


class TopologyRegistryTests(unittest.TestCase):
    def test_registry_exposes_multiple_topologies(self) -> None:
        topology_ids = list_topology_ids()
        self.assertGreaterEqual(len(topology_ids), 2)
        self.assertIn("distance_regressor_2d_cnn", topology_ids)
        self.assertIn("distance_regressor_global_pool_cnn", topology_ids)

    def test_legacy_mapping_resolves_to_default_topology(self) -> None:
        spec = resolve_topology_spec_from_mapping(
            {"model_architecture_variant": "fast_v0_2"}
        )
        self.assertEqual(spec.topology_id, DEFAULT_TOPOLOGY_ID)
        self.assertEqual(spec.topology_variant, "fast_v0_2")

    def test_build_model_from_resolved_specs(self) -> None:
        for topology_id in list_topology_ids():
            variants = list_topology_variants(topology_id)
            self.assertGreaterEqual(len(variants), 1)
            spec = resolve_topology_spec(
                topology_id=topology_id,
                topology_variant=variants[0],
                topology_params={},
            )
            model = build_model_from_spec(spec)
            self.assertIsInstance(model, nn.Module)
            text = architecture_text_from_spec(model, spec)
            self.assertIn(f"topology_id={topology_id}", text)

    def test_signature_changes_across_variants(self) -> None:
        base = resolve_topology_spec(
            topology_id="distance_regressor_2d_cnn",
            topology_variant="plain_v0_1",
            topology_params={},
        )
        changed = resolve_topology_spec(
            topology_id="distance_regressor_2d_cnn",
            topology_variant="fast_v0_2",
            topology_params={},
        )
        self.assertNotEqual(
            topology_spec_signature(base),
            topology_spec_signature(changed),
        )


if __name__ == "__main__":
    unittest.main()
