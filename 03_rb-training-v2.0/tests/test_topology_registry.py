"""Tests for topology registry and multi-topology resolution."""

from __future__ import annotations

import unittest

from torch import nn

from src.topologies import (
    DEFAULT_TOPOLOGY_ID,
    architecture_text_from_spec,
    build_model_from_spec,
    canonicalize_task_contract,
    list_topology_ids,
    list_topology_variants,
    resolve_topology_spec,
    resolve_topology_spec_from_mapping,
    topology_spec_signature,
    topology_contract_signature,
    task_contract_signature,
)


class TopologyRegistryTests(unittest.TestCase):
    def test_registry_exposes_multiple_topologies(self) -> None:
        topology_ids = list_topology_ids()
        self.assertGreaterEqual(len(topology_ids), 2)
        self.assertIn("distance_regressor_2d_cnn", topology_ids)
        self.assertIn("distance_regressor_dual_stream", topology_ids)
        self.assertIn("distance_regressor_dual_stream_yaw", topology_ids)
        self.assertIn("distance_regressor_dual_stream_yaw_from_tri_stream", topology_ids)
        self.assertIn("distance_regressor_tri_stream_yaw", topology_ids)
        self.assertIn("distance_regressor_global_pool_cnn", topology_ids)

    def test_registry_can_exclude_deprecated_topologies(self) -> None:
        topology_ids = list_topology_ids(include_deprecated=False)

        self.assertIn("distance_regressor_dual_stream", topology_ids)
        self.assertIn("distance_regressor_dual_stream_yaw", topology_ids)
        self.assertIn("distance_regressor_dual_stream_yaw_from_tri_stream", topology_ids)
        self.assertIn("distance_regressor_tri_stream_yaw", topology_ids)
        self.assertNotIn("distance_regressor_2d_cnn", topology_ids)
        self.assertNotIn("distance_regressor_global_pool_cnn", topology_ids)

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

    def test_legacy_topology_receives_synthesized_topology_contract(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_2d_cnn",
            topology_variant="fast_v0_2",
            topology_params={},
        )

        self.assertIn("topology_contract", spec.to_dict())
        self.assertEqual(spec.topology_contract["reporting"]["family"], "distance_regression")
        self.assertEqual(spec.task_contract["reporting"]["family"], "distance_regression")

    def test_task_contract_signature_ignores_reporting_extensions(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_dual_stream_yaw",
            topology_variant="dual_stream_yaw_v0_1",
            topology_params={},
        )

        runtime_only = canonicalize_task_contract(spec.task_contract)

        self.assertEqual(task_contract_signature(spec), task_contract_signature(runtime_only))
        self.assertEqual(
            topology_contract_signature(spec),
            topology_contract_signature(spec.topology_contract),
        )

    def test_task_contract_signature_keeps_implicit_huber_legacy_compatible(self) -> None:
        legacy_contract = {
            "task_family": "multitask_regression",
            "prediction_mode": "distance_yaw_sincos",
            "input_mode": "tri_stream_distance_orientation_geometry",
            "output_kind": "mapping",
            "target_columns": ["distance_m", "yaw_sin", "yaw_cos"],
            "debug_target_columns": ["yaw_deg"],
            "heads": {
                "distance": {
                    "target_columns": ["distance_m"],
                    "metrics_role": "distance",
                    "loss_role": "distance",
                    "target_kind": "regression",
                    "target_npz_key": "y_distance_m",
                    "output_key": "distance_m",
                },
                "orientation": {
                    "target_columns": ["yaw_sin", "yaw_cos"],
                    "debug_target_columns": ["yaw_deg"],
                    "metrics_role": "orientation",
                    "loss_role": "orientation",
                    "target_kind": "circular_regression",
                    "target_npz_keys": ["y_yaw_sin", "y_yaw_cos"],
                    "debug_target_npz_key": "y_yaw_deg",
                    "output_key": "yaw_sin_cos",
                },
            },
        }
        explicit_huber_contract = {
            **legacy_contract,
            "heads": {
                name: {**head, "loss_kind": "huber"}
                for name, head in legacy_contract["heads"].items()
            },
        }
        non_default_contract = {
            **legacy_contract,
            "heads": {
                **legacy_contract["heads"],
                "orientation": {
                    **legacy_contract["heads"]["orientation"],
                    "loss_kind": "bce_with_logits",
                },
            },
        }

        self.assertEqual(
            task_contract_signature(legacy_contract),
            task_contract_signature(explicit_huber_contract),
        )
        self.assertNotEqual(
            task_contract_signature(legacy_contract),
            task_contract_signature(non_default_contract),
        )


if __name__ == "__main__":
    unittest.main()
