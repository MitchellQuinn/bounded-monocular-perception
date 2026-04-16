"""Tests for the dual-stream distance + yaw topology."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from src.data import load_root_metadata, validate_root_schema, validate_task_contract_schema
from src.topologies import build_model_from_spec, resolve_topology_spec


class DualStreamYawTests(unittest.TestCase):
    def test_dual_stream_yaw_forward_matches_mapping_contract(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_dual_stream_yaw",
            topology_variant="dual_stream_yaw_v0_1",
            topology_params={},
        )
        model = build_model_from_spec(spec)

        outputs = model(
            {
                "silhouette_crop": torch.rand(4, 1, 300, 300),
                "bbox_features": torch.rand(4, 10),
            }
        )

        self.assertEqual(spec.task_contract["prediction_mode"], "distance_yaw_sincos")
        self.assertEqual(spec.task_contract["input_mode"], "dual_stream_image_bbox_features")
        self.assertEqual(spec.topology_contract["reporting"]["family"], "distance_orientation_multitask")
        self.assertEqual(
            spec.topology_contract["targets"]["yaw"]["debug_columns"],
            ["yaw_deg"],
        )
        self.assertEqual(set(outputs.keys()), {"distance_m", "yaw_sin_cos"})
        self.assertEqual(tuple(outputs["distance_m"].shape), (4,))
        self.assertEqual(tuple(outputs["yaw_sin_cos"].shape), (4, 2))

    def test_smoketest_schema_matches_dual_stream_yaw_contract(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        corpus_dir = repo_root / "datasets" / "26-04-11_v020-train-shuffled-images-smoketest"
        with TemporaryDirectory() as tmp:
            temp_root = Path(tmp)
            (temp_root / corpus_dir.name).symlink_to(corpus_dir, target_is_directory=True)
            metadata_df, _ = load_root_metadata(
                temp_root,
                source_root="smoketest",
                repo_root=repo_root,
            )
            schema_df = validate_root_schema(metadata_df, root_name="smoketest")
            spec = resolve_topology_spec(
                topology_id="distance_regressor_dual_stream_yaw",
                topology_variant="dual_stream_yaw_v0_1",
                topology_params={},
            )

            validate_task_contract_schema(
                metadata_df,
                schema_df,
                spec.task_contract,
                root_name="smoketest",
            )


if __name__ == "__main__":
    unittest.main()
