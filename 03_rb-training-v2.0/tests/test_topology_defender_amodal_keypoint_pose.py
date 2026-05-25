"""Tests for the Defender amodal keypoint pose topology."""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data import (
    BBOX_FEATURE_COLUMNS,
    BBOX_FEATURE_SCHEMA,
    TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
    TRI_STREAM_GEOMETRY_ARRAY_KEY,
    TRI_STREAM_INPUT_MODE,
    TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,
    load_root_metadata,
    validate_root_schema,
    validate_task_contract_schema,
)
from src.task_runtime import (
    compute_task_loss,
    extract_prediction_heads,
    summarize_task_metrics,
)
from src.topologies import (
    architecture_text_from_spec,
    build_model_from_spec,
    list_topology_ids,
    list_topology_variants,
    resolve_topology_spec,
)

TOPOLOGY_ID = "defender_amodal_keypoint_pose"
VARIANT = "defender_amodal_keypoint_pose_v0_1"
REQUIRED_OUTPUT_KEYS = {
    "distance_m",
    "yaw_sin_cos",
    "defender_center_3d",
    "defender_keypoints_3d_flat",
    "defender_keypoints_visible_logits",
}


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY: torch.rand(batch_size, 1, 64, 64),
        TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY: torch.rand(batch_size, 1, 64, 64),
        TRI_STREAM_GEOMETRY_ARRAY_KEY: torch.rand(batch_size, 10),
    }


class DefenderAmodalKeypointPoseTests(unittest.TestCase):
    def test_registry_exposes_defender_family_and_v0_1_variant(self) -> None:
        self.assertIn(TOPOLOGY_ID, list_topology_ids())
        self.assertIn(VARIANT, list_topology_variants(TOPOLOGY_ID))

    def test_resolve_spec_and_build_model(self) -> None:
        spec = resolve_topology_spec(
            topology_id=TOPOLOGY_ID,
            topology_variant=VARIANT,
            topology_params={},
        )
        model = build_model_from_spec(spec)
        text = architecture_text_from_spec(model, spec)

        self.assertIsInstance(model, nn.Module)
        self.assertEqual(spec.model_class_name, "DefenderAmodalKeypointPoseRegressor")
        self.assertEqual(spec.task_contract["prediction_mode"], TOPOLOGY_ID)
        self.assertEqual(spec.task_contract["input_mode"], TRI_STREAM_INPUT_MODE)
        self.assertEqual(spec.task_contract["output_kind"], "mapping")
        self.assertIn(f"topology_id={TOPOLOGY_ID}", text)
        self.assertEqual(
            set(spec.task_contract["heads"]),
            {
                "distance",
                "orientation",
                "defender_center_3d",
                "defender_keypoints_3d",
                "defender_keypoints_visible",
            },
        )

    def test_forward_returns_required_mapping_keys_and_shapes(self) -> None:
        spec = resolve_topology_spec(
            topology_id=TOPOLOGY_ID,
            topology_variant=VARIANT,
            topology_params={},
        )
        model = build_model_from_spec(spec)
        model.eval()

        with torch.no_grad():
            outputs = model(_batch(batch_size=3))

        self.assertEqual(set(outputs.keys()), REQUIRED_OUTPUT_KEYS)
        self.assertEqual(tuple(outputs["distance_m"].shape), (3,))
        self.assertEqual(tuple(outputs["yaw_sin_cos"].shape), (3, 2))
        self.assertEqual(tuple(outputs["defender_center_3d"].shape), (3, 3))
        self.assertEqual(tuple(outputs["defender_keypoints_3d_flat"].shape), (3, 30))
        self.assertEqual(tuple(outputs["defender_keypoints_visible_logits"].shape), (3, 10))

    def test_missing_inputs_raise_clear_errors(self) -> None:
        model = build_model_from_spec(
            resolve_topology_spec(
                topology_id=TOPOLOGY_ID,
                topology_variant=VARIANT,
                topology_params={},
            )
        )
        batch = _batch()
        missing_orientation = dict(batch)
        missing_orientation.pop(TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY)
        with self.assertRaisesRegex(KeyError, TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY):
            model(missing_orientation)

        missing_geometry = dict(batch)
        missing_geometry.pop(TRI_STREAM_GEOMETRY_ARRAY_KEY)
        with self.assertRaisesRegex(KeyError, TRI_STREAM_GEOMETRY_ARRAY_KEY):
            model(missing_geometry)

        wrong_geometry = dict(batch)
        wrong_geometry[TRI_STREAM_GEOMETRY_ARRAY_KEY] = torch.rand(2, 9)
        with self.assertRaisesRegex(ValueError, "x_geometry width mismatch"):
            model(wrong_geometry)

    def test_task_contract_extracts_mapping_heads(self) -> None:
        spec = resolve_topology_spec(
            topology_id=TOPOLOGY_ID,
            topology_variant=VARIANT,
            topology_params={},
        )
        outputs = build_model_from_spec(spec)(_batch(batch_size=2))
        heads = extract_prediction_heads(outputs, spec.task_contract)

        self.assertEqual(
            set(heads),
            {
                "distance",
                "orientation",
                "defender_center_3d",
                "defender_keypoints_3d",
                "defender_keypoints_visible",
            },
        )
        self.assertEqual(tuple(heads["defender_keypoints_3d"].shape), (2, 30))
        self.assertEqual(tuple(heads["defender_keypoints_visible"].shape), (2, 10))

    def test_keypoint_loss_is_mean_normalized_and_visibility_uses_bce(self) -> None:
        spec = resolve_topology_spec(
            topology_id=TOPOLOGY_ID,
            topology_variant=VARIANT,
            topology_params={},
        )
        prediction_heads = {
            "distance": torch.zeros(2, 1),
            "orientation": torch.zeros(2, 2),
            "defender_center_3d": torch.zeros(2, 3),
            "defender_keypoints_3d": torch.zeros(2, 30),
            "defender_keypoints_visible": torch.zeros(2, 10),
        }
        target_heads = {
            "distance": torch.zeros(2, 1),
            "orientation": torch.zeros(2, 2),
            "defender_center_3d": torch.zeros(2, 3),
            "defender_keypoints_3d": torch.ones(2, 30),
            "defender_keypoints_visible": torch.ones(2, 10),
        }

        result = compute_task_loss(
            prediction_heads,
            target_heads,
            spec.task_contract,
            huber_delta=1.0,
            loss_weights={"keypoint_3d": 2.0, "keypoint_visibility": 3.0},
        )

        self.assertAlmostEqual(float(result.components["keypoint_3d_loss"]), 0.5, places=6)
        self.assertAlmostEqual(float(result.components["raw_keypoint_3d_loss"]), 0.5, places=6)
        self.assertAlmostEqual(float(result.components["weighted_keypoint_3d_loss"]), 1.0, places=6)
        self.assertAlmostEqual(
            float(result.components["keypoint_visibility_loss"]),
            math.log(2.0),
            places=6,
        )
        self.assertGreater(float(result.components["keypoint_visibility_loss"]), 0.5)
        self.assertIn("weighted_keypoint_visibility_loss", result.components)

    def test_center_keypoint_and_visibility_metrics(self) -> None:
        spec = resolve_topology_spec(
            topology_id=TOPOLOGY_ID,
            topology_variant=VARIANT,
            topology_params={},
        )
        pred_keypoints = np.zeros((1, 30), dtype=np.float32)
        pred_keypoints[0, 0] = 1.0
        for idx in range(1, 10):
            pred_keypoints[0, idx * 3] = 2.0
        visibility = np.zeros((1, 10), dtype=np.float32)
        visibility[0, 0] = 1.0

        metrics = summarize_task_metrics(
            prediction_heads={
                "distance": np.asarray([[0.0]], dtype=np.float32),
                "orientation": np.asarray([[0.0, 1.0]], dtype=np.float32),
                "defender_center_3d": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                "defender_keypoints_3d": pred_keypoints,
                "defender_keypoints_visible": np.asarray([[1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]], dtype=np.float32),
            },
            target_heads={
                "distance": np.asarray([[0.0]], dtype=np.float32),
                "orientation": np.asarray([[0.0, 1.0]], dtype=np.float32),
                "defender_center_3d": np.zeros((1, 3), dtype=np.float32),
                "defender_keypoints_3d": np.zeros((1, 30), dtype=np.float32),
                "defender_keypoints_visible": visibility,
            },
            task_contract=spec.task_contract,
            tolerance_values=(0.1,),
            primary_tolerance=0.1,
            collect_predictions=False,
        )

        center = metrics.task_metrics["defender_center_3d"]
        keypoints = metrics.task_metrics["defender_keypoints_3d"]
        visible = metrics.task_metrics["defender_keypoints_visible"]
        self.assertAlmostEqual(center["center_mean_error_m"], 1.0, places=6)
        self.assertAlmostEqual(keypoints["visible_keypoint_mean_error_m"], 1.0, places=6)
        self.assertAlmostEqual(keypoints["hidden_keypoint_mean_error_m"], 2.0, places=6)
        self.assertAlmostEqual(keypoints["keypoint_mean_point_error_m"], 1.9, places=6)
        self.assertAlmostEqual(visible["keypoint_visibility_accuracy"], 1.0, places=6)
        self.assertAlmostEqual(visible["keypoint_visibility_precision"], 1.0, places=6)
        self.assertAlmostEqual(visible["keypoint_visibility_recall"], 1.0, places=6)
        self.assertAlmostEqual(visible["keypoint_visibility_f1"], 1.0, places=6)

    def test_missing_keypoint_target_arrays_fail_clearly(self) -> None:
        spec = resolve_topology_spec(
            topology_id=TOPOLOGY_ID,
            topology_variant=VARIANT,
            topology_params={},
        )
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "training-data"
            self._write_fixture_corpus(
                data_root / "fixture_defender",
                spec,
                include_schema_metadata=True,
                include_keypoint_array=False,
            )
            metadata_df, _ = load_root_metadata(data_root, source_root="training", repo_root=Path(tmpdir))
            schema_df = validate_root_schema(
                metadata_df,
                root_name="training",
                image_array_key=TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
            )
            with self.assertRaisesRegex(ValueError, "y_defender_keypoints_3d_flat"):
                validate_task_contract_schema(metadata_df, schema_df, spec.task_contract, root_name="training")

    def test_missing_schema_metadata_fails_clearly(self) -> None:
        spec = resolve_topology_spec(
            topology_id=TOPOLOGY_ID,
            topology_variant=VARIANT,
            topology_params={},
        )
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "training-data"
            self._write_fixture_corpus(
                data_root / "fixture_defender",
                spec,
                include_schema_metadata=False,
                include_keypoint_array=True,
            )
            metadata_df, _ = load_root_metadata(data_root, source_root="training", repo_root=Path(tmpdir))
            schema_df = validate_root_schema(
                metadata_df,
                root_name="training",
                image_array_key=TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
            )
            with self.assertRaisesRegex(ValueError, "defender_keypoint_schema_hash"):
                validate_task_contract_schema(metadata_df, schema_df, spec.task_contract, root_name="training")

    def test_unresolved_schema_blocks_training_validation(self) -> None:
        spec = resolve_topology_spec(
            topology_id=TOPOLOGY_ID,
            topology_variant=VARIANT,
            topology_params={},
        )
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "training-data"
            self._write_fixture_corpus(
                data_root / "fixture_defender",
                spec,
                include_schema_metadata=True,
                include_keypoint_array=True,
            )
            metadata_df, _ = load_root_metadata(data_root, source_root="training", repo_root=Path(tmpdir))
            schema_df = validate_root_schema(
                metadata_df,
                root_name="training",
                image_array_key=TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
            )
            with self.assertRaisesRegex(ValueError, "schema requirement 'defender_keypoint_schema' is blocked"):
                validate_task_contract_schema(metadata_df, schema_df, spec.task_contract, root_name="training")

    def test_geometry_only_ablation_accepts_x_geometry_only(self) -> None:
        spec = resolve_topology_spec(
            topology_id=TOPOLOGY_ID,
            topology_variant=VARIANT,
            topology_params={"ablation_mode": "geometry_only"},
        )
        model = build_model_from_spec(spec)

        self.assertEqual(spec.task_contract["input_mode"], "geometry_only")
        outputs = model({TRI_STREAM_GEOMETRY_ARRAY_KEY: torch.rand(2, 10)})

        self.assertEqual(set(outputs.keys()), REQUIRED_OUTPUT_KEYS)
        self.assertEqual(tuple(outputs["defender_keypoints_3d_flat"].shape), (2, 30))

    def _write_fixture_corpus(
        self,
        corpus_dir: Path,
        spec: object,
        *,
        include_schema_metadata: bool,
        include_keypoint_array: bool,
    ) -> None:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = corpus_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        n = 2
        distance_image = np.ones((n, 1, 16, 16), dtype=np.float32)
        orientation_image = np.ones((n, 1, 16, 16), dtype=np.float32)
        bbox_features = np.asarray(
            [
                [8.0, 8.0, 6.0, 8.0, 0.5, 0.5, 0.375, 0.5, 0.75, 0.1875],
                [8.0, 8.0, 6.0, 8.0, 0.5, 0.5, 0.375, 0.5, 0.75, 0.1875],
            ],
            dtype=np.float32,
        )
        yaw_deg = np.asarray([0.0, 20.0], dtype=np.float32)
        center = np.asarray([[0.0, 0.5, 1.0], [0.1, 0.6, 1.1]], dtype=np.float32)
        keypoints = np.arange(n * 30, dtype=np.float32).reshape(n, 30) / 100.0
        visibility = np.asarray(
            [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
            dtype=np.float32,
        )
        npz_name = f"{corpus_dir.name}_shard_00000.npz"
        payload: dict[str, object] = {
            TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY: distance_image,
            TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY: orientation_image,
            TRI_STREAM_GEOMETRY_ARRAY_KEY: bbox_features,
            "x_geometry_schema": np.asarray(BBOX_FEATURE_SCHEMA, dtype=str),
            "y_distance_m": np.asarray([1.0, 2.0], dtype=np.float32),
            "y_yaw_deg": yaw_deg,
            "y_yaw_sin": np.sin(np.deg2rad(yaw_deg)).astype(np.float32),
            "y_yaw_cos": np.cos(np.deg2rad(yaw_deg)).astype(np.float32),
            "y_defender_center_3d": center,
            "y_defender_keypoints_visible": visibility,
            "sample_id": np.asarray(["sample_0", "sample_1"]),
            "image_filename": np.asarray(["frame_0.png", "frame_1.png"]),
            "npz_row_index": np.arange(n, dtype=np.int64),
        }
        if include_keypoint_array:
            payload["y_defender_keypoints_3d_flat"] = keypoints
        if include_schema_metadata:
            requirement = spec.task_contract["schema_requirements"]["defender_keypoint_schema"]
            for key, value in requirement["required_npz_metadata"].items():
                payload[str(key)] = np.asarray(value)
        np.savez(corpus_dir / npz_name, **payload)

        rows: list[dict[str, object]] = []
        target_columns = tuple(spec.task_contract["target_columns"])
        keypoint_columns = list(spec.task_contract["heads"]["defender_keypoints_3d"]["target_columns"])
        visibility_columns = list(spec.task_contract["heads"]["defender_keypoints_visible"]["target_columns"])
        for idx in range(n):
            row: dict[str, object] = {
                "run_id": corpus_dir.name,
                "sample_id": f"sample_{idx}",
                "frame_index": idx,
                "image_filename": f"frame_{idx}.png",
                "distance_m": float(idx + 1),
                "npz_filename": npz_name,
                "npz_row_index": idx,
                "yaw_deg": float(yaw_deg[idx]),
                "yaw_sin": float(np.sin(np.deg2rad(float(yaw_deg[idx])))),
                "yaw_cos": float(np.cos(np.deg2rad(float(yaw_deg[idx])))),
                "defender_center_x_m": float(center[idx, 0]),
                "defender_center_y_m": float(center[idx, 1]),
                "defender_center_z_m": float(center[idx, 2]),
            }
            row.update(
                {
                    column: float(bbox_features[idx, col_idx])
                    for col_idx, column in enumerate(BBOX_FEATURE_COLUMNS)
                }
            )
            row.update(
                {column: float(keypoints[idx, col_idx]) for col_idx, column in enumerate(keypoint_columns)}
            )
            row.update(
                {column: float(visibility[idx, col_idx]) for col_idx, column in enumerate(visibility_columns)}
            )
            for column in target_columns:
                self.assertIn(column, row)
            rows.append(row)
        pd.DataFrame(rows).to_csv(manifest_dir / "samples.csv", index=False)

        (manifest_dir / "run.json").write_text(
            json.dumps(
                {
                    "RunId": corpus_dir.name,
                    "PreprocessingContract": {
                        "ContractVersion": "rb-preprocess-v4-tri-stream-orientation-v1",
                        "CurrentStage": "pack_tri_stream",
                        "CompletedStages": ["detect", "silhouette", "pack_tri_stream"],
                        "CurrentRepresentation": {
                            "Kind": "tri_stream_npz",
                            "StorageFormat": "npz",
                            "ArrayKeys": sorted(payload),
                        },
                    },
                },
                indent=4,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
