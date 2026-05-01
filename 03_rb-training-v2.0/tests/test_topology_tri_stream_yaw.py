"""Tests for the tri-stream distance + yaw topology and loader contract."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from src.data import (
    BBOX_FEATURE_COLUMNS,
    BBOX_FEATURE_SCHEMA,
    TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
    TRI_STREAM_GEOMETRY_ARRAY_KEY,
    TRI_STREAM_INPUT_MODE,
    TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,
    iter_batches,
    load_root_metadata,
    validate_root_schema,
    validate_task_contract_schema,
)
from src.task_runtime import batch_to_model_inputs
from src.topologies import build_model_from_spec, resolve_topology_spec


class TriStreamYawTests(unittest.TestCase):
    def test_tri_stream_yaw_forward_matches_mapping_contract(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_tri_stream_yaw",
            topology_variant="tri_stream_yaw_v0_1",
            topology_params={},
        )
        model = build_model_from_spec(spec)

        outputs = model(
            {
                TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY: torch.rand(4, 1, 64, 64),
                TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY: torch.rand(4, 1, 64, 64),
                TRI_STREAM_GEOMETRY_ARRAY_KEY: torch.rand(4, 10),
            }
        )

        self.assertEqual(spec.task_contract["prediction_mode"], "distance_yaw_sincos")
        self.assertEqual(spec.task_contract["input_mode"], TRI_STREAM_INPUT_MODE)
        self.assertEqual(spec.topology_contract["reporting"]["family"], "distance_orientation_multitask")
        self.assertEqual(set(outputs.keys()), {"distance_m", "yaw_sin_cos"})
        self.assertEqual(tuple(outputs["distance_m"].shape), (4,))
        self.assertEqual(tuple(outputs["yaw_sin_cos"].shape), (4, 2))

    def test_missing_orientation_image_raises_clear_error(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_tri_stream_yaw",
            topology_variant="tri_stream_yaw_v0_1",
            topology_params={},
        )
        model = build_model_from_spec(spec)

        with self.assertRaisesRegex(KeyError, TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY):
            model(
                {
                    TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY: torch.rand(2, 1, 64, 64),
                    TRI_STREAM_GEOMETRY_ARRAY_KEY: torch.rand(2, 10),
                }
            )

    def test_loader_validates_and_batches_tri_stream_fixture(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_tri_stream_yaw",
            topology_variant="tri_stream_yaw_v0_1",
            topology_params={},
        )
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "training-data-v4-tri-stream"
            self._write_fixture_corpus(data_root / "fixture_tri", include_tri_stream=True)

            metadata_df, _ = load_root_metadata(
                data_root,
                source_root="training",
                repo_root=Path(tmpdir),
            )
            schema_df = validate_root_schema(metadata_df, root_name="training")
            validate_task_contract_schema(
                metadata_df,
                schema_df,
                spec.task_contract,
                root_name="training",
            )

            batch = next(
                iter_batches(
                    metadata_df,
                    batch_size=2,
                    target_hw=(16, 16),
                    target_columns=tuple(spec.task_contract["target_columns"]),
                    include_geometry=True,
                    extra_input_array_keys=(TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,),
                )
            )
            model_inputs = batch_to_model_inputs(
                batch,
                spec.task_contract,
                device=torch.device("cpu"),
            )

            self.assertEqual(batch.images.shape, (2, 1, 16, 16))
            self.assertIsNotNone(batch.extra_inputs)
            self.assertIsNotNone(batch.geometry)
            self.assertIsNone(batch.bbox_features)
            self.assertEqual(batch.extra_inputs[TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY].shape, (2, 1, 16, 16))
            self.assertEqual(batch.geometry.shape, (2, 10))
            self.assertEqual(
                set(model_inputs.keys()),
                {
                    TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
                    TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,
                    TRI_STREAM_GEOMETRY_ARRAY_KEY,
                },
            )
            self.assertEqual(tuple(model_inputs[TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY].shape), (2, 1, 16, 16))
            self.assertEqual(tuple(model_inputs[TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY].shape), (2, 1, 16, 16))
            self.assertEqual(tuple(model_inputs[TRI_STREAM_GEOMETRY_ARRAY_KEY].shape), (2, 10))

    def test_tri_stream_contract_rejects_dual_stream_fixture(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_tri_stream_yaw",
            topology_variant="tri_stream_yaw_v0_1",
            topology_params={},
        )
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "training-data-v4"
            self._write_fixture_corpus(data_root / "fixture_dual", include_tri_stream=False)

            metadata_df, _ = load_root_metadata(
                data_root,
                source_root="training",
                repo_root=Path(tmpdir),
            )
            schema_df = validate_root_schema(metadata_df, root_name="training")

            with self.assertRaisesRegex(ValueError, TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY):
                validate_task_contract_schema(
                    metadata_df,
                    schema_df,
                    spec.task_contract,
                    root_name="training",
                )

    def _write_fixture_corpus(self, corpus_dir: Path, *, include_tri_stream: bool) -> None:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = corpus_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        n = 2
        distance_image = np.ones((n, 1, 16, 16), dtype=np.float32)
        distance_image[:, :, 6:10, 7:9] = 0.35
        orientation_image = np.ones((n, 1, 16, 16), dtype=np.float32)
        orientation_image[:, :, 2:14, 3:13] = 0.35
        bbox_features = np.asarray(
            [
                [8.0, 8.0, 6.0, 8.0, 0.5, 0.5, 0.375, 0.5, 0.75, 0.1875],
                [8.0, 8.0, 6.0, 8.0, 0.5, 0.5, 0.375, 0.5, 0.75, 0.1875],
            ],
            dtype=np.float32,
        )
        yaw_deg = np.asarray([10.0, 25.0], dtype=np.float32)
        payload = {
            "y_distance_m": np.asarray([1.0, 2.0], dtype=np.float32),
            "y_yaw_deg": yaw_deg,
            "y_yaw_sin": np.sin(np.deg2rad(yaw_deg)).astype(np.float32),
            "y_yaw_cos": np.cos(np.deg2rad(yaw_deg)).astype(np.float32),
            "sample_id": np.asarray(["sample_0", "sample_1"]),
            "image_filename": np.asarray(["frame_0.png", "frame_1.png"]),
            "npz_row_index": np.arange(n, dtype=np.int64),
        }
        if include_tri_stream:
            payload.update(
                {
                    TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY: distance_image,
                    TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY: orientation_image,
                    TRI_STREAM_GEOMETRY_ARRAY_KEY: bbox_features,
                    "x_geometry_schema": np.asarray(BBOX_FEATURE_SCHEMA, dtype=str),
                }
            )
        else:
            payload.update(
                {
                    "silhouette_crop": distance_image,
                    "bbox_features": bbox_features,
                    "bbox_features_schema": np.asarray(BBOX_FEATURE_SCHEMA, dtype=str),
                    "y_position_3d": np.zeros((n, 3), dtype=np.float32),
                }
            )

        npz_name = f"{corpus_dir.name}_shard_00000.npz"
        np.savez(corpus_dir / npz_name, **payload)

        rows: list[dict[str, object]] = []
        for idx in range(n):
            row = {
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
            }
            row.update(
                {
                    column: float(bbox_features[idx, col_idx])
                    for col_idx, column in enumerate(BBOX_FEATURE_COLUMNS)
                }
            )
            rows.append(row)
        pd.DataFrame(rows).to_csv(manifest_dir / "samples.csv", index=False)

        contract_kind = "tri_stream_npz" if include_tri_stream else "dual_stream_npz"
        array_keys = ["silhouette_crop", "bbox_features", "y_distance_m", "y_yaw_sin", "y_yaw_cos"]
        if include_tri_stream:
            array_keys = [
                TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
                TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,
                TRI_STREAM_GEOMETRY_ARRAY_KEY,
                "y_distance_m",
                "y_yaw_sin",
                "y_yaw_cos",
                "y_yaw_deg",
            ]
        (manifest_dir / "run.json").write_text(
            json.dumps(
                {
                    "RunId": corpus_dir.name,
                    "PreprocessingContract": {
                        "ContractVersion": "rb-preprocess-v4-tri-stream-orientation-v1"
                        if include_tri_stream
                        else "rb-preprocess-v4-dual-stream-orientation-brightness-v1",
                        "CurrentStage": "pack_tri_stream" if include_tri_stream else "pack_dual_stream",
                        "CompletedStages": ["detect", "silhouette", "pack_tri_stream"]
                        if include_tri_stream
                        else ["detect", "silhouette", "pack_dual_stream"],
                        "CurrentRepresentation": {
                            "Kind": contract_kind,
                            "StorageFormat": "npz",
                            "ArrayKeys": array_keys,
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
