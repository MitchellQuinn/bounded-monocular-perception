"""Integration tests for pass-1 v2 silhouette -> npy/pack pipeline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

from rb_pipeline_v2.config import NpyPackStageConfigV2, SilhouetteStageConfigV2
from rb_pipeline_v2.manifest import PREPROCESSING_CONTRACT_VERSION_V2, load_run_json, load_samples_csv, samples_csv_path
from rb_pipeline_v2.npy_pack_stage import run_npy_pack_stage_v2
from rb_pipeline_v2.silhouette_stage import run_silhouette_stage_v2


class V2PipelineIntegrationTests(unittest.TestCase):
    def test_npy_pack_config_rejects_unknown_training_source_column(self) -> None:
        config = NpyPackStageConfigV2(
            representation_mode="filled",
            array_exporter_id="array.grayscale_v1",
            training_image_source_column="unknown_column",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported training_image_source_column"):
            config.normalized_training_image_source_column()

    def test_v2_pipeline_subset_then_pack_one_mode(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_integration"

            self._make_project_fixture(
                project_root,
                run_name,
                [self._rectangle_image(), self._circle_image(), self._rectangle_image()],
            )

            silhouette_config = SilhouetteStageConfigV2(
                representation_mode="filled",
                generator_id="silhouette.contour_v2",
                fallback_id="fallback.convex_hull_v1",
                sample_offset=0,
                sample_limit=1,
                persist_edge_debug=True,
            )

            silhouette_summary = run_silhouette_stage_v2(project_root, run_name, silhouette_config)
            self.assertEqual(silhouette_summary.successful_rows, 1)
            self.assertEqual(silhouette_summary.skipped_rows, 2)

            npy_pack_config = NpyPackStageConfigV2(
                representation_mode="filled",
                array_exporter_id="array.grayscale_v1",
                npy_output_dtype="float32",
                pack_output_dtype="preserve",
                shard_size=1,
                training_image_source_column="silhouette_debug_selected_component_filename",
            )

            npy_pack_summary = run_npy_pack_stage_v2(project_root, run_name, npy_pack_config)
            self.assertEqual(npy_pack_summary.successful_rows, 1)
            self.assertEqual(npy_pack_summary.skipped_rows, 2)

            training_run_root = project_root / "training-data-v2" / run_name
            run_payload = load_run_json(training_run_root / "manifests")
            contract = run_payload["PreprocessingContract"]

            self.assertEqual(contract["ContractVersion"], PREPROCESSING_CONTRACT_VERSION_V2)
            self.assertEqual(contract["CurrentStage"], "pack")
            self.assertEqual(contract["CompletedStages"], ["silhouette", "npy", "pack"])
            self.assertEqual(contract["CurrentRepresentation"]["RepresentationMode"], "filled")

            samples = load_samples_csv(samples_csv_path(training_run_root / "manifests"))
            self.assertIn("silhouette_fallback_used", samples.columns)
            self.assertIn("silhouette_area_px", samples.columns)

            success_rows = samples[samples["pack_stage_status"] == "success"]
            self.assertEqual(len(success_rows), 1)
            self.assertEqual(
                str(success_rows.iloc[0]["npy_source_image_column"]),
                "silhouette_debug_selected_component_filename",
            )
            self.assertTrue(str(success_rows.iloc[0]["npy_source_image_filename"]).strip())

            npz_paths = sorted(training_run_root.glob("*.npz"))
            self.assertEqual(len(npz_paths), 1)

            with np.load(npz_paths[0], allow_pickle=False) as payload:
                self.assertIn("X", payload)
                self.assertIn("y", payload)
                self.assertEqual(payload["X"].shape[0], 1)
                self.assertEqual(payload["X"].ndim, 3)
                self.assertEqual(str(payload["X"].dtype), "float32")

    def test_npy_pack_from_amalgamated_debug_source(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_amalgamated_source"

            self._make_project_fixture(
                project_root,
                run_name,
                [self._rectangle_image()],
            )

            silhouette_config = SilhouetteStageConfigV2(
                representation_mode="filled",
                generator_id="silhouette.contour_v2",
                fallback_id="fallback.convex_hull_v1",
                sample_limit=1,
                persist_edge_debug=True,
                amalgamate_debug_outputs=True,
                keep_individual_debug_outputs=False,
            )
            silhouette_summary = run_silhouette_stage_v2(project_root, run_name, silhouette_config)
            self.assertEqual(silhouette_summary.successful_rows, 1)

            silhouette_samples = load_samples_csv(
                samples_csv_path(project_root / "silhouette-images-v2" / run_name / "manifests")
            )
            self.assertTrue(str(silhouette_samples.at[0, "silhouette_debug_amalgamated_filename"]).strip())
            raw_edge_value = silhouette_samples.at[0, "silhouette_debug_raw_edge_filename"]
            self.assertTrue(pd.isna(raw_edge_value) or str(raw_edge_value).strip() == "")

            npy_pack_config = NpyPackStageConfigV2(
                representation_mode="filled",
                array_exporter_id="array.grayscale_v1",
                npy_output_dtype="float32",
                pack_output_dtype="preserve",
                shard_size=1,
                training_image_source_column="silhouette_debug_amalgamated_filename",
            )
            npy_pack_summary = run_npy_pack_stage_v2(project_root, run_name, npy_pack_config)
            self.assertEqual(npy_pack_summary.successful_rows, 1)
            self.assertEqual(npy_pack_summary.failed_rows, 0)

            training_run_root = project_root / "training-data-v2" / run_name
            npz_paths = sorted(training_run_root.glob("*.npz"))
            self.assertEqual(len(npz_paths), 1)

            with np.load(npz_paths[0], allow_pickle=False) as payload:
                self.assertIn("X", payload)
                self.assertEqual(payload["X"].shape[0], 1)
                # amalgamated debug overlay matches source frame geometry
                self.assertEqual(payload["X"].shape[1:], (64, 64))

    def _make_project_fixture(
        self,
        project_root: Path,
        run_name: str,
        images: list[np.ndarray],
    ) -> None:
        (project_root / "notebooks").mkdir(parents=True, exist_ok=True)

        run_root = project_root / "input-images" / run_name
        images_dir = run_root / "images"
        manifests_dir = run_root / "manifests"
        images_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []

        for idx, image in enumerate(images):
            filename = f"frame_{idx:03d}.png"
            image_path = images_dir / filename
            ok = cv2.imwrite(str(image_path), image)
            if not ok:
                raise RuntimeError(f"Failed to write fixture image: {image_path}")

            rows.append(
                {
                    "run_id": run_name,
                    "sample_id": f"{run_name}_sample_{idx:03d}",
                    "frame_index": idx,
                    "image_filename": filename,
                    "distance_m": float(idx + 1),
                    "image_width_px": int(image.shape[1]),
                    "image_height_px": int(image.shape[0]),
                    "capture_success": True,
                }
            )

        pd.DataFrame(rows).to_csv(manifests_dir / "samples.csv", index=False)
        (manifests_dir / "run.json").write_text(
            json.dumps({"RunId": run_name}, indent=4) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _rectangle_image() -> np.ndarray:
        image = np.full((64, 64), 255, dtype=np.uint8)
        cv2.rectangle(image, (14, 14), (50, 50), color=0, thickness=2)
        return image

    @staticmethod
    def _circle_image() -> np.ndarray:
        image = np.full((64, 64), 255, dtype=np.uint8)
        cv2.circle(image, (32, 32), 14, color=0, thickness=2)
        return image


if __name__ == "__main__":
    unittest.main()
