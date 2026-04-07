"""Integration tests for v3 threshold -> npy/pack pipeline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

from rb_pipeline_v3.config import NpyPackStageConfigV3, ThresholdStageConfigV3
from rb_pipeline_v3.manifest import PREPROCESSING_CONTRACT_VERSION_V3, load_run_json, load_samples_csv, samples_csv_path
from rb_pipeline_v3.npy_pack_stage import run_npy_pack_stage_v3
from rb_pipeline_v3.threshold_stage import run_threshold_stage_v3


class V3PipelineIntegrationTests(unittest.TestCase):
    def test_npy_pack_config_rejects_unknown_training_source_column(self) -> None:
        config = NpyPackStageConfigV3(
            representation_mode="filled",
            training_image_source_column="unknown_column",
        )
        with self.assertRaisesRegex(ValueError, "Unsupported training_image_source_column"):
            config.normalized_training_image_source_column()

    def test_v3_pipeline_threshold_then_pack(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_threshold"

            self._make_project_fixture(
                project_root,
                run_name,
                [self._scene_image(), self._scene_image()],
            )

            threshold_config = ThresholdStageConfigV3(
                representation_mode="filled",
                threshold_low_value=120,
                threshold_high_value=255,
                sample_offset=0,
                sample_limit=1,
                persist_debug=True,
            )

            threshold_summary = run_threshold_stage_v3(project_root, run_name, threshold_config)
            self.assertEqual(threshold_summary.successful_rows, 1)
            self.assertEqual(threshold_summary.skipped_rows, 1)

            threshold_samples = load_samples_csv(
                samples_csv_path(project_root / "threshold-images-v3" / run_name / "manifests")
            )
            threshold_filename = str(threshold_samples.at[0, "threshold_image_filename"])
            threshold_image = cv2.imread(
                str(project_root / "threshold-images-v3" / run_name / "images" / threshold_filename),
                cv2.IMREAD_GRAYSCALE,
            )
            self.assertIsNotNone(threshold_image)
            # Foreground should be white on black in v3 outputs.
            self.assertEqual(int(threshold_image[32, 32]), 255)
            self.assertEqual(int(threshold_image[0, 0]), 0)

            npy_pack_config = NpyPackStageConfigV3(
                representation_mode="filled",
                npy_output_dtype="float16",
                pack_output_dtype="uint8",
                shard_size=1,
                training_image_source_column="threshold_debug_selected_component_filename",
                delete_source_npy_after_pack=False,
            )

            npy_pack_summary = run_npy_pack_stage_v3(project_root, run_name, npy_pack_config)
            self.assertEqual(npy_pack_summary.successful_rows, 1)
            self.assertEqual(npy_pack_summary.skipped_rows, 1)

            training_run_root = project_root / "training-data-v3" / run_name
            run_payload = load_run_json(training_run_root / "manifests")
            contract = run_payload["PreprocessingContract"]

            self.assertEqual(contract["ContractVersion"], PREPROCESSING_CONTRACT_VERSION_V3)
            self.assertEqual(contract["CurrentStage"], "pack")
            self.assertEqual(contract["CompletedStages"], ["threshold", "npy", "pack"])
            self.assertEqual(contract["CurrentRepresentation"]["RepresentationMode"], "filled")

            samples = load_samples_csv(samples_csv_path(training_run_root / "manifests"))
            self.assertIn("threshold_area_px", samples.columns)
            self.assertIn("threshold_debug_selected_component_filename", samples.columns)

            success_rows = samples[samples["pack_stage_status"] == "success"]
            self.assertEqual(len(success_rows), 1)
            self.assertEqual(
                str(success_rows.iloc[0]["npy_source_image_column"]),
                "threshold_debug_selected_component_filename",
            )
            self.assertTrue(str(success_rows.iloc[0]["npy_source_image_filename"]).strip())

            npz_paths = sorted(training_run_root.glob("*.npz"))
            self.assertEqual(len(npz_paths), 1)

            with np.load(npz_paths[0], allow_pickle=False) as payload:
                self.assertIn("X", payload)
                self.assertIn("y", payload)
                self.assertEqual(payload["X"].shape[0], 1)
                self.assertEqual(payload["X"].ndim, 3)
                self.assertEqual(str(payload["X"].dtype), "uint8")

            npy_paths = sorted((training_run_root / "arrays").glob("*.npy"))
            self.assertEqual(len(npy_paths), 1)
            npy_array = np.load(npy_paths[0], allow_pickle=False)
            self.assertEqual(str(npy_array.dtype), "float16")
            self.assertGreaterEqual(float(np.min(npy_array)), 0.0)
            self.assertLessEqual(float(np.max(npy_array)), 255.0)
            self.assertEqual(float(npy_array[32, 32]), 255.0)
            self.assertEqual(float(npy_array[0, 0]), 0.0)

    def test_threshold_bounds_are_sorted_when_low_gt_high(self) -> None:
        config = ThresholdStageConfigV3(
            representation_mode="filled",
            threshold_low_value=230,
            threshold_high_value=40,
        )
        self.assertEqual(config.normalized_threshold_bounds(), (40, 230))

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
    def _scene_image() -> np.ndarray:
        image = np.full((64, 64), 20, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (42, 44), color=230, thickness=-1)
        # Deliberate dark pinholes inside the bright foreground target.
        image[32, 32] = 20
        image[35, 30] = 20
        cv2.rectangle(image, (6, 6), (10, 10), color=245, thickness=-1)
        return image


if __name__ == "__main__":
    unittest.main()
