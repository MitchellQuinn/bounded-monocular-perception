"""Unit tests for brightness-sensitivity diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import inference_v0_1
import inference_v0_1.brightness_analysis as brightness_analysis
from inference_v0_1.brightness_analysis import (
    _summarize_aggregate,
    _summarize_brightness_correlations,
    _summarize_per_sample,
    apply_vehicle_darkness_gain,
)
from inference_v0_1.external import ensure_external_paths, preprocessing_root
from src.task_runtime import batch_to_model_inputs


class BrightnessAnalysisTests(unittest.TestCase):
    def test_package_exports_brightness_analysis_helpers(self) -> None:
        self.assertTrue(hasattr(inference_v0_1, "run_brightness_sensitivity_analysis"))
        self.assertTrue(hasattr(inference_v0_1, "apply_vehicle_darkness_gain"))
        self.assertTrue(hasattr(inference_v0_1, "BrightnessSensitivityResult"))

    def test_external_preprocessing_root_exposes_rb_pipeline_v4(self) -> None:
        root = preprocessing_root()

        self.assertEqual(root.name, "02_synthetic-data-processing-v4.0")
        self.assertTrue((root / "rb_pipeline_v4" / "__init__.py").is_file())

        ensure_external_paths()
        self.assertIn(str(root.resolve()), sys.path)

        import rb_pipeline_v4

        self.assertEqual(Path(rb_pipeline_v4.__file__).resolve().parents[1], root.resolve())

    def test_apply_vehicle_darkness_gain_preserves_background_and_shape(self) -> None:
        image = np.array([[[1.0, 0.75], [0.5, 1.0]]], dtype=np.float32)

        adjusted = apply_vehicle_darkness_gain(image, 1.5)

        self.assertEqual(adjusted.shape, image.shape)
        np.testing.assert_allclose(
            adjusted,
            np.array([[[1.0, 0.625], [0.25, 1.0]]], dtype=np.float32),
            atol=1e-6,
        )

    def test_variant_batch_preserves_tri_stream_side_inputs(self) -> None:
        samples = [
            SimpleNamespace(
                sample_row={"sample_id": f"sample-{idx}"},
                input_mode=brightness_analysis.TRI_STREAM_INPUT_MODE,
                orientation_image=np.full((1, 2, 2), 0.8 + idx, dtype=np.float32),
                bbox_features=np.arange(10, dtype=np.float32) + idx,
                actual_distance_m=float(idx + 1),
                actual_yaw_sin=0.0,
                actual_yaw_cos=1.0,
            )
            for idx in range(2)
        ]
        variant_images = [
            np.full((1, 2, 2), 0.4 + idx, dtype=np.float32)
            for idx in range(2)
        ]

        batch = brightness_analysis._build_variant_batch(
            preprocessed_samples=samples,
            variant_images=variant_images,
        )
        model_inputs = batch_to_model_inputs(
            batch,
            {"input_mode": brightness_analysis.TRI_STREAM_INPUT_MODE},
            device=torch.device("cpu"),
        )

        self.assertIsNone(batch.bbox_features)
        self.assertEqual(batch.images.shape, (2, 1, 2, 2))
        self.assertIsNotNone(batch.geometry)
        self.assertEqual(batch.geometry.shape, (2, 10))
        self.assertIsNotNone(batch.extra_inputs)
        self.assertEqual(batch.extra_inputs["x_orientation_image"].shape, (2, 1, 2, 2))
        self.assertEqual(set(model_inputs), {"x_distance_image", "x_orientation_image", "x_geometry"})

    def test_summary_tables_capture_prediction_drift(self) -> None:
        predictions_df = pd.DataFrame(
            [
                {
                    "analysis_sample_index": 0,
                    "sample_id": "sample-001",
                    "image_filename": "a.png",
                    "darkness_gain": 1.0,
                    "truth_distance_m": 4.0,
                    "truth_orientation_deg": 90.0,
                    "truth_yaw_sin": 1.0,
                    "truth_yaw_cos": 0.0,
                    "prediction_distance_m": 4.2,
                    "prediction_orientation_deg": 92.0,
                    "distance_error_m": 0.2,
                    "abs_distance_error_m": 0.2,
                    "orientation_error_deg": 2.0,
                    "abs_orientation_error_deg": 2.0,
                    "baseline_prediction_distance_m": 4.2,
                    "baseline_prediction_orientation_deg": 92.0,
                    "distance_shift_from_baseline_m": 0.0,
                    "abs_distance_shift_from_baseline_m": 0.0,
                    "orientation_shift_from_baseline_deg": 0.0,
                    "abs_orientation_shift_from_baseline_deg": 0.0,
                    "variant_canvas_mean_intensity": 0.90,
                    "variant_canvas_std_intensity": 0.05,
                    "variant_vehicle_pixel_fraction": 0.10,
                    "variant_vehicle_mean_intensity": 0.70,
                    "variant_vehicle_std_intensity": 0.10,
                    "variant_vehicle_mean_darkness": 0.30,
                },
                {
                    "analysis_sample_index": 0,
                    "sample_id": "sample-001",
                    "image_filename": "a.png",
                    "darkness_gain": 1.4,
                    "truth_distance_m": 4.0,
                    "truth_orientation_deg": 90.0,
                    "truth_yaw_sin": 1.0,
                    "truth_yaw_cos": 0.0,
                    "prediction_distance_m": 4.5,
                    "prediction_orientation_deg": 97.0,
                    "distance_error_m": 0.5,
                    "abs_distance_error_m": 0.5,
                    "orientation_error_deg": 7.0,
                    "abs_orientation_error_deg": 7.0,
                    "baseline_prediction_distance_m": 4.2,
                    "baseline_prediction_orientation_deg": 92.0,
                    "distance_shift_from_baseline_m": 0.3,
                    "abs_distance_shift_from_baseline_m": 0.3,
                    "orientation_shift_from_baseline_deg": 5.0,
                    "abs_orientation_shift_from_baseline_deg": 5.0,
                    "variant_canvas_mean_intensity": 0.86,
                    "variant_canvas_std_intensity": 0.08,
                    "variant_vehicle_pixel_fraction": 0.10,
                    "variant_vehicle_mean_intensity": 0.58,
                    "variant_vehicle_std_intensity": 0.12,
                    "variant_vehicle_mean_darkness": 0.42,
                },
                {
                    "analysis_sample_index": 1,
                    "sample_id": "sample-002",
                    "image_filename": "b.png",
                    "darkness_gain": 1.0,
                    "truth_distance_m": 6.0,
                    "truth_orientation_deg": 180.0,
                    "truth_yaw_sin": 0.0,
                    "truth_yaw_cos": -1.0,
                    "prediction_distance_m": 5.8,
                    "prediction_orientation_deg": 178.0,
                    "distance_error_m": -0.2,
                    "abs_distance_error_m": 0.2,
                    "orientation_error_deg": -2.0,
                    "abs_orientation_error_deg": 2.0,
                    "baseline_prediction_distance_m": 5.8,
                    "baseline_prediction_orientation_deg": 178.0,
                    "distance_shift_from_baseline_m": 0.0,
                    "abs_distance_shift_from_baseline_m": 0.0,
                    "orientation_shift_from_baseline_deg": 0.0,
                    "abs_orientation_shift_from_baseline_deg": 0.0,
                    "variant_canvas_mean_intensity": 0.82,
                    "variant_canvas_std_intensity": 0.07,
                    "variant_vehicle_pixel_fraction": 0.18,
                    "variant_vehicle_mean_intensity": 0.45,
                    "variant_vehicle_std_intensity": 0.09,
                    "variant_vehicle_mean_darkness": 0.55,
                },
                {
                    "analysis_sample_index": 1,
                    "sample_id": "sample-002",
                    "image_filename": "b.png",
                    "darkness_gain": 1.4,
                    "truth_distance_m": 6.0,
                    "truth_orientation_deg": 180.0,
                    "truth_yaw_sin": 0.0,
                    "truth_yaw_cos": -1.0,
                    "prediction_distance_m": 5.6,
                    "prediction_orientation_deg": 170.0,
                    "distance_error_m": -0.4,
                    "abs_distance_error_m": 0.4,
                    "orientation_error_deg": -10.0,
                    "abs_orientation_error_deg": 10.0,
                    "baseline_prediction_distance_m": 5.8,
                    "baseline_prediction_orientation_deg": 178.0,
                    "distance_shift_from_baseline_m": -0.2,
                    "abs_distance_shift_from_baseline_m": 0.2,
                    "orientation_shift_from_baseline_deg": -8.0,
                    "abs_orientation_shift_from_baseline_deg": 8.0,
                    "variant_canvas_mean_intensity": 0.76,
                    "variant_canvas_std_intensity": 0.11,
                    "variant_vehicle_pixel_fraction": 0.18,
                    "variant_vehicle_mean_intensity": 0.28,
                    "variant_vehicle_std_intensity": 0.13,
                    "variant_vehicle_mean_darkness": 0.72,
                },
            ]
        )

        per_sample = _summarize_per_sample(predictions_df)
        aggregate = _summarize_aggregate(predictions_df)
        correlations = _summarize_brightness_correlations(per_sample)

        first_sample = per_sample.loc[per_sample["sample_id"] == "sample-001"].iloc[0]
        self.assertAlmostEqual(float(first_sample["distance_prediction_range_m"]), 0.3, places=6)
        self.assertAlmostEqual(float(first_sample["max_abs_distance_shift_m"]), 0.3, places=6)
        self.assertAlmostEqual(float(first_sample["max_abs_orientation_shift_deg"]), 5.0, places=6)

        gain_14 = aggregate.loc[np.isclose(aggregate["darkness_gain"], 1.4)].iloc[0]
        self.assertAlmostEqual(float(gain_14["mean_abs_distance_shift_m"]), 0.25, places=6)
        self.assertAlmostEqual(float(gain_14["mean_abs_orientation_shift_deg"]), 6.5, places=6)
        self.assertAlmostEqual(
            float(gain_14["delta_mean_abs_distance_error_m_vs_baseline"]),
            0.25,
            places=6,
        )

        matching = correlations.loc[
            (correlations["feature"] == "baseline_vehicle_mean_darkness")
            & (correlations["metric"] == "max_abs_distance_shift_m")
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(int(matching.iloc[0]["sample_count"]), 2)

    def test_run_brightness_sensitivity_analysis_streams_in_batches(self) -> None:
        selected_rows = pd.DataFrame(
            [
                {
                    "sample_id": "sample-001",
                    "image_filename": "a.png",
                    "distance_m": 4.0,
                    "yaw_deg": 90.0,
                },
                {
                    "sample_id": "sample-002",
                    "image_filename": "b.png",
                    "distance_m": 6.0,
                    "yaw_deg": 135.0,
                },
                {
                    "sample_id": "sample-003",
                    "image_filename": "c.png",
                    "distance_m": 8.0,
                    "yaw_deg": 180.0,
                },
            ]
        )
        observed_batch_sizes: list[int] = []
        observed_progress_updates: list[tuple[int, int]] = []

        def fake_preprocess_single_sample(*, sample_row, **_kwargs):
            sample_index = int(str(sample_row["sample_id"]).split("-")[-1])
            return SimpleNamespace(
                sample_row={
                    "sample_id": str(sample_row["sample_id"]),
                    "image_filename": str(sample_row["image_filename"]),
                    "yaw_deg": float(sample_row["yaw_deg"]),
                },
                model_image=np.array([[[0.40 + (0.05 * sample_index)]]], dtype=np.float32),
                bbox_features=np.array([float(sample_index), 1.0], dtype=np.float32),
                actual_distance_m=float(sample_row["distance_m"]),
                actual_yaw_sin=0.0,
                actual_yaw_cos=1.0,
            )

        def fake_run_prediction_batch(*, batch, **_kwargs):
            observed_batch_sizes.append(int(batch.images.shape[0]))
            records: list[dict[str, float]] = []
            for row, image, target in zip(batch.rows, batch.images, batch.targets, strict=True):
                mean_intensity = float(np.mean(image))
                truth_distance = float(target[0])
                truth_yaw_deg = float(row["yaw_deg"])
                records.append(
                    {
                        "truth_distance_m": truth_distance,
                        "prediction_distance_m": truth_distance + (1.0 - mean_intensity),
                        "truth_yaw_deg": truth_yaw_deg,
                        "prediction_yaw_deg": truth_yaw_deg + (10.0 * (1.0 - mean_intensity)),
                    }
                )
            return pd.DataFrame(records)

        with (
            patch.object(
                brightness_analysis,
                "_resolve_analysis_corpus",
                return_value=SimpleNamespace(name="demo-corpus"),
            ),
            patch.object(
                brightness_analysis,
                "_select_analysis_rows",
                return_value=selected_rows,
            ),
            patch.object(
                brightness_analysis,
                "load_model_context",
                return_value=(object(), SimpleNamespace(label="distance-model", device="cuda")),
            ),
            patch.object(
                brightness_analysis,
                "load_roi_fcn_model_context",
                return_value=(object(), SimpleNamespace(label="roi-model")),
            ),
            patch.object(
                brightness_analysis,
                "preprocess_single_sample",
                side_effect=fake_preprocess_single_sample,
            ),
            patch.object(
                brightness_analysis,
                "_run_prediction_batch",
                side_effect=fake_run_prediction_batch,
            ),
        ):
            result = brightness_analysis.run_brightness_sensitivity_analysis(
                "distance-run",
                "corpus-root",
                roi_model_run_dir="roi-run",
                darkness_gains=(1.0, 1.2),
                batch_size=2,
                progress_callback=lambda processed, total: observed_progress_updates.append(
                    (processed, total)
                ),
            )

        self.assertEqual(observed_batch_sizes, [2, 2, 1, 1])
        self.assertEqual(observed_progress_updates, [(2, 3), (3, 3)])
        self.assertEqual(result.darkness_gains, (1.0, 1.2))
        self.assertEqual(len(result.predictions), 6)
        self.assertSetEqual(
            set(result.predictions["analysis_sample_index"].tolist()),
            {0, 1, 2},
        )
        self.assertTrue((result.aggregate_summary["sample_count"] == 3).all())


if __name__ == "__main__":
    unittest.main()
