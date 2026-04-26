"""Tests for v0.3 inference brightness-normalization parity."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_v0_1.brightness_normalization import (  # noqa: E402
    BrightnessNormalizationConfigV3,
    apply_brightness_normalization_v3,
)
from inference_v0_1.external import ensure_external_paths  # noqa: E402
from inference_v0_1.pipeline import (  # noqa: E402
    ModelContext,
    RoiFcnModelContext,
    _resolve_brightness_normalization_runtime,
    build_inference_startup_report,
    format_inference_startup_log,
)

ensure_external_paths()
from rb_pipeline_v4.brightness_normalization import apply_brightness_normalization_v4  # noqa: E402
from rb_pipeline_v4.config import BrightnessNormalizationConfigV4  # noqa: E402


class BrightnessNormalizationParityTests(unittest.TestCase):
    def test_vendored_helper_matches_preprocessing_source(self) -> None:
        image = np.asarray(
            [
                [1.0, 0.82, 0.74, 1.0],
                [1.0, 0.50, 0.25, 1.0],
                [1.0, 0.70, 0.35, 1.0],
            ],
            dtype=np.float32,
        )
        mask = np.asarray(
            [
                [False, True, True, False],
                [False, True, True, False],
                [False, True, True, False],
            ],
            dtype=bool,
        )

        inference_result = apply_brightness_normalization_v3(
            image,
            mask,
            BrightnessNormalizationConfigV3(
                enabled=True,
                method="masked_median_darkness_gain",
                target_median_darkness=0.55,
                min_gain=0.5,
                max_gain=2.0,
                epsilon=1e-6,
                empty_mask_policy="skip",
            ),
        )
        preprocessing_result = apply_brightness_normalization_v4(
            image,
            mask,
            BrightnessNormalizationConfigV4(
                enabled=True,
                method="masked_median_darkness_gain",
                target_median_darkness=0.55,
                min_gain=0.5,
                max_gain=2.0,
                epsilon=1e-6,
                empty_mask_policy="skip",
            ),
        )

        np.testing.assert_allclose(inference_result.image, preprocessing_result.image, atol=0.0)
        self.assertEqual(inference_result.status, preprocessing_result.status)
        self.assertEqual(inference_result.method, preprocessing_result.method)
        self.assertEqual(
            inference_result.foreground_pixel_count,
            preprocessing_result.foreground_pixel_count,
        )
        self.assertEqual(inference_result.current_median_darkness, preprocessing_result.current_median_darkness)
        self.assertEqual(
            inference_result.effective_median_darkness,
            preprocessing_result.effective_median_darkness,
        )
        self.assertEqual(inference_result.gain, preprocessing_result.gain)

    def test_resolves_brightness_contract_from_distance_model_preprocessing_contract(self) -> None:
        brightness_contract = {
            "Enabled": True,
            "Method": "masked_median_darkness_gain",
            "TargetMedianDarkness": 0.55,
            "MinGain": 0.5,
            "MaxGain": 2.0,
            "Epsilon": 1e-6,
            "EmptyMaskPolicy": "skip",
        }
        preprocessing_contract = {
            "Stages": {"pack_dual_stream": {"BrightnessNormalization": brightness_contract}},
            "CurrentRepresentation": {"BrightnessNormalization": dict(brightness_contract)},
        }

        runtime = _resolve_brightness_normalization_runtime(preprocessing_contract)

        self.assertTrue(runtime.active())
        self.assertEqual(runtime.config.normalized_method(), "masked_median_darkness_gain")
        self.assertEqual(runtime.mask_source, "silhouette_background_mask < 0.5")
        self.assertFalse(runtime.explicit_mask_source)
        self.assertEqual(runtime.contract_source, "Stages.pack_dual_stream.BrightnessNormalization")

    def test_rejects_unsupported_active_mask_source(self) -> None:
        preprocessing_contract = {
            "Stages": {
                "pack_dual_stream": {
                    "BrightnessNormalization": {
                        "Enabled": True,
                        "Method": "masked_median_darkness_gain",
                        "MaskSource": "roi_fcn_heatmap",
                    }
                }
            }
        }

        with self.assertRaisesRegex(ValueError, "MaskSource='roi_fcn_heatmap'"):
            _resolve_brightness_normalization_runtime(preprocessing_contract)

    def test_startup_report_formats_required_brightness_fields(self) -> None:
        brightness_contract = {
            "Enabled": True,
            "Method": "masked_median_darkness_gain",
            "TargetMedianDarkness": 0.55,
            "MinGain": 0.5,
            "MaxGain": 2.0,
            "Epsilon": 1e-6,
            "EmptyMaskPolicy": "skip",
        }
        model_context = ModelContext(
            label="distance-orientation / demo / run_0001",
            run_dir=PROJECT_ROOT / "models" / "distance-orientation" / "demo" / "runs" / "run_0001",
            checkpoint_path=PROJECT_ROOT / "models" / "distance-orientation" / "demo" / "runs" / "run_0001" / "best.pt",
            device="cuda",
            run_config={},
            run_manifest={},
            task_contract={},
            preprocessing_contract={
                "ContractVersion": "rb-preprocess-v4-dual-stream-orientation-brightness-v1",
                "Stages": {"pack_dual_stream": {"BrightnessNormalization": brightness_contract}},
            },
        )
        roi_context = RoiFcnModelContext(
            label="roi-fcn / demo / run_0001",
            run_dir=PROJECT_ROOT / "models" / "roi-fcn" / "demo" / "runs" / "run_0001",
            checkpoint_path=PROJECT_ROOT / "models" / "roi-fcn" / "demo" / "runs" / "run_0001" / "best.pt",
            device="cuda",
            run_config={},
            dataset_contract={"validation_split": {"geometry": {"canvas_width_px": 480, "canvas_height_px": 300}}},
            canvas_width_px=480,
            canvas_height_px=300,
            roi_width_px=300,
            roi_height_px=300,
        )

        report = build_inference_startup_report(
            model_context=model_context,
            roi_model_context=roi_context,
        )
        lines = format_inference_startup_log(report)

        self.assertTrue(any("inference pipeline version: v0.3" in line for line in lines))
        self.assertTrue(any("brightness normalization active" in line and "true" in line for line in lines))
        self.assertTrue(any("brightness method: masked_median_darkness_gain" in line for line in lines))
        self.assertTrue(any("target median darkness: 0.55" in line for line in lines))
        self.assertTrue(any("empty mask policy: skip" in line for line in lines))
        self.assertTrue(any("mask source: silhouette_background_mask < 0.5" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
