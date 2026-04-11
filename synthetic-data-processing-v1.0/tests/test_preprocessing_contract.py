"""Tests for authoritative preprocessing contract metadata."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rb_pipeline.manifest import load_run_json, upsert_preprocessing_contract


class PreprocessingContractTests(unittest.TestCase):
    def test_upsert_preprocessing_contract_records_stage_settings(self) -> None:
        with TemporaryDirectory() as tmpdir:
            manifests_dir = Path(tmpdir) / "fixture-corpus" / "manifests"
            manifests_dir.mkdir(parents=True, exist_ok=True)
            (manifests_dir / "run.json").write_text(
                json.dumps({"RunId": "fixture-run"}, indent=4) + "\n",
                encoding="utf-8",
            )

            upsert_preprocessing_contract(
                manifests_dir,
                stage_name="edge",
                stage_parameters={
                    "BlurKernelSize": 5,
                    "BlurKernelSizeUsed": 5,
                    "CannyLowThreshold": 50,
                    "CannyHighThreshold": 150,
                },
                current_representation={
                    "Kind": "edge_png",
                    "StorageFormat": "png",
                },
            )
            upsert_preprocessing_contract(
                manifests_dir,
                stage_name="bbox",
                stage_parameters={
                    "ForegroundThreshold": 250,
                    "LineThicknessPx": 3,
                    "LineThicknessPxUsed": 3,
                    "PaddingPx": 0,
                    "PaddingPxUsed": 0,
                    "PostDrawBlur": False,
                    "PostDrawBlurKernelSize": 3,
                    "PostDrawBlurKernelSizeUsed": 3,
                },
                current_representation={
                    "Kind": "full_frame_bbox_png",
                    "StorageFormat": "png",
                    "Geometry": "full_frame_bbox_outline",
                },
            )

            payload = load_run_json(manifests_dir)
            contract = payload["PreprocessingContract"]

            self.assertEqual(contract["ContractVersion"], "rb-preprocess-v1")
            self.assertEqual(contract["CurrentStage"], "bbox")
            self.assertEqual(contract["CompletedStages"], ["edge", "bbox"])
            self.assertEqual(contract["Stages"]["bbox"]["LineThicknessPx"], 3)
            self.assertEqual(contract["Stages"]["bbox"]["LineThicknessPxUsed"], 3)
            self.assertFalse(contract["Stages"]["bbox"]["PostDrawBlur"])
            self.assertEqual(
                contract["CurrentRepresentation"]["Kind"],
                "full_frame_bbox_png",
            )


if __name__ == "__main__":
    unittest.main()
