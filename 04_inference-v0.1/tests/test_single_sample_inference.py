"""End-to-end smoke tests for the v0.1 raw-image inference pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_v0_1.discovery import (
    discover_model_runs,
    discover_raw_corpora,
    list_corpus_image_names,
)
from inference_v0_1.pipeline import run_single_sample_inference


class SingleSampleInferenceTests(unittest.TestCase):
    def test_discover_raw_corpora_ignores_npz_only_input(self) -> None:
        corpora = discover_raw_corpora(PROJECT_ROOT / "input")
        corpus_names = [corpus.name for corpus in corpora]

        self.assertIn("def90_synth_v023-validation-smoketest-raw", corpus_names)
        self.assertNotIn("26-04-11_v020-train-shuffled-images-smoketest", corpus_names)

    def test_single_sample_inference_runs_end_to_end(self) -> None:
        model = discover_model_runs(PROJECT_ROOT / "models")[0]
        corpora = {
            corpus.name: corpus
            for corpus in discover_raw_corpora(PROJECT_ROOT / "input")
        }
        corpus = corpora["def90_synth_v023-validation-smoketest-raw"]
        image_name = list_corpus_image_names(corpus)[0]

        with TemporaryDirectory() as tmp_dir:
            result = run_single_sample_inference(
                model.run_dir,
                corpus.root,
                image_name,
                save_result=True,
                results_root_path=Path(tmp_dir),
                device="cpu",
            )

            self.assertEqual(result.selected_image_name, image_name)
            self.assertEqual(result.roi_image.shape, (300, 300))
            self.assertGreaterEqual(float(result.roi_image.min()), 0.0)
            self.assertLessEqual(float(result.roi_image.max()), 1.0)
            self.assertGreater(result.actual_distance_m, 0.0)
            self.assertGreaterEqual(result.actual_orientation_deg, 0.0)
            self.assertLess(result.actual_orientation_deg, 360.0)
            self.assertIsNotNone(result.saved_json_path)
            self.assertIsNotNone(result.saved_roi_path)
            self.assertTrue(result.saved_json_path.is_file())
            self.assertTrue(result.saved_roi_path.is_file())

            payload = json.loads(result.saved_json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["selected_image"]["image_filename"], image_name)
            self.assertEqual(payload["selected_corpus"]["name"], corpus.name)


if __name__ == "__main__":
    unittest.main()
