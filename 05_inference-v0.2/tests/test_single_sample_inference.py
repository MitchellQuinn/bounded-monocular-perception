"""End-to-end smoke tests for the v0.2 raw-image inference pipeline."""

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

from inference_v0_1.discovery import discover_model_runs, discover_raw_corpora, list_corpus_image_names
from inference_v0_1.pipeline import run_single_sample_inference


class SingleSampleInferenceTests(unittest.TestCase):
    def _select_raw_corpus(self):
        corpora = discover_raw_corpora()
        self.assertTrue(corpora)
        return next((corpus for corpus in corpora if "input-images" in corpus.root.parts), corpora[0])

    def test_discover_model_runs_includes_both_runtime_families(self) -> None:
        models = discover_model_runs(PROJECT_ROOT / "models")
        families = {artifact.model_family for artifact in models}

        self.assertIn("distance-orientation", families)
        self.assertIn("roi-fcn", families)
        self.assertTrue(discover_model_runs(PROJECT_ROOT / "models", family="distance"))
        self.assertTrue(discover_model_runs(PROJECT_ROOT / "models", family="roi"))

    def test_discover_raw_corpora_ignores_npz_only_input(self) -> None:
        explicit_local = discover_raw_corpora(PROJECT_ROOT / "input")
        self.assertEqual(explicit_local, [])

        corpora = discover_raw_corpora()
        corpus_names = [corpus.name for corpus in corpora]

        self.assertTrue(corpus_names)
        self.assertNotIn("26-04-11_v021-validate-shuffled-images", corpus_names)
        self.assertTrue(any("input-images" in corpus.root.parts for corpus in corpora))

    def test_single_sample_inference_runs_end_to_end(self) -> None:
        distance_model = discover_model_runs(
            PROJECT_ROOT / "models",
            family="distance-orientation",
        )[0]
        roi_model = discover_model_runs(
            PROJECT_ROOT / "models",
            family="roi-fcn",
        )[-1]
        corpus = self._select_raw_corpus()
        image_name = list_corpus_image_names(corpus)[0]

        with TemporaryDirectory() as tmp_dir:
            result = run_single_sample_inference(
                distance_model.run_dir,
                corpus.root,
                image_name,
                roi_model_run_dir=roi_model.run_dir,
                save_result=True,
                results_root_path=Path(tmp_dir),
                device="cpu",
            )

            self.assertEqual(result.selected_image_name, image_name)
            self.assertEqual(result.selected_model_label, distance_model.label)
            self.assertEqual(result.selected_roi_model_label, roi_model.label)
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
            self.assertEqual(
                payload["selected_models"]["distance_orientation"]["label"],
                distance_model.label,
            )
            self.assertEqual(
                payload["selected_models"]["roi_fcn"]["label"],
                roi_model.label,
            )
            self.assertEqual(len(payload["roi_prediction"]["center_original_xy_px"]), 2)
            self.assertEqual(len(payload["roi_prediction"]["request_xyxy_px"]), 4)


if __name__ == "__main__":
    unittest.main()
