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
from inference_v0_1.pipeline import run_multi_sample_inference, run_single_sample_inference


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
        image_names = list_corpus_image_names(corpus)
        first_image_name = image_names[0]
        second_image_name = image_names[1] if len(image_names) > 1 else image_names[0]
        model_output_name = distance_model.run_dir.parent.parent.name

        with TemporaryDirectory() as tmp_dir:
            results_root_path = Path(tmp_dir) / model_output_name
            first_result = run_single_sample_inference(
                distance_model.run_dir,
                corpus.root,
                first_image_name,
                roi_model_run_dir=roi_model.run_dir,
                save_result=True,
                results_root_path=results_root_path,
                device="cpu",
            )
            second_result = run_single_sample_inference(
                distance_model.run_dir,
                corpus.root,
                second_image_name,
                roi_model_run_dir=roi_model.run_dir,
                save_result=True,
                results_root_path=results_root_path,
                device="cpu",
            )

            self.assertEqual(first_result.selected_image_name, first_image_name)
            self.assertEqual(second_result.selected_image_name, second_image_name)
            self.assertEqual(first_result.selected_model_label, distance_model.label)
            self.assertEqual(first_result.selected_roi_model_label, roi_model.label)
            self.assertEqual(first_result.roi_image.shape, (300, 300))
            self.assertGreaterEqual(float(first_result.roi_image.min()), 0.0)
            self.assertLessEqual(float(first_result.roi_image.max()), 1.0)
            self.assertGreater(first_result.actual_distance_m, 0.0)
            self.assertGreaterEqual(first_result.actual_orientation_deg, 0.0)
            self.assertLess(first_result.actual_orientation_deg, 360.0)
            self.assertIsNotNone(first_result.saved_json_path)
            self.assertIsNotNone(first_result.saved_roi_path)
            self.assertIsNotNone(second_result.saved_json_path)
            self.assertIsNotNone(second_result.saved_roi_path)
            self.assertTrue(first_result.saved_json_path.is_file())
            self.assertTrue(first_result.saved_roi_path.is_file())
            self.assertTrue(second_result.saved_roi_path.is_file())
            self.assertEqual(first_result.saved_json_path, second_result.saved_json_path)
            self.assertEqual(first_result.saved_json_path.parent.name, model_output_name)
            self.assertEqual(
                first_result.saved_json_path.name,
                f"inference-output_{model_output_name}.json",
            )

            payload = json.loads(first_result.saved_json_path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, list)
            self.assertEqual(len(payload), 2)
            self.assertEqual(payload[0]["selected_image"]["image_filename"], first_image_name)
            self.assertEqual(payload[1]["selected_image"]["image_filename"], second_image_name)
            self.assertEqual(payload[1]["selected_corpus"]["name"], corpus.name)
            self.assertEqual(
                payload[1]["selected_models"]["distance_orientation"]["label"],
                distance_model.label,
            )
            self.assertEqual(
                payload[1]["selected_models"]["roi_fcn"]["label"],
                roi_model.label,
            )
            self.assertEqual(len(payload[1]["roi_prediction"]["center_original_xy_px"]), 2)
            self.assertEqual(len(payload[1]["roi_prediction"]["request_xyxy_px"]), 4)


    def test_multi_sample_inference_runs_requested_slice(self) -> None:
        distance_model = discover_model_runs(
            PROJECT_ROOT / "models",
            family="distance-orientation",
        )[0]
        roi_model = discover_model_runs(
            PROJECT_ROOT / "models",
            family="roi-fcn",
        )[-1]
        corpus = self._select_raw_corpus()
        image_names = list_corpus_image_names(corpus)
        offset = 1 if len(image_names) > 1 else 0
        requested_count = 2 if len(image_names) > offset + 1 else 1
        expected_image_names = image_names[offset : offset + requested_count]
        model_output_name = distance_model.run_dir.parent.parent.name

        with TemporaryDirectory() as tmp_dir:
            results_root_path = Path(tmp_dir) / model_output_name
            results = run_multi_sample_inference(
                distance_model.run_dir,
                corpus.root,
                roi_model_run_dir=roi_model.run_dir,
                offset=offset,
                num_samples=requested_count,
                save_result=True,
                results_root_path=results_root_path,
                device="cpu",
            )

            self.assertEqual(len(results), len(expected_image_names))
            self.assertEqual(
                [result.selected_image_name for result in results],
                expected_image_names,
            )
            self.assertTrue(all(result.selected_corpus_name == corpus.name for result in results))
            self.assertTrue(all(result.selected_model_label == distance_model.label for result in results))
            self.assertTrue(all(result.selected_roi_model_label == roi_model.label for result in results))
            self.assertTrue(all(result.saved_json_path is not None for result in results))
            self.assertTrue(all(result.saved_roi_path is not None for result in results))
            self.assertTrue(all(result.saved_json_path == results[0].saved_json_path for result in results))
            self.assertTrue(results[0].saved_json_path.is_file())

            payload = json.loads(results[0].saved_json_path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, list)
            self.assertEqual(len(payload), len(expected_image_names))
            self.assertEqual(
                [entry["selected_image"]["image_filename"] for entry in payload],
                expected_image_names,
            )
            self.assertTrue(all(len(entry["roi_prediction"]["request_xyxy_px"]) == 4 for entry in payload))


if __name__ == "__main__":
    unittest.main()
