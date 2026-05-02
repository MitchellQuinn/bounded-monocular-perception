"""End-to-end smoke tests for the v0.3 raw-image inference pipeline."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference_v0_1.discovery import (
    RawCorpus,
    default_raw_corpus_roots,
    discover_model_runs,
    discover_raw_corpora,
    list_corpus_image_names,
)
from inference_v0_1.pipeline import (
    InferenceResult,
    load_model_context,
    load_roi_fcn_model_context,
    run_multi_sample_inference,
    run_single_sample_inference,
    save_inference_result,
)
import inference_v0_1.pipeline as pipeline
from src.task_runtime import batch_to_model_inputs


class SingleSampleInferenceTests(unittest.TestCase):
    def _select_raw_corpus(self):
        corpora = discover_raw_corpora()
        if not corpora:
            self.skipTest("No local raw-image corpora available under 05_inference-v0.4/input.")
        return corpora[0]

    def test_discover_model_runs_includes_both_runtime_families(self) -> None:
        models = discover_model_runs(PROJECT_ROOT / "models")
        families = {artifact.model_family for artifact in models}

        self.assertIn("distance-orientation", families)
        self.assertIn("roi-fcn", families)
        self.assertTrue(discover_model_runs(PROJECT_ROOT / "models", family="distance"))
        self.assertTrue(discover_model_runs(PROJECT_ROOT / "models", family="roi"))

    def test_default_raw_corpus_roots_use_local_inference_input_only(self) -> None:
        self.assertEqual(default_raw_corpus_roots(), [(PROJECT_ROOT / "input").resolve()])

    def test_discover_raw_corpora_ignores_npz_only_input(self) -> None:
        explicit_local = discover_raw_corpora(PROJECT_ROOT / "input")
        default_local = discover_raw_corpora()

        self.assertEqual(default_local, explicit_local)

        corpus_names = [corpus.name for corpus in explicit_local]
        self.assertNotIn("26-04-11_v021-validate-shuffled-images", corpus_names)
        self.assertTrue(all(PROJECT_ROOT / "input" in corpus.root.parents for corpus in explicit_local))

    def test_run_functions_reject_non_raw_corpus_paths(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            corpus_root = Path(tmp_dir) / "npz-only-corpus"
            corpus_root.mkdir()

            with self.assertRaisesRegex(FileNotFoundError, "only supports raw-image corpora"):
                pipeline._resolve_raw_corpus(corpus_root)

    def test_load_model_context_reads_dataset_summary_preprocessing_contract(self) -> None:
        preprocessing_contract = {
            "ContractVersion": "rb-preprocess-v4-tri-stream-orientation-v1",
            "CurrentStage": "pack_tri_stream",
            "CompletedStages": ["detect", "silhouette", "pack_tri_stream"],
            "CurrentRepresentation": {"Kind": "tri_stream_npz"},
            "Stages": {"pack_tri_stream": {"CanvasWidth": 300, "CanvasHeight": 300}},
        }

        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "model" / "runs" / "run_0001"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text("{}", encoding="utf-8")
            (run_dir / "dataset_summary.json").write_text(
                json.dumps({"preprocessing_contract": preprocessing_contract}),
                encoding="utf-8",
            )
            (run_dir / "best.pt").touch()

            with (
                patch("inference_v0_1.pipeline.torch.cuda.is_available", return_value=True),
                patch.object(
                    pipeline,
                    "_load_model_from_run",
                    return_value=(
                        torch.nn.Identity(),
                        SimpleNamespace(
                            task_contract={"input_mode": pipeline.TRI_STREAM_INPUT_MODE}
                        ),
                    ),
                ),
            ):
                _, context = load_model_context(run_dir, device="cuda")

        self.assertEqual(
            context.preprocessing_contract["ContractVersion"],
            "rb-preprocess-v4-tri-stream-orientation-v1",
        )
        self.assertEqual(context.preprocessing_contract["CurrentStage"], "pack_tri_stream")
        self.assertEqual(context.dataset_summary["preprocessing_contract"], preprocessing_contract)

    def test_tri_stream_batch_includes_orientation_and_geometry_inputs(self) -> None:
        samples = [
            SimpleNamespace(
                sample_row={"sample_id": f"sample-{idx}"},
                model_image=np.full((1, 4, 4), 0.25 + idx, dtype=np.float32),
                orientation_image=np.full((1, 4, 4), 0.75 + idx, dtype=np.float32),
                bbox_features=np.arange(10, dtype=np.float32) + idx,
                input_mode=pipeline.TRI_STREAM_INPUT_MODE,
                actual_distance_m=float(idx + 1),
                actual_yaw_sin=0.0,
                actual_yaw_cos=1.0,
            )
            for idx in range(2)
        ]

        batch = pipeline._build_multi_sample_batch(samples)
        model_inputs = batch_to_model_inputs(
            batch,
            {"input_mode": pipeline.TRI_STREAM_INPUT_MODE},
            device=torch.device("cpu"),
        )

        self.assertIsNone(batch.bbox_features)
        self.assertEqual(batch.images.shape, (2, 1, 4, 4))
        self.assertIsNotNone(batch.geometry)
        self.assertEqual(batch.geometry.shape, (2, 10))
        self.assertIsNotNone(batch.extra_inputs)
        self.assertEqual(
            batch.extra_inputs[pipeline.TRI_STREAM_ORIENTATION_IMAGE_KEY].shape,
            (2, 1, 4, 4),
        )
        self.assertEqual(
            set(model_inputs),
            {"x_distance_image", "x_orientation_image", "x_geometry"},
        )

    def test_tri_stream_preprocess_orientation_uses_inverted_white_background(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            corpus_root = Path(tmp_dir) / "demo-corpus"
            images_dir = corpus_root / "images"
            manifests_dir = corpus_root / "manifests"
            images_dir.mkdir(parents=True)
            manifests_dir.mkdir(parents=True)

            image = np.zeros((64, 64), dtype=np.uint8)
            cv2.rectangle(image, (18, 18), (46, 46), color=85, thickness=-1)
            cv2.circle(image, (32, 32), 9, color=165, thickness=-1)
            cv2.line(image, (20, 44), (44, 20), color=225, thickness=2)
            image_path = images_dir / "frame.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            run_json_path = manifests_dir / "run.json"
            samples_csv_path = manifests_dir / "samples.csv"
            run_json_path.write_text("{}", encoding="utf-8")
            samples_csv_path.write_text("", encoding="utf-8")

            sample_row = pd.Series(
                {
                    "__image_path__": str(image_path),
                    "image_filename": "frame.png",
                    "sample_id": "sample-001",
                    "distance_m": 4.0,
                    "final_rot_y_deg": 90.0,
                }
            )
            preprocessing_contract = {
                "ContractVersion": "rb-preprocess-v4-tri-stream-orientation-v1",
                "CurrentStage": "pack_tri_stream",
                "CompletedStages": ["detect", "silhouette", "pack_tri_stream"],
                "CurrentRepresentation": {
                    "Kind": "tri_stream_npz",
                    "CanvasWidth": 64,
                    "CanvasHeight": 64,
                    "OrientationContextScale": 1.25,
                },
                "Stages": {
                    "silhouette": {
                        "RepresentationMode": "filled",
                        "ROICanvasWidthPx": 64,
                        "ROICanvasHeightPx": 64,
                    },
                    "pack_tri_stream": {
                        "CanvasWidth": 64,
                        "CanvasHeight": 64,
                        "ClipPolicy": "fail",
                        "OrientationContextScale": 1.25,
                    },
                },
            }
            model_context = pipeline.ModelContext(
                label="distance-orientation / demo / run",
                run_dir=Path(tmp_dir) / "distance-run",
                checkpoint_path=Path(tmp_dir) / "distance-run" / "best.pt",
                device="cpu",
                run_config={},
                run_manifest={},
                dataset_summary={},
                task_contract={"input_mode": pipeline.TRI_STREAM_INPUT_MODE},
                preprocessing_contract=preprocessing_contract,
            )
            roi_model_context = pipeline.RoiFcnModelContext(
                label="roi-fcn / demo / run",
                run_dir=Path(tmp_dir) / "roi-run",
                checkpoint_path=Path(tmp_dir) / "roi-run" / "best.pt",
                device="cpu",
                run_config={},
                dataset_contract={},
                canvas_width_px=64,
                canvas_height_px=64,
                roi_width_px=64,
                roi_height_px=64,
            )

            with patch.object(
                pipeline,
                "_predict_roi_center",
                return_value=(32.0, 32.0, np.asarray([0.0, 0.0, 64.0, 64.0], dtype=np.float32)),
            ):
                preprocessed = pipeline.preprocess_single_sample(
                    corpus=RawCorpus(
                        name="demo-corpus",
                        root=corpus_root,
                        images_dir=images_dir,
                        run_json_path=run_json_path,
                        samples_csv_path=samples_csv_path,
                    ),
                    sample_row=sample_row,
                    model_context=model_context,
                    roi_model=object(),
                    roi_model_context=roi_model_context,
                )

            self.assertIsNotNone(preprocessed.orientation_image)
            orientation = preprocessed.orientation_image[0]
            self.assertGreater(float(np.mean(orientation[:4, :4])), 0.99)
            self.assertGreater(float(np.mean(orientation[-4:, -4:])), 0.99)
            self.assertTrue(bool(np.any(orientation < 0.99)))

    def test_save_inference_result_can_skip_roi_image_output(self) -> None:
        run_dir = PROJECT_ROOT / "models" / "distance-orientation" / "demo-model" / "runs" / "run_demo"
        checkpoint_path = run_dir / "best.pt"
        roi_run_dir = PROJECT_ROOT / "models" / "roi-fcn" / "demo-roi" / "runs" / "run_demo"
        roi_checkpoint_path = roi_run_dir / "best.pt"
        source_image_path = PROJECT_ROOT / "tests" / "fixtures" / "sample.png"
        source_run_json_path = PROJECT_ROOT / "tests" / "fixtures" / "run.json"
        source_samples_csv_path = PROJECT_ROOT / "tests" / "fixtures" / "samples.csv"
        result = InferenceResult(
            selected_model_label="distance-orientation / demo-model / run_demo",
            selected_roi_model_label="roi-fcn / demo-roi / run_demo",
            selected_corpus_name="demo-corpus",
            selected_image_name="sample.png",
            sample_id="sample-001",
            roi_image=np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32),
            predicted_crop_center_x_px=123.0,
            predicted_crop_center_y_px=456.0,
            predicted_roi_request_xyxy_px=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
            predicted_distance_m=4.5,
            actual_distance_m=4.0,
            distance_delta_m=0.5,
            absolute_distance_error_m=0.5,
            predicted_orientation_deg=91.0,
            actual_orientation_deg=89.0,
            orientation_delta_deg=2.0,
            absolute_orientation_error_deg=2.0,
            device="cuda",
            roi_device="cuda",
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            roi_run_dir=roi_run_dir,
            roi_checkpoint_path=roi_checkpoint_path,
            source_image_path=source_image_path,
            source_run_json_path=source_run_json_path,
            source_samples_csv_path=source_samples_csv_path,
            preprocessing_contract_version="v-test",
        )

        with TemporaryDirectory() as tmp_dir:
            json_path, roi_path = save_inference_result(
                result,
                root=tmp_dir,
                save_roi_image=False,
            )

            self.assertTrue(json_path.is_file())
            self.assertIsNone(roi_path)
            self.assertEqual(list(Path(tmp_dir).glob("*.roi.png")), [])

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIsInstance(payload, list)
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["selected_image"]["image_filename"], "sample.png")
            self.assertEqual(payload[0]["artifacts"]["json_path"], str(json_path.resolve()))
            self.assertNotIn("roi_image_path", payload[0]["artifacts"])

    def test_batched_json_appender_keeps_output_valid_after_each_batch(self) -> None:
        run_dir = PROJECT_ROOT / "models" / "distance-orientation" / "demo-model" / "runs" / "run_demo"
        checkpoint_path = run_dir / "best.pt"
        roi_run_dir = PROJECT_ROOT / "models" / "roi-fcn" / "demo-roi" / "runs" / "run_demo"
        roi_checkpoint_path = roi_run_dir / "best.pt"

        def make_result(sample_id: str) -> InferenceResult:
            return InferenceResult(
                selected_model_label="distance-orientation / demo-model / run_demo",
                selected_roi_model_label="roi-fcn / demo-roi / run_demo",
                selected_corpus_name="demo-corpus",
                selected_image_name=f"{sample_id}.png",
                sample_id=sample_id,
                roi_image=np.ones((2, 2), dtype=np.float32),
                predicted_crop_center_x_px=1.0,
                predicted_crop_center_y_px=1.0,
                predicted_roi_request_xyxy_px=np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32),
                predicted_distance_m=4.0,
                actual_distance_m=4.0,
                distance_delta_m=0.0,
                absolute_distance_error_m=0.0,
                predicted_orientation_deg=90.0,
                actual_orientation_deg=90.0,
                orientation_delta_deg=0.0,
                absolute_orientation_error_deg=0.0,
                device="cuda",
                roi_device="cuda",
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                roi_run_dir=roi_run_dir,
                roi_checkpoint_path=roi_checkpoint_path,
                source_image_path=PROJECT_ROOT / "tests" / "fixtures" / f"{sample_id}.png",
                source_run_json_path=PROJECT_ROOT / "tests" / "fixtures" / "run.json",
                source_samples_csv_path=PROJECT_ROOT / "tests" / "fixtures" / "samples.csv",
                preprocessing_contract_version="v-test",
            )

        with TemporaryDirectory() as tmp_dir:
            target_root = Path(tmp_dir)
            json_path = target_root / "inference-output_demo-model.json"
            appender = pipeline._JsonArrayBatchAppender(json_path)

            first_batch = pipeline._save_inference_result_batch(
                [make_result("sample-001"), make_result("sample-002")],
                json_appender=appender,
                target_root=target_root,
                save_roi_images=False,
            )
            first_payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(first_payload), 2)
            self.assertTrue(all(result.saved_json_path == json_path for result in first_batch))

            second_batch = pipeline._save_inference_result_batch(
                [make_result("sample-003")],
                json_appender=appender,
                target_root=target_root,
                save_roi_images=False,
            )
            second_payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(second_payload), 3)
            self.assertEqual(second_payload[-1]["selected_image"]["sample_id"], "sample-003")
            self.assertTrue(all(result.saved_json_path == json_path for result in second_batch))

    def test_distance_model_loading_requires_cuda(self) -> None:
        distance_model = discover_model_runs(
            PROJECT_ROOT / "models",
            family="distance-orientation",
        )[0]

        with patch("inference_v0_1.pipeline.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "Inference requires CUDA"):
                load_model_context(distance_model.run_dir)

    def test_roi_fcn_model_loading_rejects_cpu_override(self) -> None:
        roi_model = discover_model_runs(
            PROJECT_ROOT / "models",
            family="roi-fcn",
        )[-1]

        with patch("inference_v0_1.pipeline.torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(ValueError, "Requested device 'cpu' cannot be used for inference"):
                load_roi_fcn_model_context(roi_model.run_dir, device="cpu")

    def test_multi_sample_inference_reports_preprocessing_progress(self) -> None:
        selected_rows = pd.DataFrame(
            [
                {"sample_id": "sample-001"},
                {"sample_id": "sample-002"},
                {"sample_id": "sample-003"},
            ]
        )
        observed_progress: list[tuple[int, int]] = []
        observed_batch_sizes: list[int] = []

        def fake_preprocess_single_sample(*, sample_row, **_kwargs):
            return SimpleNamespace(
                sample_row={"sample_id": str(sample_row["sample_id"])},
                model_image=np.zeros((1, 2, 2), dtype=np.float32),
                bbox_features=np.zeros((2,), dtype=np.float32),
                actual_distance_m=1.0,
                actual_yaw_sin=0.0,
                actual_yaw_cos=1.0,
            )

        def fake_build_inference_result(*, preprocessed, **_kwargs):
            return preprocessed.sample_row["sample_id"]

        def fake_run_prediction_batch(*, batch, **_kwargs):
            observed_batch_sizes.append(int(batch.images.shape[0]))
            return pd.DataFrame([{} for _ in range(int(batch.images.shape[0]))])

        with (
            patch.object(
                pipeline,
                "_resolve_raw_corpus",
                return_value=SimpleNamespace(name="demo-corpus"),
            ),
            patch.object(pipeline, "load_corpus_samples", return_value=selected_rows),
            patch.object(
                pipeline,
                "load_model_context",
                return_value=(object(), SimpleNamespace(device="cuda")),
            ),
            patch.object(
                pipeline,
                "load_roi_fcn_model_context",
                return_value=(object(), SimpleNamespace(device="cuda")),
            ),
            patch.object(
                pipeline,
                "preprocess_single_sample",
                side_effect=fake_preprocess_single_sample,
            ),
            patch.object(
                pipeline,
                "_run_prediction_batch",
                side_effect=fake_run_prediction_batch,
            ),
            patch.object(
                pipeline,
                "_build_inference_result",
                side_effect=fake_build_inference_result,
            ),
        ):
            results = run_multi_sample_inference(
                "distance-run",
                "corpus-root",
                roi_model_run_dir="roi-run",
                num_samples=3,
                batch_size=2,
                progress_callback=lambda processed, total: observed_progress.append(
                    (processed, total)
                ),
            )

        self.assertEqual(observed_progress, [(2, 3), (3, 3)])
        self.assertEqual(observed_batch_sizes, [2, 1])
        self.assertEqual(results, ["sample-001", "sample-002", "sample-003"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference smoke tests.")
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
                device="cuda",
            )
            second_result = run_single_sample_inference(
                distance_model.run_dir,
                corpus.root,
                second_image_name,
                roi_model_run_dir=roi_model.run_dir,
                save_result=True,
                results_root_path=results_root_path,
                device="cuda",
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


    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for inference smoke tests.")
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
                device="cuda",
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
