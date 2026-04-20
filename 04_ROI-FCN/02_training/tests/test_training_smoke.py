from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from _test_support import build_dataset_reference, ensure_training_root, pushd
from roi_fcn_training_v0_1.contracts import RESUME_STATE_FILENAME
from roi_fcn_training_v0_1.evaluate import resolve_device
from roi_fcn_training_v0_1.paths import build_model_run_dir_path
from roi_fcn_training_v0_1.resume_state import load_resume_state
from roi_fcn_training_v0_1.train import train_roi_fcn


class TrainingSmokeTests(unittest.TestCase):
    def test_end_to_end_smoke_training_writes_artifacts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            build_dataset_reference(
                root,
                "fixture",
                train_centers=[(10.0, 8.0), (16.0, 10.0), (22.0, 12.0), (28.0, 14.0), (34.0, 16.0), (20.0, 20.0)],
                validate_centers=[(12.0, 8.0), (24.0, 14.0), (30.0, 18.0)],
                canvas_width=48,
                canvas_height=32,
                roi_width=10,
                roi_height=10,
            )
            with pushd(root):
                log_messages: list[str] = []
                with patch("roi_fcn_training_v0_1.train.resolve_device", return_value=torch.device("cpu")) as mocked_resolve_device:
                    summary = train_roi_fcn(
                        {
                            "training_dataset": "fixture",
                            "validation_dataset": "fixture",
                            "model_name": "roi-fcn-smoke",
                            "batch_size": 2,
                            "epochs": 5,
                            "early_stopping_patience": 3,
                            "learning_rate": 1e-2,
                            "weight_decay": 0.0,
                            "gaussian_sigma_px": 1.5,
                            "roi_width_px": 10,
                            "roi_height_px": 10,
                            "evaluation_max_visual_examples": 2,
                        },
                        log_sink=log_messages.append,
                    )
                mocked_resolve_device.assert_called_once_with(None, require_cuda=True)
                self.assertTrue(any("bch_roi_acc=" in message for message in log_messages))
                self.assertTrue(any("running_roi_acc=" in message for message in log_messages))
                self.assertTrue(any("train_roi_acc=" in message and "val_roi_acc=" in message for message in log_messages))
            run_dir = Path(summary["run_dir"])
            self.assertEqual(summary["run_id"], "run_0001")
            self.assertEqual(run_dir.name, "run_0001")
            self.assertEqual(run_dir.parent.name, "runs")
            self.assertTrue(run_dir.parent.parent.name.endswith("_roi-fcn-smoke"))
            self.assertEqual(summary["model_directory"], run_dir.parent.parent.name)
            self.assertTrue((run_dir / "run_config.json").is_file())
            self.assertTrue((run_dir / "dataset_contract.json").is_file())
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "validation_predictions.csv").is_file())
            self.assertTrue((run_dir / "validation_metrics.json").is_file())
            self.assertIn("best_validation_mean_center_error_px", summary)
            self.assertLessEqual(summary["validation_metrics"]["mean_center_error_px"], 35.0)

    def test_end_to_end_smoke_training_reuses_precreated_log_only_run_dir(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            build_dataset_reference(
                root,
                "fixture",
                train_centers=[(10.0, 8.0), (16.0, 10.0), (22.0, 12.0), (28.0, 14.0)],
                validate_centers=[(12.0, 8.0), (24.0, 14.0)],
                canvas_width=48,
                canvas_height=32,
                roi_width=10,
                roi_height=10,
            )
            run_dir = build_model_run_dir_path(
                root / "models",
                model_directory="260420-1024_roi-fcn-smoke",
                run_id="run_0001",
            )
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "train.log").write_text("", encoding="utf-8")

            with pushd(root):
                with patch("roi_fcn_training_v0_1.train.resolve_device", return_value=torch.device("cpu")):
                    summary = train_roi_fcn(
                        {
                            "training_dataset": "fixture",
                            "validation_dataset": "fixture",
                            "model_name": "roi-fcn-smoke",
                            "model_directory": "260420-1024_roi-fcn-smoke",
                            "run_id": "run_0001",
                            "batch_size": 2,
                            "epochs": 2,
                            "early_stopping_patience": 2,
                            "learning_rate": 1e-2,
                            "weight_decay": 0.0,
                            "gaussian_sigma_px": 1.5,
                            "roi_width_px": 10,
                            "roi_height_px": 10,
                            "evaluation_max_visual_examples": 1,
                        },
                    )

            self.assertEqual(Path(summary["run_dir"]), run_dir)
            self.assertEqual(summary["model_directory"], "260420-1024_roi-fcn-smoke")
            self.assertEqual(summary["run_id"], "run_0001")
            self.assertTrue((run_dir / "train.log").is_file())
            self.assertTrue((run_dir / "best.pt").is_file())

    def test_end_to_end_smoke_training_uses_next_run_id_for_existing_model_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            build_dataset_reference(
                root,
                "fixture",
                train_centers=[(10.0, 8.0), (16.0, 10.0), (22.0, 12.0), (28.0, 14.0)],
                validate_centers=[(12.0, 8.0), (24.0, 14.0)],
                canvas_width=48,
                canvas_height=32,
                roi_width=10,
                roi_height=10,
            )
            existing_run_dir = build_model_run_dir_path(
                root / "models",
                model_directory="260420-1024_roi-fcn-smoke",
                run_id="run_0001",
            )
            existing_run_dir.mkdir(parents=True, exist_ok=False)
            (existing_run_dir / "best.pt").write_text("placeholder", encoding="utf-8")

            with pushd(root):
                with patch("roi_fcn_training_v0_1.train.resolve_device", return_value=torch.device("cpu")):
                    summary = train_roi_fcn(
                        {
                            "training_dataset": "fixture",
                            "validation_dataset": "fixture",
                            "model_name": "roi-fcn-smoke",
                            "model_directory": "260420-1024_roi-fcn-smoke",
                            "batch_size": 2,
                            "epochs": 2,
                            "early_stopping_patience": 2,
                            "learning_rate": 1e-2,
                            "weight_decay": 0.0,
                            "gaussian_sigma_px": 1.5,
                            "roi_width_px": 10,
                            "roi_height_px": 10,
                            "evaluation_max_visual_examples": 1,
                        },
                    )

            self.assertEqual(summary["model_directory"], "260420-1024_roi-fcn-smoke")
            self.assertEqual(summary["run_id"], "run_0002")
            self.assertEqual(
                Path(summary["run_dir"]),
                build_model_run_dir_path(
                    root / "models",
                    model_directory="260420-1024_roi-fcn-smoke",
                    run_id="run_0002",
                ),
            )

    def test_resume_training_continues_into_child_run(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            build_dataset_reference(
                root,
                "fixture",
                train_centers=[(10.0, 8.0), (16.0, 10.0), (22.0, 12.0), (28.0, 14.0), (34.0, 16.0), (20.0, 20.0)],
                validate_centers=[(12.0, 8.0), (24.0, 14.0), (30.0, 18.0)],
                canvas_width=48,
                canvas_height=32,
                roi_width=10,
                roi_height=10,
            )
            with pushd(root):
                with patch("roi_fcn_training_v0_1.train.resolve_device", return_value=torch.device("cpu")):
                    base_summary = train_roi_fcn(
                        {
                            "training_dataset": "fixture",
                            "validation_dataset": "fixture",
                            "model_name": "roi-fcn-smoke",
                            "model_directory": "260420-1024_roi-fcn-smoke",
                            "run_id": "run_0001",
                            "batch_size": 2,
                            "epochs": 2,
                            "early_stopping_patience": 10,
                            "learning_rate": 1e-2,
                            "weight_decay": 0.0,
                            "gaussian_sigma_px": 1.5,
                            "roi_width_px": 10,
                            "roi_height_px": 10,
                            "evaluation_max_visual_examples": 1,
                        },
                    )
                    resumed_summary = train_roi_fcn(
                        {
                            "training_dataset": "fixture",
                            "validation_dataset": "fixture",
                            "model_name": "roi-fcn-smoke",
                            "model_directory": "260420-1024_roi-fcn-smoke",
                            "run_id": "run_0002",
                            "batch_size": 2,
                            "epochs": 2,
                            "early_stopping_patience": 10,
                            "learning_rate": 1e-2,
                            "weight_decay": 0.0,
                            "gaussian_sigma_px": 1.5,
                            "roi_width_px": 10,
                            "roi_height_px": 10,
                            "evaluation_max_visual_examples": 1,
                            "resume_from_run_dir": str(base_summary["run_dir"]),
                            "additional_epochs": 2,
                        },
                    )

            resumed_run_dir = Path(resumed_summary["run_dir"])
            history_rows = json.loads((resumed_run_dir / "history.json").read_text(encoding="utf-8"))
            self.assertEqual([int(row["epoch"]) for row in history_rows], [1, 2, 3, 4])
            resume_state = load_resume_state(resumed_run_dir / RESUME_STATE_FILENAME)
            self.assertEqual(int(resume_state["epoch"]), 4)
            self.assertTrue((resumed_run_dir / RESUME_STATE_FILENAME).is_file())
            self.assertTrue(bool(resumed_summary["resume"]["enabled"]))
            self.assertEqual(str(resumed_summary["resume"]["source_run_id"]), "run_0001")

    def test_training_device_resolution_requires_cuda(self) -> None:
        with patch("roi_fcn_training_v0_1.evaluate.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                resolve_device(None, require_cuda=True)
            with self.assertRaisesRegex(ValueError, "cannot be used for ROI-FCN training"):
                resolve_device("cpu", require_cuda=True)


if __name__ == "__main__":
    unittest.main()
