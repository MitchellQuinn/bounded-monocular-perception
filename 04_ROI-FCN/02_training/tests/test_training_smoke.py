from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from _test_support import build_dataset_reference, ensure_training_root, pushd
from roi_fcn_training_v0_1.evaluate import resolve_device
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
                        }
                    )
                mocked_resolve_device.assert_called_once_with(None, require_cuda=True)
            run_dir = Path(summary["run_dir"])
            self.assertTrue((run_dir / "run_config.json").is_file())
            self.assertTrue((run_dir / "dataset_contract.json").is_file())
            self.assertTrue((run_dir / "best.pt").is_file())
            self.assertTrue((run_dir / "validation_predictions.csv").is_file())
            self.assertTrue((run_dir / "validation_metrics.json").is_file())
            self.assertLessEqual(summary["validation_metrics"]["mean_center_error_px"], 35.0)


    def test_training_device_resolution_requires_cuda(self) -> None:
        with patch("roi_fcn_training_v0_1.evaluate.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                resolve_device(None, require_cuda=True)
            with self.assertRaisesRegex(ValueError, "cannot be used for ROI-FCN training"):
                resolve_device("cpu", require_cuda=True)


if __name__ == "__main__":
    unittest.main()
