"""Tests for structured epoch-summary helpers."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from src.epoch_summary import read_epoch_summary_panel
from src.resume.state import RESUME_STATE_FILENAME
from src.topologies import resolve_topology_spec


class EpochSummaryTests(unittest.TestCase):
    def test_distance_only_summary_uses_shared_val_loss_rule(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_2d_cnn",
            topology_variant="fast_v0_2",
            topology_params={},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_config(run_dir, spec.task_contract)
            pd.DataFrame(
                [
                    {
                        "epoch": 1,
                        "learning_rate": 1e-3,
                        "next_learning_rate": 1e-3,
                        "train_loss": 0.70,
                        "val_loss": 0.60,
                        "val_mae": 0.30,
                        "val_rmse": 0.35,
                        "train_acc_at_0p10m": 0.20,
                        "train_acc_at_0p25m": 0.45,
                        "train_acc_at_0p50m": 0.72,
                        "val_acc_at_0p10m": 0.18,
                        "val_acc_at_0p25m": 0.41,
                        "val_acc_at_0p50m": 0.68,
                    },
                    {
                        "epoch": 2,
                        "learning_rate": 1e-3,
                        "next_learning_rate": 5e-4,
                        "train_loss": 0.52,
                        "val_loss": 0.44,
                        "val_mae": 0.22,
                        "val_rmse": 0.28,
                        "train_acc_at_0p10m": 0.44,
                        "train_acc_at_0p25m": 0.70,
                        "train_acc_at_0p50m": 0.87,
                        "val_acc_at_0p10m": 0.39,
                        "val_acc_at_0p25m": 0.63,
                        "val_acc_at_0p50m": 0.82,
                    },
                    {
                        "epoch": 3,
                        "learning_rate": 5e-4,
                        "next_learning_rate": 5e-4,
                        "train_loss": 0.49,
                        "val_loss": 0.44,
                        "val_mae": 0.20,
                        "val_rmse": 0.26,
                        "train_acc_at_0p10m": 0.48,
                        "train_acc_at_0p25m": 0.74,
                        "train_acc_at_0p50m": 0.89,
                        "val_acc_at_0p10m": 0.42,
                        "val_acc_at_0p25m": 0.66,
                        "val_acc_at_0p50m": 0.85,
                    },
                ]
            ).to_csv(run_dir / "history.csv", index=False)

            panel = read_epoch_summary_panel(run_dir)

        self.assertEqual(panel.criterion_metric, "val_loss")
        self.assertEqual(panel.latest_epoch, 3)
        self.assertEqual(panel.best_epoch, 2)
        self.assertIn("Best selected by: val_loss", panel.text)
        self.assertIn("Latest completed", panel.text)
        self.assertIn("Best so far", panel.text)
        self.assertIn("train_acc@0.10m=0.4800", panel.text)
        self.assertIn("val_acc@0.50m=0.8500", panel.text)
        self.assertNotIn("yaw_mean_error_deg", panel.text)

    def test_multitask_summary_falls_back_to_resume_state_history(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_dual_stream_yaw",
            topology_variant="dual_stream_yaw_v0_1",
            topology_params={},
        )

        history_records = [
            {
                "epoch": 4,
                "learning_rate": 5e-4,
                "next_learning_rate": 5e-4,
                "train_loss": 0.40,
                "val_loss": 0.36,
                "val_mae": 0.17,
                "val_rmse": 0.21,
                "train_acc_at_0p10m": 0.51,
                "train_acc_at_0p25m": 0.79,
                "train_acc_at_0p50m": 0.93,
                "val_acc_at_0p10m": 0.47,
                "val_acc_at_0p25m": 0.74,
                "val_acc_at_0p50m": 0.90,
                "train_distance_loss": 0.28,
                "train_orientation_loss": 0.12,
                "val_distance_loss": 0.25,
                "val_orientation_loss": 0.11,
                "val_yaw_mean_error_deg": 8.4,
                "val_yaw_median_error_deg": 6.1,
                "val_yaw_p95_error_deg": 18.2,
                "val_yaw_acc@5deg": 0.31,
                "val_yaw_acc@10deg": 0.67,
                "val_yaw_acc@15deg": 0.84,
            },
            {
                "epoch": 5,
                "learning_rate": 5e-4,
                "next_learning_rate": 2.5e-4,
                "train_loss": 0.34,
                "val_loss": 0.29,
                "val_mae": 0.15,
                "val_rmse": 0.18,
                "train_acc_at_0p10m": 0.58,
                "train_acc_at_0p25m": 0.84,
                "train_acc_at_0p50m": 0.95,
                "val_acc_at_0p10m": 0.53,
                "val_acc_at_0p25m": 0.79,
                "val_acc_at_0p50m": 0.92,
                "train_distance_loss": 0.23,
                "train_orientation_loss": 0.11,
                "val_distance_loss": 0.20,
                "val_orientation_loss": 0.09,
                "val_yaw_mean_error_deg": 6.7,
                "val_yaw_median_error_deg": 4.9,
                "val_yaw_p95_error_deg": 15.0,
                "val_yaw_acc@5deg": 0.42,
                "val_yaw_acc@10deg": 0.75,
                "val_yaw_acc@15deg": 0.90,
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            self._write_config(run_dir, spec.task_contract)
            torch.save(
                {
                    "format_version": 1,
                    "epoch": 5,
                    "run_id": "run_0002",
                    "model_state_dict": {},
                    "optimizer_state_dict": {},
                    "best_epoch": 5,
                    "best_val_loss": 0.29,
                    "no_improvement_epochs": 0,
                    "history_records": history_records,
                },
                run_dir / RESUME_STATE_FILENAME,
            )

            panel = read_epoch_summary_panel(run_dir)

        self.assertEqual(panel.latest_epoch, 5)
        self.assertEqual(panel.best_epoch, 5)
        self.assertIn("train_distance_loss=0.2300", panel.text)
        self.assertIn("train_orientation_loss=0.1100", panel.text)
        self.assertIn("val_distance_loss=0.2000", panel.text)
        self.assertIn("val_orientation_loss=0.0900", panel.text)
        self.assertIn("yaw_mean_error_deg=6.7000", panel.text)
        self.assertIn("yaw_acc@10deg=0.7500", panel.text)

    def _write_config(self, run_dir: Path, task_contract: dict) -> None:
        (run_dir / "config.json").write_text(
            json.dumps({"task_contract": task_contract}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
