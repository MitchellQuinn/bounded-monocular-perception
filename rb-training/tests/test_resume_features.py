"""Tests for resume-training state and control helpers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.resume.control_panel import (
    build_resume_launch_command,
    latest_resumable_candidate,
    list_resume_candidates,
)
from src.resume.state import RESUME_STATE_FILENAME, load_resume_state, save_resume_state


class ResumeStateTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        with TemporaryDirectory() as tmp:
            state_path = Path(tmp) / RESUME_STATE_FILENAME
            payload = {
                "format_version": 1,
                "epoch": 7,
                "run_id": "run_0003",
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "lr_scheduler_state_dict": None,
                "best_epoch": 5,
                "best_val_loss": 0.123,
                "no_improvement_epochs": 2,
                "history_records": [{"epoch": 1, "val_loss": 0.5}],
            }

            save_resume_state(state_path, payload)
            loaded = load_resume_state(state_path)

            self.assertEqual(int(loaded["epoch"]), 7)
            self.assertEqual(str(loaded["run_id"]), "run_0003")
            self.assertEqual(float(loaded["best_val_loss"]), 0.123)

    def test_load_rejects_missing_required_keys(self) -> None:
        with TemporaryDirectory() as tmp:
            state_path = Path(tmp) / RESUME_STATE_FILENAME
            save_resume_state(state_path, {"epoch": 1})
            with self.assertRaisesRegex(ValueError, "missing required keys"):
                load_resume_state(state_path)


class ResumeControlPanelTests(unittest.TestCase):
    def test_resume_candidate_discovery_and_command_build(self) -> None:
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            models_root = repo_root / "models"
            model_directory = "260401-1200_2d-cnn"
            run_id = "run_0001"

            model_dir = models_root / model_directory
            run_dir = model_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "model_name": model_directory,
                        "model_architecture_variant": "fast_v0_2",
                        "seed": 42,
                        "batch_size": 8,
                        "epochs": 8,
                        "learning_rate": 1e-3,
                        "weight_decay": 1e-5,
                        "huber_delta": 1.0,
                        "early_stopping_patience": 3,
                        "padding_mode": "disabled",
                        "progress_log_interval_batches": 250,
                        "accuracy_tolerance_m": 0.10,
                        "extra_accuracy_tolerances_m": [0.25, 0.50],
                        "enable_lr_scheduler": True,
                        "lr_scheduler_factor": 0.5,
                        "lr_scheduler_patience": 1,
                        "lr_scheduler_min_lr": 1e-5,
                        "train_cache_budget_gb": 48.0,
                        "train_shuffle_mode": "shard",
                        "train_active_shard_count": 3,
                        "cache_validation_in_ram": True,
                        "validation_cache_budget_gb": 40.0,
                        "enable_internal_test_split": False,
                        "internal_test_fraction": 0.1,
                        "training_data_root_resolved": str(repo_root / "training-data"),
                        "validation_data_root_resolved": str(repo_root / "validation-data"),
                    }
                ),
                encoding="utf-8",
            )

            save_resume_state(
                run_dir / RESUME_STATE_FILENAME,
                {
                    "format_version": 1,
                    "epoch": 5,
                    "run_id": run_id,
                    "model_state_dict": {},
                    "optimizer_state_dict": {},
                    "lr_scheduler_state_dict": None,
                    "best_epoch": 4,
                    "best_val_loss": 0.15,
                    "no_improvement_epochs": 1,
                    "history_records": [{"epoch": 1}],
                },
            )

            (model_dir / "run_register.json").write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "run_id": run_id,
                                "run_dir": f"models/{model_directory}/runs/{run_id}",
                                "session_name": "rb-test-run_0001",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            candidates = list_resume_candidates(models_root=models_root, model_directory=model_directory)
            self.assertEqual(len(candidates), 1)
            self.assertTrue(bool(candidates[0]["is_resumable"]))
            self.assertEqual(int(candidates[0]["last_completed_epoch"]), 5)

            latest = latest_resumable_candidate(models_root=models_root, model_directory=model_directory)
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(str(latest["run_id"]), run_id)

            command = build_resume_launch_command(
                run_id="run_0002",
                model_directory=model_directory,
                source_run_dir=run_dir,
                additional_epochs=3,
                python_executable="python",
                training_module="src.train",
                output_root=models_root,
                change_note="resume test",
            )
            self.assertIn("--resume-from-run-dir", command)
            self.assertIn("--additional-epochs 3", command)
            self.assertIn("--run-id run_0002", command)


if __name__ == "__main__":
    unittest.main()
