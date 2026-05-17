from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import shlex
import subprocess
import unittest
from unittest.mock import patch

from _test_support import ensure_training_root
from roi_fcn_training_v0_1.paths import PROJECT_TIMEZONE, suggest_model_run_id
from roi_fcn_training_v0_1.resume_state import build_resume_state_payload, save_resume_state
from roi_fcn_training_v0_1.tmux_launcher_v0_2 import (
    build_tmux_log_path,
    default_session_name,
    end_session,
    launch_session,
    latest_resume_candidate,
    list_model_directories,
    list_resume_candidates,
    plan_tmux_resume_launch,
    plan_tmux_training_launch,
    read_log_tail,
    resolve_session_run_paths,
)


class TmuxLauncherV02Tests(unittest.TestCase):
    def test_suggest_model_run_id_uses_shared_timestamp_naming(self) -> None:
        model_directory = suggest_model_run_id(
            "roi-fcn-tiny",
            run_name_suffix="tmux",
            now_local=datetime(2026, 4, 19, 17, 44, tzinfo=PROJECT_TIMEZONE),
        )
        self.assertEqual(model_directory, "260419-1744_roi-fcn-tiny_tmux")

    def test_default_session_name_uses_model_directory_and_run_id(self) -> None:
        self.assertEqual(
            default_session_name("260420-1024_roi-fcn-tiny", "run_0001"),
            "roi_fcn_260420-1024_roi-fcn-tiny_run_0001",
        )

    def test_plan_tmux_training_launch_builds_expected_paths_and_command(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            plan = plan_tmux_training_launch(
                root,
                {
                    "training_dataset": "fixture_train",
                    "validation_dataset": "fixture_validate",
                    "model_name": "roi-fcn-tiny",
                    "model_directory": "260420-1024_roi-fcn-tiny",
                    "run_id": "run_0007",
                    "topology_id": "roi_fcn_tiny",
                    "topology_variant": "tiny_v1",
                    "batch_size": 8,
                    "epochs": 12,
                    "progress_log_interval_steps": 25,
                    "roi_width_px": 300,
                    "roi_height_px": 220,
                    "device": "cuda",
                },
                python_executable="/tmp/fake-python",
                session_name="roi_fcn_260420-1024_roi-fcn-tiny_run_0007",
            )

            expected_run_dir = root / "models" / "260420-1024_roi-fcn-tiny" / "runs" / "run_0007"
            expected_log_path = expected_run_dir / "train.log"
            self.assertEqual(plan.model_directory, "260420-1024_roi-fcn-tiny")
            self.assertEqual(Path(plan.run_dir), expected_run_dir.resolve())
            self.assertEqual(Path(plan.log_path), expected_log_path.resolve())
            self.assertEqual(Path(plan.working_directory), (root / "src").resolve())
            self.assertEqual(plan.session_name, "roi_fcn_260420-1024_roi-fcn-tiny_run_0007")

            tokens = shlex.split(plan.command)
            self.assertEqual(tokens[:4], ["/tmp/fake-python", "-u", "-m", "roi_fcn_training_v0_1.train"])
            self.assertIn("--training-dataset", tokens)
            self.assertIn("fixture_train", tokens)
            self.assertIn("--validation-dataset", tokens)
            self.assertIn("fixture_validate", tokens)
            self.assertIn("--model-directory", tokens)
            self.assertIn("260420-1024_roi-fcn-tiny", tokens)
            self.assertIn("--run-id", tokens)
            self.assertIn("run_0007", tokens)
            self.assertIn("--device", tokens)
            self.assertIn("cuda", tokens)

    def test_plan_tmux_training_launch_suggests_next_run_id_for_existing_model_directory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            existing_run_dir = root / "models" / "260420-1024_roi-fcn-tiny" / "runs" / "run_0001"
            existing_run_dir.mkdir(parents=True, exist_ok=False)

            plan = plan_tmux_training_launch(
                root,
                {
                    "training_dataset": "fixture_train",
                    "model_name": "roi-fcn-tiny",
                    "model_directory": "260420-1024_roi-fcn-tiny",
                },
                python_executable="/tmp/fake-python",
            )

            self.assertEqual(plan.run_id, "run_0002")
            self.assertEqual(
                Path(plan.run_dir),
                (root / "models" / "260420-1024_roi-fcn-tiny" / "runs" / "run_0002").resolve(),
            )

    def test_resume_candidate_discovery_and_plan_resume_launch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            models_root = root / "models"
            model_directory = "260420-1024_roi-fcn-tiny"
            source_run_dir = models_root / model_directory / "runs" / "run_0001"
            source_run_dir.mkdir(parents=True, exist_ok=False)
            (source_run_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "training_dataset": "fixture_train",
                        "validation_dataset": "fixture_validate",
                        "datasets_root": "datasets",
                        "models_root": "models",
                        "seed": 42,
                        "batch_size": 8,
                        "epochs": 12,
                        "learning_rate": 1e-3,
                        "weight_decay": 1e-5,
                        "gaussian_sigma_px": 2.5,
                        "heatmap_positive_threshold": 0.05,
                        "early_stopping_patience": 4,
                        "topology_id": "roi_fcn_tiny",
                        "topology_variant": "tiny_v1",
                        "topology_params": {},
                        "model_name": "roi-fcn-tiny",
                        "model_directory": model_directory,
                        "progress_log_interval_steps": 25,
                        "roi_width_px": 300,
                        "roi_height_px": 220,
                        "evaluation_max_visual_examples": 6,
                    }
                ) + "\n",
                encoding="utf-8",
            )
            save_resume_state(
                source_run_dir / "resume_state.pt",
                build_resume_state_payload(
                    epoch=5,
                    run_id="run_0001",
                    training_dataset="fixture_train",
                    validation_dataset="fixture_validate",
                    topology_id="roi_fcn_tiny",
                    topology_variant="tiny_v1",
                    topology_params={},
                    topology_spec_signature="sig-spec",
                    topology_contract_signature="sig-contract",
                    output_hw=(16, 24),
                    train_split_contract={
                        "dataset_reference": "fixture_train",
                        "split_name": "train",
                        "row_count": 10,
                        "shard_count": 1,
                        "geometry": {
                            "canvas_width_px": 48,
                            "canvas_height_px": 32,
                            "image_layout": "N,C,H,W",
                            "channels": 1,
                            "normalization_range": [0.0, 1.0],
                            "geometry_schema": ["schema"],
                        },
                        "preprocessing_contract_version": "rb-preprocess-roi-fcn-v0_1",
                        "representation_kind": "roi_fcn_locator_npz",
                        "representation_storage_format": "npz",
                        "representation_array_keys": ["locator_input_image"],
                        "bootstrap_bbox_available": True,
                        "fixed_roi_width_px": 300,
                        "fixed_roi_height_px": 220,
                    },
                    validation_split_contract={
                        "dataset_reference": "fixture_validate",
                        "split_name": "validate",
                        "row_count": 4,
                        "shard_count": 1,
                        "geometry": {
                            "canvas_width_px": 48,
                            "canvas_height_px": 32,
                            "image_layout": "N,C,H,W",
                            "channels": 1,
                            "normalization_range": [0.0, 1.0],
                            "geometry_schema": ["schema"],
                        },
                        "preprocessing_contract_version": "rb-preprocess-roi-fcn-v0_1",
                        "representation_kind": "roi_fcn_locator_npz",
                        "representation_storage_format": "npz",
                        "representation_array_keys": ["locator_input_image"],
                        "bootstrap_bbox_available": True,
                        "fixed_roi_width_px": 300,
                        "fixed_roi_height_px": 220,
                    },
                    best_epoch=4,
                    best_validation_loss=0.25,
                    best_validation_mean_center_error_px=3.5,
                    epochs_without_improvement=1,
                    history_rows=[{"epoch": 1}],
                    model_state_dict={},
                    optimizer_state_dict={},
                ),
            )

            self.assertEqual(list_model_directories(models_root), [model_directory])
            candidates = list_resume_candidates(models_root, model_directory=model_directory)
            self.assertEqual(len(candidates), 1)
            self.assertTrue(bool(candidates[0]["is_resumable"]))
            self.assertEqual(int(candidates[0]["completed_epochs"]), 5)
            latest = latest_resume_candidate(models_root, model_directory=model_directory)
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(str(latest["run_id"]), "run_0001")

            plan = plan_tmux_resume_launch(
                root,
                source_run_dir=source_run_dir,
                additional_epochs=3,
                python_executable="/tmp/fake-python",
                device_override="cuda",
            )

            self.assertEqual(plan.model_directory, model_directory)
            self.assertEqual(plan.run_id, "run_0002")
            tokens = shlex.split(plan.command)
            self.assertIn("--resume-from-run-dir", tokens)
            self.assertIn(str(source_run_dir.resolve()), tokens)
            self.assertIn("--additional-epochs", tokens)
            self.assertIn("3", tokens)
            self.assertIn("--device", tokens)
            self.assertIn("cuda", tokens)

    def test_plan_tmux_training_launch_fails_when_predicted_run_dir_exists(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            existing_run_dir = root / "models" / "260420-1024_roi-fcn-tiny" / "runs" / "run_0002"
            existing_run_dir.mkdir(parents=True, exist_ok=False)
            with self.assertRaisesRegex(FileExistsError, "Run directory already exists"):
                plan_tmux_training_launch(
                    root,
                    {
                        "training_dataset": "fixture_train",
                        "model_name": "roi-fcn-tiny",
                        "model_directory": "260420-1024_roi-fcn-tiny",
                        "run_id": "run_0002",
                    },
                    python_executable="/tmp/fake-python",
                )

    def test_build_tmux_log_path_is_inside_run_directory(self) -> None:
        path = build_tmux_log_path(
            "/tmp/models",
            model_directory="260420-1024_roi-fcn-tiny",
            run_id="run_0003",
        )
        self.assertEqual(
            path,
            Path("/tmp/models").resolve() / "260420-1024_roi-fcn-tiny" / "runs" / "run_0003" / "train.log",
        )

    def test_launch_session_creates_log_and_invokes_tmux(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "train.log"
            with patch("roi_fcn_training_v0_1.tmux_launcher_v0_2.session_exists", return_value=False):
                with patch(
                    "roi_fcn_training_v0_1.tmux_launcher_v0_2._run_tmux",
                    return_value=subprocess.CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
                ) as mocked_run_tmux:
                    payload = launch_session(
                        "roi_fcn_260420-1024_roi-fcn-tiny_run_0004",
                        "python -u -m roi_fcn_training_v0_1.train --training-dataset fixture",
                        log_path,
                        working_directory="/tmp",
                    )

            self.assertTrue(log_path.exists())
            self.assertEqual(payload["session_name"], "roi_fcn_260420-1024_roi-fcn-tiny_run_0004")
            self.assertEqual(payload["log_path"], str(log_path.resolve()))
            tmux_args = mocked_run_tmux.call_args.args[0]
            self.assertEqual(tmux_args[:4], ["new-session", "-d", "-s", "roi_fcn_260420-1024_roi-fcn-tiny_run_0004"])
            self.assertIn(">>", tmux_args[-1])

    def test_end_session_returns_false_when_missing(self) -> None:
        with patch(
            "roi_fcn_training_v0_1.tmux_launcher_v0_2._run_tmux",
            return_value=subprocess.CompletedProcess(args=["tmux"], returncode=1, stdout="", stderr="can't find session"),
        ):
            self.assertFalse(end_session("roi_fcn_260420-1024_roi-fcn-tiny_run_0005"))

    def test_end_session_kills_without_list_session_precheck(self) -> None:
        with patch(
            "roi_fcn_training_v0_1.tmux_launcher_v0_2._run_tmux",
            return_value=subprocess.CompletedProcess(args=["tmux"], returncode=0, stdout="", stderr=""),
        ) as mocked_run_tmux:
            self.assertTrue(end_session("roi_fcn_260420-1024_roi-fcn-tiny_run_0006"))
        self.assertEqual(
            mocked_run_tmux.call_args.args[0],
            ["kill-session", "-t", "roi_fcn_260420-1024_roi-fcn-tiny_run_0006"],
        )

    def test_resolve_session_run_paths_reads_running_pane_command(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            session_name = "roi_fcn_260420-1024_roi-fcn-tiny_run_0007"
            log_path = root / "models" / "260420-1024_roi-fcn-tiny" / "runs" / "run_0007" / "train.log"
            pane_command = (
                f'"/tmp/fake-python -u -m roi_fcn_training_v0_1.train '
                f'--models-root models --model-directory 260420-1024_roi-fcn-tiny --run-id run_0007 '
                f'>> {log_path} 2>&1"'
            )
            with patch(
                "roi_fcn_training_v0_1.tmux_launcher_v0_2._run_tmux",
                return_value=subprocess.CompletedProcess(
                    args=["tmux"],
                    returncode=0,
                    stdout=f"{root / 'src'}\t{pane_command}\n",
                    stderr="",
                ),
            ):
                info = resolve_session_run_paths(root, session_name)

        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info["model_directory"], "260420-1024_roi-fcn-tiny")
        self.assertEqual(info["run_id"], "run_0007")
        self.assertEqual(Path(info["run_dir"]), log_path.parent.resolve())
        self.assertEqual(Path(info["log_path"]), log_path.resolve())

    def test_launch_session_removes_empty_run_dir_when_tmux_start_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "260420-1024_roi-fcn-tiny" / "runs" / "run_0005" / "train.log"
            with patch("roi_fcn_training_v0_1.tmux_launcher_v0_2.session_exists", return_value=False):
                with patch(
                    "roi_fcn_training_v0_1.tmux_launcher_v0_2._run_tmux",
                    return_value=subprocess.CompletedProcess(args=["tmux"], returncode=1, stdout="", stderr="boom"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "tmux new-session failed: boom"):
                        launch_session(
                            "roi_fcn_260420-1024_roi-fcn-tiny_run_0005",
                            "python -u -m roi_fcn_training_v0_1.train --training-dataset fixture",
                            log_path,
                            working_directory="/tmp",
                        )

            self.assertFalse(log_path.exists())
            self.assertFalse(log_path.parent.exists())

    def test_read_log_tail_returns_last_lines(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "train.log"
            log_path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
            self.assertEqual(read_log_tail(log_path, max_lines=2), "three\nfour\n")


if __name__ == "__main__":
    unittest.main()
