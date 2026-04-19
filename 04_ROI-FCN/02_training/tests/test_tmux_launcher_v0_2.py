from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import shlex
import subprocess
import unittest
from unittest.mock import patch

from _test_support import ensure_training_root
from roi_fcn_training_v0_1.paths import PROJECT_TIMEZONE, suggest_model_run_id
from roi_fcn_training_v0_1.tmux_launcher_v0_2 import (
    TMUX_LOGS_DIRECTORY_NAME,
    build_tmux_log_path,
    end_session,
    launch_session,
    plan_tmux_training_launch,
    read_log_tail,
)


class TmuxLauncherV02Tests(unittest.TestCase):
    def test_suggest_model_run_id_uses_shared_timestamp_naming(self) -> None:
        run_id = suggest_model_run_id(
            "roi-fcn-tiny",
            run_name_suffix="tmux",
            now_local=datetime(2026, 4, 19, 17, 44, tzinfo=PROJECT_TIMEZONE),
        )
        self.assertEqual(run_id, "260419-1744_roi-fcn-tiny_tmux")

    def test_plan_tmux_training_launch_builds_expected_paths_and_command(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            plan = plan_tmux_training_launch(
                root,
                {
                    "training_dataset": "fixture_train",
                    "validation_dataset": "fixture_validate",
                    "model_name": "roi-fcn-tiny",
                    "run_id": "custom_run_0001",
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
                session_name="roi_fcn_custom_run_0001",
            )

            expected_run_dir = root / "models" / "roi-fcn-tiny" / "runs" / "custom_run_0001"
            expected_log_path = root / "models" / "roi-fcn-tiny" / TMUX_LOGS_DIRECTORY_NAME / "custom_run_0001__train.log"
            self.assertEqual(Path(plan.run_dir), expected_run_dir.resolve())
            self.assertEqual(Path(plan.log_path), expected_log_path.resolve())
            self.assertEqual(Path(plan.working_directory), (root / "src").resolve())
            self.assertEqual(plan.session_name, "roi_fcn_custom_run_0001")

            tokens = shlex.split(plan.command)
            self.assertEqual(tokens[:4], ["/tmp/fake-python", "-u", "-m", "roi_fcn_training_v0_1.train"])
            self.assertIn("--training-dataset", tokens)
            self.assertIn("fixture_train", tokens)
            self.assertIn("--validation-dataset", tokens)
            self.assertIn("fixture_validate", tokens)
            self.assertIn("--run-id", tokens)
            self.assertIn("custom_run_0001", tokens)
            self.assertIn("--device", tokens)
            self.assertIn("cuda", tokens)

    def test_plan_tmux_training_launch_fails_when_predicted_run_dir_exists(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            existing_run_dir = root / "models" / "roi-fcn-tiny" / "runs" / "custom_run_0002"
            existing_run_dir.mkdir(parents=True, exist_ok=False)
            with self.assertRaisesRegex(FileExistsError, "Run directory already exists"):
                plan_tmux_training_launch(
                    root,
                    {
                        "training_dataset": "fixture_train",
                        "model_name": "roi-fcn-tiny",
                        "run_id": "custom_run_0002",
                    },
                    python_executable="/tmp/fake-python",
                )

    def test_build_tmux_log_path_is_outside_run_directory(self) -> None:
        path = build_tmux_log_path(
            "/tmp/models",
            model_name="roi-fcn-tiny",
            run_id="custom_run_0003",
        )
        self.assertEqual(
            path,
            Path("/tmp/models").resolve() / "roi-fcn-tiny" / TMUX_LOGS_DIRECTORY_NAME / "custom_run_0003__train.log",
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
                        "roi_fcn_custom_run_0004",
                        "python -u -m roi_fcn_training_v0_1.train --training-dataset fixture",
                        log_path,
                        working_directory="/tmp",
                    )

            self.assertTrue(log_path.exists())
            self.assertEqual(payload["session_name"], "roi_fcn_custom_run_0004")
            self.assertEqual(payload["log_path"], str(log_path.resolve()))
            tmux_args = mocked_run_tmux.call_args.args[0]
            self.assertEqual(tmux_args[:4], ["new-session", "-d", "-s", "roi_fcn_custom_run_0004"])
            self.assertIn(">>", tmux_args[-1])

    def test_end_session_returns_false_when_missing(self) -> None:
        with patch("roi_fcn_training_v0_1.tmux_launcher_v0_2.session_exists", return_value=False):
            self.assertFalse(end_session("roi_fcn_custom_run_0005"))

    def test_read_log_tail_returns_last_lines(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "train.log"
            log_path.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
            self.assertEqual(read_log_tail(log_path, max_lines=2), "three\nfour\n")


if __name__ == "__main__":
    unittest.main()
