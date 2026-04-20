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
    build_tmux_log_path,
    default_session_name,
    end_session,
    launch_session,
    plan_tmux_training_launch,
    read_log_tail,
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
