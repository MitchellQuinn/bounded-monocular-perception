"""Launch module for the PySide6 ChArUco calibration app."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from rb_camera_calibration.gui.main_window import MainWindow
from rb_camera_calibration.io.config_loader import build_session_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raccoon Ball ChArUco camera calibration GUI.")
    parser.add_argument("--board-config", required=True, help="Path to board TOML config.")
    parser.add_argument("--camera-config", required=True, help="Path to camera TOML config.")
    parser.add_argument(
        "--capture-policy",
        default=None,
        help="Optional path to capture policy TOML config.",
    )
    parser.add_argument(
        "--session-root",
        default=None,
        help=(
            "Optional calibration run directory. Existing manifests are loaded so a "
            "session can be resumed. Defaults under charuco-calibration/calibration_runs/."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    session_config = build_session_config(
        board_config_path=Path(args.board_config),
        camera_config_path=Path(args.camera_config),
        capture_policy_path=Path(args.capture_policy) if args.capture_policy else None,
        session_root=Path(args.session_root) if args.session_root else None,
    )
    app = QApplication(sys.argv[:1])
    window = MainWindow(session_config)
    window.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover - GUI entry
    raise SystemExit(main())
