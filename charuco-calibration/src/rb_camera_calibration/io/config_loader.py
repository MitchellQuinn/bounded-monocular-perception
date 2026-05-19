"""TOML configuration loading for the calibration app."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Mapping

from rb_camera_calibration.contracts import (
    CalibrationPatternType,
    CalibrationSessionConfig,
    CameraCaptureConfig,
    CameraSourceType,
    CapturePolicyConfig,
    CharucoBoardConfig,
)
from rb_camera_calibration.utils.timestamps import utc_run_slug


def default_calibration_runs_root() -> Path:
    """Return the package-local default run directory.

    Resolve from this file rather than the process working directory so the app
    consistently writes to ``<CHARUCO_CALIBRATION_DIR>/calibration_runs``.
    """
    return Path(__file__).resolve().parents[3] / "calibration_runs"


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML in {path}: {exc}") from exc


def _section(data: Mapping[str, Any], name: str, path: Path) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a [{name}] section.")
    return value


def _enum_value(enum_type: type, value: Any, field_name: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ValueError(f"{field_name} must be one of: {choices}. Got {value!r}.") from exc


def load_board_config(path: str | Path) -> CharucoBoardConfig:
    """Load and validate a ChArUco board config."""
    resolved = Path(path)
    board = _section(_load_toml(resolved), "board", resolved)
    return CharucoBoardConfig(
        pattern_type=_enum_value(
            CalibrationPatternType,
            board.get("pattern_type", CalibrationPatternType.CHARUCO.value),
            "board.pattern_type",
        ),
        squares_x=int(board["squares_x"]),
        squares_y=int(board["squares_y"]),
        square_length_m=float(board["square_length_m"]),
        marker_length_m=float(board["marker_length_m"]),
        aruco_dictionary=str(board["aruco_dictionary"]),
        board_name=str(board.get("board_name", "")),
        extras={str(k): v for k, v in board.items() if k not in {
            "pattern_type",
            "squares_x",
            "squares_y",
            "square_length_m",
            "marker_length_m",
            "aruco_dictionary",
            "board_name",
        }},
    )


def load_camera_config(path: str | Path) -> CameraCaptureConfig:
    """Load and validate an OpenCV camera config."""
    resolved = Path(path)
    camera = _section(_load_toml(resolved), "camera", resolved)
    return CameraCaptureConfig(
        camera_source_type=_enum_value(
            CameraSourceType,
            camera.get("camera_source_type", CameraSourceType.OPENCV_V4L2.value),
            "camera.camera_source_type",
        ),
        camera_device=camera.get("camera_device", "/dev/video0"),
        width_px=int(camera.get("width_px", 1920)),
        height_px=int(camera.get("height_px", 1200)),
        fps=float(camera.get("fps", 80)),
        pixel_format=str(camera.get("pixel_format", "YUYV")),
        backend=str(camera.get("backend", "V4L2")),
        exposure=_optional_float(camera.get("exposure")),
        gain=_optional_float(camera.get("gain")),
        focus=_optional_float(camera.get("focus")),
        auto_exposure=_optional_bool(camera.get("auto_exposure")),
        auto_white_balance=_optional_bool(camera.get("auto_white_balance")),
        extras={str(k): v for k, v in camera.items() if k not in {
            "camera_source_type",
            "camera_device",
            "width_px",
            "height_px",
            "fps",
            "pixel_format",
            "backend",
            "exposure",
            "gain",
            "focus",
            "auto_exposure",
            "auto_white_balance",
        }},
    )


def load_capture_policy_config(path: str | Path | None) -> CapturePolicyConfig:
    """Load a capture policy config, or return defaults when omitted."""
    if path is None:
        return CapturePolicyConfig()
    resolved = Path(path)
    policy = _section(_load_toml(resolved), "capture_policy", resolved)
    return CapturePolicyConfig(
        target_accepted_frame_count=int(policy.get("target_accepted_frame_count", 45)),
        min_marker_count=int(policy.get("min_marker_count", 8)),
        min_charuco_corner_count=int(policy.get("min_charuco_corner_count", 24)),
        min_board_area_fraction=float(policy.get("min_board_area_fraction", 0.03)),
        max_board_area_fraction=float(policy.get("max_board_area_fraction", 0.85)),
        min_edge_margin_px=float(policy.get("min_edge_margin_px", 12)),
        min_laplacian_variance=float(policy.get("min_laplacian_variance", 80.0)),
        cooldown_seconds=float(policy.get("cooldown_seconds", 1.5)),
        require_pose_novelty=bool(policy.get("require_pose_novelty", True)),
        require_stability=bool(policy.get("require_stability", True)),
        stability_window_frames=int(policy.get("stability_window_frames", 3)),
        pose_grid_cols=int(policy.get("pose_grid_cols", 3)),
        pose_grid_rows=int(policy.get("pose_grid_rows", 3)),
        scale_bin_count=int(policy.get("scale_bin_count", 3)),
        tilt_bin_count=int(policy.get("tilt_bin_count", 2)),
        extras={str(k): v for k, v in policy.items() if k not in {
            "target_accepted_frame_count",
            "min_marker_count",
            "min_charuco_corner_count",
            "min_board_area_fraction",
            "max_board_area_fraction",
            "min_edge_margin_px",
            "min_laplacian_variance",
            "cooldown_seconds",
            "require_pose_novelty",
            "require_stability",
            "stability_window_frames",
            "pose_grid_cols",
            "pose_grid_rows",
            "scale_bin_count",
            "tilt_bin_count",
        }},
    )


def build_session_config(
    *,
    board_config_path: str | Path,
    camera_config_path: str | Path,
    capture_policy_path: str | Path | None = None,
    session_root: str | Path | None = None,
) -> CalibrationSessionConfig:
    """Build a complete session config from individual config files."""
    board_config = load_board_config(board_config_path)
    camera_config = load_camera_config(camera_config_path)
    capture_policy = load_capture_policy_config(capture_policy_path)
    root = Path(session_root) if session_root is not None else (
        default_calibration_runs_root() / f"{utc_run_slug()}_{board_config.board_name or 'charuco'}"
    )
    return CalibrationSessionConfig(
        session_root=root,
        board_config=board_config,
        camera_config=camera_config,
        capture_policy=capture_policy,
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)
