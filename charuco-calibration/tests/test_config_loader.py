from __future__ import annotations

import pytest

from rb_camera_calibration.io.config_loader import (
    build_session_config,
    default_calibration_runs_root,
    load_board_config,
    load_camera_config,
)


def test_load_board_config(tmp_path) -> None:
    path = tmp_path / "board.toml"
    path.write_text(
        """
[board]
pattern_type = "charuco"
squares_x = 15
squares_y = 10
square_length_m = 0.015
marker_length_m = 0.011
aruco_dictionary = "DICT_5X5_100"
board_name = "test_board"
""",
        encoding="utf-8",
    )

    config = load_board_config(path)
    assert config.squares_x == 15
    assert config.squares_y == 10
    assert config.aruco_dictionary == "DICT_5X5_100"


def test_load_board_config_rejects_invalid_marker_size(tmp_path) -> None:
    path = tmp_path / "board.toml"
    path.write_text(
        """
[board]
squares_x = 15
squares_y = 10
square_length_m = 0.015
marker_length_m = 0.020
aruco_dictionary = "DICT_5X5_100"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="marker_length_m must be smaller"):
        load_board_config(path)


def test_load_camera_config(tmp_path) -> None:
    path = tmp_path / "camera.toml"
    path.write_text(
        """
[camera]
camera_source_type = "opencv_v4l2"
camera_device = "/dev/video0"
width_px = 960
height_px = 600
fps = 80
pixel_format = "YUYV"
backend = "V4L2"
""",
        encoding="utf-8",
    )

    config = load_camera_config(path)
    assert config.width_px == 960
    assert config.height_px == 600
    assert config.camera_device == "/dev/video0"


def test_default_session_root_is_package_local(tmp_path) -> None:
    board_path = tmp_path / "board.toml"
    board_path.write_text(
        """
[board]
pattern_type = "charuco"
squares_x = 10
squares_y = 15
square_length_m = 0.015
marker_length_m = 0.011
aruco_dictionary = "DICT_4X4_100"
board_name = "test_board"
""",
        encoding="utf-8",
    )
    camera_path = tmp_path / "camera.toml"
    camera_path.write_text(
        """
[camera]
camera_source_type = "opencv_v4l2"
camera_device = "/dev/video0"
width_px = 1920
height_px = 1200
fps = 50
pixel_format = "YUYV"
backend = "V4L2"
""",
        encoding="utf-8",
    )

    session = build_session_config(
        board_config_path=board_path,
        camera_config_path=camera_path,
    )

    assert session.session_root.parent == default_calibration_runs_root()
