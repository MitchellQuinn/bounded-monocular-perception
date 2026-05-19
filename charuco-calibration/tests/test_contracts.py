from __future__ import annotations

import json

import pytest

from rb_camera_calibration.contracts import (
    CAMERA_CALIBRATION_CONTRACT_VERSION,
    CalibrationPatternType,
    CharucoBoardConfig,
)


def test_contract_objects_serialize_cleanly() -> None:
    config = CharucoBoardConfig(
        pattern_type=CalibrationPatternType.CHARUCO,
        squares_x=15,
        squares_y=10,
        square_length_m=0.015,
        marker_length_m=0.011,
        aruco_dictionary="DICT_5X5_100",
        board_name="calibio_charuco_15mm_mdf",
    )

    payload = config.to_dict()
    assert payload["contract_version"] == CAMERA_CALIBRATION_CONTRACT_VERSION
    assert payload["pattern_type"] == "charuco"
    json.dumps(payload)


def test_board_config_validates_positive_lengths() -> None:
    with pytest.raises(ValueError, match="square_length_m must be positive"):
        CharucoBoardConfig(
            squares_x=15,
            squares_y=10,
            square_length_m=0.0,
            marker_length_m=0.011,
            aruco_dictionary="DICT_5X5_100",
        )


def test_marker_size_must_be_smaller_than_checker_size() -> None:
    with pytest.raises(ValueError, match="marker_length_m must be smaller"):
        CharucoBoardConfig(
            squares_x=15,
            squares_y=10,
            square_length_m=0.015,
            marker_length_m=0.015,
            aruco_dictionary="DICT_5X5_100",
        )
