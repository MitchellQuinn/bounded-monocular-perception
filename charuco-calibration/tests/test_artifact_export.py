from __future__ import annotations

import json

from rb_camera_calibration.calibration.artifact_export import CalibrationArtifactExporter
from rb_camera_calibration.contracts import (
    CalibrationResult,
    CalibrationSessionConfig,
    CameraCaptureConfig,
    CapturePolicyConfig,
    CharucoBoardConfig,
    PerViewCalibrationError,
)


def test_artifact_export_writes_json_compatible_output(tmp_path) -> None:
    board = CharucoBoardConfig(
        squares_x=15,
        squares_y=10,
        square_length_m=0.015,
        marker_length_m=0.011,
        aruco_dictionary="DICT_5X5_100",
    )
    session = CalibrationSessionConfig(
        session_root=tmp_path,
        board_config=board,
        camera_config=CameraCaptureConfig(camera_device="/dev/video0"),
        capture_policy=CapturePolicyConfig(),
    )
    result = CalibrationResult(
        success=True,
        rms_reprojection_error_px=0.25,
        camera_matrix=((1.0, 0.0, 10.0), (0.0, 1.0, 20.0), (0.0, 0.0, 1.0)),
        distortion_coefficients=(0.1, 0.01, 0.0, 0.0, 0.0),
        image_size_wh_px=(960, 600),
        board_config=board,
        accepted_frame_count=3,
        used_frame_count=3,
        rejected_outlier_count=0,
        per_view_errors=(
            PerViewCalibrationError("frame-1", 0.2, 0.18, 0.3, 24),
        ),
        generated_at_utc="2026-05-19T00:00:00Z",
        opencv_version="4.x",
    )

    manifest = CalibrationArtifactExporter().export(result, session)

    payload = json.loads(manifest.result_json_path.read_text(encoding="utf-8"))
    assert payload["board"]["type"] == "charuco"
    assert payload["camera_matrix"][0][2] == 10.0
    assert payload["distortion_coefficients"] == [0.1, 0.01, 0.0, 0.0, 0.0]
    assert manifest.result_yaml_path.exists()
    assert manifest.report_csv_path.exists()
