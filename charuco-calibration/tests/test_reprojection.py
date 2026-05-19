from __future__ import annotations

from rb_camera_calibration.calibration.reprojection import compute_reprojection_error


def test_reprojection_error_zero_for_known_projected_points() -> None:
    import cv2
    import numpy as np

    object_points = np.array(
        [[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[1.0, 1.0, 0.0]]],
        dtype=np.float32,
    )
    camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros((5, 1))
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [5.0]])
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist)

    error = compute_reprojection_error(
        frame_id="frame-1",
        object_points=object_points,
        image_points=image_points,
        rvec=rvec,
        tvec=tvec,
        camera_matrix=camera_matrix,
        distortion_coefficients=dist,
    )

    assert error.rms_error_px == 0.0
    assert error.point_count == 4
