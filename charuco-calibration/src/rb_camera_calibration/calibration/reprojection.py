"""Reprojection error diagnostics."""

from __future__ import annotations

from rb_camera_calibration.contracts import PerViewCalibrationError
from rb_camera_calibration.utils import opencv_compat as cvx


def compute_reprojection_error(
    *,
    frame_id: str,
    object_points: object,
    image_points: object,
    rvec: object,
    tvec: object,
    camera_matrix: object,
    distortion_coefficients: object,
    include_in_final: bool = True,
) -> PerViewCalibrationError:
    """Compute RMS/mean/max reprojection error for one calibration view."""
    cv2 = cvx.import_cv2()
    import numpy as np

    projected, _jacobian = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        distortion_coefficients,
    )
    observed = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    predicted = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    if observed.shape != predicted.shape:
        raise ValueError(
            f"Observed/projected point shape mismatch: {observed.shape} != {predicted.shape}."
        )
    if observed.size == 0:
        return PerViewCalibrationError(
            frame_id=frame_id,
            rms_error_px=0.0,
            mean_error_px=0.0,
            max_error_px=0.0,
            point_count=0,
            include_in_final=include_in_final,
        )
    errors = np.linalg.norm(observed - predicted, axis=1)
    rms = float(np.sqrt(np.mean(errors**2)))
    return PerViewCalibrationError(
        frame_id=frame_id,
        rms_error_px=rms,
        mean_error_px=float(np.mean(errors)),
        max_error_px=float(np.max(errors)),
        point_count=int(errors.shape[0]),
        include_in_final=include_in_final,
    )
