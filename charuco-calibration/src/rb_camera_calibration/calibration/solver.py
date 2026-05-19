"""Camera calibration solver using matched ChArUco image/object points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rb_camera_calibration.calibration.reprojection import compute_reprojection_error
from rb_camera_calibration.contracts import (
    CalibrationRequest,
    CalibrationResult,
    PerViewCalibrationError,
)
from rb_camera_calibration.utils import opencv_compat as cvx
from rb_camera_calibration.utils.timestamps import utc_now_iso


class OpenCvCalibrationSolver:
    """Solve camera intrinsics from accepted ChArUco detections."""

    def solve(self, request: CalibrationRequest) -> CalibrationResult:
        cv2 = cvx.import_cv2()
        import numpy as np

        board = cvx.create_charuco_board(request.board_config)
        object_points: list[Any] = []
        image_points: list[Any] = []
        frame_ids: list[str] = []
        rejected = 0
        errors: list[PerViewCalibrationError] = []

        for accepted in request.accepted_frames:
            if accepted.extras.get("include_in_calibration", True) is False:
                rejected += 1
                continue
            detection_payload = _load_detection_payload(accepted.detection_json_path)
            extras = detection_payload.get("extras", {})
            corners_xy = extras.get("charuco_corners_xy", [])
            charuco_ids = extras.get("charuco_ids", detection_payload.get("charuco_ids", []))
            if len(corners_xy) < 4 or len(charuco_ids) < 4:
                rejected += 1
                continue
            corners = cvx.corners_from_list(corners_xy)
            ids = cvx.ids_from_list(charuco_ids)
            current_obj, current_img = board.matchImagePoints(corners, ids)
            if int(np.asarray(current_img).reshape(-1, 2).shape[0]) < 4:
                rejected += 1
                continue
            object_points.append(current_obj)
            image_points.append(current_img)
            frame_ids.append(accepted.frame_id)

        flags = _calibration_flags_to_int(tuple(request.calibration_flags))
        if len(object_points) < 3:
            return _failed_result(
                request,
                cv2.__version__,
                rejected,
                f"Need at least 3 usable accepted frames; found {len(object_points)}.",
            )

        try:
            rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points,
                tuple(int(v) for v in request.image_size_wh_px),
                None,
                None,
                flags=flags,
            )
        except cv2.error as exc:
            return _failed_result(request, cv2.__version__, rejected, str(exc))

        for frame_id, obj, img, rvec, tvec in zip(frame_ids, object_points, image_points, rvecs, tvecs):
            errors.append(
                compute_reprojection_error(
                    frame_id=frame_id,
                    object_points=obj,
                    image_points=img,
                    rvec=rvec,
                    tvec=tvec,
                    camera_matrix=camera_matrix,
                    distortion_coefficients=dist_coeffs,
                )
            )

        return CalibrationResult(
            success=True,
            rms_reprojection_error_px=float(rms),
            camera_matrix=_matrix3_tuple(camera_matrix),
            distortion_coefficients=tuple(float(v) for v in np.asarray(dist_coeffs).reshape(-1)),
            image_size_wh_px=request.image_size_wh_px,
            board_config=request.board_config,
            accepted_frame_count=len(request.accepted_frames),
            used_frame_count=len(object_points),
            rejected_outlier_count=rejected,
            per_view_errors=tuple(errors),
            generated_at_utc=utc_now_iso(),
            opencv_version=str(cv2.__version__),
            calibration_flags=tuple(request.calibration_flags),
        )


def _load_detection_payload(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Accepted frame detection JSON is missing: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Detection JSON must contain an object: {path}")
    return payload


def _calibration_flags_to_int(flag_names: tuple[str, ...]) -> int:
    cv2 = cvx.import_cv2()
    flags = 0
    for name in flag_names:
        if not name:
            continue
        normalized = name if name.startswith("CALIB_") else f"CALIB_{name}"
        if not hasattr(cv2, normalized):
            raise ValueError(f"Unknown OpenCV calibration flag: {name!r}.")
        flags |= int(getattr(cv2, normalized))
    return flags


def _matrix3_tuple(matrix: object) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    import numpy as np

    arr = np.asarray(matrix, dtype=float).reshape(3, 3)
    return tuple(tuple(float(value) for value in row) for row in arr)  # type: ignore[return-value]


def _failed_result(
    request: CalibrationRequest,
    opencv_version: str,
    rejected_count: int,
    message: str,
) -> CalibrationResult:
    return CalibrationResult(
        success=False,
        rms_reprojection_error_px=0.0,
        camera_matrix=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        distortion_coefficients=(),
        image_size_wh_px=request.image_size_wh_px,
        board_config=request.board_config,
        accepted_frame_count=len(request.accepted_frames),
        used_frame_count=0,
        rejected_outlier_count=rejected_count,
        per_view_errors=(),
        generated_at_utc=utc_now_iso(),
        opencv_version=opencv_version,
        calibration_flags=tuple(request.calibration_flags),
        extras={"error": message},
    )
