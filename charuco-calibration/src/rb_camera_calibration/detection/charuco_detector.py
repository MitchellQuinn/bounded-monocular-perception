"""ChArUco detection implementation."""

from __future__ import annotations

import time
from dataclasses import replace

from rb_camera_calibration.contracts import (
    BoardDimensionDebugReport,
    CameraFrame,
    CharucoBoardConfig,
    CharucoDetection,
)
from rb_camera_calibration.utils import opencv_compat as cvx


class OpenCvCharucoDetector:
    """Detect ChArUco boards while returning dependency-free contracts."""

    def __init__(self, board_config: CharucoBoardConfig) -> None:
        self.board_config = board_config
        # Resolve early so placeholder/unknown dictionary values fail helpfully.
        cvx.resolve_aruco_dictionary(board_config.aruco_dictionary)

    def detect(self, frame: CameraFrame) -> CharucoDetection:
        """Detect the configured board in a camera frame."""
        image = cvx.decode_image_bytes(frame.image_bytes)
        return self.detect_image(image, frame_id=frame.frame_id)

    def detect_image(self, image: object, *, frame_id: str = "") -> CharucoDetection:
        """Detect the configured board in an already-decoded image."""
        started = time.perf_counter()
        raw = cvx.detect_charuco_board(image, self.board_config)
        detection_time_ms = (time.perf_counter() - started) * 1000.0
        marker_ids = cvx.ids_to_tuple(raw.get("marker_ids"))
        charuco_ids = cvx.ids_to_tuple(raw.get("charuco_ids"))
        marker_corners = raw.get("marker_corners")
        metrics = cvx.board_point_metrics(marker_corners, image.shape)
        marker_count = len(marker_ids)
        charuco_corner_count = len(charuco_ids)
        warnings = []
        if marker_count == 0:
            warnings.append(
                "No ArUco markers detected. Check lighting, focus, dictionary, and visibility."
            )
        elif charuco_corner_count == 0:
            warnings.append(
                "AruCo markers were detected but no ChArUco corners were interpolated. "
                "Check squares_x/squares_y and the selected dictionary."
            )
        extras = {
            "image_size_wh_px": (int(image.shape[1]), int(image.shape[0])),
            "marker_ids": marker_ids,
            "charuco_ids": charuco_ids,
            "marker_corners_xy": cvx.marker_corners_to_list(marker_corners),
            "charuco_corners_xy": cvx.charuco_corners_to_list(raw.get("charuco_corners")),
            "board_quad_xy_px": metrics["quad_xy_px"],
            "roll_like_angle_deg": metrics["roll_like_angle_deg"],
            "perspective_skew_score": metrics["perspective_skew_score"],
            "configured_squares_xy": (self.board_config.squares_x, self.board_config.squares_y),
        }
        return CharucoDetection(
            frame_id=frame_id,
            detected=marker_count > 0,
            marker_count=marker_count,
            charuco_corner_count=charuco_corner_count,
            marker_ids=marker_ids,
            charuco_ids=charuco_ids,
            board_center_xy_px=metrics["center_xy_px"],
            board_bounds_xyxy_px=metrics["bounds_xyxy_px"],
            board_area_fraction=float(metrics["area_fraction"]),
            edge_margin_px=metrics["edge_margin_px"],
            detection_time_ms=detection_time_ms,
            warning_messages=tuple(warnings),
            extras=extras,
        )

    def debug_reversed_dimensions(self, frame: CameraFrame) -> BoardDimensionDebugReport:
        """Compare configured and reversed board dimensions on the same frame."""
        image = cvx.decode_image_bytes(frame.image_bytes)
        return self.debug_reversed_dimensions_image(image)

    def debug_reversed_dimensions_image(self, image: object) -> BoardDimensionDebugReport:
        """Compare configured and reversed board dimensions on a decoded image."""
        configured = self.detect_image(image, frame_id="configured")
        reversed_config = replace(
            self.board_config,
            squares_x=self.board_config.squares_y,
            squares_y=self.board_config.squares_x,
        )
        reversed_detector = OpenCvCharucoDetector(reversed_config)
        reversed_detection = reversed_detector.detect_image(image, frame_id="reversed")
        reversing_better = (
            reversed_detection.charuco_corner_count > configured.charuco_corner_count
            and reversed_detection.marker_count >= configured.marker_count
        )
        if reversing_better:
            message = (
                "Reversed squares_x/squares_y produced more ChArUco corners; "
                "verify the OpenCV board dimensions before calibration."
            )
        else:
            message = (
                "Configured squares_x/squares_y are at least as plausible as the reversed "
                "orientation for this frame."
            )
        return BoardDimensionDebugReport(
            configured_squares_xy=(self.board_config.squares_x, self.board_config.squares_y),
            reversed_squares_xy=(self.board_config.squares_y, self.board_config.squares_x),
            configured_marker_count=configured.marker_count,
            configured_charuco_corner_count=configured.charuco_corner_count,
            reversed_marker_count=reversed_detection.marker_count,
            reversed_charuco_corner_count=reversed_detection.charuco_corner_count,
            reversing_appears_more_plausible=reversing_better,
            message=message,
        )


def render_detection_overlay(frame: CameraFrame, detection: CharucoDetection) -> bytes:
    """Render marker/corner/bounds overlay PNG for storage or preview."""
    image = cvx.decode_image_bytes(frame.image_bytes)
    overlay = cvx.draw_detection_overlay(image, detection.extras)
    return cvx.encode_png(overlay)
