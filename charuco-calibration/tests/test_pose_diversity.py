from __future__ import annotations

from rb_camera_calibration.capture.pose_diversity import SimplePoseDiversityTracker
from rb_camera_calibration.contracts import CapturePolicyConfig, CharucoDetection, FrameQualityMetrics


def _quality() -> FrameQualityMetrics:
    return FrameQualityMetrics(
        laplacian_variance=150.0,
        mean_luma=120.0,
        luma_std=40.0,
        clipped_black_fraction=0.0,
        clipped_white_fraction=0.0,
        contrast_score=0.5,
        blur_score=0.5,
    )


def test_pose_diversity_assigns_different_bins() -> None:
    tracker = SimplePoseDiversityTracker(CapturePolicyConfig())
    left = CharucoDetection(
        frame_id="left",
        detected=True,
        marker_count=12,
        charuco_corner_count=30,
        board_center_xy_px=(100.0, 100.0),
        board_bounds_xyxy_px=(50.0, 50.0, 150.0, 150.0),
        board_area_fraction=0.05,
        edge_margin_px=50.0,
        extras={"image_size_wh_px": (900, 600), "perspective_skew_score": 0.1},
    )
    right_large_tilted = CharucoDetection(
        frame_id="right",
        detected=True,
        marker_count=12,
        charuco_corner_count=30,
        board_center_xy_px=(800.0, 500.0),
        board_bounds_xyxy_px=(650.0, 350.0, 890.0, 590.0),
        board_area_fraction=0.50,
        edge_margin_px=10.0,
        extras={"image_size_wh_px": (900, 600), "perspective_skew_score": 0.8},
    )

    sig_a = tracker.evaluate(left, _quality())
    sig_b = tracker.evaluate(right_large_tilted, _quality())

    assert sig_a.grid_cell != sig_b.grid_cell
    assert sig_a.scale_bin != sig_b.scale_bin
    assert sig_a.tilt_bin != sig_b.tilt_bin


def test_pose_diversity_splits_practical_fixed_focus_area_range() -> None:
    tracker = SimplePoseDiversityTracker(CapturePolicyConfig())
    bins = []
    for area in (0.04, 0.07, 0.09):
        detection = CharucoDetection(
            frame_id=str(area),
            detected=True,
            marker_count=12,
            charuco_corner_count=30,
            board_center_xy_px=(450.0, 300.0),
            board_bounds_xyxy_px=(250.0, 120.0, 650.0, 480.0),
            board_area_fraction=area,
            edge_margin_px=50.0,
            extras={"image_size_wh_px": (900, 600), "perspective_skew_score": 0.1},
        )
        bins.append(tracker.evaluate(detection, _quality()).scale_bin)

    assert bins == [0, 1, 2]
