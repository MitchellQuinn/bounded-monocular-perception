from __future__ import annotations

from rb_camera_calibration.capture.capture_controller import AutomaticCaptureController
from rb_camera_calibration.contracts import (
    CameraFrame,
    CapturePolicyConfig,
    CaptureRejectReason,
    CharucoDetection,
    FrameHash,
    FrameMetadata,
    FrameQualityMetrics,
)


def _frame() -> CameraFrame:
    metadata = FrameMetadata(
        frame_id="frame-1",
        sequence_index=1,
        captured_at_utc="2026-05-19T00:00:00Z",
        width_px=900,
        height_px=600,
        pixel_format="png",
        source_name="test",
    )
    return CameraFrame(
        frame_id="frame-1",
        metadata=metadata,
        frame_hash=FrameHash(value="abc"),
        image_bytes=b"fake",
    )


def _detection(**overrides) -> CharucoDetection:
    data = {
        "frame_id": "frame-1",
        "detected": True,
        "marker_count": 12,
        "charuco_corner_count": 30,
        "board_center_xy_px": (450.0, 300.0),
        "board_bounds_xyxy_px": (250.0, 120.0, 650.0, 480.0),
        "board_area_fraction": 0.20,
        "edge_margin_px": 120.0,
        "extras": {"image_size_wh_px": (900, 600), "perspective_skew_score": 0.2},
    }
    data.update(overrides)
    return CharucoDetection(**data)


def _quality(**overrides) -> FrameQualityMetrics:
    data = {
        "laplacian_variance": 150.0,
        "mean_luma": 120.0,
        "luma_std": 45.0,
        "clipped_black_fraction": 0.0,
        "clipped_white_fraction": 0.0,
        "contrast_score": 0.7,
        "blur_score": 0.7,
    }
    data.update(overrides)
    return FrameQualityMetrics(**data)


def test_capture_controller_rejects_low_quality() -> None:
    policy = CapturePolicyConfig(require_stability=False)
    controller = AutomaticCaptureController(policy)

    decision = controller.evaluate_frame(_frame(), _detection(), _quality(laplacian_variance=10.0))

    assert decision.reason == CaptureRejectReason.IMAGE_TOO_BLURRY


def test_capture_controller_rejects_too_close_to_edge() -> None:
    policy = CapturePolicyConfig(require_stability=False)
    controller = AutomaticCaptureController(policy)

    decision = controller.evaluate_frame(_frame(), _detection(edge_margin_px=2.0), _quality())

    assert decision.reason == CaptureRejectReason.TOO_CLOSE_TO_EDGE


def test_capture_controller_rejects_duplicate_pose_after_accept() -> None:
    now = 0.0

    def clock() -> float:
        return now

    policy = CapturePolicyConfig(require_stability=False, cooldown_seconds=0.0)
    controller = AutomaticCaptureController(policy, clock=clock)

    first = controller.evaluate_frame(_frame(), _detection(), _quality())
    second = controller.evaluate_frame(_frame(), _detection(frame_id="frame-2"), _quality())

    assert first.accepted
    assert second.reason == CaptureRejectReason.DUPLICATE_POSE
