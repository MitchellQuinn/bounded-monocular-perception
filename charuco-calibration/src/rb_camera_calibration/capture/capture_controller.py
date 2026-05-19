"""Automatic capture policy controller."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

from rb_camera_calibration.capture.pose_diversity import SimplePoseDiversityTracker
from rb_camera_calibration.contracts import (
    CameraFrame,
    CaptureDecision,
    CaptureDecisionType,
    CapturePolicyConfig,
    CaptureRejectReason,
    CharucoDetection,
    FrameQualityMetrics,
    PoseCoverageState,
    PoseSignature,
)


class AutomaticCaptureController:
    """Accept only frames that improve calibration quality and pose coverage."""

    def __init__(
        self,
        policy: CapturePolicyConfig,
        pose_tracker: SimplePoseDiversityTracker | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.pose_tracker = pose_tracker or SimplePoseDiversityTracker(policy)
        self.clock = clock
        self._last_accept_time: float | None = None
        self._recent_pose_keys: deque[tuple[tuple[int, int], int, int]] = deque(
            maxlen=max(policy.stability_window_frames, 1)
        )

    def evaluate_frame(
        self,
        frame: CameraFrame,
        detection: CharucoDetection,
        quality: FrameQualityMetrics,
    ) -> CaptureDecision:
        """Evaluate one frame against the automatic capture policy."""
        del frame
        coverage = self.pose_tracker.coverage_state()
        basic_reject = self._basic_reject_reason(detection, quality)
        if basic_reject is not None:
            return self._reject(basic_reject, detection, quality, None, coverage)

        signature = self.pose_tracker.evaluate(detection, quality)
        self._remember_signature(signature)
        cooldown_remaining = self._cooldown_remaining()
        if cooldown_remaining > 0.0:
            return self._reject(
                CaptureRejectReason.COOLDOWN_ACTIVE,
                detection,
                quality,
                signature,
                coverage,
                cooldown_remaining_s=cooldown_remaining,
            )
        if self.policy.require_stability and not self._current_pose_is_stable():
            return self._reject(
                CaptureRejectReason.UNSTABLE_DETECTION,
                detection,
                quality,
                signature,
                coverage,
            )
        if self.policy.require_pose_novelty and self.pose_tracker.has_seen(signature):
            return self._reject(
                CaptureRejectReason.DUPLICATE_POSE,
                detection,
                quality,
                signature,
                coverage,
            )

        coverage = self.pose_tracker.update_accepted(signature)
        self._last_accept_time = self.clock()
        return CaptureDecision(
            decision_type=CaptureDecisionType.ACCEPT,
            accepted=True,
            reason=None,
            message="accepted useful calibration frame",
            detection=detection,
            quality=quality,
            pose_signature=signature,
            coverage_state=coverage,
            cooldown_remaining_s=0.0,
        )

    def force_accept(
        self,
        detection: CharucoDetection,
        quality: FrameQualityMetrics,
    ) -> CaptureDecision:
        """Accept the current detected frame regardless of cooldown/novelty."""
        signature = self.pose_tracker.evaluate(detection, quality)
        coverage = self.pose_tracker.update_accepted(signature)
        self._last_accept_time = self.clock()
        return CaptureDecision(
            decision_type=CaptureDecisionType.ACCEPT,
            accepted=True,
            reason=None,
            message="force accepted by operator",
            detection=detection,
            quality=quality,
            pose_signature=signature,
            coverage_state=coverage,
            cooldown_remaining_s=0.0,
            extras={"manual": True},
        )

    def _basic_reject_reason(
        self,
        detection: CharucoDetection,
        quality: FrameQualityMetrics,
    ) -> CaptureRejectReason | None:
        if not detection.detected:
            return CaptureRejectReason.NO_BOARD
        if detection.marker_count < self.policy.min_marker_count:
            return CaptureRejectReason.TOO_FEW_MARKERS
        if detection.charuco_corner_count < self.policy.min_charuco_corner_count:
            return CaptureRejectReason.TOO_FEW_CHARUCO_CORNERS
        if detection.board_area_fraction < self.policy.min_board_area_fraction:
            return CaptureRejectReason.BOARD_TOO_SMALL
        if detection.board_area_fraction > self.policy.max_board_area_fraction:
            return CaptureRejectReason.BOARD_TOO_LARGE
        if (
            detection.edge_margin_px is not None
            and detection.edge_margin_px < self.policy.min_edge_margin_px
        ):
            return CaptureRejectReason.TOO_CLOSE_TO_EDGE
        if quality.laplacian_variance < self.policy.min_laplacian_variance:
            return CaptureRejectReason.IMAGE_TOO_BLURRY
        if (
            quality.clipped_black_fraction > 0.20
            or quality.clipped_white_fraction > 0.20
            or quality.mean_luma < 20.0
            or quality.mean_luma > 235.0
        ):
            return CaptureRejectReason.EXPOSURE_POOR
        return None

    def _remember_signature(self, signature: PoseSignature) -> None:
        self._recent_pose_keys.append((signature.grid_cell, signature.scale_bin, signature.tilt_bin))

    def _current_pose_is_stable(self) -> bool:
        if len(self._recent_pose_keys) < self.policy.stability_window_frames:
            return False
        return len(set(self._recent_pose_keys)) == 1

    def _cooldown_remaining(self) -> float:
        if self._last_accept_time is None:
            return 0.0
        elapsed = self.clock() - self._last_accept_time
        return max(0.0, self.policy.cooldown_seconds - elapsed)

    def _reject(
        self,
        reason: CaptureRejectReason,
        detection: CharucoDetection,
        quality: FrameQualityMetrics,
        signature: PoseSignature | None,
        coverage: PoseCoverageState,
        *,
        cooldown_remaining_s: float = 0.0,
    ) -> CaptureDecision:
        return CaptureDecision(
            decision_type=CaptureDecisionType.REJECT,
            accepted=False,
            reason=reason,
            message=_message_for_reason(reason),
            detection=detection,
            quality=quality,
            pose_signature=signature,
            coverage_state=coverage,
            cooldown_remaining_s=cooldown_remaining_s,
        )


def _message_for_reason(reason: CaptureRejectReason) -> str:
    messages = {
        CaptureRejectReason.NO_BOARD: "no ChArUco board detected",
        CaptureRejectReason.TOO_FEW_MARKERS: "too few ArUco markers detected",
        CaptureRejectReason.TOO_FEW_CHARUCO_CORNERS: "too few ChArUco corners detected",
        CaptureRejectReason.BOARD_TOO_SMALL: "board is too small in the image",
        CaptureRejectReason.BOARD_TOO_LARGE: "board is too large in the image",
        CaptureRejectReason.TOO_CLOSE_TO_EDGE: "board is too close to the image edge",
        CaptureRejectReason.IMAGE_TOO_BLURRY: "image is too blurry",
        CaptureRejectReason.EXPOSURE_POOR: "exposure is clipped or too dark/bright",
        CaptureRejectReason.DUPLICATE_POSE: "pose bin is already covered",
        CaptureRejectReason.COOLDOWN_ACTIVE: "waiting for capture cooldown",
        CaptureRejectReason.UNSTABLE_DETECTION: "hold board steady for a few frames",
        CaptureRejectReason.CONFIG_MISMATCH: "board configuration appears mismatched",
    }
    return messages[reason]
