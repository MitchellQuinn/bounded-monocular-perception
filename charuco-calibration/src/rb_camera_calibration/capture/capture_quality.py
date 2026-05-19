"""Frame quality scoring for automatic capture."""

from __future__ import annotations

from rb_camera_calibration.contracts import CameraFrame, FrameQualityMetrics
from rb_camera_calibration.utils import opencv_compat as cvx


class OpenCvFrameQualityScorer:
    """Compute simple, deterministic image quality metrics with OpenCV."""

    def score(self, frame: CameraFrame) -> FrameQualityMetrics:
        image = cvx.decode_image_bytes(frame.image_bytes)
        return score_decoded_image(image)


def score_decoded_image(image: object) -> FrameQualityMetrics:
    """Score a decoded OpenCV image."""
    cv2 = cvx.import_cv2()
    import numpy as np

    gray = cvx.to_gray(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = float(laplacian.var())
    luma = gray.astype("float32")
    mean_luma = float(luma.mean())
    luma_std = float(luma.std())
    clipped_black_fraction = float(np.mean(gray <= 3))
    clipped_white_fraction = float(np.mean(gray >= 252))
    contrast_score = max(0.0, min(1.0, luma_std / 64.0))
    blur_score = max(0.0, min(1.0, laplacian_variance / 500.0))
    return FrameQualityMetrics(
        laplacian_variance=laplacian_variance,
        mean_luma=mean_luma,
        luma_std=luma_std,
        clipped_black_fraction=clipped_black_fraction,
        clipped_white_fraction=clipped_white_fraction,
        contrast_score=contrast_score,
        blur_score=blur_score,
    )
