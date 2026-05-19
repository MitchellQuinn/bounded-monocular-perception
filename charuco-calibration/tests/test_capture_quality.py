from __future__ import annotations

from rb_camera_calibration.capture.capture_quality import score_decoded_image


def test_capture_quality_scores_blur_and_exposure() -> None:
    import cv2
    import numpy as np

    image = np.zeros((80, 80), dtype=np.uint8)
    image[::2, :] = 255
    sharp = score_decoded_image(image)
    blurred = score_decoded_image(cv2.GaussianBlur(image, (9, 9), 0))
    white = score_decoded_image(np.full((80, 80), 255, dtype=np.uint8))

    assert sharp.laplacian_variance > blurred.laplacian_variance
    assert white.clipped_white_fraction == 1.0
