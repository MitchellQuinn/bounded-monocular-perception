"""Image I/O and per-image transform helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np



def read_image_unchanged(image_path: Path) -> np.ndarray:
    """Load an image without changing channels or dtype."""

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image



def to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convert input image array to single-channel uint8 grayscale."""

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray



def read_grayscale_uint8(image_path: Path) -> np.ndarray:
    """Read image and return single-channel uint8 grayscale."""

    return to_grayscale_uint8(read_image_unchanged(image_path))



def write_grayscale_png(image_path: Path, gray_image: np.ndarray, dry_run: bool = False) -> None:
    """Write single-channel uint8 PNG image with no alpha channel."""

    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    if gray_image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got dtype {gray_image.dtype}")

    if dry_run:
        return

    image_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(image_path), gray_image)
    if not success:
        raise IOError(f"Failed to write image: {image_path}")



def edge_image_black_on_white(
    source_gray: np.ndarray,
    blur_kernel_size: int,
    canny_low_threshold: int,
    canny_high_threshold: int,
) -> np.ndarray:
    """Generate full-frame black-edge-on-white image from grayscale source."""

    if source_gray.ndim != 2:
        raise ValueError("source_gray must be 2D grayscale.")

    processed = source_gray
    if blur_kernel_size > 1:
        processed = cv2.GaussianBlur(processed, (blur_kernel_size, blur_kernel_size), 0)

    edges = cv2.Canny(processed, int(canny_low_threshold), int(canny_high_threshold))

    output = np.full(source_gray.shape, 255, dtype=np.uint8)
    output[edges > 0] = 0
    return output



def bbox_png_to_training_array(
    bbox_gray: np.ndarray,
    *,
    normalize: bool = True,
    invert: bool = True,
) -> np.ndarray:
    """Convert bbox PNG grayscale image to float32 training array (H, W)."""

    if bbox_gray.ndim != 2:
        raise ValueError("bbox_gray must be 2D grayscale.")

    arr = bbox_gray.astype(np.float32)

    if normalize:
        arr = arr / 255.0

    if invert:
        arr = 1.0 - arr

    return arr.astype(np.float32)
