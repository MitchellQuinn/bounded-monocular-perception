"""Image I/O helpers used by v4 pipeline stages."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image_unchanged(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def to_grayscale_uint8(image: np.ndarray) -> np.ndarray:
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


def to_bgr_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(to_grayscale_uint8(image), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported image shape for BGR conversion: {image.shape}")


def read_grayscale_uint8(image_path: Path) -> np.ndarray:
    return to_grayscale_uint8(read_image_unchanged(image_path))


def write_grayscale_png(image_path: Path, gray_image: np.ndarray, dry_run: bool = False) -> None:
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {gray_image.shape}")

    image_u8 = gray_image if gray_image.dtype == np.uint8 else np.clip(gray_image, 0, 255).astype(np.uint8)
    if dry_run:
        return

    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(image_path), image_u8)
    if not ok:
        raise IOError(f"Failed to write image: {image_path}")


def write_bgr_png(image_path: Path, bgr_image: np.ndarray, dry_run: bool = False) -> None:
    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError(f"Expected BGR image shape (H, W, 3), got {bgr_image.shape}")

    image_u8 = np.clip(bgr_image, 0, 255).astype(np.uint8)
    if dry_run:
        return

    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(image_path), image_u8)
    if not ok:
        raise IOError(f"Failed to write image: {image_path}")
