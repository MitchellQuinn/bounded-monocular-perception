"""Deterministic brightness normalization helpers for packed ROI representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BrightnessNormalizationConfigV4


@dataclass(frozen=True)
class BrightnessNormalizationResultV4:
    """Result of applying one brightness normalization policy to one ROI."""

    image: np.ndarray
    status: str
    method: str
    foreground_pixel_count: int
    current_median_darkness: float
    effective_median_darkness: float
    gain: float


def apply_brightness_normalization_v4(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    config: BrightnessNormalizationConfigV4,
) -> BrightnessNormalizationResultV4:
    """Apply the configured deterministic foreground-only brightness normalization."""

    method = config.normalized_method()
    image_float = _validate_image(image)
    mask_bool = _validate_foreground_mask(foreground_mask, image_float.shape)

    if not config.normalized_enabled() or method == "none":
        return BrightnessNormalizationResultV4(
            image=image_float.copy(),
            status="disabled",
            method=method,
            foreground_pixel_count=int(np.count_nonzero(mask_bool)),
            current_median_darkness=float("nan"),
            effective_median_darkness=float("nan"),
            gain=1.0,
        )

    if method == "masked_median_darkness_gain":
        return _apply_masked_median_darkness_gain(image_float, mask_bool, config)

    raise ValueError(f"Unsupported brightness normalization method: {method}")


def _apply_masked_median_darkness_gain(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    config: BrightnessNormalizationConfigV4,
) -> BrightnessNormalizationResultV4:
    foreground_count = int(np.count_nonzero(foreground_mask))
    if foreground_count <= 0:
        return _handle_empty_mask(image, config)

    darkness = 1.0 - image
    foreground_darkness = darkness[foreground_mask]
    if foreground_darkness.size == 0:
        return _handle_empty_mask(image, config)

    current_median = float(np.median(foreground_darkness))
    epsilon = config.normalized_epsilon()
    effective_median = max(current_median, epsilon)
    gain = config.normalized_target_median_darkness() / effective_median
    gain = max(config.normalized_min_gain(), min(config.normalized_max_gain(), gain))

    normalized_darkness = darkness.copy()
    normalized_darkness[foreground_mask] = np.clip(
        darkness[foreground_mask] * gain,
        0.0,
        1.0,
    )
    normalized = 1.0 - normalized_darkness
    normalized[~foreground_mask] = image[~foreground_mask]

    return BrightnessNormalizationResultV4(
        image=normalized.astype(np.float32, copy=False),
        status="success",
        method="masked_median_darkness_gain",
        foreground_pixel_count=foreground_count,
        current_median_darkness=current_median,
        effective_median_darkness=float(effective_median),
        gain=float(gain),
    )


def _handle_empty_mask(
    image: np.ndarray,
    config: BrightnessNormalizationConfigV4,
) -> BrightnessNormalizationResultV4:
    policy = config.normalized_empty_mask_policy()
    if policy == "fail":
        raise ValueError("brightness normalization foreground mask is empty")
    return BrightnessNormalizationResultV4(
        image=image.copy(),
        status="skipped_empty_mask",
        method=config.normalized_method(),
        foreground_pixel_count=0,
        current_median_darkness=float("nan"),
        effective_median_darkness=float("nan"),
        gain=1.0,
    )


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"brightness normalization image must be 2D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("brightness normalization image contains NaN or Inf")
    if array.size and (float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6):
        raise ValueError("brightness normalization image values must be in [0, 1]")
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def _validate_foreground_mask(mask: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"brightness normalization foreground mask must be 2D, got {mask_array.shape}")
    if tuple(mask_array.shape) != tuple(expected_shape):
        raise ValueError(
            "brightness normalization image/mask shape mismatch: "
            f"image={expected_shape}, mask={mask_array.shape}"
        )
    return mask_array.astype(bool, copy=False)
