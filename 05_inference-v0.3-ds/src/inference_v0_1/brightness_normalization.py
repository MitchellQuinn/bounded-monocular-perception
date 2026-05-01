"""Vendored deterministic brightness normalization for v0.3 inference.

This module is copied from the preprocessing source of truth:
`02_synthetic-data-processing-v4.0/rb_pipeline_v4/brightness_normalization.py`
and the matching `BrightnessNormalizationConfigV4` contract behavior in
`rb_pipeline_v4/config.py`.

Keep this behavior-equivalent to preprocessing until the repo has a shared
representation/preprocessing package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np


_VALID_BRIGHTNESS_NORMALIZATION_METHODS = {"none", "masked_median_darkness_gain"}
_VALID_EMPTY_MASK_POLICIES = {"skip", "fail"}


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return bool(value)


@dataclass(frozen=True)
class BrightnessNormalizationConfigV3:
    """Config for deterministic foreground-only brightness normalization."""

    enabled: bool = False
    method: str = "none"
    target_median_darkness: float = 0.55
    min_gain: float = 0.5
    max_gain: float = 2.0
    epsilon: float = 1e-6
    empty_mask_policy: str = "skip"

    def normalized_enabled(self) -> bool:
        return bool(self.enabled)

    def normalized_method(self) -> str:
        value = str(self.method).strip().lower()
        if value not in _VALID_BRIGHTNESS_NORMALIZATION_METHODS:
            allowed = ", ".join(sorted(_VALID_BRIGHTNESS_NORMALIZATION_METHODS))
            raise ValueError(f"Unsupported brightness normalization method '{self.method}'. Allowed: {allowed}.")
        return value

    def normalized_target_median_darkness(self) -> float:
        value = float(self.target_median_darkness)
        if not math.isfinite(value) or value < 0.05 or value > 0.95:
            raise ValueError("target_median_darkness must be finite and in [0.05, 0.95]")
        return value

    def normalized_min_gain(self) -> float:
        value = float(self.min_gain)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("min_gain must be finite and > 0")
        return value

    def normalized_max_gain(self) -> float:
        value = float(self.max_gain)
        min_gain = self.normalized_min_gain()
        if not math.isfinite(value) or value < min_gain:
            raise ValueError("max_gain must be finite and >= min_gain")
        return value

    def normalized_epsilon(self) -> float:
        value = float(self.epsilon)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("epsilon must be finite and > 0")
        return value

    def normalized_empty_mask_policy(self) -> str:
        value = str(self.empty_mask_policy).strip().lower()
        if value not in _VALID_EMPTY_MASK_POLICIES:
            allowed = ", ".join(sorted(_VALID_EMPTY_MASK_POLICIES))
            raise ValueError(f"Unsupported empty_mask_policy '{self.empty_mask_policy}'. Allowed: {allowed}.")
        return value

    def active_method(self) -> str:
        method = self.normalized_method()
        return method if self.normalized_enabled() else "none"

    def to_contract_dict(self) -> dict[str, object]:
        return {
            "Enabled": self.normalized_enabled(),
            "Method": self.normalized_method(),
            "TargetMedianDarkness": self.normalized_target_median_darkness(),
            "MinGain": self.normalized_min_gain(),
            "MaxGain": self.normalized_max_gain(),
            "Epsilon": self.normalized_epsilon(),
            "EmptyMaskPolicy": self.normalized_empty_mask_policy(),
        }

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["enabled"] = self.normalized_enabled()
        payload["method"] = self.normalized_method()
        payload["target_median_darkness"] = self.normalized_target_median_darkness()
        payload["min_gain"] = self.normalized_min_gain()
        payload["max_gain"] = self.normalized_max_gain()
        payload["epsilon"] = self.normalized_epsilon()
        payload["empty_mask_policy"] = self.normalized_empty_mask_policy()
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "BrightnessNormalizationConfigV3":
        def read(*names: str, default: object) -> object:
            for name in names:
                if name in payload:
                    return payload[name]
            return default

        return cls(
            enabled=_coerce_bool(read("enabled", "Enabled", default=False)),
            method=str(read("method", "Method", default="none")),
            target_median_darkness=float(
                read("target_median_darkness", "TargetMedianDarkness", default=0.55)
            ),
            min_gain=float(read("min_gain", "MinGain", default=0.5)),
            max_gain=float(read("max_gain", "MaxGain", default=2.0)),
            epsilon=float(read("epsilon", "Epsilon", default=1e-6)),
            empty_mask_policy=str(read("empty_mask_policy", "EmptyMaskPolicy", default="skip")),
        )


@dataclass(frozen=True)
class BrightnessNormalizationResultV3:
    """Result of applying one brightness normalization policy to one ROI."""

    image: np.ndarray
    status: str
    method: str
    foreground_pixel_count: int
    current_median_darkness: float
    effective_median_darkness: float
    gain: float

    def to_log_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "method": self.method,
            "foreground_pixel_count": int(self.foreground_pixel_count),
            "current_median_darkness": float(self.current_median_darkness),
            "effective_median_darkness": float(self.effective_median_darkness),
            "gain": float(self.gain),
        }


def apply_brightness_normalization_v3(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    config: BrightnessNormalizationConfigV3,
) -> BrightnessNormalizationResultV3:
    """Apply the configured deterministic foreground-only brightness normalization."""

    method = config.normalized_method()
    image_float = _validate_image(image)
    mask_bool = _validate_foreground_mask(foreground_mask, image_float.shape)

    if not config.normalized_enabled() or method == "none":
        return BrightnessNormalizationResultV3(
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
    config: BrightnessNormalizationConfigV3,
) -> BrightnessNormalizationResultV3:
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

    return BrightnessNormalizationResultV3(
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
    config: BrightnessNormalizationConfigV3,
) -> BrightnessNormalizationResultV3:
    policy = config.normalized_empty_mask_policy()
    if policy == "fail":
        raise ValueError("brightness normalization foreground mask is empty")
    return BrightnessNormalizationResultV3(
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
