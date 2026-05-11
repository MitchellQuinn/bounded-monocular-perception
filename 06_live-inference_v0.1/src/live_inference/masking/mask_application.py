"""Shared mask/background-removal operations for preview and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .background_state import BackgroundSnapshot


@dataclass(frozen=True)
class BackgroundRemovalResult:
    """Computed static-background removal mask for one source frame."""

    mask: np.ndarray | None
    applied: bool
    warning: str | None = None

    @property
    def pixel_count(self) -> int:
        """Return the number of background pixels selected for removal."""
        if self.mask is None:
            return 0
        return int(np.count_nonzero(self.mask))


def compute_background_removal_mask(
    current_gray: np.ndarray,
    snapshot: BackgroundSnapshot | None,
) -> BackgroundRemovalResult:
    """Return pixels that match the captured static background.

    Pixels are considered removable background when the grayscale absolute
    difference from the captured background is strictly less than the threshold.
    """
    current = np.asarray(current_gray, dtype=np.uint8)
    if current.ndim != 2:
        raise ValueError(f"Background removal requires a 2D grayscale image; got {current.shape}.")
    if snapshot is None or not snapshot.captured or not snapshot.enabled:
        return BackgroundRemovalResult(mask=None, applied=False)

    height, width = int(current.shape[0]), int(current.shape[1])
    if not snapshot.dimensions_match(width, height):
        warning = (
            "background removal skipped: background size "
            f"{(snapshot.width_px, snapshot.height_px)} does not match source image "
            f"size {(width, height)}."
        )
        return BackgroundRemovalResult(mask=None, applied=False, warning=warning)

    mask = compute_background_removal_mask_from_arrays(
        current,
        snapshot.grayscale_background,
        threshold=snapshot.threshold,
    )
    return BackgroundRemovalResult(mask=mask, applied=True)


def compute_background_removal_mask_from_arrays(
    current_gray: np.ndarray,
    background_gray: np.ndarray,
    *,
    threshold: int,
) -> np.ndarray:
    """Return pixels whose grayscale values match within ``threshold``."""
    current = np.asarray(current_gray, dtype=np.uint8)
    background = np.asarray(background_gray, dtype=np.uint8)
    if current.ndim != 2:
        raise ValueError(f"Background removal requires a 2D grayscale image; got {current.shape}.")
    if background.shape != current.shape:
        raise ValueError(
            "Background shape must match current image shape: "
            f"background={background.shape}, current={current.shape}."
        )
    diff = np.empty(current.shape, dtype=np.int16)
    np.subtract(current, background, out=diff, dtype=np.int16)
    np.abs(diff, out=diff)
    return diff < int(threshold)


def combine_ignore_masks(
    *,
    shape: tuple[int, int],
    manual_mask: np.ndarray | None = None,
    background_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Combine manual and background-removal masks with logical OR."""
    height, width = int(shape[0]), int(shape[1])
    combined: np.ndarray | None = None
    combined_is_owned = False
    for mask in (manual_mask, background_mask):
        if mask is None:
            continue
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (height, width):
            raise ValueError(
                "Ignore mask shape must match source image shape: "
                f"shape={mask_array.shape}, expected={(height, width)}."
            )
        if combined is None:
            combined = mask_array
            combined_is_owned = False
        else:
            if not combined_is_owned:
                combined = np.array(combined, dtype=bool, copy=True)
                combined_is_owned = True
            combined |= mask_array
    return combined


def apply_fill_to_mask(
    image: np.ndarray,
    ignore_mask: np.ndarray | None,
    *,
    fill_value: int,
) -> np.ndarray:
    """Return a copy of image with ignored pixels filled black or white."""
    source = np.asarray(image, dtype=np.uint8)
    if ignore_mask is None:
        return np.array(source, dtype=np.uint8, copy=True)

    mask = np.asarray(ignore_mask, dtype=bool)
    if mask.shape != source.shape[:2]:
        raise ValueError(
            "Ignore mask shape must match image height/width: "
            f"mask={mask.shape}, image={source.shape}."
        )
    value = _coerce_fill_value(fill_value)
    result = np.array(source, dtype=np.uint8, copy=True)
    result[mask] = value
    return result


def _coerce_fill_value(fill_value: int) -> int:
    value = int(fill_value)
    if value not in {0, 255}:
        raise ValueError(f"Mask fill_value must be 0 or 255; got {fill_value!r}.")
    return value


__all__ = [
    "BackgroundRemovalResult",
    "apply_fill_to_mask",
    "combine_ignore_masks",
    "compute_background_removal_mask",
    "compute_background_removal_mask_from_arrays",
]
