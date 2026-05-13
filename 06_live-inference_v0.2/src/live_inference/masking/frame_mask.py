"""Thread-safe binary source-frame mask state for live preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import numpy as np


@dataclass(frozen=True)
class FrameMaskSnapshot:
    """Immutable, copy-safe snapshot of the current source-pixel mask."""

    revision: int
    width_px: int
    height_px: int
    mask: np.ndarray
    fill_value: int
    enabled: bool = True

    def __post_init__(self) -> None:
        width = max(0, int(self.width_px))
        height = max(0, int(self.height_px))
        fill_value = _coerce_fill_value(self.fill_value)
        mask = np.asarray(self.mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                "Frame mask shape must match height/width: "
                f"shape={mask.shape}, expected={(height, width)}."
            )
        safe_mask = np.array(mask, dtype=bool, copy=True)
        safe_mask.setflags(write=False)
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "width_px", width)
        object.__setattr__(self, "height_px", height)
        object.__setattr__(self, "mask", safe_mask)
        object.__setattr__(self, "fill_value", fill_value)
        object.__setattr__(self, "enabled", bool(self.enabled))

    @property
    def pixel_count(self) -> int:
        """Return the number of masked source pixels."""
        return int(np.count_nonzero(self.mask))

    @property
    def has_geometry(self) -> bool:
        """Return whether this snapshot contains a usable mask bitmap."""
        return (
            bool(self.enabled)
            and int(self.width_px) > 0
            and int(self.height_px) > 0
            and self.mask.shape == (int(self.height_px), int(self.width_px))
        )

    def dimensions_match(self, width_px: int, height_px: int) -> bool:
        """Return whether this snapshot is tied to the provided source size."""
        return int(self.width_px) == int(width_px) and int(self.height_px) == int(height_px)


class FrameMaskState:
    """Lock-protected mask state shared by GUI and preprocessing threads."""

    def __init__(self, *, fill_value: int = 255) -> None:
        self._lock = RLock()
        self._revision = 0
        self._width_px = 0
        self._height_px = 0
        self._mask: np.ndarray | None = None
        self._fill_value = _coerce_fill_value(fill_value)
        self._enabled = False

    def get_snapshot(self) -> FrameMaskSnapshot:
        """Return a stable snapshot that callers may safely retain."""
        with self._lock:
            if self._mask is None:
                mask = np.zeros((0, 0), dtype=bool)
                return FrameMaskSnapshot(
                    revision=self._revision,
                    width_px=0,
                    height_px=0,
                    mask=mask,
                    fill_value=self._fill_value,
                    enabled=False,
                )
            return FrameMaskSnapshot(
                revision=self._revision,
                width_px=self._width_px,
                height_px=self._height_px,
                mask=self._mask,
                fill_value=self._fill_value,
                enabled=self._enabled,
            )

    def commit_mask(
        self,
        mask: np.ndarray,
        width_px: int,
        height_px: int,
        fill_value: int,
        enabled: bool = True,
    ) -> int:
        """Store a new committed binary mask and return the updated revision."""
        width = int(width_px)
        height = int(height_px)
        if width <= 0 or height <= 0:
            raise ValueError(f"Frame mask dimensions must be positive; got {(width, height)}.")
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != (height, width):
            raise ValueError(
                "Frame mask shape must match height/width: "
                f"shape={mask_array.shape}, expected={(height, width)}."
            )
        with self._lock:
            self._mask = np.array(mask_array, dtype=bool, copy=True)
            self._width_px = width
            self._height_px = height
            self._fill_value = _coerce_fill_value(fill_value)
            self._enabled = bool(enabled)
            self._revision += 1
            return self._revision

    def set_fill_value(self, fill_value: int) -> int:
        """Update the fill value used with the current mask."""
        value = _coerce_fill_value(fill_value)
        with self._lock:
            if self._fill_value == value:
                return self._revision
            self._fill_value = value
            if self._mask is not None:
                self._revision += 1
            return self._revision

    def clear(self) -> int:
        """Clear the committed mask and return the updated revision."""
        with self._lock:
            self._mask = None
            self._width_px = 0
            self._height_px = 0
            self._enabled = False
            self._revision += 1
            return self._revision

    def revision(self) -> int:
        """Return the current mask revision."""
        with self._lock:
            return int(self._revision)


def _coerce_fill_value(fill_value: int) -> int:
    value = int(fill_value)
    if value not in {0, 255}:
        raise ValueError(f"Frame mask fill_value must be 0 or 255; got {fill_value!r}.")
    return value


__all__ = ["FrameMaskSnapshot", "FrameMaskState"]
