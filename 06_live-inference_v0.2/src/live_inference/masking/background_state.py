"""Thread-safe static background state for live preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock

import numpy as np


DEFAULT_BACKGROUND_THRESHOLD = 25


@dataclass(frozen=True)
class BackgroundSnapshot:
    """Immutable, copy-safe snapshot of the captured background image."""

    revision: int
    width_px: int
    height_px: int
    grayscale_background: np.ndarray
    enabled: bool = False
    threshold: int = DEFAULT_BACKGROUND_THRESHOLD
    captured_at_utc: str | None = None

    def __post_init__(self) -> None:
        width = max(0, int(self.width_px))
        height = max(0, int(self.height_px))
        background = np.asarray(self.grayscale_background, dtype=np.uint8)
        if background.shape != (height, width):
            raise ValueError(
                "Background shape must match height/width: "
                f"shape={background.shape}, expected={(height, width)}."
            )
        if _is_immutable_uint8_array(background):
            safe_background = background
        else:
            safe_background = _readonly_uint8_array(background)
        object.__setattr__(self, "revision", int(self.revision))
        object.__setattr__(self, "width_px", width)
        object.__setattr__(self, "height_px", height)
        object.__setattr__(self, "grayscale_background", safe_background)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "threshold", _coerce_threshold(self.threshold))
        object.__setattr__(self, "captured_at_utc", self.captured_at_utc)

    @property
    def captured(self) -> bool:
        """Return whether this snapshot contains a usable captured background."""
        return (
            int(self.width_px) > 0
            and int(self.height_px) > 0
            and self.grayscale_background.shape == (int(self.height_px), int(self.width_px))
        )

    def dimensions_match(self, width_px: int, height_px: int) -> bool:
        """Return whether this snapshot is tied to the provided source size."""
        return int(self.width_px) == int(width_px) and int(self.height_px) == int(height_px)


class BackgroundState:
    """Lock-protected static background shared by GUI and preprocessing threads."""

    def __init__(self, *, threshold: int = DEFAULT_BACKGROUND_THRESHOLD) -> None:
        self._lock = RLock()
        self._revision = 0
        self._width_px = 0
        self._height_px = 0
        self._grayscale_background: np.ndarray | None = None
        self._enabled = False
        self._threshold = _coerce_threshold(threshold)
        self._captured_at_utc: str | None = None
        self._snapshot_cache: BackgroundSnapshot | None = None

    def get_snapshot(self) -> BackgroundSnapshot:
        """Return a stable snapshot that callers may safely retain."""
        with self._lock:
            if self._snapshot_cache is not None:
                return self._snapshot_cache
            if self._grayscale_background is None:
                background = _readonly_uint8_array(np.zeros((0, 0), dtype=np.uint8))
                self._snapshot_cache = BackgroundSnapshot(
                    revision=self._revision,
                    width_px=0,
                    height_px=0,
                    grayscale_background=background,
                    enabled=self._enabled,
                    threshold=self._threshold,
                    captured_at_utc=None,
                )
                return self._snapshot_cache
            self._snapshot_cache = BackgroundSnapshot(
                revision=self._revision,
                width_px=self._width_px,
                height_px=self._height_px,
                grayscale_background=self._grayscale_background,
                enabled=self._enabled,
                threshold=self._threshold,
                captured_at_utc=self._captured_at_utc,
            )
            return self._snapshot_cache

    def capture_background(self, gray_image: np.ndarray) -> int:
        """Capture one static grayscale background and return the updated revision."""
        image = np.asarray(gray_image, dtype=np.uint8)
        if image.ndim != 2:
            raise ValueError(f"Background capture requires a 2D grayscale image; got {image.shape}.")
        height, width = int(image.shape[0]), int(image.shape[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"Background dimensions must be positive; got {(width, height)}.")
        captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00",
            "Z",
        )
        safe_image = _readonly_uint8_array(image)
        with self._lock:
            self._grayscale_background = safe_image
            self._width_px = width
            self._height_px = height
            self._captured_at_utc = captured_at
            self._revision += 1
            self._snapshot_cache = None
            return self._revision

    def clear(self) -> int:
        """Clear the captured background, disable removal, and return the revision."""
        with self._lock:
            self._grayscale_background = None
            self._width_px = 0
            self._height_px = 0
            self._enabled = False
            self._captured_at_utc = None
            self._revision += 1
            self._snapshot_cache = None
            return self._revision

    def set_enabled(self, enabled: bool) -> int:
        """Update whether the captured background should be used."""
        value = bool(enabled)
        with self._lock:
            if self._enabled == value:
                return self._revision
            self._enabled = value
            self._revision += 1
            self._snapshot_cache = None
            return self._revision

    def set_threshold(self, threshold: int) -> int:
        """Update the grayscale absolute-difference threshold."""
        value = _coerce_threshold(threshold)
        with self._lock:
            if self._threshold == value:
                return self._revision
            self._threshold = value
            self._revision += 1
            self._snapshot_cache = None
            return self._revision

    def revision(self) -> int:
        """Return the current background revision."""
        with self._lock:
            return int(self._revision)


def _coerce_threshold(threshold: int) -> int:
    value = int(threshold)
    return min(255, max(0, value))


def _readonly_uint8_array(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.uint8))
    return np.frombuffer(contiguous.tobytes(), dtype=np.uint8).reshape(contiguous.shape)


def _is_immutable_uint8_array(array: np.ndarray) -> bool:
    if array.dtype != np.uint8 or array.flags.writeable:
        return False
    base: object = array
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    return isinstance(base, bytes)


__all__ = [
    "BackgroundSnapshot",
    "BackgroundState",
    "DEFAULT_BACKGROUND_THRESHOLD",
]
