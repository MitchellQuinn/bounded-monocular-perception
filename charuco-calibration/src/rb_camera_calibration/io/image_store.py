"""Image persistence helpers."""

from __future__ import annotations

from pathlib import Path

from rb_camera_calibration.io.atomic_write import atomic_write_bytes


def write_png_bytes(path: Path, image_bytes: bytes) -> Path:
    """Write encoded PNG bytes to ``path`` atomically."""
    atomic_write_bytes(path, image_bytes)
    return path
