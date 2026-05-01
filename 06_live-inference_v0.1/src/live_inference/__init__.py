"""Live inference implementation services."""

from .frame_handoff import (
    AtomicFrameHandoffWriter,
    LatestFrameHandoffReader,
    compute_frame_hash,
)

__all__ = [
    "AtomicFrameHandoffWriter",
    "LatestFrameHandoffReader",
    "compute_frame_hash",
]
