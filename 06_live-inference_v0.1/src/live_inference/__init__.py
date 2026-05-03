"""Live inference implementation services."""

from .frame_handoff import (
    AtomicFrameHandoffWriter,
    LatestFrameHandoffReader,
    compute_frame_hash,
)
from .frame_selection import (
    FrameSelectionResult,
    InferenceFrameSelector,
    SelectedFrameForInference,
)
from .runtime_parameters import RuntimeParameterStateManager

__all__ = [
    "AtomicFrameHandoffWriter",
    "FrameSelectionResult",
    "InferenceFrameSelector",
    "LatestFrameHandoffReader",
    "RuntimeParameterStateManager",
    "SelectedFrameForInference",
    "compute_frame_hash",
]
