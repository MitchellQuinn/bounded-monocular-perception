"""Frame masking state shared between the live GUI and preprocessing."""

from .background_state import (
    DEFAULT_BACKGROUND_THRESHOLD,
    BackgroundSnapshot,
    BackgroundState,
)
from .frame_mask import FrameMaskSnapshot, FrameMaskState
from .mask_application import (
    BackgroundRemovalResult,
    apply_fill_to_mask,
    combine_ignore_masks,
    compute_background_removal_mask,
)

__all__ = [
    "BackgroundRemovalResult",
    "BackgroundSnapshot",
    "BackgroundState",
    "DEFAULT_BACKGROUND_THRESHOLD",
    "FrameMaskSnapshot",
    "FrameMaskState",
    "apply_fill_to_mask",
    "combine_ignore_masks",
    "compute_background_removal_mask",
]
