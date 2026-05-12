"""Thread-safe stage policy for live preprocessing transforms."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

import interfaces.contracts as contracts


ROI_LOCATOR_INPUT_POLARITY_AS_IS = "as_is"
ROI_LOCATOR_INPUT_POLARITY_INVERTED = "inverted"
SUPPORTED_ROI_LOCATOR_INPUT_POLARITIES = (
    ROI_LOCATOR_INPUT_POLARITY_AS_IS,
    ROI_LOCATOR_INPUT_POLARITY_INVERTED,
)


@dataclass(frozen=True)
class StageTransformPolicySnapshot:
    """Immutable stage-aware transform and ROI guard settings."""

    revision: int = 0
    roi_locator_input_polarity: str = ROI_LOCATOR_INPUT_POLARITY_AS_IS
    apply_manual_mask_to_roi_locator: bool = False
    apply_background_removal_to_roi_locator: bool = False
    apply_manual_mask_to_regressor_preprocessing: bool = True
    apply_background_removal_to_regressor_preprocessing: bool = False
    roi_min_confidence: float = 0.30
    reject_clipped_roi: bool = True
    roi_clip_tolerance_px: int = 0
    roi_min_content_fraction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "roi_locator_input_polarity",
            normalize_roi_locator_input_polarity(self.roi_locator_input_polarity),
        )
        object.__setattr__(
            self,
            "roi_clip_tolerance_px",
            max(0, int(self.roi_clip_tolerance_px)),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return serializable policy values for trace/debug metadata."""
        return {
            "stage_policy_revision": int(self.revision),
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY: str(
                self.roi_locator_input_polarity
            ),
            "apply_manual_mask_to_roi_locator": bool(
                self.apply_manual_mask_to_roi_locator
            ),
            "apply_background_removal_to_roi_locator": bool(
                self.apply_background_removal_to_roi_locator
            ),
            "apply_manual_mask_to_regressor_preprocessing": bool(
                self.apply_manual_mask_to_regressor_preprocessing
            ),
            "apply_background_removal_to_regressor_preprocessing": bool(
                self.apply_background_removal_to_regressor_preprocessing
            ),
            "roi_min_confidence": float(self.roi_min_confidence),
            "reject_clipped_roi": bool(self.reject_clipped_roi),
            contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX: int(
                self.roi_clip_tolerance_px
            ),
            "roi_min_content_fraction": float(self.roi_min_content_fraction),
        }


class StageTransformPolicyState:
    """Lock-protected mutable holder for stage-aware preprocessing policy."""

    def __init__(
        self,
        initial: StageTransformPolicySnapshot | None = None,
    ) -> None:
        self._lock = RLock()
        self._snapshot = initial or StageTransformPolicySnapshot()

    def get_snapshot(self) -> StageTransformPolicySnapshot:
        """Return the current immutable policy snapshot."""
        with self._lock:
            return self._snapshot

    def update(self, **updates: Any) -> StageTransformPolicySnapshot:
        """Apply supported policy updates and return the new snapshot."""
        if not updates:
            return self.get_snapshot()

        with self._lock:
            normalized = {
                key: _normalized_policy_value(key, value)
                for key, value in updates.items()
            }
            changed = any(
                getattr(self._snapshot, key) != value
                for key, value in normalized.items()
            )
            if not changed:
                return self._snapshot
            self._snapshot = replace(
                self._snapshot,
                revision=int(self._snapshot.revision) + 1,
                **normalized,
            )
            return self._snapshot

    def revision(self) -> int:
        """Return the current policy revision."""
        return int(self.get_snapshot().revision)


def _normalized_policy_value(name: str, value: Any) -> Any:
    bool_fields = {
        "apply_manual_mask_to_roi_locator",
        "apply_background_removal_to_roi_locator",
        "apply_manual_mask_to_regressor_preprocessing",
        "apply_background_removal_to_regressor_preprocessing",
        "reject_clipped_roi",
    }
    float_fields = {"roi_min_confidence", "roi_min_content_fraction"}
    int_fields = {"roi_clip_tolerance_px"}
    if name in bool_fields:
        return bool(value)
    if name in float_fields:
        number = float(value)
        if name == "roi_min_content_fraction":
            return max(0.0, min(1.0, number))
        return max(0.0, number)
    if name in int_fields:
        return max(0, int(value))
    if name == "roi_locator_input_polarity":
        return normalize_roi_locator_input_polarity(value)
    raise AttributeError(f"Unknown stage transform policy field: {name!r}.")


def normalize_roi_locator_input_polarity(value: Any) -> str:
    """Return a canonical ROI locator input polarity value."""
    text = str(value).strip().lower().replace("-", "_")
    if text in {"as_is", "asis", "as is", "raw", "none"}:
        return ROI_LOCATOR_INPUT_POLARITY_AS_IS
    if text in {"inverted", "invert"}:
        return ROI_LOCATOR_INPUT_POLARITY_INVERTED
    raise ValueError(
        "roi_locator_input_polarity must be one of "
        f"{SUPPORTED_ROI_LOCATOR_INPUT_POLARITIES!r}; got {value!r}."
    )


__all__ = [
    "ROI_LOCATOR_INPUT_POLARITY_AS_IS",
    "ROI_LOCATOR_INPUT_POLARITY_INVERTED",
    "StageTransformPolicySnapshot",
    "StageTransformPolicyState",
    "SUPPORTED_ROI_LOCATOR_INPUT_POLARITIES",
    "normalize_roi_locator_input_polarity",
]
