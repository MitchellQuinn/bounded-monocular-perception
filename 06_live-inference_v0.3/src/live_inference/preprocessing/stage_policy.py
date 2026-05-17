"""Thread-safe stage policy for live preprocessing transforms."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

import interfaces.contracts as contracts


ROI_LOCATOR_INPUT_POLARITY_AS_IS = "as_is"
ROI_LOCATOR_INPUT_POLARITY_INVERTED = "inverted"
ROI_LOCATOR_INPUT_MODE_AS_IS = ROI_LOCATOR_INPUT_POLARITY_AS_IS
ROI_LOCATOR_INPUT_MODE_INVERTED = ROI_LOCATOR_INPUT_POLARITY_INVERTED
ROI_LOCATOR_INPUT_MODE_SHEET_DARK_FOREGROUND = "sheet_dark_foreground"
SUPPORTED_ROI_LOCATOR_INPUT_POLARITIES = (
    ROI_LOCATOR_INPUT_POLARITY_AS_IS,
    ROI_LOCATOR_INPUT_POLARITY_INVERTED,
)
SUPPORTED_ROI_LOCATOR_INPUT_MODES = (
    ROI_LOCATOR_INPUT_MODE_AS_IS,
    ROI_LOCATOR_INPUT_MODE_INVERTED,
    ROI_LOCATOR_INPUT_MODE_SHEET_DARK_FOREGROUND,
)
DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR = (
    "baseline_inverted_masked_locator"
)
_DIAGNOSTIC_PROFILE_VALUES: dict[str, dict[str, Any]] = {
    DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR: {
        "roi_locator_input_mode": ROI_LOCATOR_INPUT_MODE_INVERTED,
        "apply_manual_mask_to_roi_locator": True,
        "apply_manual_mask_to_regressor_preprocessing": True,
        "apply_background_removal_to_roi_locator": False,
        "apply_background_removal_to_regressor_preprocessing": False,
    }
}
SUPPORTED_DIAGNOSTIC_PROFILES = tuple(_DIAGNOSTIC_PROFILE_VALUES)


@dataclass(frozen=True)
class StageTransformPolicySnapshot:
    """Immutable stage-aware transform and ROI guard settings."""

    revision: int = 0
    roi_locator_input_polarity: str = ROI_LOCATOR_INPUT_POLARITY_AS_IS
    roi_locator_input_mode: str | None = None
    apply_manual_mask_to_roi_locator: bool = False
    apply_background_removal_to_roi_locator: bool = False
    apply_manual_mask_to_regressor_preprocessing: bool = True
    apply_background_removal_to_regressor_preprocessing: bool = False
    roi_min_confidence: float = 0.30
    reject_clipped_roi: bool = True
    roi_clip_tolerance_px: int = 0
    roi_min_content_fraction: float = 0.0
    sheet_min_gray: int = 190
    target_max_gray: int = 130
    min_component_area_px: int = 75
    morphology_close_kernel_px: int = 3
    dilate_kernel_px: int = 0
    restrict_to_lower_frame_fraction: float = 0.0
    diagnostic_profile_name: str | None = None

    def __post_init__(self) -> None:
        mode = normalize_roi_locator_input_mode(
            self.roi_locator_input_mode
            if self.roi_locator_input_mode is not None
            else self.roi_locator_input_polarity
        )
        object.__setattr__(self, "roi_locator_input_mode", mode)
        object.__setattr__(self, "roi_locator_input_polarity", mode)
        object.__setattr__(
            self,
            "roi_clip_tolerance_px",
            max(0, int(self.roi_clip_tolerance_px)),
        )
        object.__setattr__(
            self,
            "sheet_min_gray",
            _clamped_uint8(self.sheet_min_gray),
        )
        object.__setattr__(
            self,
            "target_max_gray",
            _clamped_uint8(self.target_max_gray),
        )
        object.__setattr__(
            self,
            "min_component_area_px",
            max(0, int(self.min_component_area_px)),
        )
        object.__setattr__(
            self,
            "morphology_close_kernel_px",
            max(0, int(self.morphology_close_kernel_px)),
        )
        object.__setattr__(
            self,
            "dilate_kernel_px",
            max(0, int(self.dilate_kernel_px)),
        )
        object.__setattr__(
            self,
            "restrict_to_lower_frame_fraction",
            max(0.0, min(1.0, float(self.restrict_to_lower_frame_fraction))),
        )
        if self.diagnostic_profile_name is not None:
            object.__setattr__(
                self,
                "diagnostic_profile_name",
                normalize_diagnostic_profile_name(self.diagnostic_profile_name),
            )

    def to_metadata(self) -> dict[str, Any]:
        """Return serializable policy values for trace/debug metadata."""
        return {
            "stage_policy_revision": int(self.revision),
            "diagnostic_profile_name": self.diagnostic_profile_name,
            "roi_locator_input_mode": str(self.roi_locator_input_mode),
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY: str(
                self.roi_locator_input_polarity
            ),
            "roi_locator_sheet_min_gray": int(self.sheet_min_gray),
            "roi_locator_target_max_gray": int(self.target_max_gray),
            "roi_locator_min_component_area_px": int(self.min_component_area_px),
            "roi_locator_morphology_close_kernel_px": int(
                self.morphology_close_kernel_px
            ),
            "roi_locator_dilate_kernel_px": int(self.dilate_kernel_px),
            "roi_locator_restrict_to_lower_frame_fraction": float(
                self.restrict_to_lower_frame_fraction
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
            if "diagnostic_profile_name" not in normalized:
                normalized["diagnostic_profile_name"] = None
            self._snapshot = replace(
                self._snapshot,
                revision=int(self._snapshot.revision) + 1,
                **normalized,
            )
            return self._snapshot

    def apply_diagnostic_profile(
        self,
        profile_name: str,
    ) -> StageTransformPolicySnapshot:
        """Apply a named diagnostic policy profile and return the new snapshot."""
        normalized_name = normalize_diagnostic_profile_name(profile_name)
        return self.update(
            **diagnostic_profile_updates(normalized_name),
            diagnostic_profile_name=normalized_name,
        )

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
    int_fields = {
        "roi_clip_tolerance_px",
        "sheet_min_gray",
        "target_max_gray",
        "min_component_area_px",
        "morphology_close_kernel_px",
        "dilate_kernel_px",
    }
    if name in bool_fields:
        return bool(value)
    if name in float_fields:
        number = float(value)
        if name == "roi_min_content_fraction":
            return max(0.0, min(1.0, number))
        return max(0.0, number)
    if name == "restrict_to_lower_frame_fraction":
        return max(0.0, min(1.0, float(value)))
    if name in int_fields:
        if name in {"sheet_min_gray", "target_max_gray"}:
            return _clamped_uint8(value)
        return max(0, int(value))
    if name == "roi_locator_input_polarity":
        return normalize_roi_locator_input_mode(value)
    if name == "roi_locator_input_mode":
        return normalize_roi_locator_input_mode(value)
    if name == "diagnostic_profile_name":
        return (
            None if value is None else normalize_diagnostic_profile_name(value)
        )
    raise AttributeError(f"Unknown stage transform policy field: {name!r}.")


def normalize_roi_locator_input_polarity(value: Any) -> str:
    """Return a canonical ROI locator input polarity value."""
    return normalize_roi_locator_input_mode(value)


def normalize_roi_locator_input_mode(value: Any) -> str:
    """Return a canonical ROI locator input representation mode."""
    text = str(value).strip().lower().replace("-", "_")
    if text in {"as_is", "asis", "as is", "raw", "none"}:
        return ROI_LOCATOR_INPUT_MODE_AS_IS
    if text in {"inverted", "invert"}:
        return ROI_LOCATOR_INPUT_MODE_INVERTED
    if text in {"sheet_dark_foreground", "sheet dark foreground", "sheet"}:
        return ROI_LOCATOR_INPUT_MODE_SHEET_DARK_FOREGROUND
    raise ValueError(
        "roi_locator_input_mode must be one of "
        f"{SUPPORTED_ROI_LOCATOR_INPUT_MODES!r}; got {value!r}."
    )


def normalize_diagnostic_profile_name(value: Any) -> str:
    """Return a canonical diagnostic profile name."""
    text = str(value).strip().lower().replace("-", "_")
    if text in _DIAGNOSTIC_PROFILE_VALUES:
        return text
    raise ValueError(
        "diagnostic_profile must be one of "
        f"{SUPPORTED_DIAGNOSTIC_PROFILES!r}; got {value!r}."
    )


def diagnostic_profile_updates(profile_name: str) -> dict[str, Any]:
    """Return a mutable copy of the named diagnostic profile updates."""
    normalized_name = normalize_diagnostic_profile_name(profile_name)
    return dict(_DIAGNOSTIC_PROFILE_VALUES[normalized_name])


def _clamped_uint8(value: Any) -> int:
    return max(0, min(255, int(value)))


__all__ = [
    "DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR",
    "ROI_LOCATOR_INPUT_MODE_AS_IS",
    "ROI_LOCATOR_INPUT_MODE_INVERTED",
    "ROI_LOCATOR_INPUT_MODE_SHEET_DARK_FOREGROUND",
    "ROI_LOCATOR_INPUT_POLARITY_AS_IS",
    "ROI_LOCATOR_INPUT_POLARITY_INVERTED",
    "StageTransformPolicySnapshot",
    "StageTransformPolicyState",
    "SUPPORTED_DIAGNOSTIC_PROFILES",
    "SUPPORTED_ROI_LOCATOR_INPUT_MODES",
    "SUPPORTED_ROI_LOCATOR_INPUT_POLARITIES",
    "diagnostic_profile_updates",
    "normalize_diagnostic_profile_name",
    "normalize_roi_locator_input_mode",
    "normalize_roi_locator_input_polarity",
]
