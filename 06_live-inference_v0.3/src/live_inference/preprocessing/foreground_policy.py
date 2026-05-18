"""Runtime policy for live foreground extraction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import RLock
from typing import Any

import interfaces.contracts as contracts


DEFAULT_FOREGROUND_EXTRACTION_MODE = (
    contracts.ForegroundExtractionMode.THRESHOLD_FOREGROUND_V1.value
)
LEGACY_SILHOUETTE_FOREGROUND_EXTRACTION_MODE = (
    contracts.ForegroundExtractionMode.SILHOUETTE_CONTOUR_V2.value
)
SUPPORTED_FOREGROUND_EXTRACTION_MODES = tuple(
    mode.value for mode in contracts.ForegroundExtractionMode
)


@dataclass(frozen=True)
class ForegroundExtractionPolicySnapshot:
    """Immutable foreground extraction runtime settings."""

    revision: int = 0
    foreground_extraction_mode: str = DEFAULT_FOREGROUND_EXTRACTION_MODE
    threshold_white_percentile: float = 90.0
    threshold_margin_px: int = 35
    threshold_min_foreground_fraction: float = 0.001
    threshold_max_foreground_fraction: float = 0.80
    threshold_morphology_close_kernel_px: int = 5
    threshold_fill_holes: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "foreground_extraction_mode",
            normalize_foreground_extraction_mode(self.foreground_extraction_mode),
        )
        object.__setattr__(
            self,
            "threshold_white_percentile",
            max(50.0, min(99.9, float(self.threshold_white_percentile))),
        )
        object.__setattr__(
            self,
            "threshold_margin_px",
            max(0, min(255, int(self.threshold_margin_px))),
        )
        object.__setattr__(
            self,
            "threshold_min_foreground_fraction",
            max(0.0, min(1.0, float(self.threshold_min_foreground_fraction))),
        )
        object.__setattr__(
            self,
            "threshold_max_foreground_fraction",
            max(0.0, min(1.0, float(self.threshold_max_foreground_fraction))),
        )
        object.__setattr__(
            self,
            "threshold_morphology_close_kernel_px",
            max(0, int(self.threshold_morphology_close_kernel_px)),
        )
        object.__setattr__(
            self,
            "threshold_fill_holes",
            bool(self.threshold_fill_holes),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return serializable policy values for trace/debug metadata."""
        return {
            contracts.PREPROCESSING_METADATA_FOREGROUND_EXTRACTION_MODE: (
                self.foreground_extraction_mode
            ),
            contracts.PREPROCESSING_METADATA_FOREGROUND_EXTRACTION_REVISION: int(
                self.revision
            ),
            "threshold_white_percentile": float(self.threshold_white_percentile),
            "threshold_margin_px": int(self.threshold_margin_px),
            "threshold_min_foreground_fraction": float(
                self.threshold_min_foreground_fraction
            ),
            "threshold_max_foreground_fraction": float(
                self.threshold_max_foreground_fraction
            ),
            "threshold_morphology_close_kernel_px": int(
                self.threshold_morphology_close_kernel_px
            ),
            "threshold_fill_holes": bool(self.threshold_fill_holes),
        }


class ForegroundExtractionPolicyState:
    """Lock-protected foreground extraction policy holder."""

    def __init__(
        self,
        initial: ForegroundExtractionPolicySnapshot | None = None,
    ) -> None:
        self._lock = RLock()
        self._snapshot = initial or ForegroundExtractionPolicySnapshot()

    def snapshot(self) -> ForegroundExtractionPolicySnapshot:
        """Return the current immutable policy snapshot."""
        with self._lock:
            return self._snapshot

    def update(self, **updates: Any) -> tuple[ForegroundExtractionPolicySnapshot, int]:
        """Apply supported policy updates and return the new snapshot and revision."""
        if not updates:
            snapshot = self.snapshot()
            return snapshot, int(snapshot.revision)

        with self._lock:
            normalized = {
                _normalized_key(key): _normalized_value(key, value)
                for key, value in updates.items()
                if value is not None
            }
            if not normalized:
                return self._snapshot, int(self._snapshot.revision)
            changed = any(
                getattr(self._snapshot, key) != value
                for key, value in normalized.items()
            )
            if not changed:
                return self._snapshot, int(self._snapshot.revision)
            self._snapshot = replace(
                self._snapshot,
                revision=int(self._snapshot.revision) + 1,
                **normalized,
            )
            return self._snapshot, int(self._snapshot.revision)

    def revision(self) -> int:
        """Return the current policy revision."""
        return int(self.snapshot().revision)


def normalize_foreground_extraction_mode(value: Any) -> str:
    """Return a canonical foreground extraction mode."""
    text = str(value).strip().lower().replace("-", "_")
    if text in {"threshold", "threshold_foreground", "threshold_foreground_v1"}:
        return contracts.ForegroundExtractionMode.THRESHOLD_FOREGROUND_V1.value
    if text in {
        "silhouette",
        "silhouette_contour",
        "silhouette_contour_v2",
        "contour_silhouette",
        "legacy_silhouette",
    }:
        return contracts.ForegroundExtractionMode.SILHOUETTE_CONTOUR_V2.value
    raise ValueError(
        "foreground_extraction_mode must be one of "
        f"{SUPPORTED_FOREGROUND_EXTRACTION_MODES!r}; got {value!r}."
    )


def _normalized_key(name: str) -> str:
    if name in {
        "mode",
        contracts.PREPROCESSING_RUNTIME_PARAMETER_FOREGROUND_EXTRACTION_MODE,
    }:
        return "foreground_extraction_mode"
    return str(name)


def _normalized_value(name: str, value: Any) -> Any:
    key = _normalized_key(name)
    if key == "foreground_extraction_mode":
        return normalize_foreground_extraction_mode(value)
    if key == "threshold_white_percentile":
        return max(50.0, min(99.9, float(value)))
    if key == "threshold_margin_px":
        return max(0, min(255, int(value)))
    if key in {
        "threshold_min_foreground_fraction",
        "threshold_max_foreground_fraction",
    }:
        return max(0.0, min(1.0, float(value)))
    if key == "threshold_morphology_close_kernel_px":
        return max(0, int(value))
    if key == "threshold_fill_holes":
        return bool(value)
    raise AttributeError(f"Unknown foreground extraction policy field: {name!r}.")


__all__ = [
    "DEFAULT_FOREGROUND_EXTRACTION_MODE",
    "LEGACY_SILHOUETTE_FOREGROUND_EXTRACTION_MODE",
    "SUPPORTED_FOREGROUND_EXTRACTION_MODES",
    "ForegroundExtractionPolicySnapshot",
    "ForegroundExtractionPolicyState",
    "normalize_foreground_extraction_mode",
]
