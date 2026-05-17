"""Manifest-derived configuration for concrete live preprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import interfaces.contracts as contracts
from live_inference.model_registry.model_manifest import (
    ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE_ON_WHITE,
    SUPPORTED_ORIENTATION_SOURCE_MODES,
    LiveModelManifest,
    resolve_orientation_source_mode,
)


DEFAULT_BRIGHTNESS_MASK_SOURCE = "silhouette_background_mask < 0.5"
_SUPPORTED_BRIGHTNESS_MASK_SOURCES = {
    DEFAULT_BRIGHTNESS_MASK_SOURCE,
    "silhouette_background_mask_lt_0_5",
    "silhouette_foreground_mask",
    "foreground_mask",
}


def _ensure_preprocessing_paths() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    for path in (repo_root / "02_synthetic-data-processing-v4.0",):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


_ensure_preprocessing_paths()

from rb_pipeline_v4.config import (  # noqa: E402
    BrightnessNormalizationConfigV4,
    ForegroundEnhancementConfigV4,
    SilhouetteStageConfigV4,
)


@dataclass(frozen=True)
class BrightnessNormalizationRuntimeConfig:
    """Resolved inference-time brightness normalization contract."""

    config: BrightnessNormalizationConfigV4
    contract_source: str
    mask_source: str
    explicit_mask_source: bool
    raw_contract: Mapping[str, Any] = field(default_factory=dict)

    def active(self) -> bool:
        return self.config.normalized_enabled() and self.config.normalized_method() != "none"

    def to_log_dict(self) -> dict[str, Any]:
        payload = self.config.to_contract_dict()
        payload.update(
            {
                "Active": self.active(),
                "ContractSource": self.contract_source,
                "MaskSource": self.mask_source,
                "MaskSourceExplicit": bool(self.explicit_mask_source),
            }
        )
        return payload


@dataclass(frozen=True)
class ForegroundEnhancementRuntimeConfig:
    """Resolved inference-time foreground enhancement contract."""

    config: ForegroundEnhancementConfigV4
    contract_source: str
    raw_contract: Mapping[str, Any] = field(default_factory=dict)

    def active(self) -> bool:
        return self.config.normalized_enabled() and self.config.normalized_method() != "none"

    def to_log_dict(self) -> dict[str, Any]:
        payload = self.config.to_contract_dict()
        payload.update(
            {
                "Active": self.active(),
                "ContractSource": self.contract_source,
            }
        )
        return payload


@dataclass(frozen=True)
class TriStreamPreprocessingConfig:
    """Concrete settings needed to reproduce the tri-stream v4 representation."""

    preprocessing_contract: Mapping[str, Any]
    preprocessing_contract_name: str
    preprocessing_contract_version: str | None
    representation_kind: str
    input_mode: str
    input_keys: tuple[str, ...]
    geometry_schema: tuple[str, ...]
    geometry_dim: int
    distance_canvas_size: tuple[int, int]
    orientation_canvas_size: tuple[int, int]
    orientation_source_mode: str
    image_representation_mode: str
    orientation_context_scale: float
    clip_policy: str
    silhouette_config: SilhouetteStageConfigV4
    foreground_runtime: ForegroundEnhancementRuntimeConfig
    brightness_runtime: BrightnessNormalizationRuntimeConfig
    source_model_root: Path | None = None

    @classmethod
    def from_manifest(
        cls,
        manifest: LiveModelManifest,
        *,
        orientation_source_mode: str | None = None,
    ) -> "TriStreamPreprocessingConfig":
        """Build live preprocessing settings from a normalized model manifest."""
        preprocessing_contract = _resolve_preprocessing_contract(manifest.raw_metadata)
        if not preprocessing_contract:
            raise ValueError(
                "Live model manifest does not include a preprocessing contract; "
                "cannot configure tri-stream preprocessing."
            )

        resolved_orientation_source_mode = (
            orientation_source_mode
            or manifest.orientation_source_mode
            or resolve_orientation_source_mode(preprocessing_contract)
        )
        current = _current_representation(preprocessing_contract)
        stage = _stage_parameters(preprocessing_contract, "pack_tri_stream")
        distance_canvas_size = (
            manifest.distance_canvas_size
            or _canvas_size(current, stage)
            or (300, 300)
        )
        orientation_canvas_size = (
            manifest.orientation_canvas_size
            or _canvas_size(current, stage)
            or distance_canvas_size
        )
        geometry_schema = manifest.geometry_schema or _text_tuple(
            current.get("GeometrySchema", current.get("geometry_schema"))
        )
        if not geometry_schema:
            geometry_schema = contracts.TRI_STREAM_GEOMETRY_SCHEMA
        geometry_dim = (
            int(manifest.geometry_dim)
            if manifest.geometry_dim is not None
            else int(_int_value(current.get("GeometryDim")) or len(geometry_schema))
        )

        return cls(
            preprocessing_contract=dict(preprocessing_contract),
            preprocessing_contract_name=(
                manifest.preprocessing_contract_name
                or _text(preprocessing_contract.get("ContractVersion"))
                or contracts.PREPROCESSING_CONTRACT_NAME
            ),
            preprocessing_contract_version=_text(preprocessing_contract.get("ContractVersion")),
            representation_kind=(
                manifest.representation_kind
                or _text(current.get("Kind"))
                or contracts.TRI_STREAM_REPRESENTATION_KIND
            ),
            input_mode=manifest.input_mode or contracts.TRI_STREAM_INPUT_MODE,
            input_keys=contracts.TRI_STREAM_INPUT_KEYS,
            geometry_schema=tuple(geometry_schema),
            geometry_dim=geometry_dim,
            distance_canvas_size=distance_canvas_size,
            orientation_canvas_size=orientation_canvas_size,
            orientation_source_mode=str(resolved_orientation_source_mode),
            image_representation_mode=_resolve_image_representation_mode(
                preprocessing_contract
            ),
            orientation_context_scale=float(
                stage.get(
                    "OrientationContextScale",
                    current.get("OrientationContextScale", 1.25),
                )
            ),
            clip_policy=str(stage.get("ClipPolicy", "fail")).strip().lower() or "fail",
            silhouette_config=_silhouette_config_from_contract(preprocessing_contract),
            foreground_runtime=_resolve_foreground_enhancement_runtime(
                preprocessing_contract,
                pack_stage_name="pack_tri_stream",
            ),
            brightness_runtime=_resolve_brightness_normalization_runtime(
                preprocessing_contract,
                pack_stage_name="pack_tri_stream",
            ),
            source_model_root=manifest.model_root,
        )

    def validate(self) -> None:
        """Raise when settings cannot reproduce the old tri-stream representation."""
        if self.input_mode != contracts.TRI_STREAM_INPUT_MODE:
            raise ValueError(
                "TriStreamLivePreprocessor requires input_mode="
                f"{contracts.TRI_STREAM_INPUT_MODE!r}; got {self.input_mode!r}."
            )
        if tuple(self.input_keys) != contracts.TRI_STREAM_INPUT_KEYS:
            raise ValueError(
                "TriStreamLivePreprocessor requires live tri-stream input keys "
                f"{contracts.TRI_STREAM_INPUT_KEYS!r}; got {self.input_keys!r}."
            )
        if self.representation_kind != contracts.TRI_STREAM_REPRESENTATION_KIND:
            raise ValueError(
                "TriStreamLivePreprocessor requires representation_kind="
                f"{contracts.TRI_STREAM_REPRESENTATION_KIND!r}; got "
                f"{self.representation_kind!r}."
            )
        if self.geometry_dim != len(contracts.TRI_STREAM_GEOMETRY_SCHEMA):
            raise ValueError(
                "TriStreamLivePreprocessor requires geometry_dim="
                f"{len(contracts.TRI_STREAM_GEOMETRY_SCHEMA)}; got {self.geometry_dim}."
            )
        if tuple(self.geometry_schema) != contracts.TRI_STREAM_GEOMETRY_SCHEMA:
            raise ValueError(
                "TriStreamLivePreprocessor requires geometry_schema="
                f"{contracts.TRI_STREAM_GEOMETRY_SCHEMA!r}; got {self.geometry_schema!r}."
            )
        if self.orientation_source_mode not in SUPPORTED_ORIENTATION_SOURCE_MODES:
            raise ValueError(
                "Unsupported resolved tri-stream orientation source mode: "
                f"{self.orientation_source_mode!r}. Supported: "
                f"{SUPPORTED_ORIENTATION_SOURCE_MODES!r}."
            )
        if self.image_representation_mode not in {
            "inverted_vehicle_on_white",
            "raw_grayscale_on_white",
        }:
            raise ValueError(
                "Unsupported tri-stream image representation mode: "
                f"{self.image_representation_mode!r}."
            )
        silhouette_size = (
            int(self.silhouette_config.normalized_roi_canvas_width_px()),
            int(self.silhouette_config.normalized_roi_canvas_height_px()),
        )
        if tuple(self.distance_canvas_size) != silhouette_size:
            raise ValueError(
                "Tri-stream distance canvas must match the silhouette ROI canvas so "
                "x_distance_image remains unscaled: "
                f"distance={self.distance_canvas_size!r}, silhouette={silhouette_size!r}."
            )
        if tuple(self.orientation_canvas_size) != tuple(self.distance_canvas_size):
            raise ValueError(
                "Tri-stream orientation canvas must match the distance canvas for the "
                "current live engine contract: "
                f"orientation={self.orientation_canvas_size!r}, "
                f"distance={self.distance_canvas_size!r}."
            )
        if self.clip_policy not in {"fail", "clip"}:
            raise ValueError(f"Unsupported ClipPolicy={self.clip_policy!r}.")


def _resolve_preprocessing_contract(raw_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = (
        ("live_model_manifest", ("preprocessing_contract",)),
        ("live_model_manifest", ("PreprocessingContract",)),
        ("dataset_summary", ("preprocessing_contract",)),
        ("dataset_summary", ("PreprocessingContract",)),
        ("run_manifest", ("dataset_summary", "preprocessing_contract")),
        ("run_manifest", ("dataset_summary", "PreprocessingContract")),
        ("run_manifest", ("preprocessing_contract",)),
        ("run_manifest", ("PreprocessingContract",)),
        ("config", ("dataset_summary", "preprocessing_contract")),
        ("config", ("dataset_summary", "PreprocessingContract")),
        ("config", ("preprocessing_contract",)),
        ("config", ("PreprocessingContract",)),
    )
    for source_name, path in candidates:
        candidate = _mapping_at(_as_mapping(raw_metadata.get(source_name)), path)
        if candidate:
            return candidate
    return {}


def _stage_parameters(
    preprocessing_contract: Mapping[str, Any],
    stage_name: str,
) -> dict[str, Any]:
    stages = preprocessing_contract.get("Stages", preprocessing_contract.get("stages"))
    if not isinstance(stages, Mapping):
        return {}
    stage = stages.get(stage_name)
    return dict(stage) if isinstance(stage, Mapping) else {}


def _current_representation(preprocessing_contract: Mapping[str, Any]) -> dict[str, Any]:
    current = preprocessing_contract.get(
        "CurrentRepresentation",
        preprocessing_contract.get("current_representation"),
    )
    return dict(current) if isinstance(current, Mapping) else {}


def _brightness_contract_from_preprocessing_contract(
    preprocessing_contract: Mapping[str, Any],
    *,
    pack_stage_name: str,
) -> tuple[dict[str, Any], str]:
    stage = _stage_parameters(preprocessing_contract, pack_stage_name)
    stage_contract = stage.get("BrightnessNormalization")
    current = _current_representation(preprocessing_contract)
    current_contract = current.get("BrightnessNormalization")

    stage_payload = dict(stage_contract) if isinstance(stage_contract, Mapping) else {}
    current_payload = dict(current_contract) if isinstance(current_contract, Mapping) else {}
    if stage_payload and current_payload and stage_payload != current_payload:
        raise ValueError(
            "Brightness normalization contract mismatch between "
            f"Stages.{pack_stage_name}.BrightnessNormalization and "
            "CurrentRepresentation.BrightnessNormalization."
        )
    if stage_payload:
        return stage_payload, f"Stages.{pack_stage_name}.BrightnessNormalization"
    if current_payload:
        return current_payload, "CurrentRepresentation.BrightnessNormalization"
    return {}, "absent"


def _foreground_contract_from_preprocessing_contract(
    preprocessing_contract: Mapping[str, Any],
    *,
    pack_stage_name: str,
) -> tuple[dict[str, Any], str]:
    stage = _stage_parameters(preprocessing_contract, pack_stage_name)
    stage_contract = stage.get("ForegroundEnhancement")
    current = _current_representation(preprocessing_contract)
    current_contract = current.get("ForegroundEnhancement")

    stage_payload = dict(stage_contract) if isinstance(stage_contract, Mapping) else {}
    current_payload = dict(current_contract) if isinstance(current_contract, Mapping) else {}
    if stage_payload and current_payload and stage_payload != current_payload:
        raise ValueError(
            "Foreground enhancement contract mismatch between "
            f"Stages.{pack_stage_name}.ForegroundEnhancement and "
            "CurrentRepresentation.ForegroundEnhancement."
        )
    if stage_payload:
        return stage_payload, f"Stages.{pack_stage_name}.ForegroundEnhancement"
    if current_payload:
        return current_payload, "CurrentRepresentation.ForegroundEnhancement"
    return {}, "absent"


def _resolve_foreground_enhancement_runtime(
    preprocessing_contract: Mapping[str, Any],
    *,
    pack_stage_name: str,
) -> ForegroundEnhancementRuntimeConfig:
    payload, source = _foreground_contract_from_preprocessing_contract(
        preprocessing_contract,
        pack_stage_name=pack_stage_name,
    )
    config = (
        ForegroundEnhancementConfigV4.from_mapping(payload)
        if payload
        else ForegroundEnhancementConfigV4()
    )
    return ForegroundEnhancementRuntimeConfig(
        config=config,
        contract_source=source,
        raw_contract=payload,
    )


def _resolve_brightness_normalization_runtime(
    preprocessing_contract: Mapping[str, Any],
    *,
    pack_stage_name: str,
) -> BrightnessNormalizationRuntimeConfig:
    payload, source = _brightness_contract_from_preprocessing_contract(
        preprocessing_contract,
        pack_stage_name=pack_stage_name,
    )
    config = (
        BrightnessNormalizationConfigV4.from_mapping(payload)
        if payload
        else BrightnessNormalizationConfigV4()
    )

    explicit_mask_source = "MaskSource" in payload or "mask_source" in payload
    raw_mask_source = payload.get("MaskSource", payload.get("mask_source", ""))
    mask_source = str(raw_mask_source).strip() if explicit_mask_source else DEFAULT_BRIGHTNESS_MASK_SOURCE
    if not mask_source:
        mask_source = DEFAULT_BRIGHTNESS_MASK_SOURCE

    if config.normalized_enabled() and config.normalized_method() != "none":
        if mask_source not in _SUPPORTED_BRIGHTNESS_MASK_SOURCES:
            allowed = ", ".join(sorted(_SUPPORTED_BRIGHTNESS_MASK_SOURCES))
            raise ValueError(
                "Distance/yaw model expects brightness normalization but inference cannot "
                f"produce the requested MaskSource={mask_source!r}. "
                f"Supported mask sources: {allowed}."
            )

    return BrightnessNormalizationRuntimeConfig(
        config=config,
        contract_source=source,
        mask_source=mask_source,
        explicit_mask_source=explicit_mask_source,
        raw_contract=payload,
    )


def _resolve_image_representation_mode(preprocessing_contract: Mapping[str, Any]) -> str:
    stage = _stage_parameters(preprocessing_contract, "pack_tri_stream")
    raw = _text(stage.get("ImageRepresentationMode"))
    normalized = str(raw or "").strip().lower()
    if normalized in {
        "roi_grayscale_inverted_vehicle_on_white",
        "inverted_vehicle_on_white",
    }:
        return "inverted_vehicle_on_white"
    if normalized in {
        "roi_raw_grayscale_vehicle_on_white",
        "raw_grayscale_on_white",
    }:
        return "raw_grayscale_on_white"

    current = _current_representation(preprocessing_contract)
    polarity = str(
        _text(
            current.get("DistanceImagePolarity")
            or current.get("OrientationImagePolarity")
        )
        or ""
    ).strip()
    if polarity == "source_grayscale_vehicle_detail_on_white_background":
        return "raw_grayscale_on_white"
    return "inverted_vehicle_on_white"


def _silhouette_config_from_contract(
    preprocessing_contract: Mapping[str, Any],
) -> SilhouetteStageConfigV4:
    stage = _stage_parameters(preprocessing_contract, "silhouette")
    return SilhouetteStageConfigV4(
        representation_mode=str(stage.get("RepresentationMode", "filled")).strip().lower()
        or "filled",
        generator_id=str(stage.get("GeneratorId", "silhouette.contour_v2")).strip()
        or "silhouette.contour_v2",
        fallback_id=str(stage.get("FallbackId", "fallback.convex_hull_v1")).strip()
        or "fallback.convex_hull_v1",
        roi_padding_px=int(stage.get("ROIPaddingPx", 0)),
        roi_canvas_width_px=int(stage.get("ROICanvasWidthPx", 300)),
        roi_canvas_height_px=int(stage.get("ROICanvasHeightPx", 300)),
        blur_kernel_size=int(stage.get("BlurKernelSize", 5)),
        canny_low_threshold=int(stage.get("CannyLowThreshold", 50)),
        canny_high_threshold=int(stage.get("CannyHighThreshold", 150)),
        close_kernel_size=int(stage.get("CloseKernelSize", 1)),
        dilate_kernel_size=int(stage.get("DilateKernelSize", 1)),
        min_component_area_px=int(stage.get("MinComponentAreaPx", 50)),
        outline_thickness=int(stage.get("OutlineThicknessPx", 1)),
        fill_holes=bool(stage.get("FillHoles", True)),
        use_convex_hull_fallback=bool(stage.get("UseConvexHullFallback", True)),
        overwrite=True,
        dry_run=False,
        continue_on_error=False,
        persist_debug=bool(stage.get("PersistDebug", False)),
        sample_offset=0,
        sample_limit=1,
    )


def _canvas_size(
    current: Mapping[str, Any],
    stage: Mapping[str, Any],
) -> tuple[int, int] | None:
    width = _int_value(stage.get("CanvasWidth", current.get("CanvasWidth")))
    height = _int_value(stage.get("CanvasHeight", current.get("CanvasHeight")))
    if width is None or height is None:
        return None
    return (width, height)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_at(source: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
    current: Any = source
    for key in path:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence):
        items: list[str] = []
        for item in value:
            text = _text(item)
            if text:
                items.append(text)
        return tuple(items)
    return ()


__all__ = [
    "BrightnessNormalizationRuntimeConfig",
    "ForegroundEnhancementRuntimeConfig",
    "ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE",
    "ORIENTATION_SOURCE_RAW_GRAYSCALE",
    "ORIENTATION_SOURCE_RAW_GRAYSCALE_ON_WHITE",
    "TriStreamPreprocessingConfig",
]
