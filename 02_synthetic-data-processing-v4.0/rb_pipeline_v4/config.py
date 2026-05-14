"""Configuration dataclasses for the v4 dual-stream synthetic preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Mapping


_VALID_REPRESENTATION_MODES = {"outline", "filled"}
_VALID_IMAGE_REPRESENTATION_MODES = {"inverted_vehicle_on_white", "raw_grayscale_on_white"}
_VALID_CLIP_POLICIES = {"fail", "clip"}
_VALID_DETECTOR_BACKENDS = {"yolo", "edge"}
_VALID_BRIGHTNESS_NORMALIZATION_METHODS = {"none", "masked_median_darkness_gain"}
_VALID_FOREGROUND_ENHANCEMENT_METHODS = {"none", "masked_median_darkness_gain"}
_VALID_EMPTY_MASK_POLICIES = {"skip", "fail"}


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n", ""}:
        return False
    return bool(value)


@dataclass(frozen=True)
class BrightnessNormalizationConfigV4:
    """Config for deterministic foreground-only brightness normalization."""

    enabled: bool = False
    method: str = "none"
    target_median_darkness: float = 0.55
    min_gain: float = 0.5
    max_gain: float = 2.0
    epsilon: float = 1e-6
    empty_mask_policy: str = "skip"

    def normalized_enabled(self) -> bool:
        return bool(self.enabled)

    def normalized_method(self) -> str:
        value = str(self.method).strip().lower()
        if value not in _VALID_BRIGHTNESS_NORMALIZATION_METHODS:
            allowed = ", ".join(sorted(_VALID_BRIGHTNESS_NORMALIZATION_METHODS))
            raise ValueError(f"Unsupported brightness normalization method '{self.method}'. Allowed: {allowed}.")
        return value

    def normalized_target_median_darkness(self) -> float:
        value = float(self.target_median_darkness)
        if not math.isfinite(value) or value < 0.05 or value > 0.95:
            raise ValueError("target_median_darkness must be finite and in [0.05, 0.95]")
        return value

    def normalized_min_gain(self) -> float:
        value = float(self.min_gain)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("min_gain must be finite and > 0")
        return value

    def normalized_max_gain(self) -> float:
        value = float(self.max_gain)
        min_gain = self.normalized_min_gain()
        if not math.isfinite(value) or value < min_gain:
            raise ValueError("max_gain must be finite and >= min_gain")
        return value

    def normalized_epsilon(self) -> float:
        value = float(self.epsilon)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("epsilon must be finite and > 0")
        return value

    def normalized_empty_mask_policy(self) -> str:
        value = str(self.empty_mask_policy).strip().lower()
        if value not in _VALID_EMPTY_MASK_POLICIES:
            allowed = ", ".join(sorted(_VALID_EMPTY_MASK_POLICIES))
            raise ValueError(f"Unsupported empty_mask_policy '{self.empty_mask_policy}'. Allowed: {allowed}.")
        return value

    def active_method(self) -> str:
        method = self.normalized_method()
        return method if self.normalized_enabled() else "none"

    def to_contract_dict(self) -> dict[str, object]:
        return {
            "Enabled": self.normalized_enabled(),
            "Method": self.normalized_method(),
            "TargetMedianDarkness": self.normalized_target_median_darkness(),
            "MinGain": self.normalized_min_gain(),
            "MaxGain": self.normalized_max_gain(),
            "Epsilon": self.normalized_epsilon(),
            "EmptyMaskPolicy": self.normalized_empty_mask_policy(),
        }

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["enabled"] = self.normalized_enabled()
        payload["method"] = self.normalized_method()
        payload["target_median_darkness"] = self.normalized_target_median_darkness()
        payload["min_gain"] = self.normalized_min_gain()
        payload["max_gain"] = self.normalized_max_gain()
        payload["epsilon"] = self.normalized_epsilon()
        payload["empty_mask_policy"] = self.normalized_empty_mask_policy()
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "BrightnessNormalizationConfigV4":
        def read(*names: str, default: object) -> object:
            for name in names:
                if name in payload:
                    return payload[name]
            return default

        return cls(
            enabled=_coerce_bool(read("enabled", "Enabled", default=False)),
            method=str(read("method", "Method", default="none")),
            target_median_darkness=float(
                read("target_median_darkness", "TargetMedianDarkness", default=0.55)
            ),
            min_gain=float(read("min_gain", "MinGain", default=0.5)),
            max_gain=float(read("max_gain", "MaxGain", default=2.0)),
            epsilon=float(read("epsilon", "Epsilon", default=1e-6)),
            empty_mask_policy=str(read("empty_mask_policy", "EmptyMaskPolicy", default="skip")),
        )


@dataclass(frozen=True)
class ForegroundEnhancementConfigV4:
    """Config for deterministic foreground-only representation strengthening."""

    enabled: bool = False
    method: str = "none"
    target_median_darkness: float = 0.70
    min_gain: float = 1.0
    max_gain: float = 3.0
    epsilon: float = 1e-6
    empty_mask_policy: str = "skip"

    def normalized_enabled(self) -> bool:
        return bool(self.enabled)

    def normalized_method(self) -> str:
        value = str(self.method).strip().lower()
        if value not in _VALID_FOREGROUND_ENHANCEMENT_METHODS:
            allowed = ", ".join(sorted(_VALID_FOREGROUND_ENHANCEMENT_METHODS))
            raise ValueError(f"Unsupported foreground enhancement method '{self.method}'. Allowed: {allowed}.")
        return value

    def normalized_target_median_darkness(self) -> float:
        value = float(self.target_median_darkness)
        if not math.isfinite(value) or value < 0.05 or value > 0.95:
            raise ValueError("target_median_darkness must be finite and in [0.05, 0.95]")
        return value

    def normalized_min_gain(self) -> float:
        value = float(self.min_gain)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("min_gain must be finite and > 0")
        return value

    def normalized_max_gain(self) -> float:
        value = float(self.max_gain)
        min_gain = self.normalized_min_gain()
        if not math.isfinite(value) or value < min_gain:
            raise ValueError("max_gain must be finite and >= min_gain")
        return value

    def normalized_epsilon(self) -> float:
        value = float(self.epsilon)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("epsilon must be finite and > 0")
        return value

    def normalized_empty_mask_policy(self) -> str:
        value = str(self.empty_mask_policy).strip().lower()
        if value not in _VALID_EMPTY_MASK_POLICIES:
            allowed = ", ".join(sorted(_VALID_EMPTY_MASK_POLICIES))
            raise ValueError(f"Unsupported empty_mask_policy '{self.empty_mask_policy}'. Allowed: {allowed}.")
        return value

    def active_method(self) -> str:
        method = self.normalized_method()
        return method if self.normalized_enabled() else "none"

    def to_contract_dict(self) -> dict[str, object]:
        return {
            "Enabled": self.normalized_enabled(),
            "Method": self.normalized_method(),
            "TargetMedianDarkness": self.normalized_target_median_darkness(),
            "MinGain": self.normalized_min_gain(),
            "MaxGain": self.normalized_max_gain(),
            "Epsilon": self.normalized_epsilon(),
            "EmptyMaskPolicy": self.normalized_empty_mask_policy(),
        }

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["enabled"] = self.normalized_enabled()
        payload["method"] = self.normalized_method()
        payload["target_median_darkness"] = self.normalized_target_median_darkness()
        payload["min_gain"] = self.normalized_min_gain()
        payload["max_gain"] = self.normalized_max_gain()
        payload["epsilon"] = self.normalized_epsilon()
        payload["empty_mask_policy"] = self.normalized_empty_mask_policy()
        return payload

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ForegroundEnhancementConfigV4":
        def read(*names: str, default: object) -> object:
            for name in names:
                if name in payload:
                    return payload[name]
            return default

        return cls(
            enabled=_coerce_bool(read("enabled", "Enabled", default=False)),
            method=str(read("method", "Method", default="none")),
            target_median_darkness=float(
                read("target_median_darkness", "TargetMedianDarkness", default=0.70)
            ),
            min_gain=float(read("min_gain", "MinGain", default=1.0)),
            max_gain=float(read("max_gain", "MaxGain", default=3.0)),
            epsilon=float(read("epsilon", "Epsilon", default=1e-6)),
            empty_mask_policy=str(read("empty_mask_policy", "EmptyMaskPolicy", default="skip")),
        )


@dataclass(frozen=True)
class DetectStageConfigV4:
    """Config for detector-based defender ROI stage (YOLO or edge)."""

    detector_backend: str = "edge"

    model_path: str = ""
    default_model_ref: str = "yolov8n.pt"
    defender_class_ids: tuple[int, ...] = ()
    defender_class_names: tuple[str, ...] = ()
    conf_threshold: float = 0.10
    iou_threshold: float = 0.70
    max_det: int = 32
    imgsz: int = 1280
    device: str = ""

    edge_blur_kernel_size: int = 5
    edge_canny_low_threshold: int = 50
    edge_canny_high_threshold: int = 150
    edge_foreground_threshold: int = 250
    edge_padding_px: int = 0
    edge_min_foreground_px: int = 16
    edge_close_kernel_size: int = 1
    edge_ignore_border_px: int = 0

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True
    persist_debug: bool = False

    sample_offset: int = 0
    sample_limit: int = 0

    def normalized_detector_backend(self) -> str:
        backend = str(self.detector_backend).strip().lower()
        if backend not in _VALID_DETECTOR_BACKENDS:
            allowed = ", ".join(sorted(_VALID_DETECTOR_BACKENDS))
            raise ValueError(f"Unsupported detector_backend '{self.detector_backend}'. Allowed: {allowed}.")
        return backend

    def normalized_model_path(self) -> str:
        return str(self.model_path).strip()

    def normalized_default_model_ref(self) -> str:
        value = str(self.default_model_ref).strip()
        return value

    def normalized_defender_class_ids(self) -> tuple[int, ...]:
        values = []
        for value in self.defender_class_ids:
            values.append(int(value))
        return tuple(sorted(set(values)))

    def normalized_defender_class_names(self) -> tuple[str, ...]:
        values = []
        for value in self.defender_class_names:
            text = str(value).strip().lower()
            if text:
                values.append(text)
        return tuple(sorted(set(values)))

    def normalized_conf_threshold(self) -> float:
        return max(0.0, min(1.0, float(self.conf_threshold)))

    def normalized_iou_threshold(self) -> float:
        return max(0.0, min(1.0, float(self.iou_threshold)))

    def normalized_max_det(self) -> int:
        return max(1, int(self.max_det))

    def normalized_imgsz(self) -> int:
        return max(32, int(self.imgsz))

    def normalized_device(self) -> str | None:
        value = str(self.device).strip()
        return value or None

    def normalized_edge_blur_kernel_size(self) -> int:
        kernel = max(1, int(self.edge_blur_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def normalized_edge_canny_low_threshold(self) -> int:
        return max(0, min(255, int(self.edge_canny_low_threshold)))

    def normalized_edge_canny_high_threshold(self) -> int:
        return max(0, min(255, int(self.edge_canny_high_threshold)))

    def normalized_edge_foreground_threshold(self) -> int:
        return max(0, min(255, int(self.edge_foreground_threshold)))

    def normalized_edge_padding_px(self) -> int:
        return max(0, int(self.edge_padding_px))

    def normalized_edge_min_foreground_px(self) -> int:
        return max(1, int(self.edge_min_foreground_px))

    def normalized_edge_close_kernel_size(self) -> int:
        return max(1, int(self.edge_close_kernel_size))

    def normalized_edge_ignore_border_px(self) -> int:
        return max(0, int(self.edge_ignore_border_px))

    def normalized_sample_offset(self) -> int:
        return max(0, int(self.sample_offset))

    def normalized_sample_limit(self) -> int:
        return max(0, int(self.sample_limit))

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["detector_backend"] = self.normalized_detector_backend()
        payload["model_path"] = self.normalized_model_path()
        payload["default_model_ref"] = self.normalized_default_model_ref()
        payload["defender_class_ids"] = self.normalized_defender_class_ids()
        payload["defender_class_names"] = self.normalized_defender_class_names()
        payload["edge_ignore_border_px"] = self.normalized_edge_ignore_border_px()
        return payload


@dataclass(frozen=True)
class SilhouetteStageConfigV4:
    """Config for ROI silhouette extraction after detection."""

    representation_mode: str = "filled"
    generator_id: str = "silhouette.contour_v2"
    fallback_id: str = "fallback.convex_hull_v1"

    roi_padding_px: int = 0
    roi_canvas_width_px: int = 224
    roi_canvas_height_px: int = 224
    blur_kernel_size: int = 5
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    close_kernel_size: int = 1
    dilate_kernel_size: int = 1
    min_component_area_px: int = 50
    outline_thickness: int = 1
    fill_holes: bool = True
    use_convex_hull_fallback: bool = True

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True
    persist_debug: bool = False

    sample_offset: int = 0
    sample_limit: int = 0

    def normalized_representation_mode(self) -> str:
        mode = str(self.representation_mode).strip().lower()
        if mode not in _VALID_REPRESENTATION_MODES:
            allowed = ", ".join(sorted(_VALID_REPRESENTATION_MODES))
            raise ValueError(f"Unsupported representation_mode '{self.representation_mode}'. Allowed: {allowed}.")
        return mode

    def normalized_generator_id(self) -> str:
        value = str(self.generator_id).strip()
        if not value:
            raise ValueError("generator_id cannot be blank")
        return value

    def normalized_fallback_id(self) -> str:
        value = str(self.fallback_id).strip()
        if not value:
            raise ValueError("fallback_id cannot be blank")
        return value

    def normalized_roi_padding_px(self) -> int:
        return max(0, int(self.roi_padding_px))

    def normalized_roi_canvas_width_px(self) -> int:
        return max(32, int(self.roi_canvas_width_px))

    def normalized_roi_canvas_height_px(self) -> int:
        return max(32, int(self.roi_canvas_height_px))

    def normalized_blur_kernel_size(self) -> int:
        kernel = max(1, int(self.blur_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def normalized_close_kernel_size(self) -> int:
        return max(1, int(self.close_kernel_size))

    def normalized_dilate_kernel_size(self) -> int:
        return max(1, int(self.dilate_kernel_size))

    def normalized_min_component_area_px(self) -> int:
        return max(1, int(self.min_component_area_px))

    def normalized_outline_thickness(self) -> int:
        return max(1, int(self.outline_thickness))

    def normalized_sample_offset(self) -> int:
        return max(0, int(self.sample_offset))

    def normalized_sample_limit(self) -> int:
        return max(0, int(self.sample_limit))

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["representation_mode"] = self.normalized_representation_mode()
        payload["generator_id"] = self.normalized_generator_id()
        payload["fallback_id"] = self.normalized_fallback_id()
        return payload


@dataclass(frozen=True)
class PackDualStreamStageConfigV4:
    """Config for assembling dual-stream training shards."""

    canvas_width_px: int = 224
    canvas_height_px: int = 224
    clip_policy: str = "fail"
    image_representation_mode: str = "inverted_vehicle_on_white"
    foreground_enhancement: ForegroundEnhancementConfigV4 | Mapping[str, object] | None = field(
        default_factory=ForegroundEnhancementConfigV4
    )
    include_v1_compat_arrays: bool = False
    include_optional_metadata_arrays: bool = True
    use_intermediate_npy: bool = True
    delete_source_npy_after_pack: bool = True
    brightness_normalization: BrightnessNormalizationConfigV4 | Mapping[str, object] | None = field(
        default_factory=BrightnessNormalizationConfigV4
    )

    shard_size: int = 8192
    compress: bool = True
    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True

    sample_offset: int = 0
    sample_limit: int = 0

    def normalized_canvas_width_px(self) -> int:
        return max(32, int(self.canvas_width_px))

    def normalized_canvas_height_px(self) -> int:
        return max(32, int(self.canvas_height_px))

    def normalized_clip_policy(self) -> str:
        value = str(self.clip_policy).strip().lower()
        if value not in _VALID_CLIP_POLICIES:
            allowed = ", ".join(sorted(_VALID_CLIP_POLICIES))
            raise ValueError(f"Unsupported clip_policy '{self.clip_policy}'. Allowed: {allowed}.")
        return value

    def normalized_image_representation_mode(self) -> str:
        value = str(self.image_representation_mode).strip().lower()
        if value not in _VALID_IMAGE_REPRESENTATION_MODES:
            allowed = ", ".join(sorted(_VALID_IMAGE_REPRESENTATION_MODES))
            raise ValueError(
                f"Unsupported image_representation_mode '{self.image_representation_mode}'. Allowed: {allowed}."
            )
        return value

    def normalized_shard_size(self) -> int:
        size = int(self.shard_size)
        if size < 0:
            raise ValueError("shard_size must be >= 0")
        return size

    def normalized_brightness_normalization(self) -> BrightnessNormalizationConfigV4:
        config = self.brightness_normalization
        if config is None:
            return BrightnessNormalizationConfigV4()
        if isinstance(config, BrightnessNormalizationConfigV4):
            return config
        if isinstance(config, Mapping):
            return BrightnessNormalizationConfigV4.from_mapping(config)
        raise TypeError(
            "brightness_normalization must be a BrightnessNormalizationConfigV4, mapping, or None"
        )

    def normalized_foreground_enhancement(self) -> ForegroundEnhancementConfigV4:
        config = self.foreground_enhancement
        if config is None:
            return ForegroundEnhancementConfigV4()
        if isinstance(config, ForegroundEnhancementConfigV4):
            return config
        if isinstance(config, Mapping):
            return ForegroundEnhancementConfigV4.from_mapping(config)
        raise TypeError(
            "foreground_enhancement must be a ForegroundEnhancementConfigV4, mapping, or None"
        )

    def normalized_sample_offset(self) -> int:
        return max(0, int(self.sample_offset))

    def normalized_sample_limit(self) -> int:
        return max(0, int(self.sample_limit))

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["clip_policy"] = self.normalized_clip_policy()
        payload["image_representation_mode"] = self.normalized_image_representation_mode()
        payload["canvas_width_px"] = self.normalized_canvas_width_px()
        payload["canvas_height_px"] = self.normalized_canvas_height_px()
        payload["foreground_enhancement"] = self.normalized_foreground_enhancement().to_log_dict()
        payload["brightness_normalization"] = self.normalized_brightness_normalization().to_log_dict()
        return payload


@dataclass(frozen=True)
class PackTriStreamStageConfigV4:
    """Config for assembling tri-stream training shards."""

    canvas_width_px: int = 300
    canvas_height_px: int = 300
    clip_policy: str = "fail"
    image_representation_mode: str = "inverted_vehicle_on_white"
    foreground_enhancement: ForegroundEnhancementConfigV4 | Mapping[str, object] | None = field(
        default_factory=ForegroundEnhancementConfigV4
    )
    include_v1_compat_arrays: bool = False
    include_optional_metadata_arrays: bool = True
    use_intermediate_npy: bool = True
    delete_source_npy_after_pack: bool = True
    orientation_context_scale: float = 1.25
    brightness_normalization: BrightnessNormalizationConfigV4 | Mapping[str, object] | None = field(
        default_factory=BrightnessNormalizationConfigV4
    )

    shard_size: int = 8192
    compress: bool = True
    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True

    sample_offset: int = 0
    sample_limit: int = 0

    def normalized_canvas_width_px(self) -> int:
        return max(32, int(self.canvas_width_px))

    def normalized_canvas_height_px(self) -> int:
        return max(32, int(self.canvas_height_px))

    def normalized_clip_policy(self) -> str:
        value = str(self.clip_policy).strip().lower()
        if value not in _VALID_CLIP_POLICIES:
            allowed = ", ".join(sorted(_VALID_CLIP_POLICIES))
            raise ValueError(f"Unsupported clip_policy '{self.clip_policy}'. Allowed: {allowed}.")
        return value

    def normalized_image_representation_mode(self) -> str:
        value = str(self.image_representation_mode).strip().lower()
        if value not in _VALID_IMAGE_REPRESENTATION_MODES:
            allowed = ", ".join(sorted(_VALID_IMAGE_REPRESENTATION_MODES))
            raise ValueError(
                f"Unsupported image_representation_mode '{self.image_representation_mode}'. Allowed: {allowed}."
            )
        return value

    def normalized_shard_size(self) -> int:
        size = int(self.shard_size)
        if size < 0:
            raise ValueError("shard_size must be >= 0")
        return size

    def normalized_orientation_context_scale(self) -> float:
        value = float(self.orientation_context_scale)
        if not math.isfinite(value) or value < 1.0:
            raise ValueError("orientation_context_scale must be finite and >= 1.0")
        return value

    def normalized_brightness_normalization(self) -> BrightnessNormalizationConfigV4:
        config = self.brightness_normalization
        if config is None:
            return BrightnessNormalizationConfigV4()
        if isinstance(config, BrightnessNormalizationConfigV4):
            return config
        if isinstance(config, Mapping):
            return BrightnessNormalizationConfigV4.from_mapping(config)
        raise TypeError(
            "brightness_normalization must be a BrightnessNormalizationConfigV4, mapping, or None"
        )

    def normalized_foreground_enhancement(self) -> ForegroundEnhancementConfigV4:
        config = self.foreground_enhancement
        if config is None:
            return ForegroundEnhancementConfigV4()
        if isinstance(config, ForegroundEnhancementConfigV4):
            return config
        if isinstance(config, Mapping):
            return ForegroundEnhancementConfigV4.from_mapping(config)
        raise TypeError(
            "foreground_enhancement must be a ForegroundEnhancementConfigV4, mapping, or None"
        )

    def normalized_sample_offset(self) -> int:
        return max(0, int(self.sample_offset))

    def normalized_sample_limit(self) -> int:
        return max(0, int(self.sample_limit))

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["clip_policy"] = self.normalized_clip_policy()
        payload["image_representation_mode"] = self.normalized_image_representation_mode()
        payload["canvas_width_px"] = self.normalized_canvas_width_px()
        payload["canvas_height_px"] = self.normalized_canvas_height_px()
        payload["orientation_context_scale"] = self.normalized_orientation_context_scale()
        payload["foreground_enhancement"] = self.normalized_foreground_enhancement().to_log_dict()
        payload["brightness_normalization"] = self.normalized_brightness_normalization().to_log_dict()
        return payload


@dataclass(frozen=True)
class ShuffleStageConfigV4:
    """Config for optional post-pack corpus shuffling."""

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True
    output_root_name: str = "training-data-v4-shuffled"
    random_seed: int = 42
    compress: bool = True
    strict_unique_sample_ids: bool = True
    ledger_filename: str = "shuffle_ledger.csv"

    def normalized_output_root_name(self) -> str:
        value = str(self.output_root_name).strip()
        if not value:
            raise ValueError("output_root_name cannot be blank")
        return value

    def normalized_ledger_filename(self) -> str:
        value = str(self.ledger_filename).strip()
        if not value:
            raise ValueError("ledger_filename cannot be blank")
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError("ledger_filename must be a simple relative filename")
        return value

    def to_log_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class StageSummaryV4:
    """Standard summary emitted by each v4 run-level stage."""

    run_name: str
    stage_name: str
    total_rows: int
    successful_rows: int
    failed_rows: int
    skipped_rows: int
    output_path: str
    log_path: str | None
    dry_run: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
