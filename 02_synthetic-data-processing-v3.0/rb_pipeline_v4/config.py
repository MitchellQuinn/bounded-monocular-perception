"""Configuration dataclasses for the v4 dual-stream synthetic preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


_VALID_REPRESENTATION_MODES = {"outline", "filled"}
_VALID_CLIP_POLICIES = {"fail", "clip"}
_VALID_DETECTOR_BACKENDS = {"yolo", "edge"}


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
    include_v1_compat_arrays: bool = False
    include_optional_metadata_arrays: bool = True
    use_intermediate_npy: bool = True
    delete_source_npy_after_pack: bool = True

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

    def normalized_shard_size(self) -> int:
        size = int(self.shard_size)
        if size < 0:
            raise ValueError("shard_size must be >= 0")
        return size

    def normalized_sample_offset(self) -> int:
        return max(0, int(self.sample_offset))

    def normalized_sample_limit(self) -> int:
        return max(0, int(self.sample_limit))

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["clip_policy"] = self.normalized_clip_policy()
        payload["canvas_width_px"] = self.normalized_canvas_width_px()
        payload["canvas_height_px"] = self.normalized_canvas_height_px()
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
