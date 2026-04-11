"""Configuration dataclasses for the extensible v2 preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


_VALID_REPRESENTATION_MODES = {"outline", "filled"}
_VALID_NPY_DTYPES = {"float32", "float16", "uint8"}
_VALID_PACK_DTYPES = {"preserve", "float32", "float16", "uint8"}
_VALID_TRAINING_IMAGE_SOURCE_COLUMNS = {
    "silhouette_image_filename",
    "silhouette_edge_debug_filename",
    "silhouette_debug_raw_edge_filename",
    "silhouette_debug_post_morph_filename",
    "silhouette_debug_components_mask_filename",
    "silhouette_debug_selected_component_filename",
    "silhouette_debug_external_contour_filename",
    "silhouette_debug_final_filled_filename",
    "silhouette_debug_fallback_hull_filename",
    "silhouette_debug_amalgamated_filename",
}


@dataclass(frozen=True)
class SilhouetteStageConfigV2:
    """Config for v2 source PNG -> silhouette PNG processing."""

    representation_mode: str
    generator_id: str
    fallback_id: str

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True

    blur_kernel_size: int = 5
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150

    close_kernel_size: int = 1
    dilate_kernel_size: int = 1
    min_component_area_px: int = 50
    outline_thickness: int = 1
    fill_holes: bool = True
    use_convex_hull_fallback: bool = True

    blob_border_fraction: float = 0.05
    blob_hue_delta: float = 16.0
    blob_sat_min: float = 42.0
    blob_val_min: float = 50.0
    blob_bright_val_min: float = 95.0
    blob_bright_sat_min: float = 35.0
    blob_use_bright_rescue: bool = True
    blob_reject_large_border_components: bool = True
    blob_large_border_component_ratio: float = 0.01

    persist_edge_debug: bool = False
    debug_persist: bool | None = None
    amalgamate_debug_outputs: bool = False
    keep_individual_debug_outputs: bool = True

    sample_offset: int = 0
    sample_limit: int = 0

    def normalized_representation_mode(self) -> str:
        mode = str(self.representation_mode).strip().lower()
        if mode not in _VALID_REPRESENTATION_MODES:
            allowed = ", ".join(sorted(_VALID_REPRESENTATION_MODES))
            raise ValueError(
                f"Unsupported representation_mode '{self.representation_mode}'. Allowed: {allowed}."
            )
        return mode

    def normalized_generator_id(self) -> str:
        value = str(self.generator_id).strip()
        if not value:
            raise ValueError("generator_id is required.")
        return value

    def normalized_fallback_id(self) -> str:
        value = str(self.fallback_id).strip()
        if not value:
            raise ValueError("fallback_id is required.")
        return value

    def normalized_blur_kernel_size(self) -> int:
        kernel = max(1, int(self.blur_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def normalized_close_kernel_size(self) -> int:
        return max(1, int(self.close_kernel_size))

    def normalized_dilate_kernel_size(self) -> int:
        return max(1, int(self.dilate_kernel_size))

    def normalized_outline_thickness(self) -> int:
        return max(1, int(self.outline_thickness))

    def normalized_min_component_area_px(self) -> int:
        return max(1, int(self.min_component_area_px))

    def normalized_debug_persist(self) -> bool:
        if self.debug_persist is None:
            return bool(self.persist_edge_debug)
        return bool(self.debug_persist)

    def blob_params(self) -> dict[str, object]:
        return {
            "blob_border_fraction": max(0.01, min(0.25, float(self.blob_border_fraction))),
            "blob_hue_delta": max(1.0, min(90.0, float(self.blob_hue_delta))),
            "blob_sat_min": max(0.0, min(255.0, float(self.blob_sat_min))),
            "blob_val_min": max(0.0, min(255.0, float(self.blob_val_min))),
            "blob_bright_val_min": max(0.0, min(255.0, float(self.blob_bright_val_min))),
            "blob_bright_sat_min": max(0.0, min(255.0, float(self.blob_bright_sat_min))),
            "blob_use_bright_rescue": bool(self.blob_use_bright_rescue),
            "blob_reject_large_border_components": bool(self.blob_reject_large_border_components),
            "blob_large_border_component_ratio": max(
                0.0,
                min(0.5, float(self.blob_large_border_component_ratio)),
            ),
        }

    def normalized_sample_offset(self) -> int:
        return max(0, int(self.sample_offset))

    def normalized_sample_limit(self) -> int:
        return max(0, int(self.sample_limit))

    def to_log_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NpyPackStageConfigV2:
    """Config for v2 silhouette PNG -> NPY -> NPZ processing."""

    representation_mode: str
    array_exporter_id: str

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True

    normalize: bool = True
    invert: bool = True
    npy_output_dtype: str = "float32"

    pack_output_dtype: str = "preserve"
    compress: bool = True
    shard_size: int = 0
    delete_source_npy_after_pack: bool = False
    include_optional_filename_arrays: bool = True
    training_image_source_column: str = "silhouette_image_filename"

    def normalized_representation_mode(self) -> str:
        mode = str(self.representation_mode).strip().lower()
        if mode not in _VALID_REPRESENTATION_MODES:
            allowed = ", ".join(sorted(_VALID_REPRESENTATION_MODES))
            raise ValueError(
                f"Unsupported representation_mode '{self.representation_mode}'. Allowed: {allowed}."
            )
        return mode

    def normalized_array_exporter_id(self) -> str:
        value = str(self.array_exporter_id).strip()
        if not value:
            raise ValueError("array_exporter_id is required.")
        return value

    def normalized_npy_output_dtype(self) -> str:
        dtype_name = str(self.npy_output_dtype).strip().lower()
        if dtype_name not in _VALID_NPY_DTYPES:
            allowed = ", ".join(sorted(_VALID_NPY_DTYPES))
            raise ValueError(f"Unsupported npy_output_dtype '{self.npy_output_dtype}'. Allowed: {allowed}.")
        return dtype_name

    def normalized_pack_output_dtype(self) -> str:
        dtype_name = str(self.pack_output_dtype).strip().lower()
        if dtype_name not in _VALID_PACK_DTYPES:
            allowed = ", ".join(sorted(_VALID_PACK_DTYPES))
            raise ValueError(f"Unsupported pack_output_dtype '{self.pack_output_dtype}'. Allowed: {allowed}.")
        return dtype_name

    def normalized_shard_size(self) -> int:
        size = int(self.shard_size)
        if size < 0:
            raise ValueError("shard_size must be >= 0")
        return size

    def normalized_training_image_source_column(self) -> str:
        value = str(self.training_image_source_column).strip()
        if not value:
            raise ValueError("training_image_source_column is required.")
        if value not in _VALID_TRAINING_IMAGE_SOURCE_COLUMNS:
            allowed = ", ".join(sorted(_VALID_TRAINING_IMAGE_SOURCE_COLUMNS))
            raise ValueError(
                f"Unsupported training_image_source_column '{self.training_image_source_column}'. "
                f"Allowed: {allowed}."
            )
        return value

    def to_log_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PipelineRunConfigV2:
    """Optional holder for v2 stage configs when orchestrating full runs."""

    silhouette: SilhouetteStageConfigV2 | None = None
    npy_pack: NpyPackStageConfigV2 | None = None


@dataclass(frozen=True)
class ShuffleStageConfigV2:
    """Config for shuffled corpus generation from existing v2 NPZ shards."""

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True
    output_root_name: str = "training-data-v2-shuffled"
    random_seed: int = 42
    compress: bool = True
    strict_unique_sample_ids: bool = True
    ledger_filename: str = "shuffle_ledger.csv"

    def to_log_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class StageSummaryV2:
    """Simple summary returned by each run-level v2 stage."""

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
