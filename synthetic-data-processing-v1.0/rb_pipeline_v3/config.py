"""Configuration dataclasses for the v3 threshold preprocessing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


_VALID_REPRESENTATION_MODES = {"outline", "filled"}
_VALID_NPY_DTYPES = {"float32", "float16", "uint8"}
_VALID_PACK_DTYPES = {"preserve", "float32", "float16", "uint8"}
_VALID_TRAINING_IMAGE_SOURCE_COLUMNS = {
    "threshold_image_filename",
    "threshold_debug_binary_filename",
    "threshold_debug_selected_component_filename",
    "threshold_debug_amalgamated_filename",
}


@dataclass(frozen=True)
class ThresholdStageConfigV3:
    """Config for v3 source PNG -> threshold PNG processing."""

    representation_mode: str

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True

    threshold_low_value: int = 128
    threshold_high_value: int = 255
    invert_selection: bool = False

    min_component_area_px: int = 1
    fill_internal_holes: bool = True
    hole_close_kernel_size: int = 3
    outline_thickness: int = 1

    persist_debug: bool = False
    keep_individual_debug_outputs: bool = True
    amalgamate_debug_outputs: bool = False

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

    def normalized_threshold_bounds(self) -> tuple[int, int]:
        low = int(self.threshold_low_value)
        high = int(self.threshold_high_value)

        low = max(0, min(255, low))
        high = max(0, min(255, high))
        if low > high:
            low, high = high, low
        return low, high

    def normalized_outline_thickness(self) -> int:
        return max(1, int(self.outline_thickness))

    def normalized_min_component_area_px(self) -> int:
        return max(1, int(self.min_component_area_px))

    def normalized_hole_close_kernel_size(self) -> int:
        kernel = max(1, int(self.hole_close_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def normalized_sample_offset(self) -> int:
        return max(0, int(self.sample_offset))

    def normalized_sample_limit(self) -> int:
        return max(0, int(self.sample_limit))

    def to_log_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NpyPackStageConfigV3:
    """Config for v3 threshold PNG -> NPY -> NPZ processing."""

    representation_mode: str

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True

    normalize: bool = False
    invert: bool = False
    npy_output_dtype: str = "float32"

    pack_output_dtype: str = "preserve"
    compress: bool = True
    shard_size: int = 0
    delete_source_npy_after_pack: bool = False
    include_optional_filename_arrays: bool = True
    training_image_source_column: str = "threshold_image_filename"

    def normalized_representation_mode(self) -> str:
        mode = str(self.representation_mode).strip().lower()
        if mode not in _VALID_REPRESENTATION_MODES:
            allowed = ", ".join(sorted(_VALID_REPRESENTATION_MODES))
            raise ValueError(
                f"Unsupported representation_mode '{self.representation_mode}'. Allowed: {allowed}."
            )
        return mode

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
class PipelineRunConfigV3:
    """Optional holder for v3 stage configs when orchestrating full runs."""

    threshold: ThresholdStageConfigV3 | None = None
    npy_pack: NpyPackStageConfigV3 | None = None


@dataclass(frozen=True)
class ShuffleStageConfigV3:
    """Config for shuffled corpus generation from existing v3 NPZ shards."""

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True
    output_root_name: str = "training-data-v3-shuffled"
    random_seed: int = 42
    compress: bool = True
    strict_unique_sample_ids: bool = True
    ledger_filename: str = "shuffle_ledger.csv"

    def to_log_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class StageSummaryV3:
    """Simple summary returned by each run-level v3 stage."""

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
