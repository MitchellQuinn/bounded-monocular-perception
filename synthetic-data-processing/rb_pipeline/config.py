"""Configuration dataclasses for the Raccoon Ball post-processing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CommonStageOptions:
    """Options shared by all stages."""

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True

    def to_log_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EdgeStageConfig(CommonStageOptions):
    """Config for source PNG -> edge PNG processing."""

    blur_kernel_size: int = 5
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150

    def normalized_blur_kernel_size(self) -> int:
        kernel = max(1, int(self.blur_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        return kernel


@dataclass(frozen=True)
class BBoxStageConfig(CommonStageOptions):
    """Config for edge PNG -> bbox PNG processing."""

    foreground_threshold: int = 250
    line_thickness: int = 3
    padding_px: int = 0
    post_draw_blur: bool = False
    post_draw_blur_kernel_size: int = 3

    def normalized_blur_kernel_size(self) -> int:
        kernel = max(1, int(self.post_draw_blur_kernel_size))
        if kernel % 2 == 0:
            kernel += 1
        return kernel


@dataclass(frozen=True)
class NpyStageConfig(CommonStageOptions):
    """Config for bbox PNG -> NPY processing."""

    normalize: bool = True
    invert: bool = True
    output_dtype: str = "float32"


@dataclass(frozen=True)
class PackStageConfig(CommonStageOptions):
    """Config for NPY -> NPZ processing."""

    output_dtype: str = "preserve"
    compress: bool = True
    shard_size: int = 0
    delete_source_npy_after_pack: bool = False
    include_optional_filename_arrays: bool = True


@dataclass(frozen=True)
class ShuffleStageConfig(CommonStageOptions):
    """Config for shuffled corpus generation from existing NPZ shards."""

    output_root_name: str = "training-data-shuffled"
    random_seed: int = 42
    compress: bool = True
    strict_unique_sample_ids: bool = True
    ledger_filename: str = "shuffle_ledger.csv"


@dataclass
class StageSummary:
    """Simple summary returned by each run-level stage function."""

    run_name: str
    stage_name: str
    total_rows: int
    successful_rows: int
    failed_rows: int
    skipped_rows: int
    output_path: str
    log_path: str | None
    dry_run: bool

    def as_dict(self) -> dict:
        return asdict(self)
