"""Raccoon Ball synthetic preprocessing pipeline v4 (dual-stream)."""

from .config import (
    BrightnessNormalizationConfigV4,
    DetectStageConfigV4,
    PackDualStreamStageConfigV4,
    PackTriStreamStageConfigV4,
    ShuffleStageConfigV4,
    SilhouetteStageConfigV4,
    StageSummaryV4,
)
from .brightness_normalization import BrightnessNormalizationResultV4, apply_brightness_normalization_v4
from .detector import EdgeRoiDetector, UltralyticsYoloDetector
from .detect_stage import run_detect_stage_v4
from .input_corpus_shuffle import (
    InputCorpusSummary,
    InputCorpusShuffleError,
    InputCorpusShuffleResult,
    InputCorpusShuffleValidationError,
    default_input_shuffle_destination,
    discover_input_corpuses,
    parse_shuffle_seed,
    shuffle_input_corpus,
)
from .pack_dual_stream_stage import run_pack_dual_stream_stage_v4
from .pack_tri_stream_stage import build_tri_stream_sample_preview, run_pack_tri_stream_stage_v4
from .pipeline import (
    STAGE_ORDER_V4,
    TRI_STREAM_STAGE_ORDER,
    run_tri_stream_stage_for_run,
    run_tri_stream_stage_sequence_for_run,
    run_v4_stage_for_run,
    run_v4_stage_sequence_for_run,
)
from .shuffle_stage import run_shuffle_stage_v4
from .silhouette_stage import run_silhouette_stage_v4
from .tri_stream_control import (
    TriStreamPreviewResult,
    TriStreamRunStatus,
    build_pack_tri_stream_config,
    discover_tri_stream_runs,
    infer_tri_stream_run_canvas_size,
    preview_tri_stream_sample,
    run_tri_stream_pack,
    tri_stream_output_root,
)

__all__ = [
    "DetectStageConfigV4",
    "BrightnessNormalizationConfigV4",
    "BrightnessNormalizationResultV4",
    "EdgeRoiDetector",
    "InputCorpusSummary",
    "InputCorpusShuffleError",
    "InputCorpusShuffleResult",
    "InputCorpusShuffleValidationError",
    "PackDualStreamStageConfigV4",
    "PackTriStreamStageConfigV4",
    "ShuffleStageConfigV4",
    "SilhouetteStageConfigV4",
    "StageSummaryV4",
    "STAGE_ORDER_V4",
    "TRI_STREAM_STAGE_ORDER",
    "TriStreamPreviewResult",
    "TriStreamRunStatus",
    "default_input_shuffle_destination",
    "apply_brightness_normalization_v4",
    "build_tri_stream_sample_preview",
    "build_pack_tri_stream_config",
    "discover_tri_stream_runs",
    "infer_tri_stream_run_canvas_size",
    "discover_input_corpuses",
    "parse_shuffle_seed",
    "preview_tri_stream_sample",
    "run_detect_stage_v4",
    "run_pack_dual_stream_stage_v4",
    "run_pack_tri_stream_stage_v4",
    "run_tri_stream_stage_for_run",
    "run_tri_stream_stage_sequence_for_run",
    "run_tri_stream_pack",
    "run_shuffle_stage_v4",
    "run_silhouette_stage_v4",
    "run_v4_stage_for_run",
    "run_v4_stage_sequence_for_run",
    "shuffle_input_corpus",
    "tri_stream_output_root",
    "UltralyticsYoloDetector",
]
