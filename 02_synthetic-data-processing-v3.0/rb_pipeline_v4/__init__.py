"""Raccoon Ball synthetic preprocessing pipeline v4 (dual-stream)."""

from .config import (
    DetectStageConfigV4,
    PackDualStreamStageConfigV4,
    ShuffleStageConfigV4,
    SilhouetteStageConfigV4,
    StageSummaryV4,
)
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
from .pipeline import STAGE_ORDER_V4, run_v4_stage_for_run, run_v4_stage_sequence_for_run
from .shuffle_stage import run_shuffle_stage_v4
from .silhouette_stage import run_silhouette_stage_v4

__all__ = [
    "DetectStageConfigV4",
    "EdgeRoiDetector",
    "InputCorpusSummary",
    "InputCorpusShuffleError",
    "InputCorpusShuffleResult",
    "InputCorpusShuffleValidationError",
    "PackDualStreamStageConfigV4",
    "ShuffleStageConfigV4",
    "SilhouetteStageConfigV4",
    "StageSummaryV4",
    "STAGE_ORDER_V4",
    "default_input_shuffle_destination",
    "discover_input_corpuses",
    "parse_shuffle_seed",
    "run_detect_stage_v4",
    "run_pack_dual_stream_stage_v4",
    "run_shuffle_stage_v4",
    "run_silhouette_stage_v4",
    "run_v4_stage_for_run",
    "run_v4_stage_sequence_for_run",
    "shuffle_input_corpus",
    "UltralyticsYoloDetector",
]
