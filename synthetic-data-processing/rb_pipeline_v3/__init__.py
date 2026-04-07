"""Threshold-first v3 preprocessing pipeline for binary representations."""

from __future__ import annotations

from .config import (
    NpyPackStageConfigV3,
    PipelineRunConfigV3,
    ShuffleStageConfigV3,
    StageSummaryV3,
    ThresholdStageConfigV3,
)
from .npy_pack_stage import run_npy_pack_stage_v3
from .paths import (
    THRESHOLD_ROOT_NAME,
    TRAINING_V3_ROOT_NAME,
    find_project_root,
    list_training_v3_runs,
)
from .pipeline import STAGE_ORDER_V3, run_v3_stage_for_run, run_v3_stage_sequence_for_run
from .shuffle_stage import run_shuffle_stage_v3
from .threshold_stage import run_threshold_stage_v3
from .widgets import NpyPackPanelV3, PipelineLauncherV3, PreviewPanelV3, ShuffleCorpusPanelV3, ThresholdStagePanelV3

__all__ = [
    "ThresholdStageConfigV3",
    "NpyPackStageConfigV3",
    "ShuffleStageConfigV3",
    "PipelineRunConfigV3",
    "StageSummaryV3",
    "THRESHOLD_ROOT_NAME",
    "TRAINING_V3_ROOT_NAME",
    "STAGE_ORDER_V3",
    "find_project_root",
    "list_training_v3_runs",
    "run_v3_stage_for_run",
    "run_v3_stage_sequence_for_run",
    "run_threshold_stage_v3",
    "run_npy_pack_stage_v3",
    "run_shuffle_stage_v3",
    "PreviewPanelV3",
    "PipelineLauncherV3",
    "ThresholdStagePanelV3",
    "NpyPackPanelV3",
    "ShuffleCorpusPanelV3",
]
