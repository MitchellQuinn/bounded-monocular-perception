"""Extensible pass-1 v2 preprocessing pipeline for silhouette representations."""

from __future__ import annotations

from .algorithms import register_default_components
from .config import (
    NpyPackStageConfigV2,
    PipelineRunConfigV2,
    ShuffleStageConfigV2,
    SilhouetteStageConfigV2,
    StageSummaryV2,
)
from .inspect import (
    load_source_edge_silhouette_preview,
    save_contour_comparison_debug_batch,
    show_source_edge_silhouette_preview,
)
from .npy_pack_stage import run_npy_pack_stage_v2
from .paths import (
    SILHOUETTE_ROOT_NAME,
    TRAINING_V2_ROOT_NAME,
    find_project_root,
    list_training_v2_runs,
)
from .pipeline import STAGE_ORDER_V2, run_v2_stage_for_run, run_v2_stage_sequence_for_run
from .registry import list_registered_component_ids
from .shuffle_stage import run_shuffle_stage_v2
from .silhouette_stage import run_silhouette_stage_v2
from .widgets import NpyPackPanelV2, PipelineLauncherV2, PreviewPanelV2, ShuffleCorpusPanelV2, SilhouetteStagePanelV2

register_default_components()

__all__ = [
    "SilhouetteStageConfigV2",
    "NpyPackStageConfigV2",
    "ShuffleStageConfigV2",
    "PipelineRunConfigV2",
    "StageSummaryV2",
    "SILHOUETTE_ROOT_NAME",
    "TRAINING_V2_ROOT_NAME",
    "STAGE_ORDER_V2",
    "find_project_root",
    "list_training_v2_runs",
    "run_v2_stage_for_run",
    "run_v2_stage_sequence_for_run",
    "run_silhouette_stage_v2",
    "run_npy_pack_stage_v2",
    "run_shuffle_stage_v2",
    "load_source_edge_silhouette_preview",
    "show_source_edge_silhouette_preview",
    "save_contour_comparison_debug_batch",
    "list_registered_component_ids",
    "PreviewPanelV2",
    "PipelineLauncherV2",
    "SilhouetteStagePanelV2",
    "NpyPackPanelV2",
    "ShuffleCorpusPanelV2",
]
