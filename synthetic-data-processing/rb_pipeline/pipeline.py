"""Convenience orchestration helpers for running stages from notebooks/widgets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .bbox_stage import run_bbox_stage
from .config import BBoxStageConfig, EdgeStageConfig, NpyStageConfig, PackStageConfig, StageSummary
from .edge_stage import run_edge_stage
from .npy_pack_stage import run_npy_pack_stage
from .pack_stage import run_pack_stage

STAGE_ORDER = ["edge", "bbox", "npy"]



def run_stage_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    edge_config: EdgeStageConfig | None = None,
    bbox_config: BBoxStageConfig | None = None,
    npy_config: NpyStageConfig | None = None,
    pack_config: PackStageConfig | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummary:
    """Run one stage for one run."""

    stage = stage_name.strip().lower()

    if stage == "edge":
        return run_edge_stage(project_root, run_name, edge_config, log_sink=log_sink)
    if stage == "bbox":
        return run_bbox_stage(project_root, run_name, bbox_config, log_sink=log_sink)
    if stage == "npy":
        return run_npy_pack_stage(project_root, run_name, npy_config, pack_config, log_sink=log_sink)
    if stage == "pack":
        return run_pack_stage(project_root, run_name, pack_config, log_sink=log_sink)

    raise ValueError(f"Unsupported stage name: {stage_name}")



def run_stage_sequence_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    edge_config: EdgeStageConfig | None = None,
    bbox_config: BBoxStageConfig | None = None,
    npy_config: NpyStageConfig | None = None,
    pack_config: PackStageConfig | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> list[StageSummary]:
    """Run one stage or all stages for one run."""

    selected = stage_name.strip().lower()
    stages = STAGE_ORDER if selected == "all" else [selected]

    summaries: list[StageSummary] = []
    for stage in stages:
        summaries.append(
            run_stage_for_run(
                project_root,
                run_name,
                stage,
                edge_config=edge_config,
                bbox_config=bbox_config,
                npy_config=npy_config,
                pack_config=pack_config,
                log_sink=log_sink,
            )
        )
    return summaries
