"""Orchestration helpers for v2 preprocessing stages."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import NpyPackStageConfigV2, SilhouetteStageConfigV2, StageSummaryV2
from .npy_pack_stage import run_npy_pack_stage_v2
from .silhouette_stage import run_silhouette_stage_v2

STAGE_ORDER_V2 = ["silhouette", "npy"]



def run_v2_stage_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    silhouette_config: SilhouetteStageConfigV2 | None = None,
    npy_pack_config: NpyPackStageConfigV2 | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV2:
    """Run one v2 stage for one run."""

    stage = str(stage_name).strip().lower()

    if stage == "silhouette":
        if silhouette_config is None:
            raise ValueError("silhouette_config is required for stage 'silhouette'.")
        return run_silhouette_stage_v2(project_root, run_name, silhouette_config, log_sink=log_sink)

    if stage == "npy":
        if npy_pack_config is None:
            raise ValueError("npy_pack_config is required for stage 'npy'.")
        return run_npy_pack_stage_v2(project_root, run_name, npy_pack_config, log_sink=log_sink)

    raise ValueError(f"Unsupported v2 stage name: {stage_name}")



def run_v2_stage_sequence_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    silhouette_config: SilhouetteStageConfigV2 | None = None,
    npy_pack_config: NpyPackStageConfigV2 | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> list[StageSummaryV2]:
    """Run one stage or all stages for one v2 run."""

    selected = str(stage_name).strip().lower()
    stages = STAGE_ORDER_V2 if selected == "all" else [selected]

    summaries: list[StageSummaryV2] = []
    for stage in stages:
        summaries.append(
            run_v2_stage_for_run(
                project_root,
                run_name,
                stage,
                silhouette_config=silhouette_config,
                npy_pack_config=npy_pack_config,
                log_sink=log_sink,
            )
        )

    return summaries
