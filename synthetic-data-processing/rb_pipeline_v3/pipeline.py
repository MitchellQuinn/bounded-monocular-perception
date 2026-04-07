"""Orchestration helpers for v3 preprocessing stages."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import NpyPackStageConfigV3, StageSummaryV3, ThresholdStageConfigV3
from .npy_pack_stage import run_npy_pack_stage_v3
from .threshold_stage import run_threshold_stage_v3

STAGE_ORDER_V3 = ["threshold", "npy"]



def run_v3_stage_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    threshold_config: ThresholdStageConfigV3 | None = None,
    npy_pack_config: NpyPackStageConfigV3 | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV3:
    """Run one v3 stage for one run."""

    stage = str(stage_name).strip().lower()

    if stage == "threshold":
        if threshold_config is None:
            raise ValueError("threshold_config is required for stage 'threshold'.")
        return run_threshold_stage_v3(project_root, run_name, threshold_config, log_sink=log_sink)

    if stage == "npy":
        if npy_pack_config is None:
            raise ValueError("npy_pack_config is required for stage 'npy'.")
        return run_npy_pack_stage_v3(project_root, run_name, npy_pack_config, log_sink=log_sink)

    raise ValueError(f"Unsupported v3 stage name: {stage_name}")



def run_v3_stage_sequence_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    threshold_config: ThresholdStageConfigV3 | None = None,
    npy_pack_config: NpyPackStageConfigV3 | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> list[StageSummaryV3]:
    """Run one stage or all stages for one v3 run."""

    selected = str(stage_name).strip().lower()
    stages = STAGE_ORDER_V3 if selected == "all" else [selected]

    summaries: list[StageSummaryV3] = []
    for stage in stages:
        summaries.append(
            run_v3_stage_for_run(
                project_root,
                run_name,
                stage,
                threshold_config=threshold_config,
                npy_pack_config=npy_pack_config,
                log_sink=log_sink,
            )
        )

    return summaries
