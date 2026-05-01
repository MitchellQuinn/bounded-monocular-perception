"""Orchestration helpers for the v4 dual-stream preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .config import (
    DetectStageConfigV4,
    PackDualStreamStageConfigV4,
    PackTriStreamStageConfigV4,
    SilhouetteStageConfigV4,
    StageSummaryV4,
)
from .constants import TRI_STREAM_STAGE_ORDER_V1
from .detect_stage import run_detect_stage_v4
from .pack_dual_stream_stage import run_pack_dual_stream_stage_v4
from .pack_tri_stream_stage import run_pack_tri_stream_stage_v4
from .silhouette_stage import run_silhouette_stage_v4

STAGE_ORDER_V4 = ["detect", "silhouette", "pack_dual_stream"]
TRI_STREAM_STAGE_ORDER = list(TRI_STREAM_STAGE_ORDER_V1)



def run_v4_stage_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    detect_config: DetectStageConfigV4 | None = None,
    silhouette_config: SilhouetteStageConfigV4 | None = None,
    pack_dual_stream_config: PackDualStreamStageConfigV4 | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV4:
    """Run one v4 stage for one run."""

    stage = str(stage_name).strip().lower()

    if stage == "detect":
        if detect_config is None:
            raise ValueError("detect_config is required for stage 'detect'.")
        return run_detect_stage_v4(project_root, run_name, detect_config, log_sink=log_sink)

    if stage == "silhouette":
        if silhouette_config is None:
            raise ValueError("silhouette_config is required for stage 'silhouette'.")
        return run_silhouette_stage_v4(project_root, run_name, silhouette_config, log_sink=log_sink)

    if stage == "pack_dual_stream":
        if pack_dual_stream_config is None:
            raise ValueError("pack_dual_stream_config is required for stage 'pack_dual_stream'.")
        return run_pack_dual_stream_stage_v4(project_root, run_name, pack_dual_stream_config, log_sink=log_sink)

    raise ValueError(f"Unsupported v4 stage name: {stage_name}")



def run_v4_stage_sequence_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    detect_config: DetectStageConfigV4 | None = None,
    silhouette_config: SilhouetteStageConfigV4 | None = None,
    pack_dual_stream_config: PackDualStreamStageConfigV4 | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> list[StageSummaryV4]:
    """Run one stage or all stages in order for one run."""

    selected = str(stage_name).strip().lower()
    stages = STAGE_ORDER_V4 if selected == "all" else [selected]

    summaries: list[StageSummaryV4] = []
    for stage in stages:
        summaries.append(
            run_v4_stage_for_run(
                project_root,
                run_name,
                stage,
                detect_config=detect_config,
                silhouette_config=silhouette_config,
                pack_dual_stream_config=pack_dual_stream_config,
                log_sink=log_sink,
            )
        )

    return summaries


def run_tri_stream_stage_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    detect_config: DetectStageConfigV4 | None = None,
    silhouette_config: SilhouetteStageConfigV4 | None = None,
    pack_tri_stream_config: PackTriStreamStageConfigV4 | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV4:
    """Run one explicit tri-stream stage for one run."""

    stage = str(stage_name).strip().lower()

    if stage == "detect":
        if detect_config is None:
            raise ValueError("detect_config is required for stage 'detect'.")
        return run_detect_stage_v4(project_root, run_name, detect_config, log_sink=log_sink)

    if stage == "silhouette":
        if silhouette_config is None:
            raise ValueError("silhouette_config is required for stage 'silhouette'.")
        return run_silhouette_stage_v4(project_root, run_name, silhouette_config, log_sink=log_sink)

    if stage == "pack_tri_stream":
        if pack_tri_stream_config is None:
            raise ValueError("pack_tri_stream_config is required for stage 'pack_tri_stream'.")
        return run_pack_tri_stream_stage_v4(project_root, run_name, pack_tri_stream_config, log_sink=log_sink)

    raise ValueError(f"Unsupported tri-stream stage name: {stage_name}")


def run_tri_stream_stage_sequence_for_run(
    project_root: Path,
    run_name: str,
    stage_name: str,
    *,
    detect_config: DetectStageConfigV4 | None = None,
    silhouette_config: SilhouetteStageConfigV4 | None = None,
    pack_tri_stream_config: PackTriStreamStageConfigV4 | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> list[StageSummaryV4]:
    """Run one explicit tri-stream stage or all tri-stream stages in order for one run."""

    selected = str(stage_name).strip().lower()
    stages = TRI_STREAM_STAGE_ORDER if selected == "all" else [selected]

    summaries: list[StageSummaryV4] = []
    for stage in stages:
        summaries.append(
            run_tri_stream_stage_for_run(
                project_root,
                run_name,
                stage,
                detect_config=detect_config,
                silhouette_config=silhouette_config,
                pack_tri_stream_config=pack_tri_stream_config,
                log_sink=log_sink,
            )
        )

    return summaries
