"""Dataset-run orchestration for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

from .bootstrap_center_target_stage import run_bootstrap_center_target_stage
from .config import BootstrapCenterTargetConfig, PackRoiFcnConfig
from .contracts import DatasetRunSummaryV01, SPLIT_ORDER
from .pack_roi_fcn_stage import run_pack_roi_fcn_stage
from .paths import dataset_output_root, find_preprocessing_root, resolve_split_paths
from .validation import ensure_valid_input_dataset_reference


def run_preprocessing_for_dataset(
    preprocessing_root: Path | None,
    dataset_reference: str,
    *,
    bootstrap_config: BootstrapCenterTargetConfig | None = None,
    pack_config: PackRoiFcnConfig | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> DatasetRunSummaryV01:
    """Run train then validate preprocessing for one dataset reference."""

    root = find_preprocessing_root(preprocessing_root)
    dataset_name = str(dataset_reference).strip()
    if not dataset_name:
        raise ValueError("dataset_reference cannot be blank")

    ensure_valid_input_dataset_reference(root, dataset_name)

    bootstrap = bootstrap_config or BootstrapCenterTargetConfig()
    pack = pack_config or PackRoiFcnConfig()

    output_root = dataset_output_root(root, dataset_name)
    overwrite_enabled = bool(bootstrap.overwrite or pack.overwrite)
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite_enabled:
            raise FileExistsError(
                "Output dataset already exists. Enable overwrite=True to replace it: "
                f"{output_root}"
            )
        if not (bootstrap.dry_run or pack.dry_run):
            shutil.rmtree(output_root)

    stage_summaries = []
    for split_name in SPLIT_ORDER:
        split_paths = resolve_split_paths(root, dataset_name, split_name)
        bootstrap_summary = run_bootstrap_center_target_stage(
            split_paths,
            bootstrap,
            log_sink=log_sink,
        )
        stage_summaries.append(bootstrap_summary)

        pack_summary = run_pack_roi_fcn_stage(
            split_paths,
            pack,
            log_sink=log_sink,
        )
        stage_summaries.append(pack_summary)

    return DatasetRunSummaryV01(
        dataset_reference=dataset_name,
        output_root=str(output_root),
        stage_summaries=stage_summaries,
    )
