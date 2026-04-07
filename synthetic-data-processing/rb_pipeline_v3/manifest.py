"""Manifest and preprocessing contract helpers for v3 pipeline."""

from __future__ import annotations

from typing import Any

from rb_pipeline.manifest import (
    PREPROCESSING_CONTRACT_KEY,
    UNITY_REQUIRED_COLUMNS,
    append_columns,
    copy_run_json,
    load_run_json,
    load_samples_csv,
    run_json_path,
    samples_csv_path,
    write_run_json,
    write_samples_csv,
)

THRESHOLD_STAGE_COLUMNS = [
    "threshold_image_filename",
    "threshold_stage_status",
    "threshold_stage_error",
    "threshold_mode",
    "threshold_low_value",
    "threshold_high_value",
    "threshold_invert_selection",
    "threshold_area_px",
    "threshold_bbox_x1",
    "threshold_bbox_y1",
    "threshold_bbox_x2",
    "threshold_bbox_y2",
    "threshold_num_components_total",
    "threshold_num_components_after_filter",
    "threshold_quality_flags",
    "threshold_debug_binary_filename",
    "threshold_debug_selected_component_filename",
    "threshold_debug_amalgamated_filename",
]

NPY_STAGE_COLUMNS = [
    "npy_filename",
    "npy_source_image_column",
    "npy_source_image_filename",
    "npy_stage_status",
    "npy_stage_error",
]

PACK_STAGE_COLUMNS = [
    "npz_filename",
    "npz_row_index",
    "pack_stage_status",
    "pack_stage_error",
]

PREPROCESSING_CONTRACT_VERSION_V3 = "rb-preprocess-v3"
PREPROCESSING_STAGE_ORDER_V3 = ("threshold", "npy", "pack")


def upsert_preprocessing_contract_v3(
    manifest_dir,
    *,
    stage_name: str,
    stage_parameters: dict[str, Any],
    current_representation: dict[str, Any],
    dry_run: bool = False,
):
    """Add or update the v3 authoritative preprocessing contract in run.json."""

    normalized_stage_name = str(stage_name).strip().lower()
    if normalized_stage_name not in PREPROCESSING_STAGE_ORDER_V3:
        allowed = ", ".join(PREPROCESSING_STAGE_ORDER_V3)
        raise ValueError(f"Unsupported v3 stage '{stage_name}'. Allowed: {allowed}.")

    payload = load_run_json(manifest_dir)
    existing = payload.get(PREPROCESSING_CONTRACT_KEY)
    contract = existing.copy() if isinstance(existing, dict) else {}

    completed_stages_raw = contract.get("CompletedStages")
    completed_stages = (
        [
            str(value).strip().lower()
            for value in completed_stages_raw
            if str(value).strip().lower() in PREPROCESSING_STAGE_ORDER_V3
        ]
        if isinstance(completed_stages_raw, list)
        else []
    )

    if normalized_stage_name not in completed_stages:
        completed_stages.append(normalized_stage_name)

    completed_stages = [
        stage for stage in PREPROCESSING_STAGE_ORDER_V3 if stage in completed_stages
    ]

    stages_raw = contract.get("Stages")
    stages = stages_raw.copy() if isinstance(stages_raw, dict) else {}
    stages[normalized_stage_name] = dict(stage_parameters)

    contract["ContractVersion"] = PREPROCESSING_CONTRACT_VERSION_V3
    contract["CurrentStage"] = normalized_stage_name
    contract["CompletedStages"] = completed_stages
    contract["CurrentRepresentation"] = dict(current_representation)
    contract["Stages"] = stages

    payload[PREPROCESSING_CONTRACT_KEY] = contract
    return write_run_json(manifest_dir, payload, dry_run=dry_run)


__all__ = [
    "PREPROCESSING_CONTRACT_KEY",
    "PREPROCESSING_CONTRACT_VERSION_V3",
    "PREPROCESSING_STAGE_ORDER_V3",
    "UNITY_REQUIRED_COLUMNS",
    "THRESHOLD_STAGE_COLUMNS",
    "NPY_STAGE_COLUMNS",
    "PACK_STAGE_COLUMNS",
    "append_columns",
    "copy_run_json",
    "load_run_json",
    "load_samples_csv",
    "run_json_path",
    "samples_csv_path",
    "upsert_preprocessing_contract_v3",
    "write_samples_csv",
]
