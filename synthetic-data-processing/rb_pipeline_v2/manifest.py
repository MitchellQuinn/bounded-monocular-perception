"""Manifest and preprocessing contract helpers for v2 pipeline."""

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

SILHOUETTE_STAGE_COLUMNS = [
    "silhouette_image_filename",
    "silhouette_stage_status",
    "silhouette_stage_error",
    "silhouette_mode",
    "silhouette_generator",
    "silhouette_fallback_used",
    "silhouette_used_fallback",
    "silhouette_fallback_reason",
    "silhouette_area_px",
    "silhouette_bbox_x1",
    "silhouette_bbox_y1",
    "silhouette_bbox_x2",
    "silhouette_bbox_y2",
    "silhouette_selected_component_area_px",
    "silhouette_selected_component_bbox_x1",
    "silhouette_selected_component_bbox_y1",
    "silhouette_selected_component_bbox_x2",
    "silhouette_selected_component_bbox_y2",
    "silhouette_contour_area_px",
    "silhouette_contour_bbox_x1",
    "silhouette_contour_bbox_y1",
    "silhouette_contour_bbox_x2",
    "silhouette_contour_bbox_y2",
    "silhouette_hull_area_px",
    "silhouette_num_components_total",
    "silhouette_num_components_after_filter",
    "silhouette_quality_flags",
    "silhouette_edge_debug_filename",
    "silhouette_debug_raw_edge_filename",
    "silhouette_debug_post_morph_filename",
    "silhouette_debug_components_mask_filename",
    "silhouette_debug_selected_component_filename",
    "silhouette_debug_external_contour_filename",
    "silhouette_debug_final_filled_filename",
    "silhouette_debug_fallback_hull_filename",
]

NPY_STAGE_COLUMNS = [
    "npy_filename",
    "npy_stage_status",
    "npy_stage_error",
]

PACK_STAGE_COLUMNS = [
    "npz_filename",
    "npz_row_index",
    "pack_stage_status",
    "pack_stage_error",
]

PREPROCESSING_CONTRACT_VERSION_V2 = "rb-preprocess-v2"
PREPROCESSING_STAGE_ORDER_V2 = ("silhouette", "npy", "pack")


def upsert_preprocessing_contract_v2(
    manifest_dir,
    *,
    stage_name: str,
    stage_parameters: dict[str, Any],
    current_representation: dict[str, Any],
    dry_run: bool = False,
):
    """Add or update the v2 authoritative preprocessing contract in run.json."""

    normalized_stage_name = str(stage_name).strip().lower()
    if normalized_stage_name not in PREPROCESSING_STAGE_ORDER_V2:
        allowed = ", ".join(PREPROCESSING_STAGE_ORDER_V2)
        raise ValueError(f"Unsupported v2 stage '{stage_name}'. Allowed: {allowed}.")

    payload = load_run_json(manifest_dir)
    existing = payload.get(PREPROCESSING_CONTRACT_KEY)
    contract = existing.copy() if isinstance(existing, dict) else {}

    completed_stages_raw = contract.get("CompletedStages")
    completed_stages = (
        [
            str(value).strip().lower()
            for value in completed_stages_raw
            if str(value).strip().lower() in PREPROCESSING_STAGE_ORDER_V2
        ]
        if isinstance(completed_stages_raw, list)
        else []
    )

    if normalized_stage_name not in completed_stages:
        completed_stages.append(normalized_stage_name)

    completed_stages = [
        stage for stage in PREPROCESSING_STAGE_ORDER_V2 if stage in completed_stages
    ]

    stages_raw = contract.get("Stages")
    stages = stages_raw.copy() if isinstance(stages_raw, dict) else {}
    stages[normalized_stage_name] = dict(stage_parameters)

    contract["ContractVersion"] = PREPROCESSING_CONTRACT_VERSION_V2
    contract["CurrentStage"] = normalized_stage_name
    contract["CompletedStages"] = completed_stages
    contract["CurrentRepresentation"] = dict(current_representation)
    contract["Stages"] = stages

    payload[PREPROCESSING_CONTRACT_KEY] = contract
    return write_run_json(manifest_dir, payload, dry_run=dry_run)


__all__ = [
    "PREPROCESSING_CONTRACT_KEY",
    "PREPROCESSING_CONTRACT_VERSION_V2",
    "PREPROCESSING_STAGE_ORDER_V2",
    "UNITY_REQUIRED_COLUMNS",
    "SILHOUETTE_STAGE_COLUMNS",
    "NPY_STAGE_COLUMNS",
    "PACK_STAGE_COLUMNS",
    "append_columns",
    "copy_run_json",
    "load_run_json",
    "load_samples_csv",
    "run_json_path",
    "samples_csv_path",
    "upsert_preprocessing_contract_v2",
    "write_samples_csv",
]
