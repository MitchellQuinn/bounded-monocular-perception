"""Manifest utilities for reading, validating, and writing samples CSV files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import RUN_JSON_FILENAME, SAMPLES_FILENAME

UNITY_REQUIRED_COLUMNS = [
    "run_id",
    "sample_id",
    "frame_index",
    "image_filename",
    "distance_m",
    "image_width_px",
    "image_height_px",
    "capture_success",
]

EDGE_STAGE_COLUMNS = [
    "edge_image_filename",
    "edge_stage_status",
    "edge_stage_error",
]

BBOX_STAGE_COLUMNS = [
    "bbox_image_filename",
    "bbox_stage_status",
    "bbox_stage_error",
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

PREPROCESSING_CONTRACT_KEY = "PreprocessingContract"
PREPROCESSING_CONTRACT_VERSION = "rb-preprocess-v1"
PREPROCESSING_STAGE_ORDER = ("edge", "bbox", "npy", "pack")



def load_samples_csv(samples_path: Path) -> pd.DataFrame:
    """Load a samples CSV while preserving row order."""

    return pd.read_csv(samples_path)



def write_samples_csv(samples_df: pd.DataFrame, samples_path: Path, dry_run: bool = False) -> None:
    """Write samples CSV unless dry-run mode is enabled."""

    if dry_run:
        return

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(samples_path, index=False)



def ensure_columns_exist(samples_df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """Return a list of missing columns."""

    return [column for column in required_columns if column not in samples_df.columns]



def append_columns(samples_df: pd.DataFrame, appended_columns: list[str], default_value: object = "") -> pd.DataFrame:
    """Append new columns at the end while preserving existing columns and row order."""

    for column in appended_columns:
        if column not in samples_df.columns:
            samples_df[column] = default_value
    return samples_df



def copy_run_json(source_manifest_dir: Path, target_manifest_dir: Path, dry_run: bool = False) -> Path:
    """Copy run.json from source stage manifests to target stage manifests."""

    source = source_manifest_dir / RUN_JSON_FILENAME
    target = target_manifest_dir / RUN_JSON_FILENAME

    if dry_run:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target



def samples_csv_path(manifest_dir: Path) -> Path:
    return manifest_dir / SAMPLES_FILENAME



def run_json_path(manifest_dir: Path) -> Path:
    return manifest_dir / RUN_JSON_FILENAME


def load_run_json(manifest_dir: Path) -> dict[str, Any]:
    """Load one manifest run.json as a dictionary."""

    path = run_json_path(manifest_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run.json must contain a JSON object: {path}")
    return payload


def write_run_json(
    manifest_dir: Path,
    payload: dict[str, Any],
    *,
    dry_run: bool = False,
) -> Path:
    """Write run.json unless dry-run mode is enabled."""

    path = run_json_path(manifest_dir)
    if dry_run:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return path


def upsert_preprocessing_contract(
    manifest_dir: Path,
    *,
    stage_name: str,
    stage_parameters: dict[str, Any],
    current_representation: dict[str, Any],
    dry_run: bool = False,
) -> Path:
    """Add or update the authoritative preprocessing contract inside run.json."""

    normalized_stage_name = str(stage_name).strip().lower()
    if normalized_stage_name not in PREPROCESSING_STAGE_ORDER:
        allowed = ", ".join(PREPROCESSING_STAGE_ORDER)
        raise ValueError(
            f"Unsupported preprocessing contract stage '{stage_name}'. Allowed: {allowed}."
        )

    payload = load_run_json(manifest_dir)
    existing = payload.get(PREPROCESSING_CONTRACT_KEY)
    contract = existing.copy() if isinstance(existing, dict) else {}

    completed_stages_raw = contract.get("CompletedStages")
    completed_stages = (
        [
            str(value).strip().lower()
            for value in completed_stages_raw
            if str(value).strip().lower() in PREPROCESSING_STAGE_ORDER
        ]
        if isinstance(completed_stages_raw, list)
        else []
    )
    if normalized_stage_name not in completed_stages:
        completed_stages.append(normalized_stage_name)
    completed_stages = [
        stage for stage in PREPROCESSING_STAGE_ORDER if stage in completed_stages
    ]

    stages_raw = contract.get("Stages")
    stages = stages_raw.copy() if isinstance(stages_raw, dict) else {}
    stages[normalized_stage_name] = dict(stage_parameters)

    contract["ContractVersion"] = PREPROCESSING_CONTRACT_VERSION
    contract["CurrentStage"] = normalized_stage_name
    contract["CompletedStages"] = completed_stages
    contract["CurrentRepresentation"] = dict(current_representation)
    contract["Stages"] = stages

    payload[PREPROCESSING_CONTRACT_KEY] = contract
    return write_run_json(manifest_dir, payload, dry_run=dry_run)
