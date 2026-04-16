"""Manifest and preprocessing-contract helpers for v4 pipeline."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import (
    PREPROCESSING_CONTRACT_KEY,
    PREPROCESSING_CONTRACT_VERSION_V4,
    PREPROCESSING_STAGE_ORDER_V4,
    RUN_JSON_FILENAME,
    SAMPLES_FILENAME,
)


def samples_csv_path(manifest_dir: Path) -> Path:
    return manifest_dir / SAMPLES_FILENAME


def run_json_path(manifest_dir: Path) -> Path:
    return manifest_dir / RUN_JSON_FILENAME


def load_samples_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def write_samples_csv(samples_df: pd.DataFrame, path: Path, dry_run: bool = False) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    samples_df.to_csv(path, index=False)


def append_columns(samples_df: pd.DataFrame, columns: list[str], default_value: object = "") -> pd.DataFrame:
    for column in columns:
        if column not in samples_df.columns:
            samples_df[column] = default_value
    return samples_df


def copy_run_json(source_manifest_dir: Path, target_manifest_dir: Path, dry_run: bool = False) -> Path:
    source = source_manifest_dir / RUN_JSON_FILENAME
    target = target_manifest_dir / RUN_JSON_FILENAME

    if dry_run:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def load_run_json(manifest_dir: Path) -> dict[str, Any]:
    payload = json.loads(run_json_path(manifest_dir).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"run.json must contain a JSON object: {run_json_path(manifest_dir)}")
    return payload


def write_run_json(manifest_dir: Path, payload: dict[str, Any], *, dry_run: bool = False) -> Path:
    path = run_json_path(manifest_dir)
    if dry_run:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    return path


def upsert_preprocessing_contract_v4(
    manifest_dir: Path,
    *,
    stage_name: str,
    stage_parameters: dict[str, Any],
    current_representation: dict[str, Any],
    dry_run: bool = False,
) -> Path:
    normalized_stage = str(stage_name).strip().lower()
    if normalized_stage not in PREPROCESSING_STAGE_ORDER_V4:
        allowed = ", ".join(PREPROCESSING_STAGE_ORDER_V4)
        raise ValueError(f"Unsupported v4 stage '{stage_name}'. Allowed: {allowed}.")

    payload = load_run_json(manifest_dir)
    existing = payload.get(PREPROCESSING_CONTRACT_KEY)
    contract = existing.copy() if isinstance(existing, dict) else {}

    completed_raw = contract.get("CompletedStages")
    completed = (
        [
            str(value).strip().lower()
            for value in completed_raw
            if str(value).strip().lower() in PREPROCESSING_STAGE_ORDER_V4
        ]
        if isinstance(completed_raw, list)
        else []
    )
    if normalized_stage not in completed:
        completed.append(normalized_stage)
    completed = [stage for stage in PREPROCESSING_STAGE_ORDER_V4 if stage in completed]

    stages_raw = contract.get("Stages")
    stages = stages_raw.copy() if isinstance(stages_raw, dict) else {}
    stages[normalized_stage] = dict(stage_parameters)

    contract["ContractVersion"] = PREPROCESSING_CONTRACT_VERSION_V4
    contract["CurrentStage"] = normalized_stage
    contract["CompletedStages"] = completed
    contract["CurrentRepresentation"] = dict(current_representation)
    contract["Stages"] = stages

    payload[PREPROCESSING_CONTRACT_KEY] = contract
    return write_run_json(manifest_dir, payload, dry_run=dry_run)
