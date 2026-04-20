"""Persistence helpers for resumable ROI-FCN training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .contracts import RESUME_STATE_FILENAME

_REQUIRED_KEYS = {
    "format_version",
    "epoch",
    "run_id",
    "training_dataset",
    "validation_dataset",
    "topology_id",
    "topology_variant",
    "topology_params",
    "topology_spec_signature",
    "topology_contract_signature",
    "output_hw",
    "train_split_contract",
    "validation_split_contract",
    "best_epoch",
    "best_validation_loss",
    "best_validation_mean_center_error_px",
    "epochs_without_improvement",
    "history_rows",
    "model_state_dict",
    "optimizer_state_dict",
}


def load_resume_state(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate a resume-state payload."""
    state_path = Path(path).expanduser().resolve()
    if not state_path.exists():
        raise FileNotFoundError(f"Resume state file not found: {state_path}")
    try:
        payload = torch.load(state_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(state_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Resume state payload must be a mapping: {state_path}")
    missing = sorted(_REQUIRED_KEYS.difference(payload.keys()))
    if missing:
        raise ValueError(f"Resume state missing required keys {missing}: {state_path}")
    return dict(payload)


def save_resume_state(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save one normalized resume-state payload."""
    state_path = Path(path).expanduser().resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, state_path)
    return state_path


def build_resume_state_payload(
    *,
    epoch: int,
    run_id: str,
    training_dataset: str,
    validation_dataset: str,
    topology_id: str,
    topology_variant: str,
    topology_params: dict[str, Any],
    topology_spec_signature: str,
    topology_contract_signature: str,
    output_hw: tuple[int, int],
    train_split_contract: dict[str, Any],
    validation_split_contract: dict[str, Any],
    best_epoch: int,
    best_validation_loss: float,
    best_validation_mean_center_error_px: float,
    epochs_without_improvement: int,
    history_rows: list[dict[str, Any]],
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical resume payload for the current epoch."""
    return {
        "format_version": 1,
        "epoch": int(epoch),
        "run_id": str(run_id),
        "training_dataset": str(training_dataset),
        "validation_dataset": str(validation_dataset),
        "topology_id": str(topology_id),
        "topology_variant": str(topology_variant),
        "topology_params": dict(topology_params),
        "topology_spec_signature": str(topology_spec_signature),
        "topology_contract_signature": str(topology_contract_signature),
        "output_hw": [int(output_hw[0]), int(output_hw[1])],
        "train_split_contract": dict(train_split_contract),
        "validation_split_contract": dict(validation_split_contract),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_mean_center_error_px": float(best_validation_mean_center_error_px),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history_rows": list(history_rows),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }


__all__ = [
    "RESUME_STATE_FILENAME",
    "build_resume_state_payload",
    "load_resume_state",
    "save_resume_state",
]
