"""Persistence helpers for resumable training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

RESUME_STATE_FILENAME = "resume_state.pt"


_REQUIRED_KEYS = {
    "format_version",
    "epoch",
    "run_id",
    "model_state_dict",
    "optimizer_state_dict",
    "best_epoch",
    "best_val_loss",
    "no_improvement_epochs",
    "history_records",
}


def load_resume_state(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load and validate a resume-state checkpoint payload."""
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
        raise ValueError(
            "Resume state missing required keys "
            f"{missing}: {state_path}"
        )
    return dict(payload)


def save_resume_state(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save a resume-state payload to disk."""
    state_path = Path(path).expanduser().resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, state_path)
    return state_path


def build_resume_state_payload(
    *,
    epoch: int,
    run_id: str,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    lr_scheduler_state_dict: dict[str, Any] | None,
    best_epoch: int,
    best_val_loss: float,
    no_improvement_epochs: int,
    history_records: list[dict[str, Any]],
    topology_id: str,
    topology_variant: str,
    topology_params: dict[str, Any],
    topology_signature: str,
    model_architecture_variant: str | None,
    training_data_root_resolved: str,
    validation_data_root_resolved: str,
    target_hw: tuple[int, int],
) -> dict[str, Any]:
    """Build a normalized resume-state payload for the current epoch."""
    return {
        "format_version": 1,
        "epoch": int(epoch),
        "run_id": str(run_id),
        "topology_id": str(topology_id),
        "topology_variant": str(topology_variant),
        "topology_params": dict(topology_params),
        "topology_signature": str(topology_signature),
        "model_architecture_variant": (
            str(model_architecture_variant)
            if model_architecture_variant is not None
            else str(topology_variant)
        ),
        "training_data_root_resolved": str(training_data_root_resolved),
        "validation_data_root_resolved": str(validation_data_root_resolved),
        "target_hw": [int(target_hw[0]), int(target_hw[1])],
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "no_improvement_epochs": int(no_improvement_epochs),
        "history_records": list(history_records),
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "lr_scheduler_state_dict": lr_scheduler_state_dict,
    }
