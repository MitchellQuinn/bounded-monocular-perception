"""Torch device policy resolution for live inference runtime adapters."""

from __future__ import annotations

import importlib
from typing import Any


TORCH_DEVICE_POLICIES = ("auto", "cuda", "cpu")


def normalize_torch_device_policy(requested: str) -> str:
    """Normalize and validate a supported Torch device policy string."""
    policy = str(requested).strip().lower()
    if policy not in TORCH_DEVICE_POLICIES:
        expected = ", ".join(TORCH_DEVICE_POLICIES)
        raise ValueError(
            f"Unsupported Torch device policy {requested!r}; expected one of: {expected}."
        )
    return policy


def resolve_torch_device(requested: str) -> str:
    """Resolve ``auto``, ``cuda``, or ``cpu`` to a concrete Torch device name."""
    policy = normalize_torch_device_policy(requested)
    if policy == "cpu":
        return "cpu"

    torch = _import_torch()
    cuda_available = bool(torch.cuda.is_available())
    if policy == "cuda":
        if cuda_available:
            return "cuda"
        raise RuntimeError(
            "CUDA was requested for live inference, but torch.cuda.is_available() is false."
        )
    return "cuda" if cuda_available else "cpu"


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Torch is required to resolve CUDA availability for live inference."
        ) from exc
