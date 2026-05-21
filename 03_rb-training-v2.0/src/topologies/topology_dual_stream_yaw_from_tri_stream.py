"""Dual-stream distance + yaw topology backed by tri-stream shard inputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from torch import nn

from . import topology_dual_stream_yaw
from .contracts import task_contract_from_topology_contract

TOPOLOGY_ID = "distance_regressor_dual_stream_yaw_from_tri_stream"
MODEL_CLASS_NAME = topology_dual_stream_yaw.MODEL_CLASS_NAME
DEFAULT_VARIANT = "dual_stream_yaw_from_tri_stream_v0_1"
TRI_STREAM_DUAL_INPUT_MODE = "tri_stream_distance_geometry_as_dual"
BASE_ARCHITECTURE_VARIANT = topology_dual_stream_yaw.DEFAULT_VARIANT
TOPOLOGY_METADATA = {
    "status": "experimental",
    "display_name": "Distance Regressor Dual Stream + Yaw from Tri-Stream Shards",
    "note": (
        "Reuses the dual-stream yaw architecture while binding tri-stream shard "
        "x_distance_image and x_geometry data into the dual image + bbox feature contract."
    ),
    "replacement": "",
}
_SUPPORTED_VARIANTS = {DEFAULT_VARIANT}


def _topology_contract() -> dict[str, Any]:
    contract = deepcopy(topology_dual_stream_yaw.TOPOLOGY_CONTRACT)
    contract["runtime"]["input_mode"] = TRI_STREAM_DUAL_INPUT_MODE
    contract["runtime"]["input_source"] = "tri_stream_distance_image_geometry"
    return contract


def supported_variants() -> tuple[str, ...]:
    """Return all allowed variants."""
    return tuple(sorted(_SUPPORTED_VARIANTS))


def resolve_topology_contract(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the declared output/reporting contract for this topology."""
    variant = str(topology_variant).strip()
    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported topology_variant={topology_variant}; "
            f"expected one of {supported_variants()}"
        )
    _ = topology_params
    return _topology_contract()


def resolve_task_contract(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the training/evaluation contract for this topology family."""
    return task_contract_from_topology_contract(
        resolve_topology_contract(topology_variant, topology_params)
    )


def build_model(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build the reused dual-stream yaw architecture for tri-stream-backed training."""
    variant = str(topology_variant).strip()
    if variant not in _SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported topology_variant={topology_variant}; "
            f"expected one of {supported_variants()}"
        )
    model = topology_dual_stream_yaw.build_model(
        BASE_ARCHITECTURE_VARIANT,
        topology_params,
    )
    model.architecture_variant = variant
    return model


def architecture_text(model: nn.Module) -> str:
    """Render architecture text persisted in run artifacts."""
    variant = getattr(model, "architecture_variant", "unknown")
    return (
        f"architecture_variant={variant}\n"
        f"base_architecture_variant={BASE_ARCHITECTURE_VARIANT}\n"
        f"{model}"
    )
