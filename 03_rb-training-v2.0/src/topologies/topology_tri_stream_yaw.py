"""Tri-stream distance + yaw topology family dispatcher."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from torch import nn

from . import (
    topology_tri_stream_yaw_v0_1,
    topology_tri_stream_yaw_v0_2,
    topology_tri_stream_yaw_v0_3,
    topology_tri_stream_yaw_v0_4,
)
from .contracts import TOPOLOGY_CONTRACT_VERSION, task_contract_from_topology_contract
from .topology_tri_stream_yaw_common import parse_common_topology_params

TOPOLOGY_ID = "distance_regressor_tri_stream_yaw"
MODEL_CLASS_NAME = "DistanceRegressorTriStreamYaw"
DEFAULT_VARIANT = topology_tri_stream_yaw_v0_1.VARIANT
TOPOLOGY_METADATA = {
    "status": "experimental",
    "display_name": "Distance Regressor Tri Stream + Yaw",
    "note": "Distance plus yaw multitask topology with distance image, orientation image, and geometry streams.",
    "replacement": "",
}

_VARIANT_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    topology_tri_stream_yaw_v0_1.VARIANT: topology_tri_stream_yaw_v0_1.build_model,
    topology_tri_stream_yaw_v0_2.VARIANT: topology_tri_stream_yaw_v0_2.build_model,
    topology_tri_stream_yaw_v0_3.VARIANT: topology_tri_stream_yaw_v0_3.build_model,
    topology_tri_stream_yaw_v0_4.VARIANT: topology_tri_stream_yaw_v0_4.build_model,
}
_SUPPORTED_VARIANTS = frozenset(_VARIANT_BUILDERS)

TOPOLOGY_CONTRACT = {
    "contract_version": TOPOLOGY_CONTRACT_VERSION,
    "task_family": "multitask_regression",
    "targets": {
        "distance": {
            "kind": "regression",
            "columns": ["distance_m"],
            "target_npz_key": "y_distance_m",
        },
        "yaw": {
            "kind": "circular_regression",
            "columns": ["yaw_sin", "yaw_cos"],
            "debug_columns": ["yaw_deg"],
            "target_npz_keys": ["y_yaw_sin", "y_yaw_cos"],
            "debug_target_npz_key": "y_yaw_deg",
        },
    },
    "outputs": {
        "distance": {
            "kind": "regression",
            "columns": ["distance_m"],
            "output_key": "distance_m",
        },
        "yaw": {
            "kind": "circular_regression",
            "columns": ["yaw_sin", "yaw_cos"],
            "output_key": "yaw_sin_cos",
        },
    },
    "runtime": {
        "prediction_mode": "distance_yaw_sincos",
        "input_mode": "tri_stream_distance_orientation_geometry",
        "output_kind": "mapping",
        "heads": {
            "distance": {
                "output": "distance",
                "target": "distance",
                "metrics_role": "distance",
                "loss_role": "distance",
            },
            "orientation": {
                "output": "yaw",
                "target": "yaw",
                "metrics_role": "orientation",
                "loss_role": "orientation",
            },
        },
    },
    "reporting": {
        "family": "distance_orientation_multitask",
        "train_losses": ["total_loss", "distance_loss", "orientation_loss"],
        "validation_metrics": [
            "yaw_mean_error_deg",
            "yaw_median_error_deg",
            "yaw_p95_error_deg",
            "yaw_acc@5deg",
            "yaw_acc@10deg",
            "yaw_acc@15deg",
        ],
        "orientation_accuracy_thresholds_deg": [5.0, 10.0, 15.0],
    },
}


def supported_variants() -> tuple[str, ...]:
    """Return all allowed variants."""
    return tuple(sorted(_SUPPORTED_VARIANTS))


def resolve_topology_contract(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the declared output/reporting contract for this topology."""
    _ = topology_variant
    _ = topology_params
    return dict(TOPOLOGY_CONTRACT)


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
    """Build one tri-stream multitask model instance from topology variant + params."""
    builder = _VARIANT_BUILDERS.get(str(topology_variant).strip())
    if builder is None:
        raise ValueError(
            f"Unsupported topology_variant={topology_variant}; "
            f"expected one of {supported_variants()}"
        )
    parsed_params = parse_common_topology_params(topology_params, topology_id=TOPOLOGY_ID)
    return builder(**parsed_params)


def architecture_text(model: nn.Module) -> str:
    """Render architecture text persisted in run artifacts."""
    variant = getattr(model, "architecture_variant", "unknown")
    return f"architecture_variant={variant}\n{model}"
