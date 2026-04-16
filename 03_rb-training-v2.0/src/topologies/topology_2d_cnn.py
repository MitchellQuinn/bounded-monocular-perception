"""Topology adapter for the original 2D CNN distance regressor."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from ..model_2d_cnn import (
    MODEL_ARCHITECTURE_VARIANTS,
    DistanceRegressor2DCNN,
    architecture_text as _architecture_text_2d_cnn,
)

TOPOLOGY_ID = "distance_regressor_2d_cnn"
MODEL_CLASS_NAME = "DistanceRegressor2DCNN"
DEFAULT_VARIANT = "fast_v0_2"
TOPOLOGY_METADATA = {
    "status": "deprecated",
    "display_name": "Distance Regressor 2D CNN",
    "note": "Legacy full-frame baseline",
    "replacement": "",
}


def resolve_task_contract(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the training/evaluation contract for this topology family."""
    _ = topology_variant
    _ = topology_params
    return {
        "task_family": "regression",
        "prediction_mode": "scalar_distance",
        "input_mode": "image_tensor",
        "output_kind": "tensor",
        "target_columns": ["distance_m"],
        "debug_target_columns": [],
        "heads": {
            "distance": {
                "target_columns": ["distance_m"],
                "metrics_role": "distance",
                "loss_role": "distance",
            }
        },
    }


def supported_variants() -> tuple[str, ...]:
    """Return supported variants for this topology."""
    return tuple(sorted(MODEL_ARCHITECTURE_VARIANTS))



def build_model(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build the original 2D CNN model using standardized topology inputs."""
    params = dict(topology_params or {})
    input_channels = int(params.pop("input_channels", 1))
    dropout_p = float(params.pop("dropout_p", 0.2))
    if params:
        raise ValueError(
            "Unsupported topology_params for distance_regressor_2d_cnn: "
            f"{sorted(params.keys())}. Supported keys: ['dropout_p', 'input_channels']"
        )

    return DistanceRegressor2DCNN(
        input_channels=input_channels,
        dropout_p=dropout_p,
        architecture_variant=topology_variant,
    )



def architecture_text(model: nn.Module) -> str:
    """Render architecture text for logging/artifacts."""
    return _architecture_text_2d_cnn(model)
