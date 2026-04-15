"""Alternative global-pooling CNN topology for distance regression."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

TOPOLOGY_ID = "distance_regressor_global_pool_cnn"
MODEL_CLASS_NAME = "DistanceRegressorGlobalPoolCNN"
DEFAULT_VARIANT = "tiny_v0_1"
TOPOLOGY_METADATA = {
    "status": "deprecated",
    "display_name": "Distance Regressor Global Pool CNN",
    "note": "Compact global-pooling alternative baseline",
    "replacement": "",
}
_SUPPORTED_VARIANTS = {"tiny_v0_1", "wide_v0_1"}


class DistanceRegressorGlobalPoolCNN(nn.Module):
    """Compact CNN family with global pooling and an MLP regression head."""

    def __init__(
        self,
        input_channels: int = 1,
        dropout_p: float = 0.2,
        architecture_variant: str = DEFAULT_VARIANT,
    ) -> None:
        super().__init__()
        if architecture_variant not in _SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported architecture_variant={architecture_variant}; "
                f"expected one of {sorted(_SUPPORTED_VARIANTS)}"
            )
        self.architecture_variant = architecture_variant

        if architecture_variant == "tiny_v0_1":
            channels = (16, 24, 32, 48, 64)
            head_hidden = 64
        else:
            channels = (24, 36, 48, 72, 96)
            head_hidden = 96

        layers: list[nn.Module] = []
        in_channels = int(input_channels)
        for idx, out_channels in enumerate(channels):
            stride = 2 if idx < 4 else 1
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout_p)),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)



def supported_variants() -> tuple[str, ...]:
    """Return supported variants for this topology."""
    return tuple(sorted(_SUPPORTED_VARIANTS))


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



def build_model(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build the global-pool CNN topology model."""
    params = dict(topology_params or {})
    input_channels = int(params.pop("input_channels", 1))
    dropout_p = float(params.pop("dropout_p", 0.2))
    if params:
        raise ValueError(
            "Unsupported topology_params for distance_regressor_global_pool_cnn: "
            f"{sorted(params.keys())}. Supported keys: ['dropout_p', 'input_channels']"
        )
    return DistanceRegressorGlobalPoolCNN(
        input_channels=input_channels,
        dropout_p=dropout_p,
        architecture_variant=topology_variant,
    )



def architecture_text(model: nn.Module) -> str:
    """Render architecture text for logging/artifacts."""
    variant = getattr(model, "architecture_variant", "unknown")
    return f"architecture_variant={variant}\\n{model}"
