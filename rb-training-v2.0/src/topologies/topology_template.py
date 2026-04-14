"""Template for adding a new model topology module.

Copy this file to `topology_<your_name>.py`, then replace placeholder values.

Required exports for registry integration:
- TOPOLOGY_ID
- MODEL_CLASS_NAME
- DEFAULT_VARIANT
- supported_variants()
- build_model(topology_variant, topology_params)
- architecture_text(model)
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

TOPOLOGY_ID = "distance_regressor_template"
MODEL_CLASS_NAME = "DistanceRegressorTemplate"
DEFAULT_VARIANT = "base_v0_1"
TOPOLOGY_METADATA = {
    "status": "deprecated",
    "display_name": "Distance Regressor Template",
    "note": "Template topology scaffold for new implementations.",
    "replacement": "",
}
_SUPPORTED_VARIANTS = {"base_v0_1"}


class DistanceRegressorTemplate(nn.Module):
    """Example topology skeleton for scalar distance regression."""

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

        # Replace this with your actual network design.
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout_p)),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)



def supported_variants() -> tuple[str, ...]:
    """Return all allowed topology variants."""
    return tuple(sorted(_SUPPORTED_VARIANTS))



def build_model(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build one model instance from topology variant + params."""
    params = dict(topology_params or {})

    # Keep params strict to prevent silent typos in launch JSON.
    input_channels = int(params.pop("input_channels", 1))
    dropout_p = float(params.pop("dropout_p", 0.2))
    if params:
        raise ValueError(
            "Unsupported topology_params for distance_regressor_template: "
            f"{sorted(params.keys())}. Supported keys: ['dropout_p', 'input_channels']"
        )

    return DistanceRegressorTemplate(
        input_channels=input_channels,
        dropout_p=dropout_p,
        architecture_variant=topology_variant,
    )



def architecture_text(model: nn.Module) -> str:
    """Render architecture text persisted in run artifacts."""
    variant = getattr(model, "architecture_variant", "unknown")
    return f"architecture_variant={variant}\\n{model}"
