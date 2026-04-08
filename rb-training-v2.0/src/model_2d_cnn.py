"""2D CNN regressors for distance prediction."""

from __future__ import annotations

import torch
from torch import nn

MODEL_ARCHITECTURE_VARIANTS = {"plain_v0_1", "fast_v0_2"}


class DistanceRegressor2DCNN(nn.Module):
    """Strided-convolution model for scalar distance regression."""

    def __init__(
        self,
        input_channels: int = 1,
        dropout_p: float = 0.2,
        architecture_variant: str = "plain_v0_1",
    ) -> None:
        super().__init__()
        if architecture_variant not in MODEL_ARCHITECTURE_VARIANTS:
            raise ValueError(
                f"Unsupported architecture_variant={architecture_variant}; "
                f"expected one of {sorted(MODEL_ARCHITECTURE_VARIANTS)}"
            )
        self.architecture_variant = architecture_variant

        if architecture_variant == "plain_v0_1":
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            head_width = 128
            head_hidden = 64
        else:
            # Fast variant for large full-HD inputs: downsample early to reduce per-batch compute.
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 24, kernel_size=7, stride=4, padding=3, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            head_width = 96
            head_hidden = 48

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_width, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.head(x)
        return x.squeeze(-1)


def architecture_text(model: nn.Module) -> str:
    """Return architecture text suitable for artifact logging."""
    variant = getattr(model, "architecture_variant", "unknown")
    return f"architecture_variant={variant}\n{model}"
