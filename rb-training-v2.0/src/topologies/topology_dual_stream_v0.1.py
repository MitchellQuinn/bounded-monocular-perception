"""Dual-stream monocular distance regressor topology.

Dual-stream model for a known vehicle instance observed by a fixed, calibrated
camera. The model estimates 3D position (or scalar distance) from two
complementary views of a single frame:

- Geometric stream: a small MLP over YOLO-derived bounding-box features,
  carrying the projective-geometry signal (image-plane location + apparent
  size -> direction + coarse depth).
- Shape stream: a compact 2D CNN over a fixed-size silhouette crop in which
  the vehicle is NOT rescaled -- it sits at its native pixel size, padded
  with background to fill the canvas. This preserves apparent-size as a
  depth cue and lets the CNN learn a pose-dependent refinement.

The two streams are fused in a small head that produces the final prediction.

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

TOPOLOGY_ID = "distance_regressor_dual_stream"
MODEL_CLASS_NAME = "DistanceRegressorDualStream"
DEFAULT_VARIANT = "dual_stream_v0_1"
_SUPPORTED_VARIANTS = {"dual_stream_v0_1"}

_SUPPORTED_OUTPUT_MODES = {"position_3d", "scalar_distance"}

_SUPPORTED_PARAM_KEYS = (
    "input_channels",
    "bbox_feature_dim",
    "canvas_size",
    "output_dim",
    "output_mode",
    "dropout_p",
    "geom_hidden",
    "geom_feature_dim",
    "shape_feature_dim",
    "fusion_hidden",
)


class DistanceRegressorDualStream(nn.Module):
    """Dual-stream distance regressor.

    Accepts a dict-style batch with keys:
      - ``silhouette_crop``: (B, C_in, H, W) float tensor in [0, 1]
      - ``bbox_features``:   (B, F_bbox)     float tensor

    Produces:
      - (B, output_dim) for ``output_mode="position_3d"`` (default)
      - (B,)            for ``output_mode="scalar_distance"``
    """

    def __init__(
        self,
        input_channels: int = 1,
        bbox_feature_dim: int = 10,
        canvas_size: int = 224,
        output_dim: int = 3,
        output_mode: str = "position_3d",
        dropout_p: float = 0.1,
        geom_hidden: int = 64,
        geom_feature_dim: int = 32,
        shape_feature_dim: int = 128,
        fusion_hidden: int = 128,
        architecture_variant: str = DEFAULT_VARIANT,
    ) -> None:
        super().__init__()

        if architecture_variant not in _SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported architecture_variant={architecture_variant}; "
                f"expected one of {sorted(_SUPPORTED_VARIANTS)}"
            )
        if output_mode not in _SUPPORTED_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported output_mode={output_mode!r}; "
                f"expected one of {sorted(_SUPPORTED_OUTPUT_MODES)}"
            )
        if output_mode == "scalar_distance" and output_dim != 1:
            raise ValueError(
                "output_mode='scalar_distance' requires output_dim=1; "
                f"got output_dim={output_dim}"
            )
        if output_mode == "position_3d" and output_dim != 3:
            raise ValueError(
                "output_mode='position_3d' requires output_dim=3; "
                f"got output_dim={output_dim}"
            )
        if input_channels < 1:
            raise ValueError(f"input_channels must be >= 1, got {input_channels}")
        if bbox_feature_dim < 1:
            raise ValueError(f"bbox_feature_dim must be >= 1, got {bbox_feature_dim}")
        if canvas_size < 32:
            raise ValueError(f"canvas_size must be >= 32, got {canvas_size}")

        self.architecture_variant = architecture_variant
        self.input_channels = int(input_channels)
        self.bbox_feature_dim = int(bbox_feature_dim)
        self.canvas_size = int(canvas_size)
        self.output_dim = int(output_dim)
        self.output_mode = output_mode
        self.dropout_p = float(dropout_p)
        self.geom_hidden = int(geom_hidden)
        self.geom_feature_dim = int(geom_feature_dim)
        self.shape_feature_dim = int(shape_feature_dim)
        self.fusion_hidden = int(fusion_hidden)

        # Geometric stream: small MLP over bbox features.
        self.geom_mlp = nn.Sequential(
            nn.Linear(self.bbox_feature_dim, self.geom_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.geom_hidden, self.geom_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.geom_hidden, self.geom_feature_dim),
            nn.ReLU(inplace=True),
        )

        # Shape stream: compact 2D CNN over the fixed-canvas silhouette crop.
        # Five stride-2 conv stages -> global average pool.
        # At canvas_size=224 the spatial flow is 224 -> 112 -> 56 -> 28 -> 14 -> 7 -> 1.
        self.shape_cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, self.shape_feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.shape_feature_dim),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Fusion head.
        fused_dim = self.geom_feature_dim + self.shape_feature_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, self.fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.fusion_hidden, self.fusion_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.fusion_hidden // 2, self.output_dim),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "DistanceRegressorDualStream expects a dict-style batch with keys "
                "'silhouette_crop' and 'bbox_features'; got "
                f"{type(batch).__name__}"
            )
        try:
            silhouette_crop = batch["silhouette_crop"]
            bbox_features = batch["bbox_features"]
        except KeyError as missing:
            raise KeyError(
                "DistanceRegressorDualStream batch missing required key "
                f"{missing!s}. Expected keys: 'silhouette_crop', 'bbox_features'."
            ) from None

        geom = self.geom_mlp(bbox_features)          # (B, geom_feature_dim)
        shape = self.shape_cnn(silhouette_crop)      # (B, shape_feature_dim)
        fused = torch.cat([geom, shape], dim=1)      # (B, fused_dim)
        out = self.fusion_head(fused)                # (B, output_dim)

        if self.output_mode == "scalar_distance":
            out = out.squeeze(-1)
        return out


def supported_variants() -> tuple[str, ...]:
    """Return all allowed topology variants."""
    return tuple(sorted(_SUPPORTED_VARIANTS))


def build_model(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build one dual-stream model instance from topology variant + params.

    Strict parsing: unknown keys raise ValueError, preventing silent typos in
    launch JSON.
    """
    params = dict(topology_params or {})

    input_channels = int(params.pop("input_channels", 1))
    bbox_feature_dim = int(params.pop("bbox_feature_dim", 10))
    canvas_size = int(params.pop("canvas_size", 224))
    output_dim = int(params.pop("output_dim", 3))
    output_mode = str(params.pop("output_mode", "position_3d"))
    dropout_p = float(params.pop("dropout_p", 0.1))
    geom_hidden = int(params.pop("geom_hidden", 64))
    geom_feature_dim = int(params.pop("geom_feature_dim", 32))
    shape_feature_dim = int(params.pop("shape_feature_dim", 128))
    fusion_hidden = int(params.pop("fusion_hidden", 128))

    if params:
        raise ValueError(
            "Unsupported topology_params for "
            f"{TOPOLOGY_ID}: {sorted(params.keys())}. "
            f"Supported keys: {list(_SUPPORTED_PARAM_KEYS)}"
        )

    return DistanceRegressorDualStream(
        input_channels=input_channels,
        bbox_feature_dim=bbox_feature_dim,
        canvas_size=canvas_size,
        output_dim=output_dim,
        output_mode=output_mode,
        dropout_p=dropout_p,
        geom_hidden=geom_hidden,
        geom_feature_dim=geom_feature_dim,
        shape_feature_dim=shape_feature_dim,
        fusion_hidden=fusion_hidden,
        architecture_variant=topology_variant,
    )


def architecture_text(model: nn.Module) -> str:
    """Render architecture text persisted in run artifacts."""
    variant = getattr(model, "architecture_variant", "unknown")
    return f"architecture_variant={variant}\n{model}"