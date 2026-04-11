"""Deprecated dual-stream monocular distance regressor topology.

This module is kept only as an archival reference for the retired v0.1
experiment. It is no longer registered for new runs.

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

    Accepts either:
      - a dict-style batch with keys:
      - ``silhouette_crop``: (B, C_in, H, W) float tensor in [0, 1]
      - ``bbox_features``:   (B, F_bbox)     float tensor
      - or a raw silhouette tensor ``(B, C_in, H, W)``. In this mode, bbox
        features are derived from non-zero silhouette extents.

    Produces:
      - (B,)            for ``output_mode="scalar_distance"`` (default)
      - (B, output_dim) for ``output_mode="position_3d"``
    """

    def __init__(
        self,
        input_channels: int = 1,
        bbox_feature_dim: int = 10,
        canvas_size: int = 224,
        output_dim: int = 1,
        output_mode: str = "scalar_distance",
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

    def _derive_bbox_features_from_silhouette(
        self,
        silhouette_crop: torch.Tensor,
    ) -> torch.Tensor:
        """Derive normalized bbox-style features from silhouette pixels.

        This keeps the dual-stream topology compatible with existing scalar
        training/evaluation loops that currently pass only an image tensor.
        """
        if silhouette_crop.ndim != 4:
            raise ValueError(
                "silhouette_crop must have shape (B, C, H, W); "
                f"got shape={tuple(silhouette_crop.shape)}"
            )
        batch_size, _, height, width = silhouette_crop.shape
        device = silhouette_crop.device
        dtype = silhouette_crop.dtype

        # Treat non-zero pixels as foreground.
        mask = silhouette_crop.amax(dim=1) > 0  # (B, H, W)
        row_any = mask.any(dim=2)               # (B, H)
        col_any = mask.any(dim=1)               # (B, W)
        valid = row_any.any(dim=1) & col_any.any(dim=1)  # (B,)

        row_idx = torch.arange(height, device=device, dtype=torch.int64).unsqueeze(0)
        col_idx = torch.arange(width, device=device, dtype=torch.int64).unsqueeze(0)

        y_min = torch.where(row_any, row_idx, torch.full_like(row_idx, height)).min(dim=1).values
        y_max = torch.where(row_any, row_idx, torch.full_like(row_idx, -1)).max(dim=1).values
        x_min = torch.where(col_any, col_idx, torch.full_like(col_idx, width)).min(dim=1).values
        x_max = torch.where(col_any, col_idx, torch.full_like(col_idx, -1)).max(dim=1).values

        valid_i64 = valid.to(dtype=torch.int64)
        y_min = y_min * valid_i64
        y_max = y_max * valid_i64
        x_min = x_min * valid_i64
        x_max = x_max * valid_i64

        width_px = ((x_max - x_min + 1).clamp(min=0)).to(dtype=dtype)
        height_px = ((y_max - y_min + 1).clamp(min=0)).to(dtype=dtype)
        x_min_f = x_min.to(dtype=dtype)
        x_max_f = x_max.to(dtype=dtype)
        y_min_f = y_min.to(dtype=dtype)
        y_max_f = y_max.to(dtype=dtype)
        cx = 0.5 * (x_min_f + x_max_f)
        cy = 0.5 * (y_min_f + y_max_f)

        width_denom = float(max(width - 1, 1))
        height_denom = float(max(height - 1, 1))
        area_denom = float(max(height * width, 1))
        eps = torch.tensor(1e-6, device=device, dtype=dtype)

        base_features = torch.stack(
            [
                cx / width_denom,
                cy / height_denom,
                width_px / float(max(width, 1)),
                height_px / float(max(height, 1)),
                (width_px * height_px) / area_denom,
                width_px / (height_px + eps),
                x_min_f / width_denom,
                y_min_f / height_denom,
                x_max_f / width_denom,
                y_max_f / height_denom,
            ],
            dim=1,
        )  # (B, 10)

        feature_dim = int(self.bbox_feature_dim)
        if base_features.shape[1] == feature_dim:
            return base_features
        if base_features.shape[1] > feature_dim:
            return base_features[:, :feature_dim]
        pad = torch.zeros(
            (batch_size, feature_dim - base_features.shape[1]),
            dtype=dtype,
            device=device,
        )
        return torch.cat([base_features, pad], dim=1)

    def _coerce_bbox_features(
        self,
        bbox_features: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if bbox_features.ndim != 2 or int(bbox_features.shape[0]) != int(batch_size):
            raise ValueError(
                "bbox_features must have shape (B, F) aligned with silhouette batch size; "
                f"got shape={tuple(bbox_features.shape)} batch_size={batch_size}"
            )
        features = bbox_features.to(device=device, dtype=dtype)
        feature_dim = int(self.bbox_feature_dim)
        if int(features.shape[1]) == feature_dim:
            return features
        if int(features.shape[1]) > feature_dim:
            return features[:, :feature_dim]
        pad = torch.zeros(
            (batch_size, feature_dim - int(features.shape[1])),
            dtype=dtype,
            device=device,
        )
        return torch.cat([features, pad], dim=1)

    def forward(self, batch: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(batch, Mapping):
            try:
                silhouette_crop = batch["silhouette_crop"]
            except KeyError as missing:
                raise KeyError(
                    "DistanceRegressorDualStream batch missing required key "
                    f"{missing!s}. Expected key: 'silhouette_crop'."
                ) from None
            raw_bbox_features = batch.get("bbox_features")
            if raw_bbox_features is None:
                bbox_features = self._derive_bbox_features_from_silhouette(silhouette_crop)
            else:
                bbox_features = self._coerce_bbox_features(
                    raw_bbox_features,
                    batch_size=int(silhouette_crop.shape[0]),
                    device=silhouette_crop.device,
                    dtype=silhouette_crop.dtype,
                )
        elif torch.is_tensor(batch):
            silhouette_crop = batch
            bbox_features = self._derive_bbox_features_from_silhouette(silhouette_crop)
        else:
            raise TypeError(
                "DistanceRegressorDualStream expects either a dict-style batch "
                "with key 'silhouette_crop' (optionally 'bbox_features') or a "
                f"tensor silhouette input; got {type(batch).__name__}"
            )

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
    output_dim = int(params.pop("output_dim", 1))
    output_mode = str(params.pop("output_mode", "scalar_distance"))
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
