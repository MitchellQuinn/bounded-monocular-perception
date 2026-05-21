"""Tri-stream distance + yaw v0.5 implementation."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .topology_tri_stream_yaw_common import TriStreamYawBase, make_dropout

VARIANT = "tri_stream_yaw_v0_5"
DEFAULT_RESIDUAL_LIMIT_M = 0.35
RESIDUAL_GEOMETRY_SCHEMA = (
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "aspect_ratio",
    "log_aspect_ratio",
    "area_norm",
    "sqrt_area_norm",
)


class DistanceRegressorTriStreamYaw(TriStreamYawBase):
    """v0.5 distance-protected topology with bounded pose-conditioned residual."""

    def __init__(
        self,
        input_channels: int = 1,
        orientation_input_channels: int = 1,
        geometry_feature_dim: int = 10,
        canvas_size: int = 300,
        dropout_p: float = 0.0,
        geom_hidden: int = 64,
        geom_feature_dim: int = 32,
        distance_feature_dim: int = 128,
        orientation_feature_dim: int = 128,
        fusion_hidden: int = 128,
        residual_limit_m: float = DEFAULT_RESIDUAL_LIMIT_M,
        architecture_variant: str = VARIANT,
    ) -> None:
        if architecture_variant != VARIANT:
            raise ValueError(
                f"Unsupported architecture_variant={architecture_variant}; expected {VARIANT}"
            )
        if int(geometry_feature_dim) != 10:
            raise ValueError(
                f"{VARIANT} requires the 10-field tri-stream geometry schema; "
                f"got geometry_feature_dim={geometry_feature_dim}."
            )
        if float(residual_limit_m) <= 0.0:
            raise ValueError(f"residual_limit_m must be positive; got {residual_limit_m}")
        super().__init__(
            architecture_variant=architecture_variant,
            input_channels=input_channels,
            orientation_input_channels=orientation_input_channels,
            geometry_feature_dim=geometry_feature_dim,
            canvas_size=canvas_size,
            dropout_p=dropout_p,
            geom_hidden=geom_hidden,
            geom_feature_dim=geom_feature_dim,
            distance_feature_dim=distance_feature_dim,
            orientation_feature_dim=orientation_feature_dim,
            fusion_hidden=fusion_hidden,
        )

        self.residual_limit_m = float(residual_limit_m)

        camera_input_dim = self.geom_feature_dim + self.distance_feature_dim
        camera_dim = max(16, self.fusion_hidden // 2)
        yaw_input_dim = (
            self.geom_feature_dim
            + self.distance_feature_dim
            + camera_dim
            + self.orientation_feature_dim
        )
        yaw_dim = max(16, self.fusion_hidden // 2)
        residual_input_dim = camera_dim + 2 + len(RESIDUAL_GEOMETRY_SCHEMA)
        residual_hidden = max(16, self.fusion_hidden // 2)

        self.camera_trunk = nn.Sequential(
            nn.Linear(camera_input_dim, self.fusion_hidden),
            nn.ReLU(inplace=True),
            make_dropout(self.dropout_p),
            nn.Linear(self.fusion_hidden, camera_dim),
            nn.ReLU(inplace=True),
            make_dropout(self.dropout_p),
        )
        self.distance_head = nn.Linear(camera_dim, 1)
        self.yaw_trunk = nn.Sequential(
            nn.Linear(yaw_input_dim, self.fusion_hidden),
            nn.ReLU(inplace=True),
            make_dropout(self.dropout_p),
            nn.Linear(self.fusion_hidden, yaw_dim),
            nn.ReLU(inplace=True),
            make_dropout(self.dropout_p),
        )
        self.orientation_head = nn.Linear(yaw_dim, 2)
        self.distance_residual_head = nn.Sequential(
            nn.Linear(residual_input_dim, residual_hidden),
            nn.ReLU(inplace=True),
            make_dropout(self.dropout_p),
            nn.Linear(residual_hidden, residual_hidden),
            nn.ReLU(inplace=True),
            make_dropout(self.dropout_p),
            nn.Linear(residual_hidden, 1),
        )
        _initialize_near_zero_residual(self.distance_residual_head)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        geom, distance_features, orientation_features = self._encode_streams(batch)
        raw_geometry = self._require_geometry(
            batch["x_geometry"],
            batch_size=int(distance_features.shape[0]),
            device=distance_features.device,
            dtype=distance_features.dtype,
        )

        camera_input = torch.cat([geom, distance_features], dim=1)
        camera_feat = self.camera_trunk(camera_input)
        distance_base = self.distance_head(camera_feat).squeeze(-1)

        yaw_input = torch.cat(
            [geom, distance_features, camera_feat, orientation_features],
            dim=1,
        )
        yaw_feat = self.yaw_trunk(yaw_input)
        yaw_sin_cos = self.orientation_head(yaw_feat)

        yaw_context = F.normalize(yaw_sin_cos.detach(), dim=1, eps=1e-6)
        residual_geometry = self._residual_geometry_features(raw_geometry)
        residual_input = torch.cat(
            [camera_feat, yaw_context, residual_geometry],
            dim=1,
        )
        residual_raw = self.distance_residual_head(residual_input).squeeze(-1)
        distance_delta = self.residual_limit_m * torch.tanh(residual_raw)
        distance = distance_base + distance_delta

        return {
            "distance_m": distance,
            "yaw_sin_cos": yaw_sin_cos,
        }

    def _residual_geometry_features(self, x_geometry: torch.Tensor) -> torch.Tensor:
        cx_norm = x_geometry[:, 4:5]
        cy_norm = x_geometry[:, 5:6]
        w_norm = x_geometry[:, 6:7]
        h_norm = x_geometry[:, 7:8]
        aspect_ratio = torch.clamp(x_geometry[:, 8:9], min=1e-6)
        area_norm = torch.clamp(x_geometry[:, 9:10], min=1e-12)
        return torch.cat(
            [
                cx_norm,
                cy_norm,
                w_norm,
                h_norm,
                aspect_ratio,
                torch.log(aspect_ratio),
                area_norm,
                torch.sqrt(area_norm),
            ],
            dim=1,
        )


def _initialize_near_zero_residual(module: nn.Sequential) -> None:
    final_linear = next(
        child for child in reversed(module) if isinstance(child, nn.Linear)
    )
    nn.init.normal_(final_linear.weight, mean=0.0, std=1e-4)
    nn.init.zeros_(final_linear.bias)


def build_model(**kwargs: object) -> nn.Module:
    """Build a v0.5 tri-stream yaw model from parsed topology params."""
    return DistanceRegressorTriStreamYaw(**kwargs)
