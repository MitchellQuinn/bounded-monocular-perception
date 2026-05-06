"""Tri-stream distance + yaw v0.1 implementation."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .topology_tri_stream_yaw_common import TriStreamYawBase, make_dropout

VARIANT = "tri_stream_yaw_v0_1"


class DistanceRegressorTriStreamYaw(TriStreamYawBase):
    """v0.1 shared-trunk tri-stream distance + yaw regressor."""

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
        architecture_variant: str = VARIANT,
    ) -> None:
        if architecture_variant != VARIANT:
            raise ValueError(
                f"Unsupported architecture_variant={architecture_variant}; expected {VARIANT}"
            )
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

        fused_dim = self.geom_feature_dim + self.distance_feature_dim + self.orientation_feature_dim
        trunk_dim = max(16, self.fusion_hidden // 2)
        self.fusion_trunk = nn.Sequential(
            nn.Linear(fused_dim, self.fusion_hidden),
            nn.ReLU(inplace=True),
            make_dropout(self.dropout_p),
            nn.Linear(self.fusion_hidden, trunk_dim),
            nn.ReLU(inplace=True),
            make_dropout(self.dropout_p),
        )
        self.distance_head = nn.Linear(trunk_dim, 1)
        self.orientation_head = nn.Linear(trunk_dim, 2)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        geom, distance_features, orientation_features = self._encode_streams(batch)
        fused = torch.cat([geom, distance_features, orientation_features], dim=1)
        shared = self.fusion_trunk(fused)
        distance = self.distance_head(shared).squeeze(-1)
        yaw_sin_cos = self.orientation_head(shared)
        return {
            "distance_m": distance,
            "yaw_sin_cos": yaw_sin_cos,
        }


def build_model(**kwargs: object) -> nn.Module:
    """Build a v0.1 tri-stream yaw model from parsed topology params."""
    return DistanceRegressorTriStreamYaw(**kwargs)
