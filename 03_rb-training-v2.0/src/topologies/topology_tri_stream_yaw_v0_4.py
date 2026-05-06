"""Tri-stream distance + yaw v0.4 implementation."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .topology_tri_stream_yaw_common import TriStreamYawBase, make_dropout

VARIANT = "tri_stream_yaw_v0_4"


class DistanceRegressorTriStreamYaw(TriStreamYawBase):
    """v0.4 distance-protected, yaw-coupled tri-stream regressor."""

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

        camera_input_dim = self.geom_feature_dim + self.distance_feature_dim
        camera_dim = max(16, self.fusion_hidden // 2)
        yaw_input_dim = (
            self.geom_feature_dim
            + self.distance_feature_dim
            + camera_dim
            + self.orientation_feature_dim
        )
        yaw_dim = max(16, self.fusion_hidden // 2)

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

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        geom, distance_features, orientation_features = self._encode_streams(batch)
        camera_input = torch.cat([geom, distance_features], dim=1)
        camera_feat = self.camera_trunk(camera_input)
        distance = self.distance_head(camera_feat).squeeze(-1)

        yaw_input = torch.cat(
            [geom, distance_features, camera_feat, orientation_features],
            dim=1,
        )
        yaw_feat = self.yaw_trunk(yaw_input)
        yaw_sin_cos = self.orientation_head(yaw_feat)
        return {
            "distance_m": distance,
            "yaw_sin_cos": yaw_sin_cos,
        }


def build_model(**kwargs: object) -> nn.Module:
    """Build a v0.4 tri-stream yaw model from parsed topology params."""
    return DistanceRegressorTriStreamYaw(**kwargs)
