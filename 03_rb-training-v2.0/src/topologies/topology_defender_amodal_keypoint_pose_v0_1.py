"""Defender amodal keypoint pose v0.1 implementation."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .topology_tri_stream_yaw_common import TriStreamYawBase, make_dropout

VARIANT = "defender_amodal_keypoint_pose_v0_1"
NUM_DEFENDER_KEYPOINTS = 10
COORDINATE_WIDTH = 3
KEYPOINT_OUTPUT_WIDTH = NUM_DEFENDER_KEYPOINTS * COORDINATE_WIDTH
ABLATION_TRI_STREAM = "tri_stream"
ABLATION_GEOMETRY_ONLY = "geometry_only"
SUPPORTED_ABLATION_MODES = (ABLATION_TRI_STREAM, ABLATION_GEOMETRY_ONLY)


class DefenderAmodalKeypointPoseRegressor(TriStreamYawBase):
    """v0.1 shared-trunk Defender amodal pose/keypoint regressor."""

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
        keypoint_hidden: int | None = None,
        ablation_mode: str = ABLATION_TRI_STREAM,
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
        normalized_ablation = str(ablation_mode).strip().lower() or ABLATION_TRI_STREAM
        if normalized_ablation not in SUPPORTED_ABLATION_MODES:
            raise ValueError(
                f"Unsupported ablation_mode={ablation_mode!r}; "
                f"expected one of {list(SUPPORTED_ABLATION_MODES)}."
            )
        if keypoint_hidden is not None and int(keypoint_hidden) < 1:
            raise ValueError(f"keypoint_hidden must be positive; got {keypoint_hidden}")

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

        self.ablation_mode = normalized_ablation
        fused_dim = self.geom_feature_dim
        if self.ablation_mode == ABLATION_TRI_STREAM:
            fused_dim += self.distance_feature_dim + self.orientation_feature_dim
        trunk_dim = max(16, self.fusion_hidden // 2)
        keypoint_hidden_dim = int(keypoint_hidden) if keypoint_hidden is not None else trunk_dim

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
        self.center_3d_head = nn.Linear(trunk_dim, COORDINATE_WIDTH)
        self.keypoints_3d_head = nn.Sequential(
            nn.Linear(trunk_dim, keypoint_hidden_dim),
            nn.ReLU(inplace=True),
            make_dropout(self.dropout_p),
            nn.Linear(keypoint_hidden_dim, KEYPOINT_OUTPUT_WIDTH),
        )
        self.visibility_head = nn.Linear(trunk_dim, NUM_DEFENDER_KEYPOINTS)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        shared = self.fusion_trunk(self._fused_features(batch))
        distance = self.distance_head(shared).squeeze(-1)
        yaw_sin_cos = self.orientation_head(shared)
        center_3d = self.center_3d_head(shared)
        keypoints_3d_flat = self.keypoints_3d_head(shared)
        visibility_logits = self.visibility_head(shared)
        return {
            "distance_m": distance,
            "yaw_sin_cos": yaw_sin_cos,
            "defender_center_3d": center_3d,
            "defender_keypoints_3d_flat": keypoints_3d_flat,
            "defender_keypoints_visible_logits": visibility_logits,
        }

    def _fused_features(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.ablation_mode == ABLATION_GEOMETRY_ONLY:
            return self._encode_geometry_only(batch)
        geom, distance_features, orientation_features = self._encode_streams(batch)
        return torch.cat([geom, distance_features, orientation_features], dim=1)

    def _encode_geometry_only(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not isinstance(batch, Mapping):
            raise TypeError(
                f"{type(self).__name__} expects a mapping with key 'x_geometry' "
                "for ablation_mode='geometry_only'."
            )
        if "x_geometry" not in batch:
            raise KeyError(
                f"{type(self).__name__} batch missing required key 'x_geometry' "
                "for ablation_mode='geometry_only'."
            )
        x_geometry = batch["x_geometry"]
        if not torch.is_tensor(x_geometry):
            raise TypeError(f"x_geometry must be a torch.Tensor; got {type(x_geometry).__name__}")
        if x_geometry.ndim != 2:
            raise ValueError(f"x_geometry must have shape (B, F); got {tuple(x_geometry.shape)}")
        if int(x_geometry.shape[1]) != int(self.geometry_feature_dim):
            raise ValueError(
                f"x_geometry width mismatch; expected {self.geometry_feature_dim}, "
                f"got {int(x_geometry.shape[1])}"
            )
        parameter = next(self.parameters())
        return self.geom_mlp(x_geometry.to(device=parameter.device, dtype=parameter.dtype))


def build_model(**kwargs: object) -> nn.Module:
    """Build a v0.1 Defender amodal keypoint pose model from parsed params."""
    return DefenderAmodalKeypointPoseRegressor(**kwargs)
