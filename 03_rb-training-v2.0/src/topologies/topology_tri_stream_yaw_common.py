"""Shared building blocks for tri-stream distance + yaw topology variants."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .topology_dual_stream_v0_2 import _make_dropout, _make_group_norm

SUPPORTED_PARAM_KEYS = (
    "input_channels",
    "orientation_input_channels",
    "geometry_feature_dim",
    "canvas_size",
    "dropout_p",
    "geom_hidden",
    "geom_feature_dim",
    "distance_feature_dim",
    "orientation_feature_dim",
    "fusion_hidden",
)


def parse_common_topology_params(
    topology_params: Mapping[str, Any] | None,
    *,
    topology_id: str,
) -> dict[str, Any]:
    """Parse shared tri-stream topology params for one variant constructor."""
    params = dict(topology_params or {})

    parsed = {
        "input_channels": int(params.pop("input_channels", 1)),
        "orientation_input_channels": int(params.pop("orientation_input_channels", 1)),
        "geometry_feature_dim": int(params.pop("geometry_feature_dim", 10)),
        "canvas_size": int(params.pop("canvas_size", 300)),
        "dropout_p": (
            0.0
            if (raw_dropout_p := params.pop("dropout_p", None)) is None
            else float(raw_dropout_p)
        ),
        "geom_hidden": int(params.pop("geom_hidden", 64)),
        "geom_feature_dim": int(params.pop("geom_feature_dim", 32)),
        "distance_feature_dim": int(params.pop("distance_feature_dim", 128)),
        "orientation_feature_dim": int(params.pop("orientation_feature_dim", 128)),
        "fusion_hidden": int(params.pop("fusion_hidden", 128)),
    }

    if params:
        raise ValueError(
            "Unsupported topology_params for "
            f"{topology_id}: {sorted(params.keys())}. "
            f"Supported keys: {list(SUPPORTED_PARAM_KEYS)}"
        )
    return parsed


def make_dropout(dropout_p: float) -> nn.Module:
    """Return the topology-standard dropout/identity layer."""
    return _make_dropout(dropout_p)


def make_image_encoder(input_channels: int, output_dim: int) -> nn.Sequential:
    """Build the shared CNN encoder used by distance and orientation streams."""
    return nn.Sequential(
        nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
        _make_group_norm(16),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        _make_group_norm(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        _make_group_norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        _make_group_norm(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1),
        _make_group_norm(output_dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )


class TriStreamYawBase(nn.Module):
    """Shared input validation and stream encoders for tri-stream yaw variants."""

    def __init__(
        self,
        *,
        architecture_variant: str,
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
    ) -> None:
        super().__init__()

        if input_channels < 1:
            raise ValueError(f"input_channels must be >= 1, got {input_channels}")
        if orientation_input_channels < 1:
            raise ValueError(
                f"orientation_input_channels must be >= 1, got {orientation_input_channels}"
            )
        if geometry_feature_dim < 1:
            raise ValueError(f"geometry_feature_dim must be >= 1, got {geometry_feature_dim}")
        if canvas_size < 32:
            raise ValueError(f"canvas_size must be >= 32, got {canvas_size}")
        if not (0.0 <= float(dropout_p) < 1.0):
            raise ValueError(f"dropout_p must be in [0, 1); got {dropout_p}")

        self.architecture_variant = architecture_variant
        self.input_channels = int(input_channels)
        self.orientation_input_channels = int(orientation_input_channels)
        self.geometry_feature_dim = int(geometry_feature_dim)
        self.canvas_size = int(canvas_size)
        self.dropout_p = float(dropout_p)
        self.geom_hidden = int(geom_hidden)
        self.geom_feature_dim = int(geom_feature_dim)
        self.distance_feature_dim = int(distance_feature_dim)
        self.orientation_feature_dim = int(orientation_feature_dim)
        self.fusion_hidden = int(fusion_hidden)

        self.geom_mlp = nn.Sequential(
            nn.Linear(self.geometry_feature_dim, self.geom_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.geom_hidden, self.geom_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.geom_hidden, self.geom_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.distance_cnn = make_image_encoder(self.input_channels, self.distance_feature_dim)
        self.orientation_cnn = make_image_encoder(
            self.orientation_input_channels,
            self.orientation_feature_dim,
        )

    def _require_4d(
        self,
        value: torch.Tensor,
        *,
        key: str,
        batch_size: int | None = None,
        channels: int,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"{key} must be a torch.Tensor; got {type(value).__name__}")
        if value.ndim != 4:
            raise ValueError(f"{key} must have shape (B, C, H, W); got {tuple(value.shape)}")
        if int(value.shape[1]) != int(channels):
            raise ValueError(
                f"{key} channel mismatch; expected {channels}, got {int(value.shape[1])}"
            )
        if batch_size is not None and int(value.shape[0]) != int(batch_size):
            raise ValueError(
                f"{key} batch size mismatch; expected {batch_size}, got {int(value.shape[0])}"
            )
        return value

    def _require_geometry(
        self,
        value: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"x_geometry must be a torch.Tensor; got {type(value).__name__}")
        if value.ndim != 2 or int(value.shape[0]) != int(batch_size):
            raise ValueError(
                "x_geometry must have shape (B, F) aligned with image batch size; "
                f"got shape={tuple(value.shape)} batch_size={batch_size}"
            )
        if int(value.shape[1]) != int(self.geometry_feature_dim):
            raise ValueError(
                f"x_geometry width mismatch; expected {self.geometry_feature_dim}, "
                f"got {int(value.shape[1])}"
            )
        return value.to(device=device, dtype=dtype)

    def _encode_streams(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                f"{type(self).__name__} expects a mapping with keys "
                "'x_distance_image', 'x_orientation_image', and 'x_geometry'."
            )
        missing = [
            key
            for key in ("x_distance_image", "x_orientation_image", "x_geometry")
            if key not in batch
        ]
        if missing:
            raise KeyError(
                f"{type(self).__name__} batch missing required key(s): "
                f"{missing}. Expected keys: ['x_distance_image', 'x_orientation_image', "
                "'x_geometry']."
            )

        distance_image = self._require_4d(
            batch["x_distance_image"],
            key="x_distance_image",
            channels=self.input_channels,
        )
        orientation_image = self._require_4d(
            batch["x_orientation_image"],
            key="x_orientation_image",
            batch_size=int(distance_image.shape[0]),
            channels=self.orientation_input_channels,
        ).to(device=distance_image.device, dtype=distance_image.dtype)
        if tuple(orientation_image.shape[2:]) != tuple(distance_image.shape[2:]):
            raise ValueError(
                "x_orientation_image geometry must match x_distance_image; "
                f"distance={tuple(distance_image.shape)}, orientation={tuple(orientation_image.shape)}"
            )
        x_geometry = self._require_geometry(
            batch["x_geometry"],
            batch_size=int(distance_image.shape[0]),
            device=distance_image.device,
            dtype=distance_image.dtype,
        )

        geom = self.geom_mlp(x_geometry)
        distance_features = self.distance_cnn(distance_image)
        orientation_features = self.orientation_cnn(orientation_image)
        return geom, distance_features, orientation_features
