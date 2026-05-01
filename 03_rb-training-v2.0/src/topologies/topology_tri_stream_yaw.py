"""Tri-stream distance + yaw topology with distance image, orientation image, and geometry inputs."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .contracts import TOPOLOGY_CONTRACT_VERSION, task_contract_from_topology_contract
from .topology_dual_stream_v0_2 import _make_dropout, _make_group_norm

TOPOLOGY_ID = "distance_regressor_tri_stream_yaw"
MODEL_CLASS_NAME = "DistanceRegressorTriStreamYaw"
DEFAULT_VARIANT = "tri_stream_yaw_v0_1"
TOPOLOGY_METADATA = {
    "status": "experimental",
    "display_name": "Distance Regressor Tri Stream + Yaw",
    "note": "Distance plus yaw multitask topology with distance image, orientation image, and geometry streams.",
    "replacement": "",
}
_SUPPORTED_VARIANTS = {"tri_stream_yaw_v0_1"}
_SUPPORTED_PARAM_KEYS = (
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

TOPOLOGY_CONTRACT = {
    "contract_version": TOPOLOGY_CONTRACT_VERSION,
    "task_family": "multitask_regression",
    "targets": {
        "distance": {
            "kind": "regression",
            "columns": ["distance_m"],
            "target_npz_key": "y_distance_m",
        },
        "yaw": {
            "kind": "circular_regression",
            "columns": ["yaw_sin", "yaw_cos"],
            "debug_columns": ["yaw_deg"],
            "target_npz_keys": ["y_yaw_sin", "y_yaw_cos"],
            "debug_target_npz_key": "y_yaw_deg",
        },
    },
    "outputs": {
        "distance": {
            "kind": "regression",
            "columns": ["distance_m"],
            "output_key": "distance_m",
        },
        "yaw": {
            "kind": "circular_regression",
            "columns": ["yaw_sin", "yaw_cos"],
            "output_key": "yaw_sin_cos",
        },
    },
    "runtime": {
        "prediction_mode": "distance_yaw_sincos",
        "input_mode": "tri_stream_distance_orientation_geometry",
        "output_kind": "mapping",
        "heads": {
            "distance": {
                "output": "distance",
                "target": "distance",
                "metrics_role": "distance",
                "loss_role": "distance",
            },
            "orientation": {
                "output": "yaw",
                "target": "yaw",
                "metrics_role": "orientation",
                "loss_role": "orientation",
            },
        },
    },
    "reporting": {
        "family": "distance_orientation_multitask",
        "train_losses": ["total_loss", "distance_loss", "orientation_loss"],
        "validation_metrics": [
            "yaw_mean_error_deg",
            "yaw_median_error_deg",
            "yaw_p95_error_deg",
            "yaw_acc@5deg",
            "yaw_acc@10deg",
            "yaw_acc@15deg",
        ],
        "orientation_accuracy_thresholds_deg": [5.0, 10.0, 15.0],
    },
}


def _make_image_encoder(input_channels: int, output_dim: int) -> nn.Sequential:
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


class DistanceRegressorTriStreamYaw(nn.Module):
    """Tri-stream distance + yaw regressor.

    v0.1 accepts only a mapping with:
    - ``x_distance_image``: fixed unscaled distance image tensor ``(B, C, H, W)``
    - ``x_orientation_image``: target-normalised orientation image tensor ``(B, C, H, W)``
    - ``x_geometry``: geometry/context vector ``(B, F)``
    """

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
        architecture_variant: str = DEFAULT_VARIANT,
    ) -> None:
        super().__init__()

        if architecture_variant not in _SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported architecture_variant={architecture_variant}; "
                f"expected one of {sorted(_SUPPORTED_VARIANTS)}"
            )
        if input_channels < 1:
            raise ValueError(f"input_channels must be >= 1, got {input_channels}")
        if orientation_input_channels < 1:
            raise ValueError(f"orientation_input_channels must be >= 1, got {orientation_input_channels}")
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
        self.distance_cnn = _make_image_encoder(self.input_channels, self.distance_feature_dim)
        self.orientation_cnn = _make_image_encoder(
            self.orientation_input_channels,
            self.orientation_feature_dim,
        )

        fused_dim = self.geom_feature_dim + self.distance_feature_dim + self.orientation_feature_dim
        trunk_dim = max(16, self.fusion_hidden // 2)
        self.fusion_trunk = nn.Sequential(
            nn.Linear(fused_dim, self.fusion_hidden),
            nn.ReLU(inplace=True),
            _make_dropout(self.dropout_p),
            nn.Linear(self.fusion_hidden, trunk_dim),
            nn.ReLU(inplace=True),
            _make_dropout(self.dropout_p),
        )
        self.distance_head = nn.Linear(trunk_dim, 1)
        self.orientation_head = nn.Linear(trunk_dim, 2)

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
            raise ValueError(f"{key} channel mismatch; expected {channels}, got {int(value.shape[1])}")
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
                f"x_geometry width mismatch; expected {self.geometry_feature_dim}, got {int(value.shape[1])}"
            )
        return value.to(device=device, dtype=dtype)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise TypeError(
                "DistanceRegressorTriStreamYaw expects a mapping with keys "
                "'x_distance_image', 'x_orientation_image', and 'x_geometry'."
            )
        missing = [
            key
            for key in ("x_distance_image", "x_orientation_image", "x_geometry")
            if key not in batch
        ]
        if missing:
            raise KeyError(
                "DistanceRegressorTriStreamYaw batch missing required key(s): "
                f"{missing}. Expected keys: ['x_distance_image', 'x_orientation_image', 'x_geometry']."
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
        fused = torch.cat([geom, distance_features, orientation_features], dim=1)
        shared = self.fusion_trunk(fused)
        distance = self.distance_head(shared).squeeze(-1)
        yaw_sin_cos = self.orientation_head(shared)
        return {
            "distance_m": distance,
            "yaw_sin_cos": yaw_sin_cos,
        }


def supported_variants() -> tuple[str, ...]:
    """Return all allowed variants."""
    return tuple(sorted(_SUPPORTED_VARIANTS))


def resolve_topology_contract(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the declared output/reporting contract for this topology."""
    _ = topology_variant
    _ = topology_params
    return dict(TOPOLOGY_CONTRACT)


def resolve_task_contract(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the training/evaluation contract for this topology family."""
    return task_contract_from_topology_contract(
        resolve_topology_contract(topology_variant, topology_params)
    )


def build_model(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build one tri-stream multitask model instance from topology variant + params."""
    params = dict(topology_params or {})

    input_channels = int(params.pop("input_channels", 1))
    orientation_input_channels = int(params.pop("orientation_input_channels", 1))
    geometry_feature_dim = int(params.pop("geometry_feature_dim", 10))
    canvas_size = int(params.pop("canvas_size", 300))
    raw_dropout_p = params.pop("dropout_p", None)
    dropout_p = 0.0 if raw_dropout_p is None else float(raw_dropout_p)
    geom_hidden = int(params.pop("geom_hidden", 64))
    geom_feature_dim = int(params.pop("geom_feature_dim", 32))
    distance_feature_dim = int(params.pop("distance_feature_dim", 128))
    orientation_feature_dim = int(params.pop("orientation_feature_dim", 128))
    fusion_hidden = int(params.pop("fusion_hidden", 128))

    if params:
        raise ValueError(
            "Unsupported topology_params for "
            f"{TOPOLOGY_ID}: {sorted(params.keys())}. "
            f"Supported keys: {list(_SUPPORTED_PARAM_KEYS)}"
        )

    return DistanceRegressorTriStreamYaw(
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
        architecture_variant=topology_variant,
    )


def architecture_text(model: nn.Module) -> str:
    """Render architecture text persisted in run artifacts."""
    variant = getattr(model, "architecture_variant", "unknown")
    return f"architecture_variant={variant}\n{model}"
