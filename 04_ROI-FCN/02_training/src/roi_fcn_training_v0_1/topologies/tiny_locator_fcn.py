"""Tiny fully-convolutional centre-localiser topology family."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from ..contracts import TOPOLOGY_CONTRACT_VERSION

TOPOLOGY_ID = "roi_fcn_tiny"
MODEL_CLASS_NAME = "TinyRoiFcnLocaliser"
DEFAULT_VARIANT = "tiny_v1"
TOPOLOGY_METADATA = {
    "display_name": "ROI FCN Tiny",
    "status": "active",
    "note": "Single-head centre-likelihood heatmap localiser.",
}

_VARIANT_CHANNELS: dict[str, tuple[int, int, int]] = {
    "tiny_v1": (16, 32, 64),
    "tiny_wide_v1": (24, 48, 96),
}


class TinyRoiFcnLocaliser(nn.Module):
    """A very small encoder-decoder FCN that predicts one heatmap."""

    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)).sigmoid()


def supported_variants() -> tuple[str, ...]:
    """Return the supported architecture variants."""
    return tuple(sorted(_VARIANT_CHANNELS.keys()))


def _canonicalize_topology_params(topology_params: Mapping[str, Any] | None) -> dict[str, Any]:
    params = dict(topology_params or {})
    if not params:
        return {}
    unsupported = sorted(params.keys())
    raise ValueError(
        "Unsupported topology_params for roi_fcn_tiny; expected no params in v0.1, "
        f"got {unsupported}."
    )


def resolve_topology_contract(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the declared topology contract for this family."""
    _canonicalize_topology_params(topology_params)
    if topology_variant not in _VARIANT_CHANNELS:
        raise ValueError(
            f"Unsupported topology_variant={topology_variant!r}; "
            f"expected one of {sorted(_VARIANT_CHANNELS)}"
        )
    return {
        "contract_version": TOPOLOGY_CONTRACT_VERSION,
        "topology_id": TOPOLOGY_ID,
        "topology_variant": topology_variant,
        "family": "tiny_fcn_centre_localiser",
        "input": {
            "layout": "N,C,H,W",
            "channels": 1,
            "kind": "full_frame_locator_canvas",
        },
        "output": {
            "kind": "center_likelihood_heatmap",
            "channels": 1,
            "decode": "argmax",
            "coordinate_owner": "validated_output_shape_from_forward_pass",
        },
        "loss": {
            "name": "mse_heatmap",
            "target": "gaussian_heatmap_from_canvas_center",
        },
        "non_goals": [
            "bbox_regression",
            "classification",
            "orientation_regression",
            "segmentation",
            "multi_object_support",
        ],
    }


def build_model(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build one model instance from topology variant + params."""
    _canonicalize_topology_params(topology_params)
    channels = _VARIANT_CHANNELS.get(topology_variant)
    if channels is None:
        raise ValueError(
            f"Unsupported topology_variant={topology_variant!r}; "
            f"expected one of {sorted(_VARIANT_CHANNELS)}"
        )
    return TinyRoiFcnLocaliser(channels=channels)


def architecture_text(model: nn.Module) -> str:
    """Render a human-readable architecture summary."""
    return str(model)
