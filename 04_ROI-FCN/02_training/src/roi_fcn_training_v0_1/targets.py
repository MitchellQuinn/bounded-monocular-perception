"""Gaussian heatmap target generation for ROI-FCN supervision."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def build_gaussian_heatmaps(
    centers_canvas_px: torch.Tensor,
    *,
    canvas_hw: tuple[int, int],
    output_hw: tuple[int, int],
    sigma_px: float,
) -> torch.Tensor:
    """Create Gaussian targets in output space from canvas-space centres."""
    if centers_canvas_px.ndim != 2 or centers_canvas_px.shape[1] != 2:
        raise ValueError(
            f"centers_canvas_px must have shape (N, 2), got {tuple(centers_canvas_px.shape)}"
        )
    if float(sigma_px) <= 0.0:
        raise ValueError(f"sigma_px must be > 0; got {sigma_px}")

    canvas_h, canvas_w = int(canvas_hw[0]), int(canvas_hw[1])
    output_h, output_w = int(output_hw[0]), int(output_hw[1])
    scale_x = float(output_w) / float(canvas_w)
    scale_y = float(output_h) / float(canvas_h)

    device = centers_canvas_px.device
    dtype = centers_canvas_px.dtype
    ys = torch.arange(output_h, device=device, dtype=dtype)
    xs = torch.arange(output_w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    centers_x = centers_canvas_px[:, 0] * scale_x
    centers_y = centers_canvas_px[:, 1] * scale_y
    dist_sq = (grid_x.unsqueeze(0) - centers_x[:, None, None]) ** 2
    dist_sq = dist_sq + (grid_y.unsqueeze(0) - centers_y[:, None, None]) ** 2
    heatmaps = torch.exp(-dist_sq / (2.0 * float(sigma_px) * float(sigma_px)))
    return heatmaps.unsqueeze(1)


def build_balanced_heatmap_weights(
    target_heatmaps: torch.Tensor,
    *,
    positive_threshold: float = 0.05,
) -> torch.Tensor:
    """Balance positive and negative heatmap regions to avoid blank-map collapse."""
    if target_heatmaps.ndim != 4:
        raise ValueError(
            f"target_heatmaps must have shape (N, C, H, W), got {tuple(target_heatmaps.shape)}"
        )
    if not 0.0 < float(positive_threshold) < 1.0:
        raise ValueError(f"positive_threshold must be in (0, 1); got {positive_threshold}")

    positive_mask = target_heatmaps >= float(positive_threshold)
    total_pixels = int(target_heatmaps.shape[1] * target_heatmaps.shape[2] * target_heatmaps.shape[3])
    if total_pixels <= 0:
        raise ValueError("target_heatmaps must have non-zero spatial extent.")

    dtype = target_heatmaps.dtype
    positive_count = positive_mask.reshape(target_heatmaps.shape[0], -1).sum(dim=1, keepdim=True).to(dtype=dtype)
    total_pixels_tensor = torch.full_like(positive_count, float(total_pixels), dtype=dtype)
    negative_count = total_pixels_tensor - positive_count
    positive_weight = torch.where(
        positive_count > 0.0,
        negative_count / positive_count.clamp_min(1.0),
        torch.ones_like(positive_count, dtype=dtype),
    )

    weights = torch.ones_like(target_heatmaps, dtype=dtype)
    weights = torch.where(positive_mask, positive_weight.view(-1, 1, 1, 1), weights)
    weights = weights / weights.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return weights


def balanced_heatmap_mse_loss(
    predicted_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    *,
    positive_threshold: float = 0.05,
) -> torch.Tensor:
    """Compute a sample-balanced heatmap MSE loss."""
    if predicted_heatmaps.shape != target_heatmaps.shape:
        raise ValueError(
            "predicted_heatmaps and target_heatmaps must have matching shapes; "
            f"got {tuple(predicted_heatmaps.shape)} vs {tuple(target_heatmaps.shape)}"
        )
    weights = build_balanced_heatmap_weights(
        target_heatmaps,
        positive_threshold=float(positive_threshold),
    )
    return torch.mean(torch.square(predicted_heatmaps - target_heatmaps) * weights)


def compute_heatmap_loss(
    predicted_heatmaps: torch.Tensor,
    target_heatmaps: torch.Tensor,
    *,
    loss_name: str,
    positive_threshold: float = 0.05,
) -> torch.Tensor:
    """Compute one supported heatmap supervision loss."""
    normalized_name = str(loss_name).strip().lower()
    if normalized_name == "mse_heatmap":
        return F.mse_loss(predicted_heatmaps, target_heatmaps, reduction="mean")
    if normalized_name == "balanced_mse_heatmap":
        return balanced_heatmap_mse_loss(
            predicted_heatmaps,
            target_heatmaps,
            positive_threshold=float(positive_threshold),
        )
    raise ValueError(
        f"Unsupported heatmap loss {loss_name!r}; expected 'mse_heatmap' or 'balanced_mse_heatmap'."
    )
