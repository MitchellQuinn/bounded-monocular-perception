"""Gaussian heatmap target generation for ROI-FCN supervision."""

from __future__ import annotations

import torch


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
