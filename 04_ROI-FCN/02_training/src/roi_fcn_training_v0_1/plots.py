"""Plot helpers for ROI-FCN training and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_history_plot(history_df: pd.DataFrame, path: Path) -> None:
    """Save the train/validation loss history."""
    if history_df.empty:
        raise ValueError("history_df cannot be empty")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Loss")
    ax.plot(history_df["epoch"], history_df["validation_loss"], marker="o", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Heatmap Loss")
    ax.set_title("ROI-FCN Training Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_prediction_scatter(predictions_df: pd.DataFrame, path: Path) -> None:
    """Save predicted-vs-target scatter plots for original-space centres."""
    if predictions_df.empty:
        raise ValueError("predictions_df cannot be empty")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(predictions_df["target_center_x_px"], predictions_df["predicted_center_x_px"], s=8, alpha=0.4)
    axes[0].set_title("Center X")
    axes[0].set_xlabel("Target X")
    axes[0].set_ylabel("Predicted X")
    axes[0].grid(True, alpha=0.2)

    axes[1].scatter(predictions_df["target_center_y_px"], predictions_df["predicted_center_y_px"], s=8, alpha=0.4)
    axes[1].set_title("Center Y")
    axes[1].set_xlabel("Target Y")
    axes[1].set_ylabel("Predicted Y")
    axes[1].grid(True, alpha=0.2)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_center_error_histogram(predictions_df: pd.DataFrame, path: Path) -> None:
    """Save a histogram of original-space centre error."""
    if predictions_df.empty:
        raise ValueError("predictions_df cannot be empty")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(predictions_df["center_error_px"], bins=40, color="#4C78A8", alpha=0.9)
    ax.set_xlabel("Center Error (px)")
    ax.set_ylabel("Count")
    ax.set_title("Validation Center Error Distribution")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _draw_xyxy(ax, xyxy: np.ndarray, *, color: str, linewidth: float = 1.5, label: str | None = None) -> None:
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linewidth=linewidth, label=label)


def save_heatmap_examples(examples: list[dict[str, Any]], path: Path) -> None:
    """Save a compact montage of input/target/predicted heatmaps."""
    if not examples:
        raise ValueError("examples cannot be empty")
    rows = len(examples)
    fig, axes = plt.subplots(rows, 3, figsize=(10.5, max(3.5, rows * 3.0)))
    axes = np.atleast_2d(axes)
    for row_index, example in enumerate(examples):
        input_ax, target_ax, pred_ax = axes[row_index]
        input_ax.imshow(example["input_image"], cmap="gray", vmin=0.0, vmax=1.0)
        input_ax.set_title(f"Input: {example['sample_id']}")
        input_ax.axis("off")

        target_ax.imshow(example["target_heatmap"], cmap="magma", vmin=0.0, vmax=1.0)
        target_ax.scatter([example["target_output_xy"][0]], [example["target_output_xy"][1]], s=20, c="cyan")
        target_ax.set_title("Target Heatmap")
        target_ax.axis("off")

        pred_ax.imshow(example["predicted_heatmap"], cmap="magma", vmin=0.0, vmax=1.0)
        pred_ax.scatter([example["predicted_output_xy"][0]], [example["predicted_output_xy"][1]], s=20, c="lime")
        pred_ax.set_title(f"Pred Heatmap ({example['confidence']:.3f})")
        pred_ax.axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_center_examples(examples: list[dict[str, Any]], path: Path) -> None:
    """Save locator-canvas overlays showing target/predicted centres and ROI bounds."""
    if not examples:
        raise ValueError("examples cannot be empty")
    rows = len(examples)
    fig, axes = plt.subplots(rows, 1, figsize=(10.5, max(3.0, rows * 3.0)))
    axes = np.atleast_1d(axes)
    for row_index, example in enumerate(examples):
        ax = axes[row_index]
        ax.imshow(example["input_image"], cmap="gray", vmin=0.0, vmax=1.0)
        ax.scatter([example["target_canvas_xy"][0]], [example["target_canvas_xy"][1]], c="cyan", s=18, label="Target")
        ax.scatter([example["predicted_canvas_xy"][0]], [example["predicted_canvas_xy"][1]], c="lime", s=18, label="Predicted")
        if example.get("target_roi_canvas_xyxy") is not None:
            _draw_xyxy(ax, np.asarray(example["target_roi_canvas_xyxy"], dtype=np.float32), color="cyan", label="Target ROI")
        if example.get("predicted_roi_canvas_xyxy") is not None:
            _draw_xyxy(ax, np.asarray(example["predicted_roi_canvas_xyxy"], dtype=np.float32), color="lime", label="Pred ROI")
        ax.set_title(
            f"{example['sample_id']} | target=({example['target_original_xy'][0]:.1f},{example['target_original_xy'][1]:.1f}) "
            f"pred=({example['predicted_original_xy'][0]:.1f},{example['predicted_original_xy'][1]:.1f})"
        )
        ax.axis("off")
        if row_index == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
