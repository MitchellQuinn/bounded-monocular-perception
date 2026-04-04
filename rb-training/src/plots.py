"""Plot helpers for training/evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_history_plot(history_df: pd.DataFrame, output_path: Path) -> None:
    """Save train/validation loss curves."""
    if history_df.empty:
        raise ValueError("Cannot plot empty training history.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    ax.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Save prediction-vs-truth scatter plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y_true.size == 0:
        raise ValueError("Cannot create scatter plot with zero samples.")

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=6, alpha=0.35)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black")
    ax.set_title("Prediction vs Truth")
    ax.set_xlabel("True distance_m")
    ax.set_ylabel("Predicted distance_m")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    """Save residual plot (residuals vs true distance)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    if y_true.size == 0:
        raise ValueError("Cannot create residual plot with zero samples.")

    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_true, residuals, s=6, alpha=0.35)
    ax.axhline(0.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_title("Residuals vs Truth")
    ax.set_xlabel("True distance_m")
    ax.set_ylabel("Residual (pred - true)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
