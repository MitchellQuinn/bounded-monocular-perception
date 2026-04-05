"""Small inspection helpers for source/edge/silhouette previews in v2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rb_pipeline.image_io import read_grayscale_uint8

from .manifest import load_samples_csv, samples_csv_path
from .paths import input_run_paths, resolve_manifest_path, silhouette_run_paths



def load_source_edge_silhouette_preview(
    project_root: Path,
    run_name: str,
    row_idx: int,
) -> dict[str, np.ndarray | None]:
    """Load source, edge-debug, and silhouette images for one row if available."""

    source_paths = input_run_paths(project_root, run_name)
    silhouette_paths = silhouette_run_paths(project_root, run_name)

    source_df = load_samples_csv(samples_csv_path(source_paths.manifests_dir))
    if row_idx not in source_df.index:
        raise ValueError(f"Row index {row_idx} not found in source samples.")

    silhouette_df = None
    silhouette_samples_path = samples_csv_path(silhouette_paths.manifests_dir)
    if silhouette_samples_path.exists():
        silhouette_df = load_samples_csv(silhouette_samples_path)

    source_filename = source_df.at[row_idx, "image_filename"]
    source_path = resolve_manifest_path(source_paths.root, "images", source_filename)

    source_gray = _safe_gray(source_path)

    edge_gray: np.ndarray | None = None
    silhouette_gray: np.ndarray | None = None

    if silhouette_df is not None and row_idx in silhouette_df.index:
        edge_filename = str(silhouette_df.at[row_idx, "silhouette_edge_debug_filename"]).strip()
        silhouette_filename = str(silhouette_df.at[row_idx, "silhouette_image_filename"]).strip()

        if edge_filename:
            edge_path = resolve_manifest_path(silhouette_paths.root, "images", edge_filename)
            edge_gray = _safe_gray(edge_path)

        if silhouette_filename:
            silhouette_path = resolve_manifest_path(silhouette_paths.root, "images", silhouette_filename)
            silhouette_gray = _safe_gray(silhouette_path)

    return {
        "source": source_gray,
        "edge_debug": edge_gray,
        "silhouette": silhouette_gray,
    }



def show_source_edge_silhouette_preview(
    project_root: Path,
    run_name: str,
    row_idx: int,
) -> dict[str, np.ndarray | None]:
    """Render a quick 3-panel preview and return loaded image arrays."""

    loaded = load_source_edge_silhouette_preview(project_root, run_name, row_idx)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("Source", loaded["source"]),
        ("Edge Debug", loaded["edge_debug"]),
        ("Silhouette", loaded["silhouette"]),
    ]

    for axis, (title, image) in zip(axes, panels):
        axis.set_title(title)
        axis.axis("off")
        if image is None:
            axis.text(0.5, 0.5, "N/A", ha="center", va="center")
        else:
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)

    fig.tight_layout()
    return loaded



def _safe_gray(path: Path) -> np.ndarray | None:
    try:
        if not path.is_file():
            return None
        return read_grayscale_uint8(path)
    except Exception:
        return None
