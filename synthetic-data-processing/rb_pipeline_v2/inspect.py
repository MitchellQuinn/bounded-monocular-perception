"""Small inspection helpers for source/edge/silhouette previews in v2."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rb_pipeline.image_io import read_grayscale_uint8

from .algorithms import register_default_components
from .manifest import load_samples_csv, samples_csv_path
from .paths import input_run_paths, resolve_manifest_path, silhouette_run_paths
from .registry import get_artifact_writer_by_mode, get_fallback_strategy, get_representation_generator



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


def save_contour_comparison_debug_batch(
    project_root: Path,
    run_name: str,
    *,
    output_dir: Path | None = None,
    sample_limit: int = 8,
    sample_offset: int = 0,
    blur_k: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    close_k: int = 1,
    dilate_k: int = 1,
    min_area: int = 50,
    outline_px: int = 1,
    fill_holes: bool = True,
    use_convex_hull_fallback: bool = True,
    representation_mode: str = "filled",
) -> Path:
    """Save side-by-side v1/v2 silhouette comparison grids for a small input subset."""

    register_default_components()
    generator_v1 = get_representation_generator("silhouette.contour_v1")
    generator_v2 = get_representation_generator("silhouette.contour_v2")
    fallback = get_fallback_strategy("fallback.convex_hull_v1")
    writer = get_artifact_writer_by_mode(representation_mode)

    input_paths = input_run_paths(project_root, run_name)
    samples_df = load_samples_csv(samples_csv_path(input_paths.manifests_dir))
    selected_rows = _selected_input_rows(samples_df, sample_offset=sample_offset, sample_limit=sample_limit)

    if output_dir is None:
        output_dir = project_root / "silhouette-images-v2" / run_name / "debug-contour-v1-v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    written_count = 0

    for row_idx in selected_rows:
        source_filename = str(samples_df.at[row_idx, "image_filename"])
        sample_id = (
            str(samples_df.at[row_idx, "sample_id"])
            if "sample_id" in samples_df.columns
            else f"row_{row_idx}"
        )
        if "capture_success" in samples_df.columns:
            capture_value = samples_df.at[row_idx, "capture_success"]
            if pd.isna(capture_value):
                capture_success = False
            elif isinstance(capture_value, str):
                capture_success = capture_value.strip().lower() in {"1", "true", "yes", "y"}
            else:
                capture_success = bool(capture_value)
        else:
            capture_success = True
        if not capture_success:
            continue

        source_path = resolve_manifest_path(input_paths.root, "images", source_filename)
        source_gray = _safe_gray(source_path)
        if source_gray is None:
            continue

        v1_output = generator_v1.generate(
            source_gray,
            blur_kernel_size=max(1, int(blur_k) | 1),
            canny_low_threshold=int(canny_low),
            canny_high_threshold=int(canny_high),
            close_kernel_size=max(1, int(close_k)),
            dilate_kernel_size=max(1, int(dilate_k)),
            min_component_area_px=max(1, int(min_area)),
            fill_holes=bool(fill_holes),
        )
        v2_output = generator_v2.generate(
            source_gray,
            blur_kernel_size=max(1, int(blur_k) | 1),
            canny_low_threshold=int(canny_low),
            canny_high_threshold=int(canny_high),
            close_kernel_size=max(1, int(close_k)),
            dilate_kernel_size=max(1, int(dilate_k)),
            min_component_area_px=max(1, int(min_area)),
            fill_holes=bool(fill_holes),
        )

        edge_debug = _resolve_debug_edge(v2_output)
        v1_render, v1_status, _ = _render_with_optional_fallback(
            source_gray,
            v1_output.contour,
            fallback_mask=v1_output.fallback_mask,
            writer=writer,
            outline_px=outline_px,
            fallback=fallback,
            use_convex_hull_fallback=use_convex_hull_fallback,
            primary_reason=v1_output.primary_reason,
        )
        v2_render, v2_status, v2_fallback_render = _render_with_optional_fallback(
            source_gray,
            v2_output.contour,
            fallback_mask=v2_output.fallback_mask,
            writer=writer,
            outline_px=outline_px,
            fallback=fallback,
            use_convex_hull_fallback=use_convex_hull_fallback,
            primary_reason=v2_output.primary_reason,
        )

        image_stem = f"{written_count:03d}_row_{row_idx:05d}_{sample_id}"
        image_stem = image_stem.replace("/", "_")
        grid_path = output_dir / f"{image_stem}.png"

        _save_comparison_grid(
            grid_path,
            edge_map=edge_debug,
            contour_v1=v1_render,
            contour_v2=v2_render,
            fallback_hull=v2_fallback_render,
            title_suffix=f"row={row_idx}, sample={sample_id}",
        )

        records.append(
            {
                "row_index": int(row_idx),
                "sample_id": sample_id,
                "image_filename": source_filename,
                "comparison_grid_filename": grid_path.name,
                "contour_v1_status": v1_status,
                "contour_v2_status": v2_status,
                "contour_v2_quality_flags": ";".join(v2_output.quality_flags),
                "contour_v2_primary_reason": str(v2_output.primary_reason),
                "contour_v2_used_fallback": str(v2_fallback_render is not None).lower(),
            }
        )
        written_count += 1

    pd.DataFrame(records).to_csv(output_dir / "comparison_manifest.csv", index=False)
    return output_dir


def _resolve_debug_edge(generator_output) -> np.ndarray:
    debug_images = getattr(generator_output, "debug_images", None)
    if isinstance(debug_images, dict):
        raw = debug_images.get("raw_edge")
        if isinstance(raw, np.ndarray) and raw.ndim == 2:
            return raw

    edge_binary = generator_output.edge_binary
    edge_debug = np.full(edge_binary.shape, 255, dtype=np.uint8)
    edge_debug[edge_binary > 0] = 0
    return edge_debug


def _render_with_optional_fallback(
    source_gray: np.ndarray,
    contour: np.ndarray | None,
    *,
    fallback_mask: np.ndarray,
    writer,
    outline_px: int,
    fallback,
    use_convex_hull_fallback: bool,
    primary_reason: str,
) -> tuple[np.ndarray | None, str, np.ndarray | None]:
    current_contour = contour
    status = "primary_success"
    fallback_render = None

    if _is_contour_broken(current_contour):
        if not use_convex_hull_fallback:
            reason = primary_reason or "primary_contour_failed"
            return None, f"failed_no_fallback:{reason}", None

        current_contour, recovery_reason = fallback.recover(fallback_mask)
        if current_contour is None:
            reason = primary_reason or recovery_reason or "fallback_failed"
            return None, f"fallback_failed:{reason}", None

        status = "fallback_success"
        fallback_render = writer.render(
            source_gray.shape,
            current_contour,
            line_thickness=max(1, int(outline_px)),
        )

    rendered = writer.render(
        source_gray.shape,
        current_contour,
        line_thickness=max(1, int(outline_px)),
    )
    return rendered, status, fallback_render


def _is_contour_broken(contour: np.ndarray | None) -> bool:
    if contour is None:
        return True
    if contour.ndim != 3 or contour.shape[0] < 3:
        return True
    area = float(abs(cv2.contourArea(contour)))
    return area < 1.0


def _selected_input_rows(samples_df: pd.DataFrame, *, sample_offset: int, sample_limit: int) -> list[int]:
    rows = list(samples_df.index)
    if not rows:
        return []

    start = max(0, int(sample_offset))
    if start >= len(rows):
        return []

    if int(sample_limit) <= 0:
        return [int(row) for row in rows[start:]]

    end = start + int(sample_limit)
    return [int(row) for row in rows[start:end]]


def _save_comparison_grid(
    output_path: Path,
    *,
    edge_map: np.ndarray | None,
    contour_v1: np.ndarray | None,
    contour_v2: np.ndarray | None,
    fallback_hull: np.ndarray | None,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        (axes[0, 0], edge_map, "Original Edge Map"),
        (axes[0, 1], contour_v1, "contour_v1"),
        (axes[1, 0], contour_v2, "contour_v2"),
        (axes[1, 1], fallback_hull, "Fallback Hull (if used)"),
    ]

    for axis, image, title in panels:
        axis.set_title(title)
        axis.axis("off")
        if image is None:
            axis.text(0.5, 0.5, "Not used / failed", ha="center", va="center")
            continue
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)

    fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
