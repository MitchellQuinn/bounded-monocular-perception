"""Logic helpers for the opt-in tri-stream pack control panel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import BrightnessNormalizationConfigV4, PackTriStreamStageConfigV4, StageSummaryV4
from .manifest import load_samples_csv, samples_csv_path
from .pack_tri_stream_stage import (
    build_tri_stream_sample_preview,
    infer_tri_stream_silhouette_canvas_size,
    run_pack_tri_stream_stage_v4,
)
from .paths import (
    RunPathsV4,
    find_project_root,
    get_tri_stream_training_root,
    input_run_paths,
    list_input_runs,
    silhouette_run_paths,
    tri_stream_training_run_paths,
)
from .validation import validate_run_structure


@dataclass(frozen=True)
class TriStreamRunStatus:
    """Discovery status for one candidate input run."""

    run_name: str
    input_ready: bool
    silhouette_ready: bool
    output_exists: bool
    output_root: str
    row_count: int | None
    issues: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TriStreamPreviewResult:
    """Structured preview payload returned to the notebook."""

    run_name: str
    row_index: int
    sample_id: str
    image_filename: str
    expected_output_root: str
    geometry_summary: dict[str, Any]
    arrays: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_project_root(start: Path | None = None) -> Path:
    """Resolve the preprocessing project root."""
    return find_project_root(start)


def tri_stream_output_root(project_root: Path) -> Path:
    """Return the dedicated tri-stream training root."""
    return get_tri_stream_training_root(project_root)


def discover_tri_stream_runs(project_root: Path) -> list[TriStreamRunStatus]:
    """List input runs with status needed by the tri-stream control panel."""
    statuses: list[TriStreamRunStatus] = []
    for run_name in list_input_runs(project_root):
        input_paths = input_run_paths(project_root, run_name)
        silhouette_paths = silhouette_run_paths(project_root, run_name)
        output_paths = tri_stream_training_run_paths(project_root, run_name)
        issues: list[str] = []
        input_errors = validate_run_structure(input_paths, require_images=True)
        silhouette_errors = validate_run_structure(silhouette_paths, require_images=True)
        issues.extend(input_errors)
        issues.extend(silhouette_errors)
        row_count: int | None = None
        if not silhouette_errors:
            try:
                row_count = int(len(load_samples_csv(samples_csv_path(silhouette_paths.manifests_dir))))
            except Exception as exc:
                issues.append(f"Could not read silhouette samples.csv: {exc}")
        statuses.append(
            TriStreamRunStatus(
                run_name=run_name,
                input_ready=not input_errors,
                silhouette_ready=not silhouette_errors,
                output_exists=output_paths.root.exists(),
                output_root=str(output_paths.root),
                row_count=row_count,
                issues=tuple(issues),
            )
        )
    return statuses


def require_packable_run(project_root: Path, run_name: str) -> tuple[RunPathsV4, RunPathsV4]:
    """Validate that input and silhouette artifacts are present for one run."""
    input_paths = input_run_paths(project_root, run_name)
    silhouette_paths = silhouette_run_paths(project_root, run_name)
    errors = validate_run_structure(input_paths, require_images=True)
    errors.extend(validate_run_structure(silhouette_paths, require_images=True))
    if errors:
        raise ValueError("\n".join(errors))
    return input_paths, silhouette_paths


def infer_tri_stream_run_canvas_size(project_root: Path, run_name: str) -> tuple[int, int]:
    """Infer the required tri-stream pack canvas from the selected run's silhouette artifacts."""
    _, silhouette_paths = require_packable_run(project_root, run_name)
    samples_df = load_samples_csv(samples_csv_path(silhouette_paths.manifests_dir))
    return infer_tri_stream_silhouette_canvas_size(samples_df)


def build_pack_tri_stream_config(
    *,
    canvas_width_px: int = 300,
    canvas_height_px: int = 300,
    clip_policy: str = "fail",
    orientation_context_scale: float = 1.25,
    shard_size: int = 8192,
    overwrite: bool = False,
    sample_offset: int = 0,
    sample_limit: int = 0,
    brightness_normalization: BrightnessNormalizationConfigV4 | Mapping[str, object] | None = None,
) -> PackTriStreamStageConfigV4:
    """Construct the pack config from notebook control values."""
    return PackTriStreamStageConfigV4(
        canvas_width_px=int(canvas_width_px),
        canvas_height_px=int(canvas_height_px),
        clip_policy=str(clip_policy),
        orientation_context_scale=float(orientation_context_scale),
        shard_size=int(shard_size),
        overwrite=bool(overwrite),
        sample_offset=int(sample_offset),
        sample_limit=int(sample_limit),
        brightness_normalization=brightness_normalization,
    )


def preview_tri_stream_sample(
    project_root: Path,
    run_name: str,
    config: PackTriStreamStageConfigV4,
    *,
    row_index: int,
) -> TriStreamPreviewResult:
    """Build the preview payload for one selected row."""
    require_packable_run(project_root, run_name)
    preview = build_tri_stream_sample_preview(
        project_root,
        run_name,
        config,
        row_index=int(row_index),
    )
    geometry_summary = {
        "roi_request_xyxy_px": preview["roi_request_xyxy_px"].astype(float).tolist(),
        "roi_source_xyxy_px": preview["roi_source_xyxy_px"].astype(float).tolist(),
        "roi_canvas_insert_xyxy_px": preview["roi_canvas_insert_xyxy_px"].astype(float).tolist(),
        "orientation_source_extent_xyxy_px": preview["orientation_source_extent_xyxy_px"].astype(float).tolist(),
        "orientation_crop_source_xyxy_px": preview["orientation_crop_source_xyxy_px"].astype(float).tolist(),
        "orientation_crop_size_px": float(preview["orientation_crop_size_px"]),
        "distance_clipped": bool(preview["distance_clipped"]),
        "x_geometry": preview["x_geometry"].astype(float).tolist(),
    }
    return TriStreamPreviewResult(
        run_name=run_name,
        row_index=int(row_index),
        sample_id=str(preview["sample_id"]),
        image_filename=str(preview["image_filename"]),
        expected_output_root=str(preview["expected_output_root"]),
        geometry_summary=geometry_summary,
        arrays={
            "source_roi_canvas": preview["source_roi_canvas"],
            "x_distance_image": preview["x_distance_image"],
            "x_orientation_image": preview["x_orientation_image"],
        },
    )


def run_tri_stream_pack(
    project_root: Path,
    run_name: str,
    config: PackTriStreamStageConfigV4,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV4:
    """Run the opt-in tri-stream pack stage after validating source availability."""
    require_packable_run(project_root, run_name)
    return run_pack_tri_stream_stage_v4(
        project_root,
        run_name,
        config,
        log_sink=log_sink,
    )
