"""Path helpers for v3 preprocessing artifacts and training outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rb_pipeline.paths import (
    INPUT_ROOT_NAME,
    RUN_JSON_FILENAME,
    SAMPLES_FILENAME,
    ensure_run_dirs,
    find_project_root as _find_project_root_legacy,
    normalize_relative_filename,
    resolve_manifest_path,
    to_posix_path,
)

THRESHOLD_ROOT_NAME = "threshold-images-v3"
TRAINING_V3_ROOT_NAME = "training-data-v3"


@dataclass(frozen=True)
class RunPathsV3:
    """Resolved paths for one run in one v3 stage root."""

    run_name: str
    root: Path
    manifests_dir: Path
    images_dir: Path | None = None
    arrays_dir: Path | None = None



def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root for v3 notebooks and pipeline entrypoints."""

    candidate = (start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if not (current / INPUT_ROOT_NAME).exists():
            continue

        if (current / "rb_pipeline_v3").exists() or (current / "rb_ui_v3").exists():
            return current

        if (current / "rb_pipeline_v2").exists() or (current / "rb_ui_v2").exists():
            return current

        if (current / "rb_pipeline").exists() and ((current / "rb_ui_v1").exists() or (current / "notebooks").exists()):
            return current

    try:
        return _find_project_root_legacy(candidate)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not locate project root containing 'input-images/' and 'rb_pipeline_v3/' or 'rb_ui_v3/'."
        ) from exc



def get_input_root(project_root: Path) -> Path:
    return project_root / INPUT_ROOT_NAME



def get_threshold_root(project_root: Path) -> Path:
    return project_root / THRESHOLD_ROOT_NAME



def get_training_v3_root(project_root: Path) -> Path:
    return project_root / TRAINING_V3_ROOT_NAME



def input_run_paths(project_root: Path, run_name: str) -> RunPathsV3:
    run_root = get_input_root(project_root) / run_name
    return RunPathsV3(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )



def threshold_run_paths(project_root: Path, run_name: str) -> RunPathsV3:
    run_root = get_threshold_root(project_root) / run_name
    return RunPathsV3(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )



def training_v3_run_paths(project_root: Path, run_name: str) -> RunPathsV3:
    run_root = get_training_v3_root(project_root) / run_name
    return RunPathsV3(
        run_name=run_name,
        root=run_root,
        arrays_dir=run_root / "arrays",
        manifests_dir=run_root / "manifests",
    )



def ensure_run_dirs_v3(run_paths: RunPathsV3, dry_run: bool = False) -> None:
    """Create run output directories unless dry-run mode is enabled."""

    ensure_run_dirs(
        run_paths,
        dry_run=dry_run,
    )



def list_training_v3_runs(project_root: Path) -> list[str]:
    """List available packed training runs under training-data-v3/."""

    root = get_training_v3_root(project_root)
    if not root.exists():
        return []

    runs: list[str] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        manifests_dir = run_dir / "manifests"
        if not (
            (manifests_dir / RUN_JSON_FILENAME).is_file()
            and (manifests_dir / SAMPLES_FILENAME).is_file()
        ):
            continue

        if not any(run_dir.glob("*.npz")):
            continue

        runs.append(run_dir.name)

    return runs
