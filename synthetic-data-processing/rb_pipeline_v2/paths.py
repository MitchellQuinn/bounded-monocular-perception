"""Path helpers for v2 preprocessing artifacts and training outputs."""

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

SILHOUETTE_ROOT_NAME = "silhouette-images-v2"
TRAINING_V2_ROOT_NAME = "training-data-v2"


@dataclass(frozen=True)
class RunPathsV2:
    """Resolved paths for one run in one v2 stage root."""

    run_name: str
    root: Path
    manifests_dir: Path
    images_dir: Path | None = None
    arrays_dir: Path | None = None



def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root for v2 notebooks and pipeline entrypoints."""

    candidate = (start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if not (current / INPUT_ROOT_NAME).exists():
            continue

        if (current / "rb_pipeline_v2").exists() or (current / "rb_ui_v2").exists():
            return current

        if (current / "rb_pipeline").exists() and ((current / "rb_ui_v1").exists() or (current / "notebooks").exists()):
            return current

    try:
        return _find_project_root_legacy(candidate)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not locate project root containing 'input-images/' and 'rb_pipeline_v2/' or 'rb_ui_v2/'."
        ) from exc



def get_input_root(project_root: Path) -> Path:
    return project_root / INPUT_ROOT_NAME



def get_silhouette_root(project_root: Path) -> Path:
    return project_root / SILHOUETTE_ROOT_NAME



def get_training_v2_root(project_root: Path) -> Path:
    return project_root / TRAINING_V2_ROOT_NAME



def input_run_paths(project_root: Path, run_name: str) -> RunPathsV2:
    run_root = get_input_root(project_root) / run_name
    return RunPathsV2(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )



def silhouette_run_paths(project_root: Path, run_name: str) -> RunPathsV2:
    run_root = get_silhouette_root(project_root) / run_name
    return RunPathsV2(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )



def training_v2_run_paths(project_root: Path, run_name: str) -> RunPathsV2:
    run_root = get_training_v2_root(project_root) / run_name
    return RunPathsV2(
        run_name=run_name,
        root=run_root,
        arrays_dir=run_root / "arrays",
        manifests_dir=run_root / "manifests",
    )



def ensure_run_dirs_v2(run_paths: RunPathsV2, dry_run: bool = False) -> None:
    """Create run output directories unless dry-run mode is enabled."""

    ensure_run_dirs(
        run_paths,
        dry_run=dry_run,
    )



def list_training_v2_runs(project_root: Path) -> list[str]:
    """List available packed training runs under training-data-v2/."""

    root = get_training_v2_root(project_root)
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
