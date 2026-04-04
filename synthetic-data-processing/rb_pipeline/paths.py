"""Path helpers for the Raccoon Ball pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

INPUT_ROOT_NAME = "input-images"
EDGE_ROOT_NAME = "edge-images"
BBOX_ROOT_NAME = "boundbox-images"
TRAINING_ROOT_NAME = "training-data"
TRAINING_SHUFFLED_ROOT_NAME = "training-data-shuffled"
NOTEBOOKS_ROOT_NAME = "notebooks"

RUN_JSON_FILENAME = "run.json"
SAMPLES_FILENAME = "samples.csv"

_KNOWN_STAGE_SUBDIRS = {"images", "arrays", "manifests"}


@dataclass(frozen=True)
class RunPaths:
    """Resolved paths for one run in one stage root."""

    run_name: str
    root: Path
    manifests_dir: Path
    images_dir: Path | None = None
    arrays_dir: Path | None = None



def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking up from `start`."""

    candidate = (start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if (current / INPUT_ROOT_NAME).exists() and (current / NOTEBOOKS_ROOT_NAME).exists():
            return current

    raise FileNotFoundError(
        "Could not locate project root containing 'input-images/' and 'notebooks/'."
    )



def get_input_root(project_root: Path) -> Path:
    return project_root / INPUT_ROOT_NAME



def get_edge_root(project_root: Path) -> Path:
    return project_root / EDGE_ROOT_NAME



def get_bbox_root(project_root: Path) -> Path:
    return project_root / BBOX_ROOT_NAME



def get_training_root(project_root: Path) -> Path:
    return project_root / TRAINING_ROOT_NAME



def get_training_shuffled_root(project_root: Path) -> Path:
    return project_root / TRAINING_SHUFFLED_ROOT_NAME



def list_input_runs(project_root: Path) -> list[str]:
    """List available run directories under input-images/."""

    root = get_input_root(project_root)
    if not root.exists():
        return []

    runs: list[str] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        if (
            (run_dir / "images").is_dir()
            and (run_dir / "manifests" / RUN_JSON_FILENAME).is_file()
            and (run_dir / "manifests" / SAMPLES_FILENAME).is_file()
        ):
            runs.append(run_dir.name)
    return runs



def list_training_runs(project_root: Path, *, root_name: str = TRAINING_ROOT_NAME) -> list[str]:
    """List available packed training runs under a training root directory."""

    root = project_root / root_name
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



def input_run_paths(project_root: Path, run_name: str) -> RunPaths:
    run_root = get_input_root(project_root) / run_name
    return RunPaths(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )



def edge_run_paths(project_root: Path, run_name: str) -> RunPaths:
    run_root = get_edge_root(project_root) / run_name
    return RunPaths(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )



def bbox_run_paths(project_root: Path, run_name: str) -> RunPaths:
    run_root = get_bbox_root(project_root) / run_name
    return RunPaths(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )



def training_run_paths(project_root: Path, run_name: str) -> RunPaths:
    run_root = get_training_root(project_root) / run_name
    return RunPaths(
        run_name=run_name,
        root=run_root,
        arrays_dir=run_root / "arrays",
        manifests_dir=run_root / "manifests",
    )



def training_shuffled_run_paths(project_root: Path, run_name: str) -> RunPaths:
    run_root = get_training_shuffled_root(project_root) / run_name
    return RunPaths(
        run_name=run_name,
        root=run_root,
        arrays_dir=run_root / "arrays",
        manifests_dir=run_root / "manifests",
    )



def ensure_run_dirs(run_paths: RunPaths, dry_run: bool = False) -> None:
    """Create run output directories unless dry-run mode is enabled."""

    if dry_run:
        return

    run_paths.root.mkdir(parents=True, exist_ok=True)
    run_paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    if run_paths.images_dir is not None:
        run_paths.images_dir.mkdir(parents=True, exist_ok=True)
    if run_paths.arrays_dir is not None:
        run_paths.arrays_dir.mkdir(parents=True, exist_ok=True)



def resolve_manifest_path(run_root: Path, default_subdir: str, manifest_value: object) -> Path:
    """Resolve a manifest filename/path value to an absolute file path."""

    if manifest_value is None:
        raise ValueError("Manifest filename is missing.")

    text = str(manifest_value).strip()
    if not text:
        raise ValueError("Manifest filename is blank.")

    rel = Path(text)
    if rel.is_absolute():
        return rel

    if rel.parts and rel.parts[0].lower() in _KNOWN_STAGE_SUBDIRS:
        return run_root / rel

    return run_root / default_subdir / rel



def normalize_relative_filename(manifest_value: object, *, new_suffix: str | None = None) -> Path:
    """Normalize a manifest filename to a relative path suitable for output columns."""

    if manifest_value is None:
        raise ValueError("Manifest filename is missing.")

    text = str(manifest_value).strip()
    if not text:
        raise ValueError("Manifest filename is blank.")

    rel = Path(text)
    if rel.is_absolute():
        raise ValueError("Manifest filename must be relative.")

    if rel.parts and rel.parts[0].lower() in _KNOWN_STAGE_SUBDIRS and len(rel.parts) > 1:
        rel = Path(*rel.parts[1:])

    if new_suffix is not None:
        rel = rel.with_suffix(new_suffix)

    return rel



def to_posix_path(path: Path) -> str:
    return path.as_posix()
