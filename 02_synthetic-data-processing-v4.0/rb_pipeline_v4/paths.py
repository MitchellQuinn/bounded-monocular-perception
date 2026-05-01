"""Path helpers for v4 dual-stream preprocessing artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .constants import (
    DETECT_ROOT_NAME,
    INPUT_ROOT_NAME,
    KNOWN_STAGE_SUBDIRS,
    RUN_JSON_FILENAME,
    SAMPLES_FILENAME,
    SILHOUETTE_ROOT_NAME,
    TRAINING_ROOT_NAME,
    TRI_STREAM_TRAINING_ROOT_NAME,
)


@dataclass(frozen=True)
class RunPathsV4:
    """Resolved paths for one run in one stage root."""

    run_name: str
    root: Path
    manifests_dir: Path
    images_dir: Path | None = None
    arrays_dir: Path | None = None


def find_project_root(start: Path | None = None) -> Path:
    candidate = (start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if not (current / INPUT_ROOT_NAME).exists():
            continue
        if (current / "rb_pipeline_v4").exists() or (current / "rb_ui_v4").exists():
            return current

    raise FileNotFoundError(
        "Could not locate project root containing 'input-images/' and 'rb_pipeline_v4/' or 'rb_ui_v4/'."
    )


def get_input_root(project_root: Path) -> Path:
    return project_root / INPUT_ROOT_NAME


def get_detect_root(project_root: Path) -> Path:
    return project_root / DETECT_ROOT_NAME


def get_silhouette_root(project_root: Path) -> Path:
    return project_root / SILHOUETTE_ROOT_NAME


def get_training_root(project_root: Path) -> Path:
    return project_root / TRAINING_ROOT_NAME


def get_tri_stream_training_root(project_root: Path) -> Path:
    return project_root / TRI_STREAM_TRAINING_ROOT_NAME


def list_input_runs(project_root: Path) -> list[str]:
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


def input_run_paths(project_root: Path, run_name: str) -> RunPathsV4:
    run_root = get_input_root(project_root) / run_name
    return RunPathsV4(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )


def detect_run_paths(project_root: Path, run_name: str) -> RunPathsV4:
    run_root = get_detect_root(project_root) / run_name
    return RunPathsV4(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )


def silhouette_run_paths(project_root: Path, run_name: str) -> RunPathsV4:
    run_root = get_silhouette_root(project_root) / run_name
    return RunPathsV4(
        run_name=run_name,
        root=run_root,
        images_dir=run_root / "images",
        manifests_dir=run_root / "manifests",
    )


def training_run_paths(project_root: Path, run_name: str) -> RunPathsV4:
    run_root = get_training_root(project_root) / run_name
    return RunPathsV4(
        run_name=run_name,
        root=run_root,
        manifests_dir=run_root / "manifests",
        arrays_dir=run_root / "arrays",
    )


def tri_stream_training_run_paths(project_root: Path, run_name: str) -> RunPathsV4:
    run_root = get_tri_stream_training_root(project_root) / run_name
    return RunPathsV4(
        run_name=run_name,
        root=run_root,
        manifests_dir=run_root / "manifests",
        arrays_dir=run_root / "arrays",
    )


def ensure_run_dirs(run_paths: RunPathsV4, dry_run: bool = False) -> None:
    if dry_run:
        return

    run_paths.root.mkdir(parents=True, exist_ok=True)
    run_paths.manifests_dir.mkdir(parents=True, exist_ok=True)
    if run_paths.images_dir is not None:
        run_paths.images_dir.mkdir(parents=True, exist_ok=True)
    if run_paths.arrays_dir is not None:
        run_paths.arrays_dir.mkdir(parents=True, exist_ok=True)


def resolve_manifest_path(run_root: Path, default_subdir: str, manifest_value: object) -> Path:
    if manifest_value is None:
        raise ValueError("Manifest filename is missing.")

    text = str(manifest_value).strip()
    if not text:
        raise ValueError("Manifest filename is blank.")

    rel = Path(text)
    if rel.is_absolute():
        return rel

    if rel.parts and rel.parts[0].lower() in KNOWN_STAGE_SUBDIRS:
        return run_root / rel

    return run_root / default_subdir / rel


def normalize_relative_filename(manifest_value: object, *, new_suffix: str | None = None) -> Path:
    if manifest_value is None:
        raise ValueError("Manifest filename is missing.")

    text = str(manifest_value).strip()
    if not text:
        raise ValueError("Manifest filename is blank.")

    rel = Path(text)
    if rel.is_absolute():
        raise ValueError("Manifest filename must be relative.")

    if rel.parts and rel.parts[0].lower() in KNOWN_STAGE_SUBDIRS and len(rel.parts) > 1:
        rel = Path(*rel.parts[1:])

    if new_suffix is not None:
        rel = rel.with_suffix(new_suffix)
    return rel


def to_posix_path(path: Path) -> str:
    return path.as_posix()
