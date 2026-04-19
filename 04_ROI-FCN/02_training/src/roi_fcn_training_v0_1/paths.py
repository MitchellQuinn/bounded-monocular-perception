"""Path helpers for ROI-FCN training v0.1."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from .contracts import DATASETS_ROOT_NAME, MODELS_ROOT_NAME, NOTEBOOKS_ROOT_NAME, SplitPaths

PROJECT_TIMEZONE = ZoneInfo("Europe/London")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def module_training_root() -> Path:
    """Return the repository copy of 02_training."""
    return Path(__file__).resolve().parents[2]


def find_training_root(start: Path | None = None) -> Path:
    """Locate 04_ROI-FCN/02_training from a nearby path."""
    candidate = (start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if all((current / name).exists() for name in (DATASETS_ROOT_NAME, MODELS_ROOT_NAME, "src", NOTEBOOKS_ROOT_NAME)):
            return current
        nested = current / "04_ROI-FCN" / "02_training"
        if all((nested / name).exists() for name in (DATASETS_ROOT_NAME, MODELS_ROOT_NAME, "src", NOTEBOOKS_ROOT_NAME)):
            return nested

    fallback = module_training_root()
    if all((fallback / name).exists() for name in (DATASETS_ROOT_NAME, MODELS_ROOT_NAME, "src", NOTEBOOKS_ROOT_NAME)):
        return fallback
    raise FileNotFoundError("Could not locate 04_ROI-FCN/02_training root.")


def datasets_root(training_root: Path | None = None) -> Path:
    return find_training_root(training_root) / DATASETS_ROOT_NAME


def models_root(training_root: Path | None = None) -> Path:
    root = find_training_root(training_root) / MODELS_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def resolve_datasets_root(training_root: Path, override: str | Path | None) -> Path:
    """Resolve a datasets root, relative to the training root when needed."""
    path = datasets_root(training_root) if override is None else Path(override)
    if not path.is_absolute():
        path = (training_root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Datasets root does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Datasets root is not a directory: {path}")
    return path


def resolve_models_root(training_root: Path, override: str | Path | None) -> Path:
    """Resolve a models root, relative to the training root when needed."""
    path = models_root(training_root) if override is None else Path(override)
    if not path.is_absolute():
        path = (training_root / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_split_paths(
    training_root: Path | None,
    dataset_reference: str,
    split_name: str,
    *,
    datasets_root_override: str | Path | None = None,
) -> SplitPaths:
    root = find_training_root(training_root)
    datasets = resolve_datasets_root(root, datasets_root_override)
    dataset_name = str(dataset_reference).strip()
    split = str(split_name).strip().lower()
    split_root = datasets / dataset_name / split
    return SplitPaths(
        dataset_reference=dataset_name,
        split_name=split,
        split_root=split_root,
        arrays_dir=split_root / "arrays",
        manifests_dir=split_root / "manifests",
    )


def make_model_run_dir(
    models_root_path: Path,
    *,
    model_name: str,
    run_id: str | None = None,
    run_name_suffix: str | None = None,
    now_local: datetime | None = None,
) -> Path:
    """Create a new run directory under models/<model_name>/runs/."""
    model_text = str(model_name).strip()
    if not _IDENTIFIER_RE.fullmatch(model_text):
        raise ValueError(f"model_name must match {_IDENTIFIER_RE.pattern!r}; got {model_name!r}")

    if run_id is not None and run_name_suffix:
        raise ValueError("run_name_suffix cannot be used together with run_id.")

    if run_id is None:
        timestamp = (now_local or datetime.now(PROJECT_TIMEZONE)).strftime("%y%m%d-%H%M")
        run_name = f"{timestamp}_{model_text}"
        if run_name_suffix:
            suffix = "".join(ch for ch in str(run_name_suffix).strip() if ch.isalnum() or ch in ("-", "_"))
            if not suffix:
                raise ValueError("run_name_suffix produced an empty sanitized value.")
            run_name = f"{run_name}_{suffix}"
    else:
        run_name = str(run_id).strip()
        if not _IDENTIFIER_RE.fullmatch(run_name):
            raise ValueError(f"run_id must match {_IDENTIFIER_RE.pattern!r}; got {run_id!r}")

    run_dir = models_root_path / model_text / "runs" / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def to_repo_relative(repo_root: Path, path: str | Path) -> str:
    """Convert a path to a repository-relative string when possible."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(repo_root.resolve()))
