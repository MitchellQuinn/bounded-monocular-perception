"""Path utilities and canonical directory handling."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from zoneinfo import ZoneInfo

PROJECT_TIMEZONE = ZoneInfo("Europe/London")

DEFAULT_TRAINING_ROOT = Path("training-data")
DEFAULT_VALIDATION_ROOT = Path("validation-data")
DEFAULT_MODELS_ROOT = Path("models")
DEFAULT_NOTEBOOKS_ROOT = Path("notebooks")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root by looking for canonical project folders."""
    origin = (start or Path.cwd()).resolve()
    candidates = [origin, *origin.parents]
    required = ("training-data", "validation-data", "src", "notebooks")
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError(
        "Could not infer repo root. Expected to find src/, notebooks/, "
        "training-data/, and validation-data/ in a parent directory."
    )


def resolve_data_root(repo_root: Path, override: str | Path | None, default_rel: Path) -> Path:
    """Resolve and validate a data root directory."""
    path = (repo_root / default_rel) if override is None else Path(override)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data root does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Data root is not a directory: {path}")
    return path


def resolve_output_root(repo_root: Path, override: str | Path | None = None) -> Path:
    """Resolve output root and create it if missing."""
    path = (repo_root / DEFAULT_MODELS_ROOT) if override is None else Path(override)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_model_run_dir(
    models_root: Path,
    model_name: str = "2d-cnn",
    run_id: str | None = None,
    run_name_suffix: str | None = None,
    now_local: datetime | None = None,
) -> Path:
    """Create a run directory at models/<model_name>/runs/<run_id>."""
    model = str(model_name).strip()
    if not _IDENTIFIER_RE.fullmatch(model):
        raise ValueError(
            f"model_name must match {_IDENTIFIER_RE.pattern!r}; got {model_name!r}"
        )

    if run_id is not None and run_name_suffix:
        raise ValueError("run_name_suffix cannot be used together with run_id.")

    if run_id is None:
        timestamp = (now_local or datetime.now(PROJECT_TIMEZONE)).strftime("%y%m%d-%H%M")
        base_run_id = f"{timestamp}_{model}"
        if run_name_suffix:
            sanitized = "".join(
                ch for ch in run_name_suffix.strip() if ch.isalnum() or ch in ("-", "_")
            )
            if not sanitized:
                raise ValueError("run_name_suffix produced an empty sanitized value.")
            run_name = f"{base_run_id}_{sanitized}"
        else:
            run_name = base_run_id
    else:
        run_name = str(run_id).strip()
        if not _IDENTIFIER_RE.fullmatch(run_name):
            raise ValueError(
                f"run_id must match {_IDENTIFIER_RE.pattern!r}; got {run_id!r}"
            )

    run_dir = models_root / model / "runs" / run_name
    if run_dir.exists():
        if run_id is not None and run_dir.is_dir():
            # Control-panel launches may pre-create an empty run folder (or train.log only)
            # so tmux can start writing logs immediately before train.py begins.
            existing_names = {entry.name for entry in run_dir.iterdir()}
            allowed_precreated = {"train.log"}
            if existing_names.issubset(allowed_precreated):
                return run_dir
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Choose a different run_id or wait for a new timestamp."
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def to_repo_relative(repo_root: Path, path: str | Path) -> str:
    """Convert a path to a repository-relative string when possible."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(repo_root.resolve()))
