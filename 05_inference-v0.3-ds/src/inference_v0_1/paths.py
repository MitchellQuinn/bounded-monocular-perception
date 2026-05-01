"""Path helpers for the v0.3 inference package."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def inference_project_root() -> Path:
    """Return the v0.3 inference project root."""
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Return the repository root that contains the inference project."""
    return inference_project_root().parent


def input_root() -> Path:
    """Return the inference input root."""
    return inference_project_root() / "input"


def models_root() -> Path:
    """Return the inference models root."""
    return inference_project_root() / "models"


def output_root() -> Path:
    """Return the inference output root, creating it when needed."""
    path = inference_project_root() / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_root() -> Path:
    """Return the canonical results output folder."""
    path = output_root() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_repo_relative(path: str | Path) -> str:
    """Return a repository-relative path string when possible."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(repo_root().resolve()))
    except ValueError:
        return str(resolved)


def timestamp_slug(now: datetime | None = None) -> str:
    """Return a UTC timestamp suitable for filenames."""
    value = now or datetime.now(timezone.utc)
    return value.strftime("%Y%m%dT%H%M%SZ")


def sanitize_identifier(value: object) -> str:
    """Normalize arbitrary text into a filesystem-safe identifier."""
    text = _SANITIZE_RE.sub("-", str(value).strip())
    text = text.strip("-._")
    return text or "artifact"
