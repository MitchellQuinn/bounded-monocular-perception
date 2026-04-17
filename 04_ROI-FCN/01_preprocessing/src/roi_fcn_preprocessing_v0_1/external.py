"""Bootstrap helpers for reusing sibling preprocessing code."""

from __future__ import annotations

from pathlib import Path
import sys


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def synthetic_preprocessing_root() -> Path:
    return repo_root() / "02_synthetic-data-processing-v3.0"


def ensure_external_paths() -> None:
    external_root = str(synthetic_preprocessing_root().resolve())
    if external_root not in sys.path:
        sys.path.insert(0, external_root)
