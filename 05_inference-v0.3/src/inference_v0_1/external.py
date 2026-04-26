"""Bootstrap helpers for reusing the existing preprocessing and training packages."""

from __future__ import annotations

from pathlib import Path
import sys

from .paths import repo_root


def preprocessing_root() -> Path:
    """Return the sibling synthetic preprocessing project root."""
    return repo_root() / "02_synthetic-data-processing-v4.0"


def training_root() -> Path:
    """Return the sibling training project root."""
    return repo_root() / "03_rb-training-v2.0"


def roi_fcn_training_src_root() -> Path:
    """Return the ROI-FCN training src root."""
    return repo_root() / "04_ROI-FCN" / "02_training" / "src"


def ensure_external_paths() -> None:
    """Add sibling project roots to `sys.path` so we can reuse their modules."""
    for path in (training_root(), preprocessing_root(), roi_fcn_training_src_root()):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
