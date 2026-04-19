"""Dataset discovery helpers for ROI-FCN training v0.1."""

from __future__ import annotations

from pathlib import Path

from .contracts import (
    ARRAYS_DIR_NAME,
    DatasetReference,
    MANIFESTS_DIR_NAME,
    RUN_JSON_FILENAME,
    SAMPLES_FILENAME,
    SPLIT_ORDER,
)
from .paths import datasets_root, find_training_root


def _looks_like_valid_dataset_reference(dataset_root: Path) -> bool:
    for split_name in SPLIT_ORDER:
        split_root = dataset_root / split_name
        if not split_root.is_dir():
            return False
        if not (split_root / ARRAYS_DIR_NAME).is_dir():
            return False
        manifests_dir = split_root / MANIFESTS_DIR_NAME
        if not manifests_dir.is_dir():
            return False
        if not (manifests_dir / RUN_JSON_FILENAME).is_file():
            return False
        if not (manifests_dir / SAMPLES_FILENAME).is_file():
            return False
        if not any((split_root / ARRAYS_DIR_NAME).glob("*.npz")):
            return False
    return True


def discover_dataset_references(training_root: Path | None = None) -> list[DatasetReference]:
    """List selectable dataset references under 02_training/datasets."""
    root = find_training_root(training_root)
    data_root = datasets_root(root)
    discovered: list[DatasetReference] = []
    for child in sorted(path for path in data_root.iterdir() if path.is_dir()):
        if _looks_like_valid_dataset_reference(child):
            discovered.append(DatasetReference(name=child.name, datasets_root=data_root))
    return discovered
