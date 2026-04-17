"""Dataset discovery helpers for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

from pathlib import Path

from .contracts import DatasetReference
from .paths import dataset_output_root, find_preprocessing_root, input_root
from .validation import validate_input_dataset_reference


def discover_dataset_references(preprocessing_root: Path | None = None) -> list[DatasetReference]:
    """List only dataset references that contain valid train and validate splits."""

    root = find_preprocessing_root(preprocessing_root)
    discovery_root = input_root(root)
    if not discovery_root.exists():
        return []

    discovered: list[DatasetReference] = []
    for candidate in sorted(discovery_root.iterdir()):
        if not candidate.is_dir():
            continue
        errors = validate_input_dataset_reference(root, candidate.name)
        if errors:
            continue
        discovered.append(
            DatasetReference(
                name=candidate.name,
                input_root=candidate,
                output_root=dataset_output_root(root, candidate.name),
            )
        )
    return discovered
