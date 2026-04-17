"""Path helpers for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contracts import (
    INPUT_ROOT_NAME,
    OUTPUT_ROOT_NAME,
    RUN_JSON_FILENAME,
    SAMPLES_FILENAME,
)


@dataclass(frozen=True)
class SplitPaths:
    """Resolved input and output paths for one dataset split."""

    dataset_reference: str
    split_name: str
    input_root: Path
    input_images_dir: Path
    input_manifests_dir: Path
    output_root: Path
    output_arrays_dir: Path
    output_manifests_dir: Path

    @property
    def input_run_json_path(self) -> Path:
        return self.input_manifests_dir / RUN_JSON_FILENAME

    @property
    def input_samples_csv_path(self) -> Path:
        return self.input_manifests_dir / SAMPLES_FILENAME

    @property
    def output_run_json_path(self) -> Path:
        return self.output_manifests_dir / RUN_JSON_FILENAME

    @property
    def output_samples_csv_path(self) -> Path:
        return self.output_manifests_dir / SAMPLES_FILENAME


def module_preprocessing_root() -> Path:
    """Return the repository copy of 01_preprocessing."""

    return Path(__file__).resolve().parents[2]


def find_preprocessing_root(start: Path | None = None) -> Path:
    """Locate the ROI-FCN preprocessing root from a nearby path."""

    candidate = (start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if (current / INPUT_ROOT_NAME).is_dir() and (current / OUTPUT_ROOT_NAME).is_dir():
            return current

        nested = current / "04_ROI-FCN" / "01_preprocessing"
        if (nested / INPUT_ROOT_NAME).is_dir() and (nested / OUTPUT_ROOT_NAME).is_dir():
            return nested

    fallback = module_preprocessing_root()
    if (fallback / INPUT_ROOT_NAME).is_dir() and (fallback / OUTPUT_ROOT_NAME).is_dir():
        return fallback

    raise FileNotFoundError("Could not locate 04_ROI-FCN/01_preprocessing root.")


def input_root(preprocessing_root: Path | None = None) -> Path:
    return find_preprocessing_root(preprocessing_root) / INPUT_ROOT_NAME


def output_root(preprocessing_root: Path | None = None) -> Path:
    return find_preprocessing_root(preprocessing_root) / OUTPUT_ROOT_NAME


def dataset_input_root(preprocessing_root: Path | None, dataset_reference: str) -> Path:
    return input_root(preprocessing_root) / str(dataset_reference).strip()


def dataset_output_root(preprocessing_root: Path | None, dataset_reference: str) -> Path:
    return output_root(preprocessing_root) / str(dataset_reference).strip()


def resolve_split_paths(
    preprocessing_root: Path | None,
    dataset_reference: str,
    split_name: str,
) -> SplitPaths:
    root = find_preprocessing_root(preprocessing_root)
    dataset_name = str(dataset_reference).strip()
    split = str(split_name).strip().lower()
    input_split_root = root / INPUT_ROOT_NAME / dataset_name / split
    output_split_root = root / OUTPUT_ROOT_NAME / dataset_name / split
    return SplitPaths(
        dataset_reference=dataset_name,
        split_name=split,
        input_root=input_split_root,
        input_images_dir=input_split_root / "images",
        input_manifests_dir=input_split_root / "manifests",
        output_root=output_split_root,
        output_arrays_dir=output_split_root / "arrays",
        output_manifests_dir=output_split_root / "manifests",
    )


def ensure_split_output_dirs(split_paths: SplitPaths, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    split_paths.output_root.mkdir(parents=True, exist_ok=True)
    split_paths.output_arrays_dir.mkdir(parents=True, exist_ok=True)
    split_paths.output_manifests_dir.mkdir(parents=True, exist_ok=True)


def resolve_input_image_path(split_paths: SplitPaths, manifest_value: object) -> Path:
    if manifest_value is None:
        raise ValueError("image_filename is missing")

    text = str(manifest_value).strip()
    if not text:
        raise ValueError("image_filename is blank")

    rel = Path(text)
    if rel.is_absolute():
        return rel
    if rel.parts and rel.parts[0] == "images":
        return split_paths.input_root / rel
    return split_paths.input_images_dir / rel


def split_relative_path(split_root: Path, path: Path) -> str:
    return path.resolve().relative_to(split_root.resolve()).as_posix()
