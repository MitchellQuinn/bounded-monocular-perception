"""Input-corpora shuffle utilities for ROI-FCN preprocessing."""

from __future__ import annotations

import csv
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .contracts import SPLIT_ORDER
from .paths import dataset_input_root, find_preprocessing_root
from .validation import ensure_valid_input_dataset_reference

_REQUIRED_COLUMNS = ("frame_index", "sample_id", "image_filename")
_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
_FALSE_VALUES = {"0", "false", "f", "no", "n"}
_FRAME_TOKEN_PATTERN = re.compile(r"f\d{6}")


class RoiFcnInputCorporaShuffleError(RuntimeError):
    """Base exception for ROI-FCN input-corpora shuffle failures."""


class RoiFcnInputCorporaShuffleValidationError(RoiFcnInputCorporaShuffleError):
    """Raised when ROI-FCN input-corpora shuffle validation fails."""


@dataclass(frozen=True)
class InputSplitShuffleResult:
    """Result metadata from shuffling one split corpus."""

    split_name: str
    source_split_path: Path
    destination_split_path: Path
    seed: int
    total_rows: int
    output_row_count: int
    excluded_capture_failed_rows: int


@dataclass(frozen=True)
class InputDatasetShuffleResult:
    """Result metadata from shuffling an ROI-FCN input dataset."""

    source_dataset_reference: str
    destination_dataset_reference: str
    source_dataset_path: Path
    destination_dataset_path: Path
    seed: int
    split_results: list[InputSplitShuffleResult]


@dataclass(frozen=True)
class _SampleRecord:
    source_row_index: int
    source_frame_index: int
    source_row: dict[str, str]
    source_image_path: Path
    source_image_relative_path: Path


@dataclass(frozen=True)
class _LoadResult:
    samples_csv_path: Path
    run_json_path: Path
    fieldnames: list[str]
    records: list[_SampleRecord]
    total_rows: int
    included_rows: int
    excluded_capture_failed_rows: int


def parse_shuffle_seed(seed_value: object) -> int:
    """Parse and validate a deterministic shuffle seed."""

    if seed_value is None:
        raise RoiFcnInputCorporaShuffleValidationError("Seed is required and cannot be empty.")
    if isinstance(seed_value, bool):
        raise RoiFcnInputCorporaShuffleValidationError("Seed must be an integer, not a boolean.")
    if isinstance(seed_value, int):
        return seed_value
    if isinstance(seed_value, str):
        stripped = seed_value.strip()
        if not stripped:
            raise RoiFcnInputCorporaShuffleValidationError("Seed is required and cannot be empty.")
        try:
            return int(stripped, 10)
        except ValueError as exc:
            raise RoiFcnInputCorporaShuffleValidationError(
                f"Seed must be a valid integer. Got: {seed_value!r}."
            ) from exc
    raise RoiFcnInputCorporaShuffleValidationError(
        f"Seed must be an integer. Got type={type(seed_value).__name__}."
    )


def default_shuffled_dataset_reference(dataset_reference: str) -> str:
    """Return the default shuffled sibling dataset reference."""

    dataset_name = str(dataset_reference).strip()
    if not dataset_name:
        raise RoiFcnInputCorporaShuffleValidationError("dataset_reference cannot be blank.")
    return f"{dataset_name}-shuffled"


def shuffle_input_dataset_corpora(
    preprocessing_root: Path | None,
    dataset_reference: str,
    seed: int | str,
    *,
    destination_dataset_reference: str | None = None,
) -> InputDatasetShuffleResult:
    """Shuffle both input corpora for one ROI-FCN dataset reference."""

    root = find_preprocessing_root(preprocessing_root)
    source_dataset_name = str(dataset_reference).strip()
    if not source_dataset_name:
        raise RoiFcnInputCorporaShuffleValidationError("dataset_reference cannot be blank.")

    ensure_valid_input_dataset_reference(root, source_dataset_name)

    validated_seed = parse_shuffle_seed(seed)
    destination_dataset_name = (
        default_shuffled_dataset_reference(source_dataset_name)
        if destination_dataset_reference is None
        else str(destination_dataset_reference).strip()
    )
    if not destination_dataset_name:
        raise RoiFcnInputCorporaShuffleValidationError(
            "destination_dataset_reference cannot be blank."
        )
    if destination_dataset_name == source_dataset_name:
        raise RoiFcnInputCorporaShuffleValidationError(
            "destination_dataset_reference must differ from dataset_reference."
        )

    source_dataset_path = dataset_input_root(root, source_dataset_name)
    destination_dataset_path = dataset_input_root(root, destination_dataset_name)
    if destination_dataset_path.exists():
        raise RoiFcnInputCorporaShuffleValidationError(
            f"Destination dataset already exists and will not be overwritten: {destination_dataset_path}"
        )

    destination_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    destination_dataset_path.mkdir(parents=False, exist_ok=False)

    split_results: list[InputSplitShuffleResult] = []
    try:
        for split_name in SPLIT_ORDER:
            split_results.append(
                _shuffle_split_corpus(
                    source_split_path=source_dataset_path / split_name,
                    destination_split_path=destination_dataset_path / split_name,
                    split_name=split_name,
                    seed=validated_seed,
                )
            )
    except Exception:
        shutil.rmtree(destination_dataset_path, ignore_errors=True)
        raise

    return InputDatasetShuffleResult(
        source_dataset_reference=source_dataset_name,
        destination_dataset_reference=destination_dataset_name,
        source_dataset_path=source_dataset_path,
        destination_dataset_path=destination_dataset_path,
        seed=validated_seed,
        split_results=split_results,
    )


def _shuffle_split_corpus(
    *,
    source_split_path: Path,
    destination_split_path: Path,
    split_name: str,
    seed: int,
) -> InputSplitShuffleResult:
    load_result = _load_sample_records(source_split_path)

    canonical_records = sorted(
        load_result.records,
        key=lambda record: (
            record.source_frame_index,
            record.source_row["image_filename"],
            record.source_row_index,
        ),
    )
    shuffled_records = list(canonical_records)
    random.Random(seed).shuffle(shuffled_records)

    destination_split_path.mkdir(parents=True, exist_ok=False)

    try:
        _replicate_directory_structure(source_split_path, destination_split_path)

        record_image_paths = {record.source_image_path for record in shuffled_records}
        _copy_non_record_files(
            source_root=source_split_path,
            destination_root=destination_split_path,
            run_json_path=load_result.run_json_path,
            samples_csv_path=load_result.samples_csv_path,
            record_image_paths=record_image_paths,
        )

        destination_run_json_path = destination_split_path / load_result.run_json_path.relative_to(
            source_split_path
        )
        destination_run_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(load_result.run_json_path, destination_run_json_path)

        output_rows: list[dict[str, str]] = []
        for new_frame_index, record in enumerate(shuffled_records):
            output_image_filename = _build_output_image_filename(
                record.source_row["image_filename"],
                new_frame_index,
            )
            destination_image_relative = record.source_image_relative_path.with_name(
                _rename_leaf_name(record.source_image_relative_path.name, new_frame_index)
            )
            destination_image_path = destination_split_path / destination_image_relative
            destination_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.source_image_path, destination_image_path)

            output_row = dict(record.source_row)
            output_row["frame_index"] = str(new_frame_index)
            output_row["image_filename"] = output_image_filename
            output_row["sample_id"] = _build_output_sample_id(
                record.source_row["sample_id"],
                new_frame_index,
                output_image_filename=output_image_filename,
            )
            output_rows.append(output_row)

        destination_samples_csv_path = destination_split_path / load_result.samples_csv_path.relative_to(
            source_split_path
        )
        destination_samples_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_samples_csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=load_result.fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
    except Exception:
        shutil.rmtree(destination_split_path, ignore_errors=True)
        raise

    return InputSplitShuffleResult(
        split_name=split_name,
        source_split_path=source_split_path,
        destination_split_path=destination_split_path,
        seed=seed,
        total_rows=load_result.total_rows,
        output_row_count=len(shuffled_records),
        excluded_capture_failed_rows=load_result.excluded_capture_failed_rows,
    )


def _resolve_manifest_paths(split_path: Path) -> tuple[Path, Path]:
    manifests_dir = split_path / "manifests"
    samples_csv_path = manifests_dir / "samples.csv"
    run_json_path = manifests_dir / "run.json"

    if not samples_csv_path.is_file():
        raise RoiFcnInputCorporaShuffleValidationError(
            f"Missing required samples.csv under {split_path / 'manifests'}."
        )
    if not run_json_path.is_file():
        raise RoiFcnInputCorporaShuffleValidationError(
            f"Missing required run.json under {split_path / 'manifests'}."
        )
    return samples_csv_path, run_json_path


def _image_filename_to_relative_path(image_filename: str) -> Path:
    normalized = str(image_filename).strip().replace("\\", "/")
    if not normalized:
        raise RoiFcnInputCorporaShuffleValidationError("image_filename is blank.")

    pure_path = PurePosixPath(normalized)
    if pure_path.is_absolute():
        raise RoiFcnInputCorporaShuffleValidationError(
            f"image_filename must be relative, got absolute path {image_filename!r}."
        )
    if any(part == ".." for part in pure_path.parts):
        raise RoiFcnInputCorporaShuffleValidationError(
            f"image_filename must not contain '..', got {image_filename!r}."
        )

    return Path(*pure_path.parts)


def _parse_capture_success(raw_value: str, row_index: int) -> bool:
    text = str(raw_value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise RoiFcnInputCorporaShuffleValidationError(
        f"Row {row_index}: capture_success={raw_value!r} is not a recognized boolean."
    )


def _resolve_image_path(split_path: Path, image_filename: str, row_index: int) -> tuple[Path, Path]:
    relative_from_csv = _image_filename_to_relative_path(image_filename)
    candidates = [split_path / relative_from_csv]

    if relative_from_csv.parts and relative_from_csv.parts[0] != "images":
        if len(relative_from_csv.parts) == 1:
            candidates.append(split_path / "images" / relative_from_csv.name)
        else:
            candidates.append(split_path / "images" / relative_from_csv)

    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidate.relative_to(split_path)

    tried = ", ".join(str(candidate.relative_to(split_path)) for candidate in candidates)
    raise RoiFcnInputCorporaShuffleValidationError(
        f"Row {row_index}: missing image for image_filename={image_filename!r}. Tried: {tried}."
    )


def _replace_frame_token_if_present(value: str, new_frame_index: int) -> str | None:
    replacement = f"f{new_frame_index:06d}"
    replaced, count = _FRAME_TOKEN_PATTERN.subn(replacement, value, count=1)
    if count == 0:
        return None
    return replaced


def _rename_leaf_name(leaf_name: str, new_frame_index: int) -> str:
    replaced = _replace_frame_token_if_present(str(leaf_name), new_frame_index)
    if replaced is not None:
        return replaced

    leaf_path = Path(leaf_name)
    return f"{leaf_path.stem}__f{new_frame_index:06d}{leaf_path.suffix}"


def _build_output_image_filename(source_image_filename: str, new_frame_index: int) -> str:
    source_path = PurePosixPath(str(source_image_filename).replace("\\", "/"))
    return str(source_path.with_name(_rename_leaf_name(source_path.name, new_frame_index)))


def _build_output_sample_id(
    source_sample_id: str,
    new_frame_index: int,
    *,
    output_image_filename: str,
) -> str:
    sample_id = str(source_sample_id).strip()
    replaced = _replace_frame_token_if_present(sample_id, new_frame_index) if sample_id else None
    if replaced is not None:
        return replaced
    if sample_id:
        return f"{sample_id}__f{new_frame_index:06d}"
    return Path(PurePosixPath(output_image_filename).name).stem


def _load_sample_records(split_path: Path) -> _LoadResult:
    samples_csv_path, run_json_path = _resolve_manifest_paths(split_path)

    with samples_csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise RoiFcnInputCorporaShuffleValidationError(f"{samples_csv_path} has no header row.")
        fieldnames = list(reader.fieldnames)

        missing = [column for column in _REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise RoiFcnInputCorporaShuffleValidationError(
                "samples.csv is missing required columns: " + ", ".join(missing)
            )

        total_rows = 0
        excluded_capture_failed_rows = 0
        records: list[_SampleRecord] = []

        for csv_row_number, row in enumerate(reader, start=2):
            total_rows += 1

            frame_raw = str(row.get("frame_index", "")).strip()
            if not frame_raw:
                raise RoiFcnInputCorporaShuffleValidationError(
                    f"Row {csv_row_number}: frame_index is empty."
                )
            try:
                source_frame_index = int(frame_raw)
            except ValueError as exc:
                raise RoiFcnInputCorporaShuffleValidationError(
                    f"Row {csv_row_number}: frame_index={frame_raw!r} is not an integer."
                ) from exc

            if "capture_success" in fieldnames:
                capture_raw = row.get("capture_success", "")
                if not _parse_capture_success(str(capture_raw), csv_row_number):
                    excluded_capture_failed_rows += 1
                    continue

            image_filename = str(row.get("image_filename", "")).strip()
            source_image_path, source_image_relative_path = _resolve_image_path(
                split_path,
                image_filename,
                csv_row_number,
            )
            row_copy = {key: ("" if value is None else str(value)) for key, value in row.items()}
            records.append(
                _SampleRecord(
                    source_row_index=csv_row_number,
                    source_frame_index=source_frame_index,
                    source_row=row_copy,
                    source_image_path=source_image_path,
                    source_image_relative_path=source_image_relative_path,
                )
            )

    if total_rows == 0:
        raise RoiFcnInputCorporaShuffleValidationError(f"{samples_csv_path} contains zero data rows.")
    if not records:
        if excluded_capture_failed_rows > 0:
            raise RoiFcnInputCorporaShuffleValidationError(
                "No rows remained after filtering capture_success == true. "
                f"Filtered rows: {excluded_capture_failed_rows}."
            )
        raise RoiFcnInputCorporaShuffleValidationError("No valid sample rows found in samples.csv.")

    return _LoadResult(
        samples_csv_path=samples_csv_path,
        run_json_path=run_json_path,
        fieldnames=fieldnames,
        records=records,
        total_rows=total_rows,
        included_rows=len(records),
        excluded_capture_failed_rows=excluded_capture_failed_rows,
    )


def _replicate_directory_structure(source_root: Path, destination_root: Path) -> None:
    for directory in sorted(path for path in source_root.rglob("*") if path.is_dir()):
        relative = directory.relative_to(source_root)
        (destination_root / relative).mkdir(parents=True, exist_ok=True)


def _copy_non_record_files(
    *,
    source_root: Path,
    destination_root: Path,
    run_json_path: Path,
    samples_csv_path: Path,
    record_image_paths: set[Path],
) -> None:
    for file_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        if file_path == run_json_path or file_path == samples_csv_path:
            continue
        if file_path in record_image_paths:
            continue

        relative = file_path.relative_to(source_root)
        destination_file_path = destination_root / relative
        destination_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination_file_path)
