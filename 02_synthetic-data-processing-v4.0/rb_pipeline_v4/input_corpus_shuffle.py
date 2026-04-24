"""Fast input-corpus shuffle utilities for preprocessing."""

from __future__ import annotations

import csv
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_REQUIRED_COLUMNS = ("frame_index", "sample_id", "image_filename")
_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
_FALSE_VALUES = {"0", "false", "f", "no", "n"}
_FRAME_TOKEN_PATTERN = re.compile(r"f\d{6}")


class InputCorpusShuffleError(RuntimeError):
    """Base exception for input corpus shuffle failures."""


class InputCorpusShuffleValidationError(InputCorpusShuffleError):
    """Raised when input corpus validation fails."""


@dataclass(frozen=True)
class InputCorpusSummary:
    """Discovery summary for one candidate input corpus."""

    name: str
    path: Path
    has_samples_csv: bool
    has_run_json: bool
    samples_row_count: int | None
    referenced_images_found: int | None
    referenced_images_missing: int | None
    selectable: bool
    validates_cleanly: bool
    validation_message: str


@dataclass(frozen=True)
class InputCorpusShuffleResult:
    """Result metadata from a completed input corpus shuffle."""

    source_corpus_path: Path
    destination_corpus_path: Path
    seed: int
    output_row_count: int
    excluded_capture_failed_rows: int


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
    """Parse and validate seed as an integer."""

    if seed_value is None:
        raise InputCorpusShuffleValidationError("Seed is required and cannot be empty.")
    if isinstance(seed_value, bool):
        raise InputCorpusShuffleValidationError("Seed must be an integer, not a boolean.")
    if isinstance(seed_value, int):
        return seed_value
    if isinstance(seed_value, str):
        stripped = seed_value.strip()
        if not stripped:
            raise InputCorpusShuffleValidationError("Seed is required and cannot be empty.")
        try:
            return int(stripped, 10)
        except ValueError as exc:
            raise InputCorpusShuffleValidationError(
                f"Seed must be a valid integer. Got: {seed_value!r}."
            ) from exc
    raise InputCorpusShuffleValidationError(
        f"Seed must be an integer. Got type={type(seed_value).__name__}."
    )


def default_input_shuffle_destination(source_corpus_path: str | Path) -> Path:
    """Return default sibling output path with '-shuffled' suffix."""

    source = Path(source_corpus_path)
    return source.parent / f"{source.name}-shuffled"


def discover_input_corpuses(input_images_root: str | Path) -> list[InputCorpusSummary]:
    """Discover candidate input corpuses under input-images root."""

    root = Path(input_images_root)
    if not root.is_dir():
        return []

    summaries: list[InputCorpusSummary] = []
    for entry in sorted(root.iterdir(), key=lambda item: item.name):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        has_samples_csv = (entry / "samples.csv").is_file() or (entry / "manifests" / "samples.csv").is_file()
        has_run_json = (entry / "run.json").is_file() or (entry / "manifests" / "run.json").is_file()
        selectable = has_samples_csv and has_run_json

        if (entry / "manifests" / "samples.csv").is_file():
            samples_csv_path = entry / "manifests" / "samples.csv"
        else:
            samples_csv_path = entry / "samples.csv"

        if samples_csv_path.is_file():
            row_count, referenced_found, referenced_missing = _inspect_csv_references(entry, samples_csv_path)
        else:
            row_count, referenced_found, referenced_missing = None, None, None

        if not selectable:
            missing: list[str] = []
            if not has_samples_csv:
                missing.append("samples.csv")
            if not has_run_json:
                missing.append("run.json")
            validates_cleanly = False
            validation_message = "Missing required files: " + ", ".join(missing)
        else:
            try:
                load_result = _load_sample_records(entry)
            except Exception as exc:
                validates_cleanly = False
                validation_message = str(exc)
            else:
                validates_cleanly = True
                if load_result.excluded_capture_failed_rows > 0:
                    validation_message = (
                        "Valid (capture_success=false rows excluded: "
                        f"{load_result.excluded_capture_failed_rows})"
                    )
                else:
                    validation_message = "Valid"

        summaries.append(
            InputCorpusSummary(
                name=entry.name,
                path=entry,
                has_samples_csv=has_samples_csv,
                has_run_json=has_run_json,
                samples_row_count=row_count,
                referenced_images_found=referenced_found,
                referenced_images_missing=referenced_missing,
                selectable=selectable,
                validates_cleanly=validates_cleanly,
                validation_message=validation_message,
            )
        )

    return summaries


def shuffle_input_corpus(
    source_corpus_path: str | Path,
    seed: int | str,
    destination_path: str | Path | None = None,
) -> InputCorpusShuffleResult:
    """Shuffle one input corpus by sample record and write a re-indexed copy."""

    source = Path(source_corpus_path)
    if not source.is_dir():
        raise InputCorpusShuffleValidationError(f"Selected corpus path does not exist: {source}")

    validated_seed = parse_shuffle_seed(seed)
    destination = (
        default_input_shuffle_destination(source) if destination_path is None else Path(destination_path)
    )

    if destination.exists():
        raise InputCorpusShuffleValidationError(
            f"Destination already exists and will not be overwritten: {destination}"
        )

    load_result = _load_sample_records(source)
    canonical_records = sorted(
        load_result.records,
        key=lambda record: (
            record.source_frame_index,
            record.source_row["image_filename"],
        ),
    )
    shuffled_records = list(canonical_records)
    random.Random(validated_seed).shuffle(shuffled_records)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=False, exist_ok=False)

    run_json_relative = load_result.run_json_path.relative_to(source)
    samples_csv_relative = load_result.samples_csv_path.relative_to(source)

    try:
        _replicate_directory_structure(source, destination)

        record_image_paths = {record.source_image_path for record in shuffled_records}
        _copy_non_record_files(
            source_root=source,
            destination_root=destination,
            run_json_path=load_result.run_json_path,
            samples_csv_path=load_result.samples_csv_path,
            record_image_paths=record_image_paths,
        )

        destination_run_json_path = destination / run_json_relative
        destination_run_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(load_result.run_json_path, destination_run_json_path)

        output_rows: list[dict[str, str]] = []
        for new_frame_index, record in enumerate(shuffled_records):
            output_image_filename = _build_output_name(record.source_row["image_filename"], new_frame_index)
            output_sample_id = _build_output_sample_id(record.source_row["sample_id"], new_frame_index)

            renamed_basename = Path(
                _build_output_name(record.source_image_relative_path.name, new_frame_index)
            ).name
            destination_image_relative = record.source_image_relative_path.with_name(renamed_basename)
            destination_image_path = destination / destination_image_relative
            destination_image_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record.source_image_path, destination_image_path)

            output_row = dict(record.source_row)
            output_row["frame_index"] = str(new_frame_index)
            output_row["image_filename"] = output_image_filename
            output_row["sample_id"] = output_sample_id
            output_rows.append(output_row)

        destination_samples_csv_path = destination / samples_csv_relative
        destination_samples_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_samples_csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=load_result.fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise

    return InputCorpusShuffleResult(
        source_corpus_path=source,
        destination_corpus_path=destination,
        seed=validated_seed,
        output_row_count=len(shuffled_records),
        excluded_capture_failed_rows=load_result.excluded_capture_failed_rows,
    )


def _resolve_manifest_paths(corpus_path: Path) -> tuple[Path, Path]:
    manifest_samples = corpus_path / "manifests" / "samples.csv"
    root_samples = corpus_path / "samples.csv"
    if manifest_samples.is_file():
        samples_csv_path = manifest_samples
    elif root_samples.is_file():
        samples_csv_path = root_samples
    else:
        raise InputCorpusShuffleValidationError(
            f"Missing required samples.csv under {corpus_path} (checked root and manifests/)."
        )

    manifest_run_json = corpus_path / "manifests" / "run.json"
    root_run_json = corpus_path / "run.json"
    if manifest_run_json.is_file():
        run_json_path = manifest_run_json
    elif root_run_json.is_file():
        run_json_path = root_run_json
    else:
        raise InputCorpusShuffleValidationError(
            f"Missing required run.json under {corpus_path} (checked root and manifests/)."
        )

    return samples_csv_path, run_json_path


def _image_filename_to_relative_path(image_filename: str) -> Path:
    normalized = str(image_filename).strip().replace("\\", "/")
    if not normalized:
        raise InputCorpusShuffleValidationError("image_filename is blank.")

    pure_path = PurePosixPath(normalized)
    if pure_path.is_absolute():
        raise InputCorpusShuffleValidationError(
            f"image_filename must be relative, got absolute path {image_filename!r}."
        )
    if any(part == ".." for part in pure_path.parts):
        raise InputCorpusShuffleValidationError(
            f"image_filename must not contain '..', got {image_filename!r}."
        )

    return Path(*pure_path.parts)


def _parse_capture_success(raw_value: str, row_index: int) -> bool:
    text = str(raw_value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise InputCorpusShuffleValidationError(
        f"Row {row_index}: capture_success={raw_value!r} is not a recognized boolean."
    )


def _resolve_image_path(corpus_path: Path, image_filename: str, row_index: int) -> tuple[Path, Path]:
    relative_from_csv = _image_filename_to_relative_path(image_filename)
    candidates = [corpus_path / relative_from_csv]

    if relative_from_csv.parts and relative_from_csv.parts[0] != "images":
        if len(relative_from_csv.parts) == 1:
            candidates.append(corpus_path / "images" / relative_from_csv.name)
        else:
            candidates.append(corpus_path / "images" / relative_from_csv)

    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidate.relative_to(corpus_path)

    tried = ", ".join(str(candidate.relative_to(corpus_path)) for candidate in candidates)
    raise InputCorpusShuffleValidationError(
        f"Row {row_index}: missing image for image_filename={image_filename!r}. Tried: {tried}."
    )


def _ensure_frame_token(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InputCorpusShuffleValidationError(
            f"{field_name} is required and must be a non-empty string."
        )
    if _FRAME_TOKEN_PATTERN.search(value) is None:
        raise InputCorpusShuffleValidationError(
            f"{field_name}={value!r} does not contain a parseable f###### token."
        )


def _replace_frame_token(value: str, new_frame_index: int, field_name: str) -> str:
    _ensure_frame_token(value, field_name)
    replacement = f"f{new_frame_index:06d}"
    replaced, count = _FRAME_TOKEN_PATTERN.subn(replacement, value, count=1)
    if count != 1:
        raise InputCorpusShuffleValidationError(
            f"Could not replace frame token in {field_name}={value!r}."
        )
    return replaced


def _build_output_name(source_image_filename: str, new_frame_index: int) -> str:
    normalized = str(source_image_filename).replace("\\", "/")
    source_path = PurePosixPath(normalized)
    replaced_name = _replace_frame_token(source_path.name, new_frame_index, "image_filename")
    return str(source_path.with_name(replaced_name))


def _build_output_sample_id(source_sample_id: str, new_frame_index: int) -> str:
    return _replace_frame_token(source_sample_id, new_frame_index, "sample_id")


def _load_sample_records(
    corpus_path: Path,
    *,
    include_capture_success_only: bool = True,
) -> _LoadResult:
    samples_csv_path, run_json_path = _resolve_manifest_paths(corpus_path)

    with samples_csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise InputCorpusShuffleValidationError(f"{samples_csv_path} has no header row.")
        fieldnames = list(reader.fieldnames)

        missing = [column for column in _REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise InputCorpusShuffleValidationError(
                "samples.csv is missing required columns: " + ", ".join(missing)
            )

        total_rows = 0
        excluded_capture_failed_rows = 0
        records: list[_SampleRecord] = []

        for csv_row_number, row in enumerate(reader, start=2):
            total_rows += 1

            frame_raw = str(row.get("frame_index", "")).strip()
            sample_id = str(row.get("sample_id", "")).strip()
            image_filename = str(row.get("image_filename", "")).strip()

            if not frame_raw:
                raise InputCorpusShuffleValidationError(f"Row {csv_row_number}: frame_index is empty.")
            try:
                source_frame_index = int(frame_raw)
            except ValueError as exc:
                raise InputCorpusShuffleValidationError(
                    f"Row {csv_row_number}: frame_index={frame_raw!r} is not an integer."
                ) from exc

            _ensure_frame_token(sample_id, "sample_id")
            _ensure_frame_token(image_filename, "image_filename")

            if include_capture_success_only and "capture_success" in fieldnames:
                capture_raw = row.get("capture_success", "")
                if not _parse_capture_success(capture_raw, csv_row_number):
                    excluded_capture_failed_rows += 1
                    continue

            source_image_path, source_image_relative_path = _resolve_image_path(
                corpus_path,
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
        raise InputCorpusShuffleValidationError(f"{samples_csv_path} contains zero data rows.")
    if not records:
        if excluded_capture_failed_rows > 0:
            raise InputCorpusShuffleValidationError(
                "No rows remained after filtering capture_success == true. "
                f"Filtered rows: {excluded_capture_failed_rows}."
            )
        raise InputCorpusShuffleValidationError("No valid sample rows found in samples.csv.")

    return _LoadResult(
        samples_csv_path=samples_csv_path,
        run_json_path=run_json_path,
        fieldnames=fieldnames,
        records=records,
        total_rows=total_rows,
        included_rows=len(records),
        excluded_capture_failed_rows=excluded_capture_failed_rows,
    )


def _inspect_csv_references(corpus_path: Path, samples_csv_path: Path) -> tuple[int, int, int]:
    row_count = 0
    referenced_found = 0
    referenced_missing = 0

    with samples_csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None or "image_filename" not in reader.fieldnames:
            return 0, 0, 0

        for row in reader:
            row_count += 1
            image_filename = str(row.get("image_filename", "")).strip()
            if not image_filename:
                referenced_missing += 1
                continue

            try:
                rel = _image_filename_to_relative_path(image_filename)
            except Exception:
                referenced_missing += 1
                continue

            candidate_a = corpus_path / rel
            candidate_b = corpus_path / "images" / rel.name
            candidate_c = corpus_path / "images" / rel
            if candidate_a.is_file() or candidate_b.is_file() or candidate_c.is_file():
                referenced_found += 1
            else:
                referenced_missing += 1

    return row_count, referenced_found, referenced_missing


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

