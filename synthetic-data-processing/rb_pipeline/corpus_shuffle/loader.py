"""Validation and record loading for input corpus shuffle."""

from __future__ import annotations

import csv
from pathlib import Path, PurePosixPath

from .exceptions import CorpusShuffleValidationError
from .models import LoadResult, SampleRecord
from .naming import ensure_frame_token

REQUIRED_COLUMNS = ("frame_index", "sample_id", "image_filename")
_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
_FALSE_VALUES = {"0", "false", "f", "no", "n"}


def parse_seed(seed_value: object) -> int:
    """Parse and validate seed as an explicit integer."""

    if seed_value is None:
        raise CorpusShuffleValidationError("Seed is required and cannot be empty.")
    if isinstance(seed_value, bool):
        raise CorpusShuffleValidationError("Seed must be an integer, not a boolean.")
    if isinstance(seed_value, int):
        return seed_value
    if isinstance(seed_value, str):
        stripped = seed_value.strip()
        if not stripped:
            raise CorpusShuffleValidationError("Seed is required and cannot be empty.")
        try:
            return int(stripped, 10)
        except ValueError as exc:
            raise CorpusShuffleValidationError(
                f"Seed must be a valid integer. Got: {seed_value!r}."
            ) from exc
    raise CorpusShuffleValidationError(
        f"Seed must be an integer. Got type={type(seed_value).__name__}."
    )


def resolve_manifest_paths(corpus_path: Path) -> tuple[Path, Path]:
    """
    Resolve (samples.csv, run.json) paths.

    Supports both:
    - <corpus>/manifests/samples.csv and <corpus>/manifests/run.json
    - <corpus>/samples.csv and <corpus>/run.json
    """

    manifest_samples = corpus_path / "manifests" / "samples.csv"
    root_samples = corpus_path / "samples.csv"
    if manifest_samples.is_file():
        samples_csv_path = manifest_samples
    elif root_samples.is_file():
        samples_csv_path = root_samples
    else:
        raise CorpusShuffleValidationError(
            f"Missing required samples.csv under {corpus_path} (checked root and manifests/)."
        )

    manifest_run_json = corpus_path / "manifests" / "run.json"
    root_run_json = corpus_path / "run.json"
    if manifest_run_json.is_file():
        run_json_path = manifest_run_json
    elif root_run_json.is_file():
        run_json_path = root_run_json
    else:
        raise CorpusShuffleValidationError(
            f"Missing required run.json under {corpus_path} (checked root and manifests/)."
        )

    return samples_csv_path, run_json_path


def image_filename_to_relative_path(image_filename: str) -> Path:
    """Normalize CSV image filename/path into a safe relative Path."""

    normalized = str(image_filename).strip().replace("\\", "/")
    if not normalized:
        raise CorpusShuffleValidationError("image_filename is blank.")

    pure_path = PurePosixPath(normalized)
    if pure_path.is_absolute():
        raise CorpusShuffleValidationError(
            f"image_filename must be relative, got absolute path {image_filename!r}."
        )
    if any(part == ".." for part in pure_path.parts):
        raise CorpusShuffleValidationError(
            f"image_filename must not contain '..', got {image_filename!r}."
        )

    return Path(*pure_path.parts)


def _parse_capture_success(raw_value: str, row_index: int) -> bool:
    text = str(raw_value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise CorpusShuffleValidationError(
        f"Row {row_index}: capture_success={raw_value!r} is not a recognized boolean."
    )


def _resolve_image_path(corpus_path: Path, image_filename: str, row_index: int) -> tuple[Path, Path]:
    """
    Resolve image file from image_filename.

    Tries:
    - <corpus>/<image_filename>
    - <corpus>/images/<image_filename basename or relative path>
    """

    relative_from_csv = image_filename_to_relative_path(image_filename)
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
    raise CorpusShuffleValidationError(
        f"Row {row_index}: missing image for image_filename={image_filename!r}. Tried: {tried}."
    )


def load_sample_records(
    corpus_path: str | Path,
    *,
    include_capture_success_only: bool = True,
) -> LoadResult:
    """Load and validate sample records from one corpus."""

    source_corpus = Path(corpus_path)
    if not source_corpus.is_dir():
        raise CorpusShuffleValidationError(f"Selected corpus path does not exist: {source_corpus}")

    samples_csv_path, run_json_path = resolve_manifest_paths(source_corpus)

    with samples_csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            raise CorpusShuffleValidationError(f"{samples_csv_path} has no header row.")
        fieldnames = list(reader.fieldnames)

        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise CorpusShuffleValidationError(
                "samples.csv is missing required columns: " + ", ".join(missing)
            )

        total_rows = 0
        excluded_capture_failed_rows = 0
        records: list[SampleRecord] = []

        for csv_row_number, row in enumerate(reader, start=2):
            total_rows += 1

            frame_raw = str(row.get("frame_index", "")).strip()
            sample_id = str(row.get("sample_id", "")).strip()
            image_filename = str(row.get("image_filename", "")).strip()

            if not frame_raw:
                raise CorpusShuffleValidationError(f"Row {csv_row_number}: frame_index is empty.")
            try:
                source_frame_index = int(frame_raw)
            except ValueError as exc:
                raise CorpusShuffleValidationError(
                    f"Row {csv_row_number}: frame_index={frame_raw!r} is not an integer."
                ) from exc

            ensure_frame_token(sample_id, "sample_id")
            ensure_frame_token(image_filename, "image_filename")

            if include_capture_success_only and "capture_success" in fieldnames:
                capture_raw = row.get("capture_success", "")
                if not _parse_capture_success(capture_raw, csv_row_number):
                    excluded_capture_failed_rows += 1
                    continue

            source_image_path, source_image_relative_path = _resolve_image_path(
                source_corpus,
                image_filename,
                csv_row_number,
            )

            row_copy = {key: ("" if value is None else str(value)) for key, value in row.items()}
            records.append(
                SampleRecord(
                    source_row_index=csv_row_number,
                    source_frame_index=source_frame_index,
                    source_row=row_copy,
                    source_image_path=source_image_path,
                    source_image_relative_path=source_image_relative_path,
                )
            )

    if total_rows == 0:
        raise CorpusShuffleValidationError(f"{samples_csv_path} contains zero data rows.")
    if not records:
        if excluded_capture_failed_rows > 0:
            raise CorpusShuffleValidationError(
                "No rows remained after filtering capture_success == true. "
                f"Filtered rows: {excluded_capture_failed_rows}."
            )
        raise CorpusShuffleValidationError("No valid sample rows found in samples.csv.")

    return LoadResult(
        samples_csv_path=samples_csv_path,
        run_json_path=run_json_path,
        fieldnames=fieldnames,
        records=records,
        total_rows=total_rows,
        included_rows=len(records),
        excluded_capture_failed_rows=excluded_capture_failed_rows,
    )

