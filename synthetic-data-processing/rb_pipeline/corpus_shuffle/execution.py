"""Record-level seeded shuffle execution for input corpuses."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from .exceptions import CorpusShuffleValidationError
from .loader import load_sample_records, parse_seed
from .models import SampleRecord, ShuffleResult
from .naming import build_output_name, build_output_sample_id


def default_destination_path(source_corpus_path: str | Path) -> Path:
    """Return the default sibling output path with '-shuffled' suffix."""

    source = Path(source_corpus_path)
    return source.parent / f"{source.name}-shuffled"


def _canonical_sort(records: list[SampleRecord]) -> list[SampleRecord]:
    return sorted(
        records,
        key=lambda record: (
            record.source_frame_index,
            record.source_row["image_filename"],
        ),
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
    """
    Copy support files unchanged.

    Excludes run.json, samples.csv, and all images that are copied as part of shuffled records.
    """

    for file_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        if file_path == run_json_path or file_path == samples_csv_path:
            continue
        if file_path in record_image_paths:
            continue

        relative = file_path.relative_to(source_root)
        destination_file_path = destination_root / relative
        destination_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination_file_path)


def shuffle_corpus(
    source_corpus_path: str | Path,
    seed: int | str,
    destination_path: str | Path | None = None,
) -> ShuffleResult:
    """Shuffle one corpus by sample records and write a re-indexed corpus copy."""

    source = Path(source_corpus_path)
    if not source.is_dir():
        raise CorpusShuffleValidationError(f"Selected corpus path does not exist: {source}")

    validated_seed = parse_seed(seed)
    destination = default_destination_path(source) if destination_path is None else Path(destination_path)

    if destination.exists():
        raise CorpusShuffleValidationError(
            f"Destination already exists and will not be overwritten: {destination}"
        )

    load_result = load_sample_records(source)
    canonical_records = _canonical_sort(load_result.records)
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
            output_image_filename = build_output_name(record.source_row["image_filename"], new_frame_index)
            output_sample_id = build_output_sample_id(record.source_row["sample_id"], new_frame_index)

            renamed_basename = Path(
                build_output_name(record.source_image_relative_path.name, new_frame_index)
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

    return ShuffleResult(
        source_corpus_path=source,
        destination_corpus_path=destination,
        seed=validated_seed,
        output_row_count=len(shuffled_records),
        excluded_capture_failed_rows=load_result.excluded_capture_failed_rows,
    )

