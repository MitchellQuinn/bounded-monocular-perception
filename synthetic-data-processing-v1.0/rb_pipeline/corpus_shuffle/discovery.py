"""Corpus discovery for the input corpus shuffle notebook UI."""

from __future__ import annotations

import csv
from pathlib import Path

from .loader import image_filename_to_relative_path, load_sample_records
from .models import CorpusSummary


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
                rel = image_filename_to_relative_path(image_filename)
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


def discover_corpuses(input_images_root: str | Path) -> list[CorpusSummary]:
    """Discover candidate corpuses under input-images root."""

    root = Path(input_images_root)
    if not root.is_dir():
        return []

    summaries: list[CorpusSummary] = []
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
            row_count, referenced_found, referenced_missing = _inspect_csv_references(
                entry,
                samples_csv_path,
            )
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
                load_result = load_sample_records(entry)
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
            CorpusSummary(
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

