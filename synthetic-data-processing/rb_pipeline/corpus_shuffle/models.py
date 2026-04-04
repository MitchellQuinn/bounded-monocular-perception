"""Data models for input corpus shuffle operations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRecord:
    """One row from samples.csv tied to its source image file."""

    source_row_index: int
    source_frame_index: int
    source_row: dict[str, str]
    source_image_path: Path
    source_image_relative_path: Path


@dataclass(frozen=True)
class LoadResult:
    """Validated sample record load output."""

    samples_csv_path: Path
    run_json_path: Path
    fieldnames: list[str]
    records: list[SampleRecord]
    total_rows: int
    included_rows: int
    excluded_capture_failed_rows: int


@dataclass(frozen=True)
class CorpusSummary:
    """UI-friendly discovery summary for one input corpus."""

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
class ShuffleResult:
    """Result metadata from a completed corpus shuffle."""

    source_corpus_path: Path
    destination_corpus_path: Path
    seed: int
    output_row_count: int
    excluded_capture_failed_rows: int

