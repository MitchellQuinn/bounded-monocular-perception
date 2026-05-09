#!/usr/bin/env python3
"""Add placement_bin_id to an existing synthetic corpus manifest.

The current Unity generator uses placement_bin_id as its resumable source of
truth. Older manifests already have position_step_index, which was the
StratifiedPlacementCell.CellIndex, so this script copies that value into the
new column and leaves images untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


DEFAULT_BIN_COLUMN = "placement_bin_id"
DEFAULT_SOURCE_COLUMN = "position_step_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upgrade an existing synthetic dataset manifest for safe resume."
    )
    parser.add_argument(
        "corpus_root",
        type=Path,
        help="Corpus root containing images/ and manifests/samples.csv.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override manifest path. Defaults to <corpus_root>/manifests/samples.csv.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override images directory. Defaults to <corpus_root>/images.",
    )
    parser.add_argument(
        "--source-column",
        default=DEFAULT_SOURCE_COLUMN,
        help=f"Column to copy into {DEFAULT_BIN_COLUMN}. Default: {DEFAULT_SOURCE_COLUMN}.",
    )
    parser.add_argument(
        "--bin-column",
        default=DEFAULT_BIN_COLUMN,
        help=f"Resume bin column to add. Default: {DEFAULT_BIN_COLUMN}.",
    )
    parser.add_argument(
        "--valid-bin-count",
        type=int,
        default=None,
        help="Expected valid bin count for capacity validation. Defaults to observed bins.",
    )
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Do not fail when manifest row count and PNG count differ.",
    )
    parser.add_argument(
        "--skip-capacity-check",
        action="store_true",
        help="Do not validate existing bin counts against run.json Sweep.TotalSamples.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create a timestamped backup before replacing the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report without writing changes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus_root = args.corpus_root.resolve()
    manifest_path = (args.manifest or corpus_root / "manifests" / "samples.csv").resolve()
    images_dir = (args.images_dir or corpus_root / "images").resolve()

    if not corpus_root.exists():
        print(f"ERROR: corpus root does not exist: {corpus_root}", file=sys.stderr)
        return 2
    if not manifest_path.exists():
        print(f"ERROR: manifest does not exist: {manifest_path}", file=sys.stderr)
        return 2
    if not images_dir.exists():
        print(f"ERROR: images directory does not exist: {images_dir}", file=sys.stderr)
        return 2

    try:
        result = inspect_manifest(
            manifest_path,
            source_column=args.source_column,
            bin_column=args.bin_column,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    image_count = count_pngs(images_dir)
    print(f"Manifest rows: {result.row_count}")
    print(f"PNG images:     {image_count}")

    if result.row_count != image_count and not args.allow_count_mismatch:
        print(
            "ERROR: manifest row count and PNG image count differ. "
            "Fix the corpus or rerun with --allow-count-mismatch.",
            file=sys.stderr,
        )
        return 2

    if not result.bin_counts:
        print("ERROR: no bin ids were found in the manifest.", file=sys.stderr)
        return 2

    print(f"Observed bins:  {len(result.bin_counts)}")
    print(f"Max bin count:  {max(result.bin_counts.values())}")

    if not args.skip_capacity_check:
        capacity_error = validate_capacity(
            corpus_root,
            result.bin_counts,
            args.valid_bin_count,
        )
        if capacity_error:
            print(f"ERROR: {capacity_error}", file=sys.stderr)
            return 2

    if result.has_bin_column:
        print(f"Manifest already has {args.bin_column}; no rewrite needed.")
        return 0

    if args.dry_run:
        print(f"Dry run only; would add {args.bin_column} from {args.source_column}.")
        return 0

    backup_path = None
    if not args.no_backup:
        backup_path = build_backup_path(manifest_path)
        shutil.copy2(manifest_path, backup_path)

    temp_path = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        rewrite_manifest(
            manifest_path,
            temp_path,
            source_column=args.source_column,
            bin_column=args.bin_column,
        )
        os.replace(temp_path, manifest_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"Added {args.bin_column} to {manifest_path}")
    if backup_path is not None:
        print(f"Backup written to {backup_path}")

    return 0


class ManifestInspection:
    def __init__(self, row_count: int, bin_counts: Counter[int], has_bin_column: bool):
        self.row_count = row_count
        self.bin_counts = bin_counts
        self.has_bin_column = has_bin_column


def inspect_manifest(
    manifest_path: Path,
    *,
    source_column: str,
    bin_column: str,
) -> ManifestInspection:
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"manifest is empty: {manifest_path}") from exc

        has_bin_column = bin_column in header
        if not has_bin_column and source_column not in header:
            raise ValueError(
                f"manifest is missing both {bin_column!r} and source column {source_column!r}: "
                f"{manifest_path}"
            )

        bin_index = header.index(bin_column) if has_bin_column else header.index(source_column)
        row_count = 0
        bin_counts: Counter[int] = Counter()

        for line_number, row in enumerate(reader, start=2):
            if not row or all(value == "" for value in row):
                continue
            if len(row) != len(header):
                raise ValueError(
                    f"line {line_number} has {len(row)} columns; expected {len(header)}"
                )
            bin_id = parse_bin_id(row[bin_index], line_number)
            bin_counts[bin_id] += 1
            row_count += 1

    return ManifestInspection(row_count, bin_counts, has_bin_column)


def rewrite_manifest(
    manifest_path: Path,
    temp_path: Path,
    *,
    source_column: str,
    bin_column: str,
) -> None:
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as src:
        reader = csv.reader(src)
        header = next(reader)
        if bin_column in header:
            raise ValueError(f"manifest already contains {bin_column!r}")
        if source_column not in header:
            raise ValueError(f"manifest is missing source column {source_column!r}")

        source_index = header.index(source_column)
        insert_index = header.index("image_filename") + 1 if "image_filename" in header else source_index
        new_header = header[:insert_index] + [bin_column] + header[insert_index:]

        with temp_path.open("w", newline="", encoding="utf-8") as dst:
            writer = csv.writer(dst, lineterminator="\n")
            writer.writerow(new_header)

            for line_number, row in enumerate(reader, start=2):
                if not row or all(value == "" for value in row):
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"line {line_number} has {len(row)} columns; expected {len(header)}"
                    )
                bin_text = row[source_index]
                _ = parse_bin_id(bin_text, line_number)
                writer.writerow(row[:insert_index] + [bin_text] + row[insert_index:])


def parse_bin_id(value: str, line_number: int) -> int:
    try:
        bin_id = int(value)
    except ValueError as exc:
        raise ValueError(f"line {line_number} has invalid bin id {value!r}") from exc
    if bin_id < 0:
        raise ValueError(f"line {line_number} has negative bin id {bin_id}")
    return bin_id


def count_pngs(images_dir: Path) -> int:
    return sum(1 for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() == ".png")


def validate_capacity(
    corpus_root: Path,
    bin_counts: Counter[int],
    valid_bin_count: int | None,
) -> str | None:
    run_json_path = corpus_root / "manifests" / "run.json"
    if not run_json_path.exists():
        print("Capacity check skipped: manifests/run.json not found.")
        return None

    with run_json_path.open("r", encoding="utf-8-sig") as handle:
        run_metadata = json.load(handle)

    total_samples = run_metadata.get("Sweep", {}).get("TotalSamples")
    if not isinstance(total_samples, int) or total_samples <= 0:
        print("Capacity check skipped: run.json has no positive Sweep.TotalSamples.")
        return None

    capacity_bin_count = valid_bin_count or len(bin_counts)
    if capacity_bin_count <= 0:
        return "valid bin count must be positive."

    capacity = math.ceil(total_samples / capacity_bin_count)
    max_count = max(bin_counts.values())
    print(f"Target samples: {total_samples}")
    print(f"Capacity bins:  {capacity_bin_count}")
    print(f"Per-bin cap:    {capacity}")

    if max_count > capacity:
        return (
            "existing manifest exceeds the resumable per-bin capacity. "
            f"max_bin_count={max_count}, per_bin_capacity={capacity}"
        )

    return None


def build_backup_path(manifest_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return manifest_path.with_name(f"{manifest_path.name}.pre-resume-{stamp}.bak")


if __name__ == "__main__":
    raise SystemExit(main())
