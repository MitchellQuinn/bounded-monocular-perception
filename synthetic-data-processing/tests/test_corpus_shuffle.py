"""Tests for input corpus shuffle implementation."""

from __future__ import annotations

import csv
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rb_pipeline.corpus_shuffle import (
    build_output_name,
    build_output_sample_id,
    default_destination_path,
    discover_corpuses,
    parse_seed,
    shuffle_corpus,
)
from rb_pipeline.corpus_shuffle.exceptions import CorpusShuffleValidationError


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _write_fixture_corpus(
    root: Path,
    *,
    name: str = "fixture-corpus",
    row_count: int = 12,
    failing_capture_indices: set[int] | None = None,
    missing_image_indices: set[int] | None = None,
    drop_required_column: str | None = None,
) -> Path:
    failing_capture_indices = failing_capture_indices or set()
    missing_image_indices = missing_image_indices or set()

    corpus = root / name
    images_dir = corpus / "images"
    manifests_dir = corpus / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    (corpus / "notes.txt").write_text("fixture support file\n", encoding="utf-8")
    (manifests_dir / "run.json").write_text(json.dumps({"RunId": name}), encoding="utf-8")

    rows: list[dict[str, str]] = []
    for idx in range(row_count):
        source_frame = 1000 + idx
        z_token = f"z{1.5 + (idx / 100):05.3f}"
        j_token = f"j{idx:03d}"
        sample_id = f"defender90_f{source_frame:06d}_{z_token}_{j_token}"
        image_filename = f"defender90_f{source_frame:06d}_{z_token}_{j_token}.png"

        row = {
            "run_id": name,
            "sample_id": sample_id,
            "frame_index": str(source_frame),
            "image_filename": image_filename,
            "position_step_index": str(idx),
            "sample_at_position_index": str(idx % 4),
            "capture_success": "false" if idx in failing_capture_indices else "true",
            "distance_m": f"{2.5 + idx / 10:.6f}",
        }
        rows.append(row)

        if idx not in missing_image_indices:
            (images_dir / image_filename).write_text(
                f"content::{sample_id}\n",
                encoding="utf-8",
            )

    if drop_required_column is not None:
        for row in rows:
            row.pop(drop_required_column, None)

    fieldnames = list(rows[0].keys())
    with (manifests_dir / "samples.csv").open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return corpus


class CorpusShuffleTests(unittest.TestCase):
    def test_same_seed_produces_same_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root)

            out_a = root / "out-a"
            out_b = root / "out-b"
            shuffle_corpus(source, 17, destination_path=out_a)
            shuffle_corpus(source, 17, destination_path=out_b)

            rows_a = _read_csv_rows(out_a / "manifests" / "samples.csv")
            rows_b = _read_csv_rows(out_b / "manifests" / "samples.csv")
            self.assertEqual(
                [row["sample_id"] for row in rows_a],
                [row["sample_id"] for row in rows_b],
            )

    def test_different_seed_produces_different_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root)

            out_a = root / "out-a"
            out_b = root / "out-b"
            shuffle_corpus(source, 17, destination_path=out_a)
            shuffle_corpus(source, 18, destination_path=out_b)

            rows_a = _read_csv_rows(out_a / "manifests" / "samples.csv")
            rows_b = _read_csv_rows(out_b / "manifests" / "samples.csv")
            self.assertNotEqual(
                [row["sample_id"] for row in rows_a],
                [row["sample_id"] for row in rows_b],
            )

    def test_default_destination_path_appends_shuffled(self) -> None:
        source = Path("/tmp/example-corpus")
        expected = Path("/tmp/example-corpus-shuffled")
        self.assertEqual(default_destination_path(source), expected)

    def test_run_json_is_copied(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root)
            destination = root / "out"

            shuffle_corpus(source, 17, destination_path=destination)

            self.assertEqual(
                (source / "manifests" / "run.json").read_text(encoding="utf-8"),
                (destination / "manifests" / "run.json").read_text(encoding="utf-8"),
            )

    def test_samples_csv_is_regenerated_with_sequential_frame_index(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root)
            destination = root / "out"

            shuffle_corpus(source, 17, destination_path=destination)

            source_csv_text = (source / "manifests" / "samples.csv").read_text(encoding="utf-8-sig")
            output_csv_text = (destination / "manifests" / "samples.csv").read_text(encoding="utf-8-sig")
            self.assertNotEqual(source_csv_text, output_csv_text)

            output_rows = _read_csv_rows(destination / "manifests" / "samples.csv")
            self.assertEqual(
                [int(row["frame_index"]) for row in output_rows],
                list(range(len(output_rows))),
            )

    def test_output_filenames_sort_in_shuffled_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root)
            destination = root / "out"

            shuffle_corpus(source, 17, destination_path=destination)

            output_rows = _read_csv_rows(destination / "manifests" / "samples.csv")
            filenames = [row["image_filename"] for row in output_rows]
            self.assertEqual(filenames, sorted(filenames))

    def test_output_name_and_sample_id_tokens_are_updated(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root)
            destination = root / "out"
            shuffle_corpus(source, 17, destination_path=destination)

            output_rows = _read_csv_rows(destination / "manifests" / "samples.csv")
            token_pattern = re.compile(
                r"^(?P<prefix>.+)_f(?P<frame>\d{6})_(?P<z>z\d+\.\d{3})_(?P<j>j\d{3})$"
            )

            for expected_index, row in enumerate(output_rows):
                sample_id = row["sample_id"]
                image_stem = Path(row["image_filename"]).stem

                self.assertEqual(sample_id, image_stem)
                self.assertIn(f"f{expected_index:06d}", sample_id)

                match = token_pattern.match(sample_id)
                self.assertIsNotNone(match)
                self.assertTrue(match.group("z").startswith("z"))
                self.assertTrue(match.group("j").startswith("j"))

    def test_image_content_stays_attached_to_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root)
            destination = root / "out"

            source_rows = _read_csv_rows(source / "manifests" / "samples.csv")
            source_by_content: dict[str, dict[str, str]] = {}
            for row in source_rows:
                content = (source / "images" / row["image_filename"]).read_text(encoding="utf-8").strip()
                source_by_content[content] = row

            shuffle_corpus(source, 17, destination_path=destination)
            output_rows = _read_csv_rows(destination / "manifests" / "samples.csv")

            for row in output_rows:
                output_image = destination / "images" / row["image_filename"]
                content = output_image.read_text(encoding="utf-8").strip()
                source_row = source_by_content[content]

                self.assertEqual(row["position_step_index"], source_row["position_step_index"])
                self.assertEqual(
                    row["sample_at_position_index"],
                    source_row["sample_at_position_index"],
                )
                self.assertEqual(
                    row["sample_id"].split("_")[2:],
                    source_row["sample_id"].split("_")[2:],
                )

    def test_missing_seed_is_rejected(self) -> None:
        with self.assertRaises(CorpusShuffleValidationError):
            parse_seed("")

    def test_non_integer_seed_is_rejected(self) -> None:
        with self.assertRaises(CorpusShuffleValidationError):
            parse_seed("not-an-int")

    def test_missing_required_columns_fail_cleanly(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root, drop_required_column="sample_id")
            with self.assertRaises(CorpusShuffleValidationError):
                shuffle_corpus(source, 17, destination_path=root / "out")

    def test_missing_image_file_fails_cleanly(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root, missing_image_indices={3})
            with self.assertRaises(CorpusShuffleValidationError):
                shuffle_corpus(source, 17, destination_path=root / "out")

    def test_existing_destination_fails_safely(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root)
            destination = root / "out"
            destination.mkdir(parents=True, exist_ok=True)

            with self.assertRaises(CorpusShuffleValidationError):
                shuffle_corpus(source, 17, destination_path=destination)

    def test_discovery_reports_selectable_corpus(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _write_fixture_corpus(root, name="discover-me")

            summaries = discover_corpuses(root)
            summary_by_name = {summary.name: summary for summary in summaries}
            self.assertIn(corpus.name, summary_by_name)
            self.assertTrue(summary_by_name[corpus.name].has_samples_csv)
            self.assertTrue(summary_by_name[corpus.name].has_run_json)
            self.assertTrue(summary_by_name[corpus.name].selectable)

    def test_capture_failure_rows_are_excluded_explicitly(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _write_fixture_corpus(root, failing_capture_indices={1, 2})
            destination = root / "out"

            result = shuffle_corpus(source, 17, destination_path=destination)
            output_rows = _read_csv_rows(destination / "manifests" / "samples.csv")

            self.assertEqual(result.excluded_capture_failed_rows, 2)
            self.assertEqual(len(output_rows), 10)

    def test_output_helpers_replace_frame_token_only(self) -> None:
        image_filename = "defender90_f007255_z05.120_j015.png"
        sample_id = "defender90_f007255_z05.120_j015"

        self.assertEqual(
            build_output_name(image_filename, 42),
            "defender90_f000042_z05.120_j015.png",
        )
        self.assertEqual(
            build_output_sample_id(sample_id, 42),
            "defender90_f000042_z05.120_j015",
        )


if __name__ == "__main__":
    unittest.main()
