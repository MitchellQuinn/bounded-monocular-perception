"""Input-corpora shuffle tests for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

import random
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from _test_support import build_dataset, ensure_preprocessing_root
from roi_fcn_preprocessing_v0_1.config import BootstrapCenterTargetConfig, PackRoiFcnConfig
from roi_fcn_preprocessing_v0_1.discovery import discover_dataset_references
from roi_fcn_preprocessing_v0_1.input_corpora_shuffle import (
    RoiFcnInputCorporaShuffleValidationError,
    default_shuffled_dataset_reference,
    shuffle_input_dataset_corpora,
)
from roi_fcn_preprocessing_v0_1.manifest import load_samples_csv
from roi_fcn_preprocessing_v0_1.pipeline import run_preprocessing_for_dataset


class InputCorporaShuffleTests(unittest.TestCase):
    def test_shuffle_creates_preprocessable_dataset_and_copies_sidecars(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_preprocessing_root(Path(tmpdir))
            build_dataset(
                root,
                "shuffle-fixture",
                train_rows=[
                    {
                        "filename": "traincar_f000000_view.png",
                        "sample_id": "traincar_f000000_view",
                        "distance_m": 1.1,
                        "width": 96,
                        "height": 64,
                        "box_xyxy": (20, 10, 76, 54),
                    },
                    {
                        "filename": "traincar_f000001_view.png",
                        "sample_id": "traincar_f000001_view",
                        "distance_m": 2.2,
                        "width": 96,
                        "height": 64,
                        "capture_success": False,
                        "write_image": False,
                    },
                    {
                        "filename": "traincar_f000002_view.png",
                        "sample_id": "traincar_f000002_view",
                        "distance_m": 3.3,
                        "width": 96,
                        "height": 64,
                        "box_xyxy": (24, 12, 72, 52),
                    },
                    {
                        "filename": "traincar_f000003_view.png",
                        "sample_id": "traincar_f000003_view",
                        "distance_m": 4.4,
                        "width": 96,
                        "height": 64,
                        "box_xyxy": (22, 14, 70, 50),
                    },
                ],
                validate_rows=[
                    {
                        "filename": "validcar_f000000_view.png",
                        "sample_id": "validcar_f000000_view",
                        "distance_m": 5.5,
                        "width": 96,
                        "height": 64,
                        "box_xyxy": (20, 10, 76, 54),
                    },
                    {
                        "filename": "validcar_f000001_view.png",
                        "sample_id": "validcar_f000001_view",
                        "distance_m": 6.6,
                        "width": 96,
                        "height": 64,
                        "box_xyxy": (24, 12, 72, 52),
                    },
                ],
            )
            (root / "input" / "shuffle-fixture" / "train" / "runlog.txt").write_text(
                "train-sidecar\n",
                encoding="utf-8",
            )
            (root / "input" / "shuffle-fixture" / "validate" / "runlog.txt").write_text(
                "validate-sidecar\n",
                encoding="utf-8",
            )

            result = shuffle_input_dataset_corpora(root, "shuffle-fixture", 7)

            self.assertEqual(
                result.destination_dataset_reference,
                default_shuffled_dataset_reference("shuffle-fixture"),
            )
            self.assertEqual(
                [(item.split_name, item.output_row_count, item.excluded_capture_failed_rows) for item in result.split_results],
                [("train", 3, 1), ("validate", 2, 0)],
            )

            discovered = [dataset.name for dataset in discover_dataset_references(root)]
            self.assertEqual(discovered, ["shuffle-fixture", "shuffle-fixture-shuffled"])

            destination_root = root / "input" / result.destination_dataset_reference
            self.assertTrue((destination_root / "train" / "runlog.txt").is_file())
            self.assertTrue((destination_root / "validate" / "runlog.txt").is_file())

            train_df = load_samples_csv(destination_root / "train" / "manifests" / "samples.csv")
            validate_df = load_samples_csv(destination_root / "validate" / "manifests" / "samples.csv")

            self.assertEqual(train_df["frame_index"].tolist(), [0, 1, 2])
            self.assertEqual(validate_df["frame_index"].tolist(), [0, 1])
            self.assertTrue(train_df["capture_success"].astype(bool).all())
            self.assertTrue(validate_df["capture_success"].astype(bool).all())

            expected_train_distances = [1.1, 3.3, 4.4]
            expected_validate_distances = [5.5, 6.6]
            random.Random(7).shuffle(expected_train_distances)
            random.Random(7).shuffle(expected_validate_distances)
            self.assertEqual(train_df["distance_m"].tolist(), expected_train_distances)
            self.assertEqual(validate_df["distance_m"].tolist(), expected_validate_distances)

            for index, image_filename in enumerate(train_df["image_filename"].tolist()):
                self.assertIn(f"f{index:06d}", image_filename)
                self.assertTrue((destination_root / "train" / "images" / Path(image_filename).name).is_file())
            for index, image_filename in enumerate(validate_df["image_filename"].tolist()):
                self.assertIn(f"f{index:06d}", image_filename)
                self.assertTrue((destination_root / "validate" / "images" / Path(image_filename).name).is_file())

            summary = run_preprocessing_for_dataset(
                root,
                result.destination_dataset_reference,
                bootstrap_config=BootstrapCenterTargetConfig(num_workers=2),
                pack_config=PackRoiFcnConfig(canvas_width=128, canvas_height=128, shard_size=2, compress=False, num_workers=2),
            )
            self.assertEqual(summary.dataset_reference, result.destination_dataset_reference)
            self.assertEqual(len(summary.stage_summaries), 4)

    def test_shuffle_rejects_existing_destination_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_preprocessing_root(Path(tmpdir))
            build_dataset(
                root,
                "shuffle-source",
                train_rows=[
                    {
                        "filename": "train_f000000.png",
                        "sample_id": "train_f000000",
                        "width": 64,
                        "height": 64,
                    }
                ],
                validate_rows=[
                    {
                        "filename": "validate_f000000.png",
                        "sample_id": "validate_f000000",
                        "width": 64,
                        "height": 64,
                    }
                ],
            )
            (root / "input" / "existing-destination").mkdir(parents=True, exist_ok=True)

            with self.assertRaises(RoiFcnInputCorporaShuffleValidationError):
                shuffle_input_dataset_corpora(
                    root,
                    "shuffle-source",
                    11,
                    destination_dataset_reference="existing-destination",
                )


if __name__ == "__main__":
    unittest.main()
