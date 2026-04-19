from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from _test_support import build_dataset_reference, ensure_training_root
from roi_fcn_training_v0_1.data import RoiFcnDatasetValidationError, load_and_validate_split_dataset


class DataContractTests(unittest.TestCase):
    def test_valid_split_contract_loads(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            build_dataset_reference(
                root,
                "fixture",
                train_centers=[(20.0, 18.0), (40.0, 24.0)],
                validate_centers=[(22.0, 20.0)],
            )
            split = load_and_validate_split_dataset(root, "fixture", "train")
            self.assertEqual(split.contract.row_count, 2)
            self.assertEqual(split.contract.geometry.canvas_width_px, 96)
            self.assertEqual(split.contract.geometry.canvas_height_px, 64)
            self.assertTrue(split.contract.bootstrap_bbox_available)

    def test_invalid_geometry_fails_loudly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_training_root(Path(tmpdir))
            build_dataset_reference(
                root,
                "fixture",
                train_centers=[(20.0, 18.0)],
                validate_centers=[(22.0, 20.0)],
            )
            samples_path = root / "datasets" / "fixture" / "train" / "manifests" / "samples.csv"
            text = samples_path.read_text(encoding="utf-8")
            samples_path.write_text(text.replace(",20.0,18.0,20.0,18.0,", ",21.0,18.0,20.0,18.0,"), encoding="utf-8")
            with self.assertRaises(RoiFcnDatasetValidationError):
                load_and_validate_split_dataset(root, "fixture", "train")


if __name__ == "__main__":
    unittest.main()
