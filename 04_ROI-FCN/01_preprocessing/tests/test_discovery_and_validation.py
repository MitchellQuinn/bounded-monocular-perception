"""Discovery and validation tests for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from _test_support import build_dataset, build_input_split, ensure_preprocessing_root
from roi_fcn_preprocessing_v0_1.discovery import discover_dataset_references
from roi_fcn_preprocessing_v0_1.validation import (
    RoiFcnPreprocessingValidationError,
    ensure_valid_input_dataset_reference,
    validate_input_dataset_reference,
)


class DiscoveryValidationTests(unittest.TestCase):
    def test_discovery_only_lists_valid_train_and_validate_datasets(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_preprocessing_root(Path(tmpdir))
            build_dataset(
                root,
                "valid-dataset",
                train_rows=[{"width": 64, "height": 64}],
                validate_rows=[{"width": 64, "height": 64}],
            )
            build_input_split(root, "missing-validate", "train", [{"width": 64, "height": 64}])
            build_input_split(root, "broken-dataset", "train", [{"width": 64, "height": 64}])
            bad_validate_root = root / "input" / "broken-dataset" / "validate"
            (bad_validate_root / "images").mkdir(parents=True, exist_ok=True)
            (bad_validate_root / "manifests").mkdir(parents=True, exist_ok=True)
            (bad_validate_root / "manifests" / "run.json").write_text("{}\n", encoding="utf-8")

            discovered = discover_dataset_references(root)

            self.assertEqual([dataset.name for dataset in discovered], ["valid-dataset"])

    def test_malformed_split_structure_fails_loudly(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_preprocessing_root(Path(tmpdir))
            build_input_split(root, "bad-dataset", "train", [{"width": 64, "height": 64}])
            validate_root = root / "input" / "bad-dataset" / "validate"
            (validate_root / "images").mkdir(parents=True, exist_ok=True)
            (validate_root / "manifests").mkdir(parents=True, exist_ok=True)
            (validate_root / "manifests" / "run.json").write_text("{}\n", encoding="utf-8")

            errors = validate_input_dataset_reference(root, "bad-dataset")
            self.assertTrue(any("validate: Missing samples.csv" in error for error in errors))

            with self.assertRaises(RoiFcnPreprocessingValidationError):
                ensure_valid_input_dataset_reference(root, "bad-dataset")


if __name__ == "__main__":
    unittest.main()
