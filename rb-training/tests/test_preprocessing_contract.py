"""Tests for training-side preprocessing contract handling."""

from __future__ import annotations

import unittest

from src.train import _describe_input_representation, _resolve_preprocessing_contract


class TrainingPreprocessingContractTests(unittest.TestCase):
    def test_resolve_preprocessing_contract_accepts_shared_contract(self) -> None:
        contract = {
            "ContractVersion": "rb-preprocess-v1",
            "CurrentRepresentation": {
                "StorageFormat": "npz",
                "ArrayKey": "X",
                "Kind": "full_frame_bbox_array",
                "ArrayDType": "float32",
                "Normalize": True,
                "Invert": True,
            },
        }
        records = [
            {
                "source_root": "training",
                "dataset_id": "train-a",
                "relative_run_json_path": "training-data/train-a/manifests/run.json",
                "has_preprocessing_contract": True,
                "preprocessing_contract": contract,
            },
            {
                "source_root": "validation",
                "dataset_id": "val-a",
                "relative_run_json_path": "validation-data/val-a/manifests/run.json",
                "has_preprocessing_contract": True,
                "preprocessing_contract": contract,
            },
        ]

        resolved, warnings = _resolve_preprocessing_contract(records)

        self.assertEqual(resolved, contract)
        self.assertEqual(warnings, [])
        self.assertIn("npz key X", _describe_input_representation(resolved))

    def test_resolve_preprocessing_contract_rejects_mixed_presence(self) -> None:
        records = [
            {
                "source_root": "training",
                "dataset_id": "train-a",
                "relative_run_json_path": "training-data/train-a/manifests/run.json",
                "has_preprocessing_contract": True,
                "preprocessing_contract": {"ContractVersion": "rb-preprocess-v1"},
            },
            {
                "source_root": "validation",
                "dataset_id": "val-a",
                "relative_run_json_path": "validation-data/val-a/manifests/run.json",
                "has_preprocessing_contract": False,
                "preprocessing_contract": None,
            },
        ]

        with self.assertRaisesRegex(ValueError, "mix run.json files with and without"):
            _resolve_preprocessing_contract(records)


if __name__ == "__main__":
    unittest.main()
