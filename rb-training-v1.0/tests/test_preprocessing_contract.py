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

    def test_resolve_preprocessing_contract_accepts_shared_v2_contract(self) -> None:
        contract = {
            "ContractVersion": "rb-preprocess-v2",
            "CurrentRepresentation": {
                "StorageFormat": "npz",
                "ArrayKey": "X",
                "Kind": "silhouette_filled_array",
                "RepresentationMode": "filled",
                "ArrayDType": "float32",
                "Normalize": True,
                "Invert": True,
            },
        }
        records = [
            {
                "source_root": "training",
                "dataset_id": "train-v2-a",
                "relative_run_json_path": "training-data-v2/train-v2-a/manifests/run.json",
                "has_preprocessing_contract": True,
                "preprocessing_contract": contract,
            },
            {
                "source_root": "validation",
                "dataset_id": "val-v2-a",
                "relative_run_json_path": "validation-data-v2/val-v2-a/manifests/run.json",
                "has_preprocessing_contract": True,
                "preprocessing_contract": contract,
            },
        ]

        resolved, warnings = _resolve_preprocessing_contract(records)

        self.assertEqual(resolved, contract)
        self.assertEqual(warnings, [])
        self.assertIn("silhouette_filled_array", _describe_input_representation(resolved))

    def test_resolve_preprocessing_contract_rejects_conflicting_versions(self) -> None:
        records = [
            {
                "source_root": "training",
                "dataset_id": "train-v1",
                "relative_run_json_path": "training-data/train-v1/manifests/run.json",
                "has_preprocessing_contract": True,
                "preprocessing_contract": {
                    "ContractVersion": "rb-preprocess-v1",
                    "CurrentRepresentation": {"StorageFormat": "npz", "ArrayKey": "X"},
                },
            },
            {
                "source_root": "validation",
                "dataset_id": "val-v2",
                "relative_run_json_path": "validation-data-v2/val-v2/manifests/run.json",
                "has_preprocessing_contract": True,
                "preprocessing_contract": {
                    "ContractVersion": "rb-preprocess-v2",
                    "CurrentRepresentation": {"StorageFormat": "npz", "ArrayKey": "X"},
                },
            },
        ]

        with self.assertRaisesRegex(ValueError, "conflicting PreprocessingContract"):
            _resolve_preprocessing_contract(records)


if __name__ == "__main__":
    unittest.main()
