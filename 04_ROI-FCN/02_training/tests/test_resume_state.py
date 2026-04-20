from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from roi_fcn_training_v0_1.resume_state import build_resume_state_payload, load_resume_state, save_resume_state


class ResumeStateTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "resume_state.pt"
            payload = build_resume_state_payload(
                epoch=3,
                run_id="run_0002",
                training_dataset="fixture_train",
                validation_dataset="fixture_validate",
                topology_id="roi_fcn_tiny",
                topology_variant="tiny_v1",
                topology_params={"channels": 1},
                topology_spec_signature="spec-signature",
                topology_contract_signature="contract-signature",
                output_hw=(16, 24),
                train_split_contract={
                    "dataset_reference": "fixture_train",
                    "split_name": "train",
                    "row_count": 10,
                    "shard_count": 1,
                    "geometry": {
                        "canvas_width_px": 48,
                        "canvas_height_px": 32,
                        "image_layout": "N,C,H,W",
                        "channels": 1,
                        "normalization_range": [0.0, 1.0],
                        "geometry_schema": ["schema"],
                    },
                    "preprocessing_contract_version": "rb-preprocess-roi-fcn-v0_1",
                    "representation_kind": "roi_fcn_locator_npz",
                    "representation_storage_format": "npz",
                    "representation_array_keys": ["locator_input_image"],
                    "bootstrap_bbox_available": True,
                    "fixed_roi_width_px": 300,
                    "fixed_roi_height_px": 300,
                },
                validation_split_contract={
                    "dataset_reference": "fixture_validate",
                    "split_name": "validate",
                    "row_count": 4,
                    "shard_count": 1,
                    "geometry": {
                        "canvas_width_px": 48,
                        "canvas_height_px": 32,
                        "image_layout": "N,C,H,W",
                        "channels": 1,
                        "normalization_range": [0.0, 1.0],
                        "geometry_schema": ["schema"],
                    },
                    "preprocessing_contract_version": "rb-preprocess-roi-fcn-v0_1",
                    "representation_kind": "roi_fcn_locator_npz",
                    "representation_storage_format": "npz",
                    "representation_array_keys": ["locator_input_image"],
                    "bootstrap_bbox_available": True,
                    "fixed_roi_width_px": 300,
                    "fixed_roi_height_px": 300,
                },
                best_epoch=2,
                best_validation_loss=0.5,
                best_validation_mean_center_error_px=7.0,
                epochs_without_improvement=1,
                history_rows=[{"epoch": 1, "validation_loss": 1.0}],
                model_state_dict={},
                optimizer_state_dict={},
            )
            save_resume_state(state_path, payload)
            loaded = load_resume_state(state_path)
            self.assertEqual(int(loaded["epoch"]), 3)
            self.assertEqual(str(loaded["run_id"]), "run_0002")
            self.assertEqual([int(value) for value in loaded["output_hw"]], [16, 24])

    def test_load_rejects_missing_required_keys(self) -> None:
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "resume_state.pt"
            save_resume_state(state_path, {"epoch": 1})
            with self.assertRaisesRegex(ValueError, "missing required keys"):
                load_resume_state(state_path)


if __name__ == "__main__":
    unittest.main()
