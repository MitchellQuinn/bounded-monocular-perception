from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from roi_fcn_training_v0_1.contracts import HISTORY_FILENAME
from roi_fcn_training_v0_1.epoch_summary import read_epoch_summary_panel
from roi_fcn_training_v0_1.resume_state import build_resume_state_payload, save_resume_state


class EpochSummaryTests(unittest.TestCase):
    def test_reads_history_json_and_selects_best_epoch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            history_rows = [
                {
                    "epoch": 1,
                    "train_loss": 0.40,
                    "validation_loss": 0.50,
                    "validation_mean_center_error_px": 6.0,
                },
                {
                    "epoch": 2,
                    "train_loss": 0.30,
                    "validation_loss": 0.70,
                    "validation_mean_center_error_px": 5.0,
                },
                {
                    "epoch": 3,
                    "train_loss": 0.25,
                    "validation_loss": 0.60,
                    "validation_mean_center_error_px": 5.0,
                    "validation_p95_center_error_px": 8.5,
                },
                {
                    "epoch": 4,
                    "train_loss": 0.20,
                    "validation_loss": 0.80,
                    "validation_mean_center_error_px": 7.0,
                },
            ]
            (run_dir / HISTORY_FILENAME).write_text(json.dumps(history_rows), encoding="utf-8")

            panel = read_epoch_summary_panel(run_dir)

        self.assertEqual(panel.latest_epoch, 4)
        self.assertEqual(panel.best_epoch, 3)
        self.assertIn("Best selected by: validation_mean_center_error_px", panel.text)
        self.assertIn("Latest completed", panel.text)
        self.assertIn("Best so far", panel.text)
        self.assertIn("validation_p95_center_error_px=8.5000", panel.text)

    def test_falls_back_to_resume_state_history_rows(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            payload = build_resume_state_payload(
                epoch=2,
                run_id="run_0001",
                training_dataset="fixture_train",
                validation_dataset="fixture_validate",
                topology_id="roi_fcn_tiny",
                topology_variant="tiny_v1",
                topology_params={},
                topology_spec_signature="spec",
                topology_contract_signature="contract",
                output_hw=(16, 16),
                train_split_contract={},
                validation_split_contract={},
                best_epoch=2,
                best_validation_loss=0.25,
                best_validation_mean_center_error_px=4.0,
                epochs_without_improvement=0,
                history_rows=[
                    {
                        "epoch": 1,
                        "validation_loss": 0.50,
                        "validation_mean_center_error_px": 5.0,
                    },
                    {
                        "epoch": 2,
                        "validation_loss": 0.25,
                        "validation_mean_center_error_px": 4.0,
                    },
                ],
                model_state_dict={},
                optimizer_state_dict={},
            )
            save_resume_state(run_dir / "resume_state.pt", payload)

            panel = read_epoch_summary_panel(run_dir)

        self.assertEqual(panel.latest_epoch, 2)
        self.assertEqual(panel.best_epoch, 2)
        self.assertIn("validation_loss=0.2500", panel.text)


if __name__ == "__main__":
    unittest.main()
