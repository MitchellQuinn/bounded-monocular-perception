"""End-to-end integration tests for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from _test_support import build_dataset, ensure_preprocessing_root
from roi_fcn_preprocessing_v0_1.config import BootstrapCenterTargetConfig, PackRoiFcnConfig
from roi_fcn_preprocessing_v0_1.manifest import load_run_json, load_samples_csv
from roi_fcn_preprocessing_v0_1.pipeline import run_preprocessing_for_dataset
from roi_fcn_preprocessing_v0_1.validation import validate_roi_fcn_npz_file


class PipelineIntegrationTests(unittest.TestCase):
    def test_end_to_end_dataset_runs_train_then_validate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_preprocessing_root(Path(tmpdir))
            build_dataset(
                root,
                "integration-fixture",
                train_rows=[
                    {"width": 96, "height": 64, "box_xyxy": (20, 10, 76, 54)},
                    {"width": 96, "height": 64, "box_xyxy": (24, 12, 72, 52)},
                ],
                validate_rows=[
                    {"width": 96, "height": 64, "box_xyxy": (20, 10, 76, 54)},
                ],
            )

            summary = run_preprocessing_for_dataset(
                root,
                "integration-fixture",
                bootstrap_config=BootstrapCenterTargetConfig(num_workers=2),
                pack_config=PackRoiFcnConfig(canvas_width=128, canvas_height=128, shard_size=1, compress=False, num_workers=2),
            )

            self.assertEqual(
                [(item.split_name, item.stage_name) for item in summary.stage_summaries],
                [
                    ("train", "bootstrap_center_target"),
                    ("train", "pack_roi_fcn"),
                    ("validate", "bootstrap_center_target"),
                    ("validate", "pack_roi_fcn"),
                ],
            )

            train_npz_paths = sorted((root / "output" / "integration-fixture" / "train" / "arrays").glob("*.npz"))
            validate_npz_paths = sorted((root / "output" / "integration-fixture" / "validate" / "arrays").glob("*.npz"))
            self.assertEqual(
                [path.name for path in train_npz_paths],
                [
                    "integration-fixture__train__shard_0000.npz",
                    "integration-fixture__train__shard_0001.npz",
                ],
            )
            self.assertEqual(
                [path.name for path in validate_npz_paths],
                ["integration-fixture__validate__shard_0000.npz"],
            )
            validate_roi_fcn_npz_file(train_npz_paths[0], expected_canvas_height=128, expected_canvas_width=128)

            train_samples = load_samples_csv(root / "output" / "integration-fixture" / "train" / "manifests" / "samples.csv")
            self.assertTrue((train_samples["bootstrap_center_target_stage_status"] == "success").all())
            self.assertTrue((train_samples["pack_roi_fcn_stage_status"] == "success").all())

            run_json = load_run_json(root / "output" / "integration-fixture" / "validate" / "manifests")
            contract = run_json["PreprocessingContract"]
            self.assertEqual(contract["ContractVersion"], "rb-preprocess-roi-fcn-v0_1")
            self.assertEqual(contract["CompletedStages"], ["bootstrap_center_target", "pack_roi_fcn"])
            self.assertEqual(contract["CurrentRepresentation"]["Kind"], "roi_fcn_locator_npz")
            self.assertEqual(contract["StageSummaries"]["pack_roi_fcn"]["SucceededRows"], 1)

    def test_train_failure_aborts_before_validate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_preprocessing_root(Path(tmpdir))
            build_dataset(
                root,
                "abort-fixture",
                train_rows=[
                    {
                        "width": 64,
                        "height": 64,
                        "corrupt_image": True,
                    },
                ],
                validate_rows=[
                    {"width": 64, "height": 64, "box_xyxy": (16, 16, 48, 48)},
                ],
            )

            with self.assertRaises(RuntimeError):
                run_preprocessing_for_dataset(
                    root,
                    "abort-fixture",
                    bootstrap_config=BootstrapCenterTargetConfig(continue_on_error=False),
                    pack_config=PackRoiFcnConfig(compress=False),
                )

            self.assertFalse((root / "output" / "abort-fixture" / "validate").exists())


if __name__ == "__main__":
    unittest.main()
