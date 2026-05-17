"""Stage-level tests for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from _test_support import build_dataset, ensure_preprocessing_root
from roi_fcn_preprocessing_v0_1.bootstrap_center_target_stage import run_bootstrap_center_target_stage
from roi_fcn_preprocessing_v0_1.config import BootstrapCenterTargetConfig, PackRoiFcnConfig
from roi_fcn_preprocessing_v0_1.edge_roi_adapter import build_detect_stage_config, build_edge_roi_detector
from roi_fcn_preprocessing_v0_1.manifest import load_run_json, load_samples_csv
from roi_fcn_preprocessing_v0_1.pack_roi_fcn_stage import run_pack_roi_fcn_stage
from roi_fcn_preprocessing_v0_1.paths import resolve_input_image_path, resolve_split_paths
from roi_fcn_preprocessing_v0_1.validation import validate_roi_fcn_npz_file

from rb_pipeline_v4.detector import EdgeRoiDetector
from rb_pipeline_v4.image_io import read_image_unchanged, to_bgr_uint8


class BootstrapAndPackTests(unittest.TestCase):
    def test_edge_roi_adapter_preserves_parameter_meanings(self) -> None:
        config = BootstrapCenterTargetConfig(
            edge_blur_k=4,
            edge_low=10,
            edge_high=240,
            fg_threshold=180,
            edge_pad=7,
            edge_ignore_border_px=11,
            min_edge_pixels=22,
            edge_close_kernel_size=3,
        )
        detect_config = build_detect_stage_config(config)
        self.assertEqual(detect_config.detector_backend, "edge")
        self.assertEqual(detect_config.normalized_edge_blur_kernel_size(), 5)
        self.assertEqual(detect_config.normalized_edge_canny_low_threshold(), 10)
        self.assertEqual(detect_config.normalized_edge_canny_high_threshold(), 240)
        self.assertEqual(detect_config.normalized_edge_foreground_threshold(), 180)
        self.assertEqual(detect_config.normalized_edge_padding_px(), 7)
        self.assertEqual(detect_config.normalized_edge_ignore_border_px(), 11)
        self.assertEqual(detect_config.normalized_edge_min_foreground_px(), 22)
        self.assertEqual(detect_config.normalized_edge_close_kernel_size(), 3)

        with TemporaryDirectory() as tmpdir:
            root = ensure_preprocessing_root(Path(tmpdir))
            build_dataset(
                root,
                "adapter-fixture",
                train_rows=[{"width": 96, "height": 64, "box_xyxy": (20, 10, 76, 54)}],
                validate_rows=[{"width": 96, "height": 64, "box_xyxy": (20, 10, 76, 54)}],
            )
            split_paths = resolve_split_paths(root, "adapter-fixture", "train")
            image_path = resolve_input_image_path(split_paths, "train_000.png")
            image_bgr = to_bgr_uint8(read_image_unchanged(image_path))

            adapter_detector = build_edge_roi_detector(config)
            self.assertEqual(adapter_detector.ignore_border_px, 11)
            adapter_detection = adapter_detector.detect(image_bgr)[0]
            direct_detection = EdgeRoiDetector(
                blur_kernel_size=5,
                canny_low_threshold=10,
                canny_high_threshold=240,
                foreground_threshold=180,
                padding_px=7,
                ignore_border_px=11,
                min_foreground_px=22,
                close_kernel_size=3,
                class_id=0,
                class_name="defender",
            ).detect(image_bgr)[0]

            self.assertAlmostEqual(adapter_detection.x1, direct_detection.x1)
            self.assertAlmostEqual(adapter_detection.y1, direct_detection.y1)
            self.assertAlmostEqual(adapter_detection.x2, direct_detection.x2)
            self.assertAlmostEqual(adapter_detection.y2, direct_detection.y2)
            self.assertAlmostEqual(adapter_detection.center_x_px, direct_detection.center_x_px)
            self.assertAlmostEqual(adapter_detection.center_y_px, direct_detection.center_y_px)

    def test_bootstrap_center_target_writes_expected_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_preprocessing_root(Path(tmpdir))
            build_dataset(
                root,
                "bootstrap-fixture",
                train_rows=[
                    {"width": 80, "height": 60, "box_xyxy": (18, 12, 62, 48)},
                    {"width": 80, "height": 60, "capture_success": False},
                ],
                validate_rows=[{"width": 80, "height": 60, "box_xyxy": (18, 12, 62, 48)}],
            )
            split_paths = resolve_split_paths(root, "bootstrap-fixture", "train")
            summary = run_bootstrap_center_target_stage(split_paths, BootstrapCenterTargetConfig(num_workers=2))

            self.assertEqual(summary.successful_rows, 1)
            self.assertEqual(summary.skipped_rows, 1)
            samples_df = load_samples_csv(split_paths.output_samples_csv_path)
            success_row = samples_df.iloc[0]

            image_path = resolve_input_image_path(split_paths, success_row["image_filename"])
            expected_detection = build_edge_roi_detector(BootstrapCenterTargetConfig(num_workers=2)).detect(
                to_bgr_uint8(read_image_unchanged(image_path))
            )[0]

            self.assertEqual(success_row["bootstrap_center_target_stage_status"], "success")
            self.assertEqual(success_row["bootstrap_target_algorithm"], "edge_roi_v1")
            self.assertAlmostEqual(float(success_row["bootstrap_center_x_px"]), float(expected_detection.center_x_px))
            self.assertAlmostEqual(float(success_row["bootstrap_center_y_px"]), float(expected_detection.center_y_px))
            self.assertAlmostEqual(float(success_row["bootstrap_bbox_x1"]), float(expected_detection.x1))
            self.assertAlmostEqual(float(success_row["bootstrap_bbox_y1"]), float(expected_detection.y1))

            run_json = load_run_json(split_paths.output_manifests_dir)
            contract = run_json["PreprocessingContract"]
            self.assertEqual(contract["ContractVersion"], "rb-preprocess-roi-fcn-v0_1")
            self.assertEqual(contract["CurrentStage"], "bootstrap_center_target")
            self.assertEqual(contract["Stages"]["bootstrap_center_target"]["EdgeIgnoreBorderPx"], 8)
            self.assertEqual(contract["StageSummaries"]["bootstrap_center_target"]["SucceededRows"], 1)
            self.assertEqual(contract["StageSummaries"]["bootstrap_center_target"]["SkippedRows"], 1)

    def test_pack_roi_fcn_preserves_geometry_and_npz_contract(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = ensure_preprocessing_root(Path(tmpdir))
            build_dataset(
                root,
                "pack-fixture",
                train_rows=[{"width": 100, "height": 50, "box_xyxy": (40, 10, 90, 40)}],
                validate_rows=[{"width": 100, "height": 50, "box_xyxy": (40, 10, 90, 40)}],
            )
            split_paths = resolve_split_paths(root, "pack-fixture", "train")
            run_bootstrap_center_target_stage(split_paths, BootstrapCenterTargetConfig(num_workers=2))
            summary = run_pack_roi_fcn_stage(
                split_paths,
                PackRoiFcnConfig(
                    canvas_width=300,
                    canvas_height=300,
                    fixed_roi_crop_width_px=320,
                    fixed_roi_crop_height_px=280,
                    shard_size=0,
                    compress=False,
                    num_workers=2,
                ),
            )

            self.assertEqual(summary.successful_rows, 1)
            samples_df = load_samples_csv(split_paths.output_samples_csv_path)
            row = samples_df.iloc[0]
            scale = float(row["locator_resize_scale"])
            pad_left = int(row["locator_pad_left_px"])
            pad_top = int(row["locator_pad_top_px"])
            expected_center_x = float(row["bootstrap_center_x_px"]) * scale + pad_left
            expected_center_y = float(row["bootstrap_center_y_px"]) * scale + pad_top

            self.assertEqual(row["pack_roi_fcn_stage_status"], "success")
            self.assertEqual(int(row["locator_resized_width_px"]), 300)
            self.assertEqual(int(row["locator_resized_height_px"]), 150)
            self.assertEqual(int(row["locator_pad_top_px"]), 75)
            self.assertEqual(int(row["locator_pad_bottom_px"]), 75)
            self.assertAlmostEqual(float(row["locator_center_x_px"]), expected_center_x)
            self.assertAlmostEqual(float(row["locator_center_y_px"]), expected_center_y)
            self.assertEqual(row["npz_filename"], "pack-fixture__train.npz")
            self.assertEqual(int(row["npz_row_index"]), 0)

            npz_path = split_paths.output_arrays_dir / "pack-fixture__train.npz"
            validate_roi_fcn_npz_file(npz_path, expected_canvas_height=300, expected_canvas_width=300)
            with np.load(npz_path, allow_pickle=False) as payload:
                self.assertEqual(payload["locator_input_image"].shape, (1, 1, 300, 300))
                self.assertAlmostEqual(float(payload["target_center_xy_canvas_px"][0, 0]), expected_center_x)
                self.assertAlmostEqual(float(payload["target_center_xy_canvas_px"][0, 1]), expected_center_y)
                self.assertEqual(payload["source_image_wh_px"].tolist(), [[100, 50]])
                self.assertEqual(payload["resized_image_wh_px"].tolist(), [[300, 150]])
                self.assertEqual(payload["padding_ltrb_px"].tolist(), [[0, 75, 0, 75]])

            run_json = load_run_json(split_paths.output_manifests_dir)
            pack_stage = run_json["PreprocessingContract"]["Stages"]["pack_roi_fcn"]
            representation = run_json["PreprocessingContract"]["CurrentRepresentation"]
            self.assertEqual(pack_stage["FixedROICropWidthPx"], 320)
            self.assertEqual(pack_stage["FixedROICropHeightPx"], 280)
            self.assertEqual(representation["FixedROICropWidthPx"], 320)
            self.assertEqual(representation["FixedROICropHeightPx"], 280)


if __name__ == "__main__":
    unittest.main()
