"""Integration tests for v4 detect -> silhouette -> dual-stream pack pipeline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

from rb_pipeline_v4.config import (
    BrightnessNormalizationConfigV4,
    DetectStageConfigV4,
    PackDualStreamStageConfigV4,
    SilhouetteStageConfigV4,
)
from rb_pipeline_v4.contracts import Detection
from rb_pipeline_v4.detect_stage import run_detect_stage_v4
from rb_pipeline_v4.manifest import load_samples_csv, samples_csv_path
from rb_pipeline_v4.pack_dual_stream_stage import run_pack_dual_stream_stage_v4
from rb_pipeline_v4.silhouette_stage import run_silhouette_stage_v4
from rb_pipeline_v4.validation import validate_dual_stream_npz_file


class _FakeDetector:
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        return [
            Detection(
                class_id=7,
                class_name="defender",
                confidence=0.99,
                x1=float(w * 0.30),
                y1=float(h * 0.30),
                x2=float(w * 0.70),
                y2=float(h * 0.70),
            )
        ]


class _LargeFakeDetector:
    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        return [
            Detection(
                class_id=7,
                class_name="defender",
                confidence=0.99,
                x1=float(w * 0.05),
                y1=float(h * 0.05),
                x2=float(w * 0.95),
                y2=float(h * 0.95),
            )
        ]


class V4PipelineIntegrationTests(unittest.TestCase):
    def test_v4_detect_silhouette_pack_flow(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_v4"
            self._make_fixture(project_root, run_name)

            detect_summary = run_detect_stage_v4(
                project_root,
                run_name,
                DetectStageConfigV4(defender_class_names=("defender",)),
                detector=_FakeDetector(),
            )
            self.assertEqual(detect_summary.successful_rows, 2)
            self.assertEqual(detect_summary.failed_rows, 0)

            silhouette_summary = run_silhouette_stage_v4(
                project_root,
                run_name,
                SilhouetteStageConfigV4(
                    representation_mode="filled",
                    roi_canvas_width_px=64,
                    roi_canvas_height_px=64,
                ),
            )
            self.assertEqual(silhouette_summary.successful_rows, 2)
            self.assertEqual(silhouette_summary.failed_rows, 0)

            pack_summary = run_pack_dual_stream_stage_v4(
                project_root,
                run_name,
                PackDualStreamStageConfigV4(
                    canvas_width_px=64,
                    canvas_height_px=64,
                    include_v1_compat_arrays=True,
                    shard_size=1,
                ),
            )
            self.assertEqual(pack_summary.successful_rows, 2)
            self.assertEqual(pack_summary.failed_rows, 0)

            output_samples = load_samples_csv(
                samples_csv_path(project_root / "training-data-v4" / run_name / "manifests")
            )
            self.assertIn("bbox_feat_area_norm", output_samples.columns)
            self.assertIn("yaw_deg", output_samples.columns)
            self.assertIn("yaw_sin", output_samples.columns)
            self.assertIn("yaw_cos", output_samples.columns)
            self.assertTrue((output_samples["pack_dual_stream_stage_status"] == "success").all())

            npz_paths = sorted((project_root / "training-data-v4" / run_name).glob("*.npz"))
            self.assertEqual(len(npz_paths), 2)
            self.assertEqual(
                [path.name for path in npz_paths],
                [f"{run_name}_shard_00000.npz", f"{run_name}_shard_00001.npz"],
            )
            self.assertEqual(
                output_samples["npz_filename"].astype(str).tolist(),
                [f"{run_name}_shard_00000.npz", f"{run_name}_shard_00001.npz"],
            )
            staged_npy_paths = sorted(
                (project_root / "training-data-v4" / run_name / "arrays").rglob("*.npy")
            )
            self.assertEqual(staged_npy_paths, [])
            validate_dual_stream_npz_file(npz_paths[0], require_v1_compat_arrays=True)

            with np.load(npz_paths[0], allow_pickle=False) as payload:
                self.assertIn("silhouette_crop", payload)
                self.assertIn("bbox_features", payload)
                self.assertIn("y_yaw_deg", payload)
                self.assertIn("y_yaw_sin", payload)
                self.assertIn("y_yaw_cos", payload)
                self.assertEqual(payload["silhouette_crop"].shape[1:], (1, 64, 64))
                self.assertEqual(payload["bbox_features"].shape[1], 10)
                self.assertEqual(payload["y_position_3d"].shape[1], 3)
                self.assertEqual(payload["y_yaw_deg"].shape, (1,))
                self.assertEqual(payload["y_yaw_sin"].shape, (1,))
                self.assertEqual(payload["y_yaw_cos"].shape, (1,))
                yaw_deg = float(payload["y_yaw_deg"][0])
                yaw_sin = float(payload["y_yaw_sin"][0])
                yaw_cos = float(payload["y_yaw_cos"][0])
                self.assertAlmostEqual(yaw_sin, float(np.sin(np.deg2rad(yaw_deg))), places=5)
                self.assertAlmostEqual(yaw_cos, float(np.cos(np.deg2rad(yaw_deg))), places=5)
                packed_u8 = np.rint(np.clip(payload["silhouette_crop"][0, 0], 0.0, 1.0) * 255.0).astype(np.uint8)
                self.assertGreater(len(np.unique(packed_u8)), 2)
                self.assertIn("X", payload)
                self.assertIn("y", payload)

    def test_pack_can_keep_intermediate_npy_when_deletion_disabled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_v4_keep_npy"
            self._make_fixture(project_root, run_name)

            run_detect_stage_v4(
                project_root,
                run_name,
                DetectStageConfigV4(defender_class_names=("defender",)),
                detector=_FakeDetector(),
            )
            run_silhouette_stage_v4(
                project_root,
                run_name,
                SilhouetteStageConfigV4(
                    representation_mode="filled",
                    roi_canvas_width_px=64,
                    roi_canvas_height_px=64,
                ),
            )

            summary = run_pack_dual_stream_stage_v4(
                project_root,
                run_name,
                PackDualStreamStageConfigV4(
                    canvas_width_px=64,
                    canvas_height_px=64,
                    include_v1_compat_arrays=False,
                    shard_size=1,
                    use_intermediate_npy=True,
                    delete_source_npy_after_pack=False,
                ),
            )
            self.assertEqual(summary.successful_rows, 2)
            staged_npy_paths = sorted(
                (project_root / "training-data-v4" / run_name / "arrays").rglob("*.npy")
            )
            self.assertEqual(len(staged_npy_paths), 2)

    def test_pack_applies_masked_median_darkness_gain_when_enabled(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_v4_brightness_norm"
            self._make_fixture(project_root, run_name)

            run_detect_stage_v4(
                project_root,
                run_name,
                DetectStageConfigV4(defender_class_names=("defender",)),
                detector=_FakeDetector(),
            )
            run_silhouette_stage_v4(
                project_root,
                run_name,
                SilhouetteStageConfigV4(
                    representation_mode="filled",
                    roi_canvas_width_px=64,
                    roi_canvas_height_px=64,
                ),
            )

            target_darkness = 0.55
            summary = run_pack_dual_stream_stage_v4(
                project_root,
                run_name,
                PackDualStreamStageConfigV4(
                    canvas_width_px=64,
                    canvas_height_px=64,
                    include_v1_compat_arrays=False,
                    shard_size=0,
                    brightness_normalization=BrightnessNormalizationConfigV4(
                        enabled=True,
                        method="masked_median_darkness_gain",
                        target_median_darkness=target_darkness,
                        min_gain=0.1,
                        max_gain=10.0,
                    ),
                ),
            )

            self.assertEqual(summary.successful_rows, 2)
            output_root = project_root / "training-data-v4" / run_name
            output_samples = load_samples_csv(samples_csv_path(output_root / "manifests"))
            self.assertIn("brightness_normalization_status", output_samples.columns)
            self.assertTrue((output_samples["brightness_normalization_status"] == "success").all())

            run_manifest = json.loads((output_root / "manifests" / "run.json").read_text(encoding="utf-8"))
            brightness_contract = run_manifest["PreprocessingContract"]["Stages"]["pack_dual_stream"][
                "BrightnessNormalization"
            ]
            self.assertTrue(brightness_contract["Enabled"])
            self.assertEqual(brightness_contract["Method"], "masked_median_darkness_gain")
            self.assertAlmostEqual(float(brightness_contract["TargetMedianDarkness"]), target_darkness)

            npz_path = output_root / f"{run_name}.npz"
            with np.load(npz_path, allow_pickle=False) as payload:
                packed = payload["silhouette_crop"][:, 0, :, :]
            foreground = packed < 0.999
            self.assertTrue(bool(np.any(foreground)))
            median_darkness = float(np.median((1.0 - packed)[foreground]))
            self.assertAlmostEqual(median_darkness, target_darkness, places=5)
            np.testing.assert_array_equal(packed[~foreground], np.ones_like(packed[~foreground]))

    def test_pack_fails_when_canvas_too_small_with_fail_policy(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_v4_small_canvas"
            self._make_fixture(project_root, run_name)

            run_detect_stage_v4(
                project_root,
                run_name,
                DetectStageConfigV4(defender_class_names=("defender",)),
                detector=_LargeFakeDetector(),
            )
            run_silhouette_stage_v4(
                project_root,
                run_name,
                SilhouetteStageConfigV4(
                    representation_mode="filled",
                    roi_canvas_width_px=64,
                    roi_canvas_height_px=64,
                ),
            )

            summary = run_pack_dual_stream_stage_v4(
                project_root,
                run_name,
                PackDualStreamStageConfigV4(
                    canvas_width_px=8,
                    canvas_height_px=8,
                    clip_policy="fail",
                    include_v1_compat_arrays=False,
                    shard_size=0,
                ),
            )

            self.assertEqual(summary.failed_rows, 2)
            output_samples = load_samples_csv(
                samples_csv_path(project_root / "training-data-v4" / run_name / "manifests")
            )
            self.assertTrue((output_samples["pack_dual_stream_stage_status"] == "failed").all())

    def _make_fixture(self, project_root: Path, run_name: str) -> None:
        (project_root / "rb_pipeline_v4").mkdir(parents=True, exist_ok=True)
        (project_root / "rb_ui_v4").mkdir(parents=True, exist_ok=True)

        run_root = project_root / "input-images" / run_name
        images_dir = run_root / "images"
        manifests_dir = run_root / "manifests"
        images_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []
        for idx in range(2):
            image = np.full((64, 64), 255, dtype=np.uint8)
            cv2.rectangle(image, (18, 18), (46, 46), color=170, thickness=-1)
            cv2.circle(image, (32, 32), 9, color=90, thickness=-1)
            cv2.line(image, (20, 44), (44, 20), color=30, thickness=2)

            filename = f"frame_{idx:03d}.png"
            path = images_dir / filename
            ok = cv2.imwrite(str(path), image)
            if not ok:
                raise RuntimeError(f"Failed to write fixture image: {path}")

            rows.append(
                {
                    "run_id": run_name,
                    "sample_id": f"{run_name}_sample_{idx:03d}",
                    "frame_index": idx,
                    "image_filename": filename,
                    "distance_m": float(idx + 1),
                    "image_width_px": 64,
                    "image_height_px": 64,
                    "capture_success": True,
                    "final_pos_x_m": float(idx) * 0.1,
                    "final_pos_y_m": 0.5,
                    "final_pos_z_m": 4.0 + float(idx) * 0.2,
                    "final_rot_y_deg": 170.0 + float(idx) * 7.5,
                }
            )

        pd.DataFrame(rows).to_csv(manifests_dir / "samples.csv", index=False)
        (manifests_dir / "run.json").write_text(
            json.dumps({"RunId": run_name}, indent=4) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
