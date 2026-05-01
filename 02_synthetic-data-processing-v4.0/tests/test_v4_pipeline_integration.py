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
    PackTriStreamStageConfigV4,
    SilhouetteStageConfigV4,
)
from rb_pipeline_v4.contracts import Detection
from rb_pipeline_v4.detect_stage import run_detect_stage_v4
from rb_pipeline_v4.manifest import load_samples_csv, samples_csv_path
from rb_pipeline_v4.pack_dual_stream_stage import run_pack_dual_stream_stage_v4
from rb_pipeline_v4.pack_tri_stream_stage import run_pack_tri_stream_stage_v4
from rb_pipeline_v4.silhouette_stage import run_silhouette_stage_v4
from rb_pipeline_v4.tri_stream_control import (
    build_pack_tri_stream_config,
    infer_tri_stream_run_canvas_size,
    preview_tri_stream_sample,
)
from rb_pipeline_v4.validation import (
    PipelineValidationError,
    validate_dual_stream_npz_file,
    validate_tri_stream_npz_file,
)


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

    def test_pack_tri_stream_writes_distance_orientation_geometry_and_contract(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_v4_tri"
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

            summary = run_pack_tri_stream_stage_v4(
                project_root,
                run_name,
                PackTriStreamStageConfigV4(
                    canvas_width_px=64,
                    canvas_height_px=64,
                    shard_size=1,
                ),
            )

            self.assertEqual(summary.successful_rows, 2)
            self.assertEqual(summary.failed_rows, 0)
            output_root = project_root / "training-data-v4-tri-stream" / run_name
            output_samples = load_samples_csv(samples_csv_path(output_root / "manifests"))
            self.assertIn("pack_tri_stream_stage_status", output_samples.columns)
            self.assertNotIn("pack_dual_stream_stage_status", output_samples.columns)
            self.assertTrue((output_samples["pack_tri_stream_stage_status"] == "success").all())
            self.assertEqual(
                output_samples["npz_filename"].astype(str).tolist(),
                [f"{run_name}_shard_00000.npz", f"{run_name}_shard_00001.npz"],
            )

            run_manifest = json.loads((output_root / "manifests" / "run.json").read_text(encoding="utf-8"))
            contract = run_manifest["PreprocessingContract"]
            self.assertEqual(contract["ContractVersion"], "rb-preprocess-v4-tri-stream-orientation-v1")
            self.assertEqual(contract["CurrentStage"], "pack_tri_stream")
            self.assertEqual(contract["CurrentRepresentation"]["Kind"], "tri_stream_npz")
            self.assertIn("x_distance_image", contract["CurrentRepresentation"]["ArrayKeys"])
            self.assertIn("x_orientation_image", contract["CurrentRepresentation"]["ArrayKeys"])
            self.assertIn("x_geometry", contract["CurrentRepresentation"]["ArrayKeys"])
            self.assertEqual(
                contract["CurrentRepresentation"]["DistanceImageGeometry"],
                "fixed_unscaled_roi_canvas",
            )
            self.assertEqual(
                contract["CurrentRepresentation"]["OrientationImageGeometry"],
                "target_centered_scaled_by_silhouette_extent",
            )

            npz_path = output_root / f"{run_name}_shard_00000.npz"
            validate_tri_stream_npz_file(npz_path)
            with np.load(npz_path, allow_pickle=False) as payload:
                distance = payload["x_distance_image"]
                orientation = payload["x_orientation_image"]
                geometry = payload["x_geometry"]
                self.assertEqual(distance.shape, orientation.shape)
                self.assertEqual(distance.shape[1:], (1, 64, 64))
                self.assertEqual(str(distance.dtype), "float32")
                self.assertEqual(str(orientation.dtype), "float32")
                self.assertEqual(str(geometry.dtype), "float32")
                self.assertEqual(geometry.shape[1], 10)
                self.assertIn("x_geometry_schema", payload)
                self.assertFalse(np.allclose(distance, orientation))

                distance_extent = self._nonwhite_extent(distance[0, 0])
                orientation_extent = self._nonwhite_extent(orientation[0, 0])
                self.assertGreater(orientation_extent[2] - orientation_extent[0], distance_extent[2] - distance_extent[0])
                self.assertGreater(orientation_extent[3] - orientation_extent[1], distance_extent[3] - distance_extent[1])

    def test_pack_tri_stream_brightness_changes_distance_not_orientation(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_v4_tri_brightness"
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

            base_summary = run_pack_tri_stream_stage_v4(
                project_root,
                run_name,
                PackTriStreamStageConfigV4(
                    canvas_width_px=64,
                    canvas_height_px=64,
                    shard_size=0,
                    overwrite=True,
                ),
            )
            self.assertEqual(base_summary.successful_rows, 2)
            output_root = project_root / "training-data-v4-tri-stream" / run_name
            with np.load(output_root / f"{run_name}.npz", allow_pickle=False) as payload:
                base_distance = payload["x_distance_image"].copy()
                base_orientation = payload["x_orientation_image"].copy()

            norm_summary = run_pack_tri_stream_stage_v4(
                project_root,
                run_name,
                PackTriStreamStageConfigV4(
                    canvas_width_px=64,
                    canvas_height_px=64,
                    shard_size=0,
                    overwrite=True,
                    brightness_normalization=BrightnessNormalizationConfigV4(
                        enabled=True,
                        method="masked_median_darkness_gain",
                        target_median_darkness=0.65,
                        min_gain=0.1,
                        max_gain=10.0,
                    ),
                ),
            )
            self.assertEqual(norm_summary.successful_rows, 2)
            output_samples = load_samples_csv(samples_csv_path(output_root / "manifests"))
            self.assertTrue((output_samples["brightness_normalization_status"] == "success").all())
            with np.load(output_root / f"{run_name}.npz", allow_pickle=False) as payload:
                norm_distance = payload["x_distance_image"]
                norm_orientation = payload["x_orientation_image"]

            self.assertFalse(np.allclose(base_distance, norm_distance))
            np.testing.assert_array_equal(base_orientation, norm_orientation)

    def test_tri_stream_preview_uses_run_canvas_and_rejects_mismatch(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_v4_tri_canvas"
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

            self.assertEqual(infer_tri_stream_run_canvas_size(project_root, run_name), (64, 64))
            preview = preview_tri_stream_sample(
                project_root,
                run_name,
                build_pack_tri_stream_config(canvas_width_px=64, canvas_height_px=64),
                row_index=0,
            )
            self.assertEqual(preview.arrays["x_distance_image"].shape, (64, 64))

            mismatch_config = PackTriStreamStageConfigV4(
                canvas_width_px=96,
                canvas_height_px=96,
                shard_size=0,
            )
            with self.assertRaisesRegex(ValueError, "must match silhouette ROI canvas"):
                preview_tri_stream_sample(
                    project_root,
                    run_name,
                    mismatch_config,
                    row_index=0,
                )
            with self.assertRaisesRegex(PipelineValidationError, "must match silhouette ROI canvas"):
                run_pack_tri_stream_stage_v4(
                    project_root,
                    run_name,
                    mismatch_config,
                )

    def test_validate_tri_stream_rejects_missing_orientation_key(self) -> None:
        with TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "missing_orientation.npz"
            yaw_deg = np.asarray([15.0], dtype=np.float32)
            np.savez(
                npz_path,
                x_distance_image=np.ones((1, 1, 8, 8), dtype=np.float32),
                x_geometry=np.ones((1, 10), dtype=np.float32),
                y_distance_m=np.asarray([1.0], dtype=np.float32),
                y_yaw_deg=yaw_deg,
                y_yaw_sin=np.sin(np.deg2rad(yaw_deg)).astype(np.float32),
                y_yaw_cos=np.cos(np.deg2rad(yaw_deg)).astype(np.float32),
                sample_id=np.asarray(["s0"]),
                image_filename=np.asarray(["frame.png"]),
                npz_row_index=np.asarray([0], dtype=np.int64),
            )

            with self.assertRaisesRegex(ValueError, "x_orientation_image"):
                validate_tri_stream_npz_file(npz_path)

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

    def _nonwhite_extent(self, image: np.ndarray) -> tuple[int, int, int, int]:
        mask = np.asarray(image) < 0.999
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

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
