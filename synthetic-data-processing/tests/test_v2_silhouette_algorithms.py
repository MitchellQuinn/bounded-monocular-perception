"""Unit tests for pass-1 v2 silhouette generation behavior."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd

from rb_pipeline_v2.algorithms import register_default_components
from rb_pipeline_v2.algorithms.silhouette_algorithms import (
    ContourSilhouetteGeneratorV2,
    FilledArtifactWriterV1,
    OutlineArtifactWriterV1,
)
from rb_pipeline_v2.config import SilhouetteStageConfigV2
from rb_pipeline_v2.inspect import save_blob_branch_sanity_batch, save_contour_comparison_debug_batch
from rb_pipeline_v2.manifest import load_samples_csv, samples_csv_path
from rb_pipeline_v2.registry import get_representation_generator, list_registered_component_ids
from rb_pipeline_v2.silhouette_stage import run_silhouette_stage_v2


class SilhouetteV2UnitTests(unittest.TestCase):
    def test_registry_includes_contour_v2(self) -> None:
        register_default_components()
        ids = list_registered_component_ids()
        self.assertIn("silhouette.contour_v2", ids["generators"])
        self.assertIn("silhouette.blob_bg_v1", ids["generators"])

    def test_blob_generator_isolates_single_foreground_blob(self) -> None:
        register_default_components()
        generator = get_representation_generator("silhouette.blob_bg_v1")

        source_bgr = self._controlled_scene_bgr()
        source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)

        generated = generator.generate(
            source_gray,
            source_bgr=source_bgr,
            blur_kernel_size=5,
            canny_low_threshold=50,
            canny_high_threshold=150,
            close_kernel_size=1,
            dilate_kernel_size=1,
            min_component_area_px=50,
            fill_holes=True,
            experimental_params={
                "blob_border_fraction": 0.05,
                "blob_hue_delta": 16.0,
                "blob_sat_min": 42.0,
                "blob_val_min": 50.0,
                "blob_bright_val_min": 95.0,
                "blob_bright_sat_min": 35.0,
            },
        )

        self.assertIsNotNone(generated.contour)
        filled = FilledArtifactWriterV1().render(source_gray.shape, generated.contour, line_thickness=1)
        fg = (filled == 0).astype(np.uint8)
        count, _, _, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        self.assertEqual(count - 1, 1)

    def test_contour_v2_prefers_single_external_blob(self) -> None:
        image = self._internal_and_detached_edges_image()
        generator = ContourSilhouetteGeneratorV2()

        generated = generator.generate(
            image,
            blur_kernel_size=5,
            canny_low_threshold=50,
            canny_high_threshold=150,
            close_kernel_size=1,
            dilate_kernel_size=1,
            min_component_area_px=50,
            fill_holes=True,
        )

        self.assertIsNotNone(generated.contour)
        self.assertEqual(generated.primary_reason, "")

        filled = FilledArtifactWriterV1().render(image.shape, generated.contour, line_thickness=1)
        foreground = (filled == 0).astype(np.uint8)
        count, _, _, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        self.assertEqual(count - 1, 1)

        # Detached fragment near bottom-right should be excluded from final silhouette.
        self.assertEqual(int(filled[60, 60]), 255)

    def test_outline_and_filled_writers_render_expected_geometry(self) -> None:
        contour = np.asarray([[[8, 8]], [[8, 24]], [[24, 24]], [[24, 8]]], dtype=np.int32)

        outline = OutlineArtifactWriterV1().render((32, 32), contour, line_thickness=1)
        filled = FilledArtifactWriterV1().render((32, 32), contour, line_thickness=1)

        self.assertEqual(int(outline[16, 16]), 255)
        self.assertEqual(int(filled[16, 16]), 0)
        self.assertGreater(int((filled == 0).sum()), int((outline == 0).sum()))

    def test_stage_uses_convex_fallback_when_primary_is_broken(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_fallback_success"
            self._make_project_fixture(project_root, run_name, [self._rectangle_image()])

            config = SilhouetteStageConfigV2(
                representation_mode="outline",
                generator_id="silhouette.contour_v1",
                fallback_id="fallback.convex_hull_v1",
                min_component_area_px=10_000,
            )

            summary = run_silhouette_stage_v2(project_root, run_name, config)
            self.assertEqual(summary.successful_rows, 1)
            self.assertEqual(summary.failed_rows, 0)

            samples = load_samples_csv(
                samples_csv_path(project_root / "silhouette-images-v2" / run_name / "manifests")
            )
            self.assertEqual(str(samples.at[0, "silhouette_stage_status"]), "success")
            self.assertEqual(str(samples.at[0, "silhouette_fallback_used"]).strip().lower(), "true")
            self.assertIn("primary_", str(samples.at[0, "silhouette_fallback_reason"]))

    def test_stage_marks_failure_when_fallback_cannot_recover(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_fallback_fail"
            self._make_project_fixture(project_root, run_name, [self._blank_image()])

            config = SilhouetteStageConfigV2(
                representation_mode="filled",
                generator_id="silhouette.contour_v1",
                fallback_id="fallback.convex_hull_v1",
            )

            summary = run_silhouette_stage_v2(project_root, run_name, config)
            self.assertEqual(summary.successful_rows, 0)
            self.assertEqual(summary.failed_rows, 1)

            samples = load_samples_csv(
                samples_csv_path(project_root / "silhouette-images-v2" / run_name / "manifests")
            )
            self.assertEqual(str(samples.at[0, "silhouette_stage_status"]), "failed")
            self.assertIn("Fallback failed", str(samples.at[0, "silhouette_stage_error"]))

    def test_edge_debug_persistence_toggle(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name_true = "run_debug_true"
            run_name_false = "run_debug_false"

            self._make_project_fixture(project_root, run_name_true, [self._rectangle_image()])
            self._make_project_fixture(project_root, run_name_false, [self._rectangle_image()])

            config_true = SilhouetteStageConfigV2(
                representation_mode="outline",
                generator_id="silhouette.contour_v1",
                fallback_id="fallback.convex_hull_v1",
                persist_edge_debug=True,
            )
            config_false = SilhouetteStageConfigV2(
                representation_mode="outline",
                generator_id="silhouette.contour_v1",
                fallback_id="fallback.convex_hull_v1",
                persist_edge_debug=False,
            )

            run_silhouette_stage_v2(project_root, run_name_true, config_true)
            run_silhouette_stage_v2(project_root, run_name_false, config_false)

            samples_true = load_samples_csv(
                samples_csv_path(project_root / "silhouette-images-v2" / run_name_true / "manifests")
            )
            samples_false = load_samples_csv(
                samples_csv_path(project_root / "silhouette-images-v2" / run_name_false / "manifests")
            )

            edge_debug_filename_true = str(samples_true.at[0, "silhouette_edge_debug_filename"]).strip()
            edge_debug_value_false = samples_false.at[0, "silhouette_edge_debug_filename"]
            edge_debug_filename_false = (
                ""
                if pd.isna(edge_debug_value_false)
                else str(edge_debug_value_false).strip()
            )

            self.assertTrue(edge_debug_filename_true)
            self.assertFalse(edge_debug_filename_false)

            edge_debug_path_true = (
                project_root / "silhouette-images-v2" / run_name_true / "images" / Path(edge_debug_filename_true)
            )
            self.assertTrue(edge_debug_path_true.is_file())

    def test_comparison_debug_batch_writes_grids_and_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_compare"
            self._make_project_fixture(project_root, run_name, [self._rectangle_image(), self._circle_image()])

            output_dir = save_contour_comparison_debug_batch(
                project_root,
                run_name,
                sample_limit=2,
            )

            self.assertTrue(output_dir.is_dir())
            self.assertTrue((output_dir / "comparison_manifest.csv").is_file())
            grid_files = sorted(output_dir.glob("*.png"))
            self.assertGreaterEqual(len(grid_files), 1)

    def test_blob_sanity_batch_writes_grids_and_manifest(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            run_name = "run_blob_compare"
            self._make_project_fixture(
                project_root,
                run_name,
                [self._controlled_scene_bgr(), self._controlled_scene_bgr()],
            )

            output_dir = save_blob_branch_sanity_batch(
                project_root,
                run_name,
                sample_limit=2,
            )

            self.assertTrue(output_dir.is_dir())
            self.assertTrue((output_dir / "blob_sanity_manifest.csv").is_file())
            grid_files = sorted(output_dir.glob("*.png"))
            self.assertGreaterEqual(len(grid_files), 1)

    def _make_project_fixture(
        self,
        project_root: Path,
        run_name: str,
        images: list[np.ndarray],
    ) -> None:
        (project_root / "notebooks").mkdir(parents=True, exist_ok=True)

        run_root = project_root / "input-images" / run_name
        images_dir = run_root / "images"
        manifests_dir = run_root / "manifests"
        images_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, object]] = []

        for idx, image in enumerate(images):
            filename = f"frame_{idx:03d}.png"
            image_path = images_dir / filename
            ok = cv2.imwrite(str(image_path), image)
            if not ok:
                raise RuntimeError(f"Failed to write fixture image: {image_path}")

            rows.append(
                {
                    "run_id": run_name,
                    "sample_id": f"{run_name}_sample_{idx:03d}",
                    "frame_index": idx,
                    "image_filename": filename,
                    "distance_m": float(idx + 1),
                    "image_width_px": int(image.shape[1]),
                    "image_height_px": int(image.shape[0]),
                    "capture_success": True,
                }
            )

        pd.DataFrame(rows).to_csv(manifests_dir / "samples.csv", index=False)
        (manifests_dir / "run.json").write_text(
            json.dumps({"RunId": run_name}, indent=4) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _rectangle_image() -> np.ndarray:
        image = np.full((64, 64), 255, dtype=np.uint8)
        cv2.rectangle(image, (16, 16), (48, 48), color=0, thickness=2)
        return image

    @staticmethod
    def _blank_image() -> np.ndarray:
        return np.full((64, 64), 255, dtype=np.uint8)

    @staticmethod
    def _circle_image() -> np.ndarray:
        image = np.full((64, 64), 255, dtype=np.uint8)
        cv2.circle(image, (32, 32), 14, color=0, thickness=2)
        return image

    @staticmethod
    def _internal_and_detached_edges_image() -> np.ndarray:
        image = np.full((80, 80), 255, dtype=np.uint8)
        cv2.rectangle(image, (14, 14), (58, 58), color=0, thickness=2)
        # Internal clutter that should not survive external contour extraction.
        cv2.line(image, (22, 22), (50, 22), color=0, thickness=1)
        cv2.line(image, (22, 28), (50, 46), color=0, thickness=1)
        # Detached tiny junk blob.
        cv2.circle(image, (68, 68), 1, color=0, thickness=-1)
        return image

    @staticmethod
    def _controlled_scene_bgr() -> np.ndarray:
        image = np.zeros((160, 240, 3), dtype=np.uint8)
        image[:, :] = (45, 32, 24)  # dark blue-ish controlled background
        cv2.rectangle(image, (104, 48), (136, 96), color=(40, 60, 210), thickness=-1)  # red-ish body
        cv2.rectangle(image, (106, 42), (134, 56), color=(170, 120, 200), thickness=-1)  # magenta roof
        cv2.rectangle(image, (108, 58), (132, 70), color=(120, 170, 180), thickness=-1)  # windshield detail
        return image


if __name__ == "__main__":
    unittest.main()
