"""Tests for the concrete ROI-FCN live locator adapter."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import math
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402

from live_inference.preprocessing import (  # noqa: E402
    RoiFcnLocator,
    RoiLocator,
    build_roi_fcn_locator_input,
    decode_roi_fcn_heatmap,
    load_roi_fcn_artifact_metadata,
    resolve_roi_fcn_checkpoint,
)


SELECTED_ROI_ROOT = (
    PROJECT_ROOT / "models/roi-fcn/260420-1219_roi-fcn-tiny__run_0003"
)


class RoiFcnLocatorMetadataTests(unittest.TestCase):
    def test_metadata_loader_reads_selected_live_local_roi_artifact(self) -> None:
        _require_selected_roi_artifact()

        metadata = load_roi_fcn_artifact_metadata(SELECTED_ROI_ROOT)

        self.assertEqual(metadata.roi_model_root, SELECTED_ROI_ROOT.resolve())
        self.assertEqual(metadata.topology_id, "roi_fcn_tiny")
        self.assertEqual(metadata.topology_variant, "tiny_v1")
        self.assertIn("training_contract_version", metadata.dataset_contract)
        self.assertIn("training_contract_version", metadata.run_config)

    def test_checkpoint_discovery_selects_best_pt(self) -> None:
        _require_selected_roi_artifact()

        checkpoint = resolve_roi_fcn_checkpoint(SELECTED_ROI_ROOT)

        self.assertEqual(checkpoint.name, "best.pt")
        self.assertEqual(checkpoint, (SELECTED_ROI_ROOT / "best.pt").resolve())

    def test_locator_canvas_size_resolves_to_selected_artifact_contract(self) -> None:
        _require_selected_roi_artifact()

        metadata = load_roi_fcn_artifact_metadata(SELECTED_ROI_ROOT)

        self.assertEqual(metadata.locator_canvas_size, (480, 300))

    def test_roi_crop_size_resolves_to_selected_artifact_contract(self) -> None:
        _require_selected_roi_artifact()

        metadata = load_roi_fcn_artifact_metadata(SELECTED_ROI_ROOT)

        self.assertEqual(metadata.roi_crop_size, (300, 300))

    def test_locator_can_be_instantiated_metadata_only_without_loading_model(self) -> None:
        _require_selected_roi_artifact()

        locator = RoiFcnLocator(SELECTED_ROI_ROOT, device="cpu", load_model=False)

        self.assertIsInstance(locator, RoiLocator)
        self.assertFalse(locator.model_loaded)
        self.assertEqual(locator.metadata.locator_canvas_size, (480, 300))

    def test_generic_core_modules_remain_heavy_import_free(self) -> None:
        module_paths = (
            SRC_ROOT / "interfaces/contracts.py",
            SRC_ROOT / "live_inference/frame_handoff.py",
            SRC_ROOT / "live_inference/frame_selection.py",
            SRC_ROOT / "live_inference/inference_core.py",
            SRC_ROOT / "live_inference/runtime_parameters.py",
            SRC_ROOT / "live_inference/model_registry/compatibility.py",
        )

        found = {str(path): _banned_imports(path) for path in module_paths}

        self.assertTrue(all(not imports for imports in found.values()), found)

    def test_roi_fcn_locator_does_not_import_torch_at_module_import_time(self) -> None:
        module_path = SRC_ROOT / "live_inference/preprocessing/roi_fcn_locator.py"

        imported_roots = _imported_roots(module_path)

        self.assertNotIn("torch", imported_roots)
        self.assertNotIn("PySide6", imported_roots)
        self.assertNotIn("src", imported_roots)


class RoiFcnLocatorGeometryTests(unittest.TestCase):
    def test_prepare_locator_input_shape_for_generated_grayscale_image(self) -> None:
        source_gray = _generated_source_image(width=960, height=600)

        locator_input = build_roi_fcn_locator_input(
            source_gray,
            canvas_width_px=480,
            canvas_height_px=300,
        )

        self.assertEqual(locator_input.locator_image.shape, (1, 300, 480))
        self.assertEqual(locator_input.locator_image.dtype, np.float32)
        self.assertEqual(tuple(locator_input.source_image_wh_px.tolist()), (960, 600))
        self.assertEqual(tuple(locator_input.resized_image_wh_px.tolist()), (480, 300))
        self.assertEqual(tuple(locator_input.padding_ltrb_px.tolist()), (0, 0, 0, 0))
        self.assertAlmostEqual(locator_input.resize_scale, 0.5)

    def test_heatmap_decode_maps_synthetic_peak_to_source_coordinates(self) -> None:
        source_gray = _generated_source_image(width=480, height=300)
        locator_input = build_roi_fcn_locator_input(
            source_gray,
            canvas_width_px=480,
            canvas_height_px=300,
        )
        heatmap = np.zeros((300, 480), dtype=np.float32)
        heatmap[150, 240] = 1.0

        location = decode_roi_fcn_heatmap(
            heatmap,
            locator_input=locator_input,
            canvas_width_px=480,
            canvas_height_px=300,
            roi_width_px=300,
            roi_height_px=300,
        )

        self.assertEqual(location.center_xy_px, (240.0, 150.0))
        self.assertEqual(location.roi_bounds_xyxy_px, (90.0, 0.0, 390.0, 300.0))
        self.assertEqual(location.metadata["heatmap_shape"], (300, 480))
        self.assertEqual(location.metadata["heatmap_peak_confidence"], 1.0)


class RoiFcnLocatorRuntimeTests(unittest.TestCase):
    def test_injected_torch_model_produces_deterministic_roi_location_on_cpu(self) -> None:
        _require_selected_roi_artifact()
        torch = _optional_torch()

        class PeakModel(torch.nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                output = torch.zeros(
                    (1, 1, 300, 480),
                    dtype=torch.float32,
                    device=x.device,
                )
                output[0, 0, 150, 240] = 1.0
                return output

        locator = RoiFcnLocator(
            SELECTED_ROI_ROOT,
            device="cpu",
            load_model=False,
            model=PeakModel(),
        )

        location = locator.locate(_generated_source_image(width=480, height=300))

        self.assertTrue(locator.model_loaded)
        self.assertEqual(location.center_xy_px, (240.0, 150.0))
        self.assertEqual(location.roi_bounds_xyxy_px, (90.0, 0.0, 390.0, 300.0))
        self.assertEqual(location.metadata["device"], "cpu")

    def test_selected_live_local_model_loads_on_cpu_and_runs_generated_image_if_available(self) -> None:
        _require_selected_roi_artifact()
        _optional_torch()

        locator = RoiFcnLocator(SELECTED_ROI_ROOT, device="cpu")
        location = locator.locate(_generated_source_image(width=480, height=300))

        self.assertTrue(locator.model_loaded)
        self.assertEqual(len(location.center_xy_px), 2)
        self.assertTrue(all(math.isfinite(value) for value in location.center_xy_px))
        self.assertIsNotNone(location.roi_bounds_xyxy_px)
        self.assertEqual(location.metadata["checkpoint_name"], "best.pt")
        self.assertEqual(location.metadata["heatmap_shape"], (300, 480))


def _require_selected_roi_artifact() -> None:
    if not SELECTED_ROI_ROOT.is_dir():
        raise unittest.SkipTest("selected live-local ROI-FCN artifact is not available")


def _optional_torch() -> object:
    if importlib.util.find_spec("torch") is None:
        raise unittest.SkipTest("torch is not installed in the repo venv")

    return importlib.import_module("torch")


def _generated_source_image(*, width: int, height: int) -> np.ndarray:
    image = np.full((height, width), 255, dtype=np.uint8)
    cx = width // 2
    cy = height // 2
    half_w = max(8, width // 16)
    half_h = max(8, height // 12)
    image[cy - half_h : cy + half_h, cx - half_w : cx + half_w] = 32
    image[cy - max(2, half_h // 4) : cy + max(2, half_h // 4), cx - half_w : cx] = 96
    return image


def _banned_imports(module_path: Path) -> set[str]:
    return _imported_roots(module_path) & {"PySide6", "cv2", "numpy", "torch", "pandas"}


def _imported_roots(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".", maxsplit=1)[0])
    return found


if __name__ == "__main__":
    unittest.main()
