"""End-to-end synchronous live inference integration test.

This composes the non-GUI live inference path around a single real frame:
atomic frame handoff, latest-frame selection, ROI-FCN preprocessing, Torch
distance/orientation inference, and the synchronous processing core.
"""

from __future__ import annotations

import ast
import builtins
from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import importlib.util
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from interfaces import (  # noqa: E402
    FrameMetadata,
    FrameSkipReason,
    InferenceResult,
    LiveInferenceConfig,
)
from live_inference.engines import TorchTriStreamInferenceEngine  # noqa: E402
from live_inference.frame_handoff import (  # noqa: E402
    AtomicFrameHandoffWriter,
    LatestFrameHandoffReader,
)
from live_inference.frame_selection import InferenceFrameSelector  # noqa: E402
from live_inference.inference_core import InferenceProcessingCore  # noqa: E402
from live_inference.model_registry import load_live_model_manifest  # noqa: E402
from live_inference.model_registry.model_selection import load_model_selection  # noqa: E402
from live_inference.preprocessing import RoiFcnLocator, TriStreamLivePreprocessor  # noqa: E402


REQUESTED_AT = "2026-05-05T00:00:00Z"
RESULT_AT = "2026-05-05T00:00:01Z"
RUNTIME_PARAMETER_REVISION = 17
SELECTION_PATH = PROJECT_ROOT / "models/selections/current.toml"
SOURCE_IMAGE_PATH = (
    REPO_ROOT
    / "05_inference-v0.4-ts/input/def90_synth_v023-validation-shuffled/images/"
    / "defender90_f000000_z04.709_j206.png"
)


class LiveInferenceE2ETests(unittest.TestCase):
    def test_synchronous_non_gui_live_inference_processes_one_real_frame(self) -> None:
        pyside_modules_before = _loaded_pyside6_modules()
        _optional_torch()
        selected = _load_selected_artifacts()
        _require_file(SOURCE_IMAGE_PATH, "05 inference input corpus sample image")

        manifest = load_live_model_manifest(
            selected.distance_orientation_root,
            roi_locator_root=selected.roi_fcn_root,
        )
        image_bytes = SOURCE_IMAGE_PATH.read_bytes()
        image_shape = _decode_source_shape(image_bytes)

        with TemporaryDirectory() as tmp_dir:
            config = LiveInferenceConfig(
                frame_dir=Path(tmp_dir) / "live_frames",
                latest_frame_filename="latest_frame.png",
                temp_frame_filename="latest_frame.tmp.png",
            )
            writer = AtomicFrameHandoffWriter(config)
            writer.publish_frame(
                image_bytes,
                FrameMetadata(
                    frame_id="e2e-real-frame-1",
                    source_name="05_inference-v0.4-ts",
                    width_px=image_shape.width_px,
                    height_px=image_shape.height_px,
                    pixel_format="rgb8" if image_shape.channels else "gray8",
                    encoding="png",
                    byte_size=len(image_bytes),
                ),
            )
            reader = LatestFrameHandoffReader(config)
            selector = InferenceFrameSelector(
                reader,
                request_id_factory=lambda: "live-e2e-real-frame",
                now_utc_fn=lambda: REQUESTED_AT,
            )

            with _forbid_gui_worker_imports():
                roi_locator = RoiFcnLocator(selected.roi_fcn_root, device="cpu")
                preprocessor = TriStreamLivePreprocessor(
                    model_manifest=manifest,
                    roi_locator=roi_locator,
                    runtime_parameter_revision_getter=lambda: RUNTIME_PARAMETER_REVISION,
                )
                engine = TorchTriStreamInferenceEngine(
                    model_root=selected.distance_orientation_root,
                    model_manifest=manifest,
                    device="cpu",
                    now_utc_fn=lambda: RESULT_AT,
                )
                core = InferenceProcessingCore(
                    selector,
                    preprocessor,
                    engine,
                    now_utc_fn=lambda: RESULT_AT,
                )
                outcome = core.process_once()
                duplicate = core.process_once()

        self.assertIsNone(outcome.error)
        self.assertIsNone(outcome.warning)
        self.assertIsNone(outcome.skipped)
        self.assertIsNotNone(outcome.result)
        result = outcome.result
        assert result is not None

        self.assertIsInstance(result, InferenceResult)
        self.assertTrue(result.request_id)
        self.assertIsNotNone(result.input_image_hash)
        self.assertTrue(result.input_image_hash.value)
        self.assertTrue(math.isfinite(result.predicted_distance_m))
        self.assertTrue(math.isfinite(result.predicted_yaw_sin))
        self.assertTrue(math.isfinite(result.predicted_yaw_cos))
        self.assertTrue(math.isfinite(result.predicted_yaw_deg))
        self.assertGreaterEqual(result.predicted_yaw_deg, 0.0)
        self.assertLess(result.predicted_yaw_deg, 360.0)
        self.assertGreaterEqual(result.inference_time_ms, 0.0)
        self.assertEqual(
            result.preprocessing_parameter_revision,
            RUNTIME_PARAMETER_REVISION,
        )
        self.assertIsNotNone(result.roi_metadata)
        self.assertIsInstance(result.warnings, tuple)
        self.assertEqual(selector.last_processed_hash(), result.input_image_hash)

        self.assertIsNotNone(duplicate.skipped)
        assert duplicate.skipped is not None
        self.assertEqual(duplicate.skipped.reason, FrameSkipReason.DUPLICATE_HASH)
        self.assertIsNone(duplicate.result)
        self.assertIsNone(duplicate.error)

        self.assertEqual(_loaded_pyside6_modules(), pyside_modules_before)

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


class _SelectedArtifacts:
    def __init__(
        self,
        *,
        distance_orientation_root: Path,
        roi_fcn_root: Path,
    ) -> None:
        self.distance_orientation_root = distance_orientation_root
        self.roi_fcn_root = roi_fcn_root


class _ImageShape:
    def __init__(self, *, width_px: int, height_px: int, channels: int | None) -> None:
        self.width_px = width_px
        self.height_px = height_px
        self.channels = channels


def _load_selected_artifacts() -> _SelectedArtifacts:
    _require_file(SELECTION_PATH, "selected live-local model selection config")
    selection = load_model_selection(SELECTION_PATH)
    _require_dir(
        selection.distance_orientation_root,
        "selected live-local distance/orientation artifact",
    )
    _require_dir(selection.roi_fcn_root, "selected live-local ROI-FCN artifact")
    return _SelectedArtifacts(
        distance_orientation_root=selection.distance_orientation_root,
        roi_fcn_root=selection.roi_fcn_root,
    )


def _decode_source_shape(image_bytes: bytes) -> _ImageShape:
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("test source image bytes could not be decoded")
    height_px, width_px = decoded.shape[:2]
    channels = int(decoded.shape[2]) if decoded.ndim == 3 else None
    return _ImageShape(
        width_px=int(width_px),
        height_px=int(height_px),
        channels=channels,
    )


def _optional_torch() -> object:
    if importlib.util.find_spec("torch") is None:
        raise unittest.SkipTest("torch is not installed in the repo venv")
    return importlib.import_module("torch")


@contextmanager
def _forbid_gui_worker_imports() -> Iterator[None]:
    forbidden_prefixes = (
        "PySide6",
        "live_inference.gui",
        "live_inference.worker",
        "live_inference.workers",
    )
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if level == 0 and any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        ):
            raise AssertionError(
                f"live inference e2e test must not import GUI/worker code: {name}"
            )
        return original_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", side_effect=guarded_import):
        yield


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise unittest.SkipTest(f"{label} is not available: {path}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise unittest.SkipTest(f"{label} is not available: {path}")


def _loaded_pyside6_modules() -> set[str]:
    return {
        module_name
        for module_name in sys.modules
        if module_name == "PySide6" or module_name.startswith("PySide6.")
    }


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
