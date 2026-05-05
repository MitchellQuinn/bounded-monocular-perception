"""Integration test for live ROI-FCN + tri-stream preprocessing.

This test intentionally depends on one real image from the 05 inference input
corpus because the live ROI-FCN artifact was trained for those rendered frames.
It stops at PreparedInferenceInputs and does not construct a distance/orientation
inference engine.
"""

from __future__ import annotations

import builtins
from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import interfaces.contracts as contracts  # noqa: E402
from interfaces import (  # noqa: E402
    FrameMetadata,
    FrameReference,
    InferenceRequest,
    PreparedInferenceInputs,
)
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402
from live_inference.model_registry.model_manifest import (  # noqa: E402
    ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    load_live_model_manifest,
)
from live_inference.preprocessing import RoiFcnLocator, TriStreamLivePreprocessor  # noqa: E402


REQUESTED_AT = "2026-05-05T00:00:00Z"
KNOWN_DISTANCE_ORIENTATION_ROOT = (
    PROJECT_ROOT / "models/distance-orientation/260504-1100_ts-2d-cnn__run_0001"
)
KNOWN_ROI_FCN_ROOT = (
    PROJECT_ROOT / "models/roi-fcn/260420-1219_roi-fcn-tiny__run_0003"
)
SELECTION_PATH = PROJECT_ROOT / "models/selections/current.toml"
SOURCE_IMAGE_PATH = (
    REPO_ROOT
    / "05_inference-v0.4-ts/input/def90_synth_v023-validation-shuffled/images/"
    / "defender90_f000000_z04.709_j206.png"
)


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_selected_roi_fcn_locator_prepares_real_tri_stream_inputs(self) -> None:
        torch = _optional_torch()
        selected = _load_selected_artifacts()
        _require_file(SOURCE_IMAGE_PATH, "05 inference input corpus sample image")

        manifest = load_live_model_manifest(
            selected.distance_orientation_root,
            roi_locator_root=selected.roi_fcn_root,
        )
        self.assertEqual(
            manifest.orientation_source_mode,
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
        )
        self.assertIsNotNone(manifest.checkpoint_path)
        self.assertTrue(manifest.checkpoint_path.is_file())

        roi_device = _roi_runtime_device(selected.roi_fcn_device, torch)
        roi_locator = RoiFcnLocator(selected.roi_fcn_root, device=roi_device)

        image_bytes = SOURCE_IMAGE_PATH.read_bytes()
        request = _request_for_image(SOURCE_IMAGE_PATH, image_bytes)
        image_shape = _decode_source_shape(image_bytes)
        distance_orientation_checkpoint_loads: list[Path] = []

        original_torch_load = torch.load

        def guarded_torch_load(path: object, *args: object, **kwargs: object) -> object:
            candidate = _resolved_path_or_none(path)
            if candidate == manifest.checkpoint_path:
                distance_orientation_checkpoint_loads.append(candidate)
                raise AssertionError(
                    "distance/orientation model loading must not run during preprocessing"
                )
            return original_torch_load(path, *args, **kwargs)

        with patch.object(torch, "load", side_effect=guarded_torch_load):
            with _forbid_gui_worker_imports():
                preprocessor = TriStreamLivePreprocessor(
                    model_manifest=manifest,
                    roi_locator=roi_locator,
                )
                prepared = preprocessor.prepare_model_inputs(request, image_bytes)

        self.assertEqual(distance_orientation_checkpoint_loads, [])
        self.assertIsInstance(prepared, PreparedInferenceInputs)
        self.assertIs(prepared.source_frame, request.frame)
        self.assertEqual(prepared.input_mode, contracts.InferenceInputMode.TRI_STREAM_V0_4)
        self.assertEqual(prepared.input_keys, contracts.TRI_STREAM_INPUT_KEYS)

        model_inputs = prepared.model_inputs
        self.assertEqual(
            set(model_inputs),
            {
                contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
                contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
                contracts.TRI_STREAM_GEOMETRY_KEY,
            },
        )
        distance_image = model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY]
        orientation_image = model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY]
        geometry = model_inputs[contracts.TRI_STREAM_GEOMETRY_KEY]

        self.assertEqual(distance_image.shape, (1, 300, 300))
        self.assertEqual(orientation_image.shape, (1, 300, 300))
        self.assertEqual(len(geometry), 10)
        self.assertGreater(distance_image.size, 0)
        self.assertGreater(orientation_image.size, 0)
        self.assertGreater(int(np.count_nonzero(distance_image < 1.0)), 0)
        self.assertGreater(int(np.count_nonzero(orientation_image < 1.0)), 0)
        self.assertTrue(np.isfinite(distance_image).all())
        self.assertTrue(np.isfinite(orientation_image).all())
        self.assertTrue(np.isfinite(geometry).all())

        metadata = prepared.preprocessing_metadata
        self.assertEqual(
            metadata["preprocessing_contract_name"],
            contracts.PREPROCESSING_CONTRACT_NAME,
        )
        self.assertEqual(
            metadata["orientation_source_mode"],
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
        )
        self.assertEqual(metadata["source_image_width_px"], image_shape.width_px)
        self.assertEqual(metadata["source_image_height_px"], image_shape.height_px)
        self.assertEqual(metadata["distance_canvas_width_px"], 300)
        self.assertEqual(metadata["distance_canvas_height_px"], 300)
        self.assertEqual(metadata["orientation_canvas_width_px"], 300)
        self.assertEqual(metadata["orientation_canvas_height_px"], 300)
        self.assertIn("roi_request_xyxy_px", metadata)
        self.assertIn("roi_source_xyxy_px", metadata)
        self.assertIn("roi_canvas_insert_xyxy_px", metadata)
        self.assertIn("roi_locator_bounds_xyxy_px", metadata)
        self.assertIn("roi_locator_metadata", metadata)
        self.assertIn("silhouette_bbox_xyxy_px", metadata)
        self.assertGreater(metadata["silhouette_area_px"], 0)

        locator_metadata = metadata["roi_locator_metadata"]
        self.assertEqual(locator_metadata["checkpoint_name"], "best.pt")
        self.assertEqual(locator_metadata["device"], roi_device)
        self.assertEqual(locator_metadata["heatmap_shape"], (300, 480))


class _SelectedArtifacts:
    def __init__(
        self,
        *,
        distance_orientation_root: Path,
        roi_fcn_root: Path,
        roi_fcn_device: str,
    ) -> None:
        self.distance_orientation_root = distance_orientation_root
        self.roi_fcn_root = roi_fcn_root
        self.roi_fcn_device = roi_fcn_device


class _ImageShape:
    def __init__(self, *, width_px: int, height_px: int, channels: int | None) -> None:
        self.width_px = width_px
        self.height_px = height_px
        self.channels = channels


def _load_selected_artifacts() -> _SelectedArtifacts:
    try:
        from live_inference.model_registry.model_selection import (  # noqa: PLC0415
            load_model_selection,
        )
    except ImportError:
        load_model_selection = None

    if load_model_selection is not None:
        if SELECTION_PATH.is_file():
            selection = load_model_selection(SELECTION_PATH)
            _require_dir(
                selection.distance_orientation_root,
                "selected live-local distance/orientation artifact",
            )
            _require_dir(selection.roi_fcn_root, "selected live-local ROI-FCN artifact")
            return _SelectedArtifacts(
                distance_orientation_root=selection.distance_orientation_root,
                roi_fcn_root=selection.roi_fcn_root,
                roi_fcn_device=selection.roi_fcn_device,
            )

    _require_dir(
        KNOWN_DISTANCE_ORIENTATION_ROOT,
        "known live-local distance/orientation artifact",
    )
    _require_dir(KNOWN_ROI_FCN_ROOT, "known live-local ROI-FCN artifact")
    return _SelectedArtifacts(
        distance_orientation_root=KNOWN_DISTANCE_ORIENTATION_ROOT.resolve(),
        roi_fcn_root=KNOWN_ROI_FCN_ROOT.resolve(),
        roi_fcn_device="cuda",
    )


def _optional_torch() -> object:
    if importlib.util.find_spec("torch") is None:
        raise unittest.SkipTest("torch is not installed in the repo venv")
    return importlib.import_module("torch")


def _roi_runtime_device(requested_device: str, torch: object) -> str:
    requested = str(requested_device).strip() or "cuda"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _request_for_image(image_path: Path, image_bytes: bytes) -> InferenceRequest:
    image_shape = _decode_source_shape(image_bytes)
    frame = FrameReference(
        image_path=image_path,
        metadata=FrameMetadata(
            width_px=image_shape.width_px,
            height_px=image_shape.height_px,
            pixel_format="rgb8" if image_shape.channels else "gray8",
            encoding=image_path.suffix.lstrip(".") or "png",
            byte_size=len(image_bytes),
        ),
        frame_hash=compute_frame_hash(image_bytes),
        byte_size=len(image_bytes),
    )
    return InferenceRequest(
        request_id="preprocessing-integration-real-roi-fcn",
        frame=frame,
        requested_at_utc=REQUESTED_AT,
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


def _resolved_path_or_none(value: object) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    return Path(value).expanduser().resolve()


@contextmanager
def _forbid_gui_worker_imports() -> Iterator[None]:
    forbidden_prefixes = (
        "PySide6",
        "live_inference.gui",
        "live_inference.worker",
        "live_inference.workers",
        "live_inference.inference_core",
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
                f"preprocessing integration must not import GUI/worker code: {name}"
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


if __name__ == "__main__":
    unittest.main()
