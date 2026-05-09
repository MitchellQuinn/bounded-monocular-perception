"""Tests for the concrete live tri-stream preprocessor adapter."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import interfaces.contracts as contracts  # noqa: E402
from interfaces import (  # noqa: E402
    FrameHash,
    FrameMetadata,
    FrameReference,
    InferenceRequest,
    RawImagePreprocessor,
)
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402
from live_inference.model_registry import load_live_model_manifest  # noqa: E402
from live_inference.model_registry.model_manifest import (  # noqa: E402
    ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE,
)
from live_inference.preprocessing import (  # noqa: E402
    RoiLocation,
    TriStreamLivePreprocessor,
)
from live_inference.preprocessing.debug_artifacts import (  # noqa: E402
    ARTIFACT_ACCEPTED_RAW_FRAME,
    ARTIFACT_DISTANCE_IMAGE,
    ARTIFACT_LOCATOR_INPUT,
    ARTIFACT_ORIENTATION_IMAGE,
    ARTIFACT_ROI_OVERLAY_METADATA,
)


REQUESTED_AT = "2026-05-04T12:00:00Z"

# TODO: Add a parity test against the 05 preprocessor once a concrete ROI-FCN
# locator can be wired without loading model runtimes; this slice intentionally
# uses a deterministic fake locator.


class TriStreamLivePreprocessorTests(unittest.TestCase):
    def test_instantiates_with_manifest_and_fake_roi_locator(self) -> None:
        manifest = _fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE)

        preprocessor = TriStreamLivePreprocessor(
            model_manifest=manifest,
            roi_locator=FakeRoiLocator(),
        )

        self.assertIsInstance(preprocessor, RawImagePreprocessor)

    def test_valid_image_bytes_return_required_prepared_inputs(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(image_bytes)
        preprocessor = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
            runtime_parameter_revision_getter=lambda: 7,
        )

        prepared = preprocessor.prepare_model_inputs(request, image_bytes)

        self.assertEqual(prepared.request_id, request.request_id)
        self.assertIs(prepared.source_frame, request.frame)
        self.assertEqual(
            prepared.input_mode,
            contracts.InferenceInputMode.TRI_STREAM_V0_4,
        )
        self.assertEqual(prepared.input_keys, contracts.TRI_STREAM_INPUT_KEYS)
        self.assertEqual(tuple(prepared.model_inputs), contracts.TRI_STREAM_INPUT_KEYS)
        self.assertEqual(
            set(prepared.model_inputs),
            {
                contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
                contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
                contracts.TRI_STREAM_GEOMETRY_KEY,
            },
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY].shape,
            (1, 300, 300),
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY].shape,
            (1, 300, 300),
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_GEOMETRY_KEY].shape,
            (10,),
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY].dtype,
            np.float32,
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY].dtype,
            np.float32,
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_GEOMETRY_KEY].dtype,
            np.float32,
        )
        self.assertEqual(prepared.preprocessing_metadata["runtime_parameter_revision"], 7)

    def test_invalid_image_bytes_fail_clearly(self) -> None:
        preprocessor = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        )

        with self.assertRaisesRegex(ValueError, "decode image bytes"):
            preprocessor.prepare_model_inputs(_request(b"not-an-image"), b"not-an-image")

    def test_metadata_includes_contract_geometry_orientation_and_roi_information(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(image_bytes)
        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)

        metadata = prepared.preprocessing_metadata

        self.assertEqual(
            metadata["preprocessing_contract_name"],
            contracts.PREPROCESSING_CONTRACT_NAME,
        )
        self.assertEqual(metadata["geometry_schema"], contracts.TRI_STREAM_GEOMETRY_SCHEMA)
        self.assertEqual(metadata["geometry_dim"], 10)
        self.assertEqual(metadata["orientation_source_mode"], ORIENTATION_SOURCE_RAW_GRAYSCALE)
        self.assertEqual(metadata["input_image_hash"], request.frame.frame_hash.value)
        self.assertIn("roi_request_xyxy_px", metadata)
        self.assertIn("roi_source_xyxy_px", metadata)
        self.assertIn("silhouette_bbox_xyxy_px", metadata)
        self.assertGreater(metadata["silhouette_area_px"], 0)
        self.assertEqual(metadata["predicted_roi_center_xy_px"], (240.0, 150.0))
        self.assertEqual(metadata["roi_locator_metadata"], {"locator": "fake"})

    def test_raw_grayscale_orientation_source_mode_is_supported(self) -> None:
        image_bytes = _fixture_image_bytes()
        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        orientation = prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY]

        self.assertEqual(
            prepared.preprocessing_metadata["orientation_source_mode"],
            ORIENTATION_SOURCE_RAW_GRAYSCALE,
        )
        self.assertGreater(int(np.count_nonzero(orientation < 1.0)), 0)

    def test_inverted_vehicle_orientation_source_mode_is_supported_by_current_manifest(self) -> None:
        image_bytes = _fixture_image_bytes()
        prepared = TriStreamLivePreprocessor(
            model_manifest=_current_or_fixture_inverted_manifest(),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        distance = prepared.model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY]
        orientation = prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY]
        brightness = prepared.preprocessing_metadata["brightness_normalization"]

        self.assertEqual(
            prepared.preprocessing_metadata["orientation_source_mode"],
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
        )
        self.assertGreater(int(np.count_nonzero(distance < 1.0)), 0)
        self.assertGreater(int(np.count_nonzero(orientation < 1.0)), 0)
        self.assertIn("Status", brightness)
        self.assertGreater(brightness["ForegroundPixelCount"], 0)

    def test_orientation_source_modes_use_distinct_source_images(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(image_bytes)
        raw_prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)
        inverted_prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)

        raw_orientation = raw_prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY]
        inverted_orientation = inverted_prepared.model_inputs[
            contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY
        ]

        self.assertFalse(np.allclose(raw_orientation, inverted_orientation))

    def test_frame_hash_mismatch_is_reported_as_metadata_warning(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(
            image_bytes,
            frame_hash=FrameHash("0" * 32),
        )

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)

        self.assertTrue(
            any(
                "frame_hash does not match" in item
                for item in prepared.preprocessing_metadata["warnings"]
            )
        )

    def test_debug_enabled_populates_paths_and_writes_valid_images(self) -> None:
        image_bytes = _fixture_image_bytes()
        with TemporaryDirectory() as tmp_dir:
            request = _request(
                image_bytes,
                save_debug_images=True,
                debug_output_dir=Path(tmp_dir),
            )

            prepared = TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=FakeDebugRoiLocator(),
            ).prepare_model_inputs(request, image_bytes)

            paths = prepared.preprocessing_metadata["debug_paths"]
            for key in (
                ARTIFACT_ACCEPTED_RAW_FRAME,
                ARTIFACT_LOCATOR_INPUT,
                ARTIFACT_DISTANCE_IMAGE,
                ARTIFACT_ORIENTATION_IMAGE,
            ):
                self.assertIn(key, paths)
                self.assertIsNotNone(cv2.imread(str(paths[key]), cv2.IMREAD_UNCHANGED))

            metadata_path = Path(paths[ARTIFACT_ROI_OVERLAY_METADATA])
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["request_id"], request.request_id)
            self.assertEqual(metadata["input_image_hash"], request.frame.frame_hash.value)
            self.assertEqual(metadata["orientation_source_mode"], ORIENTATION_SOURCE_RAW_GRAYSCALE)

    def test_debug_disabled_does_not_write_debug_paths(self) -> None:
        image_bytes = _fixture_image_bytes()
        with TemporaryDirectory() as tmp_dir:
            prepared = TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=FakeDebugRoiLocator(),
            ).prepare_model_inputs(
                _request(
                    image_bytes,
                    save_debug_images=False,
                    debug_output_dir=Path(tmp_dir),
                ),
                image_bytes,
            )

            self.assertNotIn("debug_paths", prepared.preprocessing_metadata)
            self.assertEqual(list(Path(tmp_dir).iterdir()), [])

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

    def test_preprocessor_adds_no_pyside_or_model_inference_imports(self) -> None:
        module_paths = (
            SRC_ROOT / "live_inference/preprocessing/__init__.py",
            SRC_ROOT / "live_inference/preprocessing/preprocessing_config.py",
            SRC_ROOT / "live_inference/preprocessing/roi_fcn_locator.py",
            SRC_ROOT / "live_inference/preprocessing/roi_locator.py",
            SRC_ROOT / "live_inference/preprocessing/tri_stream_live_preprocessor.py",
        )

        found = {
            str(path): _banned_imports(
                path,
                banned_roots={"PySide6", "torch", "src"},
            )
            for path in module_paths
        }

        self.assertTrue(all(not imports for imports in found.values()), found)


class FakeRoiLocator:
    def locate(self, source_gray_image: object) -> RoiLocation:
        return RoiLocation(
            center_xy_px=(240.0, 150.0),
            roi_bounds_xyxy_px=(90.0, 0.0, 390.0, 300.0),
            metadata={"locator": "fake"},
        )


class FakeDebugRoiLocator:
    def locate(self, source_gray_image: object) -> RoiLocation:
        return RoiLocation(
            center_xy_px=(240.0, 150.0),
            roi_bounds_xyxy_px=(90.0, 0.0, 390.0, 300.0),
            metadata={
                "locator": "fake",
                "locator_canvas_width_px": 480,
                "locator_canvas_height_px": 300,
            },
        )


def _fixture_image_bytes() -> bytes:
    image = np.full((300, 480), 255, dtype=np.uint8)
    cv2.rectangle(image, (210, 115), (270, 185), 30, -1)
    cv2.line(image, (220, 130), (260, 170), 95, 5)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode fixture image")
    return encoded.tobytes()


def _request(
    image_bytes: bytes,
    *,
    frame_hash: FrameHash | None = None,
    save_debug_images: bool = False,
    debug_output_dir: Path | None = None,
) -> InferenceRequest:
    frame_hash = frame_hash or compute_frame_hash(image_bytes)
    frame = FrameReference(
        image_path=Path("live_frames/latest_frame.png"),
        metadata=FrameMetadata(
            width_px=480,
            height_px=300,
            pixel_format="gray8",
            encoding="png",
            byte_size=len(image_bytes),
        ),
        frame_hash=frame_hash,
        byte_size=len(image_bytes),
    )
    return InferenceRequest(
        request_id="request-1",
        frame=frame,
        requested_at_utc=REQUESTED_AT,
        save_debug_images=save_debug_images,
        debug_output_dir=debug_output_dir,
    )


def _current_or_fixture_inverted_manifest() -> object:
    current_root = (
        PROJECT_ROOT
        / "models/distance-orientation/260504-1100_ts-2d-cnn__run_0001"
    )
    if current_root.is_dir():
        return load_live_model_manifest(current_root)
    return _fixture_manifest(ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE)


def _fixture_manifest(orientation_source_mode: str) -> object:
    temp_dir = TemporaryDirectory()
    _TEMP_DIRS.append(temp_dir)
    root = Path(temp_dir.name) / "model-run"
    root.mkdir()
    _write_json(root / "config.json", _model_config())
    _write_json(
        root / "dataset_summary.json",
        {"preprocessing_contract": _preprocessing_contract(orientation_source_mode)},
    )
    return load_live_model_manifest(root)


_TEMP_DIRS: list[TemporaryDirectory] = []


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _model_config() -> dict[str, object]:
    return {
        "model_name": "distance-orientation-test",
        "topology_id": "distance_regressor_tri_stream_yaw",
        "topology_variant": "tri_stream_yaw_v0_1",
        "topology_contract": {
            "contract_version": contracts.MODEL_TOPOLOGY_CONTRACT_VERSION,
            "runtime": {"input_mode": contracts.TRI_STREAM_INPUT_MODE},
            "outputs": {
                "distance": {
                    "output_key": contracts.MODEL_OUTPUT_DISTANCE_KEY,
                    "columns": ["distance_m"],
                },
                "yaw": {
                    "output_key": contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY,
                    "columns": ["yaw_sin", "yaw_cos"],
                },
            },
        },
        "topology_params": {
            "geometry_feature_dim": len(contracts.TRI_STREAM_GEOMETRY_SCHEMA)
        },
    }


def _orientation_semantics(orientation_source_mode: str) -> dict[str, str | None]:
    if orientation_source_mode == ORIENTATION_SOURCE_RAW_GRAYSCALE:
        return {
            "representation": "target_centered_raw_grayscale_scaled_by_silhouette_extent",
            "content": "raw_grayscale_detail_preserving_no_brightness_normalization",
            "polarity": None,
        }
    if orientation_source_mode == ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE:
        return {
            "representation": (
                "target_centered_inverted_vehicle_on_white_scaled_by_silhouette_extent"
            ),
            "content": "inverted_vehicle_detail_on_white_no_brightness_normalization",
            "polarity": "dark_vehicle_detail_on_white_background",
        }
    raise ValueError(f"Unsupported test orientation source mode: {orientation_source_mode}")


def _preprocessing_contract(orientation_source_mode: str) -> dict[str, object]:
    orientation_semantics = _orientation_semantics(orientation_source_mode)
    current_representation: dict[str, object] = {
        "Kind": contracts.TRI_STREAM_REPRESENTATION_KIND,
        "StorageFormat": "npz",
        "ArrayKeys": [
            contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
            contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
            contracts.TRI_STREAM_GEOMETRY_KEY,
            "y_distance_m",
            "y_yaw_deg",
            "y_yaw_sin",
            "y_yaw_cos",
        ],
        "DistanceImageKey": contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
        "OrientationImageKey": contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
        "GeometryKey": contracts.TRI_STREAM_GEOMETRY_KEY,
        "GeometrySchema": list(contracts.TRI_STREAM_GEOMETRY_SCHEMA),
        "GeometryDim": len(contracts.TRI_STREAM_GEOMETRY_SCHEMA),
        "CanvasWidth": 300,
        "CanvasHeight": 300,
        "OrientationContextScale": 1.25,
        "OrientationImageContent": orientation_semantics["content"],
        "BrightnessNormalization": {
            "Enabled": False,
            "Method": "none",
            "TargetMedianDarkness": 0.55,
            "MinGain": 0.5,
            "MaxGain": 2.0,
            "Epsilon": 1e-6,
            "EmptyMaskPolicy": "skip",
        },
    }
    if orientation_semantics["polarity"]:
        current_representation["OrientationImagePolarity"] = orientation_semantics["polarity"]
    return {
        "ContractVersion": contracts.PREPROCESSING_CONTRACT_NAME,
        "CurrentStage": "pack_tri_stream",
        "CompletedStages": ["detect", "silhouette", "pack_tri_stream"],
        "CurrentRepresentation": current_representation,
        "Stages": {
            "pack_tri_stream": {
                "CanvasWidth": 300,
                "CanvasHeight": 300,
                "ClipPolicy": "fail",
                "OrientationContextScale": 1.25,
                "OrientationImageRepresentation": orientation_semantics["representation"],
                "BrightnessNormalization": current_representation["BrightnessNormalization"],
            },
            "silhouette": {
                "BlurKernelSize": 5,
                "CannyHighThreshold": 150,
                "CannyLowThreshold": 50,
                "CloseKernelSize": 3,
                "DilateKernelSize": 1,
                "FallbackId": "fallback.convex_hull_v1",
                "FillHoles": True,
                "GeneratorId": "silhouette.contour_v2",
                "MinComponentAreaPx": 50,
                "OutlineThicknessPx": 1,
                "PersistDebug": False,
                "ROICanvasHeightPx": 300,
                "ROICanvasWidthPx": 300,
                "ROIPaddingPx": 0,
                "RepresentationMode": "filled",
                "UseConvexHullFallback": True,
            },
        },
    }


def _banned_imports(
    module_path: Path,
    *,
    banned_roots: set[str] | None = None,
) -> set[str]:
    banned_roots = banned_roots or {"PySide6", "cv2", "numpy", "torch", "pandas"}
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    return imported_roots & banned_roots


if __name__ == "__main__":
    unittest.main()
