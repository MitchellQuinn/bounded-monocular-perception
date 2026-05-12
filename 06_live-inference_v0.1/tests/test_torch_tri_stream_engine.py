"""Tests for the concrete tri-stream distance/orientation inference engine."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
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
    PreparedInferenceInputs,
)
from live_inference.engines import (  # noqa: E402
    TorchTriStreamInferenceEngine,
    yaw_degrees_from_sin_cos,
)
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402
from live_inference.model_registry import (  # noqa: E402
    ModelCompatibilityError,
    load_live_model_manifest,
)
from live_inference.preprocessing import RoiFcnLocator, TriStreamLivePreprocessor  # noqa: E402
from live_inference.runtime.device import (  # noqa: E402
    normalize_torch_device_policy,
    resolve_torch_device,
)


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


class OutputDecodingTests(unittest.TestCase):
    def test_output_decoding_decodes_yaw_cardinal_angles(self) -> None:
        cases = (
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 90.0),
            (0.0, -1.0, 180.0),
            (-1.0, 0.0, 270.0),
        )
        for yaw_sin, yaw_cos, expected_deg in cases:
            with self.subTest(yaw_sin=yaw_sin, yaw_cos=yaw_cos):
                self.assertAlmostEqual(
                    yaw_degrees_from_sin_cos(yaw_sin, yaw_cos),
                    expected_deg,
                    places=6,
                )


class TorchTriStreamInferenceEngineUnitTests(unittest.TestCase):
    def test_engine_implements_inference_engine_protocol(self) -> None:
        self.assertIsInstance(_fake_engine(), contracts.InferenceEngine)

    def test_engine_rejects_missing_x_distance_image(self) -> None:
        engine = _fake_engine()
        prepared = _prepared_inputs(omit=contracts.TRI_STREAM_DISTANCE_IMAGE_KEY)

        with self.assertRaisesRegex(KeyError, "x_distance_image"):
            engine.run_inference(prepared)

    def test_engine_rejects_missing_x_orientation_image(self) -> None:
        engine = _fake_engine()
        prepared = _prepared_inputs(omit=contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY)

        with self.assertRaisesRegex(KeyError, "x_orientation_image"):
            engine.run_inference(prepared)

    def test_engine_rejects_missing_x_geometry(self) -> None:
        engine = _fake_engine()
        prepared = _prepared_inputs(omit=contracts.TRI_STREAM_GEOMETRY_KEY)

        with self.assertRaisesRegex(KeyError, "x_geometry"):
            engine.run_inference(prepared)

    def test_engine_reports_bad_shapes_clearly(self) -> None:
        engine = _fake_engine()
        prepared = _prepared_inputs(
            overrides={contracts.TRI_STREAM_DISTANCE_IMAGE_KEY: np.zeros((4, 4))}
        )

        with self.assertRaisesRegex(ValueError, "x_distance_image.*shape"):
            engine.run_inference(prepared)

    def test_fake_model_output_maps_to_inference_result_fields(self) -> None:
        engine = _fake_engine(
            model_outputs={
                contracts.MODEL_OUTPUT_DISTANCE_KEY: [12.5],
                contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY: [[1.0, 0.0]],
            }
        )

        result = engine.run_inference(_prepared_inputs())

        self.assertEqual(result.predicted_distance_m, 12.5)
        self.assertEqual(result.predicted_yaw_sin, 1.0)
        self.assertEqual(result.predicted_yaw_cos, 0.0)
        self.assertEqual(result.predicted_yaw_deg, 90.0)
        self.assertEqual(result.extras["device"], "cpu")
        self.assertEqual(result.extras["device_policy"], "cpu")

    def test_request_id_is_preserved(self) -> None:
        result = _fake_engine().run_inference(_prepared_inputs(request_id="request-123"))

        self.assertEqual(result.request_id, "request-123")

    def test_input_image_hash_is_preserved(self) -> None:
        frame_hash = FrameHash("hash-123")
        result = _fake_engine().run_inference(_prepared_inputs(frame_hash=frame_hash))

        self.assertIs(result.input_image_hash, frame_hash)

    def test_roi_metadata_is_populated_from_preprocessing_metadata(self) -> None:
        result = _fake_engine().run_inference(_prepared_inputs())

        self.assertIsNotNone(result.roi_metadata)
        assert result.roi_metadata is not None
        self.assertEqual(result.roi_metadata.bbox_xyxy_px, (10.0, 20.0, 110.0, 220.0))
        self.assertEqual(result.roi_metadata.center_xy_px, (60.0, 120.0))
        self.assertEqual(result.roi_metadata.source_image_wh_px, (480, 300))
        self.assertEqual(result.roi_metadata.geometry_schema, contracts.TRI_STREAM_GEOMETRY_SCHEMA)

    def test_background_metadata_is_carried_in_roi_metadata_extras(self) -> None:
        prepared = _prepared_inputs(
            preprocessing_metadata_overrides={
                contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED: True,
                contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED: True,
                contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_FCN_APPLIED: True,
                contracts.PREPROCESSING_METADATA_BACKGROUND_APPLICATION_SPACE: (
                    contracts.BACKGROUND_APPLICATION_SPACE_ROI_FCN_INPUT_AND_ROI_CROP
                ),
            }
        )

        result = _fake_engine().run_inference(prepared)

        self.assertIsNotNone(result.roi_metadata)
        assert result.roi_metadata is not None
        extras = result.roi_metadata.extras
        self.assertTrue(
            extras[contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED]
        )
        self.assertEqual(
            extras[contracts.PREPROCESSING_METADATA_BACKGROUND_APPLICATION_SPACE],
            contracts.BACKGROUND_APPLICATION_SPACE_ROI_FCN_INPUT_AND_ROI_CROP,
        )

    def test_roi_guard_metadata_is_carried_in_roi_metadata_extras(self) -> None:
        prepared = _prepared_inputs(
            preprocessing_metadata_overrides={
                "roi_confidence": 0.41,
                "roi_clipped": False,
                "roi_accepted": True,
                "apply_manual_mask_to_roi_locator": False,
                "apply_background_removal_to_roi_locator": True,
                "manual_mask_applied_to_roi_locator": False,
                "background_removal_applied_to_roi_locator": True,
            }
        )

        result = _fake_engine().run_inference(prepared)

        self.assertIsNotNone(result.roi_metadata)
        assert result.roi_metadata is not None
        extras = result.roi_metadata.extras
        self.assertEqual(extras["roi_confidence"], 0.41)
        self.assertFalse(extras["roi_clipped"])
        self.assertTrue(extras["roi_accepted"])
        self.assertTrue(extras["apply_background_removal_to_roi_locator"])
        self.assertTrue(extras["background_removal_applied_to_roi_locator"])

    def test_preprocessing_parameter_revision_is_preserved(self) -> None:
        result = _fake_engine().run_inference(_prepared_inputs(parameter_revision=9))

        self.assertEqual(result.preprocessing_parameter_revision, 9)

    def test_debug_paths_are_populated_from_preprocessing_metadata(self) -> None:
        result = _fake_engine().run_inference(_prepared_inputs())

        self.assertEqual(
            set(result.debug_paths),
            {
                contracts.DISPLAY_ARTIFACT_ACCEPTED_RAW_FRAME,
                contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
                contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
                contracts.DISPLAY_ARTIFACT_ROI_OVERLAY_METADATA,
            },
        )
        self.assertEqual(
            result.debug_paths[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY],
            Path("debug/x_distance.png"),
        )

    def test_inference_time_ms_is_populated(self) -> None:
        result = _fake_engine().run_inference(_prepared_inputs())

        self.assertGreaterEqual(result.inference_time_ms, 0.0)

    def test_model_compatibility_is_checked_before_inference(self) -> None:
        events: list[str] = []
        model = _FakeModel(events=events)

        def compatibility_checker(manifest: object) -> None:
            self.assertIsNotNone(manifest)
            events.append("compatibility")

        engine = _fake_engine(model=model, compatibility_checker=compatibility_checker)

        engine.run_inference(_prepared_inputs())

        self.assertEqual(events, ["compatibility", "model"])

    def test_incompatible_manifest_fails_clearly_before_model_call(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "empty-model"
            root.mkdir()
            manifest = load_live_model_manifest(root)
            model = _FakeModel()
            engine = TorchTriStreamInferenceEngine(
                model_root=root,
                model_manifest=manifest,
                model=model,
                device="cpu",
                load_model=False,
            )

            with self.assertRaisesRegex(
                ModelCompatibilityError,
                "Live model compatibility check failed",
            ):
                engine.run_inference(_prepared_inputs())

        self.assertEqual(model.calls, 0)

    def test_no_pyside6_or_opencv_imports_added_to_engine_modules(self) -> None:
        module_paths = (
            SRC_ROOT / "live_inference/engines/__init__.py",
            SRC_ROOT / "live_inference/engines/output_decoding.py",
            SRC_ROOT / "live_inference/engines/torch_tri_stream_engine.py",
        )
        found = {str(path): _imported_roots(path) for path in module_paths}

        self.assertTrue(
            all(not (imports & {"PySide6", "cv2"}) for imports in found.values()),
            found,
        )

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


class TorchTriStreamInferenceEngineIntegrationTests(unittest.TestCase):
    def test_selected_live_local_model_runs_on_preprocessed_real_frame_cpu(self) -> None:
        _optional_torch()
        selected = _load_selected_artifacts()
        _require_file(SOURCE_IMAGE_PATH, "05 inference input corpus sample image")

        manifest = load_live_model_manifest(
            selected.distance_orientation_root,
            roi_locator_root=selected.roi_fcn_root,
        )
        roi_locator = RoiFcnLocator(
            selected.roi_fcn_root,
            device=_runtime_device(selected.roi_fcn_device),
        )
        preprocessor = TriStreamLivePreprocessor(
            model_manifest=manifest,
            roi_locator=roi_locator,
        )
        image_bytes = SOURCE_IMAGE_PATH.read_bytes()
        prepared = preprocessor.prepare_model_inputs(
            _request_for_image(SOURCE_IMAGE_PATH, image_bytes),
            image_bytes,
        )
        engine = TorchTriStreamInferenceEngine(
            model_root=selected.distance_orientation_root,
            device="cpu",
        )

        result = engine.run_inference(prepared)

        self.assertTrue(math.isfinite(result.predicted_distance_m))
        self.assertTrue(math.isfinite(result.predicted_yaw_sin))
        self.assertTrue(math.isfinite(result.predicted_yaw_cos))
        self.assertTrue(math.isfinite(result.predicted_yaw_deg))
        self.assertGreaterEqual(result.predicted_yaw_deg, 0.0)
        self.assertLess(result.predicted_yaw_deg, 360.0)
        self.assertEqual(result.request_id, "engine-integration-real-frame")
        self.assertEqual(result.input_image_hash, prepared.source_frame.frame_hash)


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


class _FakeModel:
    def __init__(
        self,
        *,
        model_outputs: object | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.model_outputs = model_outputs or {
            contracts.MODEL_OUTPUT_DISTANCE_KEY: [4.5],
            contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY: [[0.0, 1.0]],
        }
        self.events = events
        self.calls = 0
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, batch: object) -> object:
        self.calls += 1
        if self.events is not None:
            self.events.append("model")
        if not isinstance(batch, dict):
            raise AssertionError("engine must pass model inputs as a mapping")
        return self.model_outputs


def _fake_engine(
    *,
    model_outputs: object | None = None,
    model: _FakeModel | None = None,
    compatibility_checker: object | None = None,
) -> TorchTriStreamInferenceEngine:
    manifest = _compatible_manifest()
    return TorchTriStreamInferenceEngine(
        model_root=manifest.model_root,
        model_manifest=manifest,
        model=model or _FakeModel(model_outputs=model_outputs),
        device="cpu",
        load_model=False,
        compatibility_checker=compatibility_checker,  # type: ignore[arg-type]
        now_utc_fn=lambda: "2026-05-05T12:00:00Z",
    )


def _prepared_inputs(
    *,
    request_id: str = "request-1",
    frame_hash: FrameHash | None = None,
    omit: str | None = None,
    overrides: dict[str, object] | None = None,
    preprocessing_metadata_overrides: dict[str, object] | None = None,
    parameter_revision: int = 7,
) -> PreparedInferenceInputs:
    model_inputs: dict[str, object] = {
        contracts.TRI_STREAM_DISTANCE_IMAGE_KEY: np.zeros((1, 4, 4), dtype=np.float32),
        contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY: np.ones((1, 4, 4), dtype=np.float32),
        contracts.TRI_STREAM_GEOMETRY_KEY: np.arange(10, dtype=np.float32),
    }
    if overrides:
        model_inputs.update(overrides)
    if omit is not None:
        model_inputs.pop(omit, None)

    frame = FrameReference(
        image_path=Path("live_frames/latest_frame.png"),
        frame_hash=frame_hash or FrameHash("hash-1"),
        metadata=FrameMetadata(width_px=480, height_px=300),
    )
    preprocessing_metadata: dict[str, object] = {
        contracts.PREPROCESSING_METADATA_RUNTIME_PARAMETER_REVISION: parameter_revision,
        contracts.PREPROCESSING_METADATA_SOURCE_IMAGE_WIDTH_PX: 480,
        contracts.PREPROCESSING_METADATA_SOURCE_IMAGE_HEIGHT_PX: 300,
        contracts.PREPROCESSING_METADATA_DISTANCE_CANVAS_WIDTH_PX: 4,
        contracts.PREPROCESSING_METADATA_DISTANCE_CANVAS_HEIGHT_PX: 4,
        contracts.PREPROCESSING_METADATA_ORIENTATION_CANVAS_WIDTH_PX: 4,
        contracts.PREPROCESSING_METADATA_ORIENTATION_CANVAS_HEIGHT_PX: 4,
        contracts.PREPROCESSING_METADATA_PREDICTED_ROI_CENTER_XY_PX: (60.0, 120.0),
        contracts.PREPROCESSING_METADATA_SILHOUETTE_BBOX_XYXY_PX: (
            10.0,
            20.0,
            110.0,
            220.0,
        ),
        contracts.PREPROCESSING_METADATA_GEOMETRY_SCHEMA: contracts.TRI_STREAM_GEOMETRY_SCHEMA,
        contracts.PREPROCESSING_METADATA_DEBUG_PATHS: {
            contracts.DISPLAY_ARTIFACT_ACCEPTED_RAW_FRAME: "debug/raw.png",
            contracts.TRI_STREAM_DISTANCE_IMAGE_KEY: "debug/x_distance.png",
            contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY: "debug/x_orientation.png",
            contracts.DISPLAY_ARTIFACT_ROI_OVERLAY_METADATA: "debug/metadata.json",
        },
    }
    if preprocessing_metadata_overrides:
        preprocessing_metadata.update(preprocessing_metadata_overrides)
    return PreparedInferenceInputs(
        request_id=request_id,
        source_frame=frame,
        model_inputs=model_inputs,
        preprocessing_metadata=preprocessing_metadata,
    )


def _compatible_manifest() -> object:
    tmp = TemporaryDirectory()
    root = Path(tmp.name) / "model-run"
    root.mkdir()
    _write_compatible_bundle(root)
    manifest = load_live_model_manifest(root)
    _TEMP_DIRS.append(tmp)
    return manifest


_TEMP_DIRS: list[TemporaryDirectory] = []


def _write_compatible_bundle(root: Path) -> None:
    _write_json(
        root / "config.json",
        {
            "model_name": "engine-test",
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
        },
    )
    _write_json(
        root / "dataset_summary.json",
        {
            "preprocessing_contract": {
                "ContractVersion": contracts.PREPROCESSING_CONTRACT_NAME,
                "CurrentRepresentation": {
                    "Kind": contracts.TRI_STREAM_REPRESENTATION_KIND,
                    "ArrayKeys": [
                        contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
                        contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
                        contracts.TRI_STREAM_GEOMETRY_KEY,
                        "y_distance_m",
                        "y_yaw_sin",
                        "y_yaw_cos",
                    ],
                    "DistanceImageKey": contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
                    "OrientationImageKey": contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
                    "GeometryKey": contracts.TRI_STREAM_GEOMETRY_KEY,
                    "GeometrySchema": list(contracts.TRI_STREAM_GEOMETRY_SCHEMA),
                    "GeometryDim": len(contracts.TRI_STREAM_GEOMETRY_SCHEMA),
                    "OrientationImageContent": (
                        "inverted_vehicle_detail_on_white_no_brightness_normalization"
                    ),
                },
                "Stages": {
                    "pack_tri_stream": {
                        "OrientationImageRepresentation": (
                            "target_centered_inverted_vehicle_on_white_scaled_by_silhouette_extent"
                        )
                    }
                },
            }
        },
    )
    (root / "best.pt").touch()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_selected_artifacts() -> _SelectedArtifacts:
    if SELECTION_PATH.is_file():
        from live_inference.model_registry.model_selection import (  # noqa: PLC0415
            load_model_selection,
        )

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
        roi_fcn_device="auto",
    )


def _request_for_image(image_path: Path, image_bytes: bytes) -> InferenceRequest:
    image_shape = _decode_source_shape(image_bytes)
    frame = FrameReference(
        image_path=image_path,
        metadata=FrameMetadata(
            width_px=image_shape[0],
            height_px=image_shape[1],
            pixel_format="rgb8" if image_shape[2] else "gray8",
            encoding=image_path.suffix.lstrip(".") or "png",
            byte_size=len(image_bytes),
        ),
        frame_hash=compute_frame_hash(image_bytes),
        byte_size=len(image_bytes),
    )
    return InferenceRequest(
        request_id="engine-integration-real-frame",
        frame=frame,
        requested_at_utc=REQUESTED_AT,
    )


def _decode_source_shape(image_bytes: bytes) -> tuple[int, int, int | None]:
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("test source image bytes could not be decoded")
    height_px, width_px = decoded.shape[:2]
    channels = int(decoded.shape[2]) if decoded.ndim == 3 else None
    return int(width_px), int(height_px), channels


def _runtime_device(requested_device: str) -> str:
    torch = _optional_torch()
    requested = normalize_torch_device_policy(requested_device)
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return resolve_torch_device(requested)


def _optional_torch() -> object:
    if importlib.util.find_spec("torch") is None:
        raise unittest.SkipTest("torch is not installed in the repo venv")
    return importlib.import_module("torch")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise unittest.SkipTest(f"{label} is not available: {path}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise unittest.SkipTest(f"{label} is not available: {path}")


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
