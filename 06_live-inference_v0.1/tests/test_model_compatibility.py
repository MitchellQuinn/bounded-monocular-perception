"""Tests for live model manifest compatibility diagnostics."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import interfaces.contracts as contracts  # noqa: E402
from live_inference.model_registry import (  # noqa: E402
    ModelCompatibilityError,
    check_live_model_compatibility,
    load_live_model_manifest,
    load_model_selection,
    require_live_model_compatibility,
)
from live_inference.model_registry.model_manifest import (  # noqa: E402
    ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE,
    LiveModelManifest,
)


_TEMP_DIRS: list[TemporaryDirectory] = []


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def _write_compatible_bundle(
    root: Path,
    roi_root: Path | None = None,
    *,
    orientation_source_mode: str = ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    orientation_image_representation: str | None = None,
    orientation_image_content: str | None = None,
    orientation_image_polarity: str | None = None,
) -> None:
    orientation_semantics = _orientation_semantics(orientation_source_mode)
    representation = (
        orientation_image_representation
        if orientation_image_representation is not None
        else orientation_semantics["representation"]
    )
    content = (
        orientation_image_content
        if orientation_image_content is not None
        else orientation_semantics["content"]
    )
    polarity = (
        orientation_image_polarity
        if orientation_image_polarity is not None
        else orientation_semantics["polarity"]
    )
    current_representation: dict[str, object] = {
        "Kind": contracts.TRI_STREAM_REPRESENTATION_KIND,
        "StorageFormat": "npz",
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
        "CanvasWidth": 300,
        "CanvasHeight": 300,
        "OrientationImageContent": content,
    }
    if polarity:
        current_representation["OrientationImagePolarity"] = polarity
    _write_json(
        root / "config.json",
        {
            "model_name": "compatible-test",
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
                "CurrentStage": "pack_tri_stream",
                "CompletedStages": ["detect", "silhouette", "pack_tri_stream"],
                "CurrentRepresentation": current_representation,
                "Stages": {
                    "pack_tri_stream": {
                        "CanvasWidth": 300,
                        "CanvasHeight": 300,
                        "OrientationImageRepresentation": representation,
                    }
                },
            }
        },
    )
    (root / "best.pt").touch()

    if roi_root is not None:
        _write_json(roi_root / "run_config.json", {"roi_width_px": 300, "roi_height_px": 300})
        _write_json(
            roi_root / "dataset_contract.json",
            {
                "validation_split": {
                    "geometry": {"canvas_width_px": 480, "canvas_height_px": 300}
                }
            },
        )


def _compatible_manifest(
    *,
    orientation_source_mode: str = ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    orientation_image_representation: str | None = None,
    orientation_image_content: str | None = None,
    orientation_image_polarity: str | None = None,
) -> LiveModelManifest:
    tmp = TemporaryDirectory()
    _TEMP_DIRS.append(tmp)
    root = Path(tmp.name) / "model-run"
    roi_root = Path(tmp.name) / "roi-run"
    root.mkdir()
    roi_root.mkdir()
    _write_compatible_bundle(
        root,
        roi_root,
        orientation_source_mode=orientation_source_mode,
        orientation_image_representation=orientation_image_representation,
        orientation_image_content=orientation_image_content,
        orientation_image_polarity=orientation_image_polarity,
    )
    return load_live_model_manifest(root, roi_locator_root=roi_root)


def _error_codes(manifest: LiveModelManifest) -> set[str]:
    result = check_live_model_compatibility(manifest)
    return {issue.code for issue in result.issues if issue.severity == "error"}


def _format_errors(result: object) -> str:
    issues = getattr(result, "issues", ())
    return "; ".join(
        f"{issue.code}: {issue.message}"
        for issue in issues
        if issue.severity == "error"
    )


class LiveModelCompatibilityTests(unittest.TestCase):
    def test_compatible_fixture_passes_with_no_errors(self) -> None:
        manifest = _compatible_manifest()

        result = check_live_model_compatibility(manifest)

        self.assertTrue(result.ok)
        self.assertEqual([issue for issue in result.issues if issue.severity == "error"], [])
        self.assertEqual(
            result.orientation_source_mode,
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
        )

    def test_raw_grayscale_orientation_source_mode_is_compatible(self) -> None:
        manifest = _compatible_manifest(
            orientation_source_mode=ORIENTATION_SOURCE_RAW_GRAYSCALE
        )

        result = check_live_model_compatibility(manifest)

        self.assertTrue(result.ok)
        self.assertEqual(result.orientation_source_mode, ORIENTATION_SOURCE_RAW_GRAYSCALE)

    def test_grayscale_white_preprocessing_contract_is_compatible(self) -> None:
        manifest = replace(
            _compatible_manifest(orientation_source_mode=ORIENTATION_SOURCE_RAW_GRAYSCALE),
            preprocessing_contract_name=contracts.PREPROCESSING_CONTRACT_NAME_GRAYSCALE_WHITE,
        )

        result = check_live_model_compatibility(manifest)

        self.assertTrue(result.ok, _format_errors(result))

    def test_reports_resolved_orientation_source_mode_when_manifest_field_is_empty(self) -> None:
        manifest = replace(_compatible_manifest(), orientation_source_mode=None)

        result = check_live_model_compatibility(manifest)

        self.assertTrue(result.ok)
        self.assertEqual(
            result.orientation_source_mode,
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
        )

    def test_unknown_orientation_semantics_fails(self) -> None:
        codes = _error_codes(
            _compatible_manifest(
                orientation_image_content="unsupported_orientation_encoding"
            )
        )

        self.assertIn("unsupported_orientation_image_content", codes)

    def test_conflicting_orientation_semantics_fails(self) -> None:
        codes = _error_codes(
            _compatible_manifest(
                orientation_image_representation=(
                    "target_centered_raw_grayscale_scaled_by_silhouette_extent"
                )
            )
        )

        self.assertIn("conflicting_orientation_source_mode", codes)

    def test_unknown_orientation_source_mode_fails(self) -> None:
        codes = _error_codes(
            replace(_compatible_manifest(), orientation_source_mode="unknown_mode")
        )

        self.assertIn("unsupported_orientation_source_mode", codes)

    def test_wrong_preprocessing_contract_fails(self) -> None:
        codes = _error_codes(
            replace(_compatible_manifest(), preprocessing_contract_name="wrong-contract")
        )

        self.assertIn("preprocessing_contract_name_mismatch", codes)

    def test_wrong_input_mode_fails(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), input_mode="raw_image"))

        self.assertIn("input_mode_mismatch", codes)

    def test_wrong_representation_kind_fails(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), representation_kind="dual_stream_npz"))

        self.assertIn("representation_kind_mismatch", codes)

    def test_missing_distance_input_key_fails(self) -> None:
        manifest = _compatible_manifest()
        input_keys = tuple(
            key for key in manifest.input_keys if key != contracts.TRI_STREAM_DISTANCE_IMAGE_KEY
        )

        codes = _error_codes(replace(manifest, input_keys=input_keys))

        self.assertIn("missing_input_key", codes)

    def test_missing_orientation_input_key_fails(self) -> None:
        manifest = _compatible_manifest()
        input_keys = tuple(
            key for key in manifest.input_keys if key != contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY
        )

        codes = _error_codes(replace(manifest, input_keys=input_keys))

        self.assertIn("missing_input_key", codes)

    def test_missing_geometry_input_key_fails(self) -> None:
        manifest = _compatible_manifest()
        input_keys = tuple(
            key for key in manifest.input_keys if key != contracts.TRI_STREAM_GEOMETRY_KEY
        )

        codes = _error_codes(replace(manifest, input_keys=input_keys))

        self.assertIn("missing_input_key", codes)

    def test_geometry_schema_mismatch_fails(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), geometry_schema=("cx_px",)))

        self.assertIn("geometry_schema_mismatch", codes)

    def test_geometry_dimension_mismatch_fails(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), geometry_dim=9))

        self.assertIn("geometry_dim_mismatch", codes)

    def test_missing_distance_output_key_fails(self) -> None:
        manifest = _compatible_manifest()
        output_keys = tuple(
            key for key in manifest.model_output_keys if key != contracts.MODEL_OUTPUT_DISTANCE_KEY
        )

        codes = _error_codes(replace(manifest, model_output_keys=output_keys))

        self.assertIn("missing_model_output_key", codes)

    def test_missing_yaw_output_key_fails(self) -> None:
        manifest = _compatible_manifest()
        output_keys = tuple(
            key for key in manifest.model_output_keys if key != contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY
        )

        codes = _error_codes(replace(manifest, model_output_keys=output_keys))

        self.assertIn("missing_model_output_key", codes)

    def test_yaw_output_width_not_two_fails_if_discoverable(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), yaw_output_width=3))

        self.assertIn("yaw_output_width_mismatch", codes)

    def test_distance_output_width_not_one_fails_if_discoverable(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), distance_output_width=2))

        self.assertIn("distance_output_width_mismatch", codes)

    def test_missing_checkpoint_fails(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), checkpoint_path=None))

        self.assertIn("missing_checkpoint", codes)

    def test_invalid_non_positive_canvas_size_fails_if_discoverable(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), distance_canvas_size=(0, 300)))

        self.assertIn("invalid_distance_canvas_size", codes)

    def test_invalid_orientation_canvas_size_fails_if_discoverable(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), orientation_canvas_size=(300, -1)))

        self.assertIn("invalid_orientation_canvas_size", codes)

    def test_incompatible_roi_locator_crop_fails_if_discoverable(self) -> None:
        codes = _error_codes(replace(_compatible_manifest(), roi_locator_crop_size=(224, 224)))

        self.assertIn("roi_locator_crop_size_mismatch", codes)

    def test_roi_locator_crop_larger_than_canvas_fails_if_discoverable(self) -> None:
        codes = _error_codes(
            replace(
                _compatible_manifest(),
                roi_locator_crop_size=(500, 301),
                roi_locator_canvas_size=(480, 300),
            )
        )

        self.assertIn("roi_locator_crop_exceeds_canvas", codes)

    def test_current_selected_live_local_260504_model_remains_compatible(self) -> None:
        selection_path = PROJECT_ROOT / "models/selections/current.toml"
        if not selection_path.is_file():
            self.skipTest("current live-local model selection is not available")
        selection = load_model_selection(selection_path)
        if not selection.distance_orientation_root.is_dir():
            self.skipTest("selected live-local distance/orientation artifact is not available")

        manifest = load_live_model_manifest(
            selection.distance_orientation_root,
            roi_locator_root=(
                selection.roi_fcn_root if selection.roi_fcn_root.is_dir() else None
            ),
        )
        result = check_live_model_compatibility(manifest)

        self.assertEqual(
            manifest.orientation_source_mode,
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
        )
        self.assertTrue(result.ok, _format_errors(result))

    def test_require_helper_does_not_raise_for_compatible_fixture(self) -> None:
        require_live_model_compatibility(_compatible_manifest())

    def test_require_helper_raises_for_incompatible_fixture(self) -> None:
        manifest = replace(_compatible_manifest(), preprocessing_contract_name="wrong-contract")

        with self.assertRaises(ModelCompatibilityError) as context:
            require_live_model_compatibility(manifest)

        message = str(context.exception)
        self.assertIn("Live model compatibility check failed", message)
        self.assertIn("preprocessing_contract_name", message)

    def test_compatibility_module_keeps_heavy_runtime_imports_out(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "model_registry" / "compatibility.py"

        self.assertEqual(_banned_imports(module_path), set())

    def test_generic_live_inference_core_modules_remain_heavy_import_free(self) -> None:
        module_paths = [
            SRC_ROOT / "interfaces" / "contracts.py",
            SRC_ROOT / "live_inference" / "frame_handoff.py",
            SRC_ROOT / "live_inference" / "frame_selection.py",
            SRC_ROOT / "live_inference" / "runtime_parameters.py",
            SRC_ROOT / "live_inference" / "inference_core.py",
            SRC_ROOT / "live_inference" / "model_registry" / "model_manifest.py",
            SRC_ROOT / "live_inference" / "model_registry" / "compatibility.py",
        ]

        found = {str(path): _banned_imports(path) for path in module_paths}

        self.assertEqual(found, {str(path): set() for path in module_paths})


def _banned_imports(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    banned_roots = {"PySide6", "cv2", "numpy", "torch"}
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".", maxsplit=1)[0])
    return found & banned_roots


if __name__ == "__main__":
    unittest.main()
