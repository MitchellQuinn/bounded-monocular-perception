"""Tests for lightweight live model manifest normalization."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import interfaces.contracts as contracts  # noqa: E402
from live_inference.model_registry import (  # noqa: E402
    load_live_model_manifest,
    resolve_orientation_source_mode,
)
from live_inference.model_registry.model_manifest import (  # noqa: E402
    ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE_ON_WHITE,
    OrientationSourceModeError,
)


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
        "topology_params": {"geometry_feature_dim": len(contracts.TRI_STREAM_GEOMETRY_SCHEMA)},
    }


def _orientation_semantics(orientation_source_mode: str) -> dict[str, str | None]:
    if orientation_source_mode == ORIENTATION_SOURCE_RAW_GRAYSCALE:
        return {
            "representation": "target_centered_raw_grayscale_scaled_by_silhouette_extent",
            "content": "raw_grayscale_detail_preserving_no_brightness_normalization",
            "polarity": None,
        }
    if orientation_source_mode == ORIENTATION_SOURCE_RAW_GRAYSCALE_ON_WHITE:
        return {
            "representation": (
                "target_centered_raw_grayscale_scaled_by_silhouette_extent_foreground_enhanced"
            ),
            "content": (
                "foreground_enhanced_raw_grayscale_detail_on_white_no_brightness_normalization"
            ),
            "polarity": "source_grayscale_vehicle_detail_on_white_background",
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


def _preprocessing_contract(
    orientation_source_mode: str = ORIENTATION_SOURCE_RAW_GRAYSCALE,
) -> dict[str, object]:
    orientation_semantics = _orientation_semantics(orientation_source_mode)
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
        "OrientationImageContent": orientation_semantics["content"],
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
                "OrientationImageRepresentation": orientation_semantics["representation"],
            }
        },
    }


class LiveModelManifestLoaderTests(unittest.TestCase):
    def test_loads_available_json_metadata_and_records_sources(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "model-run"
            root.mkdir()
            config = _model_config()
            run_manifest = {"run": "manifest"}
            dataset_summary = {"preprocessing_contract": _preprocessing_contract()}
            model_architecture = {"architecture": "text", **_model_config()}
            _write_json(root / "config.json", config)
            _write_json(root / "run_manifest.json", run_manifest)
            _write_json(root / "dataset_summary.json", dataset_summary)
            _write_json(root / "model_architecture.json", model_architecture)
            (root / "best.pt").touch()

            manifest = load_live_model_manifest(root)

        self.assertEqual(manifest.raw_metadata["config"], config)
        self.assertEqual(manifest.raw_metadata["run_manifest"], run_manifest)
        self.assertEqual(manifest.raw_metadata["dataset_summary"], dataset_summary)
        self.assertEqual(manifest.raw_metadata["model_architecture"], model_architecture)
        self.assertEqual(manifest.source_files["config"], (root / "config.json").resolve())
        self.assertEqual(
            manifest.source_files["dataset_summary"],
            (root / "dataset_summary.json").resolve(),
        )
        self.assertEqual(manifest.checkpoint_path, (root / "best.pt").resolve())
        self.assertEqual(manifest.checkpoint_kind, "best")

    def test_discovers_checkpoint_candidates(self) -> None:
        cases = (
            ("best.pt", "best"),
            ("best_model.pt", "best_model"),
            ("latest.pt", "latest"),
        )
        for filename, expected_kind in cases:
            with self.subTest(filename=filename):
                with TemporaryDirectory() as tmp_dir:
                    root = Path(tmp_dir) / "model-run"
                    root.mkdir()
                    (root / filename).touch()

                    manifest = load_live_model_manifest(root)

                self.assertEqual(manifest.checkpoint_path, (root / filename).resolve())
                self.assertEqual(manifest.checkpoint_kind, expected_kind)

    def test_uses_checkpoint_priority_order(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "model-run"
            root.mkdir()
            for filename in ("latest.pt", "best_model.pt", "best.pt"):
                (root / filename).touch()

            manifest = load_live_model_manifest(root)

        self.assertEqual(manifest.checkpoint_path, (root / "best.pt").resolve())
        self.assertEqual(manifest.checkpoint_kind, "best")

    def test_priority_uses_best_model_before_latest(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "model-run"
            root.mkdir()
            (root / "latest.pt").touch()
            (root / "best_model.pt").touch()

            manifest = load_live_model_manifest(root)

        self.assertEqual(manifest.checkpoint_path, (root / "best_model.pt").resolve())
        self.assertEqual(manifest.checkpoint_kind, "best_model")

    def test_normalizes_contract_fields_from_existing_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "model-run"
            root.mkdir()
            _write_json(root / "config.json", _model_config())
            _write_json(
                root / "dataset_summary.json",
                {"preprocessing_contract": _preprocessing_contract()},
            )

            manifest = load_live_model_manifest(root)

        self.assertEqual(manifest.model_label, "distance-orientation-test")
        self.assertEqual(manifest.topology_id, "distance_regressor_tri_stream_yaw")
        self.assertEqual(manifest.topology_variant, "tri_stream_yaw_v0_1")
        self.assertEqual(
            manifest.topology_contract_version,
            contracts.MODEL_TOPOLOGY_CONTRACT_VERSION,
        )
        self.assertEqual(manifest.preprocessing_contract_name, contracts.PREPROCESSING_CONTRACT_NAME)
        self.assertEqual(manifest.input_mode, contracts.TRI_STREAM_INPUT_MODE)
        self.assertEqual(manifest.representation_kind, contracts.TRI_STREAM_REPRESENTATION_KIND)
        self.assertIn(contracts.TRI_STREAM_DISTANCE_IMAGE_KEY, manifest.input_keys)
        self.assertEqual(manifest.distance_image_key, contracts.TRI_STREAM_DISTANCE_IMAGE_KEY)
        self.assertEqual(manifest.orientation_image_key, contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY)
        self.assertEqual(manifest.geometry_key, contracts.TRI_STREAM_GEOMETRY_KEY)
        self.assertEqual(manifest.geometry_schema, contracts.TRI_STREAM_GEOMETRY_SCHEMA)
        self.assertEqual(manifest.geometry_dim, len(contracts.TRI_STREAM_GEOMETRY_SCHEMA))
        self.assertEqual(manifest.distance_canvas_size, (300, 300))
        self.assertEqual(manifest.orientation_canvas_size, (300, 300))
        self.assertEqual(
            manifest.orientation_image_representation,
            "target_centered_raw_grayscale_scaled_by_silhouette_extent",
        )
        self.assertEqual(
            manifest.orientation_image_content,
            "raw_grayscale_detail_preserving_no_brightness_normalization",
        )
        self.assertIsNone(manifest.orientation_image_polarity)
        self.assertEqual(manifest.orientation_source_mode, ORIENTATION_SOURCE_RAW_GRAYSCALE)
        self.assertEqual(
            set(manifest.model_output_keys),
            {contracts.MODEL_OUTPUT_DISTANCE_KEY, contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY},
        )
        self.assertEqual(manifest.distance_output_width, 1)
        self.assertEqual(manifest.yaw_output_width, 2)

    def test_resolves_raw_grayscale_orientation_source_mode_fixture(self) -> None:
        contract = _preprocessing_contract(ORIENTATION_SOURCE_RAW_GRAYSCALE)

        self.assertEqual(
            resolve_orientation_source_mode(contract),
            ORIENTATION_SOURCE_RAW_GRAYSCALE,
        )

    def test_resolves_inverted_vehicle_orientation_source_mode_fixture(self) -> None:
        contract = _preprocessing_contract(ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE)

        self.assertEqual(
            resolve_orientation_source_mode(contract),
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
        )

    def test_resolves_raw_grayscale_on_white_orientation_source_mode_fixture(self) -> None:
        contract = _preprocessing_contract(ORIENTATION_SOURCE_RAW_GRAYSCALE_ON_WHITE)

        self.assertEqual(
            resolve_orientation_source_mode(contract),
            ORIENTATION_SOURCE_RAW_GRAYSCALE_ON_WHITE,
        )

    def test_missing_orientation_semantics_fail_clearly(self) -> None:
        contract = _preprocessing_contract()
        contract["CurrentRepresentation"].pop("OrientationImageContent")
        contract["Stages"]["pack_tri_stream"].pop("OrientationImageRepresentation")

        with self.assertRaises(OrientationSourceModeError) as context:
            resolve_orientation_source_mode(contract)

        self.assertEqual(context.exception.code, "missing_orientation_source_mode")
        self.assertIn("OrientationImageRepresentation", str(context.exception))

    def test_loads_inverted_vehicle_orientation_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "model-run"
            root.mkdir()
            _write_json(root / "config.json", _model_config())
            _write_json(
                root / "dataset_summary.json",
                {
                    "preprocessing_contract": _preprocessing_contract(
                        ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE
                    )
                },
            )

            manifest = load_live_model_manifest(root)

        self.assertEqual(
            manifest.orientation_image_representation,
            "target_centered_inverted_vehicle_on_white_scaled_by_silhouette_extent",
        )
        self.assertEqual(
            manifest.orientation_image_content,
            "inverted_vehicle_detail_on_white_no_brightness_normalization",
        )
        self.assertEqual(
            manifest.orientation_image_polarity,
            "dark_vehicle_detail_on_white_background",
        )
        self.assertEqual(
            manifest.orientation_source_mode,
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
        )

    def test_real_260430_artifact_resolves_raw_grayscale_if_available(self) -> None:
        artifact_root = (
            REPO_ROOT
            / "05_inference-v0.4-ts/models/distance-orientation"
            / "260430-1023_ts-2d-cnn/runs/run_0002"
        )
        if not artifact_root.is_dir():
            self.skipTest("260430 tri-stream artifact is not available")

        manifest = load_live_model_manifest(artifact_root)

        self.assertEqual(manifest.orientation_source_mode, ORIENTATION_SOURCE_RAW_GRAYSCALE)

    def test_current_live_local_260515_resolves_raw_grayscale_on_white_if_available(self) -> None:
        artifact_root = (
            PROJECT_ROOT
            / "models/distance-orientation/260515-1301_ts-2d-cnn"
        )
        if not artifact_root.is_dir():
            self.skipTest("live-local 260515 tri-stream artifact is not available")

        manifest = load_live_model_manifest(artifact_root)

        self.assertEqual(
            manifest.orientation_source_mode,
            ORIENTATION_SOURCE_RAW_GRAYSCALE_ON_WHITE,
        )

    def test_loads_roi_locator_metadata_when_root_is_provided(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            model_root = Path(tmp_dir) / "model-run"
            roi_root = Path(tmp_dir) / "roi-run"
            model_root.mkdir()
            roi_root.mkdir()
            _write_json(roi_root / "run_config.json", {"roi_width_px": 300, "roi_height_px": 300})
            _write_json(
                roi_root / "dataset_contract.json",
                {
                    "validation_split": {
                        "geometry": {"canvas_width_px": 480, "canvas_height_px": 300}
                    }
                },
            )

            manifest = load_live_model_manifest(model_root, roi_locator_root=roi_root)

        self.assertEqual(manifest.roi_locator_root, roi_root.resolve())
        self.assertEqual(manifest.roi_locator_crop_size, (300, 300))
        self.assertEqual(manifest.roi_locator_canvas_size, (480, 300))
        self.assertIn("roi_run_config", manifest.raw_metadata)
        self.assertIn("roi_dataset_contract", manifest.source_files)

    def test_missing_optional_json_files_leave_fields_empty(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "model-run"
            root.mkdir()

            manifest = load_live_model_manifest(root)

        self.assertEqual(manifest.raw_metadata, {})
        self.assertEqual(manifest.source_files, {})
        self.assertIsNone(manifest.model_label)
        self.assertIsNone(manifest.checkpoint_path)
        self.assertIsNone(manifest.topology_id)
        self.assertIsNone(manifest.topology_contract_version)
        self.assertIsNone(manifest.preprocessing_contract_name)
        self.assertIsNone(manifest.input_mode)
        self.assertIsNone(manifest.representation_kind)
        self.assertEqual(manifest.input_keys, ())
        self.assertEqual(manifest.geometry_schema, ())
        self.assertIsNone(manifest.geometry_dim)
        self.assertIsNone(manifest.orientation_source_mode)
        self.assertEqual(manifest.model_output_keys, ())


if __name__ == "__main__":
    unittest.main()
