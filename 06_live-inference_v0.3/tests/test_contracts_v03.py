from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import interfaces.contracts as contracts  # noqa: E402


class ContractsV03Tests(unittest.TestCase):
    def test_contract_version_bumped(self) -> None:
        self.assertEqual(
            contracts.LIVE_INFERENCE_CONTRACT_VERSION,
            "rb-live-inference-v0_3",
        )

    def test_geometry_schema_is_unchanged(self) -> None:
        self.assertEqual(
            contracts.TRI_STREAM_GEOMETRY_SCHEMA,
            (
                "cx_px",
                "cy_px",
                "w_px",
                "h_px",
                "cx_norm",
                "cy_norm",
                "w_norm",
                "h_norm",
                "aspect_ratio",
                "area_norm",
            ),
        )

    def test_foreground_extraction_modes_are_contract_values(self) -> None:
        self.assertEqual(
            contracts.ForegroundExtractionMode.THRESHOLD_FOREGROUND_V1.value,
            "threshold_foreground_v1",
        )
        self.assertEqual(
            contracts.ForegroundExtractionMode.SILHOUETTE_CONTOUR_V2.value,
            "silhouette_contour_v2",
        )

    def test_camera_intrinsics_modes_are_contract_values(self) -> None:
        self.assertEqual(
            contracts.CameraIntrinsicsMode.DISABLED.value,
            "disabled",
        )
        self.assertEqual(
            contracts.CameraIntrinsicsMode.REAL_TO_UNITY_INTRINSICS_REMAP.value,
            "real_to_unity_intrinsics_remap",
        )
        self.assertEqual(
            contracts.CameraIntrinsicsMode.REAL_UNDISTORT_ONLY.value,
            "real_undistort_only",
        )
        self.assertEqual(
            contracts.PREPROCESSING_METADATA_CAMERA_INTRINSICS_MODE,
            contracts.PREPROCESSING_RUNTIME_PARAMETER_CAMERA_INTRINSICS_MODE,
        )

    def test_locator_result_serializes_plain_values(self) -> None:
        candidate = contracts.RoiCandidate(
            candidate_id="c0",
            bbox_xyxy_px=(10.0, 20.0, 30.0, 40.0),
            center_xy_px=(20.0, 30.0),
            area_px=100.0,
            contour_area_px=80.0,
            bbox_area_px=200.0,
            aspect_ratio=1.0,
            score=0.8,
        )
        result = contracts.LocatorResult(
            request_id="r0",
            locator_kind=contracts.LocatorKind.BACKGROUND_EDGE_V1,
            accepted=True,
            confidence=0.8,
            source_image_wh_px=(640, 480),
            chosen_candidate=candidate,
            candidates=(candidate,),
        )

        payload = result.to_dict()

        self.assertEqual(payload["locator_kind"], "background_edge_v1")
        self.assertEqual(payload["chosen_candidate"]["candidate_id"], "c0")

    def test_live_performance_metrics_are_contract_values(self) -> None:
        metrics = contracts.LivePerformanceMetrics(
            camera_raw_fps=79.5,
            inference_fps=12.25,
            live_inference_running=True,
            camera_frame_sample_count=80,
            inference_sample_count=12,
        )

        payload = metrics.to_dict()

        self.assertEqual(contracts.PERFORMANCE_METRIC_CAMERA_RAW_FPS, "camera_raw_fps")
        self.assertEqual(contracts.PERFORMANCE_METRIC_INFERENCE_FPS, "inference_fps")
        self.assertEqual(payload["camera_raw_fps"], 79.5)
        self.assertEqual(payload["inference_fps"], 12.25)
        self.assertTrue(payload["live_inference_running"])

    def test_contracts_module_stays_dependency_light(self) -> None:
        tree = ast.parse(
            (SRC_ROOT / "live_inference/interfaces/contracts.py").read_text(
                encoding="utf-8"
            )
        )
        imports = {
            alias.name.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module.split(".")[0]
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        )
        self.assertFalse(
            imports
            & {
                "PySide6",
                "cv2",
                "numpy",
                "torch",
                "live_inference",
                "cameras",
            }
        )


if __name__ == "__main__":
    unittest.main()
