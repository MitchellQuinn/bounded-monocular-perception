"""Tests for single-frame inference trace recording."""

from __future__ import annotations

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
from interfaces import (  # noqa: E402
    FrameHash,
    FrameMetadata,
    FrameReference,
    InferenceRequest,
    InferenceResult,
    PreparedInferenceInputs,
    WorkerError,
    WorkerName,
    FrameFailureStage,
)
from live_inference.inspection import (  # noqa: E402
    DEFAULT_TRACE_OUTPUT_DIR,
    InferenceTraceRecorder,
    default_trace_output_dir,
)


CREATED_AT = "2026-05-10T10:30:12Z"


class InferenceTraceRecorderTests(unittest.TestCase):
    def test_record_trace_writes_required_and_available_artifacts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            debug_dir = base_dir / "debug-source"
            debug_dir.mkdir()
            distance_debug = debug_dir / "distance.png"
            distance_debug.write_bytes(b"distance image")
            recorder = InferenceTraceRecorder(
                output_dir=base_dir / "traces",
                context_metadata={
                    "model_selection_path": "models/selections/current.toml",
                    "distance_orientation_root": "distance-root",
                    "roi_fcn_root": "roi-root",
                    "device": "cpu",
                },
            )
            request = _request()
            prepared = _prepared_inputs(
                debug_paths={contracts.TRI_STREAM_DISTANCE_IMAGE_KEY: distance_debug}
            )
            result = _result(debug_paths={contracts.TRI_STREAM_DISTANCE_IMAGE_KEY: distance_debug})
            trace_dir = recorder.create_trace_directory(
                request_id=request.request_id,
                frame_hash=request.frame.frame_hash,
                created_at_utc=CREATED_AT,
            )

            recorded = recorder.record_trace(
                trace_dir=trace_dir,
                image_bytes=b"raw frame bytes",
                request=request,
                prepared_inputs=prepared,
                result=result,
                source_path=Path("live_frames/latest_frame.png"),
                created_at_utc=CREATED_AT,
            )

            self.assertEqual(recorded, trace_dir)
            self.assertTrue((trace_dir / "trace_manifest.json").is_file())
            self.assertEqual((trace_dir / "accepted_raw_frame.png").read_bytes(), b"raw frame bytes")
            self.assertTrue((trace_dir / "inference_result.json").is_file())
            self.assertTrue((trace_dir / "model_outputs.json").is_file())
            self.assertTrue((trace_dir / "preprocessing_metadata.json").is_file())
            self.assertTrue((trace_dir / "mask_background_metadata.json").is_file())
            self.assertTrue((trace_dir / "x_geometry.json").is_file())
            self.assertEqual((trace_dir / "x_distance_image.png").read_bytes(), b"distance image")

            manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
            self.assertEqual(manifest["request_id"], "req-000001")
            self.assertEqual(manifest["input_image_hash_value"], "deadbeefcafebabe")
            self.assertEqual(manifest["source_path"], "live_frames/latest_frame.png")
            self.assertEqual(manifest["model_selection_path"], "models/selections/current.toml")
            self.assertEqual(manifest["roi_fcn_artifact_root"], "roi-root")
            self.assertEqual(manifest["orientation_source_mode"], "raw_grayscale")
            self.assertEqual(manifest["mask_revision"], 7)
            self.assertEqual(manifest["background_revision"], 11)
            self.assertEqual(manifest["app_project_name"], "06_live-inference_v0.2")
            self.assertEqual(manifest["app_root_path"], str(PROJECT_ROOT))
            self.assertIn("argv", manifest)
            self.assertIn("python_executable", manifest)
            self.assertTrue(manifest["distance_orientation_regressor_reached"])

    def test_directory_name_includes_request_id_and_hash_prefix(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            recorder = InferenceTraceRecorder(output_dir=Path(tmp_dir))

            trace_dir = recorder.create_trace_directory(
                request_id="req-000001",
                frame_hash=FrameHash("deadbeefcafebabe"),
                created_at_utc=CREATED_AT,
            )

            self.assertIn("req-000001", trace_dir.name)
            self.assertIn("deadbeef", trace_dir.name)
            self.assertTrue(trace_dir.name.startswith("20260510T103012Z"))

    def test_tolerates_missing_optional_debug_artifacts(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            recorder = InferenceTraceRecorder(output_dir=Path(tmp_dir))
            request = _request()
            prepared = _prepared_inputs(
                debug_paths={contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY: Path("missing.png")}
            )
            trace_dir = recorder.create_trace_directory(
                request_id=request.request_id,
                frame_hash=request.frame.frame_hash,
                created_at_utc=CREATED_AT,
            )

            recorder.record_trace(
                trace_dir=trace_dir,
                image_bytes=b"raw",
                request=request,
                prepared_inputs=prepared,
                result=_result(),
                created_at_utc=CREATED_AT,
            )

            manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
            self.assertEqual(
                manifest["missing_optional_artifacts"],
                [{"kind": contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY, "path": "missing.png"}],
            )
            self.assertTrue((trace_dir / "accepted_raw_frame.png").is_file())

    def test_create_trace_directory_does_not_overwrite_existing_bundle(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            recorder = InferenceTraceRecorder(output_dir=Path(tmp_dir))
            frame_hash = FrameHash("deadbeefcafebabe")
            first = recorder.create_trace_directory(
                request_id="req-000001",
                frame_hash=frame_hash,
                created_at_utc=CREATED_AT,
            )
            marker = first / "marker.txt"
            marker.write_text("keep me", encoding="utf-8")

            second = recorder.create_trace_directory(
                request_id="req-000001",
                frame_hash=frame_hash,
                created_at_utc=CREATED_AT,
            )

            self.assertNotEqual(first, second)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep me")
            self.assertTrue(second.name.endswith("__2"))

    def test_trace_copies_final_locator_input_and_records_stage_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            debug_dir = base_dir / "debug-source"
            debug_dir.mkdir()
            final_locator = debug_dir / "final.png"
            final_locator.write_bytes(b"locator")
            recorder = InferenceTraceRecorder(output_dir=base_dir / "traces")
            request = _request()
            prepared = _prepared_inputs(
                debug_paths={contracts.DISPLAY_ARTIFACT_FINAL_LOCATOR_INPUT: final_locator},
                extra_metadata={
                    "roi_locator_input_mode": "sheet_dark_foreground",
                    contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY: "inverted",
                    "roi_locator_sheet_min_gray": 201,
                    "roi_locator_target_max_gray": 111,
                    "roi_locator_morphology_close_kernel_px": 5,
                    "apply_manual_mask_to_roi_locator": False,
                    "apply_background_removal_to_roi_locator": True,
                    "apply_manual_mask_to_regressor_preprocessing": True,
                    "apply_background_removal_to_regressor_preprocessing": False,
                    "manual_mask_applied_to_roi_locator": False,
                    "background_removal_applied_to_roi_locator": True,
                    contracts.PREPROCESSING_METADATA_ROI_CONFIDENCE: 0.42,
                    contracts.PREPROCESSING_METADATA_ROI_CLIPPED: False,
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX: 0,
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX: 10,
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED: False,
                    contracts.PREPROCESSING_METADATA_ROI_ACCEPTED: True,
                    contracts.PREPROCESSING_METADATA_BACKGROUND_THRESHOLD: 17,
                    "frame_mask_fill_value": 255,
                },
            )
            trace_dir = recorder.create_trace_directory(
                request_id=request.request_id,
                frame_hash=request.frame.frame_hash,
                created_at_utc=CREATED_AT,
            )

            recorder.record_trace(
                trace_dir=trace_dir,
                image_bytes=b"raw",
                request=request,
                prepared_inputs=prepared,
                result=_result(),
                created_at_utc=CREATED_AT,
            )

            self.assertEqual((trace_dir / "final_locator_input.png").read_bytes(), b"locator")
            manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
            self.assertEqual(manifest["locator_input_mode"], "sheet_dark_foreground")
            self.assertEqual(
                manifest["stage_policy_snapshot"]["roi_locator_input_mode"],
                "sheet_dark_foreground",
            )
            self.assertEqual(manifest["locator_input_parameters"]["sheet_min_gray"], 201)
            self.assertEqual(manifest["locator_input_parameters"]["target_max_gray"], 111)
            self.assertEqual(
                manifest["locator_input_parameters"]["morphology_close_kernel_px"],
                5,
            )
            self.assertEqual(manifest["roi_locator_input_polarity"], "inverted")
            self.assertEqual(manifest["roi_confidence"], 0.42)
            self.assertEqual(manifest["roi_clip_max_px"], 0)
            self.assertEqual(manifest["roi_clip_tolerance_px"], 10)
            self.assertFalse(manifest["apply_manual_mask_to_roi_locator"])
            self.assertTrue(manifest["apply_background_removal_to_roi_locator"])
            self.assertTrue(manifest["background_removal_applied_to_roi_locator"])
            self.assertEqual(manifest["background_threshold"], 17)
            self.assertEqual(manifest["fill_value"], 255)
            self.assertEqual(
                manifest["ui_state_snapshot"]["roi_locator_input_mode_dropdown"],
                "sheet_dark_foreground",
            )
            self.assertIsNone(
                manifest["ui_state_snapshot"][
                    "invert_roi_locator_input_checkbox_checked"
                ]
            )

    def test_failure_trace_uses_preprocessing_metadata_from_worker_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            debug_dir = base_dir / "debug-source"
            debug_dir.mkdir()
            final_locator = debug_dir / "final.png"
            final_locator.write_bytes(b"locator")
            recorder = InferenceTraceRecorder(output_dir=base_dir / "traces")
            request = _request()
            metadata = _prepared_inputs(
                debug_paths={contracts.DISPLAY_ARTIFACT_FINAL_LOCATOR_INPUT: final_locator},
                extra_metadata={
                    contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY: "inverted",
                    contracts.PREPROCESSING_METADATA_ROI_CONFIDENCE: 0.12,
                    contracts.PREPROCESSING_METADATA_ROI_CLIPPED: True,
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX: 126,
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX: 10,
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED: False,
                    contracts.PREPROCESSING_METADATA_ROI_ACCEPTED: False,
                    contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASON: (
                        "low_confidence:0.120<min:0.300"
                    ),
                    contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA: {
                        "heatmap_peak_confidence": 0.12
                    },
                },
            ).preprocessing_metadata
            error = WorkerError(
                worker_name=WorkerName.INFERENCE,
                error_type="roi_rejected",
                message="ROI rejected",
                recoverable=True,
                timestamp_utc=CREATED_AT,
                failure_stage=FrameFailureStage.PREPROCESS,
                details={
                    "preprocessing_metadata": metadata,
                    "debug_paths": {
                        contracts.DISPLAY_ARTIFACT_FINAL_LOCATOR_INPUT: final_locator
                    },
                    contracts.PREPROCESSING_METADATA_ROI_ACCEPTED: False,
                },
            )
            trace_dir = recorder.create_trace_directory(
                request_id=request.request_id,
                frame_hash=request.frame.frame_hash,
                created_at_utc=CREATED_AT,
            )

            recorder.record_trace(
                trace_dir=trace_dir,
                image_bytes=b"raw",
                request=request,
                prepared_inputs=None,
                result=None,
                error=error,
                created_at_utc=CREATED_AT,
            )

            self.assertTrue((trace_dir / "failure_result.json").is_file())
            self.assertTrue((trace_dir / "preprocessing_metadata.json").is_file())
            self.assertTrue((trace_dir / "roi_fcn_metadata.json").is_file())
            self.assertEqual((trace_dir / "final_locator_input.png").read_bytes(), b"locator")
            manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
            self.assertFalse(manifest["roi_accepted"])
            self.assertEqual(manifest["roi_clip_max_px"], 126)
            self.assertEqual(manifest["roi_clip_tolerance_px"], 10)
            self.assertEqual(manifest["error_type"], "roi_rejected")
            self.assertEqual(manifest["failure_stage"], "preprocess")

            roi_metadata = json.loads((trace_dir / "roi_fcn_metadata.json").read_text())
            self.assertEqual(roi_metadata["roi_locator_input_polarity"], "inverted")
            self.assertFalse(roi_metadata["roi_clip_tolerated"])

    def test_trace_copies_locator_inputs_before_and_after_polarity(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            debug_dir = base_dir / "debug-source"
            debug_dir.mkdir()
            before = debug_dir / "before.png"
            after = debug_dir / "after.png"
            before.write_bytes(b"before polarity")
            after.write_bytes(b"after polarity")
            recorder = InferenceTraceRecorder(output_dir=base_dir / "traces")
            request = _request()
            prepared = _prepared_inputs(
                debug_paths={
                    contracts.DISPLAY_ARTIFACT_LOCATOR_INPUT_BEFORE_POLARITY: before,
                    contracts.DISPLAY_ARTIFACT_LOCATOR_INPUT_AFTER_POLARITY: after,
                },
                extra_metadata={
                    contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY: "inverted",
                    contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX: 10,
                },
            )
            trace_dir = recorder.create_trace_directory(
                request_id=request.request_id,
                frame_hash=request.frame.frame_hash,
                created_at_utc=CREATED_AT,
            )

            recorder.record_trace(
                trace_dir=trace_dir,
                image_bytes=b"raw",
                request=request,
                prepared_inputs=prepared,
                result=_result(),
                created_at_utc=CREATED_AT,
            )

            self.assertEqual(
                (trace_dir / "locator_input_before_polarity.png").read_bytes(),
                b"before polarity",
            )
            self.assertEqual(
                (trace_dir / "locator_input_after_polarity.png").read_bytes(),
                b"after polarity",
            )
            manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
            self.assertEqual(manifest["roi_locator_input_polarity"], "inverted")
            self.assertEqual(manifest["roi_clip_tolerance_px"], 10)

    def test_default_trace_root_is_derived_from_v02_project_root(self) -> None:
        self.assertEqual(default_trace_output_dir(PROJECT_ROOT), PROJECT_ROOT / "live_traces")
        self.assertEqual(DEFAULT_TRACE_OUTPUT_DIR, PROJECT_ROOT / "live_traces")
        self.assertNotIn("06_live-inference_v0.1", str(DEFAULT_TRACE_OUTPUT_DIR))

    def test_manifest_records_provenance_paths_and_checkpoint_hashes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            distance_checkpoint = base_dir / "distance.pt"
            roi_checkpoint = base_dir / "roi.pt"
            distance_checkpoint.write_bytes(b"distance checkpoint")
            roi_checkpoint.write_bytes(b"roi checkpoint")
            recorder = InferenceTraceRecorder(
                output_dir=base_dir / "traces",
                context_metadata={
                    "app_root_path": str(PROJECT_ROOT),
                    "model_selection_path": str(PROJECT_ROOT / "models/selections/current.toml"),
                    "distance_orientation_root": str(PROJECT_ROOT / "models/distance-orientation/test"),
                    "roi_fcn_root": str(PROJECT_ROOT / "models/roi-fcn/test"),
                    "distance_orientation_checkpoint_path": str(distance_checkpoint),
                    "roi_fcn_checkpoint_path": str(roi_checkpoint),
                    "trace_root_overridden": True,
                },
            )
            request = _request()
            trace_dir = recorder.create_trace_directory(
                request_id=request.request_id,
                frame_hash=request.frame.frame_hash,
                created_at_utc=CREATED_AT,
            )

            recorder.record_trace(
                trace_dir=trace_dir,
                image_bytes=b"raw",
                request=request,
                prepared_inputs=_prepared_inputs(
                    extra_metadata={
                        "diagnostic_profile_name": "baseline_inverted_masked_locator",
                        "roi_locator_input_mode": "inverted",
                        "apply_manual_mask_to_roi_locator": True,
                        "apply_manual_mask_to_regressor_preprocessing": True,
                        "apply_background_removal_to_roi_locator": False,
                        "apply_background_removal_to_regressor_preprocessing": False,
                        "frame_mask_pixel_count": 5,
                        "frame_mask_fill_value": 255,
                        "background_captured": True,
                        "background_removal_enabled": False,
                        "distance_orientation_regressor_reached": True,
                    }
                ),
                result=_result(),
                created_at_utc=CREATED_AT,
            )

            manifest_text = (trace_dir / "trace_manifest.json").read_text()
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["app_root_path"], str(PROJECT_ROOT))
            self.assertEqual(manifest["trace_root_path"], str(base_dir / "traces"))
            self.assertTrue(manifest["trace_root_overridden"])
            self.assertEqual(
                manifest["diagnostic_profile_name"],
                "baseline_inverted_masked_locator",
            )
            self.assertEqual(
                manifest["stage_policy_snapshot"][
                    "apply_manual_mask_to_roi_locator"
                ],
                True,
            )
            self.assertEqual(
                manifest["ui_state_snapshot"]["mask_pixel_count"],
                5,
            )
            self.assertEqual(
                manifest["checkpoint_paths"]["distance_orientation"],
                str(distance_checkpoint),
            )
            self.assertIsNotNone(
                manifest["checkpoint_file_hashes"]["distance_orientation"]["value"]
            )
            self.assertIn("command_line", manifest)
            self.assertNotIn("06_live-inference_v0.1", manifest_text)

    def test_failure_manifest_preserves_foreground_and_locator_stats(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            recorder = InferenceTraceRecorder(output_dir=base_dir / "traces")
            request = _request()
            metadata = _prepared_inputs(
                extra_metadata={
                    "roi_locator_input_mode": "inverted",
                    contracts.PREPROCESSING_METADATA_ROI_CONFIDENCE: 0.9,
                    "roi_center_source_xy_px": (240.0, 150.0),
                    contracts.PREPROCESSING_METADATA_ROI_REQUEST_XYXY_PX: (
                        90.0,
                        0.0,
                        390.0,
                        300.0,
                    ),
                    contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX: (
                        90.0,
                        0.0,
                        390.0,
                        300.0,
                    ),
                    contracts.PREPROCESSING_METADATA_ROI_ACCEPTED: True,
                    "foreground_mask_empty": True,
                    "foreground_pixel_count": 0,
                    "distance_orientation_regressor_reached": False,
                    "final_locator_input_stats": {
                        "min": 0,
                        "max": 255,
                        "mean": 12.5,
                        "median": 0.0,
                        "nonzero_pixel_count": 12,
                        "non_whiteish_pixel_count": 140000,
                    },
                    "final_locator_input_nonzero_pixel_count": 12,
                },
            ).preprocessing_metadata
            error = WorkerError(
                worker_name=WorkerName.INFERENCE,
                error_type="preprocess_failed",
                message="empty foreground",
                recoverable=True,
                timestamp_utc=CREATED_AT,
                failure_stage=FrameFailureStage.PREPROCESS,
                details={"preprocessing_metadata": metadata},
            )
            trace_dir = recorder.create_trace_directory(
                request_id=request.request_id,
                frame_hash=request.frame.frame_hash,
                created_at_utc=CREATED_AT,
            )

            recorder.record_trace(
                trace_dir=trace_dir,
                image_bytes=b"raw",
                request=request,
                prepared_inputs=None,
                result=None,
                error=error,
                created_at_utc=CREATED_AT,
            )

            manifest = json.loads((trace_dir / "trace_manifest.json").read_text())
            self.assertEqual(manifest["foreground_pixel_count"], 0)
            self.assertTrue(manifest["foreground_mask_empty"])
            self.assertEqual(
                manifest["final_locator_input_stats"]["nonzero_pixel_count"],
                12,
            )
            self.assertFalse(manifest["distance_orientation_regressor_reached"])
            roi_metadata = json.loads((trace_dir / "roi_fcn_metadata.json").read_text())
            self.assertEqual(roi_metadata["foreground_pixel_count"], 0)


def _request() -> InferenceRequest:
    frame_hash = FrameHash("deadbeefcafebabe")
    return InferenceRequest(
        request_id="req-000001",
        frame=FrameReference(
            image_path=Path("live_frames/latest_frame.png"),
            frame_hash=frame_hash,
            metadata=FrameMetadata(frame_id="frame-1"),
        ),
        requested_at_utc=CREATED_AT,
    )


def _prepared_inputs(
    *,
    debug_paths: dict[str, Path] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> PreparedInferenceInputs:
    metadata = {
        "preprocessing_contract_name": "test-preprocess",
        "orientation_source_mode": "raw_grayscale",
        contracts.PREPROCESSING_METADATA_RUNTIME_PARAMETER_REVISION: 3,
        "frame_mask_revision": 7,
        contracts.PREPROCESSING_METADATA_BACKGROUND_REVISION: 11,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if debug_paths:
        metadata[contracts.PREPROCESSING_METADATA_DEBUG_PATHS] = debug_paths
    return PreparedInferenceInputs(
        request_id="req-000001",
        source_frame=_request().frame,
        preprocessing_metadata=metadata,
        model_inputs={contracts.TRI_STREAM_GEOMETRY_KEY: [1.0, 2.0, 3.0]},
    )


def _result(*, debug_paths: dict[str, Path] | None = None) -> InferenceResult:
    return InferenceResult(
        request_id="req-000001",
        input_image_path=Path("live_frames/latest_frame.png"),
        input_image_hash=FrameHash("deadbeefcafebabe"),
        timestamp_utc=CREATED_AT,
        predicted_distance_m=4.5,
        predicted_yaw_sin=0.0,
        predicted_yaw_cos=1.0,
        predicted_yaw_deg=0.0,
        inference_time_ms=12.5,
        preprocessing_parameter_revision=3,
        debug_paths=debug_paths or {},
        extras={"device": "cpu", "model_root": "distance-root"},
    )


if __name__ == "__main__":
    unittest.main()
