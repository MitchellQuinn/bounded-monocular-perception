"""Contract-only tests for the live inference interface layer."""

from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from interfaces import (  # noqa: E402
    DEFAULT_FRAME_HASH_ALGORITHM,
    DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES,
    DISPLAY_ARTIFACT_ACCEPTED_RAW_FRAME,
    DISPLAY_ARTIFACT_DISTANCE_IMAGE,
    DISPLAY_ARTIFACT_ORIENTATION_IMAGE,
    DISPLAY_ARTIFACT_ROI_OVERLAY,
    LIVE_INFERENCE_CONTRACT_VERSION,
    TRI_STREAM_DISTANCE_IMAGE_KEY,
    TRI_STREAM_GEOMETRY_KEY,
    TRI_STREAM_INPUT_KEYS,
    TRI_STREAM_ORIENTATION_IMAGE_KEY,
    FrameFailureStage,
    FrameHash,
    FrameReference,
    FrameSkipReason,
    FrameSkipped,
    InferenceInputMode,
    InferenceResult,
    LiveInferenceConfig,
    PreparedInferenceInputs,
    RuntimeParameterSetSpec,
    RuntimeParameterSpec,
    RuntimeParameterUpdate,
    RuntimeParameterUpdateResult,
    RuntimeParameterValueType,
    RuntimeParameterWidgetHint,
    WorkerName,
    WorkerState,
    WorkerStatus,
    contract_version_matches,
    get_contract_version,
    is_allowed_worker_state_transition,
    require_contract_version,
)
import interfaces.contracts as contracts  # noqa: E402


class LiveInferenceContractTests(unittest.TestCase):
    def _sample_instances(self) -> dict[str, object]:
        frame_hash = contracts.FrameHash("abc123")
        frame = contracts.FrameReference(
            image_path=Path("live_frames/latest_frame.png"),
            frame_hash=frame_hash,
        )
        parameter_spec = contracts.RuntimeParameterSpec(
            name="threshold",
            label="Threshold",
            value_type=contracts.RuntimeParameterValueType.FLOAT,
            default_value=0.5,
            current_value=0.75,
            widget_hint=contracts.RuntimeParameterWidgetHint.SLIDER,
        )
        return {
            "InferenceOutputContract": contracts.InferenceOutputContract(),
            "ModelContractReference": contracts.ModelContractReference(),
            "FrameHandoffPaths": contracts.FrameHandoffPaths(),
            "LiveInferenceConfig": contracts.LiveInferenceConfig(),
            "FrameMetadata": contracts.FrameMetadata(),
            "FrameHash": frame_hash,
            "FrameReference": frame,
            "InferenceRequest": contracts.InferenceRequest(
                request_id="request-1",
                frame=frame,
                requested_at_utc="2026-05-01T10:00:00Z",
            ),
            "PreparedInferenceInputs": contracts.PreparedInferenceInputs(
                request_id="request-1",
            ),
            "RoiMetadata": contracts.RoiMetadata(),
            "InferenceResult": contracts.InferenceResult(
                request_id="request-1",
                input_image_path=Path("live_frames/latest_frame.png"),
                input_image_hash=frame_hash,
                timestamp_utc="2026-05-01T10:00:00Z",
                predicted_distance_m=4.5,
                predicted_yaw_sin=0.0,
                predicted_yaw_cos=1.0,
                predicted_yaw_deg=0.0,
                inference_time_ms=12.5,
            ),
            "DebugImageReference": contracts.DebugImageReference(
                request_id="request-1",
                image_kind="distance",
                path=Path("live_debug/distance.png"),
                created_at_utc="2026-05-01T10:00:00Z",
            ),
            "RuntimeParameterSpec": parameter_spec,
            "RuntimeParameterSetSpec": contracts.RuntimeParameterSetSpec(
                owner=contracts.WorkerName.INFERENCE,
                namespace="preprocessing",
                revision=3,
                parameters=(parameter_spec,),
                timestamp_utc="2026-05-01T10:00:00Z",
            ),
            "RuntimeParameterUpdate": contracts.RuntimeParameterUpdate(
                owner=contracts.WorkerName.INFERENCE,
                namespace="preprocessing",
                updates={"threshold": 0.8},
                requested_at_utc="2026-05-01T10:00:00Z",
            ),
            "RuntimeParameterUpdateResult": contracts.RuntimeParameterUpdateResult(
                owner=contracts.WorkerName.INFERENCE,
                namespace="preprocessing",
                accepted=True,
                revision=4,
                timestamp_utc="2026-05-01T10:00:00Z",
            ),
            "CameraWorkerCounters": contracts.CameraWorkerCounters(),
            "InferenceWorkerCounters": contracts.InferenceWorkerCounters(),
            "WorkerStatus": contracts.WorkerStatus(
                worker_name=contracts.WorkerName.CAMERA,
                state=contracts.WorkerState.RUNNING,
                message="capturing",
                timestamp_utc="2026-05-01T10:00:00Z",
            ),
            "WorkerLifecycleEvent": contracts.WorkerLifecycleEvent(
                worker_name=contracts.WorkerName.CAMERA,
                event_type=contracts.WorkerEventType.STARTED,
                state=contracts.WorkerState.RUNNING,
                timestamp_utc="2026-05-01T10:00:00Z",
            ),
            "FrameSkipped": contracts.FrameSkipped(
                worker_name=contracts.WorkerName.INFERENCE,
                reason=contracts.FrameSkipReason.DUPLICATE_HASH,
                timestamp_utc="2026-05-01T10:00:00Z",
            ),
            "WorkerWarning": contracts.WorkerWarning(
                worker_name=contracts.WorkerName.INFERENCE,
                warning_type="read_warning",
                message="warning",
                timestamp_utc="2026-05-01T10:00:00Z",
            ),
            "WorkerError": contracts.WorkerError(
                worker_name=contracts.WorkerName.INFERENCE,
                error_type="read_error",
                message="error",
                recoverable=True,
                timestamp_utc="2026-05-01T10:00:00Z",
            ),
        }

    def test_default_handoff_paths(self) -> None:
        config = LiveInferenceConfig()

        self.assertEqual(config.handoff_paths.latest_frame_path, Path("live_frames/latest_frame.png"))
        self.assertEqual(config.handoff_paths.temp_frame_path, Path("live_frames/latest_frame.tmp.png"))
        self.assertEqual(config.frame_hash_algorithm, DEFAULT_FRAME_HASH_ALGORITHM)
        self.assertEqual(config.frame_hash_digest_size_bytes, DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES)
        self.assertEqual(config.contract_version, LIVE_INFERENCE_CONTRACT_VERSION)

    def test_public_dataclasses_have_contract_version_field(self) -> None:
        missing: list[str] = []
        for name in contracts.__all__:
            value = getattr(contracts, name, None)
            if isinstance(value, type) and is_dataclass(value):
                field_names = {field.name for field in fields(value)}
                if "contract_version" not in field_names:
                    missing.append(name)

        self.assertEqual(missing, [])

    def test_public_dataclass_instances_have_default_contract_version(self) -> None:
        for name, instance in self._sample_instances().items():
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(instance, "contract_version"),
                    LIVE_INFERENCE_CONTRACT_VERSION,
                )

    def test_prepared_inputs_accept_tri_stream_model_inputs(self) -> None:
        model_inputs = {
            TRI_STREAM_DISTANCE_IMAGE_KEY: object(),
            TRI_STREAM_ORIENTATION_IMAGE_KEY: object(),
            TRI_STREAM_GEOMETRY_KEY: object(),
        }

        prepared = PreparedInferenceInputs(
            request_id="request-1",
            model_inputs=model_inputs,
        )

        self.assertEqual(prepared.input_mode, InferenceInputMode.TRI_STREAM_V0_4)
        self.assertEqual(prepared.input_keys, TRI_STREAM_INPUT_KEYS)
        self.assertEqual(set(prepared.model_inputs), set(TRI_STREAM_INPUT_KEYS))

    def test_prepared_inputs_to_dict_hides_model_input_payloads(self) -> None:
        class DeepCopyBlocked:
            def __deepcopy__(self, memo: dict[object, object]) -> object:
                raise AssertionError("model_inputs payload must not be deep-copied")

        prepared = PreparedInferenceInputs(
            request_id="request-1",
            model_inputs={
                TRI_STREAM_DISTANCE_IMAGE_KEY: DeepCopyBlocked(),
                TRI_STREAM_ORIENTATION_IMAGE_KEY: DeepCopyBlocked(),
                TRI_STREAM_GEOMETRY_KEY: DeepCopyBlocked(),
            },
        )

        payload = prepared.to_dict()

        self.assertNotIn("model_inputs", payload)
        self.assertEqual(payload["contract_version"], LIVE_INFERENCE_CONTRACT_VERSION)
        self.assertEqual(payload["request_id"], "request-1")
        self.assertEqual(payload["input_mode"], "tri_stream_distance_orientation_geometry")
        self.assertEqual(payload["input_keys"], list(TRI_STREAM_INPUT_KEYS))
        self.assertEqual(set(payload["model_input_keys"]), set(TRI_STREAM_INPUT_KEYS))

    def test_skip_and_failure_vocabularies_are_distinct(self) -> None:
        skip_values = {reason.value for reason in FrameSkipReason}

        self.assertIn("duplicate_hash", skip_values)
        self.assertIn("missing_file", skip_values)
        self.assertIn("unreadable_file", skip_values)
        self.assertIn("stale_frame", skip_values)
        self.assertNotIn("decode_failed", skip_values)
        self.assertNotIn("preprocess_failed", skip_values)
        self.assertNotIn("inference_failed", skip_values)

        self.assertEqual(FrameFailureStage.DECODE.value, "decode")
        self.assertEqual(FrameFailureStage.PREPROCESS.value, "preprocess")
        self.assertEqual(FrameFailureStage.INFERENCE.value, "inference")

    def test_display_artifact_kind_constants(self) -> None:
        self.assertEqual(DISPLAY_ARTIFACT_ACCEPTED_RAW_FRAME, "accepted_raw_frame")
        self.assertEqual(DISPLAY_ARTIFACT_DISTANCE_IMAGE, "x_distance_image")
        self.assertEqual(DISPLAY_ARTIFACT_ORIENTATION_IMAGE, "x_orientation_image")
        self.assertEqual(DISPLAY_ARTIFACT_ROI_OVERLAY, "roi_overlay")
        self.assertEqual(contracts.DISPLAY_ARTIFACT_ROI_CROP, "roi_crop")
        self.assertEqual(contracts.DISPLAY_ARTIFACT_LOCATOR_INPUT, "locator_input")
        self.assertEqual(
            contracts.DISPLAY_ARTIFACT_ROI_OVERLAY_METADATA,
            "roi_overlay_metadata",
        )
        self.assertIn(
            contracts.DISPLAY_ARTIFACT_ROI_CROP,
            contracts.DISPLAY_DEBUG_ARTIFACT_KEYS,
        )

    def test_preprocessing_metadata_bridge_constants(self) -> None:
        self.assertEqual(
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA,
            "roi_locator_metadata",
        )
        self.assertEqual(
            contracts.PREPROCESSING_METADATA_BACKGROUND_APPLICATION_SPACE,
            "background_application_space",
        )
        self.assertEqual(
            contracts.PREPROCESSING_METADATA_ROI_FCN_HEATMAP_U8,
            "roi_fcn_heatmap_u8",
        )
        self.assertEqual(
            contracts.PREPROCESSING_METADATA_ROI_FCN_RESIZED_IMAGE_WH_PX,
            "resized_image_wh_px",
        )
        self.assertEqual(
            contracts.PREPROCESSING_METADATA_ROI_FCN_PADDING_LTRB_PX,
            "padding_ltrb_px",
        )
        self.assertEqual(
            contracts.BACKGROUND_APPLICATION_SPACE_ROI_FCN_INPUT_AND_ROI_CROP,
            "roi_fcn_input_and_roi_crop",
        )
        self.assertIn(
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA,
            contracts.ROI_METADATA_EXTRA_KEYS,
        )
        self.assertIn(
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED,
            contracts.BACKGROUND_REMOVAL_METADATA_KEYS,
        )
        self.assertIn(
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_EXCLUDED_FROM_FOREGROUND,
            contracts.BACKGROUND_REMOVAL_METADATA_KEYS,
        )
        self.assertEqual(
            contracts.ROI_OVERLAY_BOUNDS_METADATA_KEYS,
            (
                contracts.PREPROCESSING_METADATA_ROI_LOCATOR_BOUNDS_XYXY_PX,
                contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX,
                contracts.PREPROCESSING_METADATA_ROI_REQUEST_XYXY_PX,
            ),
        )

    def test_inference_protocols_include_runtime_parameter_hooks(self) -> None:
        self.assertTrue(hasattr(contracts.InferenceWorkerProtocol, "update_runtime_parameters"))
        self.assertTrue(hasattr(contracts.InferenceWorkerEventSink, "runtime_parameters_available"))
        self.assertTrue(hasattr(contracts.InferenceWorkerEventSink, "runtime_parameter_update_result"))

    def test_worker_state_transition_helper(self) -> None:
        self.assertTrue(
            is_allowed_worker_state_transition(WorkerState.STOPPED, WorkerState.STARTING)
        )
        self.assertTrue(
            is_allowed_worker_state_transition(WorkerState.RUNNING, WorkerState.STOPPING)
        )
        self.assertTrue(
            is_allowed_worker_state_transition(WorkerState.ERROR, WorkerState.STOPPED)
        )
        self.assertTrue(
            is_allowed_worker_state_transition(WorkerState.RUNNING, WorkerState.RUNNING)
        )
        self.assertFalse(
            is_allowed_worker_state_transition(
                WorkerState.RUNNING,
                WorkerState.RUNNING,
                allow_idempotent=False,
            )
        )
        self.assertFalse(
            is_allowed_worker_state_transition(WorkerState.STOPPED, WorkerState.RUNNING)
        )

    def test_to_dict_converts_paths_enums_and_hashes(self) -> None:
        frame_hash = FrameHash("abc123")
        frame = FrameReference(
            image_path=Path("live_frames/latest_frame.png"),
            frame_hash=frame_hash,
        )
        status = WorkerStatus(
            worker_name=WorkerName.CAMERA,
            state=WorkerState.RUNNING,
            message="capturing",
            timestamp_utc="2026-05-01T10:00:00Z",
        )

        self.assertEqual(frame.to_dict()["image_path"], "live_frames/latest_frame.png")
        self.assertEqual(frame.to_dict()["contract_version"], LIVE_INFERENCE_CONTRACT_VERSION)
        self.assertEqual(frame.to_dict()["frame_hash"]["value"], "abc123")
        self.assertEqual(status.to_dict()["worker_name"], "camera")
        self.assertEqual(status.to_dict()["state"], "RUNNING")
        self.assertEqual(status.to_dict()["contract_version"], LIVE_INFERENCE_CONTRACT_VERSION)
        self.assertEqual(frame_hash.to_dict()["contract_version"], LIVE_INFERENCE_CONTRACT_VERSION)
        self.assertEqual(frame_hash.to_dict()["digest_size_bytes"], 16)

    def test_frame_skipped_to_dict(self) -> None:
        skipped = FrameSkipped(
            worker_name=WorkerName.INFERENCE,
            reason=FrameSkipReason.DUPLICATE_HASH,
            timestamp_utc="2026-05-01T10:00:00Z",
            frame_hash=FrameHash("abc123"),
        )

        payload = skipped.to_dict()

        self.assertEqual(payload["worker_name"], "inference")
        self.assertEqual(payload["reason"], "duplicate_hash")
        self.assertEqual(payload["contract_version"], LIVE_INFERENCE_CONTRACT_VERSION)
        self.assertEqual(payload["frame_hash"]["algorithm"], DEFAULT_FRAME_HASH_ALGORITHM)

    def test_inference_result_to_dict_uses_frame_hash_object(self) -> None:
        result = InferenceResult(
            request_id="request-1",
            input_image_path=Path("live_frames/latest_frame.png"),
            input_image_hash=FrameHash("abc123"),
            timestamp_utc="2026-05-01T10:00:00Z",
            predicted_distance_m=4.5,
            predicted_yaw_sin=0.0,
            predicted_yaw_cos=1.0,
            predicted_yaw_deg=0.0,
            inference_time_ms=12.5,
        )

        payload = result.to_dict()

        self.assertEqual(payload["input_image_path"], "live_frames/latest_frame.png")
        self.assertEqual(payload["contract_version"], LIVE_INFERENCE_CONTRACT_VERSION)
        self.assertEqual(payload["input_image_hash"]["value"], "abc123")
        self.assertEqual(payload["model_input_mode"], "tri_stream_distance_orientation_geometry")

    def test_runtime_parameter_spec_to_dict_converts_enums_to_strings(self) -> None:
        spec = RuntimeParameterSpec(
            name="edge_threshold",
            label="Edge threshold",
            value_type=RuntimeParameterValueType.FLOAT,
            default_value=0.4,
            current_value=0.55,
            widget_hint=RuntimeParameterWidgetHint.SLIDER,
            group="preprocessing",
            description="Synthetic threshold exposed for runtime tuning.",
            minimum=0.0,
            maximum=1.0,
            step=0.05,
            choices=(0.25, 0.5, 0.75),
        )

        payload = spec.to_dict()

        self.assertEqual(payload["name"], "edge_threshold")
        self.assertEqual(payload["value_type"], "float")
        self.assertEqual(payload["widget_hint"], "slider")
        self.assertEqual(payload["choices"], [0.25, 0.5, 0.75])
        self.assertEqual(payload["contract_version"], LIVE_INFERENCE_CONTRACT_VERSION)

    def test_runtime_parameter_set_spec_to_dict(self) -> None:
        spec = RuntimeParameterSpec(
            name="mode",
            label="Mode",
            value_type=RuntimeParameterValueType.ENUM,
            default_value="fast",
            current_value="fast",
            widget_hint=RuntimeParameterWidgetHint.DROPDOWN,
            choices=("fast", "accurate"),
        )
        parameter_set = RuntimeParameterSetSpec(
            owner=WorkerName.INFERENCE,
            namespace="preprocessing",
            revision=7,
            parameters=(spec,),
            timestamp_utc="2026-05-01T10:00:00Z",
        )

        payload = parameter_set.to_dict()

        self.assertEqual(payload["owner"], "inference")
        self.assertEqual(payload["namespace"], "preprocessing")
        self.assertEqual(payload["revision"], 7)
        self.assertEqual(payload["parameters"][0]["value_type"], "enum")
        self.assertEqual(payload["parameters"][0]["widget_hint"], "dropdown")
        self.assertEqual(payload["timestamp_utc"], "2026-05-01T10:00:00Z")

    def test_runtime_parameter_update_to_dict(self) -> None:
        update = RuntimeParameterUpdate(
            owner=WorkerName.INFERENCE,
            namespace="preprocessing",
            updates={"edge_threshold": 0.65, "enabled": True},
            requested_at_utc="2026-05-01T10:00:00Z",
            base_revision=7,
        )

        payload = update.to_dict()

        self.assertEqual(payload["owner"], "inference")
        self.assertEqual(payload["namespace"], "preprocessing")
        self.assertEqual(payload["updates"], {"edge_threshold": 0.65, "enabled": True})
        self.assertEqual(payload["requested_at_utc"], "2026-05-01T10:00:00Z")
        self.assertEqual(payload["base_revision"], 7)

    def test_runtime_parameter_update_result_to_dict(self) -> None:
        result = RuntimeParameterUpdateResult(
            owner=WorkerName.INFERENCE,
            namespace="preprocessing",
            accepted=False,
            revision=7,
            applied_updates={"edge_threshold": 0.65},
            rejected_updates={"mode": "unsupported choice"},
            message="partial update",
            timestamp_utc="2026-05-01T10:00:00Z",
        )

        payload = result.to_dict()

        self.assertEqual(payload["owner"], "inference")
        self.assertFalse(payload["accepted"])
        self.assertEqual(payload["revision"], 7)
        self.assertEqual(payload["applied_updates"], {"edge_threshold": 0.65})
        self.assertEqual(payload["rejected_updates"], {"mode": "unsupported choice"})
        self.assertEqual(payload["message"], "partial update")
        self.assertEqual(payload["timestamp_utc"], "2026-05-01T10:00:00Z")
        self.assertEqual(payload["contract_version"], LIVE_INFERENCE_CONTRACT_VERSION)

    def test_inference_result_can_carry_preprocessing_parameter_revision(self) -> None:
        result = InferenceResult(
            request_id="request-1",
            input_image_path=Path("live_frames/latest_frame.png"),
            input_image_hash=FrameHash("abc123"),
            timestamp_utc="2026-05-01T10:00:00Z",
            predicted_distance_m=4.5,
            predicted_yaw_sin=0.0,
            predicted_yaw_cos=1.0,
            predicted_yaw_deg=0.0,
            inference_time_ms=12.5,
            preprocessing_parameter_revision=9,
        )

        self.assertEqual(result.preprocessing_parameter_revision, 9)
        self.assertEqual(result.to_dict()["preprocessing_parameter_revision"], 9)

    def test_debug_image_reference_can_carry_model_input_key_and_parameter_revision(self) -> None:
        image = contracts.DebugImageReference(
            request_id="request-1",
            image_kind=DISPLAY_ARTIFACT_DISTANCE_IMAGE,
            path=Path("live_debug/distance.png"),
            created_at_utc="2026-05-01T10:00:00Z",
            source_frame_hash=FrameHash("abc123"),
            model_input_key=TRI_STREAM_DISTANCE_IMAGE_KEY,
            parameter_revision=9,
            label="Distance image",
        )

        payload = image.to_dict()

        self.assertEqual(payload["source_frame_hash"]["value"], "abc123")
        self.assertEqual(payload["model_input_key"], TRI_STREAM_DISTANCE_IMAGE_KEY)
        self.assertEqual(payload["parameter_revision"], 9)
        self.assertEqual(payload["label"], "Distance image")

    def test_roi_metadata_includes_bbox_coordinate_space(self) -> None:
        roi = contracts.RoiMetadata(
            bbox_xyxy_px=(1.0, 2.0, 30.0, 40.0),
            source_image_wh_px=(640, 480),
        )

        payload = roi.to_dict()

        self.assertEqual(roi.bbox_coordinate_space, "source_image_px")
        self.assertEqual(payload["bbox_coordinate_space"], "source_image_px")
        self.assertEqual(payload["bbox_xyxy_px"], [1.0, 2.0, 30.0, 40.0])

    def test_contracts_module_keeps_heavy_runtime_imports_out(self) -> None:
        source = Path(contracts.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        banned_roots = {"PySide6", "cv2", "numpy", "torch"}
        found: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".", maxsplit=1)[0])

        self.assertEqual(found & banned_roots, set())

    def test_contract_version_helpers(self) -> None:
        frame_hash = FrameHash("abc123")
        matching_payload = {"contract_version": LIVE_INFERENCE_CONTRACT_VERSION}
        mismatched_payload = {"contract_version": "rb-live-inference-v9_9"}
        missing_payload = {"message": "no version"}

        self.assertEqual(get_contract_version(frame_hash), LIVE_INFERENCE_CONTRACT_VERSION)
        self.assertEqual(
            get_contract_version(matching_payload),
            LIVE_INFERENCE_CONTRACT_VERSION,
        )
        self.assertIsNone(get_contract_version(missing_payload))
        self.assertIsNone(get_contract_version(object()))

        self.assertTrue(contract_version_matches(frame_hash))
        self.assertTrue(contract_version_matches(matching_payload))
        self.assertFalse(contract_version_matches(mismatched_payload))
        self.assertFalse(contract_version_matches(missing_payload))

        require_contract_version(frame_hash)
        require_contract_version(matching_payload)
        with self.assertRaisesRegex(ValueError, "expected .* actual missing"):
            require_contract_version(missing_payload)
        with self.assertRaisesRegex(ValueError, "rb-live-inference-v9_9"):
            require_contract_version(mismatched_payload)

    def test_public_all_names_exist(self) -> None:
        missing = [name for name in contracts.__all__ if not hasattr(contracts, name)]

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
