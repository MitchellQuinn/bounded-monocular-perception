"""Contract-only tests for the live inference interface layer."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interfaces import (  # noqa: E402
    DEFAULT_FRAME_HASH_ALGORITHM,
    DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES,
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
