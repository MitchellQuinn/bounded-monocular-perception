"""Contract-only tests for the live inference interface layer."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interfaces import (  # noqa: E402
    DEFAULT_FRAME_HASH_ALGORITHM,
    DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES,
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
    is_allowed_worker_state_transition,
)
import interfaces.contracts as contracts  # noqa: E402


class LiveInferenceContractTests(unittest.TestCase):
    def test_default_handoff_paths(self) -> None:
        config = LiveInferenceConfig()

        self.assertEqual(config.handoff_paths.latest_frame_path, Path("live_frames/latest_frame.png"))
        self.assertEqual(config.handoff_paths.temp_frame_path, Path("live_frames/latest_frame.tmp.png"))
        self.assertEqual(config.frame_hash_algorithm, DEFAULT_FRAME_HASH_ALGORITHM)
        self.assertEqual(config.frame_hash_digest_size_bytes, DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES)

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
        self.assertEqual(frame.to_dict()["frame_hash"]["value"], "abc123")
        self.assertEqual(status.to_dict()["worker_name"], "camera")
        self.assertEqual(status.to_dict()["state"], "RUNNING")
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
        self.assertEqual(payload["input_image_hash"]["value"], "abc123")
        self.assertEqual(payload["model_input_mode"], "tri_stream_distance_orientation_geometry")

    def test_public_all_names_exist(self) -> None:
        missing = [name for name in contracts.__all__ if not hasattr(contracts, name)]

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
