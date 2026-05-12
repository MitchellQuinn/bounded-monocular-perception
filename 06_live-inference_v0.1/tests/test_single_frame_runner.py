"""Tests for single-frame diagnostic inference runner."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from interfaces import (  # noqa: E402
    FrameFailureStage,
    FrameHash,
    FrameMetadata,
    FrameReference,
    InferenceRequest,
    InferenceResult,
    PreparedInferenceInputs,
)
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402
from live_inference.inspection import (  # noqa: E402
    InferenceTraceRecorder,
    SingleFrameInferenceRunner,
)


REQUESTED_AT = "2026-05-10T10:30:12Z"
RESULT_AT = "2026-05-10T10:30:13Z"


class SingleFrameInferenceRunnerTests(unittest.TestCase):
    def test_successful_run_processes_exact_captured_bytes(self) -> None:
        preprocessor = FakePreprocessor()
        engine = FakeEngine()
        runner = _runner(preprocessor, engine)
        captured = b"captured-at-click-time"

        outcome = runner.run_single_frame(captured, source_path=Path("latest_frame.png"))

        self.assertIsNotNone(outcome.result)
        self.assertIsNone(outcome.error)
        self.assertEqual(preprocessor.calls[0][1], captured)
        self.assertEqual(engine.calls[0].request_id, "req-000001")

    def test_computes_frame_hash_from_captured_bytes(self) -> None:
        runner = _runner(FakePreprocessor(), FakeEngine())
        captured = b"hash these bytes"

        outcome = runner.run_single_frame(captured)

        self.assertEqual(outcome.frame_hash, compute_frame_hash(captured))
        self.assertEqual(outcome.result.input_image_hash, outcome.frame_hash)

    def test_preserves_request_id_and_frame_hash(self) -> None:
        runner = _runner(FakePreprocessor(), FakeEngine())

        outcome = runner.run_single_frame(b"frame")

        self.assertEqual(outcome.request_id, "req-000001")
        self.assertEqual(outcome.result.request_id, "req-000001")
        self.assertEqual(outcome.result.input_image_hash, outcome.frame_hash)

    def test_duplicate_hash_does_not_skip_single_frame_reprocessing(self) -> None:
        preprocessor = FakePreprocessor()
        engine = FakeEngine()
        runner = _runner(preprocessor, engine)

        first = runner.run_single_frame(b"same")
        second = runner.run_single_frame(b"same")

        self.assertIsNotNone(first.result)
        self.assertIsNotNone(second.result)
        self.assertEqual(len(preprocessor.calls), 2)
        self.assertFalse(preprocessor.calls[0][0].duplicate_hash_skip_enabled)
        self.assertFalse(preprocessor.calls[1][0].duplicate_hash_skip_enabled)

    def test_preprocessor_failure_returns_worker_error(self) -> None:
        runner = _runner(
            FakePreprocessor(exception=ValueError("decode failed")),
            FakeEngine(),
        )

        outcome = runner.run_single_frame(b"bad-image")

        self.assertIsNone(outcome.result)
        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.error.failure_stage, FrameFailureStage.PREPROCESS)
        self.assertEqual(outcome.error.error_type, "preprocess_failed")

    def test_engine_failure_returns_worker_error(self) -> None:
        runner = _runner(
            FakePreprocessor(),
            FakeEngine(exception=RuntimeError("model failed")),
        )

        outcome = runner.run_single_frame(b"frame")

        self.assertIsNone(outcome.result)
        self.assertIsNotNone(outcome.error)
        self.assertEqual(outcome.error.failure_stage, FrameFailureStage.INFERENCE)
        self.assertEqual(outcome.error.error_type, "inference_failed")

    def test_trace_path_is_returned_when_recording_enabled(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            runner = _runner(
                FakePreprocessor(),
                FakeEngine(),
                trace_recorder=InferenceTraceRecorder(output_dir=Path(tmp_dir)),
            )

            outcome = runner.run_single_frame(b"frame", record_trace=True)

            self.assertIsNotNone(outcome.result)
            self.assertIsNone(outcome.error)
            self.assertIsNotNone(outcome.trace_path)
            assert outcome.trace_path is not None
            self.assertTrue((outcome.trace_path / "trace_manifest.json").is_file())
            self.assertEqual((outcome.trace_path / "accepted_raw_frame.png").read_bytes(), b"frame")

    def test_runner_module_has_no_gui_imports(self) -> None:
        module_path = SRC_ROOT / "live_inference/inspection/single_frame_runner.py"
        imported_roots = _imported_roots(module_path)

        self.assertNotIn("PySide6", imported_roots)


class FakePreprocessor:
    def __init__(self, *, exception: Exception | None = None) -> None:
        self.exception = exception
        self.calls: list[tuple[InferenceRequest, bytes]] = []

    def prepare_model_inputs(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
    ) -> PreparedInferenceInputs:
        self.calls.append((request, image_bytes))
        if self.exception is not None:
            raise self.exception
        return PreparedInferenceInputs(
            request_id=request.request_id,
            source_frame=request.frame,
            preprocessing_metadata={
                "preprocessing_contract_name": "test-preprocess",
                "orientation_source_mode": "raw_grayscale",
            },
            model_inputs={"x_geometry": [1.0, 2.0, 3.0]},
        )


class FakeEngine:
    def __init__(self, *, exception: Exception | None = None) -> None:
        self.exception = exception
        self.calls: list[PreparedInferenceInputs] = []

    def run_inference(self, inputs: PreparedInferenceInputs) -> InferenceResult:
        self.calls.append(inputs)
        if self.exception is not None:
            raise self.exception
        assert inputs.source_frame is not None
        assert inputs.source_frame.frame_hash is not None
        return _result(inputs.request_id, inputs.source_frame)


def _runner(
    preprocessor: FakePreprocessor,
    engine: FakeEngine,
    *,
    trace_recorder: InferenceTraceRecorder | None = None,
) -> SingleFrameInferenceRunner:
    request_ids = iter(("req-000001", "req-000002", "req-000003"))
    return SingleFrameInferenceRunner(
        preprocessor,
        engine,
        trace_recorder=trace_recorder,
        request_id_factory=lambda: next(request_ids),
        now_utc_fn=lambda: REQUESTED_AT,
    )


def _result(request_id: str, frame: FrameReference) -> InferenceResult:
    frame_hash = frame.frame_hash or FrameHash("")
    return InferenceResult(
        request_id=request_id,
        input_image_path=frame.image_path,
        input_image_hash=frame_hash,
        timestamp_utc=RESULT_AT,
        predicted_distance_m=4.5,
        predicted_yaw_sin=0.0,
        predicted_yaw_cos=1.0,
        predicted_yaw_deg=0.0,
        inference_time_ms=12.5,
        extras={"device": "cpu", "model_root": "model-root"},
    )


def _imported_roots(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", maxsplit=1)[0])
    return imported


if __name__ == "__main__":
    unittest.main()
