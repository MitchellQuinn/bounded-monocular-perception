"""Unit tests for the synchronous inference processing core."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Callable
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
    FrameSkipped,
    FrameSkipReason,
    InferenceRequest,
    InferenceResult,
    LiveInferenceConfig,
    PreparedInferenceInputs,
    WorkerName,
    WorkerWarning,
)
from live_inference.frame_handoff import (  # noqa: E402
    AtomicFrameHandoffWriter,
    LatestFrameHandoffReader,
)
from live_inference.frame_selection import (  # noqa: E402
    FrameSelectionResult,
    InferenceFrameSelector,
    SelectedFrameForInference,
)
from live_inference.inference_core import InferenceProcessingCore  # noqa: E402


REQUESTED_AT = "2026-05-03T10:00:00Z"
RESULT_AT = "2026-05-03T10:00:01Z"
ERROR_AT = "2026-05-03T10:00:02Z"


class InferenceProcessingCoreTests(unittest.TestCase):
    def test_skipped_frame_returns_skipped_outcome(self) -> None:
        skipped = _skipped()
        core = _core_with_fake_selector(
            FrameSelectionResult(skipped=skipped),
            FakePreprocessor(),
            FakeEngine(),
        )

        outcome = core.process_once()

        self.assertIs(outcome.skipped, skipped)
        self.assertIsNone(outcome.result)
        self.assertIsNone(outcome.warning)
        self.assertIsNone(outcome.error)

    def test_selector_warning_returns_warning_outcome(self) -> None:
        warning = WorkerWarning(
            worker_name=WorkerName.INFERENCE,
            warning_type="selection_warning",
            message="selection warning",
            timestamp_utc=REQUESTED_AT,
        )
        preprocessor = FakePreprocessor()
        engine = FakeEngine()
        core = _core_with_fake_selector(
            FrameSelectionResult(warning=warning),
            preprocessor,
            engine,
        )

        outcome = core.process_once()

        self.assertIs(outcome.warning, warning)
        self.assertEqual(preprocessor.calls, [])
        self.assertEqual(engine.calls, [])

    def test_preprocessor_is_not_called_for_skipped_frames(self) -> None:
        preprocessor = FakePreprocessor()
        core = _core_with_fake_selector(
            FrameSelectionResult(skipped=_skipped()),
            preprocessor,
            FakeEngine(),
        )

        core.process_once()

        self.assertEqual(preprocessor.calls, [])

    def test_engine_is_not_called_for_skipped_frames(self) -> None:
        engine = FakeEngine()
        core = _core_with_fake_selector(
            FrameSelectionResult(skipped=_skipped()),
            FakePreprocessor(),
            engine,
        )

        core.process_once()

        self.assertEqual(engine.calls, [])

    def test_successful_path_calls_preprocessor_then_engine(self) -> None:
        events: list[str] = []
        core = _core_with_fake_selector(
            FrameSelectionResult(selected=_selected_frame()),
            FakePreprocessor(events=events),
            FakeEngine(events=events),
            events=events,
        )

        core.process_once()

        self.assertEqual(events, ["select", "preprocess", "engine", "mark_processed"])

    def test_successful_path_returns_inference_result(self) -> None:
        selected = _selected_frame()
        expected_result = _result(selected.request.request_id, selected.frame_hash)
        core = _core_with_fake_selector(
            FrameSelectionResult(selected=selected),
            FakePreprocessor(),
            FakeEngine(result=expected_result),
        )

        outcome = core.process_once()

        self.assertIs(outcome.result, expected_result)
        self.assertIsNone(outcome.skipped)
        self.assertIsNone(outcome.warning)
        self.assertIsNone(outcome.error)

    def test_successful_path_calls_selector_mark_processed_only_after_engine_success(self) -> None:
        events: list[str] = []
        selected = _selected_frame()
        selector = FakeSelector(FrameSelectionResult(selected=selected), events=events)
        core = InferenceProcessingCore(
            selector,  # type: ignore[arg-type]
            FakePreprocessor(events=events),
            FakeEngine(events=events),
            now_utc_fn=lambda: ERROR_AT,
        )

        outcome = core.process_once()

        self.assertIsNotNone(outcome.result)
        self.assertEqual(selector.mark_processed_calls, [selected.frame_hash])
        self.assertEqual(events, ["select", "preprocess", "engine", "mark_processed"])

    def test_preprocess_exception_returns_failure_stage_preprocess(self) -> None:
        core = _core_with_fake_selector(
            FrameSelectionResult(selected=_selected_frame()),
            FakePreprocessor(exception=ValueError("decode unavailable")),
            FakeEngine(),
        )

        outcome = core.process_once()

        self.assertIsNotNone(outcome.error)
        assert outcome.error is not None
        self.assertEqual(outcome.error.failure_stage, FrameFailureStage.PREPROCESS)
        self.assertEqual(outcome.error.error_type, "preprocess_failed")

    def test_preprocess_exception_does_not_mark_processed(self) -> None:
        selector = FakeSelector(FrameSelectionResult(selected=_selected_frame()))
        core = InferenceProcessingCore(
            selector,  # type: ignore[arg-type]
            FakePreprocessor(exception=ValueError("decode unavailable")),
            FakeEngine(),
            now_utc_fn=lambda: ERROR_AT,
        )

        core.process_once()

        self.assertEqual(selector.mark_processed_calls, [])

    def test_roi_rejection_prevents_engine_call_and_marks_frame_processed(self) -> None:
        selected = _selected_frame()
        selector = FakeSelector(FrameSelectionResult(selected=selected))
        engine = FakeEngine()
        core = InferenceProcessingCore(
            selector,  # type: ignore[arg-type]
            FakePreprocessor(exception=_StructuredRoiRejectedError()),
            engine,
            now_utc_fn=lambda: ERROR_AT,
        )

        outcome = core.process_once()

        self.assertIsNotNone(outcome.error)
        assert outcome.error is not None
        self.assertEqual(outcome.error.error_type, "roi_rejected")
        self.assertEqual(outcome.error.failure_stage, FrameFailureStage.PREPROCESS)
        self.assertEqual(engine.calls, [])
        self.assertEqual(selector.mark_processed_calls, [selected.frame_hash])
        self.assertFalse(outcome.error.details["roi_accepted"])

    def test_inference_exception_returns_failure_stage_inference(self) -> None:
        core = _core_with_fake_selector(
            FrameSelectionResult(selected=_selected_frame()),
            FakePreprocessor(),
            FakeEngine(exception=RuntimeError("model unavailable")),
        )

        outcome = core.process_once()

        self.assertIsNotNone(outcome.error)
        assert outcome.error is not None
        self.assertEqual(outcome.error.failure_stage, FrameFailureStage.INFERENCE)
        self.assertEqual(outcome.error.error_type, "inference_failed")

    def test_inference_exception_does_not_mark_processed(self) -> None:
        selector = FakeSelector(FrameSelectionResult(selected=_selected_frame()))
        core = InferenceProcessingCore(
            selector,  # type: ignore[arg-type]
            FakePreprocessor(),
            FakeEngine(exception=RuntimeError("model unavailable")),
            now_utc_fn=lambda: ERROR_AT,
        )

        core.process_once()

        self.assertEqual(selector.mark_processed_calls, [])

    def test_result_request_id_mismatch_is_handled_clearly(self) -> None:
        selected = _selected_frame()
        core = _core_with_fake_selector(
            FrameSelectionResult(selected=selected),
            FakePreprocessor(),
            FakeEngine(result=_result("wrong-request", selected.frame_hash)),
        )

        outcome = core.process_once()

        self.assertIsNotNone(outcome.result)
        assert outcome.result is not None
        self.assertEqual(outcome.result.request_id, selected.request.request_id)
        self.assertTrue(
            any("request_id mismatch" in warning for warning in outcome.result.warnings)
        )

    def test_result_hash_mismatch_is_handled_clearly(self) -> None:
        selected = _selected_frame()
        core = _core_with_fake_selector(
            FrameSelectionResult(selected=selected),
            FakePreprocessor(),
            FakeEngine(result=_result(selected.request.request_id, FrameHash("wrong-hash"))),
        )

        outcome = core.process_once()

        self.assertIsNotNone(outcome.result)
        assert outcome.result is not None
        self.assertEqual(outcome.result.input_image_hash, selected.frame_hash)
        self.assertTrue(
            any("input_image_hash mismatch" in warning for warning in outcome.result.warnings)
        )

    def test_parameter_revision_getter_value_is_attached_to_result_if_supported(self) -> None:
        selected = _selected_frame()
        core = _core_with_fake_selector(
            FrameSelectionResult(selected=selected),
            FakePreprocessor(),
            FakeEngine(result=_result(selected.request.request_id, selected.frame_hash)),
            parameter_revision_getter=lambda: 42,
        )

        outcome = core.process_once()

        self.assertIsNotNone(outcome.result)
        assert outcome.result is not None
        self.assertEqual(outcome.result.preprocessing_parameter_revision, 42)

    def test_duplicate_frame_after_successful_process_is_skipped_on_next_call(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = LiveInferenceConfig(frame_dir=Path(tmp_dir) / "live_frames")
            AtomicFrameHandoffWriter(config).publish_frame(
                b"same-frame",
                FrameMetadata(frame_id="frame-1"),
            )
            selector = InferenceFrameSelector(
                LatestFrameHandoffReader(config),
                request_id_factory=lambda: "request-1",
                now_utc_fn=lambda: REQUESTED_AT,
            )
            core = InferenceProcessingCore(
                selector,
                FakePreprocessor(),
                FakeEngine(),
                now_utc_fn=lambda: ERROR_AT,
            )

            first = core.process_once()
            second = core.process_once()

            self.assertIsNotNone(first.result)
            self.assertIsNotNone(second.skipped)
            assert second.skipped is not None
            self.assertEqual(second.skipped.reason, FrameSkipReason.DUPLICATE_HASH)

    def test_inference_core_module_keeps_heavy_runtime_imports_out(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "inference_core.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        banned_roots = {"PySide6", "cv2", "numpy", "torch"}
        found: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".", maxsplit=1)[0])

        self.assertEqual(found & banned_roots, set())


class FakeSelector:
    def __init__(
        self,
        result: FrameSelectionResult,
        *,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.events = events
        self.mark_processed_calls: list[FrameHash] = []

    def select_latest(self) -> FrameSelectionResult:
        if self.events is not None:
            self.events.append("select")
        return self.result

    def mark_processed(self, frame_hash: FrameHash) -> None:
        if self.events is not None:
            self.events.append("mark_processed")
        self.mark_processed_calls.append(frame_hash)


class FakePreprocessor:
    def __init__(
        self,
        *,
        exception: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.exception = exception
        self.events = events
        self.calls: list[tuple[InferenceRequest, bytes]] = []

    def prepare_model_inputs(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
    ) -> PreparedInferenceInputs:
        if self.events is not None:
            self.events.append("preprocess")
        self.calls.append((request, image_bytes))
        if self.exception is not None:
            raise self.exception
        return PreparedInferenceInputs(
            request_id=request.request_id,
            source_frame=request.frame,
        )


class FakeEngine:
    def __init__(
        self,
        *,
        result: InferenceResult | None = None,
        exception: Exception | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.result = result
        self.exception = exception
        self.events = events
        self.calls: list[PreparedInferenceInputs] = []

    def run_inference(self, inputs: PreparedInferenceInputs) -> InferenceResult:
        if self.events is not None:
            self.events.append("engine")
        self.calls.append(inputs)
        if self.exception is not None:
            raise self.exception
        if self.result is not None:
            return self.result
        assert inputs.source_frame is not None
        assert inputs.source_frame.frame_hash is not None
        return _result(inputs.request_id, inputs.source_frame.frame_hash)


class _StructuredRoiRejectedError(ValueError):
    worker_error_type = "roi_rejected"

    def __init__(self) -> None:
        super().__init__("ROI rejected during preprocessing")
        self.failure_details = {
            "roi_accepted": False,
            "roi_rejection_reason": "low_confidence:0.120<min:0.300",
            "mark_frame_processed": True,
        }
        self.preprocessing_metadata = {
            "roi_accepted": False,
            "roi_rejection_reason": "low_confidence:0.120<min:0.300",
        }
        self.debug_paths = {}


def _core_with_fake_selector(
    selection_result: FrameSelectionResult,
    preprocessor: FakePreprocessor,
    engine: FakeEngine,
    *,
    events: list[str] | None = None,
    parameter_revision_getter: Callable[[], int | None] | None = None,
) -> InferenceProcessingCore:
    kwargs = {}
    if parameter_revision_getter is not None:
        kwargs["parameter_revision_getter"] = parameter_revision_getter
    return InferenceProcessingCore(
        FakeSelector(selection_result, events=events),  # type: ignore[arg-type]
        preprocessor,
        engine,
        now_utc_fn=lambda: ERROR_AT,
        **kwargs,  # type: ignore[arg-type]
    )


def _selected_frame() -> SelectedFrameForInference:
    frame_hash = FrameHash("hash-1")
    frame = FrameReference(
        image_path=Path("live_frames/latest_frame.png"),
        frame_hash=frame_hash,
    )
    request = InferenceRequest(
        request_id="request-1",
        frame=frame,
        requested_at_utc=REQUESTED_AT,
    )
    return SelectedFrameForInference(
        request=request,
        image_bytes=b"frame-bytes",
        frame_hash=frame_hash,
    )


def _result(request_id: str, frame_hash: FrameHash) -> InferenceResult:
    return InferenceResult(
        request_id=request_id,
        input_image_path=Path("live_frames/latest_frame.png"),
        input_image_hash=frame_hash,
        timestamp_utc=RESULT_AT,
        predicted_distance_m=4.5,
        predicted_yaw_sin=0.0,
        predicted_yaw_cos=1.0,
        predicted_yaw_deg=0.0,
        inference_time_ms=12.5,
    )


def _skipped() -> FrameSkipped:
    return FrameSkipped(
        worker_name=WorkerName.INFERENCE,
        reason=FrameSkipReason.MISSING_FILE,
        timestamp_utc=REQUESTED_AT,
        message="No frame.",
    )


if __name__ == "__main__":
    unittest.main()
