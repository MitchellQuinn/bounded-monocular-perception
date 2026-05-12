"""Synchronous one-shot inference runner for captured inspection frames."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
import uuid

from interfaces import (
    FrameFailureStage,
    FrameHash,
    FrameMetadata,
    FrameReference,
    InferenceEngine,
    InferenceRequest,
    InferenceResult,
    PreparedInferenceInputs,
    RawImagePreprocessor,
    WorkerError,
    WorkerName,
)
from live_inference.frame_handoff import compute_frame_hash

from .trace_recorder import InferenceTraceRecorder


@dataclass(frozen=True)
class SingleFrameInferenceOutcome:
    """Result payload for one captured-frame diagnostic inference run."""

    result: InferenceResult | None
    error: WorkerError | None
    trace_path: Path | None
    frame_hash: FrameHash
    request_id: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_request_id() -> str:
    return str(uuid.uuid4())


class SingleFrameInferenceRunner:
    """Run preprocessing and model inference exactly once on supplied bytes."""

    def __init__(
        self,
        preprocessor: RawImagePreprocessor,
        engine: InferenceEngine,
        *,
        trace_recorder: InferenceTraceRecorder | None = None,
        request_id_factory: Callable[[], str] | None = None,
        now_utc_fn: Callable[[], str] | None = None,
        request_extras: Mapping[str, object] | None = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._engine = engine
        self._trace_recorder = trace_recorder
        self._request_id_factory = request_id_factory or _default_request_id
        self._now_utc_fn = now_utc_fn or _utc_now_iso
        self._request_extras = dict(request_extras or {})

    @property
    def trace_output_dir(self) -> Path | None:
        if self._trace_recorder is None:
            return None
        return self._trace_recorder.output_dir

    def run_single_frame(
        self,
        image_bytes: bytes,
        *,
        source_path: Path | None = None,
        frame_metadata: FrameMetadata | None = None,
        record_trace: bool = False,
    ) -> SingleFrameInferenceOutcome:
        """Process exactly the supplied bytes, bypassing continuous duplicate skips."""
        accepted_bytes = bytes(image_bytes)
        request_id = self._request_id_factory()
        requested_at_utc = self._now_utc_fn()
        frame_hash = compute_frame_hash(accepted_bytes)

        trace_dir: Path | None = None
        trace_setup_error: WorkerError | None = None
        if record_trace:
            if self._trace_recorder is None:
                trace_setup_error = self._worker_error(
                    request_id=request_id,
                    frame_hash=frame_hash,
                    frame=None,
                    failure_stage=FrameFailureStage.OUTPUT,
                    error_type="trace_recorder_unavailable",
                    message="Trace recording was requested, but no trace recorder is configured.",
                    exception=None,
                )
            else:
                try:
                    trace_dir = self._trace_recorder.create_trace_directory(
                        request_id=request_id,
                        frame_hash=frame_hash,
                        created_at_utc=requested_at_utc,
                    )
                except Exception as exc:
                    trace_setup_error = self._worker_error(
                        request_id=request_id,
                        frame_hash=frame_hash,
                        frame=None,
                        failure_stage=FrameFailureStage.OUTPUT,
                        error_type="trace_recording_failed",
                        message=f"Could not create single-frame trace directory: {exc}",
                        exception=exc,
                    )

        frame = self._frame_reference(
            source_path=source_path,
            frame_metadata=frame_metadata,
            frame_hash=frame_hash,
            byte_size=len(accepted_bytes),
            completed_at_utc=requested_at_utc,
        )
        request = InferenceRequest(
            request_id=request_id,
            frame=frame,
            requested_at_utc=requested_at_utc,
            duplicate_hash_skip_enabled=False,
            save_debug_images=record_trace and trace_dir is not None,
            debug_output_dir=trace_dir,
            extras={**self._request_extras, "single_frame_mode": True},
        )

        prepared = self._prepare_inputs(request, accepted_bytes, frame_hash)
        if isinstance(prepared, WorkerError):
            failure_trace_path = self._record_failure_trace(
                trace_dir=trace_dir,
                image_bytes=accepted_bytes,
                request=request,
                prepared_inputs=None,
                source_path=source_path,
                created_at_utc=requested_at_utc,
                error=prepared,
            )
            return SingleFrameInferenceOutcome(
                result=None,
                error=prepared,
                trace_path=failure_trace_path,
                frame_hash=frame_hash,
                request_id=request_id,
            )

        result = self._run_inference(prepared, request, frame_hash)
        if isinstance(result, WorkerError):
            failure_trace_path = self._record_failure_trace(
                trace_dir=trace_dir,
                image_bytes=accepted_bytes,
                request=request,
                prepared_inputs=prepared,
                source_path=source_path,
                created_at_utc=requested_at_utc,
                error=result,
            )
            return SingleFrameInferenceOutcome(
                result=None,
                error=result,
                trace_path=failure_trace_path,
                frame_hash=frame_hash,
                request_id=request_id,
            )
        result = self._normalized_result(result, request, frame_hash)

        trace_error = trace_setup_error
        if record_trace and trace_dir is not None and self._trace_recorder is not None:
            try:
                trace_dir = self._trace_recorder.record_trace(
                    trace_dir=trace_dir,
                    image_bytes=accepted_bytes,
                    request=request,
                    prepared_inputs=prepared,
                    result=result,
                    source_path=source_path,
                    created_at_utc=requested_at_utc,
                )
            except Exception as exc:
                trace_error = self._worker_error(
                    request_id=request_id,
                    frame_hash=frame_hash,
                    frame=frame,
                    failure_stage=FrameFailureStage.OUTPUT,
                    error_type="trace_recording_failed",
                    message=f"Single-frame inference completed, but trace recording failed: {exc}",
                    exception=exc,
                )
                trace_dir = None

        return SingleFrameInferenceOutcome(
            result=result,
            error=trace_error,
            trace_path=trace_dir,
            frame_hash=frame_hash,
            request_id=request_id,
        )

    def _record_failure_trace(
        self,
        *,
        trace_dir: Path | None,
        image_bytes: bytes,
        request: InferenceRequest,
        prepared_inputs: PreparedInferenceInputs | None,
        source_path: Path | None,
        created_at_utc: str,
        error: WorkerError,
    ) -> Path | None:
        if trace_dir is None or self._trace_recorder is None:
            return None
        try:
            return self._trace_recorder.record_trace(
                trace_dir=trace_dir,
                image_bytes=image_bytes,
                request=request,
                prepared_inputs=prepared_inputs,
                result=None,
                source_path=source_path,
                created_at_utc=created_at_utc,
                error=error,
            )
        except Exception:
            return None

    def _frame_reference(
        self,
        *,
        source_path: Path | None,
        frame_metadata: FrameMetadata | None,
        frame_hash: FrameHash,
        byte_size: int,
        completed_at_utc: str,
    ) -> FrameReference:
        metadata = frame_metadata or FrameMetadata()
        if metadata.byte_size is None:
            metadata = replace(metadata, byte_size=byte_size)
        return FrameReference(
            image_path=Path(source_path) if source_path is not None else _memory_frame_path(frame_hash),
            metadata=metadata,
            completed_at_utc=completed_at_utc,
            frame_hash=frame_hash,
            byte_size=byte_size,
            extras={"single_frame_capture": True},
        )

    def _prepare_inputs(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
        frame_hash: FrameHash,
    ) -> PreparedInferenceInputs | WorkerError:
        try:
            return self._preprocessor.prepare_model_inputs(request, image_bytes)
        except Exception as exc:
            return self._worker_error(
                request_id=request.request_id,
                frame_hash=frame_hash,
                frame=request.frame,
                failure_stage=FrameFailureStage.PREPROCESS,
                error_type="preprocess_failed",
                message=f"Preprocessing failed for single-frame request {request.request_id}: {exc}",
                exception=exc,
            )

    def _run_inference(
        self,
        prepared: PreparedInferenceInputs,
        request: InferenceRequest,
        frame_hash: FrameHash,
    ) -> InferenceResult | WorkerError:
        try:
            return self._engine.run_inference(prepared)
        except Exception as exc:
            return self._worker_error(
                request_id=request.request_id,
                frame_hash=frame_hash,
                frame=request.frame,
                failure_stage=FrameFailureStage.INFERENCE,
                error_type="inference_failed",
                message=f"Inference failed for single-frame request {request.request_id}: {exc}",
                exception=exc,
            )

    def _normalized_result(
        self,
        result: InferenceResult,
        request: InferenceRequest,
        frame_hash: FrameHash,
    ) -> InferenceResult:
        warnings = tuple(result.warnings)
        request_id = result.request_id
        input_image_hash = result.input_image_hash

        if result.request_id != request.request_id:
            warnings = warnings + (
                "Corrected result request_id mismatch: "
                f"{result.request_id!r} != {request.request_id!r}.",
            )
            request_id = request.request_id

        if result.input_image_hash != frame_hash:
            warnings = warnings + (
                "Corrected result input_image_hash mismatch: "
                f"{result.input_image_hash.value!r} != {frame_hash.value!r}.",
            )
            input_image_hash = frame_hash

        if (
            request_id == result.request_id
            and input_image_hash == result.input_image_hash
            and warnings == result.warnings
        ):
            return result

        return replace(
            result,
            request_id=request_id,
            input_image_hash=input_image_hash,
            warnings=warnings,
        )

    def _worker_error(
        self,
        *,
        request_id: str,
        frame_hash: FrameHash,
        frame: FrameReference | None,
        failure_stage: FrameFailureStage,
        error_type: str,
        message: str,
        exception: Exception | None,
    ) -> WorkerError:
        details: dict[str, object] = {
            "request_id": request_id,
            "frame_hash": frame_hash.value,
        }
        if exception is not None:
            details["exception_type"] = type(exception).__name__
            details.update(_exception_details(exception))
        return WorkerError(
            worker_name=WorkerName.INFERENCE,
            error_type=_exception_error_type(exception, error_type),
            message=message,
            recoverable=True,
            timestamp_utc=self._now_utc_fn(),
            frame=frame,
            failure_stage=failure_stage,
            details=details,
        )


def _memory_frame_path(frame_hash: FrameHash) -> Path:
    return Path(f"captured_single_frame_{frame_hash.value[:12]}.png")


def _exception_error_type(exception: Exception | None, default: str) -> str:
    if exception is None:
        return default
    value = getattr(exception, "worker_error_type", None)
    return str(value) if value else default


def _exception_details(exception: Exception) -> dict[str, object]:
    payload: dict[str, object] = {}
    details = getattr(exception, "failure_details", None)
    if isinstance(details, Mapping):
        payload.update(dict(details))
    preprocessing_metadata = getattr(exception, "preprocessing_metadata", None)
    if isinstance(preprocessing_metadata, Mapping):
        payload["preprocessing_metadata"] = dict(preprocessing_metadata)
    debug_paths = getattr(exception, "debug_paths", None)
    if isinstance(debug_paths, Mapping):
        payload["debug_paths"] = {
            str(key): str(value) for key, value in debug_paths.items()
        }
    return payload


__all__ = [
    "SingleFrameInferenceOutcome",
    "SingleFrameInferenceRunner",
]
