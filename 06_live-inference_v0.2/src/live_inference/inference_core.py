"""Synchronous one-frame inference processing core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Mapping
from typing import Callable

from interfaces import (
    DebugImageReference,
    FrameFailureStage,
    FrameHash,
    FrameSkipped,
    InferenceEngine,
    InferenceResult,
    PreparedInferenceInputs,
    RawImagePreprocessor,
    TRI_STREAM_INPUT_KEYS,
    WorkerError,
    WorkerName,
    WorkerWarning,
)
from live_inference.frame_selection import (
    FrameSelectionResult,
    InferenceFrameSelector,
    SelectedFrameForInference,
)


@dataclass(frozen=True)
class InferenceProcessingOutcome:
    """Plain payload returned by one synchronous inference processing step."""

    result: InferenceResult | None = None
    skipped: FrameSkipped | None = None
    warning: WorkerWarning | None = None
    error: WorkerError | None = None
    debug_images: tuple[DebugImageReference, ...] = ()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class InferenceProcessingCore:
    """Run selection, preprocessing, and inference for at most one frame."""

    def __init__(
        self,
        selector: InferenceFrameSelector,
        preprocessor: RawImagePreprocessor,
        engine: InferenceEngine,
        *,
        parameter_revision_getter: Callable[[], int | None] | None = None,
        now_utc_fn: Callable[[], str] | None = None,
    ) -> None:
        self._selector = selector
        self._preprocessor = preprocessor
        self._engine = engine
        self._parameter_revision_getter = parameter_revision_getter
        self._now_utc_fn = now_utc_fn or _utc_now_iso

    def process_once(self) -> InferenceProcessingOutcome:
        """Process at most one selected frame and return plain contract payloads."""
        selection = self._select_next()
        if selection.skipped is not None:
            return InferenceProcessingOutcome(skipped=selection.skipped)
        if selection.warning is not None:
            return InferenceProcessingOutcome(warning=selection.warning)
        if selection.selected is None:
            return InferenceProcessingOutcome(
                warning=WorkerWarning(
                    worker_name=WorkerName.INFERENCE,
                    warning_type="frame_selection_empty",
                    message="Frame selector returned no selected frame, skip, or warning.",
                    timestamp_utc=self._now_utc_fn(),
                )
            )

        selected = selection.selected
        prepared = self._prepare_inputs(selected)
        if isinstance(prepared, WorkerError):
            if _should_mark_processed_after_error(prepared):
                self._selector.mark_processed(selected.frame_hash)
            return InferenceProcessingOutcome(error=prepared)

        result = self._run_inference(prepared, selected)
        if isinstance(result, WorkerError):
            return InferenceProcessingOutcome(error=result)

        result = self._normalized_result(result, selected)
        debug_images = self._debug_images_from_result(result)
        self._selector.mark_processed(selected.frame_hash)
        return InferenceProcessingOutcome(result=result, debug_images=debug_images)

    def _select_next(self) -> FrameSelectionResult:
        for method_name in (
            "select_next",
            "select_latest",
            "select",
            "poll",
            "select_latest_frame",
        ):
            method = getattr(self._selector, method_name, None)
            if callable(method):
                return method()
        raise AttributeError("Frame selector does not expose a known selection method.")

    def _prepare_inputs(
        self,
        selected: SelectedFrameForInference,
    ) -> PreparedInferenceInputs | WorkerError:
        try:
            return self._preprocessor.prepare_model_inputs(
                selected.request,
                selected.image_bytes,
            )
        except Exception as exc:
            return self._worker_error(
                selected,
                failure_stage=FrameFailureStage.PREPROCESS,
                error_type="preprocess_failed",
                message=f"Preprocessing failed for request {selected.request.request_id}: {exc}",
                exception=exc,
            )

    def _run_inference(
        self,
        prepared: PreparedInferenceInputs,
        selected: SelectedFrameForInference,
    ) -> InferenceResult | WorkerError:
        try:
            return self._engine.run_inference(prepared)
        except Exception as exc:
            return self._worker_error(
                selected,
                failure_stage=FrameFailureStage.INFERENCE,
                error_type="inference_failed",
                message=f"Inference failed for request {selected.request.request_id}: {exc}",
                exception=exc,
            )

    def _normalized_result(
        self,
        result: InferenceResult,
        selected: SelectedFrameForInference,
    ) -> InferenceResult:
        warnings = tuple(result.warnings)
        request_id = result.request_id
        input_image_hash = result.input_image_hash

        if result.request_id != selected.request.request_id:
            warnings = warnings + (
                "Corrected result request_id mismatch: "
                f"{result.request_id!r} != {selected.request.request_id!r}.",
            )
            request_id = selected.request.request_id

        if result.input_image_hash != selected.frame_hash:
            warnings = warnings + (
                "Corrected result input_image_hash mismatch: "
                f"{_hash_value(result.input_image_hash)!r} != "
                f"{_hash_value(selected.frame_hash)!r}.",
            )
            input_image_hash = selected.frame_hash

        parameter_revision = self._result_parameter_revision(result)

        if (
            request_id == result.request_id
            and input_image_hash == result.input_image_hash
            and parameter_revision == result.preprocessing_parameter_revision
            and warnings == result.warnings
        ):
            return result

        return replace(
            result,
            request_id=request_id,
            input_image_hash=input_image_hash,
            preprocessing_parameter_revision=parameter_revision,
            warnings=warnings,
        )

    def _result_parameter_revision(self, result: InferenceResult) -> int | None:
        if self._parameter_revision_getter is None:
            return result.preprocessing_parameter_revision
        revision = self._parameter_revision_getter()
        if revision is None:
            return result.preprocessing_parameter_revision
        return revision

    def _debug_images_from_result(
        self,
        result: InferenceResult,
    ) -> tuple[DebugImageReference, ...]:
        return tuple(
            self._debug_image_from_path(result, str(image_kind), path)
            for image_kind, path in result.debug_paths.items()
        )

    def _debug_image_from_path(
        self,
        result: InferenceResult,
        image_kind: str,
        path: Path | str,
    ) -> DebugImageReference:
        return DebugImageReference(
            request_id=result.request_id,
            image_kind=image_kind,
            path=Path(path),
            created_at_utc=result.timestamp_utc,
            source_frame_hash=result.input_image_hash,
            model_input_key=image_kind if image_kind in TRI_STREAM_INPUT_KEYS else None,
            parameter_revision=result.preprocessing_parameter_revision,
            label=image_kind,
        )

    def _worker_error(
        self,
        selected: SelectedFrameForInference,
        *,
        failure_stage: FrameFailureStage,
        error_type: str,
        message: str,
        exception: Exception,
    ) -> WorkerError:
        return WorkerError(
            worker_name=WorkerName.INFERENCE,
            error_type=_exception_error_type(exception, error_type),
            message=message,
            recoverable=True,
            timestamp_utc=self._now_utc_fn(),
            frame=selected.request.frame,
            failure_stage=failure_stage,
            details={
                "request_id": selected.request.request_id,
                "frame_hash": _hash_value(selected.frame_hash),
                "exception_type": type(exception).__name__,
                **_exception_details(exception),
            },
        )


def _hash_value(frame_hash: FrameHash) -> str:
    return frame_hash.value


def _exception_error_type(exception: Exception, default: str) -> str:
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
        payload["debug_paths"] = {str(key): str(value) for key, value in debug_paths.items()}
    return payload


def _should_mark_processed_after_error(error: WorkerError) -> bool:
    return bool(error.details.get("mark_frame_processed"))


__all__ = [
    "InferenceProcessingCore",
    "InferenceProcessingOutcome",
]
