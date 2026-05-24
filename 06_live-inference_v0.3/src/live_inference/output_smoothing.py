"""Post-inference smoothing for live distance/orientation outputs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from threading import RLock
from typing import Any, Protocol

import interfaces.contracts as contracts
from live_inference.engines.output_decoding import yaw_degrees_from_sin_cos


class InferenceResultSmoothingStrategy(Protocol):
    """Callable boundary for replaceable inference-result smoothing."""

    def __call__(self, result: contracts.InferenceResult) -> contracts.InferenceResult:
        """Return a smoothed replacement for ``result``."""
        ...


@dataclass(frozen=True)
class _PredictionSample:
    timestamp_seconds: float
    distance_m: float
    yaw_sin: float
    yaw_cos: float


_UNSET = object()


class SwappableInferenceResultSmoother:
    """Delegates smoothing to a strategy that can be replaced at runtime."""

    def __init__(self, strategy: object | None = None) -> None:
        self._strategy = strategy
        self._lock = RLock()

    @property
    def strategy(self) -> object | None:
        with self._lock:
            return self._strategy

    def set_strategy(self, strategy: object | None, *, reset: bool = True) -> None:
        """Replace the active smoothing strategy."""
        with self._lock:
            self._strategy = strategy
            if reset:
                _reset_strategy(strategy)

    def reset(self) -> None:
        """Reset state held by the active strategy, when it exposes reset()."""
        with self._lock:
            _reset_strategy(self._strategy)

    def smooth_result(
        self,
        result: contracts.InferenceResult,
    ) -> contracts.InferenceResult:
        with self._lock:
            if self._strategy is None:
                return result
            return _apply_strategy(self._strategy, result)

    def __call__(self, result: contracts.InferenceResult) -> contracts.InferenceResult:
        return self.smooth_result(result)


class MovingAverageDistanceYawSmoother:
    """One-window moving average over distance and yaw sin/cos outputs."""

    def __init__(
        self,
        *,
        window_seconds: float = contracts.DEFAULT_OUTPUT_SMOOTHING_WINDOW_SECONDS,
        reset_on_preprocessing_revision_change: bool = True,
    ) -> None:
        if not math.isfinite(float(window_seconds)) or float(window_seconds) <= 0.0:
            raise ValueError("window_seconds must be a positive finite number.")
        self.window_seconds = float(window_seconds)
        self.reset_on_preprocessing_revision_change = bool(
            reset_on_preprocessing_revision_change
        )
        self._samples: deque[_PredictionSample] = deque()
        self._last_timestamp_seconds: float | None = None
        self._last_preprocessing_revision: object = _UNSET

    def reset(self) -> None:
        self._samples.clear()
        self._last_timestamp_seconds = None
        self._last_preprocessing_revision = _UNSET

    def smooth_result(
        self,
        result: contracts.InferenceResult,
    ) -> contracts.InferenceResult:
        timestamp_seconds = _timestamp_seconds(result, self._last_timestamp_seconds)
        preprocessing_revision = result.preprocessing_parameter_revision
        if self._should_reset(timestamp_seconds, preprocessing_revision):
            self.reset()

        self._last_timestamp_seconds = timestamp_seconds
        self._last_preprocessing_revision = preprocessing_revision
        self._samples.append(
            _PredictionSample(
                timestamp_seconds=timestamp_seconds,
                distance_m=_finite_float(result.predicted_distance_m),
                yaw_sin=_finite_float(result.predicted_yaw_sin),
                yaw_cos=_finite_float(result.predicted_yaw_cos),
            )
        )
        self._trim_window(timestamp_seconds)
        return _result_with_smoothed_prediction(
            result,
            samples=tuple(self._samples),
            window_seconds=self.window_seconds,
        )

    def __call__(self, result: contracts.InferenceResult) -> contracts.InferenceResult:
        return self.smooth_result(result)

    def _should_reset(
        self,
        timestamp_seconds: float,
        preprocessing_revision: int | None,
    ) -> bool:
        if (
            self._last_timestamp_seconds is not None
            and timestamp_seconds < self._last_timestamp_seconds
        ):
            return True
        if not self.reset_on_preprocessing_revision_change:
            return False
        if self._last_preprocessing_revision is _UNSET:
            return False
        return preprocessing_revision != self._last_preprocessing_revision

    def _trim_window(self, current_timestamp_seconds: float) -> None:
        cutoff = current_timestamp_seconds - self.window_seconds
        while self._samples and self._samples[0].timestamp_seconds < cutoff:
            self._samples.popleft()


def _apply_strategy(
    strategy: object,
    result: contracts.InferenceResult,
) -> contracts.InferenceResult:
    smooth_result = getattr(strategy, "smooth_result", None)
    if callable(smooth_result):
        return smooth_result(result)
    if callable(strategy):
        return strategy(result)
    raise TypeError(
        "Inference result smoother must be callable or expose smooth_result(result)."
    )


def _reset_strategy(strategy: object | None) -> None:
    reset = getattr(strategy, "reset", None)
    if callable(reset):
        reset()


def _result_with_smoothed_prediction(
    result: contracts.InferenceResult,
    *,
    samples: tuple[_PredictionSample, ...],
    window_seconds: float,
) -> contracts.InferenceResult:
    if not samples:
        return result

    sample_count = len(samples)
    distance_m = sum(sample.distance_m for sample in samples) / sample_count
    yaw_sin_mean = sum(sample.yaw_sin for sample in samples) / sample_count
    yaw_cos_mean = sum(sample.yaw_cos for sample in samples) / sample_count
    yaw_sin, yaw_cos = _normalized_yaw_components(
        yaw_sin_mean,
        yaw_cos_mean,
        fallback=samples[-1],
    )
    yaw_deg = yaw_degrees_from_sin_cos(yaw_sin, yaw_cos)

    extras = dict(result.extras)
    extras[contracts.OUTPUT_SMOOTHING_METADATA_KEY] = {
        contracts.OUTPUT_SMOOTHING_STRATEGY_FIELD: (
            contracts.OUTPUT_SMOOTHING_STRATEGY_MOVING_AVERAGE
        ),
        contracts.OUTPUT_SMOOTHING_WINDOW_SECONDS_FIELD: float(window_seconds),
        contracts.OUTPUT_SMOOTHING_SAMPLE_COUNT_FIELD: int(sample_count),
        contracts.OUTPUT_SMOOTHING_RAW_PREDICTION_KEY: _raw_prediction(result),
    }
    return replace(
        result,
        predicted_distance_m=float(distance_m),
        predicted_yaw_sin=float(yaw_sin),
        predicted_yaw_cos=float(yaw_cos),
        predicted_yaw_deg=float(yaw_deg),
        extras=extras,
    )


def _normalized_yaw_components(
    yaw_sin: float,
    yaw_cos: float,
    *,
    fallback: _PredictionSample,
) -> tuple[float, float]:
    magnitude = math.hypot(yaw_sin, yaw_cos)
    if magnitude > 1.0e-12:
        return yaw_sin / magnitude, yaw_cos / magnitude

    fallback_magnitude = math.hypot(fallback.yaw_sin, fallback.yaw_cos)
    if fallback_magnitude > 1.0e-12:
        return fallback.yaw_sin / fallback_magnitude, fallback.yaw_cos / fallback_magnitude
    return 0.0, 1.0


def _raw_prediction(result: contracts.InferenceResult) -> dict[str, float]:
    return {
        contracts.PREDICTED_DISTANCE_FIELD: float(result.predicted_distance_m),
        contracts.PREDICTED_YAW_SIN_FIELD: float(result.predicted_yaw_sin),
        contracts.PREDICTED_YAW_COS_FIELD: float(result.predicted_yaw_cos),
        contracts.PREDICTED_YAW_DEG_FIELD: float(result.predicted_yaw_deg),
    }


def _timestamp_seconds(
    result: contracts.InferenceResult,
    previous_timestamp_seconds: float | None,
) -> float:
    try:
        timestamp_text = str(result.timestamp_utc).strip()
        if timestamp_text.endswith("Z"):
            timestamp_text = f"{timestamp_text[:-1]}+00:00"
        parsed = datetime.fromisoformat(timestamp_text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return float(parsed.timestamp())
    except (TypeError, ValueError):
        return 0.0 if previous_timestamp_seconds is None else previous_timestamp_seconds


def _finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Prediction value must be finite; got {number!r}.")
    return number


__all__ = [
    "InferenceResultSmoothingStrategy",
    "MovingAverageDistanceYawSmoother",
    "SwappableInferenceResultSmoother",
]
