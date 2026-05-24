from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import interfaces.contracts as contracts  # noqa: E402
import live_inference.interfaces.contracts as canonical_contracts  # noqa: E402
from live_inference.engines.output_decoding import yaw_degrees_from_sin_cos  # noqa: E402
from live_inference.frame_selection import (  # noqa: E402
    FrameSelectionResult,
    SelectedFrameForInference,
)
from live_inference.inference_core import InferenceProcessingCore  # noqa: E402
from live_inference.output_smoothing import (  # noqa: E402
    MovingAverageDistanceYawSmoother,
    SwappableInferenceResultSmoother,
)


class OutputSmoothingTests(unittest.TestCase):
    def test_contract_smoothing_constants_are_exposed_through_shim(self) -> None:
        self.assertIs(contracts.InferenceResult, canonical_contracts.InferenceResult)
        self.assertEqual(
            contracts.OUTPUT_SMOOTHING_METADATA_KEY,
            canonical_contracts.OUTPUT_SMOOTHING_METADATA_KEY,
        )
        self.assertEqual(contracts.DEFAULT_OUTPUT_SMOOTHING_WINDOW_SECONDS, 1.0)

    def test_moving_average_uses_one_second_window_and_circular_yaw(self) -> None:
        smoother = MovingAverageDistanceYawSmoother(window_seconds=1.0)

        first = smoother.smooth_result(
            _result(distance_m=1.0, yaw_sin=0.0, yaw_cos=1.0, timestamp_utc=_ts(0.0))
        )
        second = smoother.smooth_result(
            _result(distance_m=3.0, yaw_sin=1.0, yaw_cos=0.0, timestamp_utc=_ts(0.5))
        )
        third = smoother.smooth_result(
            _result(distance_m=7.0, yaw_sin=0.0, yaw_cos=-1.0, timestamp_utc=_ts(1.6))
        )

        self.assertAlmostEqual(first.predicted_distance_m, 1.0)
        self.assertAlmostEqual(second.predicted_distance_m, 2.0)
        self.assertAlmostEqual(second.predicted_yaw_deg, 45.0)
        self.assertAlmostEqual(second.predicted_yaw_sin, math.sqrt(0.5))
        self.assertAlmostEqual(second.predicted_yaw_cos, math.sqrt(0.5))
        self.assertAlmostEqual(third.predicted_distance_m, 7.0)
        self.assertAlmostEqual(third.predicted_yaw_deg, 180.0)

        metadata = second.extras[contracts.OUTPUT_SMOOTHING_METADATA_KEY]
        self.assertEqual(
            metadata[contracts.OUTPUT_SMOOTHING_STRATEGY_FIELD],
            contracts.OUTPUT_SMOOTHING_STRATEGY_MOVING_AVERAGE,
        )
        self.assertEqual(metadata[contracts.OUTPUT_SMOOTHING_SAMPLE_COUNT_FIELD], 2)
        self.assertEqual(
            metadata[contracts.OUTPUT_SMOOTHING_RAW_PREDICTION_KEY][
                contracts.PREDICTED_DISTANCE_FIELD
            ],
            3.0,
        )

    def test_swappable_smoother_can_replace_strategy(self) -> None:
        smoother = SwappableInferenceResultSmoother(
            MovingAverageDistanceYawSmoother(window_seconds=1.0)
        )
        baseline = smoother.smooth_result(
            _result(distance_m=2.0, yaw_sin=0.0, yaw_cos=1.0, timestamp_utc=_ts(0.0))
        )

        smoother.set_strategy(
            lambda result: replace(result, predicted_distance_m=42.0),
        )
        replaced = smoother.smooth_result(
            _result(distance_m=3.0, yaw_sin=0.0, yaw_cos=1.0, timestamp_utc=_ts(0.1))
        )

        self.assertAlmostEqual(baseline.predicted_distance_m, 2.0)
        self.assertAlmostEqual(replaced.predicted_distance_m, 42.0)

    def test_core_smooths_after_result_normalization(self) -> None:
        frame_hash = contracts.FrameHash("selected-hash")
        frame = contracts.FrameReference(
            image_path=Path("frame.png"),
            frame_hash=frame_hash,
        )
        request = contracts.InferenceRequest(
            request_id="request-1",
            frame=frame,
            requested_at_utc=_ts(0.0),
        )
        selected = SelectedFrameForInference(
            request=request,
            image_bytes=b"image-bytes",
            frame_hash=frame_hash,
        )
        selector = _Selector(selected)
        smoother = _RecordingSmoother()
        core = InferenceProcessingCore(
            selector,
            _Preprocessor(),
            _Engine(),
            result_smoother=smoother,
        )

        outcome = core.process_once()

        self.assertIsNotNone(outcome.result)
        assert outcome.result is not None
        self.assertEqual(outcome.result.request_id, "request-1")
        self.assertEqual(outcome.result.input_image_hash, frame_hash)
        self.assertAlmostEqual(outcome.result.predicted_distance_m, 9.0)
        self.assertEqual(smoother.seen, [("request-1", "selected-hash")])
        self.assertEqual(selector.marked, [frame_hash])


def _result(
    *,
    distance_m: float,
    yaw_sin: float,
    yaw_cos: float,
    timestamp_utc: str,
    request_id: str = "request",
    frame_hash_value: str = "hash",
    preprocessing_parameter_revision: int | None = 0,
) -> contracts.InferenceResult:
    return contracts.InferenceResult(
        request_id=request_id,
        input_image_path=Path("frame.png"),
        input_image_hash=contracts.FrameHash(frame_hash_value),
        timestamp_utc=timestamp_utc,
        predicted_distance_m=distance_m,
        predicted_yaw_sin=yaw_sin,
        predicted_yaw_cos=yaw_cos,
        predicted_yaw_deg=yaw_degrees_from_sin_cos(yaw_sin, yaw_cos),
        inference_time_ms=1.0,
        preprocessing_parameter_revision=preprocessing_parameter_revision,
    )


def _ts(offset_seconds: float) -> str:
    return f"2026-05-24T00:00:{offset_seconds:04.1f}Z"


class _Selector:
    def __init__(self, selected: SelectedFrameForInference) -> None:
        self._selected = selected
        self.marked: list[contracts.FrameHash] = []

    def select_next(self) -> FrameSelectionResult:
        return FrameSelectionResult(selected=self._selected)

    def mark_processed(self, frame_hash: contracts.FrameHash) -> None:
        self.marked.append(frame_hash)


class _Preprocessor:
    def prepare_model_inputs(
        self,
        request: contracts.InferenceRequest,
        _image_bytes: bytes,
    ) -> contracts.PreparedInferenceInputs:
        return contracts.PreparedInferenceInputs(
            request_id=request.request_id,
            source_frame=request.frame,
        )


class _Engine:
    def run_inference(
        self,
        _inputs: contracts.PreparedInferenceInputs,
    ) -> contracts.InferenceResult:
        return _result(
            distance_m=1.0,
            yaw_sin=0.0,
            yaw_cos=1.0,
            timestamp_utc=_ts(0.0),
            request_id="engine-request",
            frame_hash_value="engine-hash",
        )


class _RecordingSmoother:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def smooth_result(
        self,
        result: contracts.InferenceResult,
    ) -> contracts.InferenceResult:
        self.seen.append((result.request_id, result.input_image_hash.value))
        return replace(result, predicted_distance_m=9.0)


if __name__ == "__main__":
    unittest.main()
