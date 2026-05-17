from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import interfaces.contracts as contracts  # noqa: E402
from interfaces import (  # noqa: E402
    FrameHash,
    FrameReference,
    InferenceRequest,
    InferenceResult,
    RawImagePreprocessor,
)
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402
from live_inference.inspection import InferenceTraceRecorder, SingleFrameInferenceRunner  # noqa: E402
from live_inference.model_registry import load_live_model_manifest  # noqa: E402
from live_inference.preprocessing import (  # noqa: E402
    BackgroundEdgeLocator,
    FixedCenterRoiLocator,
    TriStreamLivePreprocessor,
)


class GenericTriStreamPreprocessorTests(unittest.TestCase):
    def test_background_edge_preprocessor_creates_tri_stream_keys(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(image_bytes)
        manifest = _manifest()
        preprocessor = TriStreamLivePreprocessor(
            model_manifest=manifest,
            locator=BackgroundEdgeLocator(),
        )

        prepared = preprocessor.prepare_model_inputs(request, image_bytes)

        self.assertIsInstance(preprocessor, RawImagePreprocessor)
        self.assertEqual(tuple(prepared.model_inputs), contracts.TRI_STREAM_INPUT_KEYS)
        self.assertIn(contracts.TRI_STREAM_DISTANCE_IMAGE_KEY, prepared.model_inputs)
        self.assertIn(contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY, prepared.model_inputs)
        self.assertIn(contracts.TRI_STREAM_GEOMETRY_KEY, prepared.model_inputs)
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_GEOMETRY_KEY].shape,
            (10,),
        )
        self.assertEqual(
            tuple(prepared.preprocessing_metadata[contracts.PREPROCESSING_METADATA_GEOMETRY_SCHEMA]),
            contracts.TRI_STREAM_GEOMETRY_SCHEMA,
        )
        self.assertEqual(
            prepared.preprocessing_metadata[contracts.PREPROCESSING_METADATA_LOCATOR_KIND],
            contracts.LocatorKind.BACKGROUND_EDGE_V1.value,
        )

    def test_locator_only_trace_writes_required_debug_artifacts(self) -> None:
        image_bytes = _fixture_image_bytes()
        trace_dir = Path(tempfile.mkdtemp())
        request = _request(image_bytes, save_debug=True, debug_output_dir=trace_dir)
        preprocessor = TriStreamLivePreprocessor(
            model_manifest=_manifest(),
            locator=BackgroundEdgeLocator(),
        )

        diagnostic = preprocessor.run_locator_only(request, image_bytes)

        self.assertIsNotNone(diagnostic.locator_result)
        self.assertIn(contracts.DISPLAY_ARTIFACT_GRAYSCALE_FRAME, diagnostic.debug_paths)
        self.assertIn(contracts.DISPLAY_ARTIFACT_EDGE_MAP, diagnostic.debug_paths)
        self.assertIn("locator_result", diagnostic.debug_paths)

    def test_fixed_center_fallback_preprocessor_runs(self) -> None:
        image_bytes = _fixture_image_bytes()
        prepared = TriStreamLivePreprocessor(
            model_manifest=_manifest(),
            locator=FixedCenterRoiLocator(roi_wh_px=(320, 320)),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertEqual(
            prepared.preprocessing_metadata[contracts.PREPROCESSING_METADATA_LOCATOR_KIND],
            contracts.LocatorKind.FIXED_CENTER_ROI.value,
        )

    def test_single_frame_trace_contains_v03_artifacts(self) -> None:
        image_bytes = _fixture_image_bytes()
        trace_root = Path(tempfile.mkdtemp())
        runner = SingleFrameInferenceRunner(
            TriStreamLivePreprocessor(
                model_manifest=_manifest(),
                locator=BackgroundEdgeLocator(),
            ),
            _FakeEngine(),
            trace_recorder=InferenceTraceRecorder(output_dir=trace_root),
        )

        outcome = runner.run_single_frame(image_bytes, record_trace=True)

        self.assertIsNone(outcome.error)
        self.assertIsNotNone(outcome.trace_path)
        assert outcome.trace_path is not None
        trace_path = Path(outcome.trace_path)
        for filename in (
            "accepted_raw_frame.png",
            "grayscale_frame.png",
            "foreground_mask.png",
            "edge_map.png",
            "candidate_contours.png",
            "chosen_contour.png",
            "roi_crop.png",
            "x_distance_image.png",
            "x_orientation_image.png",
            "x_geometry.json",
            "locator_result.json",
            "inference_result.json",
            "trace_manifest.json",
        ):
            self.assertTrue((trace_path / filename).is_file(), filename)


def _manifest() -> object:
    return load_live_model_manifest(
        PROJECT_ROOT / "models/distance-orientation/260515-1301_ts-2d-cnn"
    )


def _fixture_image_bytes() -> bytes:
    image = np.full((600, 960), 255, dtype=np.uint8)
    cv2.rectangle(image, (430, 270), (530, 330), 80, thickness=-1)
    cv2.line(image, (430, 270), (530, 330), 40, thickness=2)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("Could not encode fixture image")
    return encoded.tobytes()


def _request(
    image_bytes: bytes,
    *,
    save_debug: bool = False,
    debug_output_dir: Path | None = None,
) -> InferenceRequest:
    return InferenceRequest(
        request_id="req",
        frame=FrameReference(
            image_path=Path("fixture.png"),
            frame_hash=compute_frame_hash(image_bytes),
        ),
        requested_at_utc="2026-05-17T00:00:00Z",
        save_debug_images=save_debug,
        debug_output_dir=debug_output_dir,
    )


class _FakeEngine:
    def run_inference(self, inputs: object) -> InferenceResult:
        source_frame = getattr(inputs, "source_frame")
        return InferenceResult(
            request_id=getattr(inputs, "request_id"),
            input_image_path=source_frame.image_path,
            input_image_hash=source_frame.frame_hash or FrameHash(""),
            timestamp_utc="2026-05-17T00:00:00Z",
            predicted_distance_m=1.23,
            predicted_yaw_sin=0.0,
            predicted_yaw_cos=1.0,
            predicted_yaw_deg=0.0,
            inference_time_ms=0.1,
            debug_paths={
                key: Path(value)
                for key, value in getattr(inputs, "preprocessing_metadata")
                .get("debug_paths", {})
                .items()
            },
        )


if __name__ == "__main__":
    unittest.main()
