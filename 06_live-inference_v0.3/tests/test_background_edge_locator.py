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
from interfaces import FrameReference, LocatorRequest  # noqa: E402
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402
from live_inference.masking import BackgroundState  # noqa: E402
from live_inference.preprocessing import (  # noqa: E402
    BackgroundEdgeLocator,
    BackgroundEdgeLocatorConfig,
    FixedCenterRoiLocator,
    LocatorRuntimeParameterState,
    ManualFixedRoiLocator,
)
import live_inference.preprocessing.locators as locators_module  # noqa: E402


class BackgroundEdgeLocatorTests(unittest.TestCase):
    def test_locates_foreground_against_captured_background(self) -> None:
        background = np.full((600, 960), 255, dtype=np.uint8)
        frame = background.copy()
        cv2.rectangle(frame, (430, 270), (530, 330), 80, thickness=-1)
        state = BackgroundState(threshold=20)
        state.capture_background(background)
        state.set_enabled(True)
        request, image_bytes, trace_dir = _request(frame)
        locator = BackgroundEdgeLocator(
            background_state=state,
            parameter_state=LocatorRuntimeParameterState(
                BackgroundEdgeLocatorConfig(
                    min_foreground_area_px=50,
                    roi_clip_tolerance_px=0,
                )
            ),
        )

        result = locator.locate(request, image_bytes)

        self.assertTrue(result.accepted, result.roi_rejection_reasons)
        self.assertEqual(result.locator_kind, contracts.LocatorKind.BACKGROUND_EDGE_V1)
        self.assertIsNotNone(result.chosen_candidate)
        assert result.center_xy_px is not None
        self.assertAlmostEqual(result.center_xy_px[0], 480.5, delta=8.0)
        self.assertIn(contracts.DISPLAY_ARTIFACT_EDGE_MAP, result.debug_artifacts.paths)
        self.assertTrue(
            result.debug_artifacts.paths[contracts.DISPLAY_ARTIFACT_EDGE_MAP].is_file()
        )
        assert result.chosen_candidate is not None
        self.assertEqual(
            result.chosen_candidate.extras.get(contracts.ROI_CANDIDATE_SOURCE_FIELD),
            contracts.ROI_CANDIDATE_SOURCE_FOREGROUND_COMPONENT,
        )

    def test_prefers_foreground_blob_over_edge_fragments(self) -> None:
        foreground_mask = np.zeros((120, 180), dtype=bool)
        foreground_mask[30:90, 40:140] = True
        edge_map = np.zeros((120, 180), dtype=np.uint8)
        edge_map[38:45, 52:59] = 255
        edge_map[70:78, 115:123] = 255

        candidates = locators_module._build_candidates(
            edge_map=edge_map,
            foreground_mask=foreground_mask,
            source_wh=(180, 120),
            config=BackgroundEdgeLocatorConfig(min_foreground_area_px=50),
        )

        accepted_candidates = tuple(
            candidate for candidate in candidates if not candidate.rejection_reason
        )
        self.assertTrue(accepted_candidates)
        chosen = max(accepted_candidates, key=lambda item: item.score)
        self.assertEqual(
            chosen.extras.get(contracts.ROI_CANDIDATE_SOURCE_FIELD),
            contracts.ROI_CANDIDATE_SOURCE_FOREGROUND_COMPONENT,
        )
        self.assertEqual(chosen.bbox_xyxy_px, (40.0, 30.0, 140.0, 90.0))
        self.assertEqual(chosen.center_xy_px, (90.0, 60.0))

    def test_diffuse_large_component_falls_back_to_edge_candidate(self) -> None:
        foreground_mask = np.zeros((400, 600), dtype=bool)
        foreground_mask[50:350, 100:400] = True
        edge_map = np.zeros((400, 600), dtype=np.uint8)
        cv2.rectangle(edge_map, (240, 180), (290, 245), 255, thickness=2)

        candidates = locators_module._build_candidates(
            edge_map=edge_map,
            foreground_mask=foreground_mask,
            source_wh=(600, 400),
            config=BackgroundEdgeLocatorConfig(min_foreground_area_px=50),
        )

        accepted_candidates = tuple(
            candidate for candidate in candidates if not candidate.rejection_reason
        )
        self.assertTrue(accepted_candidates)
        chosen = max(accepted_candidates, key=lambda item: item.score)
        self.assertEqual(
            chosen.extras.get(contracts.ROI_CANDIDATE_SOURCE_FIELD),
            contracts.ROI_CANDIDATE_SOURCE_EDGE_CONTOUR,
        )
        self.assertEqual(chosen.bbox_xyxy_px, (239.0, 179.0, 292.0, 247.0))
        component = next(
            candidate
            for candidate in candidates
            if candidate.candidate_id.startswith("component_")
        )
        self.assertIn("diffuse_large_component", component.rejection_reason)

    def test_rejects_when_requested_background_removal_is_disabled(self) -> None:
        frame = np.full((600, 960), 255, dtype=np.uint8)
        cv2.rectangle(frame, (430, 270), (530, 330), 80, thickness=-1)
        background = np.full(frame.shape, 255, dtype=np.uint8)
        state = BackgroundState(threshold=20)
        state.capture_background(background)
        state.set_enabled(False)
        request, image_bytes, _trace_dir = _request(
            frame,
            extras={
                contracts.PREPROCESSING_METADATA_APPLY_BACKGROUND_REMOVAL_TO_ROI_LOCATOR: True,
            },
        )
        locator = BackgroundEdgeLocator(background_state=state)

        result = locator.locate(request, image_bytes)

        self.assertFalse(result.accepted)
        self.assertIn(
            contracts.LocatorFailureReason.NO_FOREGROUND.value,
            result.roi_rejection_reasons,
        )
        self.assertTrue(
            any("refusing dark-on-light fallback" in item for item in result.warnings)
        )

    def test_rejects_clipped_roi_visibly(self) -> None:
        frame = np.full((300, 480), 255, dtype=np.uint8)
        cv2.rectangle(frame, (220, 230), (280, 290), 70, thickness=-1)
        request, image_bytes, _trace_dir = _request(frame)
        locator = BackgroundEdgeLocator(
            parameter_state=LocatorRuntimeParameterState(
                BackgroundEdgeLocatorConfig(min_foreground_area_px=50)
            )
        )

        result = locator.locate(request, image_bytes)

        self.assertFalse(result.accepted)
        self.assertIn(
            contracts.LocatorFailureReason.ROI_CLIPPED.value,
            result.roi_rejection_reasons,
        )

    def test_fixed_center_fallback_runs(self) -> None:
        frame = np.full((600, 960), 255, dtype=np.uint8)
        request, image_bytes, _trace_dir = _request(frame, save_debug=False)

        result = FixedCenterRoiLocator(roi_wh_px=(320, 320)).locate(request, image_bytes)

        self.assertTrue(result.accepted)
        self.assertEqual(result.locator_kind, contracts.LocatorKind.FIXED_CENTER_ROI)
        self.assertEqual(result.center_xy_px, (480.0, 300.0))

    def test_manual_fixed_roi_fallback_runs_without_explicit_center(self) -> None:
        frame = np.full((600, 960), 255, dtype=np.uint8)
        request, image_bytes, _trace_dir = _request(frame, save_debug=False)

        result = ManualFixedRoiLocator(roi_wh_px=(320, 320)).locate(request, image_bytes)

        self.assertTrue(result.accepted)
        self.assertEqual(result.locator_kind, contracts.LocatorKind.MANUAL_FIXED_ROI)
        self.assertEqual(result.center_xy_px, (480.0, 300.0))


def _request(
    gray: np.ndarray,
    *,
    save_debug: bool = True,
    extras: dict[str, object] | None = None,
) -> tuple[LocatorRequest, bytes, Path]:
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise AssertionError("Could not encode fixture image")
    image_bytes = encoded.tobytes()
    trace_dir = Path(tempfile.mkdtemp())
    request = LocatorRequest(
        request_id="t",
        frame=FrameReference(
            image_path=Path("fixture.png"),
            frame_hash=compute_frame_hash(image_bytes),
        ),
        requested_at_utc="2026-05-17T00:00:00Z",
        save_debug_images=save_debug,
        debug_output_dir=trace_dir,
        extras=dict(extras or {}),
    )
    return request, image_bytes, trace_dir


if __name__ == "__main__":
    unittest.main()
