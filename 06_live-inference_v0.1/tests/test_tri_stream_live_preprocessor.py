"""Tests for the concrete live tri-stream preprocessor adapter."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
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
    FrameMetadata,
    FrameReference,
    InferenceRequest,
    RawImagePreprocessor,
)
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402
from live_inference.masking import BackgroundState, FrameMaskState  # noqa: E402
from live_inference.model_registry import load_live_model_manifest  # noqa: E402
from live_inference.model_registry.model_manifest import (  # noqa: E402
    ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE,
)
from live_inference.preprocessing import (  # noqa: E402
    ROI_LOCATOR_INPUT_POLARITY_AS_IS,
    ROI_LOCATOR_INPUT_POLARITY_INVERTED,
    RoiLocation,
    RoiRejectedError,
    StageTransformPolicySnapshot,
    TriStreamLivePreprocessor,
)
from live_inference.preprocessing.debug_artifacts import (  # noqa: E402
    ARTIFACT_ACCEPTED_RAW_FRAME,
    ARTIFACT_DISTANCE_IMAGE,
    ARTIFACT_FINAL_LOCATOR_INPUT,
    ARTIFACT_LOCATOR_INPUT,
    ARTIFACT_LOCATOR_INPUT_AFTER_POLARITY,
    ARTIFACT_LOCATOR_INPUT_AFTER_BACKGROUND_REMOVAL,
    ARTIFACT_LOCATOR_INPUT_AFTER_MANUAL_MASK,
    ARTIFACT_LOCATOR_INPUT_BEFORE_POLARITY,
    ARTIFACT_LOCATOR_INPUT_RAW_OR_PRETRANSFORM,
    ARTIFACT_ORIENTATION_IMAGE,
    ARTIFACT_ROI_OVERLAY_METADATA,
)


REQUESTED_AT = "2026-05-04T12:00:00Z"

# TODO: Add a parity test against the 05 preprocessor once a concrete ROI-FCN
# locator can be wired without loading model runtimes; this slice intentionally
# uses a deterministic fake locator.


class TriStreamLivePreprocessorTests(unittest.TestCase):
    def test_instantiates_with_manifest_and_fake_roi_locator(self) -> None:
        manifest = _fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE)

        preprocessor = TriStreamLivePreprocessor(
            model_manifest=manifest,
            roi_locator=FakeRoiLocator(),
        )

        self.assertIsInstance(preprocessor, RawImagePreprocessor)

    def test_valid_image_bytes_return_required_prepared_inputs(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(image_bytes)
        preprocessor = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
            runtime_parameter_revision_getter=lambda: 7,
        )

        prepared = preprocessor.prepare_model_inputs(request, image_bytes)

        self.assertEqual(prepared.request_id, request.request_id)
        self.assertIs(prepared.source_frame, request.frame)
        self.assertEqual(
            prepared.input_mode,
            contracts.InferenceInputMode.TRI_STREAM_V0_4,
        )
        self.assertEqual(prepared.input_keys, contracts.TRI_STREAM_INPUT_KEYS)
        self.assertEqual(tuple(prepared.model_inputs), contracts.TRI_STREAM_INPUT_KEYS)
        self.assertEqual(
            set(prepared.model_inputs),
            {
                contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
                contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
                contracts.TRI_STREAM_GEOMETRY_KEY,
            },
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY].shape,
            (1, 300, 300),
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY].shape,
            (1, 300, 300),
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_GEOMETRY_KEY].shape,
            (10,),
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY].dtype,
            np.float32,
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY].dtype,
            np.float32,
        )
        self.assertEqual(
            prepared.model_inputs[contracts.TRI_STREAM_GEOMETRY_KEY].dtype,
            np.float32,
        )
        self.assertEqual(prepared.preprocessing_metadata["runtime_parameter_revision"], 7)

    def test_no_mask_state_reports_unapplied_mask_metadata(self) -> None:
        image_bytes = _fixture_image_bytes()

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        metadata = prepared.preprocessing_metadata
        self.assertFalse(metadata["frame_mask_applied"])
        self.assertIsNone(metadata["frame_mask_revision"])
        self.assertEqual(metadata["frame_mask_pixel_count"], 0)

    def test_no_background_state_leaves_roi_locator_input_unchanged(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()

        TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 95)

    def test_default_roi_locator_polarity_leaves_grayscale_unchanged(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 95)
        self.assertEqual(
            prepared.preprocessing_metadata[
                contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY
            ],
            ROI_LOCATOR_INPUT_POLARITY_AS_IS,
        )

    def test_inverted_roi_locator_polarity_inverts_only_locator_input(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            stage_policy=StageTransformPolicySnapshot(
                roi_locator_input_polarity=ROI_LOCATOR_INPUT_POLARITY_INVERTED
            ),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 160)
        self.assertEqual(
            prepared.preprocessing_metadata[
                contracts.PREPROCESSING_METADATA_ROI_LOCATOR_INPUT_POLARITY
            ],
            ROI_LOCATOR_INPUT_POLARITY_INVERTED,
        )
        self.assertEqual(
            prepared.preprocessing_metadata["orientation_source_mode"],
            ORIENTATION_SOURCE_RAW_GRAYSCALE,
        )

    def test_mask_state_with_no_mask_leaves_roi_locator_input_unchanged(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        mask_state = FrameMaskState()

        TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            mask_state=mask_state,
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 95)

    def test_default_manual_mask_does_not_apply_before_roi_locator_with_white_fill(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        mask_state = FrameMaskState()
        mask = np.zeros((300, 480), dtype=bool)
        mask[130, 220] = True
        mask_state.commit_mask(mask, 480, 300, 255)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            mask_state=mask_state,
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 95)
        self.assertTrue(prepared.preprocessing_metadata["frame_mask_applied"])
        self.assertFalse(
            prepared.preprocessing_metadata["manual_mask_applied_to_roi_locator"]
        )
        self.assertTrue(
            prepared.preprocessing_metadata[
                "manual_mask_applied_to_regressor_preprocessing"
            ]
        )
        self.assertEqual(prepared.preprocessing_metadata["frame_mask_revision"], 1)
        self.assertEqual(prepared.preprocessing_metadata["frame_mask_pixel_count"], 1)
        self.assertEqual(prepared.preprocessing_metadata["frame_mask_fill_value"], 255)

    def test_default_manual_mask_does_not_apply_before_roi_locator_with_black_fill(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        mask_state = FrameMaskState()
        mask = np.zeros((300, 480), dtype=bool)
        mask[0, 0] = True
        mask_state.commit_mask(mask, 480, 300, 0)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            mask_state=mask_state,
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[0, 0]), 255)
        self.assertTrue(prepared.preprocessing_metadata["frame_mask_applied"])
        self.assertFalse(
            prepared.preprocessing_metadata["manual_mask_applied_to_roi_locator"]
        )
        self.assertEqual(prepared.preprocessing_metadata["frame_mask_fill_value"], 0)

    def test_enabling_manual_mask_to_roi_locator_changes_locator_input(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        mask_state = FrameMaskState()
        mask = np.zeros((300, 480), dtype=bool)
        mask[130, 220] = True
        mask_state.commit_mask(mask, 480, 300, 255)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            mask_state=mask_state,
            stage_policy=StageTransformPolicySnapshot(
                apply_manual_mask_to_roi_locator=True
            ),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 0)
        self.assertTrue(
            prepared.preprocessing_metadata["manual_mask_applied_to_roi_locator"]
        )

    def test_manual_mask_to_roi_locator_uses_locator_background_fill_after_polarity(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        mask_state = FrameMaskState()
        mask = np.zeros((300, 480), dtype=bool)
        mask[130, 220] = True
        mask_state.commit_mask(mask, 480, 300, 255)

        TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            mask_state=mask_state,
            stage_policy=StageTransformPolicySnapshot(
                roi_locator_input_polarity=ROI_LOCATOR_INPUT_POLARITY_INVERTED,
                apply_manual_mask_to_roi_locator=True,
            ),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 0)

    def test_applied_mask_is_passed_as_roi_locator_exclusion_when_supported(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = ExclusionAwareRoiLocator()
        mask_state = FrameMaskState()
        mask = np.zeros((300, 480), dtype=bool)
        mask[130, 220] = True
        mask_state.commit_mask(mask, 480, 300, 0)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            mask_state=mask_state,
            stage_policy=StageTransformPolicySnapshot(
                apply_manual_mask_to_roi_locator=True
            ),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_excluded_source_mask)
        assert locator.last_excluded_source_mask is not None
        self.assertTrue(bool(locator.last_excluded_source_mask[130, 220]))
        self.assertTrue(
            prepared.preprocessing_metadata["frame_mask_excluded_from_roi_locator"]
        )

    def test_mismatched_mask_size_skips_mask_and_records_warning(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        mask_state = FrameMaskState()
        mask_state.commit_mask(np.ones((10, 10), dtype=bool), 10, 10, 0)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            mask_state=mask_state,
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[0, 0]), 255)
        metadata = prepared.preprocessing_metadata
        self.assertFalse(metadata["frame_mask_applied"])
        self.assertIn("frame mask skipped", metadata["frame_mask_warning"])
        self.assertTrue(
            any("frame mask skipped" in item for item in metadata["warnings"])
        )

    def test_masked_pixels_do_not_alter_latest_frame_on_disk(self) -> None:
        image_bytes = _fixture_image_bytes()
        mask_state = FrameMaskState()
        mask = np.zeros((300, 480), dtype=bool)
        mask[130, 220] = True
        mask_state.commit_mask(mask, 480, 300, 255)
        with TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "latest_frame.png"
            frame_path.write_bytes(image_bytes)

            TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=FakeRoiLocator(),
                mask_state=mask_state,
            ).prepare_model_inputs(
                _request(image_bytes, image_path=frame_path),
                image_bytes,
            )

            self.assertEqual(frame_path.read_bytes(), image_bytes)

    def test_captured_but_disabled_background_does_not_alter_roi_locator_input(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        background_state = BackgroundState()
        gray = _decode_gray(image_bytes)
        background_state.capture_background(_background_matching_one_pixel(gray, 130, 220))

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            background_state=background_state,
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 95)
        self.assertTrue(prepared.preprocessing_metadata["background_captured"])
        self.assertFalse(prepared.preprocessing_metadata["background_removal_enabled"])
        self.assertFalse(prepared.preprocessing_metadata["background_removal_applied"])

    def test_enabled_background_defaults_to_no_roi_locator_or_regressor_application(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        background_state = BackgroundState()
        gray = _decode_gray(image_bytes)
        background_state.capture_background(_background_matching_one_pixel(gray, 130, 220))
        background_state.set_enabled(True)
        background_state.set_threshold(25)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            background_state=background_state,
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 95)
        metadata = prepared.preprocessing_metadata
        self.assertTrue(metadata["background_captured"])
        self.assertTrue(metadata["background_removal_enabled"])
        self.assertFalse(metadata["background_removal_applied"])
        self.assertEqual(metadata["background_threshold"], 25)
        self.assertEqual(metadata["background_remove_pixel_count"], 0)
        self.assertEqual(metadata["background_roi_crop_remove_pixel_count"], 0)
        self.assertFalse(metadata["background_roi_crop_applied"])
        self.assertFalse(metadata["background_roi_fcn_applied"])
        self.assertFalse(metadata["background_removal_applied_to_roi_locator"])
        self.assertFalse(
            metadata["background_removal_applied_to_regressor_preprocessing"]
        )
        self.assertEqual(metadata["combined_ignore_pixel_count"], 0)
        self.assertEqual(metadata["frame_mask_fill_value"], 255)

    def test_enabling_background_removal_to_regressor_applies_to_regression_roi(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        background_state = BackgroundState()
        gray = _decode_gray(image_bytes)
        background_state.capture_background(_background_matching_one_pixel(gray, 130, 220))
        background_state.set_enabled(True)
        background_state.set_threshold(25)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            background_state=background_state,
            stage_policy=StageTransformPolicySnapshot(
                apply_background_removal_to_regressor_preprocessing=True
            ),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 95)
        metadata = prepared.preprocessing_metadata
        self.assertTrue(metadata["background_removal_applied"])
        self.assertEqual(metadata["background_remove_pixel_count"], 1)
        self.assertEqual(metadata["background_roi_crop_remove_pixel_count"], 1)
        self.assertTrue(metadata["background_roi_crop_applied"])
        self.assertFalse(metadata["background_roi_fcn_applied"])
        self.assertFalse(metadata["background_removal_applied_to_roi_locator"])
        self.assertTrue(
            metadata["background_removal_applied_to_regressor_preprocessing"]
        )

    def test_enabling_background_removal_to_roi_locator_changes_locator_debug_input(self) -> None:
        image_bytes = _fixture_image_bytes()
        background_state = BackgroundState()
        gray = _decode_gray(image_bytes)
        background_state.capture_background(_background_matching_one_pixel(gray, 130, 220))
        background_state.set_enabled(True)
        background_state.set_threshold(25)
        with TemporaryDirectory() as tmp_dir:
            request = _request(
                image_bytes,
                save_debug_images=True,
                debug_output_dir=Path(tmp_dir),
            )

            prepared = TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=FakeDebugRoiLocator(),
                background_state=background_state,
                stage_policy=StageTransformPolicySnapshot(
                    apply_background_removal_to_roi_locator=True
                ),
            ).prepare_model_inputs(request, image_bytes)

            paths = prepared.preprocessing_metadata["debug_paths"]
            raw = cv2.imread(
                str(paths[ARTIFACT_LOCATOR_INPUT_RAW_OR_PRETRANSFORM]),
                cv2.IMREAD_GRAYSCALE,
            )
            final = cv2.imread(
                str(paths[ARTIFACT_FINAL_LOCATOR_INPUT]),
                cv2.IMREAD_GRAYSCALE,
            )
            self.assertIsNotNone(raw)
            self.assertIsNotNone(final)
            assert raw is not None
            assert final is not None
            self.assertNotEqual(int(raw[130, 220]), int(final[130, 220]))
            self.assertIn(ARTIFACT_LOCATOR_INPUT_AFTER_BACKGROUND_REMOVAL, paths)

    def test_background_removal_can_remove_target_when_background_contains_target(self) -> None:
        image_bytes = _fixture_image_bytes()
        background_state = BackgroundState()
        gray = _decode_gray(image_bytes)
        background_state.capture_background(gray)
        background_state.set_enabled(True)
        background_state.set_threshold(1)
        with TemporaryDirectory() as tmp_dir:
            request = _request(
                image_bytes,
                save_debug_images=True,
                debug_output_dir=Path(tmp_dir),
            )

            prepared = TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=FakeDebugRoiLocator(),
                background_state=background_state,
                stage_policy=StageTransformPolicySnapshot(
                    apply_background_removal_to_roi_locator=True
                ),
            ).prepare_model_inputs(request, image_bytes)

            final = cv2.imread(
                str(prepared.preprocessing_metadata["debug_paths"][ARTIFACT_FINAL_LOCATOR_INPUT]),
                cv2.IMREAD_GRAYSCALE,
            )
            self.assertIsNotNone(final)
            assert final is not None
            self.assertEqual(int(final[130, 220]), 0)

    def test_background_removal_cannot_empty_silhouette_foreground(self) -> None:
        image_bytes = _fixture_image_bytes()
        background_state = BackgroundState()
        background_state.capture_background(_decode_gray(image_bytes))
        background_state.set_enabled(True)
        background_state.set_threshold(1)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
            background_state=background_state,
            stage_policy=StageTransformPolicySnapshot(
                apply_background_removal_to_regressor_preprocessing=True
            ),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        metadata = prepared.preprocessing_metadata
        self.assertTrue(metadata["background_roi_crop_applied"])
        self.assertGreater(metadata["background_roi_crop_remove_pixel_count"], 0)
        self.assertGreater(metadata["silhouette_area_px"], 0)
        self.assertGreater(
            metadata["brightness_normalization"]["ForegroundPixelCount"],
            0,
        )

    def test_background_removed_pixels_become_white_in_inverted_model_input(self) -> None:
        image_bytes = _fixture_image_bytes()
        background_state = BackgroundState()
        gray = _decode_gray(image_bytes)
        background_state.capture_background(_background_matching_one_pixel(gray, 130, 220))
        background_state.set_enabled(True)
        background_state.set_threshold(25)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE),
            roi_locator=FakeRoiLocator(),
            background_state=background_state,
            stage_policy=StageTransformPolicySnapshot(
                apply_background_removal_to_regressor_preprocessing=True
            ),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        metadata = prepared.preprocessing_metadata
        distance = prepared.model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY][0]
        self.assertTrue(
            metadata[
                contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_EXCLUDED_FROM_FOREGROUND
            ]
        )
        self.assertEqual(
            metadata[
                contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_FOREGROUND_REMOVE_PIXEL_COUNT
            ],
            1,
        )
        self.assertEqual(float(distance[130, 130]), 1.0)

    def test_manual_mask_and_background_removal_combine_with_or(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        mask_state = FrameMaskState()
        manual_mask = np.zeros((300, 480), dtype=bool)
        manual_mask[0, 0] = True
        mask_state.commit_mask(manual_mask, 480, 300, 0)
        background_state = BackgroundState()
        gray = _decode_gray(image_bytes)
        background_state.capture_background(_background_matching_one_pixel(gray, 130, 220))
        background_state.set_enabled(True)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            mask_state=mask_state,
            background_state=background_state,
            stage_policy=StageTransformPolicySnapshot(
                apply_manual_mask_to_roi_locator=True,
                apply_background_removal_to_regressor_preprocessing=True,
            ),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[0, 0]), 0)
        self.assertEqual(int(locator.last_source_gray[130, 220]), 95)
        self.assertEqual(prepared.preprocessing_metadata["combined_ignore_pixel_count"], 1)
        self.assertEqual(
            prepared.preprocessing_metadata["background_roi_crop_remove_pixel_count"],
            1,
        )

    def test_background_size_mismatch_skips_removal_and_records_warning(self) -> None:
        image_bytes = _fixture_image_bytes()
        locator = InspectingRoiLocator()
        background_state = BackgroundState()
        background_state.capture_background(np.zeros((10, 10), dtype=np.uint8))
        background_state.set_enabled(True)

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=locator,
            background_state=background_state,
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertIsNotNone(locator.last_source_gray)
        assert locator.last_source_gray is not None
        self.assertEqual(int(locator.last_source_gray[130, 220]), 95)
        metadata = prepared.preprocessing_metadata
        self.assertFalse(metadata["background_removal_applied"])
        self.assertIn("background removal skipped", metadata["background_warning"])
        self.assertTrue(
            any("background removal skipped" in item for item in metadata["warnings"])
        )

    def test_background_removal_does_not_alter_latest_frame_on_disk(self) -> None:
        image_bytes = _fixture_image_bytes()
        background_state = BackgroundState()
        gray = _decode_gray(image_bytes)
        background_state.capture_background(_background_matching_one_pixel(gray, 130, 220))
        background_state.set_enabled(True)
        with TemporaryDirectory() as tmp_dir:
            frame_path = Path(tmp_dir) / "latest_frame.png"
            frame_path.write_bytes(image_bytes)

            TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=FakeRoiLocator(),
                background_state=background_state,
            ).prepare_model_inputs(
                _request(image_bytes, image_path=frame_path),
                image_bytes,
            )

            self.assertEqual(frame_path.read_bytes(), image_bytes)

    def test_low_confidence_roi_is_rejected_before_model_inputs_are_built(self) -> None:
        image_bytes = _fixture_image_bytes()

        with self.assertRaises(RoiRejectedError) as raised:
            TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=LowConfidenceRoiLocator(),
            ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertEqual(raised.exception.failure_details["roi_accepted"], False)
        self.assertIn("low_confidence", raised.exception.failure_details["roi_rejection_reason"])
        self.assertFalse(raised.exception.preprocessing_metadata["roi_accepted"])

    def test_clipped_roi_is_rejected_before_model_inputs_are_built(self) -> None:
        image_bytes = _fixture_image_bytes()

        with self.assertRaises(RoiRejectedError) as raised:
            TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=ClippedRoiLocator(),
            ).prepare_model_inputs(_request(image_bytes), image_bytes)

        self.assertEqual(raised.exception.failure_details["roi_accepted"], False)
        self.assertIn("clipped_roi", raised.exception.failure_details["roi_rejection_reason"])
        self.assertTrue(raised.exception.failure_details["roi_clipped"])

    def test_large_clipped_roi_still_rejects_with_clip_tolerance(self) -> None:
        image_bytes = _fixture_image_bytes()

        with self.assertRaises(RoiRejectedError) as raised:
            TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=CenteredRoiLocator(center_xy_px=(240.0, 276.0)),
                stage_policy=StageTransformPolicySnapshot(roi_clip_tolerance_px=10),
            ).prepare_model_inputs(_request(image_bytes), image_bytes)

        metadata = raised.exception.preprocessing_metadata
        self.assertEqual(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_BOTTOM_PX], 126)
        self.assertEqual(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX], 126)
        self.assertEqual(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX], 10)
        self.assertFalse(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED])
        self.assertIn(
            "clipped_roi",
            metadata[contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASON],
        )

    def test_slightly_clipped_roi_is_allowed_with_clip_tolerance(self) -> None:
        image_bytes = _fixture_image_bytes()

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=CenteredRoiLocator(center_xy_px=(240.0, 156.0)),
            stage_policy=StageTransformPolicySnapshot(roi_clip_tolerance_px=10),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        metadata = prepared.preprocessing_metadata
        self.assertTrue(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIPPED])
        self.assertEqual(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_BOTTOM_PX], 6)
        self.assertEqual(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX], 6)
        self.assertEqual(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX], 10)
        self.assertTrue(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED])
        self.assertTrue(metadata[contracts.PREPROCESSING_METADATA_ROI_ACCEPTED])
        self.assertIn(contracts.TRI_STREAM_DISTANCE_IMAGE_KEY, prepared.model_inputs)

    def test_unclipped_roi_allows_and_records_zero_clip_amounts(self) -> None:
        image_bytes = _fixture_image_bytes()

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
            stage_policy=StageTransformPolicySnapshot(roi_clip_tolerance_px=10),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        metadata = prepared.preprocessing_metadata
        self.assertFalse(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIPPED])
        self.assertEqual(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX], 0)
        self.assertEqual(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX], 10)
        self.assertFalse(metadata[contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED])

    def test_invalid_image_bytes_fail_clearly(self) -> None:
        preprocessor = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        )

        with self.assertRaisesRegex(ValueError, "decode image bytes"):
            preprocessor.prepare_model_inputs(_request(b"not-an-image"), b"not-an-image")

    def test_metadata_includes_contract_geometry_orientation_and_roi_information(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(image_bytes)
        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)

        metadata = prepared.preprocessing_metadata

        self.assertEqual(
            metadata["preprocessing_contract_name"],
            contracts.PREPROCESSING_CONTRACT_NAME,
        )
        self.assertEqual(metadata["geometry_schema"], contracts.TRI_STREAM_GEOMETRY_SCHEMA)
        self.assertEqual(metadata["geometry_dim"], 10)
        self.assertEqual(metadata["orientation_source_mode"], ORIENTATION_SOURCE_RAW_GRAYSCALE)
        self.assertEqual(metadata["input_image_hash"], request.frame.frame_hash.value)
        self.assertIn("roi_request_xyxy_px", metadata)
        self.assertIn("roi_source_xyxy_px", metadata)
        self.assertIn("silhouette_bbox_xyxy_px", metadata)
        self.assertGreater(metadata["silhouette_area_px"], 0)
        self.assertEqual(metadata["predicted_roi_center_xy_px"], (240.0, 150.0))
        self.assertEqual(metadata["roi_locator_metadata"], {"locator": "fake"})

    def test_raw_grayscale_orientation_source_mode_is_supported(self) -> None:
        image_bytes = _fixture_image_bytes()
        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        orientation = prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY]

        self.assertEqual(
            prepared.preprocessing_metadata["orientation_source_mode"],
            ORIENTATION_SOURCE_RAW_GRAYSCALE,
        )
        self.assertGreater(int(np.count_nonzero(orientation < 1.0)), 0)

    def test_inverted_vehicle_orientation_source_mode_is_supported_by_current_manifest(self) -> None:
        image_bytes = _fixture_image_bytes()
        prepared = TriStreamLivePreprocessor(
            model_manifest=_current_or_fixture_inverted_manifest(),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        distance = prepared.model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY]
        orientation = prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY]
        brightness = prepared.preprocessing_metadata["brightness_normalization"]

        self.assertEqual(
            prepared.preprocessing_metadata["orientation_source_mode"],
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
        )
        self.assertGreater(int(np.count_nonzero(distance < 1.0)), 0)
        self.assertGreater(int(np.count_nonzero(orientation < 1.0)), 0)
        self.assertIn("Status", brightness)
        self.assertGreater(brightness["ForegroundPixelCount"], 0)

    def test_raw_grayscale_on_white_distance_renderer_is_manifest_driven(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(image_bytes)
        raw_prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(
                ORIENTATION_SOURCE_RAW_GRAYSCALE,
                image_representation_mode="raw_grayscale_on_white",
            ),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)
        inverted_prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(
                ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
                image_representation_mode="inverted_vehicle_on_white",
            ),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)

        raw_distance = raw_prepared.model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY][0]
        inverted_distance = inverted_prepared.model_inputs[
            contracts.TRI_STREAM_DISTANCE_IMAGE_KEY
        ][0]

        self.assertEqual(
            raw_prepared.preprocessing_metadata["image_representation_mode"],
            "raw_grayscale_on_white",
        )
        self.assertLess(float(raw_distance[120, 120]), 0.25)
        self.assertGreater(float(inverted_distance[120, 120]), 0.75)

    def test_foreground_enhancement_is_applied_to_live_raw_grayscale_inputs(self) -> None:
        image_bytes = _light_fixture_image_bytes()
        request = _request(image_bytes)
        base_prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(
                ORIENTATION_SOURCE_RAW_GRAYSCALE,
                image_representation_mode="raw_grayscale_on_white",
            ),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)
        enhanced_prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(
                ORIENTATION_SOURCE_RAW_GRAYSCALE,
                image_representation_mode="raw_grayscale_on_white",
                foreground_enhancement=True,
            ),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)

        base_distance = base_prepared.model_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY][0]
        enhanced_distance = enhanced_prepared.model_inputs[
            contracts.TRI_STREAM_DISTANCE_IMAGE_KEY
        ][0]
        base_orientation = base_prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY][0]
        enhanced_orientation = enhanced_prepared.model_inputs[
            contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY
        ][0]
        foreground = base_distance < 0.999

        self.assertEqual(
            enhanced_prepared.preprocessing_metadata["foreground_enhancement"]["Status"],
            "success",
        )
        self.assertGreater(
            enhanced_prepared.preprocessing_metadata["foreground_enhancement"]["Gain"],
            1.0,
        )
        self.assertLess(float(np.mean(enhanced_distance[foreground])), float(np.mean(base_distance[foreground])))
        self.assertLess(
            float(np.mean(enhanced_orientation[enhanced_orientation < 0.999])),
            float(np.mean(base_orientation[base_orientation < 0.999])),
        )

    def test_orientation_source_modes_use_distinct_source_images(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(image_bytes)
        raw_prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)
        inverted_prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)

        raw_orientation = raw_prepared.model_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY]
        inverted_orientation = inverted_prepared.model_inputs[
            contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY
        ]

        self.assertFalse(np.allclose(raw_orientation, inverted_orientation))

    def test_frame_hash_mismatch_is_reported_as_metadata_warning(self) -> None:
        image_bytes = _fixture_image_bytes()
        request = _request(
            image_bytes,
            frame_hash=FrameHash("0" * 32),
        )

        prepared = TriStreamLivePreprocessor(
            model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
            roi_locator=FakeRoiLocator(),
        ).prepare_model_inputs(request, image_bytes)

        self.assertTrue(
            any(
                "frame_hash does not match" in item
                for item in prepared.preprocessing_metadata["warnings"]
            )
        )

    def test_debug_enabled_populates_paths_and_writes_valid_images(self) -> None:
        image_bytes = _fixture_image_bytes()
        with TemporaryDirectory() as tmp_dir:
            request = _request(
                image_bytes,
                save_debug_images=True,
                debug_output_dir=Path(tmp_dir),
            )

            prepared = TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=FakeDebugRoiLocator(),
            ).prepare_model_inputs(request, image_bytes)

            paths = prepared.preprocessing_metadata["debug_paths"]
            for key in (
                ARTIFACT_ACCEPTED_RAW_FRAME,
                ARTIFACT_FINAL_LOCATOR_INPUT,
                ARTIFACT_LOCATOR_INPUT,
                ARTIFACT_DISTANCE_IMAGE,
                ARTIFACT_ORIENTATION_IMAGE,
            ):
                self.assertIn(key, paths)
                self.assertIsNotNone(cv2.imread(str(paths[key]), cv2.IMREAD_UNCHANGED))

            metadata_path = Path(paths[ARTIFACT_ROI_OVERLAY_METADATA])
            self.assertTrue(metadata_path.is_file())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["request_id"], request.request_id)
            self.assertEqual(metadata["input_image_hash"], request.frame.frame_hash.value)
            self.assertEqual(metadata["orientation_source_mode"], ORIENTATION_SOURCE_RAW_GRAYSCALE)

    def test_debug_trace_images_include_locator_input_before_and_after_polarity(self) -> None:
        image_bytes = _fixture_image_bytes()
        with TemporaryDirectory() as tmp_dir:
            request = _request(
                image_bytes,
                save_debug_images=True,
                debug_output_dir=Path(tmp_dir),
            )

            prepared = TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=FakeDebugRoiLocator(),
                stage_policy=StageTransformPolicySnapshot(
                    roi_locator_input_polarity=ROI_LOCATOR_INPUT_POLARITY_INVERTED
                ),
            ).prepare_model_inputs(request, image_bytes)

            paths = prepared.preprocessing_metadata["debug_paths"]
            self.assertIn(ARTIFACT_LOCATOR_INPUT_BEFORE_POLARITY, paths)
            self.assertIn(ARTIFACT_LOCATOR_INPUT_AFTER_POLARITY, paths)
            before = cv2.imread(
                str(paths[ARTIFACT_LOCATOR_INPUT_BEFORE_POLARITY]),
                cv2.IMREAD_GRAYSCALE,
            )
            after = cv2.imread(
                str(paths[ARTIFACT_LOCATOR_INPUT_AFTER_POLARITY]),
                cv2.IMREAD_GRAYSCALE,
            )
            self.assertIsNotNone(before)
            self.assertIsNotNone(after)
            assert before is not None
            assert after is not None
            self.assertEqual(int(before[130, 220]), 95)
            self.assertEqual(int(after[130, 220]), 160)

    def test_debug_disabled_does_not_write_debug_paths(self) -> None:
        image_bytes = _fixture_image_bytes()
        with TemporaryDirectory() as tmp_dir:
            prepared = TriStreamLivePreprocessor(
                model_manifest=_fixture_manifest(ORIENTATION_SOURCE_RAW_GRAYSCALE),
                roi_locator=FakeDebugRoiLocator(),
            ).prepare_model_inputs(
                _request(
                    image_bytes,
                    save_debug_images=False,
                    debug_output_dir=Path(tmp_dir),
                ),
                image_bytes,
            )

            self.assertNotIn("debug_paths", prepared.preprocessing_metadata)
            self.assertEqual(list(Path(tmp_dir).iterdir()), [])

    def test_generic_core_modules_remain_heavy_import_free(self) -> None:
        module_paths = (
            SRC_ROOT / "interfaces/contracts.py",
            SRC_ROOT / "live_inference/frame_handoff.py",
            SRC_ROOT / "live_inference/frame_selection.py",
            SRC_ROOT / "live_inference/inference_core.py",
            SRC_ROOT / "live_inference/runtime_parameters.py",
            SRC_ROOT / "live_inference/model_registry/compatibility.py",
        )

        found = {str(path): _banned_imports(path) for path in module_paths}

        self.assertTrue(all(not imports for imports in found.values()), found)

    def test_preprocessor_adds_no_pyside_or_model_inference_imports(self) -> None:
        module_paths = (
            SRC_ROOT / "live_inference/preprocessing/__init__.py",
            SRC_ROOT / "live_inference/preprocessing/preprocessing_config.py",
            SRC_ROOT / "live_inference/preprocessing/roi_fcn_locator.py",
            SRC_ROOT / "live_inference/preprocessing/roi_locator.py",
            SRC_ROOT / "live_inference/preprocessing/tri_stream_live_preprocessor.py",
        )

        found = {
            str(path): _banned_imports(
                path,
                banned_roots={"PySide6", "torch", "src"},
            )
            for path in module_paths
        }

        self.assertTrue(all(not imports for imports in found.values()), found)


class FakeRoiLocator:
    def locate(self, source_gray_image: object) -> RoiLocation:
        return RoiLocation(
            center_xy_px=(240.0, 150.0),
            roi_bounds_xyxy_px=(90.0, 0.0, 390.0, 300.0),
            metadata={"locator": "fake"},
        )


class FakeDebugRoiLocator:
    def locate(
        self,
        source_gray_image: object,
        *,
        background_snapshot: object | None = None,
        background_fill_value: int = 255,
    ) -> RoiLocation:
        metadata = {
            "locator": "fake",
            "locator_canvas_width_px": 480,
            "locator_canvas_height_px": 300,
        }
        if background_snapshot is not None:
            metadata[
                contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_REMOVAL_APPLIED
            ] = True
            metadata[
                contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_REMOVE_PIXEL_COUNT
            ] = 1
        return RoiLocation(
            center_xy_px=(240.0, 150.0),
            roi_bounds_xyxy_px=(90.0, 0.0, 390.0, 300.0),
            metadata=metadata,
        )


class LowConfidenceRoiLocator:
    def locate(self, source_gray_image: object) -> RoiLocation:
        return RoiLocation(
            center_xy_px=(240.0, 150.0),
            roi_bounds_xyxy_px=(90.0, 0.0, 390.0, 300.0),
            metadata={
                "heatmap_peak_confidence": 0.12,
                "locator_canvas_width_px": 480,
                "locator_canvas_height_px": 300,
            },
        )


class ClippedRoiLocator:
    def locate(self, source_gray_image: object) -> RoiLocation:
        return RoiLocation(
            center_xy_px=(470.0, 290.0),
            roi_bounds_xyxy_px=(320.0, 140.0, 620.0, 440.0),
            metadata={
                "heatmap_peak_confidence": 0.95,
                "locator_canvas_width_px": 480,
                "locator_canvas_height_px": 300,
            },
        )


class CenteredRoiLocator:
    def __init__(self, *, center_xy_px: tuple[float, float]) -> None:
        self.center_xy_px = center_xy_px

    def locate(self, source_gray_image: object) -> RoiLocation:
        center_x, center_y = self.center_xy_px
        return RoiLocation(
            center_xy_px=(float(center_x), float(center_y)),
            roi_bounds_xyxy_px=(
                float(center_x) - 150.0,
                float(center_y) - 150.0,
                float(center_x) + 150.0,
                float(center_y) + 150.0,
            ),
            metadata={
                "heatmap_peak_confidence": 0.95,
                "locator_canvas_width_px": 480,
                "locator_canvas_height_px": 300,
            },
        )


class InspectingRoiLocator:
    def __init__(self) -> None:
        self.last_source_gray: np.ndarray | None = None

    def locate(self, source_gray_image: object) -> RoiLocation:
        self.last_source_gray = np.array(source_gray_image, copy=True)
        return RoiLocation(
            center_xy_px=(240.0, 150.0),
            roi_bounds_xyxy_px=(90.0, 0.0, 390.0, 300.0),
            metadata={"locator": "inspecting"},
        )


class ExclusionAwareRoiLocator:
    def __init__(self) -> None:
        self.last_excluded_source_mask: np.ndarray | None = None

    def locate(
        self,
        source_gray_image: object,
        *,
        excluded_source_mask: np.ndarray | None = None,
    ) -> RoiLocation:
        if excluded_source_mask is not None:
            self.last_excluded_source_mask = np.array(excluded_source_mask, copy=True)
        return RoiLocation(
            center_xy_px=(240.0, 150.0),
            roi_bounds_xyxy_px=(90.0, 0.0, 390.0, 300.0),
            metadata={"locator": "exclusion-aware"},
        )


def _fixture_image_bytes() -> bytes:
    image = np.full((300, 480), 255, dtype=np.uint8)
    cv2.rectangle(image, (210, 115), (270, 185), 30, -1)
    cv2.line(image, (220, 130), (260, 170), 95, 5)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode fixture image")
    return encoded.tobytes()


def _light_fixture_image_bytes() -> bytes:
    image = np.full((300, 480), 255, dtype=np.uint8)
    cv2.rectangle(image, (210, 115), (270, 185), 180, -1)
    cv2.line(image, (220, 130), (260, 170), 200, 5)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode light fixture image")
    return encoded.tobytes()


def _decode_gray(image_bytes: bytes) -> np.ndarray:
    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise RuntimeError("Failed to decode fixture image")
    return decoded


def _background_matching_one_pixel(gray: np.ndarray, y: int, x: int) -> np.ndarray:
    background = np.where(gray < 128, 255, 0).astype(np.uint8)
    background[int(y), int(x)] = int(gray[int(y), int(x)])
    return background


def _request(
    image_bytes: bytes,
    *,
    frame_hash: FrameHash | None = None,
    save_debug_images: bool = False,
    debug_output_dir: Path | None = None,
    image_path: Path = Path("live_frames/latest_frame.png"),
) -> InferenceRequest:
    frame_hash = frame_hash or compute_frame_hash(image_bytes)
    frame = FrameReference(
        image_path=image_path,
        metadata=FrameMetadata(
            width_px=480,
            height_px=300,
            pixel_format="gray8",
            encoding="png",
            byte_size=len(image_bytes),
        ),
        frame_hash=frame_hash,
        byte_size=len(image_bytes),
    )
    return InferenceRequest(
        request_id="request-1",
        frame=frame,
        requested_at_utc=REQUESTED_AT,
        save_debug_images=save_debug_images,
        debug_output_dir=debug_output_dir,
    )


def _current_or_fixture_inverted_manifest() -> object:
    current_root = (
        PROJECT_ROOT
        / "models/distance-orientation/260504-1100_ts-2d-cnn__run_0001"
    )
    if current_root.is_dir():
        return load_live_model_manifest(current_root)
    return _fixture_manifest(ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE)


def _fixture_manifest(
    orientation_source_mode: str,
    *,
    image_representation_mode: str = "inverted_vehicle_on_white",
    foreground_enhancement: bool = False,
) -> object:
    temp_dir = TemporaryDirectory()
    _TEMP_DIRS.append(temp_dir)
    root = Path(temp_dir.name) / "model-run"
    root.mkdir()
    _write_json(root / "config.json", _model_config())
    _write_json(
        root / "dataset_summary.json",
        {
            "preprocessing_contract": _preprocessing_contract(
                orientation_source_mode,
                image_representation_mode=image_representation_mode,
                foreground_enhancement=foreground_enhancement,
            )
        },
    )
    return load_live_model_manifest(root)


_TEMP_DIRS: list[TemporaryDirectory] = []


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _model_config() -> dict[str, object]:
    return {
        "model_name": "distance-orientation-test",
        "topology_id": "distance_regressor_tri_stream_yaw",
        "topology_variant": "tri_stream_yaw_v0_1",
        "topology_contract": {
            "contract_version": contracts.MODEL_TOPOLOGY_CONTRACT_VERSION,
            "runtime": {"input_mode": contracts.TRI_STREAM_INPUT_MODE},
            "outputs": {
                "distance": {
                    "output_key": contracts.MODEL_OUTPUT_DISTANCE_KEY,
                    "columns": ["distance_m"],
                },
                "yaw": {
                    "output_key": contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY,
                    "columns": ["yaw_sin", "yaw_cos"],
                },
            },
        },
        "topology_params": {
            "geometry_feature_dim": len(contracts.TRI_STREAM_GEOMETRY_SCHEMA)
        },
    }


def _orientation_semantics(
    orientation_source_mode: str,
    *,
    image_representation_mode: str,
    foreground_enhancement: bool,
) -> dict[str, str | None]:
    suffix = "_foreground_enhanced" if foreground_enhancement else ""
    prefix = "foreground_enhanced_" if foreground_enhancement else ""
    if orientation_source_mode == ORIENTATION_SOURCE_RAW_GRAYSCALE:
        content = (
            f"{prefix}raw_grayscale_detail_on_white_no_brightness_normalization"
            if image_representation_mode == "raw_grayscale_on_white"
            else "raw_grayscale_detail_preserving_no_brightness_normalization"
        )
        return {
            "representation": f"target_centered_raw_grayscale_scaled_by_silhouette_extent{suffix}",
            "content": content,
            "polarity": (
                "source_grayscale_vehicle_detail_on_white_background"
                if image_representation_mode == "raw_grayscale_on_white"
                else None
            ),
        }
    if orientation_source_mode == ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE:
        return {
            "representation": (
                "target_centered_inverted_vehicle_on_white_scaled_by_silhouette_extent"
                f"{suffix}"
            ),
            "content": f"{prefix}inverted_vehicle_detail_on_white_no_brightness_normalization",
            "polarity": "dark_vehicle_detail_on_white_background",
        }
    raise ValueError(f"Unsupported test orientation source mode: {orientation_source_mode}")


def _preprocessing_contract(
    orientation_source_mode: str,
    *,
    image_representation_mode: str = "inverted_vehicle_on_white",
    foreground_enhancement: bool = False,
) -> dict[str, object]:
    orientation_semantics = _orientation_semantics(
        orientation_source_mode,
        image_representation_mode=image_representation_mode,
        foreground_enhancement=foreground_enhancement,
    )
    image_mode_contract = (
        "roi_raw_grayscale_vehicle_on_white"
        if image_representation_mode == "raw_grayscale_on_white"
        else "roi_grayscale_inverted_vehicle_on_white"
    )
    image_polarity = (
        "source_grayscale_vehicle_detail_on_white_background"
        if image_representation_mode == "raw_grayscale_on_white"
        else "dark_vehicle_detail_on_white_background"
    )
    image_content_base = (
        "raw_grayscale_vehicle_detail_inside_silhouette"
        if image_representation_mode == "raw_grayscale_on_white"
        else "inverted_grayscale_vehicle_detail_inside_silhouette"
    )
    foreground_contract = {
        "Enabled": bool(foreground_enhancement),
        "Method": "masked_median_darkness_gain" if foreground_enhancement else "none",
        "TargetMedianDarkness": 0.70,
        "MinGain": 1.0,
        "MaxGain": 3.0,
        "Epsilon": 1e-6,
        "EmptyMaskPolicy": "skip",
    }
    current_representation: dict[str, object] = {
        "Kind": contracts.TRI_STREAM_REPRESENTATION_KIND,
        "StorageFormat": "npz",
        "ArrayKeys": [
            contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
            contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
            contracts.TRI_STREAM_GEOMETRY_KEY,
            "y_distance_m",
            "y_yaw_deg",
            "y_yaw_sin",
            "y_yaw_cos",
        ],
        "DistanceImageKey": contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
        "OrientationImageKey": contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
        "GeometryKey": contracts.TRI_STREAM_GEOMETRY_KEY,
        "GeometrySchema": list(contracts.TRI_STREAM_GEOMETRY_SCHEMA),
        "GeometryDim": len(contracts.TRI_STREAM_GEOMETRY_SCHEMA),
        "CanvasWidth": 300,
        "CanvasHeight": 300,
        "DistanceImagePolarity": image_polarity,
        "DistanceImageContent": (
            f"{image_content_base}_foreground_enhanced"
            if foreground_enhancement
            else image_content_base
        ),
        "OrientationContextScale": 1.25,
        "OrientationImageContent": orientation_semantics["content"],
        "ForegroundEnhancement": foreground_contract,
        "BrightnessNormalization": {
            "Enabled": False,
            "Method": "none",
            "TargetMedianDarkness": 0.55,
            "MinGain": 0.5,
            "MaxGain": 2.0,
            "Epsilon": 1e-6,
            "EmptyMaskPolicy": "skip",
        },
    }
    if orientation_semantics["polarity"]:
        current_representation["OrientationImagePolarity"] = orientation_semantics["polarity"]
    contract_name = (
        contracts.PREPROCESSING_CONTRACT_NAME_GRAYSCALE_WHITE
        if image_representation_mode == "raw_grayscale_on_white"
        else contracts.PREPROCESSING_CONTRACT_NAME
    )
    return {
        "ContractVersion": contract_name,
        "CurrentStage": "pack_tri_stream",
        "CompletedStages": ["detect", "silhouette", "pack_tri_stream"],
        "CurrentRepresentation": current_representation,
        "Stages": {
            "pack_tri_stream": {
                "CanvasWidth": 300,
                "CanvasHeight": 300,
                "ClipPolicy": "fail",
                "ImageRepresentationMode": image_mode_contract,
                "ForegroundEnhancement": foreground_contract,
                "OrientationContextScale": 1.25,
                "OrientationImageRepresentation": orientation_semantics["representation"],
                "BrightnessNormalization": current_representation["BrightnessNormalization"],
            },
            "silhouette": {
                "BlurKernelSize": 5,
                "CannyHighThreshold": 150,
                "CannyLowThreshold": 50,
                "CloseKernelSize": 3,
                "DilateKernelSize": 1,
                "FallbackId": "fallback.convex_hull_v1",
                "FillHoles": True,
                "GeneratorId": "silhouette.contour_v2",
                "MinComponentAreaPx": 50,
                "OutlineThicknessPx": 1,
                "PersistDebug": False,
                "ROICanvasHeightPx": 300,
                "ROICanvasWidthPx": 300,
                "ROIPaddingPx": 0,
                "RepresentationMode": "filled",
                "UseConvexHullFallback": True,
            },
        },
    }


def _banned_imports(
    module_path: Path,
    *,
    banned_roots: set[str] | None = None,
) -> set[str]:
    banned_roots = banned_roots or {"PySide6", "cv2", "numpy", "torch", "pandas"}
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    return imported_roots & banned_roots


if __name__ == "__main__":
    unittest.main()
