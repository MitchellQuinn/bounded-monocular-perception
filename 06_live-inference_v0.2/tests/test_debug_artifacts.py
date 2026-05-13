"""Tests for preprocessing debug artifact writing."""

from __future__ import annotations

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
from interfaces import FrameHash  # noqa: E402
from live_inference.preprocessing.debug_artifacts import (  # noqa: E402
    ARTIFACT_ACCEPTED_RAW_FRAME,
    ARTIFACT_DISTANCE_IMAGE,
    ARTIFACT_ORIENTATION_IMAGE,
    ARTIFACT_ROI_OVERLAY_METADATA,
    DebugArtifactWriter,
)


class DebugArtifactWriterTests(unittest.TestCase):
    def test_writer_writes_expected_images_and_metadata_json(self) -> None:
        frame_hash = FrameHash("abcdef1234567890abcdef1234567890")
        with TemporaryDirectory() as tmp_dir:
            writer = DebugArtifactWriter(enabled=True, output_dir=Path(tmp_dir))

            paths = writer.write_preprocessing_artifacts(
                request_id="request-1",
                input_image_hash=frame_hash,
                preprocessing_parameter_revision=9,
                image_artifacts={
                    ARTIFACT_ACCEPTED_RAW_FRAME: np.full((12, 16), 128, dtype=np.uint8),
                    ARTIFACT_DISTANCE_IMAGE: np.zeros((8, 8), dtype=np.float32),
                    ARTIFACT_ORIENTATION_IMAGE: np.ones((8, 8), dtype=np.float32),
                },
                metadata={
                    "orientation_source_mode": "raw_grayscale",
                    "input_keys": contracts.TRI_STREAM_INPUT_KEYS,
                },
            )

            self.assertIn(ARTIFACT_ACCEPTED_RAW_FRAME, paths)
            self.assertIn(ARTIFACT_DISTANCE_IMAGE, paths)
            self.assertIn(ARTIFACT_ORIENTATION_IMAGE, paths)
            self.assertIn(ARTIFACT_ROI_OVERLAY_METADATA, paths)
            for key in (
                ARTIFACT_ACCEPTED_RAW_FRAME,
                ARTIFACT_DISTANCE_IMAGE,
                ARTIFACT_ORIENTATION_IMAGE,
            ):
                self.assertIsNotNone(cv2.imread(str(paths[key]), cv2.IMREAD_UNCHANGED))

            metadata = json.loads(paths[ARTIFACT_ROI_OVERLAY_METADATA].read_text())
            self.assertEqual(metadata["request_id"], "request-1")
            self.assertEqual(metadata["input_image_hash"], frame_hash.value)
            self.assertEqual(metadata["preprocessing_parameter_revision"], 9)
            self.assertEqual(metadata["orientation_source_mode"], "raw_grayscale")

    def test_filenames_include_request_id_and_short_hash_prefix(self) -> None:
        frame_hash = FrameHash("1234567890abcdef1234567890abcdef")
        with TemporaryDirectory() as tmp_dir:
            writer = DebugArtifactWriter(enabled=True, output_dir=Path(tmp_dir))

            paths = writer.write_preprocessing_artifacts(
                request_id="request/with spaces",
                input_image_hash=frame_hash,
                preprocessing_parameter_revision=None,
                image_artifacts={
                    ARTIFACT_ACCEPTED_RAW_FRAME: np.zeros((4, 4), dtype=np.uint8),
                },
                metadata={},
            )

            filename = paths[ARTIFACT_ACCEPTED_RAW_FRAME].name
            self.assertTrue(filename.startswith("request-with-spaces__1234567890ab__"))

    def test_disabled_writer_creates_no_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            writer = DebugArtifactWriter(enabled=False, output_dir=Path(tmp_dir))

            paths = writer.write_preprocessing_artifacts(
                request_id="request-1",
                input_image_hash=FrameHash("hash"),
                preprocessing_parameter_revision=None,
                image_artifacts={
                    ARTIFACT_ACCEPTED_RAW_FRAME: np.zeros((4, 4), dtype=np.uint8),
                },
                metadata={},
            )

            self.assertEqual(paths, {})
            self.assertEqual(list(Path(tmp_dir).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
