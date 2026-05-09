"""Unit tests for inference frame selection."""

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
    FrameMetadata,
    FrameReference,
    FrameSkipReason,
    LiveInferenceConfig,
)
from live_inference.frame_handoff import (  # noqa: E402
    AtomicFrameHandoffWriter,
    LatestFrameHandoffReader,
    compute_frame_hash,
)
from live_inference.frame_selection import InferenceFrameSelector  # noqa: E402


REQUESTED_AT = "2026-05-03T10:00:00Z"


class InferenceFrameSelectorTests(unittest.TestCase):
    def _config(self, tmp_dir: str) -> LiveInferenceConfig:
        return LiveInferenceConfig(frame_dir=Path(tmp_dir) / "live_frames")

    def _selector(self, reader: object, **kwargs: object) -> InferenceFrameSelector:
        return InferenceFrameSelector(
            reader,  # type: ignore[arg-type]
            request_id_factory=lambda: "request-1",
            now_utc_fn=lambda: REQUESTED_AT,
            **kwargs,
        )

    def test_returns_skipped_missing_file_when_no_latest_frame_exists(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            selector = self._selector(LatestFrameHandoffReader(self._config(tmp_dir)))

            result = selector.select_latest()

            self.assertIsNone(result.selected)
            self.assertIsNone(result.warning)
            self.assertIsNotNone(result.skipped)
            assert result.skipped is not None
            self.assertEqual(result.skipped.reason, FrameSkipReason.MISSING_FILE)

    def test_returns_selected_frame_when_latest_exists(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            AtomicFrameHandoffWriter(config).publish_frame(
                b"frame-bytes",
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(LatestFrameHandoffReader(config))

            result = selector.select_latest()

            self.assertIsNotNone(result.selected)
            self.assertIsNone(result.skipped)
            self.assertIsNone(result.warning)
            assert result.selected is not None
            self.assertEqual(result.selected.request.request_id, "request-1")
            self.assertEqual(result.selected.request.requested_at_utc, REQUESTED_AT)

    def test_selected_image_bytes_exactly_match_published_bytes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            image_bytes = b"\x89PNG\r\npublished bytes"
            AtomicFrameHandoffWriter(config).publish_frame(
                image_bytes,
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(LatestFrameHandoffReader(config))

            result = selector.select_latest()

            self.assertIsNotNone(result.selected)
            assert result.selected is not None
            self.assertEqual(result.selected.image_bytes, image_bytes)

    def test_selected_frame_hash_matches_compute_frame_hash_image_bytes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            image_bytes = b"hash me"
            AtomicFrameHandoffWriter(config).publish_frame(
                image_bytes,
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(LatestFrameHandoffReader(config))

            result = selector.select_latest()

            self.assertIsNotNone(result.selected)
            assert result.selected is not None
            self.assertEqual(
                result.selected.frame_hash,
                compute_frame_hash(result.selected.image_bytes),
            )

    def test_returned_inference_request_frame_hash_is_populated(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            AtomicFrameHandoffWriter(config).publish_frame(
                b"frame-bytes",
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(LatestFrameHandoffReader(config))

            result = selector.select_latest()

            self.assertIsNotNone(result.selected)
            assert result.selected is not None
            self.assertEqual(
                result.selected.request.frame.frame_hash,
                result.selected.frame_hash,
            )

    def test_returned_inference_request_frame_byte_size_is_populated(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            image_bytes = b"frame-bytes"
            AtomicFrameHandoffWriter(config).publish_frame(
                image_bytes,
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(LatestFrameHandoffReader(config))

            result = selector.select_latest()

            self.assertIsNotNone(result.selected)
            assert result.selected is not None
            self.assertEqual(result.selected.request.frame.byte_size, len(image_bytes))

    def test_duplicate_is_not_skipped_before_mark_processed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            AtomicFrameHandoffWriter(config).publish_frame(
                b"same",
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(LatestFrameHandoffReader(config))

            first = selector.select_latest()
            second = selector.select_latest()

            self.assertIsNotNone(first.selected)
            self.assertIsNotNone(second.selected)
            self.assertIsNone(second.skipped)

    def test_duplicate_is_skipped_after_mark_processed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            AtomicFrameHandoffWriter(config).publish_frame(
                b"same",
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(LatestFrameHandoffReader(config))
            first = selector.select_latest()
            self.assertIsNotNone(first.selected)
            assert first.selected is not None

            selector.mark_processed(first.selected.frame_hash)
            second = selector.select_latest()

            self.assertIsNone(second.selected)
            self.assertIsNotNone(second.skipped)
            assert second.skipped is not None
            self.assertEqual(second.skipped.reason, FrameSkipReason.DUPLICATE_HASH)
            self.assertEqual(second.skipped.frame_hash, first.selected.frame_hash)

    def test_different_bytes_after_mark_processed_are_selected(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            writer = AtomicFrameHandoffWriter(config)
            reader = LatestFrameHandoffReader(config)
            writer.publish_frame(b"first", FrameMetadata(frame_id="frame-1"))
            selector = self._selector(reader)
            first = selector.select_latest()
            self.assertIsNotNone(first.selected)
            assert first.selected is not None
            selector.mark_processed(first.selected.frame_hash)

            writer.publish_frame(b"second", FrameMetadata(frame_id="frame-2"))
            second = selector.select_latest()

            self.assertIsNotNone(second.selected)
            self.assertIsNone(second.skipped)
            assert second.selected is not None
            self.assertEqual(second.selected.image_bytes, b"second")

    def test_read_failure_returns_worker_warning_with_failure_stage_read(self) -> None:
        reader = FailingReader()
        selector = self._selector(reader)

        result = selector.select_latest()

        self.assertIsNone(result.selected)
        self.assertIsNone(result.skipped)
        self.assertIsNotNone(result.warning)
        assert result.warning is not None
        self.assertEqual(result.warning.failure_stage, FrameFailureStage.READ)
        self.assertIsNone(selector.last_processed_hash())

    def test_reset_clears_last_processed_hash(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            AtomicFrameHandoffWriter(config).publish_frame(
                b"same",
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(LatestFrameHandoffReader(config))
            result = selector.select_latest()
            self.assertIsNotNone(result.selected)
            assert result.selected is not None
            selector.mark_processed(result.selected.frame_hash)
            self.assertEqual(selector.last_processed_hash(), result.selected.frame_hash)

            selector.reset()

            self.assertIsNone(selector.last_processed_hash())

    def test_duplicate_hash_skip_disabled_always_selects_even_same_hash(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            AtomicFrameHandoffWriter(config).publish_frame(
                b"same",
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(
                LatestFrameHandoffReader(config),
                duplicate_hash_skip_enabled=False,
            )
            first = selector.select_latest()
            self.assertIsNotNone(first.selected)
            assert first.selected is not None
            selector.mark_processed(first.selected.frame_hash)

            second = selector.select_latest()

            self.assertIsNotNone(second.selected)
            self.assertIsNone(second.skipped)

    def test_selected_request_inherits_debug_artifact_settings(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            debug_dir = Path(tmp_dir) / "debug"
            AtomicFrameHandoffWriter(config).publish_frame(
                b"frame-bytes",
                FrameMetadata(frame_id="frame-1"),
            )
            selector = self._selector(
                LatestFrameHandoffReader(config),
                save_debug_images=True,
                debug_output_dir=debug_dir,
            )

            result = selector.select_latest()

            self.assertIsNotNone(result.selected)
            assert result.selected is not None
            self.assertTrue(result.selected.request.save_debug_images)
            self.assertEqual(result.selected.request.debug_output_dir, debug_dir)

    def test_frame_selection_module_keeps_heavy_runtime_imports_out(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "frame_selection.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        banned_roots = {"PySide6", "cv2", "numpy", "torch"}
        found: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".", maxsplit=1)[0])

        self.assertEqual(found & banned_roots, set())


class FailingReader:
    def latest_completed_frame(self) -> FrameReference | None:
        return FrameReference(image_path=Path("live_frames/latest_frame.png"))

    def read_frame_bytes(self, frame: FrameReference) -> bytes:
        raise OSError("simulated read failure")


if __name__ == "__main__":
    unittest.main()
