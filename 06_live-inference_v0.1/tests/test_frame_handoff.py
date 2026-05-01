"""Unit tests for atomic latest-frame handoff services."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from interfaces import (  # noqa: E402
    DEFAULT_FRAME_HASH_ALGORITHM,
    DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES,
    FrameHandoffReader,
    FrameHandoffWriter,
    FrameMetadata,
    LiveInferenceConfig,
)
import live_inference.frame_handoff as frame_handoff  # noqa: E402
from live_inference.frame_handoff import (  # noqa: E402
    AtomicFrameHandoffWriter,
    LatestFrameHandoffReader,
    compute_frame_hash,
)


class FrameHandoffServiceTests(unittest.TestCase):
    def _config(self, tmp_dir: str) -> LiveInferenceConfig:
        return LiveInferenceConfig(frame_dir=Path(tmp_dir) / "live_frames")

    def test_publish_frame_creates_dir_writes_latest_and_returns_reference(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            writer = AtomicFrameHandoffWriter(config)
            image_bytes = b"frame-bytes"

            self.assertFalse(config.frame_dir.exists())
            frame = writer.publish_frame(
                image_bytes,
                FrameMetadata(frame_id="frame-1", camera_index=2),
            )

            self.assertIsInstance(writer, FrameHandoffWriter)
            self.assertTrue(config.frame_dir.is_dir())
            self.assertTrue(config.handoff_paths.latest_frame_path.is_file())
            self.assertFalse(config.handoff_paths.temp_frame_path.exists())
            self.assertEqual(config.handoff_paths.latest_frame_path.read_bytes(), image_bytes)
            self.assertEqual(frame.image_path, config.handoff_paths.latest_frame_path)
            self.assertEqual(frame.byte_size, len(image_bytes))
            self.assertEqual(frame.metadata.frame_id, "frame-1")
            self.assertEqual(frame.metadata.camera_index, 2)
            self.assertEqual(frame.metadata.byte_size, len(image_bytes))
            self.assertIsNotNone(frame.completed_at_utc)
            self.assertEqual(frame.handoff_paths, config.handoff_paths)
            self.assertIsNotNone(frame.frame_hash)

    def test_publish_frame_replaces_existing_latest_without_deleting_first(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            writer = AtomicFrameHandoffWriter(config)
            old_bytes = b"old-frame"
            new_bytes = b"new-frame"
            original_replace = frame_handoff.os.replace

            writer.publish_frame(old_bytes, FrameMetadata(frame_id="old"))

            def assert_latest_exists_then_replace(src: Path, dst: Path) -> None:
                self.assertEqual(Path(dst), config.handoff_paths.latest_frame_path)
                self.assertTrue(Path(dst).exists())
                self.assertEqual(Path(dst).read_bytes(), old_bytes)
                original_replace(src, dst)

            with patch.object(
                frame_handoff.os,
                "replace",
                side_effect=assert_latest_exists_then_replace,
            ):
                writer.publish_frame(new_bytes, FrameMetadata(frame_id="new"))

            self.assertEqual(config.handoff_paths.latest_frame_path.read_bytes(), new_bytes)
            self.assertFalse(config.handoff_paths.temp_frame_path.exists())

    def test_latest_completed_frame_returns_none_without_latest(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            reader = LatestFrameHandoffReader(config)

            self.assertIsInstance(reader, FrameHandoffReader)
            self.assertIsNone(reader.latest_completed_frame())

    def test_latest_completed_frame_returns_latest_never_temp(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            config.frame_dir.mkdir(parents=True)
            config.handoff_paths.temp_frame_path.write_bytes(b"temp-only")
            reader = LatestFrameHandoffReader(config)

            self.assertIsNone(reader.latest_completed_frame())

            config.handoff_paths.latest_frame_path.write_bytes(b"latest")
            frame = reader.latest_completed_frame()

            self.assertIsNotNone(frame)
            assert frame is not None
            self.assertEqual(frame.image_path, config.handoff_paths.latest_frame_path)
            self.assertNotEqual(frame.image_path, config.handoff_paths.temp_frame_path)
            self.assertEqual(frame.byte_size, len(b"latest"))

    def test_read_frame_bytes_returns_exact_published_bytes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            writer = AtomicFrameHandoffWriter(config)
            reader = LatestFrameHandoffReader(config)
            image_bytes = b"\x89PNG\r\nexact bytes"

            published = writer.publish_frame(image_bytes, FrameMetadata(frame_id="frame-1"))
            latest = reader.latest_completed_frame()

            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(reader.read_frame_bytes(published), image_bytes)
            self.assertEqual(reader.read_frame_bytes(latest), image_bytes)

    def test_compute_frame_hash_is_stable_and_sensitive_to_bytes(self) -> None:
        first = compute_frame_hash(b"same")
        second = compute_frame_hash(b"same")
        different = compute_frame_hash(b"different")

        self.assertEqual(first.value, second.value)
        self.assertNotEqual(first.value, different.value)
        self.assertEqual(first.algorithm, DEFAULT_FRAME_HASH_ALGORITHM)
        self.assertEqual(first.digest_size_bytes, DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES)

    def test_compute_frame_hash_respects_configured_digest_size(self) -> None:
        frame_hash = compute_frame_hash(b"frame", digest_size_bytes=8)

        self.assertEqual(frame_hash.digest_size_bytes, 8)
        self.assertEqual(len(frame_hash.value), 16)

    def test_replace_failure_leaves_previous_latest_intact_and_surfaces_exception(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = self._config(tmp_dir)
            writer = AtomicFrameHandoffWriter(config)
            old_bytes = b"old-frame"
            new_bytes = b"new-frame"

            writer.publish_frame(old_bytes, FrameMetadata(frame_id="old"))

            with patch.object(
                frame_handoff.os,
                "replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated replace failure"):
                    writer.publish_frame(new_bytes, FrameMetadata(frame_id="new"))

            self.assertEqual(config.handoff_paths.latest_frame_path.read_bytes(), old_bytes)
            self.assertEqual(config.handoff_paths.temp_frame_path.read_bytes(), new_bytes)

    def test_implementation_imports_are_stdlib_or_local_contracts_only(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "frame_handoff.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        allowed_roots = {
            "__future__",
            "dataclasses",
            "datetime",
            "hashlib",
            "os",
            "pathlib",
            "interfaces",
        }
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

        self.assertLessEqual(imported_roots, allowed_roots)


if __name__ == "__main__":
    unittest.main()
