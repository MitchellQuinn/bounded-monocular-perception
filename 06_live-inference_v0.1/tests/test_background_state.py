"""Tests for thread-safe static background removal state."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live_inference.masking import BackgroundState  # noqa: E402


class BackgroundStateTests(unittest.TestCase):
    def test_initial_state_not_captured(self) -> None:
        state = BackgroundState()

        snapshot = state.get_snapshot()

        self.assertFalse(snapshot.captured)
        self.assertFalse(snapshot.enabled)
        self.assertEqual(snapshot.threshold, 25)
        self.assertEqual(snapshot.grayscale_background.shape, (0, 0))

    def test_capture_background_stores_grayscale_copy_and_increments_revision(self) -> None:
        state = BackgroundState()
        image = np.arange(12, dtype=np.uint8).reshape(3, 4)

        revision = state.capture_background(image)
        image[0, 0] = 255
        snapshot = state.get_snapshot()

        self.assertEqual(revision, 1)
        self.assertTrue(snapshot.captured)
        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(snapshot.width_px, 4)
        self.assertEqual(snapshot.height_px, 3)
        self.assertEqual(int(snapshot.grayscale_background[0, 0]), 0)
        self.assertIsNotNone(snapshot.captured_at_utc)

    def test_clear_resets_captured_state_and_increments_revision(self) -> None:
        state = BackgroundState()
        state.capture_background(np.zeros((2, 3), dtype=np.uint8))
        state.set_enabled(True)

        revision = state.clear()
        snapshot = state.get_snapshot()

        self.assertEqual(revision, 3)
        self.assertFalse(snapshot.captured)
        self.assertFalse(snapshot.enabled)
        self.assertEqual(snapshot.revision, 3)

    def test_enabled_flag_and_threshold_are_stored(self) -> None:
        state = BackgroundState()

        state.set_enabled(True)
        state.set_threshold(42)
        snapshot = state.get_snapshot()

        self.assertTrue(snapshot.enabled)
        self.assertEqual(snapshot.threshold, 42)

    def test_snapshot_is_copy_safe(self) -> None:
        state = BackgroundState()
        state.capture_background(np.zeros((2, 2), dtype=np.uint8))

        snapshot = state.get_snapshot()

        self.assertFalse(snapshot.grayscale_background.flags.writeable)
        with self.assertRaises(ValueError):
            snapshot.grayscale_background[0, 0] = 1
        with self.assertRaises(ValueError):
            snapshot.grayscale_background.setflags(write=True)
        self.assertEqual(int(state.get_snapshot().grayscale_background[0, 0]), 0)

    def test_repeated_snapshots_reuse_readonly_background_storage(self) -> None:
        state = BackgroundState()
        state.capture_background(np.zeros((2, 2), dtype=np.uint8))

        first = state.get_snapshot()
        second = state.get_snapshot()

        self.assertIs(first, second)
        self.assertTrue(
            np.shares_memory(first.grayscale_background, second.grayscale_background)
        )

    def test_lock_allows_concurrent_read_write(self) -> None:
        state = BackgroundState()

        def update(index: int) -> int:
            state.capture_background(np.full((2, 2), index, dtype=np.uint8))
            state.set_threshold(index % 256)
            return state.get_snapshot().revision

        with ThreadPoolExecutor(max_workers=4) as executor:
            revisions = list(executor.map(update, range(16)))

        self.assertEqual(len(revisions), 16)
        self.assertTrue(state.get_snapshot().captured)
        self.assertGreaterEqual(state.get_snapshot().revision, 16)


if __name__ == "__main__":
    unittest.main()
