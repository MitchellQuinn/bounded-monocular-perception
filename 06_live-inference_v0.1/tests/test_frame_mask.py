"""Tests for thread-safe live frame mask state."""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402

from live_inference.masking import FrameMaskSnapshot, FrameMaskState  # noqa: E402


class FrameMaskStateTests(unittest.TestCase):
    def test_initial_revision_is_zero_and_snapshot_is_disabled(self) -> None:
        state = FrameMaskState()

        snapshot = state.get_snapshot()

        self.assertEqual(state.revision(), 0)
        self.assertEqual(snapshot.revision, 0)
        self.assertFalse(snapshot.enabled)
        self.assertEqual(snapshot.pixel_count, 0)

    def test_commit_increments_revision_and_stores_fill_value(self) -> None:
        state = FrameMaskState()
        mask = np.zeros((4, 5), dtype=bool)
        mask[1:3, 2:4] = True

        revision = state.commit_mask(mask, 5, 4, 0)
        snapshot = state.get_snapshot()

        self.assertEqual(revision, 1)
        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(snapshot.width_px, 5)
        self.assertEqual(snapshot.height_px, 4)
        self.assertEqual(snapshot.fill_value, 0)
        self.assertEqual(snapshot.pixel_count, 4)
        self.assertTrue(snapshot.dimensions_match(5, 4))
        self.assertFalse(snapshot.dimensions_match(4, 5))

    def test_clear_increments_revision(self) -> None:
        state = FrameMaskState()
        state.commit_mask(np.ones((2, 3), dtype=bool), 3, 2, 255)

        revision = state.clear()
        snapshot = state.get_snapshot()

        self.assertEqual(revision, 2)
        self.assertEqual(snapshot.revision, 2)
        self.assertFalse(snapshot.enabled)
        self.assertEqual(snapshot.pixel_count, 0)

    def test_snapshot_is_stable_and_copy_safe(self) -> None:
        state = FrameMaskState()
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        state.commit_mask(mask, 3, 3, 255)

        snapshot = state.get_snapshot()
        mask[1, 1] = False
        state.commit_mask(np.zeros((3, 3), dtype=bool), 3, 3, 255)

        self.assertTrue(snapshot.mask[1, 1])
        with self.assertRaises(ValueError):
            snapshot.mask[1, 1] = False

    def test_fill_value_can_be_changed_without_repainting_geometry(self) -> None:
        state = FrameMaskState()
        mask = np.zeros((3, 3), dtype=bool)
        mask[0, 0] = True
        state.commit_mask(mask, 3, 3, 255)

        revision = state.set_fill_value(0)
        snapshot = state.get_snapshot()

        self.assertEqual(revision, 2)
        self.assertEqual(snapshot.fill_value, 0)
        self.assertTrue(snapshot.mask[0, 0])

    def test_snapshot_rejects_shape_dimension_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            FrameMaskSnapshot(
                revision=1,
                width_px=5,
                height_px=4,
                mask=np.zeros((5, 4), dtype=bool),
                fill_value=255,
            )

    def test_thread_safe_snapshot_during_simple_mutation_and_read_usage(self) -> None:
        state = FrameMaskState()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for index in range(50):
                    mask = np.zeros((10, 12), dtype=bool)
                    mask[index % 10, index % 12] = True
                    state.commit_mask(mask, 12, 10, 255 if index % 2 else 0)
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(100):
                    snapshot = state.get_snapshot()
                    self.assertIn(snapshot.fill_value, {0, 255})
                    self.assertEqual(snapshot.mask.shape, (snapshot.height_px, snapshot.width_px))
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertGreaterEqual(state.revision(), 1)


if __name__ == "__main__":
    unittest.main()
