"""Tests for named live diagnostic stage-policy profiles."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live_inference.preprocessing import (  # noqa: E402
    DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR,
    StageTransformPolicyState,
    diagnostic_profile_updates,
)


class DiagnosticProfileTests(unittest.TestCase):
    def test_baseline_profile_values_are_known_good_policy(self) -> None:
        values = diagnostic_profile_updates(
            DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR
        )

        self.assertEqual(values["roi_locator_input_mode"], "inverted")
        self.assertTrue(values["apply_manual_mask_to_roi_locator"])
        self.assertTrue(values["apply_manual_mask_to_regressor_preprocessing"])
        self.assertFalse(values["apply_background_removal_to_roi_locator"])
        self.assertFalse(
            values["apply_background_removal_to_regressor_preprocessing"]
        )

    def test_applying_baseline_profile_updates_effective_policy(self) -> None:
        state = StageTransformPolicyState()

        snapshot = state.apply_diagnostic_profile(
            DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR
        )

        self.assertEqual(snapshot.roi_locator_input_mode, "inverted")
        self.assertTrue(snapshot.apply_manual_mask_to_roi_locator)
        self.assertTrue(snapshot.apply_manual_mask_to_regressor_preprocessing)
        self.assertFalse(snapshot.apply_background_removal_to_roi_locator)
        self.assertFalse(
            snapshot.apply_background_removal_to_regressor_preprocessing
        )
        self.assertEqual(
            snapshot.diagnostic_profile_name,
            DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR,
        )
        self.assertEqual(
            state.get_snapshot().to_metadata()["diagnostic_profile_name"],
            DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR,
        )

    def test_manual_policy_change_clears_profile_name(self) -> None:
        state = StageTransformPolicyState()
        state.apply_diagnostic_profile(
            DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR
        )

        snapshot = state.update(roi_locator_input_mode="as_is")

        self.assertIsNone(snapshot.diagnostic_profile_name)
        self.assertEqual(snapshot.roi_locator_input_mode, "as_is")


if __name__ == "__main__":
    unittest.main()
