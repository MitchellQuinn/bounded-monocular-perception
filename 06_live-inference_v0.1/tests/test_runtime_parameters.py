"""Unit tests for runtime parameter state management."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from interfaces import (  # noqa: E402
    RuntimeParameterSetSpec,
    RuntimeParameterSpec,
    RuntimeParameterUpdate,
    RuntimeParameterValueType,
    RuntimeParameterWidgetHint,
    WorkerName,
)
from live_inference.runtime_parameters import RuntimeParameterStateManager  # noqa: E402


REQUESTED_AT = "2026-05-03T10:00:00Z"


class RuntimeParameterStateManagerTests(unittest.TestCase):
    def _manager(self) -> RuntimeParameterStateManager:
        return RuntimeParameterStateManager(_parameter_set_spec())

    def _update(
        self,
        updates: dict[str, object],
        *,
        owner: WorkerName = WorkerName.INFERENCE,
        namespace: str = "preprocessing",
        base_revision: int | None = None,
    ) -> RuntimeParameterUpdate:
        return RuntimeParameterUpdate(
            owner=owner,
            namespace=namespace,
            updates=updates,
            requested_at_utc=REQUESTED_AT,
            base_revision=base_revision,
        )

    def test_current_values_returns_current_values(self) -> None:
        manager = self._manager()

        self.assertEqual(
            manager.current_values(),
            {
                "enabled": False,
                "retries": 2,
                "threshold": 0.5,
                "label": "default",
                "mode": "fast",
                "read_only_note": "fixed",
            },
        )

    def test_valid_bool_update_accepted(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"enabled": True}))

        self.assertTrue(result.accepted)
        self.assertEqual(result.applied_updates, {"enabled": True})
        self.assertEqual(manager.current_values()["enabled"], True)

    def test_valid_int_update_accepted(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"retries": 3}))

        self.assertTrue(result.accepted)
        self.assertEqual(result.applied_updates, {"retries": 3})
        self.assertEqual(manager.current_values()["retries"], 3)

    def test_valid_float_update_accepted(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"threshold": 0.75}))

        self.assertTrue(result.accepted)
        self.assertEqual(result.applied_updates, {"threshold": 0.75})
        self.assertEqual(manager.current_values()["threshold"], 0.75)

    def test_valid_string_update_accepted(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"label": "tuned"}))

        self.assertTrue(result.accepted)
        self.assertEqual(result.applied_updates, {"label": "tuned"})
        self.assertEqual(manager.current_values()["label"], "tuned")

    def test_valid_enum_update_accepted(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"mode": "accurate"}))

        self.assertTrue(result.accepted)
        self.assertEqual(result.applied_updates, {"mode": "accurate"})
        self.assertEqual(manager.current_values()["mode"], "accurate")

    def test_unknown_parameter_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"missing": 1}))

        self.assertFalse(result.accepted)
        self.assertEqual(result.applied_updates, {})
        self.assertIn("missing", result.rejected_updates)
        self.assertIn("Unknown", result.rejected_updates["missing"])

    def test_read_only_parameter_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"read_only_note": "changed"}))

        self.assertFalse(result.accepted)
        self.assertEqual(result.applied_updates, {})
        self.assertIn("read-only", result.rejected_updates["read_only_note"])

    def test_wrong_owner_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(
            self._update({"enabled": True}, owner=WorkerName.CAMERA)
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.applied_updates, {})
        self.assertIn("owner", result.rejected_updates["enabled"])

    def test_wrong_namespace_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(
            self._update({"enabled": True}, namespace="camera")
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.applied_updates, {})
        self.assertIn("namespace", result.rejected_updates["enabled"])

    def test_stale_base_revision_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(
            self._update({"enabled": True}, base_revision=manager.current_revision() - 1)
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.applied_updates, {})
        self.assertIn("Stale", result.rejected_updates["enabled"])

    def test_bool_is_not_accepted_for_int(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"retries": True}))

        self.assertFalse(result.accepted)
        self.assertEqual(result.applied_updates, {})
        self.assertIn("int", result.rejected_updates["retries"])

    def test_bool_is_not_accepted_for_float(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"threshold": True}))

        self.assertFalse(result.accepted)
        self.assertEqual(result.applied_updates, {})
        self.assertIn("float", result.rejected_updates["threshold"])

    def test_int_can_be_accepted_for_float_and_converted(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"threshold": 1}))

        self.assertTrue(result.accepted)
        self.assertEqual(result.applied_updates, {"threshold": 1.0})
        self.assertEqual(manager.current_values()["threshold"], 1.0)

    def test_int_below_minimum_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"retries": -1}))

        self.assertFalse(result.accepted)
        self.assertIn("below minimum", result.rejected_updates["retries"])

    def test_int_above_maximum_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"retries": 11}))

        self.assertFalse(result.accepted)
        self.assertIn("above maximum", result.rejected_updates["retries"])

    def test_float_below_minimum_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"threshold": -0.1}))

        self.assertFalse(result.accepted)
        self.assertIn("below minimum", result.rejected_updates["threshold"])

    def test_float_above_maximum_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"threshold": 1.1}))

        self.assertFalse(result.accepted)
        self.assertIn("above maximum", result.rejected_updates["threshold"])

    def test_enum_value_outside_choices_rejected(self) -> None:
        manager = self._manager()

        result = manager.apply_update(self._update({"mode": "balanced"}))

        self.assertFalse(result.accepted)
        self.assertIn("Expected one of", result.rejected_updates["mode"])

    def test_all_or_nothing_rejects_everything_if_one_update_is_invalid(self) -> None:
        manager = self._manager()
        original_values = manager.current_values()

        result = manager.apply_update(
            self._update({"enabled": True, "threshold": 1.5})
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.applied_updates, {})
        self.assertIn("threshold", result.rejected_updates)
        self.assertEqual(manager.current_values(), original_values)

    def test_revision_increments_on_actual_accepted_change(self) -> None:
        manager = self._manager()
        original_revision = manager.current_revision()

        result = manager.apply_update(self._update({"enabled": True}))

        self.assertTrue(result.accepted)
        self.assertEqual(result.revision, original_revision + 1)
        self.assertEqual(manager.current_revision(), original_revision + 1)
        self.assertEqual(manager.current_spec().revision, original_revision + 1)

    def test_revision_does_not_increment_when_value_is_unchanged(self) -> None:
        manager = self._manager()
        original_revision = manager.current_revision()

        result = manager.apply_update(self._update({"enabled": False}))

        self.assertTrue(result.accepted)
        self.assertEqual(result.revision, original_revision)
        self.assertEqual(manager.current_revision(), original_revision)
        self.assertEqual(result.applied_updates, {"enabled": False})

    def test_update_result_contains_applied_and_rejected_updates(self) -> None:
        manager = self._manager()

        accepted = manager.apply_update(self._update({"threshold": 1}))
        rejected = manager.apply_update(self._update({"threshold": 2.0}))

        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.applied_updates, {"threshold": 1.0})
        self.assertEqual(accepted.rejected_updates, {})
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.applied_updates, {})
        self.assertEqual(
            rejected.rejected_updates,
            {"threshold": "Value 2.0 is above maximum 1.0."},
        )

    def test_runtime_parameters_module_keeps_heavy_runtime_imports_out(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "runtime_parameters.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        banned_roots = {"PySide6", "cv2", "numpy", "torch"}
        found: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                found.add(node.module.split(".", maxsplit=1)[0])

        self.assertEqual(found & banned_roots, set())


def _parameter_set_spec() -> RuntimeParameterSetSpec:
    return RuntimeParameterSetSpec(
        owner=WorkerName.INFERENCE,
        namespace="preprocessing",
        revision=3,
        parameters=(
            RuntimeParameterSpec(
                name="enabled",
                label="Enabled",
                value_type=RuntimeParameterValueType.BOOL,
                default_value=False,
                current_value=False,
                widget_hint=RuntimeParameterWidgetHint.CHECKBOX,
            ),
            RuntimeParameterSpec(
                name="retries",
                label="Retries",
                value_type=RuntimeParameterValueType.INT,
                default_value=1,
                current_value=2,
                widget_hint=RuntimeParameterWidgetHint.INT_INPUT,
                minimum=0,
                maximum=10,
            ),
            RuntimeParameterSpec(
                name="threshold",
                label="Threshold",
                value_type=RuntimeParameterValueType.FLOAT,
                default_value=0.25,
                current_value=0.5,
                widget_hint=RuntimeParameterWidgetHint.SLIDER,
                minimum=0.0,
                maximum=1.0,
                step=0.1,
            ),
            RuntimeParameterSpec(
                name="label",
                label="Label",
                value_type=RuntimeParameterValueType.STRING,
                default_value="default",
                current_value="default",
                widget_hint=RuntimeParameterWidgetHint.TEXT_INPUT,
                choices=("default", "tuned"),
            ),
            RuntimeParameterSpec(
                name="mode",
                label="Mode",
                value_type=RuntimeParameterValueType.ENUM,
                default_value="fast",
                current_value="fast",
                widget_hint=RuntimeParameterWidgetHint.DROPDOWN,
                choices=("fast", "accurate"),
            ),
            RuntimeParameterSpec(
                name="read_only_note",
                label="Read-only note",
                value_type=RuntimeParameterValueType.STRING,
                default_value="fixed",
                current_value="fixed",
                widget_hint=RuntimeParameterWidgetHint.TEXT_INPUT,
                read_only=True,
            ),
        ),
        timestamp_utc="2026-05-03T09:00:00Z",
    )


if __name__ == "__main__":
    unittest.main()
