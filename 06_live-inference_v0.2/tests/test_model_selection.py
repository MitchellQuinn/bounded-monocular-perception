"""Tests for live inference model selection config parsing."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import textwrap
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live_inference.model_registry import (  # noqa: E402
    ModelSelectionError,
    load_model_selection,
)


class SelectionFixture:
    def __init__(self, root: Path) -> None:
        self.project_root = root / "06_live-inference_v0.2"
        self.models_root = self.project_root / "models"
        self.selection_path = self.models_root / "selections" / "current.toml"
        self.distance_root = (
            self.models_root / "distance-orientation" / "260504-1100_ts-2d-cnn"
        )
        self.roi_root = self.models_root / "roi-fcn" / "roi-run-0001"
        self.distance_root.mkdir(parents=True)
        self.roi_root.mkdir(parents=True)
        self.selection_path.parent.mkdir(parents=True)

    def write_selection(
        self,
        *,
        distance_root: str = "../distance-orientation/260504-1100_ts-2d-cnn",
        roi_root: str = "../roi-fcn/roi-run-0001",
        device_block: str = textwrap.dedent(
            """\
            [device]
            distance_orientation = "cuda"
            roi_fcn = "cuda"
            """
        ),
    ) -> None:
        self.selection_path.write_text(
            textwrap.dedent(
                f"""\
                [distance_orientation]
                root = "{distance_root}"

                [roi_fcn]
                root = "{roi_root}"

                {device_block}
                """
            ),
            encoding="utf-8",
        )


class ModelSelectionTests(unittest.TestCase):
    def test_loads_valid_selection_toml(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            fixture.write_selection()

            selection = load_model_selection(fixture.selection_path)

        self.assertEqual(selection.selection_path, fixture.selection_path.resolve())
        self.assertEqual(selection.distance_orientation_root, fixture.distance_root.resolve())
        self.assertEqual(selection.roi_fcn_root, fixture.roi_root.resolve())
        self.assertEqual(selection.distance_orientation_device, "cuda")
        self.assertEqual(selection.roi_fcn_device, "cuda")

    def test_accepts_auto_and_cpu_device_policies(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            fixture.write_selection(
                device_block=textwrap.dedent(
                    """\
                    [device]
                    distance_orientation = "auto"
                    roi_fcn = "cpu"
                    """
                )
            )

            selection = load_model_selection(fixture.selection_path)

        self.assertEqual(selection.distance_orientation_device, "auto")
        self.assertEqual(selection.roi_fcn_device, "cpu")

    def test_resolves_paths_relative_to_selection_file(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            fixture.selection_path = fixture.models_root / "selections" / "nested" / "active.toml"
            fixture.selection_path.parent.mkdir(parents=True)
            fixture.write_selection(
                distance_root="../../distance-orientation/260504-1100_ts-2d-cnn",
                roi_root="../../roi-fcn/roi-run-0001",
            )

            selection = load_model_selection(fixture.selection_path)

        self.assertEqual(selection.distance_orientation_root, fixture.distance_root.resolve())
        self.assertEqual(selection.roi_fcn_root, fixture.roi_root.resolve())

    def test_rejects_absolute_distance_orientation_root(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            fixture.write_selection(distance_root=str(fixture.distance_root.resolve()))

            with self.assertRaises(ModelSelectionError):
                load_model_selection(fixture.selection_path)

    def test_rejects_absolute_roi_fcn_root(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            fixture.write_selection(roi_root=str(fixture.roi_root.resolve()))

            with self.assertRaises(ModelSelectionError):
                load_model_selection(fixture.selection_path)

    def test_rejects_paths_resolving_outside_models(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            outside_root = fixture.project_root / "outside-model"
            outside_root.mkdir()
            fixture.write_selection(distance_root="../../outside-model")

            with self.assertRaises(ModelSelectionError):
                load_model_selection(fixture.selection_path)

    def test_rejects_missing_distance_orientation_root(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            fixture.write_selection(distance_root="../distance-orientation/missing-run")

            with self.assertRaises(ModelSelectionError):
                load_model_selection(fixture.selection_path)

    def test_rejects_missing_roi_fcn_root(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            fixture.write_selection(roi_root="../roi-fcn/missing-run")

            with self.assertRaises(ModelSelectionError):
                load_model_selection(fixture.selection_path)

    def test_default_devices_are_auto_if_omitted(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            fixture.write_selection(device_block="")

            selection = load_model_selection(fixture.selection_path)

        self.assertEqual(selection.distance_orientation_device, "auto")
        self.assertEqual(selection.roi_fcn_device, "auto")

    def test_rejects_invalid_device_policy(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fixture = SelectionFixture(Path(tmp_dir))
            fixture.write_selection(
                device_block=textwrap.dedent(
                    """\
                    [device]
                    distance_orientation = "cuda:0"
                    roi_fcn = "cpu"
                    """
                )
            )

            with self.assertRaisesRegex(ModelSelectionError, "distance_orientation"):
                load_model_selection(fixture.selection_path)

    def test_model_selection_module_keeps_heavy_runtime_imports_out(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "model_registry" / "model_selection.py"

        self.assertEqual(_banned_imports(module_path), set())


def _banned_imports(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    banned_roots = {"PySide6", "cv2", "numpy", "torch"}
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".", maxsplit=1)[0])
    return found & banned_roots


if __name__ == "__main__":
    unittest.main()
