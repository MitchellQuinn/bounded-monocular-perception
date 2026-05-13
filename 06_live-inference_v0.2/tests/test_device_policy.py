"""Tests for live inference Torch device policy resolution."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live_inference.runtime.device import (  # noqa: E402
    normalize_torch_device_policy,
    resolve_torch_device,
)


class DevicePolicyTests(unittest.TestCase):
    def test_cpu_resolves_without_torch(self) -> None:
        with patch(
            "live_inference.runtime.device.importlib.import_module",
            side_effect=AssertionError("cpu policy must not import torch"),
        ):
            self.assertEqual(resolve_torch_device("cpu"), "cpu")

    def test_auto_resolves_to_cpu_when_cuda_unavailable(self) -> None:
        with patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=False)}):
            self.assertEqual(resolve_torch_device("auto"), "cpu")

    def test_auto_resolves_to_cuda_when_cuda_available(self) -> None:
        with patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=True)}):
            self.assertEqual(resolve_torch_device("auto"), "cuda")

    def test_cuda_raises_clearly_when_unavailable(self) -> None:
        with patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=False)}):
            with self.assertRaisesRegex(RuntimeError, "CUDA was requested.*is_available"):
                resolve_torch_device("cuda")

    def test_invalid_device_policy_rejects_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported Torch device policy"):
            normalize_torch_device_policy("cuda:0")

    def test_device_resolution_does_not_import_pyside6(self) -> None:
        before = set(sys.modules)
        with patch.dict(sys.modules, {"torch": _fake_torch(cuda_available=False)}):
            resolve_torch_device("auto")
        imported = set(sys.modules) - before

        self.assertFalse(
            any(name == "PySide6" or name.startswith("PySide6.") for name in imported),
            imported,
        )

    def test_generic_core_modules_remain_torch_free(self) -> None:
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


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _fake_torch(*, cuda_available: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(cuda=_FakeCuda(cuda_available))


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
