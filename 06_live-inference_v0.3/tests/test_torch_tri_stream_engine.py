from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live_inference.engines import torch_tri_stream_engine as engine_module  # noqa: E402
from live_inference.engines import TorchTriStreamInferenceEngine  # noqa: E402
from live_inference.model_registry import ModelSelectionError  # noqa: E402


class TorchTriStreamInferenceEngineTests(unittest.TestCase):
    def test_default_model_selection_file_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            with mock.patch.object(
                engine_module,
                "_live_project_root",
                return_value=project_root,
            ):
                with self.assertRaisesRegex(
                    ModelSelectionError,
                    "Selection file does not exist",
                ):
                    TorchTriStreamInferenceEngine(load_model=False)


if __name__ == "__main__":
    unittest.main()
