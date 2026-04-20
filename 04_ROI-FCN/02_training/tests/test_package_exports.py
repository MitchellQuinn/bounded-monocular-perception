from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


TRAINING_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = TRAINING_ROOT / "src"


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else f"{SRC_ROOT}{os.pathsep}{existing}"
    return env


class PackageExportTests(unittest.TestCase):
    def test_package_import_does_not_preload_train_or_evaluate_modules(self) -> None:
        probe = """
import json
import sys

import roi_fcn_training_v0_1

print(json.dumps({
    "train_loaded": "roi_fcn_training_v0_1.train" in sys.modules,
    "evaluate_loaded": "roi_fcn_training_v0_1.evaluate" in sys.modules,
    "train_proxy_callable": callable(roi_fcn_training_v0_1.train_roi_fcn),
    "evaluate_proxy_callable": callable(roi_fcn_training_v0_1.evaluate_saved_run),
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=TRAINING_ROOT,
            env=_pythonpath_env(),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        payload = json.loads(result.stdout.strip())
        self.assertFalse(payload["train_loaded"])
        self.assertFalse(payload["evaluate_loaded"])
        self.assertTrue(payload["train_proxy_callable"])
        self.assertTrue(payload["evaluate_proxy_callable"])

    def test_module_execution_help_does_not_emit_runpy_warning(self) -> None:
        result = subprocess.run(
            [sys.executable, "-W", "error::RuntimeWarning", "-m", "roi_fcn_training_v0_1.train", "--help"],
            cwd=TRAINING_ROOT,
            env=_pythonpath_env(),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertNotIn("RuntimeWarning", result.stderr)


if __name__ == "__main__":
    unittest.main()
