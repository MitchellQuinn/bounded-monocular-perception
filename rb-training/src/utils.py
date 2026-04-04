"""Generic utility helpers used across training and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Favor reproducibility for this first-pass falsification test.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write pretty JSON with deterministic key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON document."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_metadata(repo_root: Path) -> dict[str, Any]:
    """Best-effort git metadata. Returns None fields when unavailable."""
    branch = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_git(repo_root, ["rev-parse", "HEAD"])
    dirty = _run_git(repo_root, ["status", "--porcelain"])
    return {
        "git_branch": branch,
        "git_commit": commit,
        "dirty_worktree": bool(dirty) if dirty is not None else None,
    }


def environment_summary(device: str) -> dict[str, Any]:
    """Capture environment versions required by the repo standard."""
    return {
        "python_version": sys.version.split(" ", maxsplit=1)[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "matplotlib_version": matplotlib.__version__,
        "cuda_version": torch.version.cuda,
        "device": device,
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
    }


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    """Render a compact markdown table."""
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join([head, sep, *body])
