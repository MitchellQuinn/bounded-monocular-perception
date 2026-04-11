"""Small reusable utility helpers for v4 pipeline stages."""

from __future__ import annotations

import hashlib
from pathlib import Path


def selected_row_indices(total_rows: int, *, offset: int, limit: int) -> set[int]:
    rows = list(range(max(0, int(total_rows))))
    if not rows:
        return set()

    start = max(0, int(offset))
    if start >= len(rows):
        return set()

    if int(limit) <= 0:
        return set(rows[start:])

    end = start + int(limit)
    return set(rows[start:end])


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
