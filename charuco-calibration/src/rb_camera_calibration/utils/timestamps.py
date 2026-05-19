"""Timestamp helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with second-level readability."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_run_slug(now: datetime | None = None) -> str:
    """Return a compact run-directory timestamp such as ``260518-1430``."""
    current = now.astimezone(UTC) if now is not None else datetime.now(UTC)
    return current.strftime("%y%m%d-%H%M")
