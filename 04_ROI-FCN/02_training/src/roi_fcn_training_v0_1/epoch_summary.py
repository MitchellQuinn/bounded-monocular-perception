"""Structured per-epoch summary helpers for ROI-FCN notebook controls."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import HISTORY_FILENAME, RESUME_STATE_FILENAME
from .resume_state import load_resume_state


@dataclass(frozen=True)
class EpochSummaryPanel:
    """Display-ready latest/best epoch summary."""

    criterion_metric: str
    criterion_direction: str
    latest_epoch: int | None
    best_epoch: int | None
    latest_record: dict[str, Any] | None
    best_record: dict[str, Any] | None
    text: str


def best_epoch_metric_name() -> str:
    """Return the primary checkpoint-selection metric used during ROI-FCN training."""
    return "validation_mean_center_error_px"


def select_best_history_record(history_records: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Select the best epoch record using the ROI-FCN checkpointing rule."""
    best_record: dict[str, Any] | None = None
    for raw_record in history_records:
        record = _normalize_record(raw_record)
        if _is_better_record(record, best_record):
            best_record = record
    return best_record


def read_history_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load structured epoch records for a run from history.json or resume_state.pt."""
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    history_path = resolved_run_dir / HISTORY_FILENAME
    if history_path.exists():
        try:
            with history_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, list):
            records = [_normalize_record(record) for record in payload if isinstance(record, Mapping)]
            if records:
                return records

    resume_state_path = resolved_run_dir / RESUME_STATE_FILENAME
    if resume_state_path.exists():
        resume_state = load_resume_state(resume_state_path, map_location="cpu")
        raw_history = resume_state.get("history_rows")
        if isinstance(raw_history, list):
            return [_normalize_record(record) for record in raw_history if isinstance(record, Mapping)]

    return []


def read_epoch_summary_panel(run_dir: str | Path | None) -> EpochSummaryPanel:
    """Build a compact latest/best epoch summary for one run directory."""
    history_records = read_history_records(run_dir) if run_dir is not None else []
    latest_record = dict(history_records[-1]) if history_records else None
    best_record = select_best_history_record(history_records)
    criterion_metric = best_epoch_metric_name()
    text = _format_panel_text(
        latest_record=latest_record,
        best_record=best_record,
        criterion_metric=criterion_metric,
    )
    return EpochSummaryPanel(
        criterion_metric=criterion_metric,
        criterion_direction="min",
        latest_epoch=_epoch_int(latest_record),
        best_epoch=_epoch_int(best_record),
        latest_record=latest_record,
        best_record=best_record,
        text=text,
    )


def format_run_epoch_summary_panel(run_dir: str | Path | None) -> str:
    """Return display-ready latest/best epoch summary text for a run."""
    return read_epoch_summary_panel(run_dir).text


def _normalize_record(raw_record: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in raw_record.items():
        normalized[str(key)] = _normalize_value(value)
    return normalized


def _normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if hasattr(value, "item"):
        try:
            return _normalize_value(value.item())
        except (TypeError, ValueError):
            return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _epoch_int(record: Mapping[str, Any] | None) -> int | None:
    if not isinstance(record, Mapping):
        return None
    value = record.get("epoch")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_better_record(candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None) -> bool:
    candidate_center_error = _as_finite_float(candidate.get(best_epoch_metric_name()))
    if candidate_center_error is None:
        return False
    if not isinstance(incumbent, Mapping):
        return True

    incumbent_center_error = _as_finite_float(incumbent.get(best_epoch_metric_name()))
    if incumbent_center_error is None:
        return True
    if candidate_center_error < incumbent_center_error - 1e-6:
        return True
    if math.isclose(candidate_center_error, incumbent_center_error, abs_tol=1e-6):
        candidate_loss = _as_finite_float(candidate.get("validation_loss"))
        incumbent_loss = _as_finite_float(incumbent.get("validation_loss"))
        if candidate_loss is None:
            return False
        if incumbent_loss is None:
            return True
        return candidate_loss < incumbent_loss - 1e-9
    return False


def _format_panel_text(
    *,
    latest_record: Mapping[str, Any] | None,
    best_record: Mapping[str, Any] | None,
    criterion_metric: str,
) -> str:
    parts = [f"Best selected by: {criterion_metric} (lower is better; validation_loss breaks ties)", ""]
    parts.extend(_format_section("Latest completed", latest_record))
    parts.append("")
    parts.extend(_format_section("Best so far", best_record))
    return "\n".join(parts)


def _format_section(title: str, record: Mapping[str, Any] | None) -> list[str]:
    lines = [title]
    if not isinstance(record, Mapping):
        lines.append("[no completed epochs yet]")
        return lines

    header_parts: list[str] = []
    epoch_value = _epoch_int(record)
    if epoch_value is not None:
        header_parts.append(f"epoch={epoch_value}")
    for key in ("train_loss", "validation_loss", "validation_mean_center_error_px"):
        formatted = _format_key_value(key, record.get(key))
        if formatted:
            header_parts.append(formatted)
    if header_parts:
        lines.append(" | ".join(header_parts))

    detail_parts: list[str] = []
    for key in (
        "train_mean_center_error_px",
        "validation_p95_center_error_px",
        "train_roi_full_containment_success_rate",
        "validation_roi_full_containment_success_rate",
    ):
        formatted = _format_key_value(key, record.get(key))
        if formatted:
            detail_parts.append(formatted)
    lines.extend(_chunked_metric_lines(detail_parts, chunk_size=2))
    return lines


def _chunked_metric_lines(parts: Sequence[str], *, chunk_size: int) -> list[str]:
    if not parts:
        return []
    return [
        " | ".join(parts[start : start + chunk_size])
        for start in range(0, len(parts), chunk_size)
    ]


def _format_key_value(label: str, value: Any) -> str:
    numeric = _as_finite_float(value)
    if numeric is None:
        return ""
    return f"{label}={numeric:.4f}"
