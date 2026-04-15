"""Structured per-epoch summary helpers for notebook control surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from .resume.state import RESUME_STATE_FILENAME, load_resume_state
from .topologies.contracts import reporting_train_losses, reporting_validation_metrics
from .utils import read_json

_ACCURACY_RE = re.compile(r"^(train|val)_acc_at_([0-9]+p[0-9]+)m$")
_PREFERRED_DISTANCE_TOLERANCES_M = (0.10, 0.25, 0.50)


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


def best_epoch_metric_name(task_contract: Mapping[str, Any] | None = None) -> str:
    """Return the checkpoint-selection metric used during training."""
    _ = task_contract
    return "val_loss"


def is_better_epoch_metric(
    candidate_value: Any,
    incumbent_value: Any,
    *,
    task_contract: Mapping[str, Any] | None = None,
) -> bool:
    """Return True when a candidate epoch should replace the incumbent best epoch."""
    _ = task_contract
    candidate = _as_finite_float(candidate_value)
    if candidate is None:
        return False
    incumbent = _as_finite_float(incumbent_value)
    if incumbent is None:
        return True
    return candidate < incumbent


def select_best_history_record(
    history_records: Sequence[Mapping[str, Any]],
    *,
    task_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Select the best epoch record using the shared checkpointing rule."""
    metric_name = best_epoch_metric_name(task_contract)
    best_record: dict[str, Any] | None = None
    best_value: float | None = None
    for raw_record in history_records:
        record = _normalize_record(raw_record)
        candidate_value = record.get(metric_name)
        if is_better_epoch_metric(
            candidate_value,
            best_value,
            task_contract=task_contract,
        ):
            best_record = record
            best_value = _as_finite_float(candidate_value)
    return best_record


def read_history_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load structured epoch records for a run from history.csv or resume_state.pt."""
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    history_path = resolved_run_dir / "history.csv"
    if history_path.exists():
        try:
            history_df = pd.read_csv(history_path)
        except pd.errors.EmptyDataError:
            history_df = pd.DataFrame()
        if not history_df.empty:
            return [_normalize_record(record) for record in history_df.to_dict(orient="records")]

    resume_state_path = resolved_run_dir / RESUME_STATE_FILENAME
    if resume_state_path.exists():
        resume_state = load_resume_state(resume_state_path, map_location="cpu")
        raw_history = resume_state.get("history_records")
        if isinstance(raw_history, list):
            return [_normalize_record(record) for record in raw_history if isinstance(record, Mapping)]

    return []


def read_epoch_summary_panel(run_dir: str | Path | None) -> EpochSummaryPanel:
    """Build a compact latest/best epoch summary for one run directory."""
    task_contract = _load_task_contract(run_dir)
    history_records = read_history_records(run_dir) if run_dir is not None else []
    latest_record = dict(history_records[-1]) if history_records else None
    best_record = select_best_history_record(history_records, task_contract=task_contract)
    criterion_metric = best_epoch_metric_name(task_contract)
    text = _format_panel_text(
        latest_record=latest_record,
        best_record=best_record,
        task_contract=task_contract,
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


def _load_task_contract(run_dir: str | Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    config_path = Path(run_dir).expanduser().resolve() / "config.json"
    if not config_path.exists():
        return {}
    payload = read_json(config_path)
    task_contract = payload.get("task_contract")
    return dict(task_contract) if isinstance(task_contract, Mapping) else {}


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
        if not text:
            return None
        return text
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
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


def _format_panel_text(
    *,
    latest_record: Mapping[str, Any] | None,
    best_record: Mapping[str, Any] | None,
    task_contract: Mapping[str, Any],
    criterion_metric: str,
) -> str:
    parts = [f"Best selected by: {criterion_metric} (lower is better)", ""]
    parts.extend(_format_section("Latest completed", latest_record, task_contract))
    parts.append("")
    parts.extend(_format_section("Best so far", best_record, task_contract))
    return "\n".join(parts)


def _format_section(
    title: str,
    record: Mapping[str, Any] | None,
    task_contract: Mapping[str, Any],
) -> list[str]:
    lines = [title]
    if not isinstance(record, Mapping):
        lines.append("[no completed epochs yet]")
        return lines

    header_parts: list[str] = []
    epoch_value = _epoch_int(record)
    if epoch_value is not None:
        header_parts.append(f"epoch={epoch_value}")
    for key in ("train_loss", "val_loss", "val_mae", "val_rmse"):
        if key in record:
            formatted = _format_key_value(key, record[key])
            if formatted:
                header_parts.append(formatted)
    lr_text = _format_learning_rate(record)
    if lr_text:
        header_parts.append(lr_text)
    if header_parts:
        lines.append(" | ".join(header_parts))

    train_acc_parts = _format_accuracy_parts(record, split_prefix="train")
    if train_acc_parts:
        lines.extend(_chunked_metric_lines(train_acc_parts))
    val_acc_parts = _format_accuracy_parts(record, split_prefix="val")
    if val_acc_parts:
        lines.extend(_chunked_metric_lines(val_acc_parts))

    for split_prefix in ("train", "val"):
        loss_parts = _format_component_loss_parts(record, task_contract, split_prefix=split_prefix)
        if loss_parts:
            lines.extend(_chunked_metric_lines(loss_parts))

    validation_metric_parts = _format_validation_metric_parts(record, task_contract)
    if validation_metric_parts:
        lines.extend(_chunked_metric_lines(validation_metric_parts))

    return lines


def _format_learning_rate(record: Mapping[str, Any]) -> str:
    current = _as_finite_float(record.get("learning_rate"))
    if current is None:
        return ""
    next_value = _as_finite_float(record.get("next_learning_rate"))
    if next_value is None or math.isclose(current, next_value):
        return f"lr={current:.2e}"
    return f"lr={current:.2e}->{next_value:.2e}"


def _format_accuracy_parts(
    record: Mapping[str, Any],
    *,
    split_prefix: str,
) -> list[str]:
    found: list[tuple[float, str]] = []
    for key, value in record.items():
        match = _ACCURACY_RE.fullmatch(str(key))
        if match is None or match.group(1) != split_prefix:
            continue
        tolerance_m = float(match.group(2).replace("p", "."))
        formatted = _format_key_value(
            f"{split_prefix}_acc@{tolerance_m:.2f}m",
            value,
        )
        if formatted:
            found.append((tolerance_m, formatted))
    found.sort(key=_accuracy_sort_key)
    return [formatted for _, formatted in found]


def _accuracy_sort_key(item: tuple[float, str]) -> tuple[int, float]:
    tolerance_m = float(item[0])
    try:
        preferred_index = _PREFERRED_DISTANCE_TOLERANCES_M.index(round(tolerance_m, 2))
    except ValueError:
        preferred_index = len(_PREFERRED_DISTANCE_TOLERANCES_M)
    return preferred_index, tolerance_m


def _format_component_loss_parts(
    record: Mapping[str, Any],
    task_contract: Mapping[str, Any],
    *,
    split_prefix: str,
) -> list[str]:
    parts: list[str] = []
    declared_loss_names = tuple(reporting_train_losses(task_contract))
    for loss_name in declared_loss_names:
        normalized_name = str(loss_name).strip()
        if not normalized_name or normalized_name == "total_loss":
            continue
        key = f"{split_prefix}_{normalized_name}"
        formatted = _format_key_value(key, record.get(key))
        if formatted:
            parts.append(formatted)
    if parts:
        return parts

    for key in sorted(record):
        key_text = str(key)
        if not key_text.startswith(f"{split_prefix}_") or not key_text.endswith("_loss"):
            continue
        if key_text in {f"{split_prefix}_loss", f"{split_prefix}_total_loss"}:
            continue
        formatted = _format_key_value(key_text, record.get(key_text))
        if formatted:
            parts.append(formatted)
    return parts


def _format_validation_metric_parts(
    record: Mapping[str, Any],
    task_contract: Mapping[str, Any],
) -> list[str]:
    parts: list[str] = []
    declared_keys = tuple(reporting_validation_metrics(task_contract))
    if declared_keys:
        for metric_name in declared_keys:
            key = f"val_{metric_name}"
            formatted = _format_key_value(str(metric_name), record.get(key))
            if formatted:
                parts.append(formatted)
        return parts

    fallback_keys = [
        key
        for key in sorted(record)
        if str(key).startswith("val_")
        and (
            str(key).startswith("val_yaw_")
            or str(key).startswith("val_mean_angular_")
            or str(key).startswith("val_median_angular_")
            or str(key).startswith("val_p95_angular_")
            or str(key).startswith("val_mean_position_")
            or str(key).startswith("val_median_position_")
            or str(key).startswith("val_p95_position_")
        )
    ]
    for key in fallback_keys:
        formatted = _format_key_value(str(key)[4:], record.get(key))
        if formatted:
            parts.append(formatted)
    return parts


def _chunked_metric_lines(parts: Sequence[str], *, chunk_size: int = 3) -> list[str]:
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
