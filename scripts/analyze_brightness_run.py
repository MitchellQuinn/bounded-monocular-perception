#!/usr/bin/env python
"""Generate compact research-style artifacts from a saved brightness-analysis JSON."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BASELINE_GAIN = 1.0
INTERMEDIATE_DARKENING_GAIN = 1.2
MAX_DARKENING_GAIN = 1.4

DISTANCE_SUCCESS_M = 0.10
ORIENTATION_SUCCESS_DEG = 5.0
CLEAN_DISTANCE_SUCCESS_M = 0.05
CLEAN_ORIENTATION_SUCCESS_DEG = 2.5

EDGE_BAND_FRACTION = 0.20
TOP_SENSITIVE_LIMIT = 100
CORRELATION_FEATURES = [
    "baseline_vehicle_mean_darkness",
    "baseline_vehicle_mean_intensity",
    "baseline_vehicle_std_intensity",
    "baseline_canvas_mean_intensity",
    "baseline_canvas_std_intensity",
    "baseline_vehicle_pixel_fraction",
]
CORRELATION_METRICS = [
    "baseline_abs_distance_error_m",
    "baseline_abs_orientation_error_deg",
    "max_abs_distance_shift_m",
    "distance_shift_slope_m_per_gain",
    "max_abs_orientation_shift_deg",
    "orientation_shift_slope_deg_per_gain",
]


@dataclass(frozen=True)
class RunArtifacts:
    brightness_json_path: Path
    output_dir: Path
    report_path: Path
    summary_path: Path
    gain_summary_csv_path: Path
    failure_rates_csv_path: Path
    subgroup_summary_csv_path: Path
    top_sensitive_csv_path: Path
    top_yaw_flips_csv_path: Path


def _p95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 95))


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _format_float(value: Any, decimals: int = 3, suffix: str = "") -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.{decimals}f}{suffix}"


def _format_pct(value: Any, decimals: int = 1) -> str:
    return _format_float(value, decimals=decimals, suffix="%")


def _format_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "NA"


def _format_gain(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.1f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _relative_change(new_value: float, baseline_value: float) -> float | None:
    if not np.isfinite(new_value) or not np.isfinite(baseline_value):
        return None
    if abs(baseline_value) <= 1e-12:
        return None
    return float((new_value - baseline_value) / baseline_value * 100.0)


def _coverage_pct(actual_count: int, corpus_count: int | None) -> float | None:
    if corpus_count is None or corpus_count <= 0:
        return None
    return float(actual_count / corpus_count * 100.0)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _records_to_df(records: Any, *, name: str) -> pd.DataFrame:
    if isinstance(records, list):
        return pd.DataFrame(records)
    raise ValueError(f"Expected {name} to be a list of records, got {type(records).__name__}")


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _series_percent(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    if numeric.empty:
        return float("nan")
    return float(numeric.mean() * 100.0)


def _sample_count_from_csv(samples_csv_path: Path) -> int | None:
    if not samples_csv_path.exists():
        return None
    rows = pd.read_csv(samples_csv_path, usecols=["sample_id"])
    return int(len(rows))


def _corpus_samples_csv_from_selected_root(corpus_root: Path) -> Path:
    return corpus_root / "manifests" / "samples.csv"


def _discover_inference_output_path(brightness_json_path: Path) -> Path | None:
    matches = sorted(brightness_json_path.parent.glob("inference-output_*.json"))
    if len(matches) == 1:
        return matches[0]
    return None


def _load_inference_join(
    inference_output_path: Path | None,
    *,
    fallback_corpus_root: Path,
) -> tuple[pd.DataFrame | None, list[str]]:
    notes: list[str] = []
    if inference_output_path is None or not inference_output_path.exists():
        notes.append("No matching inference-output JSON was available; edge-of-frame subgroup was omitted.")
        return None, notes

    records = _load_json(inference_output_path)
    if not isinstance(records, list) or not records:
        notes.append("Matching inference-output JSON was empty or malformed; edge-of-frame subgroup was omitted.")
        return None, notes

    join_rows: list[dict[str, Any]] = []
    source_samples_csv_path: Path | None = None
    for record in records:
        selected_image = record.get("selected_image", {})
        roi_prediction = record.get("roi_prediction", {})
        center_xy = roi_prediction.get("center_original_xy_px", [None, None])
        selected_corpus = record.get("selected_corpus", {})
        if source_samples_csv_path is None:
            source_path = selected_corpus.get("source_samples_csv_path")
            if source_path:
                source_samples_csv_path = Path(str(source_path))
        join_rows.append(
            {
                "sample_id": str(selected_image.get("sample_id")),
                "image_filename": str(selected_image.get("image_filename")),
                "roi_center_x_px": _safe_float(center_xy[0] if len(center_xy) >= 1 else None),
                "roi_center_y_px": _safe_float(center_xy[1] if len(center_xy) >= 2 else None),
            }
        )

    inference_df = pd.DataFrame(join_rows)
    if source_samples_csv_path is None:
        source_samples_csv_path = _corpus_samples_csv_from_selected_root(fallback_corpus_root)
    if source_samples_csv_path.exists():
        dims_df = pd.read_csv(
            source_samples_csv_path,
            usecols=["sample_id", "image_filename", "image_width_px", "image_height_px"],
        )
        inference_df = inference_df.merge(
            dims_df,
            on=["sample_id", "image_filename"],
            how="left",
            validate="one_to_one",
        )
    else:
        notes.append(
            f"Could not load corpus dimensions from {source_samples_csv_path}; edge-of-frame subgroup was omitted."
        )
        return None, notes

    required = ["roi_center_x_px", "roi_center_y_px", "image_width_px", "image_height_px"]
    if inference_df[required].isna().any(axis=None):
        missing_rows = int(inference_df[required].isna().any(axis=1).sum())
        notes.append(
            f"Inference join was missing ROI center or frame size for {missing_rows} samples; unmatched rows were excluded from the edge-of-frame subgroup."
        )

    valid = inference_df.dropna(subset=required).copy()
    if valid.empty:
        notes.append("Inference join did not retain any rows with usable geometry; edge-of-frame subgroup was omitted.")
        return None, notes

    valid["x_norm"] = valid["roi_center_x_px"] / valid["image_width_px"]
    valid["y_norm"] = valid["roi_center_y_px"] / valid["image_height_px"]
    valid["frame_position_bucket"] = np.where(
        (valid["x_norm"] < EDGE_BAND_FRACTION)
        | (valid["x_norm"] > (1.0 - EDGE_BAND_FRACTION))
        | (valid["y_norm"] < EDGE_BAND_FRACTION)
        | (valid["y_norm"] > (1.0 - EDGE_BAND_FRACTION)),
        "edge-of-frame",
        "central",
    )
    return valid, notes


def _assign_tertiles(
    series: pd.Series,
    *,
    labels: list[str],
    descending_labels: bool = False,
) -> pd.Series:
    if descending_labels:
        labels = list(reversed(labels))
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        return pd.qcut(numeric, q=3, labels=labels, duplicates="drop")
    except ValueError:
        ranked = numeric.rank(method="first")
        return pd.qcut(ranked, q=3, labels=labels)


def _classify_baseline_bucket(df: pd.DataFrame) -> pd.Series:
    distance_pass = df["abs_distance_error_m"] <= DISTANCE_SUCCESS_M
    yaw_pass = df["abs_orientation_error_deg"] <= ORIENTATION_SUCCESS_DEG
    labels = np.select(
        [
            distance_pass & yaw_pass,
            ~distance_pass & yaw_pass,
            distance_pass & ~yaw_pass,
            ~distance_pass & ~yaw_pass,
        ],
        [
            "baseline joint success",
            "baseline distance-only failure",
            "baseline yaw-only failure",
            "baseline joint failure",
        ],
        default="unclassified",
    )
    return pd.Series(labels, index=df.index)


def _add_operational_flags(predictions_df: pd.DataFrame) -> pd.DataFrame:
    df = predictions_df.copy()
    df["distance_success"] = df["abs_distance_error_m"] <= DISTANCE_SUCCESS_M
    df["orientation_success"] = df["abs_orientation_error_deg"] <= ORIENTATION_SUCCESS_DEG
    df["joint_success"] = df["distance_success"] & df["orientation_success"]
    df["distance_only_failure"] = (~df["distance_success"]) & df["orientation_success"]
    df["yaw_only_failure"] = df["distance_success"] & (~df["orientation_success"])
    df["joint_failure"] = (~df["distance_success"]) & (~df["orientation_success"])
    df["clean_success"] = (
        (df["abs_distance_error_m"] <= CLEAN_DISTANCE_SUCCESS_M)
        & (df["abs_orientation_error_deg"] <= CLEAN_ORIENTATION_SUCCESS_DEG)
    )
    return df


def _compute_gain_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        predictions_df.groupby("darkness_gain", sort=True, dropna=False)
        .agg(
            sample_count=("sample_id", "nunique"),
            mean_abs_distance_error_m=("abs_distance_error_m", "mean"),
            median_abs_distance_error_m=("abs_distance_error_m", "median"),
            p95_abs_distance_error_m=("abs_distance_error_m", _p95),
            mean_abs_orientation_error_deg=("abs_orientation_error_deg", "mean"),
            median_abs_orientation_error_deg=("abs_orientation_error_deg", "median"),
            p95_abs_orientation_error_deg=("abs_orientation_error_deg", _p95),
            mean_abs_distance_shift_m=("abs_distance_shift_from_baseline_m", "mean"),
            p95_abs_distance_shift_m=("abs_distance_shift_from_baseline_m", _p95),
            max_abs_distance_shift_m=("abs_distance_shift_from_baseline_m", "max"),
            mean_abs_orientation_shift_deg=("abs_orientation_shift_from_baseline_deg", "mean"),
            p95_abs_orientation_shift_deg=("abs_orientation_shift_from_baseline_deg", _p95),
            max_abs_orientation_shift_deg=("abs_orientation_shift_from_baseline_deg", "max"),
            mean_variant_vehicle_mean_darkness=("variant_vehicle_mean_darkness", "mean"),
        )
        .reset_index()
        .sort_values("darkness_gain", kind="stable")
        .reset_index(drop=True)
    )
    baseline_row = grouped.loc[grouped["darkness_gain"].sub(BASELINE_GAIN).abs().idxmin()]
    baseline_map = {
        column: float(baseline_row[column])
        for column in grouped.columns
        if column not in {"darkness_gain", "sample_count"}
    }
    for metric in [
        "mean_abs_distance_error_m",
        "median_abs_distance_error_m",
        "p95_abs_distance_error_m",
        "mean_abs_orientation_error_deg",
        "median_abs_orientation_error_deg",
        "p95_abs_orientation_error_deg",
        "mean_abs_distance_shift_m",
        "p95_abs_distance_shift_m",
        "max_abs_distance_shift_m",
        "mean_abs_orientation_shift_deg",
        "p95_abs_orientation_shift_deg",
        "max_abs_orientation_shift_deg",
    ]:
        grouped[f"delta_{metric}_vs_baseline"] = grouped[metric] - baseline_map.get(metric, 0.0)
        grouped[f"relative_{metric}_vs_baseline_pct"] = grouped.apply(
            lambda row, metric_name=metric: _relative_change(
                float(row[metric_name]),
                float(baseline_map.get(metric_name, float("nan"))),
            ),
            axis=1,
        )
    return grouped


def _compute_failure_rates(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gain, group in predictions_df.groupby("darkness_gain", sort=True, dropna=False):
        sample_count = int(group["sample_id"].nunique())
        rows.append(
            {
                "darkness_gain": float(gain),
                "sample_count": sample_count,
                "within_10cm_count": int(group["distance_success"].sum()),
                "within_10cm_pct": _series_percent(group["distance_success"]),
                "within_5deg_count": int(group["orientation_success"].sum()),
                "within_5deg_pct": _series_percent(group["orientation_success"]),
                "joint_success_count": int(group["joint_success"].sum()),
                "joint_success_pct": _series_percent(group["joint_success"]),
                "clean_success_count": int(group["clean_success"].sum()),
                "clean_success_pct": _series_percent(group["clean_success"]),
                "distance_only_failure_count": int(group["distance_only_failure"].sum()),
                "distance_only_failure_pct": _series_percent(group["distance_only_failure"]),
                "yaw_only_failure_count": int(group["yaw_only_failure"].sum()),
                "yaw_only_failure_pct": _series_percent(group["yaw_only_failure"]),
                "joint_failure_count": int(group["joint_failure"].sum()),
                "joint_failure_pct": _series_percent(group["joint_failure"]),
            }
        )
    rates = pd.DataFrame(rows).sort_values("darkness_gain", kind="stable").reset_index(drop=True)
    baseline_row = rates.loc[rates["darkness_gain"].sub(BASELINE_GAIN).abs().idxmin()]
    baseline_metrics = {
        column: float(baseline_row[column])
        for column in rates.columns
        if column.endswith("_pct")
    }
    for metric in baseline_metrics:
        rates[f"delta_{metric}_vs_baseline"] = rates[metric] - baseline_metrics[metric]
        rates[f"relative_{metric}_vs_baseline_pct"] = rates.apply(
            lambda row, metric_name=metric: _relative_change(
                float(row[metric_name]),
                float(baseline_metrics[metric_name]),
            ),
            axis=1,
        )
    return rates


def _monotonic_non_decreasing(values: list[float], tolerance: float = 1e-12) -> bool | None:
    if len(values) < 2:
        return None
    return all(
        current >= previous - tolerance
        for previous, current in zip(values[:-1], values[1:], strict=True)
    )


def _monotonic_non_increasing(values: list[float], tolerance: float = 1e-12) -> bool | None:
    if len(values) < 2:
        return None
    return all(
        current <= previous + tolerance
        for previous, current in zip(values[:-1], values[1:], strict=True)
    )


def _effect_label(signals: list[int]) -> str:
    usable = [signal for signal in signals if signal != 0]
    if not usable:
        return "mixed"
    if all(signal > 0 for signal in usable):
        return "help"
    if all(signal < 0 for signal in usable):
        return "hurt"
    return "mixed"


def _metric_signal(value: float, baseline_value: float, *, higher_is_better: bool) -> int:
    delta = value - baseline_value
    if abs(delta) <= 1e-12:
        return 0
    if higher_is_better:
        return 1 if delta > 0 else -1
    return 1 if delta < 0 else -1


def _compute_monotonicity_and_asymmetry(
    gain_summary_df: pd.DataFrame,
    failure_rates_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    merged = gain_summary_df.merge(failure_rates_df, on=["darkness_gain", "sample_count"], how="inner")
    darker = merged.loc[merged["darkness_gain"] >= BASELINE_GAIN].sort_values("darkness_gain", kind="stable")

    monotonicity = {
        "distance_error_monotonic_non_decreasing_for_darkening": _monotonic_non_decreasing(
            darker["mean_abs_distance_error_m"].tolist()
        ),
        "orientation_error_monotonic_non_decreasing_for_darkening": _monotonic_non_decreasing(
            darker["mean_abs_orientation_error_deg"].tolist()
        ),
        "joint_success_monotonic_non_increasing_for_darkening": _monotonic_non_increasing(
            darker["joint_success_pct"].tolist()
        ),
        "clean_success_monotonic_non_increasing_for_darkening": _monotonic_non_increasing(
            darker["clean_success_pct"].tolist()
        ),
        "yaw_success_monotonic_non_increasing_for_darkening": _monotonic_non_increasing(
            darker["within_5deg_pct"].tolist()
        ),
        "distance_success_monotonic_non_increasing_for_darkening": _monotonic_non_increasing(
            darker["within_10cm_pct"].tolist()
        ),
    }

    baseline_row = merged.loc[merged["darkness_gain"].sub(BASELINE_GAIN).abs().idxmin()]
    bright = merged.loc[merged["darkness_gain"] < BASELINE_GAIN].sort_values("darkness_gain", kind="stable")
    monotonicity["brightening_effect_distance"] = _effect_label(
        [
            _metric_signal(float(row["mean_abs_distance_error_m"]), float(baseline_row["mean_abs_distance_error_m"]), higher_is_better=False)
            for _, row in bright.iterrows()
        ]
        + [
            _metric_signal(float(row["within_10cm_pct"]), float(baseline_row["within_10cm_pct"]), higher_is_better=True)
            for _, row in bright.iterrows()
        ]
    )
    monotonicity["brightening_effect_yaw"] = _effect_label(
        [
            _metric_signal(float(row["mean_abs_orientation_error_deg"]), float(baseline_row["mean_abs_orientation_error_deg"]), higher_is_better=False)
            for _, row in bright.iterrows()
        ]
        + [
            _metric_signal(float(row["within_5deg_pct"]), float(baseline_row["within_5deg_pct"]), higher_is_better=True)
            for _, row in bright.iterrows()
        ]
    )
    monotonicity["brightening_effect_joint"] = _effect_label(
        [
            _metric_signal(float(row["joint_success_pct"]), float(baseline_row["joint_success_pct"]), higher_is_better=True)
            for _, row in bright.iterrows()
        ]
        + [
            _metric_signal(float(row["clean_success_pct"]), float(baseline_row["clean_success_pct"]), higher_is_better=True)
            for _, row in bright.iterrows()
        ]
    )

    asymmetry_rows: list[dict[str, Any]] = []
    for bright_gain, dark_gain in [(0.8, 1.2), (0.6, 1.4)]:
        bright_match = merged.loc[np.isclose(merged["darkness_gain"], bright_gain)]
        dark_match = merged.loc[np.isclose(merged["darkness_gain"], dark_gain)]
        if bright_match.empty or dark_match.empty:
            continue
        bright_row = bright_match.iloc[0]
        dark_row = dark_match.iloc[0]
        asymmetry_rows.append(
            {
                "pair_label": f"{bright_gain:.1f}_vs_{dark_gain:.1f}",
                "bright_gain": bright_gain,
                "dark_gain": dark_gain,
                "bright_delta_mean_abs_distance_error_m": float(
                    bright_row["mean_abs_distance_error_m"] - baseline_row["mean_abs_distance_error_m"]
                ),
                "dark_delta_mean_abs_distance_error_m": float(
                    dark_row["mean_abs_distance_error_m"] - baseline_row["mean_abs_distance_error_m"]
                ),
                "dark_minus_bright_distance_error_delta_m": float(
                    (dark_row["mean_abs_distance_error_m"] - baseline_row["mean_abs_distance_error_m"])
                    - (bright_row["mean_abs_distance_error_m"] - baseline_row["mean_abs_distance_error_m"])
                ),
                "bright_delta_mean_abs_orientation_error_deg": float(
                    bright_row["mean_abs_orientation_error_deg"] - baseline_row["mean_abs_orientation_error_deg"]
                ),
                "dark_delta_mean_abs_orientation_error_deg": float(
                    dark_row["mean_abs_orientation_error_deg"] - baseline_row["mean_abs_orientation_error_deg"]
                ),
                "dark_minus_bright_orientation_error_delta_deg": float(
                    (dark_row["mean_abs_orientation_error_deg"] - baseline_row["mean_abs_orientation_error_deg"])
                    - (bright_row["mean_abs_orientation_error_deg"] - baseline_row["mean_abs_orientation_error_deg"])
                ),
                "bright_joint_success_delta_pct_points": float(
                    bright_row["joint_success_pct"] - baseline_row["joint_success_pct"]
                ),
                "dark_joint_success_delta_pct_points": float(
                    dark_row["joint_success_pct"] - baseline_row["joint_success_pct"]
                ),
                "dark_minus_bright_joint_success_delta_pct_points": float(
                    (dark_row["joint_success_pct"] - baseline_row["joint_success_pct"])
                    - (bright_row["joint_success_pct"] - baseline_row["joint_success_pct"])
                ),
                "bright_clean_success_delta_pct_points": float(
                    bright_row["clean_success_pct"] - baseline_row["clean_success_pct"]
                ),
                "dark_clean_success_delta_pct_points": float(
                    dark_row["clean_success_pct"] - baseline_row["clean_success_pct"]
                ),
                "dark_minus_bright_clean_success_delta_pct_points": float(
                    (dark_row["clean_success_pct"] - baseline_row["clean_success_pct"])
                    - (bright_row["clean_success_pct"] - baseline_row["clean_success_pct"])
                ),
            }
        )
    return monotonicity, pd.DataFrame(asymmetry_rows)


def _compute_sensitivity_delta(
    gain_summary_df: pd.DataFrame,
    failure_rates_df: pd.DataFrame,
    *,
    target_gain: float,
) -> dict[str, Any] | None:
    merged = gain_summary_df.merge(failure_rates_df, on=["darkness_gain", "sample_count"], how="inner")
    baseline_match = merged.loc[np.isclose(merged["darkness_gain"], BASELINE_GAIN)]
    target_match = merged.loc[np.isclose(merged["darkness_gain"], target_gain)]
    if baseline_match.empty or target_match.empty:
        return None
    baseline_row = baseline_match.iloc[0]
    target_row = target_match.iloc[0]
    return {
        "gain": target_gain,
        "distance_mae_delta_m": float(target_row["mean_abs_distance_error_m"] - baseline_row["mean_abs_distance_error_m"]),
        "distance_mae_relative_pct": _relative_change(
            float(target_row["mean_abs_distance_error_m"]),
            float(baseline_row["mean_abs_distance_error_m"]),
        ),
        "orientation_mae_delta_deg": float(
            target_row["mean_abs_orientation_error_deg"] - baseline_row["mean_abs_orientation_error_deg"]
        ),
        "orientation_mae_relative_pct": _relative_change(
            float(target_row["mean_abs_orientation_error_deg"]),
            float(baseline_row["mean_abs_orientation_error_deg"]),
        ),
        "within_10cm_delta_pct_points": float(target_row["within_10cm_pct"] - baseline_row["within_10cm_pct"]),
        "within_5deg_delta_pct_points": float(target_row["within_5deg_pct"] - baseline_row["within_5deg_pct"]),
        "joint_success_delta_pct_points": float(
            target_row["joint_success_pct"] - baseline_row["joint_success_pct"]
        ),
        "clean_success_delta_pct_points": float(
            target_row["clean_success_pct"] - baseline_row["clean_success_pct"]
        ),
    }


def _compute_correlations(per_sample_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in CORRELATION_FEATURES:
        feature_series = pd.to_numeric(per_sample_df[feature], errors="coerce")
        for metric in CORRELATION_METRICS:
            metric_series = pd.to_numeric(per_sample_df[metric], errors="coerce")
            valid = feature_series.notna() & metric_series.notna()
            sample_count = int(valid.sum())
            pearson_r = float("nan")
            if sample_count >= 2:
                feature_valid = feature_series[valid]
                metric_valid = metric_series[valid]
                if feature_valid.nunique(dropna=True) >= 2 and metric_valid.nunique(dropna=True) >= 2:
                    pearson_r = float(feature_valid.corr(metric_valid))
            rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "pearson_r": pearson_r,
                    "abs_pearson_r": abs(pearson_r) if np.isfinite(pearson_r) else float("nan"),
                    "sample_count": sample_count,
                }
            )
    correlations_df = pd.DataFrame(rows).sort_values(
        ["metric", "abs_pearson_r"], ascending=[True, False], kind="stable"
    )
    return correlations_df.reset_index(drop=True)


def _top_correlations_by_metric(correlations_df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for metric in CORRELATION_METRICS:
        rows = correlations_df.loc[correlations_df["metric"] == metric].head(3)
        result[metric] = rows.to_dict(orient="records")
    return result


def _build_bucketed_sample_frame(
    per_sample_df: pd.DataFrame,
    baseline_predictions_df: pd.DataFrame,
    *,
    inference_join_df: pd.DataFrame | None,
) -> pd.DataFrame:
    sample_df = per_sample_df.copy()
    baseline_selected = baseline_predictions_df[["sample_id", "image_filename", "abs_distance_error_m", "abs_orientation_error_deg"]].copy()
    sample_df = sample_df.merge(
        baseline_selected,
        on=["sample_id", "image_filename"],
        how="left",
        validate="one_to_one",
    )
    sample_df["baseline_performance_bucket"] = _classify_baseline_bucket(sample_df)
    sample_df["distance_bucket"] = _assign_tertiles(
        sample_df["truth_distance_m"],
        labels=["near", "mid", "far"],
    ).astype(str)
    sample_df["roi_size_bucket"] = _assign_tertiles(
        sample_df["baseline_vehicle_pixel_fraction"],
        labels=["small", "medium", "large"],
    ).astype(str)
    sample_df["baseline_darkness_bucket"] = _assign_tertiles(
        sample_df["baseline_vehicle_mean_darkness"],
        labels=["bright", "medium", "dark"],
    ).astype(str)
    if inference_join_df is not None:
        sample_df = sample_df.merge(
            inference_join_df[
                [
                    "sample_id",
                    "image_filename",
                    "frame_position_bucket",
                    "roi_center_x_px",
                    "roi_center_y_px",
                    "image_width_px",
                    "image_height_px",
                    "x_norm",
                    "y_norm",
                ]
            ],
            on=["sample_id", "image_filename"],
            how="left",
        )
    else:
        sample_df["frame_position_bucket"] = pd.NA
    return sample_df


def _subgroup_failure_metrics(predictions_df: pd.DataFrame) -> dict[str, float]:
    if predictions_df.empty:
        return {
            "within_10cm_pct": float("nan"),
            "within_5deg_pct": float("nan"),
            "joint_success_pct": float("nan"),
            "clean_success_pct": float("nan"),
            "distance_only_failure_pct": float("nan"),
            "yaw_only_failure_pct": float("nan"),
            "joint_failure_pct": float("nan"),
        }
    return {
        "within_10cm_pct": _series_percent(predictions_df["distance_success"]),
        "within_5deg_pct": _series_percent(predictions_df["orientation_success"]),
        "joint_success_pct": _series_percent(predictions_df["joint_success"]),
        "clean_success_pct": _series_percent(predictions_df["clean_success"]),
        "distance_only_failure_pct": _series_percent(predictions_df["distance_only_failure"]),
        "yaw_only_failure_pct": _series_percent(predictions_df["yaw_only_failure"]),
        "joint_failure_pct": _series_percent(predictions_df["joint_failure"]),
    }


def _compute_subgroup_summary(
    sample_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    *,
    darkening_gain: float = MAX_DARKENING_GAIN,
) -> pd.DataFrame:
    baseline_predictions = predictions_df.loc[np.isclose(predictions_df["darkness_gain"], BASELINE_GAIN)].copy()
    dark_predictions = predictions_df.loc[np.isclose(predictions_df["darkness_gain"], darkening_gain)].copy()

    family_specs = [
        ("baseline_performance_bucket", ["baseline joint success", "baseline distance-only failure", "baseline yaw-only failure", "baseline joint failure"]),
        ("distance_bucket", ["near", "mid", "far"]),
        ("roi_size_bucket", ["small", "medium", "large"]),
        ("baseline_darkness_bucket", ["dark", "medium", "bright"]),
    ]
    if sample_df["frame_position_bucket"].notna().any():
        family_specs.append(("frame_position_bucket", ["central", "edge-of-frame"]))

    rows: list[dict[str, Any]] = []
    for family, order in family_specs:
        for subgroup in order:
            subgroup_samples = sample_df.loc[sample_df[family] == subgroup, ["sample_id", "image_filename"]]
            if subgroup_samples.empty:
                continue
            baseline_subset = baseline_predictions.merge(subgroup_samples, on=["sample_id", "image_filename"], how="inner")
            dark_subset = dark_predictions.merge(subgroup_samples, on=["sample_id", "image_filename"], how="inner")
            baseline_failures = _subgroup_failure_metrics(baseline_subset)
            dark_failures = _subgroup_failure_metrics(dark_subset)
            rows.append(
                {
                    "subgroup_family": family,
                    "subgroup": subgroup,
                    "sample_count": int(len(subgroup_samples)),
                    "baseline_mean_abs_distance_error_m": float(baseline_subset["abs_distance_error_m"].mean()),
                    "baseline_mean_abs_orientation_error_deg": float(
                        baseline_subset["abs_orientation_error_deg"].mean()
                    ),
                    "gain_1_4_mean_abs_distance_error_m": float(dark_subset["abs_distance_error_m"].mean()),
                    "gain_1_4_mean_abs_orientation_error_deg": float(
                        dark_subset["abs_orientation_error_deg"].mean()
                    ),
                    "delta_mean_abs_distance_error_m_1_4_vs_baseline": float(
                        dark_subset["abs_distance_error_m"].mean() - baseline_subset["abs_distance_error_m"].mean()
                    ),
                    "delta_mean_abs_orientation_error_deg_1_4_vs_baseline": float(
                        dark_subset["abs_orientation_error_deg"].mean()
                        - baseline_subset["abs_orientation_error_deg"].mean()
                    ),
                    "baseline_within_10cm_pct": baseline_failures["within_10cm_pct"],
                    "baseline_within_5deg_pct": baseline_failures["within_5deg_pct"],
                    "baseline_joint_success_pct": baseline_failures["joint_success_pct"],
                    "baseline_clean_success_pct": baseline_failures["clean_success_pct"],
                    "baseline_distance_only_failure_pct": baseline_failures["distance_only_failure_pct"],
                    "baseline_yaw_only_failure_pct": baseline_failures["yaw_only_failure_pct"],
                    "baseline_joint_failure_pct": baseline_failures["joint_failure_pct"],
                    "gain_1_4_within_10cm_pct": dark_failures["within_10cm_pct"],
                    "gain_1_4_within_5deg_pct": dark_failures["within_5deg_pct"],
                    "gain_1_4_joint_success_pct": dark_failures["joint_success_pct"],
                    "gain_1_4_clean_success_pct": dark_failures["clean_success_pct"],
                    "gain_1_4_distance_only_failure_pct": dark_failures["distance_only_failure_pct"],
                    "gain_1_4_yaw_only_failure_pct": dark_failures["yaw_only_failure_pct"],
                    "gain_1_4_joint_failure_pct": dark_failures["joint_failure_pct"],
                }
            )
    return pd.DataFrame(rows)


def _per_sample_worst_rows(predictions_df: pd.DataFrame) -> pd.DataFrame:
    worst_rows: list[pd.Series] = []
    for _, group in predictions_df.groupby("sample_id", sort=False, dropna=False):
        ordered = group.sort_values(
            ["abs_orientation_shift_from_baseline_deg", "abs_distance_shift_from_baseline_m", "darkness_gain"],
            ascending=[False, False, False],
            kind="stable",
        )
        worst_rows.append(ordered.iloc[0])
    return pd.DataFrame(worst_rows).reset_index(drop=True)


def _compute_sensitive_tables(
    sample_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    worst_df = _per_sample_worst_rows(predictions_df)
    merged = sample_df.merge(
        worst_df[
            [
                "sample_id",
                "image_filename",
                "darkness_gain",
                "abs_distance_shift_from_baseline_m",
                "abs_orientation_shift_from_baseline_deg",
            ]
        ],
        on=["sample_id", "image_filename"],
        how="left",
        validate="one_to_one",
    )
    merged = merged.rename(
        columns={
            "darkness_gain": "worst_darkness_gain",
            "abs_distance_shift_from_baseline_m": "worst_gain_abs_distance_shift_m",
            "abs_orientation_shift_from_baseline_deg": "worst_gain_abs_orientation_shift_deg",
        }
    )
    top_sensitive = merged.sort_values(
        ["max_abs_orientation_shift_deg", "max_abs_distance_shift_m", "baseline_abs_orientation_error_deg"],
        ascending=[False, False, False],
        kind="stable",
    ).head(TOP_SENSITIVE_LIMIT).copy()
    top_sensitive = top_sensitive[
        [
            "sample_id",
            "image_filename",
            "truth_distance_m",
            "truth_orientation_deg",
            "baseline_abs_distance_error_m",
            "baseline_abs_orientation_error_deg",
            "baseline_performance_bucket",
            "distance_bucket",
            "roi_size_bucket",
            "baseline_darkness_bucket",
            "frame_position_bucket",
            "worst_darkness_gain",
            "mean_abs_distance_shift_m",
            "max_abs_distance_shift_m",
            "distance_shift_slope_m_per_gain",
            "mean_abs_orientation_shift_deg",
            "max_abs_orientation_shift_deg",
            "orientation_shift_slope_deg_per_gain",
            "baseline_vehicle_pixel_fraction",
            "baseline_vehicle_mean_darkness",
            "baseline_vehicle_mean_intensity",
            "baseline_vehicle_std_intensity",
            "roi_center_x_px",
            "roi_center_y_px",
            "image_width_px",
            "image_height_px",
            "x_norm",
            "y_norm",
        ]
    ]

    catastrophic = merged.loc[merged["max_abs_orientation_shift_deg"] > 90.0].copy()
    catastrophic["strict_gt_120deg"] = catastrophic["max_abs_orientation_shift_deg"] > 120.0
    catastrophic = catastrophic.sort_values(
        ["max_abs_orientation_shift_deg", "max_abs_distance_shift_m", "baseline_abs_orientation_error_deg"],
        ascending=[False, False, False],
        kind="stable",
    ).copy()
    catastrophic = catastrophic[
        [
            "sample_id",
            "image_filename",
            "truth_distance_m",
            "truth_orientation_deg",
            "baseline_abs_distance_error_m",
            "baseline_abs_orientation_error_deg",
            "baseline_performance_bucket",
            "distance_bucket",
            "roi_size_bucket",
            "baseline_darkness_bucket",
            "frame_position_bucket",
            "worst_darkness_gain",
            "max_abs_distance_shift_m",
            "max_abs_orientation_shift_deg",
            "mean_abs_distance_shift_m",
            "mean_abs_orientation_shift_deg",
            "baseline_vehicle_pixel_fraction",
            "baseline_vehicle_mean_darkness",
            "baseline_vehicle_mean_intensity",
            "baseline_vehicle_std_intensity",
            "roi_center_x_px",
            "roi_center_y_px",
            "image_width_px",
            "image_height_px",
            "x_norm",
            "y_norm",
            "strict_gt_120deg",
        ]
    ]

    catastrophic_by_gain_rows: list[dict[str, Any]] = []
    for gain, group in predictions_df.loc[predictions_df["darkness_gain"] != BASELINE_GAIN].groupby("darkness_gain", sort=True):
        catastrophic_by_gain_rows.append(
            {
                "darkness_gain": float(gain),
                "catastrophic_gt_90deg_count": int((group["abs_orientation_shift_from_baseline_deg"] > 90.0).sum()),
                "catastrophic_gt_120deg_count": int((group["abs_orientation_shift_from_baseline_deg"] > 120.0).sum()),
            }
        )
    catastrophic_by_gain = pd.DataFrame(catastrophic_by_gain_rows)
    return top_sensitive, catastrophic, catastrophic_by_gain


def _compute_catastrophic_subgroup_concentration(
    sample_df: pd.DataFrame,
    catastrophic_df: pd.DataFrame,
) -> pd.DataFrame:
    sample_ids = set(catastrophic_df["sample_id"])
    concentration_df = sample_df.copy()
    concentration_df["ever_catastrophic_yaw_flip_gt_90deg"] = concentration_df["sample_id"].isin(sample_ids)
    family_specs = [
        "baseline_performance_bucket",
        "distance_bucket",
        "roi_size_bucket",
        "baseline_darkness_bucket",
    ]
    if concentration_df["frame_position_bucket"].notna().any():
        family_specs.append("frame_position_bucket")

    rows: list[dict[str, Any]] = []
    for family in family_specs:
        for subgroup, group in concentration_df.groupby(family, dropna=True):
            rows.append(
                {
                    "subgroup_family": family,
                    "subgroup": subgroup,
                    "sample_count": int(len(group)),
                    "catastrophic_count": int(group["ever_catastrophic_yaw_flip_gt_90deg"].sum()),
                    "catastrophic_rate_pct": _series_percent(group["ever_catastrophic_yaw_flip_gt_90deg"]),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["catastrophic_rate_pct", "sample_count"], ascending=[False, False], kind="stable"
    )
    return result.reset_index(drop=True)


def _quality_checks(
    raw_data: dict[str, Any],
    gain_summary_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
) -> list[str]:
    notes: list[str] = []

    embedded_aggregate = raw_data.get("aggregate_summary")
    if isinstance(embedded_aggregate, list) and embedded_aggregate:
        embedded_df = pd.DataFrame(embedded_aggregate).sort_values("darkness_gain", kind="stable")
        merged = gain_summary_df.merge(
            embedded_df,
            on="darkness_gain",
            suffixes=("_recomputed", "_embedded"),
            how="inner",
        )
        check_columns = [
            "mean_abs_distance_error_m",
            "median_abs_distance_error_m",
            "p95_abs_distance_error_m",
            "mean_abs_orientation_error_deg",
            "median_abs_orientation_error_deg",
            "p95_abs_orientation_error_deg",
            "mean_abs_distance_shift_m",
            "mean_abs_orientation_shift_deg",
        ]
        max_diff = 0.0
        for column in check_columns:
            diffs = (
                merged[f"{column}_recomputed"] - merged[f"{column}_embedded"]
            ).abs()
            diff_value = float(diffs.max()) if not diffs.empty else 0.0
            max_diff = max(max_diff, diff_value)
        if max_diff <= 1e-9:
            notes.append("Embedded aggregate summary matched the recomputed gain summary within numerical tolerance.")
        else:
            notes.append(
                f"Embedded aggregate summary differed from the recomputed gain summary by up to {max_diff:.6g}; recomputed values were used in this report."
            )

    embedded_correlations = raw_data.get("brightness_correlations")
    if isinstance(embedded_correlations, list) and embedded_correlations:
        embedded_df = pd.DataFrame(embedded_correlations)
        merged_corr = correlations_df.merge(
            embedded_df,
            on=["feature", "metric"],
            suffixes=("_recomputed", "_embedded"),
            how="inner",
        )
        if not merged_corr.empty:
            diffs = (merged_corr["pearson_r_recomputed"] - merged_corr["pearson_r_embedded"]).abs()
            max_diff = float(diffs.max())
            if max_diff <= 1e-9:
                notes.append("Embedded correlation table matched the recomputed correlation values within numerical tolerance.")
            else:
                notes.append(
                    f"Embedded correlation table differed from the recomputed values by up to {max_diff:.6g}; recomputed values were used in this report."
                )

    return notes


def _baseline_assessment_text(baseline_rates_row: pd.Series, baseline_gain_row: pd.Series) -> dict[str, str]:
    distance_rate = float(baseline_rates_row["within_10cm_pct"])
    yaw_rate = float(baseline_rates_row["within_5deg_pct"])
    joint_rate = float(baseline_rates_row["joint_success_pct"])
    distance_mae = float(baseline_gain_row["mean_abs_distance_error_m"])
    yaw_mae = float(baseline_gain_row["mean_abs_orientation_error_deg"])

    if distance_rate >= 90.0:
        distance_assessment = "Distance is the stronger channel under the 10 cm criterion in this run, although the mean absolute distance error remains close to that threshold."
    elif distance_rate >= 75.0:
        distance_assessment = "Distance is materially stronger than yaw and shows partial operational utility under the 10 cm criterion in this run, but it is not close to ceiling."
    else:
        distance_assessment = "Distance is not yet consistently reliable under the 10 cm criterion in this run."

    if yaw_rate >= 90.0:
        yaw_assessment = "Yaw is operationally strong at baseline under the 5 deg criterion."
    elif yaw_rate >= 75.0:
        yaw_assessment = "Yaw shows some operational utility at baseline, but the margin to failure is limited."
    else:
        yaw_assessment = "Yaw is already weak at baseline under the 5 deg criterion before any brightness perturbation."

    if yaw_rate + 10.0 < distance_rate and yaw_mae > 5.0:
        brittleness = "Baseline weakness is already concentrated in yaw rather than distance."
    elif joint_rate < distance_rate and yaw_mae > distance_mae * 50.0:
        brittleness = "Joint performance is baseline-limited primarily by yaw rather than distance."
    else:
        brittleness = "Baseline weakness is shared across distance and yaw, though yaw still requires closer scrutiny."

    return {
        "distance_assessment": distance_assessment,
        "yaw_assessment": yaw_assessment,
        "brittleness_assessment": brittleness,
    }


def _brightness_sensitivity_conclusion(
    sensitivity_1_2: dict[str, Any] | None,
    sensitivity_1_4: dict[str, Any] | None,
    monotonicity: dict[str, Any],
    asymmetry_df: pd.DataFrame,
) -> list[str]:
    conclusions: list[str] = []
    if sensitivity_1_4 is not None:
        orientation_delta = sensitivity_1_4["orientation_mae_delta_deg"]
        distance_delta = sensitivity_1_4["distance_mae_delta_m"]
        if orientation_delta is not None and distance_delta is not None:
            if orientation_delta > 10.0 and distance_delta < 0.05:
                conclusions.append(
                    "The model is materially affected by brightness-linked perturbations, with the orientation channel degrading far more sharply than distance."
                )
            elif orientation_delta > distance_delta * 100.0:
                conclusions.append(
                    "Brightness-linked perturbations materially affect the model, with substantially larger deterioration in yaw than in distance."
                )
            else:
                conclusions.append(
                    "Brightness-linked perturbations materially affect the model, although the damage is not isolated to yaw alone."
                )
        if sensitivity_1_4["joint_success_delta_pct_points"] < 0.0:
            conclusions.append(
                "Operational degradation is visible not only in MAE but also in the thresholded success rates, especially joint success and clean success."
            )
    if monotonicity.get("orientation_error_monotonic_non_decreasing_for_darkening"):
        conclusions.append("Darkening above baseline produced a monotonic deterioration in mean orientation error in this run.")
    if monotonicity.get("distance_error_monotonic_non_decreasing_for_darkening"):
        conclusions.append("Distance error also worsened monotonically with darkening, but the magnitude remained much smaller.")
    conclusions.append(
        "The dominant problem appears to be both baseline yaw weakness and brightness-triggered yaw collapse under stronger darkening."
    )
    if not asymmetry_df.empty:
        if (asymmetry_df["dark_minus_bright_orientation_error_delta_deg"] > 0).all():
            conclusions.append("Darkening harmed yaw more than equally sized brightening shifts across the paired comparisons that were available.")
        if (asymmetry_df["dark_minus_bright_distance_error_delta_m"] > 0).all():
            conclusions.append("Darkening also harmed distance more than equally sized brightening shifts, although the absolute effect on distance was modest.")
    conclusions.append(
        "These results support brightness sensitivity in the deployed model-plus-preprocessing stack, but they do not by themselves prove direct reliance on brightness as a shortcut cue."
    )
    conclusions.append(
        "The perturbation changes photometric content inside the vehicle silhouette, so the evidence is consistent both with brightness dependence and with information loss under darker renderings."
    )
    return conclusions


def _run_scope_text(run_metadata: dict[str, Any]) -> dict[str, str]:
    actual = int(run_metadata["actual_sample_count"])
    corpus_count = run_metadata.get("corpus_sample_count")
    selection_mode = str(run_metadata["selection_mode"])
    offset = int(run_metadata["selection_offset"])
    coverage_pct = run_metadata.get("validation_coverage_pct")

    if run_metadata.get("is_effectively_full_validation"):
        title = f"Brightness Robustness Analysis: {actual:,}-Sample Validation Sweep"
        scope = (
            "This note analyses the completed brightness-sensitivity artifact saved in JSON form for the "
            "distance-orientation model paired with the ROI-FCN cropper. The analysis covers the full "
            "validation corpus represented by the saved JSON and does not rerun inference."
        )
        validation_note = (
            f"This artifact covers the entire validation corpus (`{actual:,}` / `{corpus_count:,}` samples)."
        )
        limitations_note = (
            "This analysis uses the full saved validation corpus rather than a small exploratory slice, so the "
            "reported rates are representative of this completed run."
        )
        return {
            "title": title,
            "scope": scope,
            "validation_note": validation_note,
            "limitations_note": limitations_note,
        }

    if (
        corpus_count is not None
        and actual + offset == corpus_count
        and selection_mode == "slice"
        and offset > 0
    ):
        title = f"Brightness Robustness Analysis: {actual:,}-Sample Near-Full Validation Sweep"
        scope = (
            "This note analyses the completed brightness-sensitivity artifact saved in JSON form for the "
            "distance-orientation model paired with the ROI-FCN cropper. The analysis covers nearly the full "
            "validation corpus represented by the saved JSON and does not rerun inference."
        )
        validation_note = (
            f"This artifact is effectively a full validation sweep in practical terms, covering `{actual:,}` of "
            f"`{corpus_count:,}` samples ({_format_pct(coverage_pct, decimals=3)}) with `selection.mode = "
            f"{selection_mode}` and `offset = {offset}`. Exactly `{offset:,}` leading sample was omitted."
        )
        limitations_note = (
            "This analysis is based on a near-complete validation sweep rather than a small exploratory slice. "
            "The only coverage gap visible in the saved metadata is the omitted leading sample implied by the "
            f"`offset = {offset}` selection."
        )
        return {
            "title": title,
            "scope": scope,
            "validation_note": validation_note,
            "limitations_note": limitations_note,
        }

    descriptor = "slice" if actual < 10000 else "partial validation run"
    title = f"Brightness Robustness Analysis: {actual:,}-Sample {descriptor.title()}"
    scope = (
        "This note analyses the completed brightness-sensitivity artifact saved in JSON form for the "
        "distance-orientation model paired with the ROI-FCN cropper. The analysis is restricted to the saved "
        "sample set and does not rerun inference."
    )
    if corpus_count is None:
        validation_note = (
            f"This artifact covers `{actual:,}` samples, but the corpus manifest count was unavailable, so "
            "full-validation coverage could not be confirmed."
        )
        limitations_note = (
            "This analysis is limited to the saved sample set, and the corpus-level coverage could not be "
            "verified from local metadata."
        )
    else:
        validation_note = (
            f"This artifact is not a full validation sweep. It covers `{actual:,}` of `{corpus_count:,}` samples "
            f"({_format_pct(coverage_pct, decimals=2)}) with `selection.mode = {selection_mode}` and `offset = "
            f"{offset}`."
        )
        limitations_note = (
            "This analysis concerns a subset of the validation corpus rather than the full validation set, so the "
            "exact rates should be treated as slice-level estimates."
        )
    return {
        "title": title,
        "scope": scope,
        "validation_note": validation_note,
        "limitations_note": limitations_note,
    }


def _recommendations_text(
    baseline_rates_row: pd.Series,
    sensitivity_1_4: dict[str, Any] | None,
) -> list[str]:
    recommendations: list[str] = []
    yaw_baseline = float(baseline_rates_row["within_5deg_pct"])
    joint_drop = float(sensitivity_1_4["joint_success_delta_pct_points"]) if sensitivity_1_4 else float("nan")
    yaw_drop = float(sensitivity_1_4["within_5deg_delta_pct_points"]) if sensitivity_1_4 else float("nan")
    recommendations.append(
        "A brightness-normalization experiment is justified as a next hypothesis test because the perturbation materially changes operational performance, especially in yaw."
    )
    if np.isfinite(yaw_drop) and yaw_drop <= -15.0:
        recommendations.append(
            "The current evidence is strong enough to make normalization a high-priority experiment, but not strong enough to claim that normalization is already known to be necessary or sufficient."
        )
    else:
        recommendations.append(
            "The current evidence supports normalization as a justified diagnostic intervention rather than a proven remedy."
        )
    if yaw_baseline < 75.0:
        recommendations.append(
            "Because yaw is already weak at baseline, a separate orientation stream is also justified as a follow-on architecture test if preprocessing alone does not recover enough robustness."
        )
    else:
        recommendations.append(
            "Architecture changes should remain secondary until preprocessing robustness has been tested, because the baseline yaw channel is not cleanly isolated from photometric fragility."
        )
    if np.isfinite(joint_drop) and joint_drop <= -10.0:
        recommendations.append(
            "The recommended order is preprocessing normalization first, then a split or orientation-specific stream if yaw remains the bottleneck after photometric stabilization."
        )
    else:
        recommendations.append(
            "The recommended order is still preprocessing normalization first, with architecture separation reserved for the next iteration if the yaw deficit persists."
        )
    return recommendations


def _correlation_interpretation_text(feature: str, pearson_r: float) -> str:
    magnitude = abs(pearson_r)
    if magnitude < 0.10:
        qualifier = "No meaningful linear association was evident"
    elif magnitude < 0.30:
        qualifier = "Only a weak linear association was evident"
    elif magnitude < 0.50:
        qualifier = "A moderate linear association was evident"
    else:
        qualifier = "A comparatively strong linear association was evident"
    return f"{qualifier}; the largest absolute correlation among the specified appearance features was `{feature}` with Pearson r = {_format_float(pearson_r, decimals=3)}."


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    table_df = df[columns].copy()
    if table_df.empty:
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        return "\n".join([header, separator])
    rows = [columns] + table_df.astype(str).values.tolist()
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(columns))]

    def render_row(row: list[Any]) -> str:
        cells = [str(cell).ljust(widths[index]) for index, cell in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    header = render_row(columns)
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row([str(cell) for cell in row]) for row in table_df.astype(str).values.tolist()]
    return "\n".join([header, separator, *body])


def _prepare_report(
    *,
    run_metadata: dict[str, Any],
    gain_summary_df: pd.DataFrame,
    failure_rates_df: pd.DataFrame,
    monotonicity: dict[str, Any],
    asymmetry_df: pd.DataFrame,
    top_correlations: dict[str, list[dict[str, Any]]],
    subgroup_summary_df: pd.DataFrame,
    top_sensitive_df: pd.DataFrame,
    catastrophic_df: pd.DataFrame,
    catastrophic_by_gain_df: pd.DataFrame,
    catastrophic_concentration_df: pd.DataFrame,
    quality_notes: list[str],
    assessment_text: dict[str, str],
    brightness_conclusions: list[str],
    recommendations: list[str],
    sensitivity_1_2: dict[str, Any] | None,
    sensitivity_1_4: dict[str, Any] | None,
) -> str:
    baseline_gain_row = gain_summary_df.loc[np.isclose(gain_summary_df["darkness_gain"], BASELINE_GAIN)].iloc[0]
    baseline_rates_row = failure_rates_df.loc[np.isclose(failure_rates_df["darkness_gain"], BASELINE_GAIN)].iloc[0]
    scope_text = _run_scope_text(run_metadata)
    gain_summary_display = gain_summary_df[
        [
            "darkness_gain",
            "sample_count",
            "mean_abs_distance_error_m",
            "median_abs_distance_error_m",
            "p95_abs_distance_error_m",
            "mean_abs_orientation_error_deg",
            "median_abs_orientation_error_deg",
            "p95_abs_orientation_error_deg",
            "mean_abs_distance_shift_m",
            "p95_abs_distance_shift_m",
            "max_abs_distance_shift_m",
            "mean_abs_orientation_shift_deg",
            "p95_abs_orientation_shift_deg",
            "max_abs_orientation_shift_deg",
        ]
    ].copy()
    for column in gain_summary_display.columns:
        if column == "darkness_gain":
            gain_summary_display[column] = gain_summary_display[column].map(_format_gain)
        elif column == "sample_count":
            gain_summary_display[column] = gain_summary_display[column].map(_format_int)
        else:
            decimals = 4 if "distance" in column else 2
            gain_summary_display[column] = gain_summary_display[column].map(lambda value, d=decimals: _format_float(value, decimals=d))

    delta_display = gain_summary_df[
        [
            "darkness_gain",
            "delta_mean_abs_distance_error_m_vs_baseline",
            "delta_mean_abs_orientation_error_deg_vs_baseline",
        ]
    ].merge(
        failure_rates_df[
            [
                "darkness_gain",
                "delta_within_10cm_pct_vs_baseline",
                "delta_within_5deg_pct_vs_baseline",
                "delta_joint_success_pct_vs_baseline",
                "delta_clean_success_pct_vs_baseline",
            ]
        ],
        on="darkness_gain",
        how="inner",
    )
    delta_display["darkness_gain"] = delta_display["darkness_gain"].map(_format_gain)
    delta_display["delta_mean_abs_distance_error_m_vs_baseline"] = delta_display[
        "delta_mean_abs_distance_error_m_vs_baseline"
    ].map(lambda value: _format_float(value, decimals=4))
    delta_display["delta_mean_abs_orientation_error_deg_vs_baseline"] = delta_display[
        "delta_mean_abs_orientation_error_deg_vs_baseline"
    ].map(lambda value: _format_float(value, decimals=2))
    for column in [
        "delta_within_10cm_pct_vs_baseline",
        "delta_within_5deg_pct_vs_baseline",
        "delta_joint_success_pct_vs_baseline",
        "delta_clean_success_pct_vs_baseline",
    ]:
        delta_display[column] = delta_display[column].map(lambda value: _format_float(value, decimals=1))

    failure_display = failure_rates_df[
        [
            "darkness_gain",
            "sample_count",
            "within_10cm_pct",
            "within_5deg_pct",
            "joint_success_pct",
            "clean_success_pct",
            "distance_only_failure_pct",
            "yaw_only_failure_pct",
            "joint_failure_pct",
        ]
    ].copy()
    failure_display["darkness_gain"] = failure_display["darkness_gain"].map(_format_gain)
    failure_display["sample_count"] = failure_display["sample_count"].map(_format_int)
    for column in [
        "within_10cm_pct",
        "within_5deg_pct",
        "joint_success_pct",
        "clean_success_pct",
        "distance_only_failure_pct",
        "yaw_only_failure_pct",
        "joint_failure_pct",
    ]:
        failure_display[column] = failure_display[column].map(_format_pct)

    highlighted_subgroups = subgroup_summary_df.sort_values(
        ["delta_mean_abs_orientation_error_deg_1_4_vs_baseline", "delta_mean_abs_distance_error_m_1_4_vs_baseline"],
        ascending=[False, False],
        kind="stable",
    ).head(8).copy()
    if not highlighted_subgroups.empty:
        highlighted_subgroups = highlighted_subgroups[
            [
                "subgroup_family",
                "subgroup",
                "sample_count",
                "baseline_mean_abs_distance_error_m",
                "baseline_mean_abs_orientation_error_deg",
                "gain_1_4_mean_abs_distance_error_m",
                "gain_1_4_mean_abs_orientation_error_deg",
                "delta_mean_abs_distance_error_m_1_4_vs_baseline",
                "delta_mean_abs_orientation_error_deg_1_4_vs_baseline",
            ]
        ]
        highlighted_subgroups["sample_count"] = highlighted_subgroups["sample_count"].map(_format_int)
        for column in highlighted_subgroups.columns:
            if column in {"subgroup_family", "subgroup", "sample_count"}:
                continue
            decimals = 4 if "distance" in column else 2
            highlighted_subgroups[column] = highlighted_subgroups[column].map(lambda value, d=decimals: _format_float(value, decimals=d))

    top_sensitive_display = top_sensitive_df[
        [
            "sample_id",
            "image_filename",
            "truth_distance_m",
            "truth_orientation_deg",
            "baseline_abs_distance_error_m",
            "baseline_abs_orientation_error_deg",
            "worst_darkness_gain",
            "max_abs_distance_shift_m",
            "max_abs_orientation_shift_deg",
            "baseline_vehicle_pixel_fraction",
            "baseline_vehicle_mean_darkness",
        ]
    ].head(10).copy()
    for column in top_sensitive_display.columns:
        if column in {"sample_id", "image_filename"}:
            continue
        decimals = 4 if "distance" in column or "fraction" in column else 2
        if "gain" in column:
            top_sensitive_display[column] = top_sensitive_display[column].map(_format_gain)
        else:
            top_sensitive_display[column] = top_sensitive_display[column].map(lambda value, d=decimals: _format_float(value, decimals=d))

    catastrophic_display = catastrophic_df[
        [
            "sample_id",
            "image_filename",
            "truth_distance_m",
            "truth_orientation_deg",
            "baseline_abs_distance_error_m",
            "baseline_abs_orientation_error_deg",
            "worst_darkness_gain",
            "max_abs_distance_shift_m",
            "max_abs_orientation_shift_deg",
            "baseline_vehicle_pixel_fraction",
            "baseline_vehicle_mean_darkness",
            "baseline_vehicle_mean_intensity",
            "baseline_vehicle_std_intensity",
            "baseline_performance_bucket",
            "distance_bucket",
            "roi_size_bucket",
            "baseline_darkness_bucket",
            "frame_position_bucket",
        ]
    ].head(10).copy()
    for column in catastrophic_display.columns:
        if column in {
            "sample_id",
            "image_filename",
            "baseline_performance_bucket",
            "distance_bucket",
            "roi_size_bucket",
            "baseline_darkness_bucket",
            "frame_position_bucket",
        }:
            continue
        decimals = 4 if "distance" in column or "fraction" in column else 2
        if "gain" in column:
            catastrophic_display[column] = catastrophic_display[column].map(_format_gain)
        else:
            catastrophic_display[column] = catastrophic_display[column].map(lambda value, d=decimals: _format_float(value, decimals=d))

    catastrophic_by_gain_display = catastrophic_by_gain_df.copy()
    if not catastrophic_by_gain_display.empty:
        catastrophic_by_gain_display["darkness_gain"] = catastrophic_by_gain_display["darkness_gain"].map(_format_gain)
        catastrophic_by_gain_display["catastrophic_gt_90deg_count"] = catastrophic_by_gain_display["catastrophic_gt_90deg_count"].map(_format_int)
        catastrophic_by_gain_display["catastrophic_gt_120deg_count"] = catastrophic_by_gain_display["catastrophic_gt_120deg_count"].map(_format_int)

    concentration_display = catastrophic_concentration_df.head(8).copy()
    if not concentration_display.empty:
        concentration_display["sample_count"] = concentration_display["sample_count"].map(_format_int)
        concentration_display["catastrophic_count"] = concentration_display["catastrophic_count"].map(_format_int)
        concentration_display["catastrophic_rate_pct"] = concentration_display["catastrophic_rate_pct"].map(_format_pct)

    top_corr_lines: list[str] = []
    metric_display_names = {
        "baseline_abs_distance_error_m": "Baseline absolute distance error",
        "baseline_abs_orientation_error_deg": "Baseline absolute orientation error",
        "max_abs_distance_shift_m": "Max absolute distance shift",
        "distance_shift_slope_m_per_gain": "Distance shift slope",
        "max_abs_orientation_shift_deg": "Max absolute orientation shift",
        "orientation_shift_slope_deg_per_gain": "Orientation shift slope",
    }
    for metric in CORRELATION_METRICS:
        top_rows = top_correlations.get(metric, [])
        if not top_rows:
            continue
        best = top_rows[0]
        top_corr_lines.append(
            f"- {metric_display_names[metric]}: {_correlation_interpretation_text(best['feature'], float(best['pearson_r']))}"
        )

    asymmetry_lines: list[str] = []
    if not asymmetry_df.empty:
        for _, row in asymmetry_df.iterrows():
            asymmetry_lines.append(
                f"- Paired comparison {row['pair_label']}: darkening increased mean orientation error by {_format_float(row['dark_delta_mean_abs_orientation_error_deg'], decimals=2)} deg from baseline versus {_format_float(row['bright_delta_mean_abs_orientation_error_deg'], decimals=2)} deg on the brightening side, and changed joint success by {_format_float(row['dark_joint_success_delta_pct_points'], decimals=1)} vs {_format_float(row['bright_joint_success_delta_pct_points'], decimals=1)} percentage points."
            )

    sensitivity_lines: list[str] = []
    for label, sensitivity in [("1.2", sensitivity_1_2), ("1.4", sensitivity_1_4)]:
        if sensitivity is None:
            continue
        sensitivity_lines.append(
            f"- From gain 1.0 to {label}, mean absolute distance error changed by {_format_float(sensitivity['distance_mae_delta_m'], decimals=4)} m ({_format_pct(sensitivity['distance_mae_relative_pct'], decimals=1)} relative), while mean absolute orientation error changed by {_format_float(sensitivity['orientation_mae_delta_deg'], decimals=2)} deg ({_format_pct(sensitivity['orientation_mae_relative_pct'], decimals=1)} relative)."
        )
        sensitivity_lines.append(
            f"- Over the same shift, within-10 cm changed by {_format_float(sensitivity['within_10cm_delta_pct_points'], decimals=1)} percentage points, within-5 deg changed by {_format_float(sensitivity['within_5deg_delta_pct_points'], decimals=1)} points, joint success changed by {_format_float(sensitivity['joint_success_delta_pct_points'], decimals=1)} points, and clean success changed by {_format_float(sensitivity['clean_success_delta_pct_points'], decimals=1)} points."
        )

    report_lines = [
        f"# {scope_text['title']}",
        "",
        "## Scope",
        "",
        scope_text["scope"],
        "",
        "The perturbation is the saved `darkness_gain` applied inside the vehicle silhouette in the model-space ROI tensor. Larger gains therefore correspond to darker vehicle appearance inside that processed tensor. The results support statements about sensitivity to brightness-linked photometric changes in the deployed stack; they do not, by themselves, identify a single causal mechanism.",
        "",
        "Operational definitions used throughout:",
        "",
        f"- Distance success: absolute distance error <= {DISTANCE_SUCCESS_M:.2f} m",
        f"- Orientation success: absolute orientation error <= {ORIENTATION_SUCCESS_DEG:.0f} deg",
        "- Joint success: both pass",
        "- Distance-only failure: distance fails, yaw passes",
        "- Yaw-only failure: distance passes, yaw fails",
        "- Joint failure: both fail",
        f"- Clean success: absolute distance error <= {CLEAN_DISTANCE_SUCCESS_M:.2f} m and absolute orientation error <= {CLEAN_ORIENTATION_SUCCESS_DEG:.1f} deg",
        "",
        "## Run Validation",
        "",
        f"- Created timestamp: `{run_metadata['created_local']}`",
        f"- Distance model: `{run_metadata['selected_model_label']}`",
        f"- Distance model run dir: `{run_metadata['selected_model_run_dir']}`",
        f"- ROI-FCN model: `{run_metadata['selected_roi_model_label']}`",
        f"- ROI-FCN run dir: `{run_metadata['selected_roi_model_run_dir']}`",
        f"- Corpus: `{run_metadata['corpus_name']}`",
        f"- Selection mode: `{run_metadata['selection_mode']}`",
        f"- Requested sample count: {_format_int(run_metadata['requested_sample_count'])}",
        f"- Actual sample count: {_format_int(run_metadata['actual_sample_count'])}",
        f"- Prediction row count: {_format_int(run_metadata['prediction_row_count'])}",
        f"- Darkness gains analysed: `{', '.join(run_metadata['darkness_gains_display'])}`",
        f"- Corpus sample count from manifest: {_format_int(run_metadata['corpus_sample_count'])}",
        f"- Validation coverage: {_format_pct(run_metadata['validation_coverage_pct'], decimals=3)}",
        f"- Effective full validation run: `{run_metadata['is_effectively_full_validation']}`",
        "",
        scope_text["validation_note"],
        "",
        "## Aggregate Performance By Gain",
        "",
        _markdown_table(
            gain_summary_display,
            list(gain_summary_display.columns),
        ),
        "",
        "Delta relative to baseline gain 1.0:",
        "",
        _markdown_table(
            delta_display,
            list(delta_display.columns),
        ),
        "",
        "## Thresholded Operational Results By Gain",
        "",
        _markdown_table(
            failure_display,
            list(failure_display.columns),
        ),
        "",
        "## Baseline Characterisation",
        "",
        f"- Baseline mean absolute distance error: {_format_float(baseline_gain_row['mean_abs_distance_error_m'], decimals=4)} m",
        f"- Baseline mean absolute orientation error: {_format_float(baseline_gain_row['mean_abs_orientation_error_deg'], decimals=2)} deg",
        f"- Baseline within 10 cm: {_format_pct(baseline_rates_row['within_10cm_pct'])}",
        f"- Baseline within 5 deg: {_format_pct(baseline_rates_row['within_5deg_pct'])}",
        f"- Baseline joint success: {_format_pct(baseline_rates_row['joint_success_pct'])}",
        f"- Baseline clean success: {_format_pct(baseline_rates_row['clean_success_pct'])}",
        f"- Baseline yaw-only failure: {_format_pct(baseline_rates_row['yaw_only_failure_pct'])}",
        "",
        assessment_text["distance_assessment"],
        "",
        assessment_text["yaw_assessment"],
        "",
        assessment_text["brittleness_assessment"],
        "",
        "## Sensitivity Relative To Baseline",
        "",
        *sensitivity_lines,
        "",
        "## Monotonicity And Brightening vs Darkening",
        "",
        f"- Distance MAE monotonic under darkening: `{monotonicity['distance_error_monotonic_non_decreasing_for_darkening']}`",
        f"- Orientation MAE monotonic under darkening: `{monotonicity['orientation_error_monotonic_non_decreasing_for_darkening']}`",
        f"- Within-10 cm monotonic under darkening: `{monotonicity['distance_success_monotonic_non_increasing_for_darkening']}`",
        f"- Within-5 deg monotonic under darkening: `{monotonicity['yaw_success_monotonic_non_increasing_for_darkening']}`",
        f"- Joint success monotonic under darkening: `{monotonicity['joint_success_monotonic_non_increasing_for_darkening']}`",
        f"- Clean success monotonic under darkening: `{monotonicity['clean_success_monotonic_non_increasing_for_darkening']}`",
        f"- Brightening effect on distance: `{monotonicity['brightening_effect_distance']}`",
        f"- Brightening effect on yaw: `{monotonicity['brightening_effect_yaw']}`",
        f"- Brightening effect on joint performance: `{monotonicity['brightening_effect_joint']}`",
        "",
        *asymmetry_lines,
        "",
        "## Correlation Findings",
        "",
        "The correlation analysis used the saved baseline appearance statistics already present in the brightness-analysis JSON. The emphasis below is on the strongest associations, not on exhaustive table dumping.",
        "",
        *top_corr_lines,
        "",
        "Interpretation should remain cautious. Correlation here indicates association between baseline appearance characteristics and either baseline error or sensitivity to perturbation. It does not isolate causality, and some appearance variables may proxy distance, silhouette scale, or other latent geometry.",
        "",
        "## Subgroup Findings",
        "",
        "Subgroups were constructed as follows:",
        "",
        "- Baseline performance buckets: baseline joint success, baseline distance-only failure, baseline yaw-only failure, baseline joint failure",
        "- Distance buckets: tertiles of truth distance (`near`, `mid`, `far`)",
        "- ROI size buckets: tertiles of baseline vehicle pixel fraction (`small`, `medium`, `large`)",
        "- Baseline darkness buckets: tertiles of baseline vehicle mean darkness (`bright`, `medium`, `dark`); higher darkness means a darker rendered vehicle in the model-space ROI tensor",
        f"- Edge-of-frame rule: ROI center within the outer {int(EDGE_BAND_FRACTION * 100)}% band of image width or height, using the matched inference-output ROI center and corpus frame size",
        "",
        _markdown_table(
            highlighted_subgroups,
            list(highlighted_subgroups.columns),
        ) if not highlighted_subgroups.empty else "No subgroup summary rows were available.",
        "",
        "The full subgroup table is provided separately in `analysis/brightness_subgroup_summary.csv`.",
        "",
        "## Catastrophic Yaw Shift Analysis",
        "",
        f"- Samples with any absolute orientation shift > 90 deg: {_format_int(len(catastrophic_df))}",
        f"- Samples with any absolute orientation shift > 120 deg: {_format_int(int(catastrophic_df['strict_gt_120deg'].sum()) if not catastrophic_df.empty else 0)}",
        "",
        _markdown_table(
            catastrophic_by_gain_display,
            list(catastrophic_by_gain_display.columns),
        ) if not catastrophic_by_gain_display.empty else "No catastrophic by-gain rows were available.",
        "",
        "Top catastrophic cases:",
        "",
        _markdown_table(
            catastrophic_display,
            list(catastrophic_display.columns),
        ) if not catastrophic_display.empty else "No catastrophic yaw flips above 90 deg were observed.",
        "",
        "Subgroups with the highest catastrophic-yaw concentration:",
        "",
        _markdown_table(
            concentration_display,
            list(concentration_display.columns),
        ) if not concentration_display.empty else "No catastrophic subgroup concentration summary was available.",
        "",
        "## Distance vs Yaw",
        "",
        *[f"- {conclusion}" for conclusion in brightness_conclusions],
        "",
        "## Implications",
        "",
        *[f"- {recommendation}" for recommendation in recommendations],
        "",
        "## Data Quality And Limitations",
        "",
        f"- {scope_text['limitations_note']}",
        "- The perturbation is applied inside the processed ROI tensor, not at raw-camera capture time. The results therefore combine photometric effects with any representation-specific vulnerability introduced by preprocessing and the shared regressor.",
        "- The edge-of-frame subgroup uses ROI center coordinates from the matched inference output. The rule is explicit, but it remains a coarse geometry descriptor rather than a full detection-quality analysis.",
        *[f"- {note}" for note in quality_notes],
        "",
        "## Sensitive Samples",
        "",
        _markdown_table(
            top_sensitive_display,
            list(top_sensitive_display.columns),
        ) if not top_sensitive_display.empty else "No sensitive sample rows were available.",
        "",
        "The complete sensitive-sample and catastrophic-yaw tables are provided separately in CSV form.",
    ]
    return "\n".join(report_lines)


def _build_summary_json(
    *,
    run_metadata: dict[str, Any],
    gain_summary_df: pd.DataFrame,
    failure_rates_df: pd.DataFrame,
    monotonicity: dict[str, Any],
    asymmetry_df: pd.DataFrame,
    top_correlations: dict[str, list[dict[str, Any]]],
    subgroup_summary_df: pd.DataFrame,
    catastrophic_df: pd.DataFrame,
    catastrophic_by_gain_df: pd.DataFrame,
    catastrophic_concentration_df: pd.DataFrame,
    quality_notes: list[str],
    assessment_text: dict[str, str],
    brightness_conclusions: list[str],
    recommendations: list[str],
    sensitivity_1_2: dict[str, Any] | None,
    sensitivity_1_4: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_gain_row = gain_summary_df.loc[np.isclose(gain_summary_df["darkness_gain"], BASELINE_GAIN)].iloc[0]
    baseline_rates_row = failure_rates_df.loc[np.isclose(failure_rates_df["darkness_gain"], BASELINE_GAIN)].iloc[0]
    subgroup_highlights = subgroup_summary_df.sort_values(
        ["delta_mean_abs_orientation_error_deg_1_4_vs_baseline", "delta_mean_abs_distance_error_m_1_4_vs_baseline"],
        ascending=[False, False],
        kind="stable",
    ).head(10)
    return {
        "run_metadata": run_metadata,
        "baseline": {
            "mean_abs_distance_error_m": float(baseline_gain_row["mean_abs_distance_error_m"]),
            "mean_abs_orientation_error_deg": float(baseline_gain_row["mean_abs_orientation_error_deg"]),
            "within_10cm_pct": float(baseline_rates_row["within_10cm_pct"]),
            "within_5deg_pct": float(baseline_rates_row["within_5deg_pct"]),
            "joint_success_pct": float(baseline_rates_row["joint_success_pct"]),
            "clean_success_pct": float(baseline_rates_row["clean_success_pct"]),
            "distance_only_failure_pct": float(baseline_rates_row["distance_only_failure_pct"]),
            "yaw_only_failure_pct": float(baseline_rates_row["yaw_only_failure_pct"]),
            "joint_failure_pct": float(baseline_rates_row["joint_failure_pct"]),
            **assessment_text,
        },
        "gain_summary": gain_summary_df.to_dict(orient="records"),
        "failure_rates": failure_rates_df.to_dict(orient="records"),
        "monotonicity": monotonicity,
        "asymmetry_pairs": asymmetry_df.to_dict(orient="records"),
        "sensitivity_from_baseline": {
            "gain_1_2": sensitivity_1_2,
            "gain_1_4": sensitivity_1_4,
        },
        "top_correlations_by_metric": top_correlations,
        "subgroup_highlights": subgroup_highlights.to_dict(orient="records"),
        "catastrophic_yaw": {
            "samples_with_any_shift_gt_90deg": int(len(catastrophic_df)),
            "samples_with_any_shift_gt_120deg": int(catastrophic_df["strict_gt_120deg"].sum()) if not catastrophic_df.empty else 0,
            "counts_by_gain": catastrophic_by_gain_df.to_dict(orient="records"),
            "top_subgroup_concentrations": catastrophic_concentration_df.head(10).to_dict(orient="records"),
        },
        "conclusions": brightness_conclusions,
        "recommendations": recommendations,
        "quality_notes": quality_notes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("brightness_json", type=Path, help="Path to the saved brightness-analysis JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory for report and table outputs",
    )
    parser.add_argument(
        "--report-name",
        default="brightness_2048_report.md",
        help="Markdown report filename",
    )
    parser.add_argument(
        "--summary-name",
        default="brightness_2048_summary.json",
        help="Machine-readable summary filename",
    )
    parser.add_argument(
        "--inference-output",
        type=Path,
        default=None,
        help="Optional matching inference-output JSON for edge-of-frame subgroup analysis",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    brightness_json_path = args.brightness_json.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = RunArtifacts(
        brightness_json_path=brightness_json_path,
        output_dir=output_dir,
        report_path=output_dir / args.report_name,
        summary_path=output_dir / args.summary_name,
        gain_summary_csv_path=output_dir / "brightness_gain_summary.csv",
        failure_rates_csv_path=output_dir / "brightness_failure_rates.csv",
        subgroup_summary_csv_path=output_dir / "brightness_subgroup_summary.csv",
        top_sensitive_csv_path=output_dir / "brightness_top_sensitive_samples.csv",
        top_yaw_flips_csv_path=output_dir / "brightness_top_yaw_flips.csv",
    )

    raw_data = _load_json(brightness_json_path)
    if not isinstance(raw_data, dict):
        raise ValueError("Brightness-analysis JSON must be a top-level object.")

    predictions_df = _records_to_df(raw_data.get("predictions"), name="predictions")
    per_sample_df = _records_to_df(raw_data.get("per_sample_summary"), name="per_sample_summary")

    _ensure_columns(
        predictions_df,
        [
            "sample_id",
            "image_filename",
            "darkness_gain",
            "abs_distance_error_m",
            "abs_orientation_error_deg",
            "abs_distance_shift_from_baseline_m",
            "abs_orientation_shift_from_baseline_deg",
            "variant_vehicle_mean_darkness",
            "truth_distance_m",
            "truth_orientation_deg",
        ],
        name="predictions",
    )
    _ensure_columns(
        per_sample_df,
        [
            "sample_id",
            "image_filename",
            "truth_distance_m",
            "truth_orientation_deg",
            "baseline_abs_distance_error_m",
            "baseline_abs_orientation_error_deg",
            "baseline_vehicle_mean_darkness",
            "baseline_vehicle_mean_intensity",
            "baseline_vehicle_std_intensity",
            "baseline_canvas_mean_intensity",
            "baseline_canvas_std_intensity",
            "baseline_vehicle_pixel_fraction",
            "max_abs_distance_shift_m",
            "distance_shift_slope_m_per_gain",
            "max_abs_orientation_shift_deg",
            "orientation_shift_slope_deg_per_gain",
        ],
        name="per_sample_summary",
    )

    predictions_df = _add_operational_flags(predictions_df)
    baseline_predictions = predictions_df.loc[np.isclose(predictions_df["darkness_gain"], BASELINE_GAIN)].copy()
    if baseline_predictions.empty:
        raise ValueError("Brightness-analysis JSON did not contain a baseline gain of 1.0.")

    corpus_root = Path(str(raw_data["selected_corpus"]["root"]))
    samples_csv_path = _corpus_samples_csv_from_selected_root(corpus_root)
    corpus_sample_count = _sample_count_from_csv(samples_csv_path)

    inference_output_path = args.inference_output.resolve() if args.inference_output else _discover_inference_output_path(brightness_json_path)
    inference_join_df, inference_notes = _load_inference_join(
        inference_output_path,
        fallback_corpus_root=corpus_root,
    )
    sample_df = _build_bucketed_sample_frame(
        per_sample_df,
        baseline_predictions,
        inference_join_df=inference_join_df,
    )
    edge_of_frame_match_count = int(sample_df["frame_position_bucket"].notna().sum())
    if inference_join_df is not None and edge_of_frame_match_count < len(sample_df):
        inference_notes.append(
            "Edge-of-frame subgroup labels were available for "
            f"{edge_of_frame_match_count:,} of {len(sample_df):,} samples because the matched inference-output "
            "JSON contained geometry for only that subset."
        )

    gain_summary_df = _compute_gain_summary(predictions_df)
    failure_rates_df = _compute_failure_rates(predictions_df)
    monotonicity, asymmetry_df = _compute_monotonicity_and_asymmetry(gain_summary_df, failure_rates_df)
    sensitivity_1_2 = _compute_sensitivity_delta(
        gain_summary_df,
        failure_rates_df,
        target_gain=INTERMEDIATE_DARKENING_GAIN,
    )
    sensitivity_1_4 = _compute_sensitivity_delta(
        gain_summary_df,
        failure_rates_df,
        target_gain=MAX_DARKENING_GAIN,
    )
    correlations_df = _compute_correlations(per_sample_df)
    top_correlations = _top_correlations_by_metric(correlations_df)
    subgroup_summary_df = _compute_subgroup_summary(sample_df, predictions_df)
    top_sensitive_df, catastrophic_df, catastrophic_by_gain_df = _compute_sensitive_tables(sample_df, predictions_df)
    catastrophic_concentration_df = _compute_catastrophic_subgroup_concentration(sample_df, catastrophic_df)

    baseline_gain_row = gain_summary_df.loc[np.isclose(gain_summary_df["darkness_gain"], BASELINE_GAIN)].iloc[0]
    baseline_rates_row = failure_rates_df.loc[np.isclose(failure_rates_df["darkness_gain"], BASELINE_GAIN)].iloc[0]
    assessment_text = _baseline_assessment_text(baseline_rates_row, baseline_gain_row)
    brightness_conclusions = _brightness_sensitivity_conclusion(
        sensitivity_1_2,
        sensitivity_1_4,
        monotonicity,
        asymmetry_df,
    )
    recommendations = _recommendations_text(baseline_rates_row, sensitivity_1_4)

    quality_notes = inference_notes + _quality_checks(raw_data, gain_summary_df, correlations_df)
    run_metadata = {
        "brightness_json_path": str(brightness_json_path),
        "created_local": str(raw_data.get("created_local")),
        "selected_model_label": str(raw_data["selected_model"]["label"]),
        "selected_model_run_dir": str(raw_data["selected_model"]["run_dir"]),
        "selected_roi_model_label": str(raw_data["selected_roi_model"]["label"]),
        "selected_roi_model_run_dir": str(raw_data["selected_roi_model"]["run_dir"]),
        "corpus_name": str(raw_data["selected_corpus"]["name"]),
        "corpus_root": str(raw_data["selected_corpus"]["root"]),
        "selection_mode": str(raw_data["selection"]["mode"]),
        "selection_offset": int(raw_data["selection"].get("offset", 0)),
        "requested_sample_count": int(raw_data["selection"]["requested_samples"]),
        "actual_sample_count": int(per_sample_df["sample_id"].nunique()),
        "prediction_row_count": int(len(predictions_df)),
        "darkness_gains": [float(value) for value in raw_data["darkness_gains"]],
        "darkness_gains_display": [_format_gain(value) for value in raw_data["darkness_gains"]],
        "corpus_sample_count": corpus_sample_count,
        "validation_coverage_pct": _coverage_pct(
            int(per_sample_df["sample_id"].nunique()),
            corpus_sample_count,
        ),
        "is_effectively_full_validation": bool(
            corpus_sample_count is not None and int(per_sample_df["sample_id"].nunique()) == int(corpus_sample_count)
        ),
        "is_near_full_validation": bool(
            corpus_sample_count is not None
            and raw_data["selection"].get("mode") == "slice"
            and int(raw_data["selection"].get("offset", 0)) > 0
            and int(per_sample_df["sample_id"].nunique()) + int(raw_data["selection"].get("offset", 0))
            == int(corpus_sample_count)
        ),
        "edge_of_frame_matched_sample_count": edge_of_frame_match_count,
        "matched_inference_output_path": str(inference_output_path) if inference_output_path else None,
        "edge_of_frame_rule": f"ROI center within the outer {int(EDGE_BAND_FRACTION * 100)}% band of image width or height",
    }

    report_text = _prepare_report(
        run_metadata=run_metadata,
        gain_summary_df=gain_summary_df,
        failure_rates_df=failure_rates_df,
        monotonicity=monotonicity,
        asymmetry_df=asymmetry_df,
        top_correlations=top_correlations,
        subgroup_summary_df=subgroup_summary_df,
        top_sensitive_df=top_sensitive_df,
        catastrophic_df=catastrophic_df,
        catastrophic_by_gain_df=catastrophic_by_gain_df,
        catastrophic_concentration_df=catastrophic_concentration_df,
        quality_notes=quality_notes,
        assessment_text=assessment_text,
        brightness_conclusions=brightness_conclusions,
        recommendations=recommendations,
        sensitivity_1_2=sensitivity_1_2,
        sensitivity_1_4=sensitivity_1_4,
    )
    summary_json = _build_summary_json(
        run_metadata=run_metadata,
        gain_summary_df=gain_summary_df,
        failure_rates_df=failure_rates_df,
        monotonicity=monotonicity,
        asymmetry_df=asymmetry_df,
        top_correlations=top_correlations,
        subgroup_summary_df=subgroup_summary_df,
        catastrophic_df=catastrophic_df,
        catastrophic_by_gain_df=catastrophic_by_gain_df,
        catastrophic_concentration_df=catastrophic_concentration_df,
        quality_notes=quality_notes,
        assessment_text=assessment_text,
        brightness_conclusions=brightness_conclusions,
        recommendations=recommendations,
        sensitivity_1_2=sensitivity_1_2,
        sensitivity_1_4=sensitivity_1_4,
    )

    artifacts.report_path.write_text(report_text, encoding="utf-8")
    artifacts.summary_path.write_text(json.dumps(_json_ready(summary_json), indent=2), encoding="utf-8")
    gain_summary_df.to_csv(artifacts.gain_summary_csv_path, index=False)
    failure_rates_df.to_csv(artifacts.failure_rates_csv_path, index=False)
    subgroup_summary_df.to_csv(artifacts.subgroup_summary_csv_path, index=False)
    top_sensitive_df.to_csv(artifacts.top_sensitive_csv_path, index=False)
    catastrophic_df.to_csv(artifacts.top_yaw_flips_csv_path, index=False)

    print(f"Wrote report: {artifacts.report_path}")
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote CSVs to: {artifacts.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
