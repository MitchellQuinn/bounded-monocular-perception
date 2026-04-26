"""Brightness-sensitivity diagnostics for the v0.3 distance-orientation regressor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .external import ensure_external_paths

ensure_external_paths()

from src.data import Batch

from .discovery import RawCorpus, load_corpus_samples
from .pipeline import (
    _resolve_raw_corpus,
    _emit_inference_startup_log,
    _run_prediction_batch,
    _signed_orientation_delta_deg,
    load_model_context,
    load_roi_fcn_model_context,
    preprocess_single_sample,
)


@dataclass(frozen=True)
class BrightnessSensitivityResult:
    """Tabular outputs from a controlled brightness-sensitivity sweep."""

    model_label: str
    roi_model_label: str
    corpus_name: str
    darkness_gains: tuple[float, ...]
    predictions: pd.DataFrame
    per_sample_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    brightness_correlations: pd.DataFrame


def apply_vehicle_darkness_gain(
    model_image: np.ndarray,
    gain: float,
    *,
    vehicle_threshold: float = 0.999,
) -> np.ndarray:
    """Scale darkness inside the vehicle silhouette while preserving white background.

    The packed model input stores inverted vehicle detail on white. In practice,
    a larger darkness gain makes model-space vehicle pixels darker, which roughly
    corresponds to a brighter raw grayscale source before inversion.
    """
    gain_value = float(gain)
    if gain_value <= 0.0:
        raise ValueError(f"gain must be > 0, got {gain_value}")

    image_2d, restore = _coerce_model_image(model_image)
    adjusted = image_2d.copy()
    vehicle_mask = adjusted < float(vehicle_threshold)
    if not np.any(vehicle_mask):
        return restore(adjusted)

    darkness = 1.0 - adjusted[vehicle_mask]
    adjusted[vehicle_mask] = 1.0 - np.clip(darkness * gain_value, 0.0, 1.0)
    return restore(adjusted.astype(np.float32, copy=False))


def run_brightness_sensitivity_analysis(
    model_run_dir: str | Path,
    corpus_dir: str | Path,
    *,
    roi_model_run_dir: str | Path | None = None,
    image_names: str | Iterable[str] | None = None,
    offset: int = 0,
    num_samples: int = 32,
    darkness_gains: Sequence[float] = (0.6, 0.8, 1.0, 1.2, 1.4),
    batch_size: int = 64,
    progress_callback: Callable[[int, int], None] | None = None,
    vehicle_threshold: float = 0.999,
    device: str | None = None,
    log_sink: Callable[[str], None] | None = None,
) -> BrightnessSensitivityResult:
    """Measure how prediction quality and stability change under controlled intensity shifts.

    The perturbation is applied after preprocessing and only affects the regressor's
    grayscale ROI tensor. This isolates sensitivity in the distance-orientation
    model from upstream ROI-FCN or silhouette-generation effects. Analysis is
    streamed through the model in smaller batches so large corpus slices do not
    need to fit on the GPU at once.
    """
    if roi_model_run_dir is None:
        raise ValueError("roi_model_run_dir is required for v0.3 brightness analysis.")

    corpus = _resolve_analysis_corpus(corpus_dir)
    selected_rows = _select_analysis_rows(
        corpus=corpus,
        image_names=image_names,
        offset=offset,
        num_samples=num_samples,
    )
    batch_size_value = int(batch_size)
    if batch_size_value <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size_value}")
    total_selected = int(len(selected_rows))

    normalized_gains = _normalize_darkness_gains(darkness_gains)
    model, model_context = load_model_context(model_run_dir, device=device)
    roi_model, roi_model_context = load_roi_fcn_model_context(roi_model_run_dir, device=device)
    _emit_inference_startup_log(
        model_context=model_context,
        roi_model_context=roi_model_context,
        log_sink=log_sink,
    )

    prediction_records: list[dict[str, object]] = []
    for start_index, chunk_rows in _iter_selected_row_chunks(selected_rows, chunk_size=batch_size_value):
        preprocessed_samples = [
            preprocess_single_sample(
                corpus=corpus,
                sample_row=sample_row,
                model_context=model_context,
                roi_model=roi_model,
                roi_model_context=roi_model_context,
            )
            for _, sample_row in chunk_rows.iterrows()
        ]
        if not preprocessed_samples:
            continue

        baseline_batch = _build_variant_batch(
            preprocessed_samples=preprocessed_samples,
            variant_images=[sample.model_image for sample in preprocessed_samples],
        )
        baseline_predictions = _run_prediction_batch(
            model=model,
            batch=baseline_batch,
            model_context=model_context,
        )

        baseline_by_index: dict[int, dict[str, float]] = {}
        for local_index, (sample, prediction_row) in enumerate(
            zip(preprocessed_samples, baseline_predictions.itertuples(index=False), strict=True)
        ):
            sample_index = start_index + local_index
            baseline_distance = float(prediction_row.prediction_distance_m)
            baseline_orientation = float(prediction_row.prediction_yaw_deg)
            baseline_by_index[sample_index] = {
                "prediction_distance_m": baseline_distance,
                "prediction_yaw_deg": baseline_orientation,
            }
            prediction_records.append(
                _prediction_record(
                    sample_index=sample_index,
                    sample=sample,
                    gain=1.0,
                    variant_image=sample.model_image,
                    prediction_row=prediction_row,
                    baseline_distance_m=baseline_distance,
                    baseline_orientation_deg=baseline_orientation,
                    vehicle_threshold=vehicle_threshold,
                )
            )

        for gain in normalized_gains:
            if np.isclose(gain, 1.0):
                continue
            variant_images = [
                apply_vehicle_darkness_gain(
                    sample.model_image,
                    gain,
                    vehicle_threshold=vehicle_threshold,
                )
                for sample in preprocessed_samples
            ]
            variant_batch = _build_variant_batch(
                preprocessed_samples=preprocessed_samples,
                variant_images=variant_images,
            )
            variant_predictions = _run_prediction_batch(
                model=model,
                batch=variant_batch,
                model_context=model_context,
            )
            for local_index, (sample, variant_image, prediction_row) in enumerate(
                zip(
                    preprocessed_samples,
                    variant_images,
                    variant_predictions.itertuples(index=False),
                    strict=True,
                )
            ):
                sample_index = start_index + local_index
                baseline = baseline_by_index[sample_index]
                prediction_records.append(
                    _prediction_record(
                        sample_index=sample_index,
                        sample=sample,
                        gain=gain,
                        variant_image=variant_image,
                        prediction_row=prediction_row,
                        baseline_distance_m=float(baseline["prediction_distance_m"]),
                        baseline_orientation_deg=float(baseline["prediction_yaw_deg"]),
                        vehicle_threshold=vehicle_threshold,
                    )
                )

        if progress_callback is not None:
            progress_callback(start_index + len(preprocessed_samples), total_selected)

    if not prediction_records:
        raise RuntimeError("Brightness analysis did not produce any prediction records.")

    predictions_df = pd.DataFrame(prediction_records).sort_values(
        ["analysis_sample_index", "darkness_gain"],
        kind="stable",
    ).reset_index(drop=True)
    per_sample_summary = _summarize_per_sample(predictions_df)
    aggregate_summary = _summarize_aggregate(predictions_df)
    brightness_correlations = _summarize_brightness_correlations(per_sample_summary)

    return BrightnessSensitivityResult(
        model_label=model_context.label,
        roi_model_label=roi_model_context.label,
        corpus_name=corpus.name,
        darkness_gains=normalized_gains,
        predictions=predictions_df,
        per_sample_summary=per_sample_summary,
        aggregate_summary=aggregate_summary,
        brightness_correlations=brightness_correlations,
    )


def _resolve_analysis_corpus(corpus_dir: str | Path) -> RawCorpus:
    return _resolve_raw_corpus(corpus_dir)


def _normalize_darkness_gains(values: Sequence[float]) -> tuple[float, ...]:
    ordered: list[float] = [1.0]
    seen = {round(1.0, 8)}
    for raw in values:
        gain = float(raw)
        if gain <= 0.0:
            raise ValueError(f"darkness gains must be > 0, got {gain}")
        key = round(gain, 8)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(gain)
    return tuple(sorted(ordered))


def _coerce_model_image(
    model_image: np.ndarray,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    image = np.asarray(model_image, dtype=np.float32)
    if image.ndim == 2:
        return image, lambda gray: gray.astype(np.float32, copy=False)
    if image.ndim == 3 and image.shape[0] == 1:
        return image[0], lambda gray: gray[None, ...].astype(np.float32, copy=False)
    raise ValueError(
        "model_image must have shape (H, W) or (1, H, W); "
        f"got {tuple(image.shape)}"
    )


def _select_analysis_rows(
    *,
    corpus: RawCorpus,
    image_names: str | Iterable[str] | None,
    offset: int,
    num_samples: int,
) -> pd.DataFrame:
    samples_df = load_corpus_samples(corpus)
    if samples_df.empty:
        raise ValueError(f"Corpus {corpus.name} has no selectable samples.")

    if image_names is not None:
        selected_names = [str(image_names)] if isinstance(image_names, str) else [str(name) for name in image_names]
        if not selected_names:
            raise ValueError("image_names must not be empty when provided.")
        order = {name: index for index, name in enumerate(selected_names)}
        selected_df = samples_df.loc[samples_df["__image_name__"].isin(order)].copy()
        missing = [name for name in selected_names if name not in set(selected_df["__image_name__"].astype(str))]
        if missing:
            raise ValueError(
                f"Requested image_names were not found in corpus {corpus.name}: {missing}"
            )
        selected_df["__selection_order__"] = selected_df["__image_name__"].map(order)
        return selected_df.sort_values("__selection_order__", kind="stable").reset_index(drop=True)

    offset_value = int(offset)
    num_samples_value = int(num_samples)
    if offset_value < 0:
        raise ValueError(f"offset must be >= 0, got {offset_value}")
    if num_samples_value <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples_value}")
    if offset_value >= len(samples_df):
        raise ValueError(
            f"offset {offset_value} is out of range for corpus {corpus.name} "
            f"with {len(samples_df)} selectable samples"
        )
    selected_df = samples_df.iloc[offset_value : offset_value + num_samples_value].copy()
    if selected_df.empty:
        raise RuntimeError("Resolved an empty sample slice for brightness analysis.")
    return selected_df.reset_index(drop=True)


def _iter_selected_row_chunks(
    selected_rows: pd.DataFrame,
    *,
    chunk_size: int,
) -> Iterable[tuple[int, pd.DataFrame]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    total_rows = int(len(selected_rows))
    for start_index in range(0, total_rows, int(chunk_size)):
        yield start_index, selected_rows.iloc[start_index : start_index + int(chunk_size)].copy()


def _build_variant_batch(
    *,
    preprocessed_samples: Sequence,
    variant_images: Sequence[np.ndarray],
) -> Batch:
    if len(preprocessed_samples) != len(variant_images):
        raise ValueError(
            "Variant image count must match preprocessed sample count; "
            f"got {len(variant_images)} and {len(preprocessed_samples)}"
        )
    return Batch(
        images=np.stack(variant_images, axis=0).astype(np.float32),
        targets=np.asarray(
            [
                [
                    float(sample.actual_distance_m),
                    float(sample.actual_yaw_sin),
                    float(sample.actual_yaw_cos),
                ]
                for sample in preprocessed_samples
            ],
            dtype=np.float32,
        ),
        rows=[dict(sample.sample_row) for sample in preprocessed_samples],
        bbox_features=np.stack(
            [np.asarray(sample.bbox_features, dtype=np.float32) for sample in preprocessed_samples],
            axis=0,
        ).astype(np.float32),
    )


def _brightness_stats(
    model_image: np.ndarray,
    *,
    vehicle_threshold: float,
) -> dict[str, float]:
    gray, _ = _coerce_model_image(model_image)
    vehicle_mask = gray < float(vehicle_threshold)
    vehicle_pixels = gray[vehicle_mask]

    stats = {
        "variant_canvas_mean_intensity": float(np.mean(gray)),
        "variant_canvas_std_intensity": float(np.std(gray, ddof=0)),
        "variant_vehicle_pixel_fraction": float(np.mean(vehicle_mask)),
        "variant_vehicle_mean_intensity": float("nan"),
        "variant_vehicle_std_intensity": float("nan"),
        "variant_vehicle_mean_darkness": float("nan"),
    }
    if vehicle_pixels.size:
        stats.update(
            {
                "variant_vehicle_mean_intensity": float(np.mean(vehicle_pixels)),
                "variant_vehicle_std_intensity": float(np.std(vehicle_pixels, ddof=0)),
                "variant_vehicle_mean_darkness": float(np.mean(1.0 - vehicle_pixels)),
            }
        )
    return stats


def _prediction_record(
    *,
    sample_index: int,
    sample,
    gain: float,
    variant_image: np.ndarray,
    prediction_row,
    baseline_distance_m: float,
    baseline_orientation_deg: float,
    vehicle_threshold: float,
) -> dict[str, object]:
    predicted_distance = float(prediction_row.prediction_distance_m)
    truth_distance = float(prediction_row.truth_distance_m)
    predicted_orientation = float(prediction_row.prediction_yaw_deg)
    truth_orientation = float(prediction_row.truth_yaw_deg)
    distance_shift = float(predicted_distance - baseline_distance_m)
    orientation_shift = float(
        _signed_orientation_delta_deg(predicted_orientation, baseline_orientation_deg)
    )
    signed_orientation_error = float(
        _signed_orientation_delta_deg(predicted_orientation, truth_orientation)
    )

    record: dict[str, object] = {
        "analysis_sample_index": int(sample_index),
        "sample_id": str(sample.sample_row["sample_id"]),
        "image_filename": str(sample.sample_row["image_filename"]),
        "darkness_gain": float(gain),
        "truth_distance_m": truth_distance,
        "truth_orientation_deg": truth_orientation,
        "truth_yaw_sin": float(sample.actual_yaw_sin),
        "truth_yaw_cos": float(sample.actual_yaw_cos),
        "prediction_distance_m": predicted_distance,
        "prediction_orientation_deg": predicted_orientation,
        "distance_error_m": float(predicted_distance - truth_distance),
        "abs_distance_error_m": float(abs(predicted_distance - truth_distance)),
        "orientation_error_deg": signed_orientation_error,
        "abs_orientation_error_deg": float(abs(signed_orientation_error)),
        "baseline_prediction_distance_m": float(baseline_distance_m),
        "baseline_prediction_orientation_deg": float(baseline_orientation_deg),
        "distance_shift_from_baseline_m": distance_shift,
        "abs_distance_shift_from_baseline_m": float(abs(distance_shift)),
        "orientation_shift_from_baseline_deg": orientation_shift,
        "abs_orientation_shift_from_baseline_deg": float(abs(orientation_shift)),
    }
    record.update(_brightness_stats(variant_image, vehicle_threshold=vehicle_threshold))
    return record


def _linear_slope(x_values: pd.Series, y_values: pd.Series) -> float:
    x = pd.to_numeric(x_values, errors="coerce").to_numpy(dtype=np.float64)
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.unique(x).size < 2:
        return float("nan")
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def _p95(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 95))


def _summarize_per_sample(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = predictions_df.groupby("analysis_sample_index", sort=True, dropna=False)
    for sample_index, group in grouped:
        ordered = group.sort_values("darkness_gain", kind="stable").reset_index(drop=True)
        baseline = ordered.loc[ordered["darkness_gain"].sub(1.0).abs().idxmin()]
        rows.append(
            {
                "analysis_sample_index": int(sample_index),
                "sample_id": str(baseline["sample_id"]),
                "image_filename": str(baseline["image_filename"]),
                "gain_count": int(len(ordered)),
                "truth_distance_m": float(baseline["truth_distance_m"]),
                "truth_orientation_deg": float(baseline["truth_orientation_deg"]),
                "truth_yaw_sin": float(baseline["truth_yaw_sin"]),
                "truth_yaw_cos": float(baseline["truth_yaw_cos"]),
                "baseline_prediction_distance_m": float(baseline["prediction_distance_m"]),
                "baseline_prediction_orientation_deg": float(baseline["prediction_orientation_deg"]),
                "baseline_distance_error_m": float(baseline["distance_error_m"]),
                "baseline_abs_distance_error_m": float(baseline["abs_distance_error_m"]),
                "baseline_orientation_error_deg": float(baseline["orientation_error_deg"]),
                "baseline_abs_orientation_error_deg": float(baseline["abs_orientation_error_deg"]),
                "baseline_canvas_mean_intensity": float(baseline["variant_canvas_mean_intensity"]),
                "baseline_canvas_std_intensity": float(baseline["variant_canvas_std_intensity"]),
                "baseline_vehicle_pixel_fraction": float(baseline["variant_vehicle_pixel_fraction"]),
                "baseline_vehicle_mean_intensity": float(baseline["variant_vehicle_mean_intensity"]),
                "baseline_vehicle_std_intensity": float(baseline["variant_vehicle_std_intensity"]),
                "baseline_vehicle_mean_darkness": float(baseline["variant_vehicle_mean_darkness"]),
                "distance_prediction_range_m": float(
                    ordered["prediction_distance_m"].max() - ordered["prediction_distance_m"].min()
                ),
                "mean_abs_distance_shift_m": float(ordered["abs_distance_shift_from_baseline_m"].mean()),
                "max_abs_distance_shift_m": float(ordered["abs_distance_shift_from_baseline_m"].max()),
                "distance_shift_slope_m_per_gain": _linear_slope(
                    ordered["darkness_gain"],
                    ordered["distance_shift_from_baseline_m"],
                ),
                "mean_abs_orientation_shift_deg": float(
                    ordered["abs_orientation_shift_from_baseline_deg"].mean()
                ),
                "max_abs_orientation_shift_deg": float(
                    ordered["abs_orientation_shift_from_baseline_deg"].max()
                ),
                "orientation_shift_slope_deg_per_gain": _linear_slope(
                    ordered["darkness_gain"],
                    ordered["orientation_shift_from_baseline_deg"],
                ),
                "distance_error_range_m": float(
                    ordered["abs_distance_error_m"].max() - ordered["abs_distance_error_m"].min()
                ),
                "orientation_error_range_deg": float(
                    ordered["abs_orientation_error_deg"].max() - ordered["abs_orientation_error_deg"].min()
                ),
                "max_abs_distance_error_m": float(ordered["abs_distance_error_m"].max()),
                "max_abs_orientation_error_deg": float(ordered["abs_orientation_error_deg"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["max_abs_distance_shift_m", "max_abs_orientation_shift_deg"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)


def _summarize_aggregate(predictions_df: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        predictions_df.groupby("darkness_gain", sort=True, dropna=False)
        .agg(
            sample_count=("analysis_sample_index", "nunique"),
            mean_variant_vehicle_mean_darkness=("variant_vehicle_mean_darkness", "mean"),
            mean_abs_distance_error_m=("abs_distance_error_m", "mean"),
            median_abs_distance_error_m=("abs_distance_error_m", "median"),
            p95_abs_distance_error_m=("abs_distance_error_m", _p95),
            mean_abs_distance_shift_m=("abs_distance_shift_from_baseline_m", "mean"),
            p95_abs_distance_shift_m=("abs_distance_shift_from_baseline_m", _p95),
            max_abs_distance_shift_m=("abs_distance_shift_from_baseline_m", "max"),
            mean_abs_orientation_error_deg=("abs_orientation_error_deg", "mean"),
            median_abs_orientation_error_deg=("abs_orientation_error_deg", "median"),
            p95_abs_orientation_error_deg=("abs_orientation_error_deg", _p95),
            mean_abs_orientation_shift_deg=("abs_orientation_shift_from_baseline_deg", "mean"),
            p95_abs_orientation_shift_deg=("abs_orientation_shift_from_baseline_deg", _p95),
            max_abs_orientation_shift_deg=("abs_orientation_shift_from_baseline_deg", "max"),
        )
        .reset_index()
    )
    baseline_row = aggregate.loc[aggregate["darkness_gain"].sub(1.0).abs().idxmin()]
    baseline_distance_mae = float(baseline_row["mean_abs_distance_error_m"])
    baseline_orientation_mae = float(baseline_row["mean_abs_orientation_error_deg"])
    baseline_distance_shift = float(baseline_row["mean_abs_distance_shift_m"])
    baseline_orientation_shift = float(baseline_row["mean_abs_orientation_shift_deg"])
    aggregate["delta_mean_abs_distance_error_m_vs_baseline"] = (
        aggregate["mean_abs_distance_error_m"] - baseline_distance_mae
    )
    aggregate["delta_mean_abs_orientation_error_deg_vs_baseline"] = (
        aggregate["mean_abs_orientation_error_deg"] - baseline_orientation_mae
    )
    aggregate["delta_mean_abs_distance_shift_m_vs_baseline"] = (
        aggregate["mean_abs_distance_shift_m"] - baseline_distance_shift
    )
    aggregate["delta_mean_abs_orientation_shift_deg_vs_baseline"] = (
        aggregate["mean_abs_orientation_shift_deg"] - baseline_orientation_shift
    )
    return aggregate.sort_values("darkness_gain", kind="stable").reset_index(drop=True)


def _summarize_brightness_correlations(per_sample_summary: pd.DataFrame) -> pd.DataFrame:
    if per_sample_summary.empty:
        return pd.DataFrame(
            columns=["feature", "metric", "pearson_r", "abs_pearson_r", "sample_count"]
        )

    feature_columns = [
        "baseline_canvas_mean_intensity",
        "baseline_canvas_std_intensity",
        "baseline_vehicle_mean_intensity",
        "baseline_vehicle_std_intensity",
        "baseline_vehicle_mean_darkness",
        "baseline_vehicle_pixel_fraction",
    ]
    metric_columns = [
        "truth_distance_m",
        "baseline_abs_distance_error_m",
        "max_abs_distance_shift_m",
        "distance_shift_slope_m_per_gain",
        "baseline_abs_orientation_error_deg",
        "max_abs_orientation_shift_deg",
        "orientation_shift_slope_deg_per_gain",
    ]

    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        feature_series = pd.to_numeric(per_sample_summary[feature], errors="coerce")
        for metric in metric_columns:
            metric_series = pd.to_numeric(per_sample_summary[metric], errors="coerce")
            valid = feature_series.notna() & metric_series.notna()
            sample_count = int(valid.sum())
            if sample_count >= 2:
                valid_feature = feature_series[valid]
                valid_metric = metric_series[valid]
                if valid_feature.nunique(dropna=True) >= 2 and valid_metric.nunique(dropna=True) >= 2:
                    pearson_r = float(valid_feature.corr(valid_metric))
                else:
                    pearson_r = float("nan")
            else:
                pearson_r = float("nan")
            rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "pearson_r": pearson_r,
                    "abs_pearson_r": float(abs(pearson_r)) if np.isfinite(pearson_r) else float("nan"),
                    "sample_count": sample_count,
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["metric", "abs_pearson_r"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)
