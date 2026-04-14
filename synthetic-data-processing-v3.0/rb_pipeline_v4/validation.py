"""Validation helpers for v4 pipeline inputs and outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import REQUIRED_DUAL_STREAM_NPZ_KEYS
from .paths import RunPathsV4, resolve_manifest_path


class PipelineValidationError(RuntimeError):
    """Raised when run-level validation fails."""


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)

    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def capture_success_mask(samples_df: pd.DataFrame, capture_col: str = "capture_success") -> pd.Series:
    if capture_col not in samples_df.columns:
        return pd.Series(False, index=samples_df.index)
    return samples_df[capture_col].map(to_bool)


def validate_run_structure(
    run_paths: RunPathsV4,
    *,
    require_images: bool = False,
    require_arrays: bool = False,
) -> list[str]:
    errors: list[str] = []

    if not run_paths.root.exists():
        errors.append(f"Run directory does not exist: {run_paths.root}")

    if require_images and (run_paths.images_dir is None or not run_paths.images_dir.exists()):
        errors.append(f"Images directory does not exist: {run_paths.images_dir}")

    if require_arrays and (run_paths.arrays_dir is None or not run_paths.arrays_dir.exists()):
        errors.append(f"Arrays directory does not exist: {run_paths.arrays_dir}")

    run_json = run_paths.manifests_dir / "run.json"
    samples_csv = run_paths.manifests_dir / "samples.csv"
    if not run_json.is_file():
        errors.append(f"Missing run.json: {run_json}")
    if not samples_csv.is_file():
        errors.append(f"Missing samples.csv: {samples_csv}")

    return errors


def validate_required_columns(samples_df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [f"Missing required column: {column}" for column in required_columns if column not in samples_df.columns]


def validate_capture_success_images(
    samples_df: pd.DataFrame,
    run_root: Path,
    *,
    image_column: str = "image_filename",
    default_subdir: str = "images",
    capture_column: str = "capture_success",
) -> list[str]:
    errors: list[str] = []

    if image_column not in samples_df.columns:
        return [f"Missing image filename column: {image_column}"]

    mask = capture_success_mask(samples_df, capture_column)
    for idx in samples_df.index[mask]:
        value = samples_df.at[idx, image_column]
        try:
            image_path = resolve_manifest_path(run_root, default_subdir, value)
        except Exception as exc:
            errors.append(f"Row {idx}: invalid image path '{value}' ({exc})")
            continue

        if not image_path.is_file():
            errors.append(f"Row {idx}: missing image file {image_path}")

    return errors


def _assert_numeric(name: str, array: np.ndarray) -> None:
    if not (
        np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{name} has unsupported dtype: {array.dtype}")



def validate_dual_stream_npz_file(npz_path: Path, *, require_v1_compat_arrays: bool = False) -> None:
    with np.load(npz_path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = sorted(REQUIRED_DUAL_STREAM_NPZ_KEYS - keys)
        if missing:
            raise ValueError(f"NPZ missing required key(s): {missing}")

        silhouette = data["silhouette_crop"]
        bbox_features = data["bbox_features"]
        y_position_3d = data["y_position_3d"]
        y_distance_m = data["y_distance_m"]
        y_yaw_deg = data["y_yaw_deg"]
        y_yaw_sin = data["y_yaw_sin"]
        y_yaw_cos = data["y_yaw_cos"]
        sample_id = data["sample_id"]
        image_filename = data["image_filename"]
        row_index = data["npz_row_index"]

        _assert_numeric("silhouette_crop", silhouette)
        _assert_numeric("bbox_features", bbox_features)
        _assert_numeric("y_position_3d", y_position_3d)
        _assert_numeric("y_distance_m", y_distance_m)
        _assert_numeric("y_yaw_deg", y_yaw_deg)
        _assert_numeric("y_yaw_sin", y_yaw_sin)
        _assert_numeric("y_yaw_cos", y_yaw_cos)

        if silhouette.ndim != 4:
            raise ValueError(f"silhouette_crop must have shape (N, C, H, W), got {silhouette.shape}")
        if bbox_features.ndim != 2 or bbox_features.shape[1] != 10:
            raise ValueError(f"bbox_features must have shape (N, 10), got {bbox_features.shape}")
        if y_position_3d.ndim != 2 or y_position_3d.shape[1] != 3:
            raise ValueError(f"y_position_3d must have shape (N, 3), got {y_position_3d.shape}")
        if y_distance_m.ndim != 1:
            raise ValueError(f"y_distance_m must have shape (N,), got {y_distance_m.shape}")
        if y_yaw_deg.ndim != 1:
            raise ValueError(f"y_yaw_deg must have shape (N,), got {y_yaw_deg.shape}")
        if y_yaw_sin.ndim != 1:
            raise ValueError(f"y_yaw_sin must have shape (N,), got {y_yaw_sin.shape}")
        if y_yaw_cos.ndim != 1:
            raise ValueError(f"y_yaw_cos must have shape (N,), got {y_yaw_cos.shape}")

        n = int(silhouette.shape[0])
        lengths = {
            "bbox_features": int(bbox_features.shape[0]),
            "y_position_3d": int(y_position_3d.shape[0]),
            "y_distance_m": int(y_distance_m.shape[0]),
            "y_yaw_deg": int(y_yaw_deg.shape[0]),
            "y_yaw_sin": int(y_yaw_sin.shape[0]),
            "y_yaw_cos": int(y_yaw_cos.shape[0]),
            "sample_id": int(sample_id.shape[0]),
            "image_filename": int(image_filename.shape[0]),
            "npz_row_index": int(row_index.shape[0]),
        }
        bad = {name: length for name, length in lengths.items() if length != n}
        if bad:
            raise ValueError(f"NPZ first-dimension mismatch against silhouette_crop N={n}: {bad}")

        if not np.array_equal(row_index.astype(np.int64), np.arange(n, dtype=np.int64)):
            raise ValueError("npz_row_index must be contiguous 0..N-1")

        if np.isnan(bbox_features).any() or np.isinf(bbox_features).any():
            raise ValueError("bbox_features contains NaN or Inf")

        if np.isnan(y_yaw_deg).any() or np.isinf(y_yaw_deg).any():
            raise ValueError("y_yaw_deg contains NaN or Inf")
        if np.isnan(y_yaw_sin).any() or np.isinf(y_yaw_sin).any():
            raise ValueError("y_yaw_sin contains NaN or Inf")
        if np.isnan(y_yaw_cos).any() or np.isinf(y_yaw_cos).any():
            raise ValueError("y_yaw_cos contains NaN or Inf")

        yaw_rad = np.deg2rad(y_yaw_deg.astype(np.float64))
        expected_sin = np.sin(yaw_rad)
        expected_cos = np.cos(yaw_rad)
        if not np.allclose(y_yaw_sin.astype(np.float64), expected_sin, atol=1e-5, rtol=1e-5):
            raise ValueError("y_yaw_sin is inconsistent with y_yaw_deg")
        if not np.allclose(y_yaw_cos.astype(np.float64), expected_cos, atol=1e-5, rtol=1e-5):
            raise ValueError("y_yaw_cos is inconsistent with y_yaw_deg")

        if require_v1_compat_arrays:
            if "X" not in data or "y" not in data:
                raise ValueError("NPZ is missing required v1 compatibility arrays X/y")
            x = data["X"]
            y = data["y"]
            if x.shape[0] != n or y.shape[0] != n:
                raise ValueError("Compatibility arrays X/y do not align with primary arrays")
