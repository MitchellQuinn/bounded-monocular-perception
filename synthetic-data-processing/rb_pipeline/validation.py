"""Validation helpers for pipeline inputs and outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import RunPaths, resolve_manifest_path


class PipelineValidationError(RuntimeError):
    """Raised when run-level or output validation fails."""



def to_bool(value: object) -> bool:
    """Convert typical CSV bool values to Python bool."""

    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)

    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}



def capture_success_mask(samples_df: pd.DataFrame, capture_col: str = "capture_success") -> pd.Series:
    """Create a bool mask of rows with capture_success truthy."""

    if capture_col not in samples_df.columns:
        return pd.Series(False, index=samples_df.index)
    return samples_df[capture_col].map(to_bool)



def validate_run_structure(
    run_paths: RunPaths,
    *,
    require_images: bool = False,
    require_arrays: bool = False,
) -> list[str]:
    """Validate stage run directory/file structure and return errors."""

    errors: list[str] = []

    if not run_paths.root.exists():
        errors.append(f"Run directory does not exist: {run_paths.root}")

    if require_images and (run_paths.images_dir is None or not run_paths.images_dir.exists()):
        errors.append(f"Images directory does not exist: {run_paths.images_dir}")

    if require_arrays and (run_paths.arrays_dir is None or not run_paths.arrays_dir.exists()):
        errors.append(f"Arrays directory does not exist: {run_paths.arrays_dir}")

    run_json_path = run_paths.manifests_dir / "run.json"
    samples_path = run_paths.manifests_dir / "samples.csv"

    if not run_json_path.is_file():
        errors.append(f"Missing run.json: {run_json_path}")

    if not samples_path.is_file():
        errors.append(f"Missing samples.csv: {samples_path}")

    return errors



def validate_required_columns(samples_df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """Return missing required column names as error messages."""

    errors: list[str] = []
    for column in required_columns:
        if column not in samples_df.columns:
            errors.append(f"Missing required column: {column}")
    return errors



def validate_capture_success_images(
    samples_df: pd.DataFrame,
    run_root: Path,
    *,
    image_column: str = "image_filename",
    default_subdir: str = "images",
    capture_column: str = "capture_success",
) -> list[str]:
    """Validate that capture_success rows can resolve image files."""

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



def validate_pack_array(
    array: np.ndarray,
    expected_shape: tuple[int, int] | None,
    *,
    allowed_dtypes: set[np.dtype] | None = None,
) -> tuple[int, int]:
    """Validate one array for pack stage and return its shape."""

    if allowed_dtypes is not None and np.dtype(array.dtype) not in allowed_dtypes:
        allowed = ", ".join(str(dtype) for dtype in sorted(allowed_dtypes, key=str))
        raise ValueError(f"Unsupported array dtype {array.dtype}; allowed dtypes: {allowed}")

    if array.ndim != 2:
        raise ValueError(f"Expected shape (H, W), got {array.shape}")

    if np.issubdtype(array.dtype, np.floating):
        if np.isnan(array).any():
            raise ValueError("Array contains NaN values")
        if np.isinf(array).any():
            raise ValueError("Array contains infinite values")
    elif not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)):
        raise ValueError(f"Unsupported array dtype for pack stage: {array.dtype}")

    shape = (int(array.shape[0]), int(array.shape[1]))
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"Array shape mismatch: expected {expected_shape}, got {shape}")

    return shape



def validate_npz_file(npz_path: Path, *, allowed_x_dtypes: set[np.dtype] | None = None) -> None:
    """Validate required keys and shape consistency in a generated NPZ shard."""

    required_keys = ["X", "y", "sample_id", "image_filename", "npz_row_index"]

    with np.load(npz_path, allow_pickle=False) as data:
        for key in required_keys:
            if key not in data:
                raise ValueError(f"NPZ missing required key: {key}")

        x = data["X"]
        y = data["y"]
        sample_id = data["sample_id"]
        image_filename = data["image_filename"]
        row_index = data["npz_row_index"]

        if allowed_x_dtypes is not None and np.dtype(x.dtype) not in allowed_x_dtypes:
            allowed = ", ".join(str(dtype) for dtype in sorted(allowed_x_dtypes, key=str))
            raise ValueError(f"X dtype must be one of ({allowed}), got {x.dtype}")
        if not (
            np.issubdtype(x.dtype, np.floating)
            or np.issubdtype(x.dtype, np.integer)
            or np.issubdtype(x.dtype, np.bool_)
        ):
            raise ValueError(f"X has unsupported dtype: {x.dtype}")

        if x.ndim != 3:
            raise ValueError(f"X must have shape (N, H, W), got {x.shape}")

        n = x.shape[0]
        if not (len(y) == len(sample_id) == len(image_filename) == len(row_index) == n):
            raise ValueError("NPZ arrays have inconsistent lengths")

        expected = np.arange(n, dtype=np.int64)
        if not np.array_equal(row_index, expected):
            raise ValueError("npz_row_index must be sequential from 0..N-1")
