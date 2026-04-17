"""Validation helpers for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import (
    BOOTSTRAP_STAGE_COLUMNS,
    INPUT_REQUIRED_COLUMNS,
    REQUIRED_ROI_FCN_NPZ_KEYS,
    SPLIT_ORDER,
    TRACEABILITY_ROI_FCN_NPZ_KEYS,
)
from .manifest import load_samples_csv
from .paths import SplitPaths, resolve_input_image_path, resolve_split_paths


class RoiFcnPreprocessingValidationError(RuntimeError):
    """Raised when dataset or split validation fails."""


def to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def capture_success_mask(
    samples_df: pd.DataFrame,
    capture_column: str = "capture_success",
) -> pd.Series:
    if capture_column not in samples_df.columns:
        return pd.Series(False, index=samples_df.index)
    return samples_df[capture_column].map(to_bool)


def validate_required_columns(samples_df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [f"Missing required column: {column}" for column in required_columns if column not in samples_df.columns]


def validate_input_split_structure(split_paths: SplitPaths) -> list[str]:
    errors: list[str] = []
    if not split_paths.input_root.exists():
        errors.append(f"Split directory does not exist: {split_paths.input_root}")
    if not split_paths.input_images_dir.is_dir():
        errors.append(f"Missing images directory: {split_paths.input_images_dir}")
    if not split_paths.input_manifests_dir.is_dir():
        errors.append(f"Missing manifests directory: {split_paths.input_manifests_dir}")
    if not split_paths.input_run_json_path.is_file():
        errors.append(f"Missing run.json: {split_paths.input_run_json_path}")
    if not split_paths.input_samples_csv_path.is_file():
        errors.append(f"Missing samples.csv: {split_paths.input_samples_csv_path}")
    return errors


def validate_capture_success_images(samples_df: pd.DataFrame, split_paths: SplitPaths) -> list[str]:
    errors: list[str] = []
    if "image_filename" not in samples_df.columns:
        return ["Missing required column: image_filename"]
    mask = capture_success_mask(samples_df)
    for row_idx in samples_df.index[mask]:
        image_value = samples_df.at[row_idx, "image_filename"]
        try:
            image_path = resolve_input_image_path(split_paths, image_value)
        except Exception as exc:
            errors.append(f"Row {row_idx}: invalid image_filename '{image_value}' ({exc})")
            continue
        if not image_path.is_file():
            errors.append(f"Row {row_idx}: missing image file {image_path}")
    return errors


def validate_input_split(preprocessing_root: Path | None, dataset_reference: str, split_name: str) -> list[str]:
    split_paths = resolve_split_paths(preprocessing_root, dataset_reference, split_name)
    errors = validate_input_split_structure(split_paths)
    if errors:
        return errors

    samples_df = load_samples_csv(split_paths.input_samples_csv_path)
    errors.extend(validate_required_columns(samples_df, INPUT_REQUIRED_COLUMNS))
    errors.extend(validate_capture_success_images(samples_df, split_paths))
    return errors


def validate_input_dataset_reference(
    preprocessing_root: Path | None,
    dataset_reference: str,
) -> list[str]:
    errors: list[str] = []
    for split_name in SPLIT_ORDER:
        split_errors = validate_input_split(preprocessing_root, dataset_reference, split_name)
        errors.extend([f"{split_name}: {error}" for error in split_errors])
    return errors


def ensure_valid_input_dataset_reference(
    preprocessing_root: Path | None,
    dataset_reference: str,
) -> None:
    errors = validate_input_dataset_reference(preprocessing_root, dataset_reference)
    if errors:
        raise RoiFcnPreprocessingValidationError("\n".join(errors))


def ensure_bootstrap_prerequisites(samples_df: pd.DataFrame) -> None:
    errors = validate_required_columns(samples_df, INPUT_REQUIRED_COLUMNS)
    if errors:
        raise RoiFcnPreprocessingValidationError("\n".join(errors))


def ensure_pack_prerequisites(samples_df: pd.DataFrame) -> None:
    required = INPUT_REQUIRED_COLUMNS + BOOTSTRAP_STAGE_COLUMNS
    errors = validate_required_columns(samples_df, required)
    if errors:
        raise RoiFcnPreprocessingValidationError("\n".join(errors))


def _assert_dtype(name: str, array: np.ndarray, dtype: np.dtype) -> None:
    if array.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}")


def validate_roi_fcn_npz_file(
    npz_path: Path,
    *,
    expected_canvas_height: int | None = None,
    expected_canvas_width: int | None = None,
) -> None:
    with np.load(npz_path, allow_pickle=False) as payload:
        keys = set(payload.files)
        missing = sorted(REQUIRED_ROI_FCN_NPZ_KEYS - keys)
        if missing:
            raise ValueError(f"NPZ missing required key(s): {missing}")

        locator_input_image = payload["locator_input_image"]
        target_center_xy_original_px = payload["target_center_xy_original_px"]
        target_center_xy_canvas_px = payload["target_center_xy_canvas_px"]
        source_image_wh_px = payload["source_image_wh_px"]
        resized_image_wh_px = payload["resized_image_wh_px"]
        padding_ltrb_px = payload["padding_ltrb_px"]
        resize_scale = payload["resize_scale"]
        sample_id = payload["sample_id"]
        image_filename = payload["image_filename"]
        npz_row_index = payload["npz_row_index"]

        _assert_dtype("locator_input_image", locator_input_image, np.dtype(np.float32))
        _assert_dtype("target_center_xy_original_px", target_center_xy_original_px, np.dtype(np.float32))
        _assert_dtype("target_center_xy_canvas_px", target_center_xy_canvas_px, np.dtype(np.float32))
        _assert_dtype("source_image_wh_px", source_image_wh_px, np.dtype(np.int32))
        _assert_dtype("resized_image_wh_px", resized_image_wh_px, np.dtype(np.int32))
        _assert_dtype("padding_ltrb_px", padding_ltrb_px, np.dtype(np.int32))
        _assert_dtype("resize_scale", resize_scale, np.dtype(np.float32))
        _assert_dtype("npz_row_index", npz_row_index, np.dtype(np.int64))

        if locator_input_image.ndim != 4 or locator_input_image.shape[1] != 1:
            raise ValueError(
                "locator_input_image must have shape (N, 1, H, W), "
                f"got {locator_input_image.shape}"
            )
        if target_center_xy_original_px.ndim != 2 or target_center_xy_original_px.shape[1] != 2:
            raise ValueError(
                "target_center_xy_original_px must have shape (N, 2), "
                f"got {target_center_xy_original_px.shape}"
            )
        if target_center_xy_canvas_px.ndim != 2 or target_center_xy_canvas_px.shape[1] != 2:
            raise ValueError(
                "target_center_xy_canvas_px must have shape (N, 2), "
                f"got {target_center_xy_canvas_px.shape}"
            )
        if source_image_wh_px.ndim != 2 or source_image_wh_px.shape[1] != 2:
            raise ValueError(f"source_image_wh_px must have shape (N, 2), got {source_image_wh_px.shape}")
        if resized_image_wh_px.ndim != 2 or resized_image_wh_px.shape[1] != 2:
            raise ValueError(
                f"resized_image_wh_px must have shape (N, 2), got {resized_image_wh_px.shape}"
            )
        if padding_ltrb_px.ndim != 2 or padding_ltrb_px.shape[1] != 4:
            raise ValueError(f"padding_ltrb_px must have shape (N, 4), got {padding_ltrb_px.shape}")
        if resize_scale.ndim != 1:
            raise ValueError(f"resize_scale must have shape (N,), got {resize_scale.shape}")

        row_count = int(locator_input_image.shape[0])
        lengths = {
            "target_center_xy_original_px": int(target_center_xy_original_px.shape[0]),
            "target_center_xy_canvas_px": int(target_center_xy_canvas_px.shape[0]),
            "source_image_wh_px": int(source_image_wh_px.shape[0]),
            "resized_image_wh_px": int(resized_image_wh_px.shape[0]),
            "padding_ltrb_px": int(padding_ltrb_px.shape[0]),
            "resize_scale": int(resize_scale.shape[0]),
            "sample_id": int(sample_id.shape[0]),
            "image_filename": int(image_filename.shape[0]),
            "npz_row_index": int(npz_row_index.shape[0]),
        }
        mismatched = {name: value for name, value in lengths.items() if value != row_count}
        if mismatched:
            raise ValueError(f"NPZ first-dimension mismatch against N={row_count}: {mismatched}")

        if not np.array_equal(npz_row_index, np.arange(row_count, dtype=np.int64)):
            raise ValueError("npz_row_index must be contiguous 0..N-1")

        if float(locator_input_image.min()) < 0.0 or float(locator_input_image.max()) > 1.0:
            raise ValueError("locator_input_image values must stay inside [0.0, 1.0]")

        if expected_canvas_height is not None and locator_input_image.shape[2] != int(expected_canvas_height):
            raise ValueError(
                "locator_input_image canvas height mismatch: "
                f"expected {expected_canvas_height}, got {locator_input_image.shape[2]}"
            )
        if expected_canvas_width is not None and locator_input_image.shape[3] != int(expected_canvas_width):
            raise ValueError(
                "locator_input_image canvas width mismatch: "
                f"expected {expected_canvas_width}, got {locator_input_image.shape[3]}"
            )

        traceability_missing = sorted(TRACEABILITY_ROI_FCN_NPZ_KEYS - keys)
        if traceability_missing:
            raise ValueError(f"NPZ missing traceability key(s): {traceability_missing}")


def stage_summary_counts_from_samples(samples_df: pd.DataFrame, status_column: str) -> tuple[int, int, int]:
    status_series = samples_df[status_column].fillna("")
    successful_rows = int((status_series == "success").sum())
    failed_rows = int((status_series == "failed").sum())
    skipped_rows = int((status_series == "skipped").sum())
    return successful_rows, failed_rows, skipped_rows
