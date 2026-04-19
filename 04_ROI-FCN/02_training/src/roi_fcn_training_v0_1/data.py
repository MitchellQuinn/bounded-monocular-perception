"""Dataset discovery, strict contract validation, and shard-batch iteration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
from numpy.lib import format as npy_format
import pandas as pd

from .contracts import (
    EXPECTED_CHANNEL_COUNT,
    EXPECTED_GEOMETRY_SCHEMA,
    EXPECTED_IMAGE_LAYOUT,
    EXPECTED_NORMALIZATION_RANGE,
    EXPECTED_PREPROCESSING_CONTRACT_VERSION,
    EXPECTED_REPRESENTATION_KIND,
    EXPECTED_STORAGE_FORMAT,
    EXPECTED_TARGET_SOURCE,
    EXPECTED_TARGET_TYPE,
    MANIFESTS_DIR_NAME,
    NUMERIC_MANIFEST_COLUMNS,
    OPTIONAL_TRACEABILITY_NPZ_KEYS,
    PREPROCESSING_CONTRACT_KEY,
    REQUIRED_MANIFEST_COLUMNS,
    REQUIRED_NPZ_ARRAY_KEYS,
    ARRAYS_DIR_NAME,
    CorpusGeometryContract,
    RoiFcnBatch,
    SplitDatasetContract,
    SplitPaths,
    TRAIN_SPLIT_NAME,
    VALIDATE_SPLIT_NAME,
)
from .paths import resolve_split_paths


class RoiFcnDatasetValidationError(ValueError):
    """Raised when a packed corpus violates the ROI-FCN training contract."""


@dataclass(frozen=True)
class ShardInfo:
    """Validated shard metadata used by the batch iterator."""

    npz_filename: str
    npz_path: Path
    row_count: int
    canvas_height_px: int
    canvas_width_px: int
    has_bootstrap_bbox: bool
    has_bootstrap_confidence: bool


@dataclass
class LoadedSplitDataset:
    """Validated split metadata and shard listing."""

    contract: SplitDatasetContract
    samples_df: pd.DataFrame
    shard_infos: dict[str, ShardInfo]
    split_paths: SplitPaths


def _read_json(path: Path) -> dict[str, object]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RoiFcnDatasetValidationError(f"run.json must contain an object: {path}")
    return payload


def _member_name_for_key(archive: zipfile.ZipFile, key: str) -> str:
    names = {name for name in archive.namelist() if name.endswith(".npy")}
    expected = f"{key}.npy"
    if expected not in names:
        raise RoiFcnDatasetValidationError(
            f"NPZ missing expected member {expected!r}; found {sorted(names)}"
        )
    return expected


def _read_npz_array_header(npz_path: Path, key: str) -> tuple[tuple[int, ...], np.dtype, bool]:
    with zipfile.ZipFile(npz_path) as archive:
        member_name = _member_name_for_key(archive, key)
        with archive.open(member_name, "r") as handle:
            major, minor = npy_format.read_magic(handle)
            if (major, minor) == (1, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_1_0(handle)
            elif (major, minor) == (2, 0):
                shape, fortran_order, dtype = npy_format.read_array_header_2_0(handle)
            else:
                raise RoiFcnDatasetValidationError(
                    f"Unsupported npy header version {(major, minor)} in {npz_path}:{member_name}"
                )
    return tuple(int(dim) for dim in shape), np.dtype(dtype), bool(fortran_order)


def _validate_preprocessing_contract(run_json: dict[str, object], *, split_paths: SplitPaths) -> dict[str, object]:
    raw = run_json.get(PREPROCESSING_CONTRACT_KEY)
    if not isinstance(raw, dict):
        raise RoiFcnDatasetValidationError(
            f"run.json missing {PREPROCESSING_CONTRACT_KEY!r} mapping: {split_paths.run_json_path}"
        )
    version = str(raw.get("ContractVersion", "")).strip()
    if version != EXPECTED_PREPROCESSING_CONTRACT_VERSION:
        raise RoiFcnDatasetValidationError(
            "Unexpected preprocessing contract version: "
            f"expected {EXPECTED_PREPROCESSING_CONTRACT_VERSION!r}, got {version!r}"
        )

    current_representation = raw.get("CurrentRepresentation")
    if not isinstance(current_representation, dict):
        raise RoiFcnDatasetValidationError(
            f"run.json missing PreprocessingContract.CurrentRepresentation: {split_paths.run_json_path}"
        )

    kind = str(current_representation.get("Kind", "")).strip()
    if kind != EXPECTED_REPRESENTATION_KIND:
        raise RoiFcnDatasetValidationError(
            f"Unexpected representation kind: expected {EXPECTED_REPRESENTATION_KIND!r}, got {kind!r}"
        )

    storage_format = str(current_representation.get("StorageFormat", "")).strip().lower()
    if storage_format != EXPECTED_STORAGE_FORMAT:
        raise RoiFcnDatasetValidationError(
            f"Unexpected storage format: expected {EXPECTED_STORAGE_FORMAT!r}, got {storage_format!r}"
        )

    image_layout = str(current_representation.get("ImageLayout", "")).strip()
    if image_layout != EXPECTED_IMAGE_LAYOUT:
        raise RoiFcnDatasetValidationError(
            f"Unexpected image layout: expected {EXPECTED_IMAGE_LAYOUT!r}, got {image_layout!r}"
        )

    channels = int(current_representation.get("Channels", -1))
    if channels != EXPECTED_CHANNEL_COUNT:
        raise RoiFcnDatasetValidationError(
            f"Unexpected channel count: expected {EXPECTED_CHANNEL_COUNT}, got {channels}"
        )

    target_type = str(current_representation.get("TargetType", "")).strip()
    if target_type != EXPECTED_TARGET_TYPE:
        raise RoiFcnDatasetValidationError(
            f"Unexpected target type: expected {EXPECTED_TARGET_TYPE!r}, got {target_type!r}"
        )

    target_source = str(current_representation.get("TargetSource", "")).strip()
    if target_source != EXPECTED_TARGET_SOURCE:
        raise RoiFcnDatasetValidationError(
            f"Unexpected target source: expected {EXPECTED_TARGET_SOURCE!r}, got {target_source!r}"
        )

    array_keys = current_representation.get("ArrayKeys")
    if not isinstance(array_keys, list):
        raise RoiFcnDatasetValidationError("CurrentRepresentation.ArrayKeys must be a list.")
    missing_keys = sorted(REQUIRED_NPZ_ARRAY_KEYS - {str(value) for value in array_keys})
    if missing_keys:
        raise RoiFcnDatasetValidationError(
            f"CurrentRepresentation.ArrayKeys is missing required keys: {missing_keys}"
        )

    normalization_range = tuple(float(value) for value in current_representation.get("NormalizationRange", []))
    if normalization_range != EXPECTED_NORMALIZATION_RANGE:
        raise RoiFcnDatasetValidationError(
            f"Unexpected normalization range: expected {EXPECTED_NORMALIZATION_RANGE}, got {normalization_range}"
        )

    return {
        "contract_version": version,
        "representation_kind": kind,
        "storage_format": storage_format,
        "image_layout": image_layout,
        "channels": channels,
        "array_keys": tuple(str(value) for value in array_keys),
        "canvas_width_px": int(current_representation.get("CanvasWidth", 0)),
        "canvas_height_px": int(current_representation.get("CanvasHeight", 0)),
        "fixed_roi_width_px": int(current_representation.get("FixedROICropWidthPx", 0) or 0) or None,
        "fixed_roi_height_px": int(current_representation.get("FixedROICropHeightPx", 0) or 0) or None,
    }


def _coerce_required_numeric_columns(df: pd.DataFrame, *, split_paths: SplitPaths) -> pd.DataFrame:
    coerced = df.copy()
    for column in NUMERIC_MANIFEST_COLUMNS:
        numeric = pd.to_numeric(coerced[column], errors="coerce")
        if numeric.isna().any():
            bad = int(numeric.isna().sum())
            raise RoiFcnDatasetValidationError(
                f"{column} must be numeric in {split_paths.samples_csv_path}; found {bad} invalid rows."
            )
        coerced[column] = numeric.astype(np.float32)

    npz_row = coerced["npz_row_index"].astype(np.float64)
    if not np.allclose(npz_row, np.round(npz_row), atol=1e-6):
        raise RoiFcnDatasetValidationError("npz_row_index must contain integer values.")
    coerced["npz_row_index"] = np.round(npz_row).astype(np.int64)

    for column in ("bootstrap_bbox_x1", "bootstrap_bbox_y1", "bootstrap_bbox_x2", "bootstrap_bbox_y2"):
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce").astype(np.float32)
    return coerced


def _load_and_validate_samples(
    split_paths: SplitPaths,
    *,
    representation_info: dict[str, object],
) -> pd.DataFrame:
    df = pd.read_csv(split_paths.samples_csv_path, low_memory=False)
    if df.empty:
        raise RoiFcnDatasetValidationError(f"samples.csv is empty: {split_paths.samples_csv_path}")

    missing_cols = sorted(set(REQUIRED_MANIFEST_COLUMNS) - set(df.columns))
    if missing_cols:
        raise RoiFcnDatasetValidationError(
            f"samples.csv is missing required columns {missing_cols}: {split_paths.samples_csv_path}"
        )

    status = df["pack_roi_fcn_stage_status"].astype("string").fillna("").str.strip().str.lower()
    invalid_status = ~status.eq("success")
    if invalid_status.any():
        bad = int(invalid_status.sum())
        raise RoiFcnDatasetValidationError(
            f"All rows must have pack_roi_fcn_stage_status=success; found {bad} invalid rows."
        )

    df["npz_filename"] = df["npz_filename"].astype("string").fillna("").str.strip()
    if (df["npz_filename"] == "").any():
        bad = int((df["npz_filename"] == "").sum())
        raise RoiFcnDatasetValidationError(f"npz_filename is blank in {bad} rows.")

    df["sample_id"] = df["sample_id"].astype("string").fillna("").str.strip()
    df["image_filename"] = df["image_filename"].astype("string").fillna("").str.strip()
    if (df["sample_id"] == "").any() or (df["image_filename"] == "").any():
        raise RoiFcnDatasetValidationError("sample_id and image_filename must be populated for all rows.")

    df = _coerce_required_numeric_columns(df, split_paths=split_paths)

    canvas_width = int(representation_info["canvas_width_px"])
    canvas_height = int(representation_info["canvas_height_px"])
    if canvas_width <= 0 or canvas_height <= 0:
        raise RoiFcnDatasetValidationError(
            f"Invalid canvas size in run.json: {canvas_width}x{canvas_height}"
        )

    if not np.all(df["locator_canvas_width_px"].to_numpy(dtype=np.int64) == canvas_width):
        raise RoiFcnDatasetValidationError("locator_canvas_width_px does not match run.json CanvasWidth.")
    if not np.all(df["locator_canvas_height_px"].to_numpy(dtype=np.int64) == canvas_height):
        raise RoiFcnDatasetValidationError("locator_canvas_height_px does not match run.json CanvasHeight.")

    if (df["locator_resize_scale"] <= 0.0).any():
        raise RoiFcnDatasetValidationError("locator_resize_scale must be > 0 for every row.")

    expected_resized_w = np.round(df["image_width_px"] * df["locator_resize_scale"]).astype(np.int64)
    expected_resized_h = np.round(df["image_height_px"] * df["locator_resize_scale"]).astype(np.int64)
    if not np.array_equal(expected_resized_w, df["locator_resized_width_px"].astype(np.int64).to_numpy()):
        raise RoiFcnDatasetValidationError("locator_resized_width_px does not match image_width_px * locator_resize_scale.")
    if not np.array_equal(expected_resized_h, df["locator_resized_height_px"].astype(np.int64).to_numpy()):
        raise RoiFcnDatasetValidationError("locator_resized_height_px does not match image_height_px * locator_resize_scale.")

    pad_sum_w = df["locator_pad_left_px"] + df["locator_resized_width_px"] + df["locator_pad_right_px"]
    pad_sum_h = df["locator_pad_top_px"] + df["locator_resized_height_px"] + df["locator_pad_bottom_px"]
    if not np.all(pad_sum_w.to_numpy(dtype=np.int64) == canvas_width):
        raise RoiFcnDatasetValidationError("Padding + resized width must equal canvas width for every row.")
    if not np.all(pad_sum_h.to_numpy(dtype=np.int64) == canvas_height):
        raise RoiFcnDatasetValidationError("Padding + resized height must equal canvas height for every row.")

    expected_center_x = (df["bootstrap_center_x_px"] * df["locator_resize_scale"]) + df["locator_pad_left_px"]
    expected_center_y = (df["bootstrap_center_y_px"] * df["locator_resize_scale"]) + df["locator_pad_top_px"]
    if not np.allclose(expected_center_x, df["locator_center_x_px"], atol=1e-3):
        raise RoiFcnDatasetValidationError("locator_center_x_px does not match bootstrap centre + resize/padding geometry.")
    if not np.allclose(expected_center_y, df["locator_center_y_px"], atol=1e-3):
        raise RoiFcnDatasetValidationError("locator_center_y_px does not match bootstrap centre + resize/padding geometry.")

    if (df["locator_center_x_px"] < 0.0).any() or (df["locator_center_x_px"] >= float(canvas_width)).any():
        raise RoiFcnDatasetValidationError("locator_center_x_px falls outside the locator canvas.")
    if (df["locator_center_y_px"] < 0.0).any() or (df["locator_center_y_px"] >= float(canvas_height)).any():
        raise RoiFcnDatasetValidationError("locator_center_y_px falls outside the locator canvas.")

    for column in ("bootstrap_bbox_x1", "bootstrap_bbox_y1", "bootstrap_bbox_x2", "bootstrap_bbox_y2"):
        if column not in df.columns:
            raise RoiFcnDatasetValidationError(f"Missing required bbox column {column!r}.")

    return df.reset_index(drop=True)


def _validate_npz_headers(
    npz_path: Path,
    *,
    expected_canvas_height: int,
    expected_canvas_width: int,
) -> tuple[int, bool, bool]:
    headers: dict[str, tuple[tuple[int, ...], np.dtype, bool]] = {}
    for key in sorted(REQUIRED_NPZ_ARRAY_KEYS | OPTIONAL_TRACEABILITY_NPZ_KEYS):
        try:
            headers[key] = _read_npz_array_header(npz_path, key)
        except RoiFcnDatasetValidationError:
            if key in OPTIONAL_TRACEABILITY_NPZ_KEYS:
                continue
            raise

    missing_required = sorted(REQUIRED_NPZ_ARRAY_KEYS - set(headers))
    if missing_required:
        raise RoiFcnDatasetValidationError(f"{npz_path} is missing required arrays: {missing_required}")

    image_shape, image_dtype, _ = headers["locator_input_image"]
    if len(image_shape) != 4 or image_shape[1] != 1:
        raise RoiFcnDatasetValidationError(
            f"locator_input_image must have shape (N, 1, H, W); got {image_shape} in {npz_path}"
        )
    if image_dtype != np.dtype(np.float32):
        raise RoiFcnDatasetValidationError(
            f"locator_input_image must be float32; got {image_dtype} in {npz_path}"
        )
    if int(image_shape[2]) != expected_canvas_height or int(image_shape[3]) != expected_canvas_width:
        raise RoiFcnDatasetValidationError(
            f"locator_input_image shape {image_shape} disagrees with canvas {expected_canvas_width}x{expected_canvas_height}"
        )

    expected_shapes = {
        "target_center_xy_original_px": (image_shape[0], 2),
        "target_center_xy_canvas_px": (image_shape[0], 2),
        "source_image_wh_px": (image_shape[0], 2),
        "resized_image_wh_px": (image_shape[0], 2),
        "padding_ltrb_px": (image_shape[0], 4),
        "resize_scale": (image_shape[0],),
        "sample_id": (image_shape[0],),
        "image_filename": (image_shape[0],),
        "npz_row_index": (image_shape[0],),
    }
    expected_dtypes = {
        "target_center_xy_original_px": np.dtype(np.float32),
        "target_center_xy_canvas_px": np.dtype(np.float32),
        "source_image_wh_px": np.dtype(np.int32),
        "resized_image_wh_px": np.dtype(np.int32),
        "padding_ltrb_px": np.dtype(np.int32),
        "resize_scale": np.dtype(np.float32),
        "npz_row_index": np.dtype(np.int64),
    }
    for key, expected_shape in expected_shapes.items():
        shape, dtype, _ = headers[key]
        if shape != expected_shape:
            raise RoiFcnDatasetValidationError(
                f"{key} has shape {shape}; expected {expected_shape} in {npz_path}"
            )
        expected_dtype = expected_dtypes.get(key)
        if expected_dtype is not None and dtype != expected_dtype:
            raise RoiFcnDatasetValidationError(
                f"{key} has dtype {dtype}; expected {expected_dtype} in {npz_path}"
            )

    return int(image_shape[0]), "bootstrap_bbox_xyxy_px" in headers, "bootstrap_confidence" in headers


def _validate_npz_payload_against_manifest(
    npz_path: Path,
    *,
    expected_rows: pd.DataFrame,
) -> None:
    with np.load(npz_path, allow_pickle=False) as payload:
        target_center_original = payload["target_center_xy_original_px"]
        target_center_canvas = payload["target_center_xy_canvas_px"]
        source_wh = payload["source_image_wh_px"]
        resized_wh = payload["resized_image_wh_px"]
        padding = payload["padding_ltrb_px"]
        resize_scale = payload["resize_scale"]
        sample_id = payload["sample_id"]
        image_filename = payload["image_filename"]
        npz_row_index = payload["npz_row_index"]

        if not np.array_equal(npz_row_index, expected_rows["npz_row_index"].to_numpy(dtype=np.int64)):
            raise RoiFcnDatasetValidationError(f"npz_row_index mismatch between {npz_path} and samples.csv")
        if not np.array_equal(sample_id.astype(str), expected_rows["sample_id"].astype(str).to_numpy()):
            raise RoiFcnDatasetValidationError(f"sample_id mismatch between {npz_path} and samples.csv")
        if not np.array_equal(image_filename.astype(str), expected_rows["image_filename"].astype(str).to_numpy()):
            raise RoiFcnDatasetValidationError(f"image_filename mismatch between {npz_path} and samples.csv")

        expected_source_wh = expected_rows[["image_width_px", "image_height_px"]].to_numpy(dtype=np.int32)
        if not np.array_equal(source_wh, expected_source_wh):
            raise RoiFcnDatasetValidationError(f"source_image_wh_px mismatch between {npz_path} and samples.csv")

        expected_resized_wh = expected_rows[["locator_resized_width_px", "locator_resized_height_px"]].to_numpy(dtype=np.int32)
        if not np.array_equal(resized_wh, expected_resized_wh):
            raise RoiFcnDatasetValidationError(f"resized_image_wh_px mismatch between {npz_path} and samples.csv")

        expected_padding = expected_rows[
            ["locator_pad_left_px", "locator_pad_top_px", "locator_pad_right_px", "locator_pad_bottom_px"]
        ].to_numpy(dtype=np.int32)
        if not np.array_equal(padding, expected_padding):
            raise RoiFcnDatasetValidationError(f"padding_ltrb_px mismatch between {npz_path} and samples.csv")

        expected_resize = expected_rows["locator_resize_scale"].to_numpy(dtype=np.float32)
        if not np.allclose(resize_scale, expected_resize, atol=1e-6):
            raise RoiFcnDatasetValidationError(f"resize_scale mismatch between {npz_path} and samples.csv")

        expected_original = expected_rows[["bootstrap_center_x_px", "bootstrap_center_y_px"]].to_numpy(dtype=np.float32)
        if not np.allclose(target_center_original, expected_original, atol=1e-3):
            raise RoiFcnDatasetValidationError(
                f"target_center_xy_original_px mismatch between {npz_path} and samples.csv"
            )

        expected_canvas = expected_rows[["locator_center_x_px", "locator_center_y_px"]].to_numpy(dtype=np.float32)
        if not np.allclose(target_center_canvas, expected_canvas, atol=1e-3):
            raise RoiFcnDatasetValidationError(
                f"target_center_xy_canvas_px mismatch between {npz_path} and samples.csv"
            )

        if "locator_geometry_schema" in payload.files:
            schema = tuple(str(value) for value in payload["locator_geometry_schema"].tolist())
            if schema != EXPECTED_GEOMETRY_SCHEMA:
                raise RoiFcnDatasetValidationError(
                    f"locator_geometry_schema mismatch in {npz_path}: {schema!r}"
                )

        if "bootstrap_bbox_xyxy_px" in payload.files:
            expected_bbox = expected_rows[
                ["bootstrap_bbox_x1", "bootstrap_bbox_y1", "bootstrap_bbox_x2", "bootstrap_bbox_y2"]
            ].to_numpy(dtype=np.float32)
            payload_bbox = payload["bootstrap_bbox_xyxy_px"]
            valid_mask = np.isfinite(expected_bbox).all(axis=1)
            if not np.allclose(payload_bbox[valid_mask], expected_bbox[valid_mask], atol=1e-3):
                raise RoiFcnDatasetValidationError(
                    f"bootstrap_bbox_xyxy_px mismatch between {npz_path} and samples.csv"
                )


def load_and_validate_split_dataset(
    training_root: Path | None,
    dataset_reference: str,
    split_name: str,
    *,
    datasets_root_override: str | Path | None = None,
) -> LoadedSplitDataset:
    """Load one split and validate the full packed corpus contract."""
    split = str(split_name).strip().lower()
    if split not in {TRAIN_SPLIT_NAME, VALIDATE_SPLIT_NAME}:
        raise RoiFcnDatasetValidationError(f"Unsupported split {split_name!r}; expected train or validate.")

    split_paths = resolve_split_paths(
        training_root,
        dataset_reference,
        split,
        datasets_root_override=datasets_root_override,
    )
    if not split_paths.split_root.is_dir():
        raise FileNotFoundError(f"Dataset split does not exist: {split_paths.split_root}")
    if not split_paths.arrays_dir.is_dir():
        raise FileNotFoundError(f"Missing arrays directory: {split_paths.arrays_dir}")
    if not split_paths.manifests_dir.is_dir():
        raise FileNotFoundError(f"Missing manifests directory: {split_paths.manifests_dir}")

    run_json = _read_json(split_paths.run_json_path)
    representation_info = _validate_preprocessing_contract(run_json, split_paths=split_paths)
    samples_df = _load_and_validate_samples(split_paths, representation_info=representation_info)

    referenced_npz = set(samples_df["npz_filename"].astype(str))
    on_disk_npz = {path.name for path in split_paths.arrays_dir.glob("*.npz")}
    missing_npz = sorted(referenced_npz - on_disk_npz)
    extra_npz = sorted(on_disk_npz - referenced_npz)
    if missing_npz:
        raise RoiFcnDatasetValidationError(f"samples.csv references missing shard files: {missing_npz}")
    if extra_npz:
        raise RoiFcnDatasetValidationError(
            f"arrays directory contains unreferenced shard files, which is ambiguous: {extra_npz}"
        )

    shard_infos: dict[str, ShardInfo] = {}
    for npz_filename, shard_rows in samples_df.groupby("npz_filename", sort=True):
        npz_path = split_paths.arrays_dir / str(npz_filename)
        ordered_rows = shard_rows.sort_values("npz_row_index").reset_index(drop=True)
        row_count, has_bbox, has_conf = _validate_npz_headers(
            npz_path,
            expected_canvas_height=int(representation_info["canvas_height_px"]),
            expected_canvas_width=int(representation_info["canvas_width_px"]),
        )
        if row_count != len(ordered_rows):
            raise RoiFcnDatasetValidationError(
                f"Shard {npz_filename} row count {row_count} does not match samples.csv rows {len(ordered_rows)}"
            )
        _validate_npz_payload_against_manifest(npz_path, expected_rows=ordered_rows)
        shard_infos[str(npz_filename)] = ShardInfo(
            npz_filename=str(npz_filename),
            npz_path=npz_path,
            row_count=row_count,
            canvas_height_px=int(representation_info["canvas_height_px"]),
            canvas_width_px=int(representation_info["canvas_width_px"]),
            has_bootstrap_bbox=has_bbox,
            has_bootstrap_confidence=has_conf,
        )

    has_any_bbox = bool(samples_df[["bootstrap_bbox_x1", "bootstrap_bbox_y1", "bootstrap_bbox_x2", "bootstrap_bbox_y2"]].notna().all(axis=1).any())
    geometry = CorpusGeometryContract(
        canvas_width_px=int(representation_info["canvas_width_px"]),
        canvas_height_px=int(representation_info["canvas_height_px"]),
        image_layout=str(representation_info["image_layout"]),
        channels=int(representation_info["channels"]),
        normalization_range=EXPECTED_NORMALIZATION_RANGE,
        geometry_schema=EXPECTED_GEOMETRY_SCHEMA,
    )
    contract = SplitDatasetContract(
        dataset_reference=str(dataset_reference),
        split_name=split,
        split_root=str(split_paths.split_root),
        run_json_path=str(split_paths.run_json_path),
        samples_csv_path=str(split_paths.samples_csv_path),
        row_count=int(len(samples_df)),
        shard_count=int(len(shard_infos)),
        geometry=geometry,
        preprocessing_contract_version=str(representation_info["contract_version"]),
        representation_kind=str(representation_info["representation_kind"]),
        representation_storage_format=str(representation_info["storage_format"]),
        representation_array_keys=tuple(str(value) for value in representation_info["array_keys"]),
        bootstrap_bbox_available=has_any_bbox,
        fixed_roi_width_px=representation_info["fixed_roi_width_px"],
        fixed_roi_height_px=representation_info["fixed_roi_height_px"],
    )
    return LoadedSplitDataset(
        contract=contract,
        samples_df=samples_df,
        shard_infos=shard_infos,
        split_paths=split_paths,
    )


def validate_run_compatibility(train_split: LoadedSplitDataset, validation_split: LoadedSplitDataset) -> None:
    """Ensure train and validation corpora are compatible inside one run."""
    train_geometry = train_split.contract.geometry
    validation_geometry = validation_split.contract.geometry
    if train_geometry.to_dict() != validation_geometry.to_dict():
        raise RoiFcnDatasetValidationError(
            "Training and validation corpora expose incompatible locator geometry contracts; "
            f"train={train_geometry.to_dict()} validation={validation_geometry.to_dict()}"
        )


def iter_split_batches(
    split_dataset: LoadedSplitDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    """Yield one split as batches, loading one shard at a time."""
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")

    rng = np.random.default_rng(int(seed))
    shard_names = list(sorted(split_dataset.shard_infos.keys()))
    if shuffle:
        rng.shuffle(shard_names)

    for shard_name in shard_names:
        shard_info = split_dataset.shard_infos[shard_name]
        with np.load(shard_info.npz_path, allow_pickle=False) as payload:
            images = payload["locator_input_image"]
            if images.dtype != np.float32:
                raise RoiFcnDatasetValidationError(
                    f"locator_input_image must be float32 in {shard_info.npz_path}; got {images.dtype}"
                )
            if images.ndim != 4 or images.shape[1] != 1:
                raise RoiFcnDatasetValidationError(
                    f"locator_input_image must have shape (N,1,H,W); got {images.shape} in {shard_info.npz_path}"
                )
            if float(images.min()) < 0.0 or float(images.max()) > 1.0:
                raise RoiFcnDatasetValidationError(
                    f"locator_input_image values must stay inside [0,1] in {shard_info.npz_path}"
                )
            if np.isnan(images).any():
                raise RoiFcnDatasetValidationError(f"locator_input_image contains NaNs in {shard_info.npz_path}")

            order = np.arange(shard_info.row_count, dtype=np.int64)
            if shuffle:
                rng.shuffle(order)

            target_center_canvas = payload["target_center_xy_canvas_px"]
            target_center_original = payload["target_center_xy_original_px"]
            source_wh = payload["source_image_wh_px"]
            resized_wh = payload["resized_image_wh_px"]
            padding = payload["padding_ltrb_px"]
            resize_scale = payload["resize_scale"]
            sample_id = payload["sample_id"]
            image_filename = payload["image_filename"]
            npz_row_index = payload["npz_row_index"]
            bbox = payload["bootstrap_bbox_xyxy_px"] if shard_info.has_bootstrap_bbox else None
            confidence = payload["bootstrap_confidence"] if shard_info.has_bootstrap_confidence else None

            for start in range(0, shard_info.row_count, int(batch_size)):
                take = order[start : start + int(batch_size)]
                yield RoiFcnBatch(
                    images=images[take],
                    target_center_canvas_px=target_center_canvas[take],
                    target_center_original_px=target_center_original[take],
                    source_image_wh_px=source_wh[take],
                    resized_image_wh_px=resized_wh[take],
                    padding_ltrb_px=padding[take],
                    resize_scale=resize_scale[take],
                    sample_id=sample_id[take],
                    image_filename=image_filename[take],
                    npz_filename=tuple([shard_name] * len(take)),
                    npz_row_index=npz_row_index[take],
                    bootstrap_bbox_xyxy_px=bbox[take] if bbox is not None else None,
                    bootstrap_confidence=confidence[take] if confidence is not None else None,
                )
