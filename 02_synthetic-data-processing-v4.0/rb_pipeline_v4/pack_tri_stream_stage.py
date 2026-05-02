"""Stage 3b (v4): distance image + orientation image + geometry -> tri-stream NPZ shards."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Callable

import cv2
import numpy as np
import pandas as pd

from .brightness_normalization import apply_brightness_normalization_v4
from .config import PackTriStreamStageConfigV4, StageSummaryV4
from .constants import (
    BBOX_FEATURE_SCHEMA,
    BRIGHTNESS_NORMALIZATION_COLUMNS,
    DETECT_STAGE_COLUMNS,
    ORIENTATION_TARGET_COLUMNS,
    SILHOUETTE_STAGE_COLUMNS,
    TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
    TRI_STREAM_GEOMETRY_ARRAY_KEY,
    TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,
    TRI_STREAM_PACK_STAGE_COLUMNS,
    UNITY_REQUIRED_COLUMNS,
)
from .image_io import read_grayscale_uint8
from .logging_utils import StageLogger
from .manifest import (
    append_columns,
    copy_run_json,
    load_samples_csv,
    samples_csv_path,
    upsert_preprocessing_contract_v4_tri_stream,
    write_samples_csv,
)
from .pack_dual_stream_stage import (
    _bbox_features_from_row,
    _place_image_on_canvas,
    _reconstruct_roi_canvas_from_source,
    _render_inverted_vehicle_detail_on_white,
    _roi_geometry_from_row,
    _safe_float,
    _shard_filename,
    _silhouette_to_background_mask,
    _validate_yaw_source_columns,
    _write_bbox_feature_columns,
    _yaw_targets_from_row,
)
from .paths import (
    ensure_run_dirs,
    input_run_paths,
    resolve_manifest_path,
    silhouette_run_paths,
    tri_stream_training_run_paths,
)
from .utils import selected_row_indices
from .validation import (
    PipelineValidationError,
    capture_success_mask,
    validate_required_columns,
    validate_run_structure,
    validate_tri_stream_npz_file,
)


def run_pack_tri_stream_stage_v4(
    project_root: Path,
    run_name: str,
    config: PackTriStreamStageConfigV4,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV4:
    """Pack tri-stream training inputs into NPZ shards."""

    input_paths = input_run_paths(project_root, run_name)
    source_paths = silhouette_run_paths(project_root, run_name)
    output_paths = tri_stream_training_run_paths(project_root, run_name)

    validation_errors = validate_run_structure(source_paths, require_images=True)
    validation_errors.extend(validate_run_structure(input_paths, require_images=True))
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    source_samples_path = samples_csv_path(source_paths.manifests_dir)
    samples_df = load_samples_csv(source_samples_path)

    required_columns = UNITY_REQUIRED_COLUMNS + DETECT_STAGE_COLUMNS + SILHOUETTE_STAGE_COLUMNS
    validation_errors.extend(validate_required_columns(samples_df, required_columns))
    validation_errors.extend(_validate_yaw_source_columns(samples_df))
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    append_columns(samples_df, TRI_STREAM_PACK_STAGE_COLUMNS)
    for column in TRI_STREAM_PACK_STAGE_COLUMNS:
        samples_df[column] = samples_df[column].astype("object")
    append_columns(samples_df, ORIENTATION_TARGET_COLUMNS)
    for column in ORIENTATION_TARGET_COLUMNS:
        samples_df[column] = samples_df[column].astype("object")
    capture_mask = capture_success_mask(samples_df)

    selected_rows = selected_row_indices(
        len(samples_df),
        offset=config.normalized_sample_offset(),
        limit=config.normalized_sample_limit(),
    )

    canvas_w = config.normalized_canvas_width_px()
    canvas_h = config.normalized_canvas_height_px()
    clip_policy = config.normalized_clip_policy()
    shard_size = config.normalized_shard_size()
    validation_errors.extend(
        _validate_tri_stream_canvas_matches_silhouette_canvas(
            samples_df,
            selected_rows=selected_rows,
            capture_mask=capture_mask,
            canvas_width=canvas_w,
            canvas_height=canvas_h,
        )
    )
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    brightness_config = config.normalized_brightness_normalization()
    brightness_contract = brightness_config.to_contract_dict()
    brightness_method = brightness_config.normalized_method()
    brightness_active = brightness_config.normalized_enabled() and brightness_method != "none"
    orientation_context_scale = config.normalized_orientation_context_scale()
    stage_via_intermediate_npy = bool(config.use_intermediate_npy and not config.dry_run)
    delete_source_npy_after_pack = bool(config.delete_source_npy_after_pack)
    stage_npy_dir = (output_paths.arrays_dir or (output_paths.root / "arrays")) / "__pack_tri_stream_stage_tmp"

    ensure_run_dirs(output_paths, dry_run=config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=config.dry_run)

    stage_parameters: dict[str, Any] = {
        "CanvasWidth": int(canvas_w),
        "CanvasHeight": int(canvas_h),
        "ClipPolicy": clip_policy,
        "ImageRepresentationMode": "roi_grayscale_inverted_vehicle_on_white",
        "DistanceImageRepresentation": "fixed_unscaled_roi_brightness_normalized_when_enabled",
        "OrientationImageRepresentation": "target_centered_inverted_vehicle_on_white_scaled_by_silhouette_extent",
        "OrientationExtentSource": "silhouette_foreground_mask",
        "OrientationContextScale": float(orientation_context_scale),
        "IncludeV1CompatArrays": bool(config.include_v1_compat_arrays),
        "IncludeOptionalMetadataArrays": bool(config.include_optional_metadata_arrays),
        "UseIntermediateNpy": bool(config.use_intermediate_npy),
        "DeleteSourceNpyAfterPack": bool(config.delete_source_npy_after_pack),
        "Compress": bool(config.compress),
        "ShardSize": int(shard_size),
        "BrightnessNormalization": brightness_contract,
        "SampleOffset": config.normalized_sample_offset(),
        "SampleLimit": config.normalized_sample_limit(),
    }
    current_representation: dict[str, Any] = {
        "Kind": "tri_stream_npz",
        "StorageFormat": "npz",
        "ArrayKeys": [
            TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
            TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,
            TRI_STREAM_GEOMETRY_ARRAY_KEY,
            "y_distance_m",
            "y_yaw_deg",
            "y_yaw_sin",
            "y_yaw_cos",
        ],
        "TargetModes": ["scalar_distance", "orientation_yaw_deg_sincos"],
        "DistanceImageKey": TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
        "DistanceImageLayout": "N,C,H,W",
        "DistanceImageGeometry": "fixed_unscaled_roi_canvas",
        "DistanceImageBrightnessNormalization": "configured_foreground_only",
        "DistanceImagePolarity": "dark_vehicle_detail_on_white_background",
        "OrientationImageKey": TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,
        "OrientationImageLayout": "N,C,H,W",
        "OrientationImageGeometry": "target_centered_scaled_by_silhouette_extent",
        "OrientationImageContent": "inverted_vehicle_detail_on_white_no_brightness_normalization",
        "OrientationImagePolarity": "dark_vehicle_detail_on_white_background",
        "OrientationExtentSource": "silhouette_foreground_mask",
        "OrientationContextScale": float(orientation_context_scale),
        "GeometryKey": TRI_STREAM_GEOMETRY_ARRAY_KEY,
        "GeometrySchema": list(BBOX_FEATURE_SCHEMA),
        "GeometryDim": int(len(BBOX_FEATURE_SCHEMA)),
        "CanvasWidth": int(canvas_w),
        "CanvasHeight": int(canvas_h),
        "DistanceImageScaling": "disabled",
        "OrientationImageScaling": "enabled_by_silhouette_extent",
        "BrightnessNormalization": brightness_contract,
    }
    upsert_preprocessing_contract_v4_tri_stream(
        output_paths.manifests_dir,
        stage_name="pack_tri_stream",
        stage_parameters=stage_parameters,
        current_representation=current_representation,
        dry_run=config.dry_run,
    )

    log_path = output_paths.manifests_dir / "pack_tri_stream_stage_log.txt"
    logger = StageLogger(
        stage_name="pack_tri_stream",
        run_name=run_name,
        log_path=log_path,
        dry_run=config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running v4 pack_tri_stream stage for run '{run_name}'")
    logger.log_parameters(config.to_log_dict())

    if brightness_active:
        append_columns(samples_df, BRIGHTNESS_NORMALIZATION_COLUMNS)
        for column in BRIGHTNESS_NORMALIZATION_COLUMNS:
            samples_df[column] = samples_df[column].astype("object")
        logger.log(
            "Brightness normalization enabled for distance stream only: "
            f"method={brightness_method}, "
            f"target_median_darkness={brightness_config.normalized_target_median_darkness():.6g}, "
            f"gain=[{brightness_config.normalized_min_gain():.6g}, "
            f"{brightness_config.normalized_max_gain():.6g}], "
            f"empty_mask_policy={brightness_config.normalized_empty_mask_policy()}"
        )
    else:
        logger.log("Brightness normalization disabled.")

    existing_npz = sorted(output_paths.root.glob(f"{run_name}*.npz"))
    if existing_npz and not config.overwrite:
        raise FileExistsError(
            f"Found existing tri-stream NPZ outputs for run '{run_name}'. Enable overwrite=True to replace them."
        )
    if existing_npz and config.overwrite and not config.dry_run:
        for path in existing_npz:
            path.unlink()
            logger.log(f"Removed existing NPZ: {path.name}")

    if stage_via_intermediate_npy and not config.dry_run:
        if stage_npy_dir.exists():
            shutil.rmtree(stage_npy_dir)
            logger.log(f"Removed stale intermediate NPY staging dir: {stage_npy_dir}")
        stage_npy_dir.mkdir(parents=True, exist_ok=True)
        logger.log(
            "Intermediate NPY staging enabled for tri-stream shard writes "
            f"(delete_after_pack={'yes' if delete_source_npy_after_pack else 'no'})"
        )
    elif bool(config.use_intermediate_npy) and config.dry_run:
        logger.log("Intermediate NPY staging requested but dry_run=True; using in-memory shard buffers.")
    else:
        logger.log("Intermediate NPY staging disabled; using in-memory shard buffers.")

    distance_image_buffer: list[np.ndarray] = []
    distance_npy_paths_buffer: list[Path] = []
    orientation_image_buffer: list[np.ndarray] = []
    orientation_npy_paths_buffer: list[Path] = []
    geometry_buffer: list[np.ndarray] = []
    y_dist_buffer: list[np.float32] = []
    y_yaw_deg_buffer: list[np.float32] = []
    y_yaw_sin_buffer: list[np.float32] = []
    y_yaw_cos_buffer: list[np.float32] = []
    sample_id_buffer: list[str] = []
    image_filename_buffer: list[str] = []
    detect_bbox_buffer: list[np.ndarray] = []
    frame_wh_buffer: list[np.ndarray] = []
    compat_x_buffer: list[np.ndarray] = []
    roi_request_buffer: list[np.ndarray] = []
    roi_source_buffer: list[np.ndarray] = []
    roi_canvas_insert_buffer: list[np.ndarray] = []
    roi_canvas_wh_buffer: list[np.ndarray] = []
    roi_padding_buffer: list[np.float32] = []
    row_indices_buffer: list[int] = []

    current_shard_idx = 0
    written_npz_paths: list[Path] = []
    aborted = False
    selected_total = int(len(selected_rows))
    progress_interval = 500
    processed_selected = 0
    progress_success = 0
    progress_failed = 0
    progress_skipped = 0
    logger.log(f"Selected rows for pack_tri_stream processing: {selected_total} / {len(samples_df)}")

    def _log_progress_if_needed() -> None:
        if (
            selected_total > 0
            and (
                processed_selected % progress_interval == 0
                or processed_selected == selected_total
            )
        ):
            logger.log(
                "Progress pack_tri_stream: "
                f"{processed_selected}/{selected_total} selected rows "
                f"(success={progress_success}, failed={progress_failed}, skipped={progress_skipped})"
            )

    def _buffered_rows_count() -> int:
        return int(len(distance_npy_paths_buffer) if stage_via_intermediate_npy else len(distance_image_buffer))

    def _reset_buffers() -> None:
        distance_image_buffer.clear()
        distance_npy_paths_buffer.clear()
        orientation_image_buffer.clear()
        orientation_npy_paths_buffer.clear()
        geometry_buffer.clear()
        y_dist_buffer.clear()
        y_yaw_deg_buffer.clear()
        y_yaw_sin_buffer.clear()
        y_yaw_cos_buffer.clear()
        sample_id_buffer.clear()
        image_filename_buffer.clear()
        detect_bbox_buffer.clear()
        frame_wh_buffer.clear()
        compat_x_buffer.clear()
        roi_request_buffer.clear()
        roi_source_buffer.clear()
        roi_canvas_insert_buffer.clear()
        roi_canvas_wh_buffer.clear()
        roi_padding_buffer.clear()
        row_indices_buffer.clear()

    def _mark_buffer_rows_failed(error_message: str) -> None:
        for buffered_row_idx in row_indices_buffer:
            samples_df.at[buffered_row_idx, "pack_tri_stream_stage_status"] = "failed"
            samples_df.at[buffered_row_idx, "pack_tri_stream_stage_error"] = error_message
            samples_df.at[buffered_row_idx, "npz_filename"] = ""
            samples_df.at[buffered_row_idx, "npz_row_index"] = pd.NA

    def _load_staged_rows(paths: list[Path], *, label: str) -> np.ndarray:
        staged_arrays: list[np.ndarray] = []
        for npy_path in paths:
            staged = np.load(npy_path, allow_pickle=False)
            if staged.ndim != 3 or int(staged.shape[0]) != 1:
                raise ValueError(
                    f"Unexpected staged {label} array shape {staged.shape} for '{npy_path.name}'. "
                    "Expected (1, H, W)."
                )
            staged_arrays.append(np.asarray(staged, dtype=np.float32))
        return np.stack(staged_arrays, axis=0).astype(np.float32)

    def _flush_shard() -> bool:
        nonlocal current_shard_idx

        if _buffered_rows_count() == 0:
            return True

        npz_name = _shard_filename(run_name, current_shard_idx, use_shards=(shard_size > 0))
        npz_path = output_paths.root / npz_name

        try:
            if stage_via_intermediate_npy:
                x_distance_image = _load_staged_rows(distance_npy_paths_buffer, label="distance image")
                x_orientation_image = _load_staged_rows(orientation_npy_paths_buffer, label="orientation image")
            else:
                x_distance_image = np.stack(distance_image_buffer, axis=0).astype(np.float32)
                x_orientation_image = np.stack(orientation_image_buffer, axis=0).astype(np.float32)
            x_geometry = np.stack(geometry_buffer, axis=0).astype(np.float32)
            y_distance_m = np.asarray(y_dist_buffer, dtype=np.float32)
            y_yaw_deg = np.asarray(y_yaw_deg_buffer, dtype=np.float32)
            y_yaw_sin = np.asarray(y_yaw_sin_buffer, dtype=np.float32)
            y_yaw_cos = np.asarray(y_yaw_cos_buffer, dtype=np.float32)
            sample_id = np.asarray(sample_id_buffer, dtype=str)
            image_filename = np.asarray(image_filename_buffer, dtype=str)
            npz_row_index = np.arange(x_distance_image.shape[0], dtype=np.int64)

            payload: dict[str, np.ndarray] = {
                TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY: x_distance_image,
                TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY: x_orientation_image,
                TRI_STREAM_GEOMETRY_ARRAY_KEY: x_geometry,
                "y_distance_m": y_distance_m,
                "y_yaw_deg": y_yaw_deg,
                "y_yaw_sin": y_yaw_sin,
                "y_yaw_cos": y_yaw_cos,
                "sample_id": sample_id,
                "image_filename": image_filename,
                "npz_row_index": npz_row_index,
                "x_geometry_schema": np.asarray(BBOX_FEATURE_SCHEMA, dtype=str),
            }

            if config.include_optional_metadata_arrays:
                n_rows = int(x_distance_image.shape[0])
                optional_lengths = {
                    "detect_bbox_buffer": len(detect_bbox_buffer),
                    "frame_wh_buffer": len(frame_wh_buffer),
                    "roi_request_buffer": len(roi_request_buffer),
                    "roi_source_buffer": len(roi_source_buffer),
                    "roi_canvas_insert_buffer": len(roi_canvas_insert_buffer),
                    "roi_canvas_wh_buffer": len(roi_canvas_wh_buffer),
                    "roi_padding_buffer": len(roi_padding_buffer),
                }
                bad_optional = {name: count for name, count in optional_lengths.items() if int(count) != n_rows}
                if bad_optional:
                    raise RuntimeError(
                        "Optional metadata buffer length mismatch before shard flush: "
                        f"{bad_optional}; expected all={n_rows}"
                    )

                payload["detect_bbox_xyxy_px"] = np.stack(detect_bbox_buffer, axis=0).astype(np.float32)
                payload["frame_wh_px"] = np.stack(frame_wh_buffer, axis=0).astype(np.int32)
                payload["roi_request_xyxy_px"] = np.stack(roi_request_buffer, axis=0).astype(np.float32)
                payload["roi_source_xyxy_px"] = np.stack(roi_source_buffer, axis=0).astype(np.float32)
                payload["roi_canvas_insert_xyxy_px"] = np.stack(roi_canvas_insert_buffer, axis=0).astype(np.float32)
                payload["roi_canvas_wh_px"] = np.stack(roi_canvas_wh_buffer, axis=0).astype(np.int32)
                payload["roi_padding_px"] = np.asarray(roi_padding_buffer, dtype=np.float32)
                payload["roi_geometry_schema"] = np.asarray(
                    (
                        "request_xyxy_px",
                        "source_xyxy_px",
                        "canvas_insert_xyxy_px",
                        "canvas_wh_px",
                        "padding_px",
                    ),
                    dtype=str,
                )

            if config.include_v1_compat_arrays:
                payload["X"] = np.stack(compat_x_buffer, axis=0).astype(np.float32)
                payload["y"] = y_distance_m.copy()

            if not config.dry_run:
                save_fn = np.savez_compressed if config.compress else np.savez
                save_fn(npz_path, **payload)
                validate_tri_stream_npz_file(
                    npz_path,
                    require_v1_compat_arrays=bool(config.include_v1_compat_arrays),
                )

            written_npz_paths.append(npz_path)

            for row_pos, buffered_row_idx in enumerate(row_indices_buffer):
                samples_df.at[buffered_row_idx, "pack_tri_stream_stage_status"] = "success"
                samples_df.at[buffered_row_idx, "pack_tri_stream_stage_error"] = ""
                samples_df.at[buffered_row_idx, "npz_filename"] = npz_name
                samples_df.at[buffered_row_idx, "npz_row_index"] = int(row_pos)

            deleted_npy_count = 0
            if stage_via_intermediate_npy and delete_source_npy_after_pack and not config.dry_run:
                for npy_path in [*distance_npy_paths_buffer, *orientation_npy_paths_buffer]:
                    try:
                        npy_path.unlink()
                        deleted_npy_count += 1
                    except FileNotFoundError:
                        continue
                    except Exception as exc:
                        logger.log(f"Could not delete staged NPY '{npy_path.name}': {exc}")

            logger.log(
                f"Wrote tri-stream NPZ shard '{npz_name}' rows={x_distance_image.shape[0]}, "
                f"canvas={canvas_h}x{canvas_w}, compression={'on' if config.compress else 'off'}"
                + (
                    f", staged_npy_deleted={deleted_npy_count}"
                    if stage_via_intermediate_npy and delete_source_npy_after_pack
                    else ""
                )
            )

            current_shard_idx += 1
            return True
        except Exception as exc:
            _mark_buffer_rows_failed(f"NPZ shard write/validate failed: {exc}")
            logger.log(f"NPZ shard write/validate failed for '{npz_name}': {exc}")
            return False
        finally:
            _reset_buffers()

    for row_idx in samples_df.index:
        samples_df.at[row_idx, "pack_tri_stream_stage_status"] = ""
        samples_df.at[row_idx, "pack_tri_stream_stage_error"] = ""
        samples_df.at[row_idx, "npz_filename"] = ""
        samples_df.at[row_idx, "npz_row_index"] = pd.NA
        samples_df.at[row_idx, "tri_stream_canvas_width_px"] = int(canvas_w)
        samples_df.at[row_idx, "tri_stream_canvas_height_px"] = int(canvas_h)
        if brightness_active:
            samples_df.at[row_idx, "brightness_normalization_enabled"] = True
            samples_df.at[row_idx, "brightness_normalization_method"] = brightness_method
            samples_df.at[row_idx, "brightness_normalization_status"] = ""
            samples_df.at[row_idx, "brightness_normalization_foreground_px"] = pd.NA
            samples_df.at[row_idx, "brightness_normalization_current_median_darkness"] = pd.NA
            samples_df.at[row_idx, "brightness_normalization_effective_median_darkness"] = pd.NA
            samples_df.at[row_idx, "brightness_normalization_gain"] = pd.NA

        if row_idx not in selected_rows:
            samples_df.at[row_idx, "pack_tri_stream_stage_status"] = "skipped"
            samples_df.at[row_idx, "pack_tri_stream_stage_error"] = "outside selected subset"
            continue

        if not bool(capture_mask.loc[row_idx]):
            samples_df.at[row_idx, "pack_tri_stream_stage_status"] = "skipped"
            samples_df.at[row_idx, "pack_tri_stream_stage_error"] = "capture_success is false"
            processed_selected += 1
            progress_skipped += 1
            _log_progress_if_needed()
            continue

        detect_status = str(samples_df.at[row_idx, "detect_stage_status"]).strip().lower()
        silhouette_status = str(samples_df.at[row_idx, "silhouette_stage_status"]).strip().lower()
        if detect_status != "success" or silhouette_status != "success":
            samples_df.at[row_idx, "pack_tri_stream_stage_status"] = "skipped"
            samples_df.at[row_idx, "pack_tri_stream_stage_error"] = "upstream stage status is not success"
            processed_selected += 1
            progress_skipped += 1
            _log_progress_if_needed()
            continue

        try:
            row_payload = _build_tri_stream_row_payload(
                project_root=project_root,
                run_name=run_name,
                samples_df=samples_df,
                row_idx=int(row_idx),
                canvas_w=canvas_w,
                canvas_h=canvas_h,
                clip_policy=clip_policy,
                orientation_context_scale=orientation_context_scale,
                brightness_active=brightness_active,
                brightness_config=brightness_config,
                include_v1_compat_arrays=bool(config.include_v1_compat_arrays),
            )
            if brightness_active and row_payload["brightness_result"] is not None:
                brightness_result = row_payload["brightness_result"]
                samples_df.at[row_idx, "brightness_normalization_status"] = brightness_result.status
                samples_df.at[row_idx, "brightness_normalization_foreground_px"] = int(
                    brightness_result.foreground_pixel_count
                )
                samples_df.at[row_idx, "brightness_normalization_current_median_darkness"] = float(
                    brightness_result.current_median_darkness
                )
                samples_df.at[row_idx, "brightness_normalization_effective_median_darkness"] = float(
                    brightness_result.effective_median_darkness
                )
                samples_df.at[row_idx, "brightness_normalization_gain"] = float(brightness_result.gain)
            if bool(row_payload["distance_clipped"]) and clip_policy == "clip":
                samples_df.at[row_idx, "pack_tri_stream_stage_error"] = "clipped_to_canvas"

            x_geometry = row_payload["x_geometry"]
            _write_bbox_feature_columns(samples_df, row_idx=int(row_idx), bbox_features=x_geometry)
            yaw_deg, yaw_sin, yaw_cos = row_payload["yaw_targets"]
            samples_df.at[row_idx, "yaw_deg"] = float(yaw_deg)
            samples_df.at[row_idx, "yaw_sin"] = float(yaw_sin)
            samples_df.at[row_idx, "yaw_cos"] = float(yaw_cos)

            if stage_via_intermediate_npy:
                distance_npy_path = stage_npy_dir / f"row_{int(row_idx):08d}_distance.npy"
                orientation_npy_path = stage_npy_dir / f"row_{int(row_idx):08d}_orientation.npy"
                distance_npy_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(distance_npy_path, row_payload["x_distance_image"][None, ...].astype(np.float32))
                np.save(orientation_npy_path, row_payload["x_orientation_image"][None, ...].astype(np.float32))
                distance_npy_paths_buffer.append(distance_npy_path)
                orientation_npy_paths_buffer.append(orientation_npy_path)
            else:
                distance_image_buffer.append(row_payload["x_distance_image"][None, ...])
                orientation_image_buffer.append(row_payload["x_orientation_image"][None, ...])
            geometry_buffer.append(x_geometry)
            y_dist_buffer.append(row_payload["y_distance"])
            y_yaw_deg_buffer.append(yaw_deg)
            y_yaw_sin_buffer.append(yaw_sin)
            y_yaw_cos_buffer.append(yaw_cos)
            sample_id_buffer.append(str(samples_df.at[row_idx, "sample_id"]))
            image_filename_buffer.append(str(samples_df.at[row_idx, "image_filename"]))
            detect_bbox_buffer.append(row_payload["detect_bbox_xyxy"])
            frame_wh_buffer.append(row_payload["frame_wh"])
            roi_request_buffer.append(row_payload["roi_request_xyxy"])
            roi_source_buffer.append(row_payload["roi_source_xyxy"])
            roi_canvas_insert_buffer.append(row_payload["roi_canvas_insert_xyxy"])
            roi_canvas_wh_buffer.append(np.asarray([int(canvas_w), int(canvas_h)], dtype=np.int32))
            roi_padding_buffer.append(row_payload["roi_padding"])
            row_indices_buffer.append(int(row_idx))

            compat_full_mask = row_payload["compat_full_mask"]
            if compat_full_mask is not None:
                compat_x_buffer.append(compat_full_mask)

            pending_error = str(samples_df.at[row_idx, "pack_tri_stream_stage_error"] or "")
            samples_df.at[row_idx, "pack_tri_stream_stage_status"] = "pending"
            samples_df.at[row_idx, "pack_tri_stream_stage_error"] = pending_error

            if shard_size > 0 and _buffered_rows_count() >= shard_size:
                buffered_count = _buffered_rows_count()
                shard_ok = _flush_shard()
                if shard_ok:
                    processed_selected += buffered_count
                    progress_success += buffered_count
                else:
                    processed_selected += buffered_count
                    progress_failed += buffered_count
                _log_progress_if_needed()
                if not shard_ok and not config.continue_on_error:
                    aborted = True
                    break
        except Exception as exc:
            samples_df.at[row_idx, "pack_tri_stream_stage_status"] = "failed"
            samples_df.at[row_idx, "pack_tri_stream_stage_error"] = str(exc)
            logger.log(f"Row {row_idx} failed: {exc}")
            processed_selected += 1
            progress_failed += 1
            _log_progress_if_needed()
            if not config.continue_on_error:
                aborted = True
                break

    if _buffered_rows_count() > 0:
        buffered_count = _buffered_rows_count()
        shard_ok = _flush_shard()
        if shard_ok:
            processed_selected += buffered_count
            progress_success += buffered_count
        else:
            processed_selected += buffered_count
            progress_failed += buffered_count
        _log_progress_if_needed()
        if not shard_ok and not config.continue_on_error:
            aborted = True

    output_samples_path = samples_csv_path(output_paths.manifests_dir)
    write_samples_csv(samples_df, output_samples_path, dry_run=config.dry_run)

    status_series = samples_df["pack_tri_stream_stage_status"].fillna("")
    successful_rows = int((status_series == "success").sum())
    failed_rows = int((status_series == "failed").sum())
    skipped_rows = int((status_series == "skipped").sum())

    output_ref = written_npz_paths[0] if len(written_npz_paths) == 1 else output_paths.root

    logger.log_summary(
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=output_ref,
    )
    if stage_via_intermediate_npy and delete_source_npy_after_pack and not config.dry_run:
        try:
            stage_npy_dir.rmdir()
        except OSError:
            pass
    logger.write()

    if aborted:
        raise RuntimeError("pack_tri_stream stage stopped after first row failure (continue_on_error=False).")

    return StageSummaryV4(
        run_name=run_name,
        stage_name="pack_tri_stream",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(output_ref),
        log_path=str(log_path),
        dry_run=config.dry_run,
    )


def build_tri_stream_sample_preview(
    project_root: Path,
    run_name: str,
    config: PackTriStreamStageConfigV4,
    *,
    row_index: int,
) -> dict[str, Any]:
    """Return preview arrays and metadata using the same row logic as the pack stage."""
    source_paths = silhouette_run_paths(project_root, run_name)
    samples_df = load_samples_csv(samples_csv_path(source_paths.manifests_dir))
    if int(row_index) < 0 or int(row_index) >= len(samples_df):
        raise IndexError(f"row_index {row_index} is outside samples.csv rows 0..{len(samples_df) - 1}")
    canvas_w = config.normalized_canvas_width_px()
    canvas_h = config.normalized_canvas_height_px()
    _require_tri_stream_canvas_matches_row(
        samples_df.iloc[int(row_index)],
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        context=f"preview row {int(row_index)}",
    )
    brightness_config = config.normalized_brightness_normalization()
    brightness_method = brightness_config.normalized_method()
    brightness_active = brightness_config.normalized_enabled() and brightness_method != "none"
    payload = _build_tri_stream_row_payload(
        project_root=project_root,
        run_name=run_name,
        samples_df=samples_df,
        row_idx=int(row_index),
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        clip_policy=config.normalized_clip_policy(),
        orientation_context_scale=config.normalized_orientation_context_scale(),
        brightness_active=brightness_active,
        brightness_config=brightness_config,
        include_v1_compat_arrays=False,
    )
    row = samples_df.iloc[int(row_index)].to_dict()
    return {
        "run_name": run_name,
        "row_index": int(row_index),
        "sample_id": str(row.get("sample_id", "")),
        "image_filename": str(row.get("image_filename", "")),
        "source_roi_canvas": payload["roi_source_gray"],
        "x_distance_image": payload["x_distance_image"],
        "x_orientation_image": payload["x_orientation_image"],
        "x_geometry": payload["x_geometry"],
        "roi_request_xyxy_px": payload["roi_request_xyxy"],
        "roi_source_xyxy_px": payload["roi_source_xyxy"],
        "roi_canvas_insert_xyxy_px": payload["roi_canvas_insert_xyxy"],
        "orientation_source_extent_xyxy_px": payload["orientation_source_extent_xyxy"],
        "orientation_crop_source_xyxy_px": payload["orientation_crop_source_xyxy"],
        "orientation_crop_size_px": float(payload["orientation_crop_size_px"]),
        "distance_clipped": bool(payload["distance_clipped"]),
        "expected_output_root": str(tri_stream_training_run_paths(project_root, run_name).root),
        "expected_array_keys": [
            TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
            TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,
            TRI_STREAM_GEOMETRY_ARRAY_KEY,
            "y_distance_m",
            "y_yaw_deg",
            "y_yaw_sin",
            "y_yaw_cos",
        ],
    }


def infer_tri_stream_silhouette_canvas_size(samples_df: pd.DataFrame) -> tuple[int, int]:
    """Infer the fixed source ROI canvas size from successful silhouette rows."""
    sizes: set[tuple[int, int]] = set()
    for row_idx in samples_df.index:
        row = samples_df.loc[row_idx]
        status = str(row.get("silhouette_stage_status", "")).strip().lower()
        if status and status != "success":
            continue
        size = _silhouette_canvas_size_from_row(row)
        if size is not None:
            sizes.add(size)

    if not sizes:
        raise ValueError(
            "Could not infer silhouette ROI canvas size from samples.csv. "
            "Run the silhouette stage first and ensure it writes silhouette ROI geometry columns."
        )
    if len(sizes) > 1:
        formatted = ", ".join(f"{width}x{height}" for width, height in sorted(sizes))
        raise ValueError(f"Silhouette ROI canvas size is inconsistent across successful rows: {formatted}.")
    return next(iter(sizes))


def _validate_tri_stream_canvas_matches_silhouette_canvas(
    samples_df: pd.DataFrame,
    *,
    selected_rows: set[int],
    capture_mask: pd.Series,
    canvas_width: int,
    canvas_height: int,
) -> list[str]:
    errors: list[str] = []
    for row_idx in samples_df.index:
        if int(row_idx) not in selected_rows:
            continue
        if row_idx in capture_mask.index and not bool(capture_mask.loc[row_idx]):
            continue
        detect_status = str(samples_df.at[row_idx, "detect_stage_status"]).strip().lower()
        silhouette_status = str(samples_df.at[row_idx, "silhouette_stage_status"]).strip().lower()
        if detect_status != "success" or silhouette_status != "success":
            continue
        try:
            _require_tri_stream_canvas_matches_row(
                samples_df.loc[row_idx],
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                context=f"row {int(row_idx)}",
            )
        except ValueError as exc:
            errors.append(str(exc))
            break
    return errors


def _require_tri_stream_canvas_matches_row(
    row: pd.Series,
    *,
    canvas_width: int,
    canvas_height: int,
    context: str,
) -> None:
    size = _silhouette_canvas_size_from_row(row)
    if size is None:
        raise ValueError(
            f"{context}: missing silhouette ROI canvas geometry; run the silhouette stage before tri-stream pack."
        )
    source_width, source_height = size
    if int(source_width) != int(canvas_width) or int(source_height) != int(canvas_height):
        raise ValueError(
            "tri-stream pack canvas must match silhouette ROI canvas so "
            "x_distance_image remains spatially unscaled: "
            f"{context} silhouette_canvas={source_width}x{source_height}, "
            f"configured_pack_canvas={int(canvas_width)}x{int(canvas_height)}. "
            f"Set Canvas W/H to {source_width}/{source_height}, or rerun silhouette with the desired fixed ROI canvas."
        )


def _silhouette_canvas_size_from_row(row: pd.Series) -> tuple[int, int] | None:
    width = _safe_float(row.get("silhouette_roi_canvas_width_px"))
    height = _safe_float(row.get("silhouette_roi_canvas_height_px"))
    if width is not None and height is not None and width > 0 and height > 0:
        return (int(round(width)), int(round(height)))

    req_x1 = _safe_float(row.get("silhouette_roi_request_x1_px"))
    req_y1 = _safe_float(row.get("silhouette_roi_request_y1_px"))
    req_x2 = _safe_float(row.get("silhouette_roi_request_x2_px"))
    req_y2 = _safe_float(row.get("silhouette_roi_request_y2_px"))
    if None in {req_x1, req_y1, req_x2, req_y2}:
        return None
    source_width = int(round(float(req_x2) - float(req_x1)))
    source_height = int(round(float(req_y2) - float(req_y1)))
    if source_width <= 0 or source_height <= 0:
        return None
    return (source_width, source_height)


def _build_tri_stream_row_payload(
    *,
    project_root: Path,
    run_name: str,
    samples_df: pd.DataFrame,
    row_idx: int,
    canvas_w: int,
    canvas_h: int,
    clip_policy: str,
    orientation_context_scale: float,
    brightness_active: bool,
    brightness_config: Any,
    include_v1_compat_arrays: bool,
) -> dict[str, Any]:
    input_paths = input_run_paths(project_root, run_name)
    source_paths = silhouette_run_paths(project_root, run_name)

    roi_rel = str(samples_df.at[row_idx, "silhouette_roi_image_filename"]).strip()
    full_rel = str(samples_df.at[row_idx, "silhouette_image_filename"]).strip()
    if not roi_rel:
        raise ValueError("silhouette_roi_image_filename is empty")
    if not full_rel and include_v1_compat_arrays:
        raise ValueError("silhouette_image_filename is empty but include_v1_compat_arrays=True")

    roi_path = resolve_manifest_path(source_paths.root, "images", roi_rel)
    roi_gray = read_grayscale_uint8(roi_path)
    silhouette_background_mask = _silhouette_to_background_mask(roi_gray)
    foreground_mask = (silhouette_background_mask < 0.5).astype(np.float32)

    source_image_rel = str(samples_df.at[row_idx, "image_filename"]).strip()
    source_image_path = resolve_manifest_path(input_paths.root, "images", source_image_rel)
    source_gray = read_grayscale_uint8(source_image_path)

    roi_request_xyxy, roi_source_xyxy, roi_canvas_insert_xyxy, roi_padding = _roi_geometry_from_row(
        samples_df.loc[row_idx],
        canvas_width=canvas_w,
        canvas_height=canvas_h,
    )

    roi_source_gray = _reconstruct_roi_canvas_from_source(
        source_gray,
        source_xyxy=roi_source_xyxy,
        canvas_insert_xyxy=roi_canvas_insert_xyxy,
        canvas_width=canvas_w,
        canvas_height=canvas_h,
    )
    roi_repr = _render_inverted_vehicle_detail_on_white(
        roi_source_gray,
        silhouette_background_mask,
    )
    orientation_repr = roi_repr
    brightness_result = None
    if brightness_active:
        brightness_result = apply_brightness_normalization_v4(
            roi_repr,
            foreground_mask.astype(bool),
            brightness_config,
        )
        roi_repr = brightness_result.image

    x_distance_image, distance_clipped = _place_image_on_canvas(
        roi_repr,
        canvas_height=canvas_h,
        canvas_width=canvas_w,
        clip_policy=clip_policy,
    )
    (
        x_orientation_image,
        orientation_source_extent_xyxy,
        orientation_crop_source_xyxy,
        orientation_crop_size_px,
    ) = _render_orientation_image_scaled_by_foreground_extent(
        orientation_repr,
        foreground_mask,
        canvas_height=canvas_h,
        canvas_width=canvas_w,
        context_scale=orientation_context_scale,
    )

    x_geometry = _bbox_features_from_row(samples_df.loc[row_idx])
    y_distance = np.float32(float(samples_df.at[row_idx, "distance_m"]))

    detect_bbox_xyxy = np.asarray(
        [
            float(samples_df.at[row_idx, "detect_bbox_x1"]),
            float(samples_df.at[row_idx, "detect_bbox_y1"]),
            float(samples_df.at[row_idx, "detect_bbox_x2"]),
            float(samples_df.at[row_idx, "detect_bbox_y2"]),
        ],
        dtype=np.float32,
    )
    frame_wh = np.asarray(
        [
            int(float(samples_df.at[row_idx, "image_width_px"])),
            int(float(samples_df.at[row_idx, "image_height_px"])),
        ],
        dtype=np.int32,
    )
    yaw_targets = _yaw_targets_from_row(samples_df.loc[row_idx])

    compat_full_mask: np.ndarray | None = None
    if include_v1_compat_arrays:
        full_path = resolve_manifest_path(source_paths.root, "images", full_rel)
        full_gray = read_grayscale_uint8(full_path)
        compat_full_mask = _silhouette_to_background_mask(full_gray)

    return {
        "roi_source_gray": roi_source_gray,
        "x_distance_image": x_distance_image.astype(np.float32, copy=False),
        "x_orientation_image": x_orientation_image.astype(np.float32, copy=False),
        "distance_clipped": bool(distance_clipped),
        "orientation_source_extent_xyxy": orientation_source_extent_xyxy,
        "orientation_crop_source_xyxy": orientation_crop_source_xyxy,
        "orientation_crop_size_px": np.float32(orientation_crop_size_px),
        "brightness_result": brightness_result,
        "x_geometry": x_geometry,
        "y_distance": y_distance,
        "detect_bbox_xyxy": detect_bbox_xyxy,
        "frame_wh": frame_wh,
        "yaw_targets": yaw_targets,
        "compat_full_mask": compat_full_mask,
        "roi_request_xyxy": roi_request_xyxy,
        "roi_source_xyxy": roi_source_xyxy,
        "roi_canvas_insert_xyxy": roi_canvas_insert_xyxy,
        "roi_padding": roi_padding,
    }


def _render_orientation_image_scaled_by_foreground_extent(
    orientation_source_image: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    canvas_height: int,
    canvas_width: int,
    context_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Render a target-centred crop with apparent target size normalised."""
    if orientation_source_image.ndim != 2:
        raise ValueError(f"orientation_source_image must be 2D, got {orientation_source_image.shape}")
    if foreground_mask.ndim != 2:
        raise ValueError(f"foreground_mask must be 2D, got {foreground_mask.shape}")
    if orientation_source_image.shape != foreground_mask.shape:
        raise ValueError(
            "orientation source and foreground mask shape mismatch: "
            f"source={orientation_source_image.shape}, mask={foreground_mask.shape}"
        )
    source_u8 = _orientation_source_to_uint8(orientation_source_image)

    extent = _foreground_extent_xyxy(foreground_mask)
    x1, y1, x2, y2 = [float(value) for value in extent]
    extent_w = max(1.0, x2 - x1)
    extent_h = max(1.0, y2 - y1)
    center_x = x1 + 0.5 * extent_w
    center_y = y1 + 0.5 * extent_h
    crop_size = max(1.0, max(extent_w, extent_h) * float(context_scale))

    crop_x1 = center_x - 0.5 * crop_size
    crop_y1 = center_y - 0.5 * crop_size
    crop_x2 = crop_x1 + crop_size
    crop_y2 = crop_y1 + crop_size
    patch = _extract_square_patch_with_padding(
        source_u8,
        crop_x1=crop_x1,
        crop_y1=crop_y1,
        crop_size=crop_size,
        fill_value=255,
    )
    interpolation = cv2.INTER_AREA if patch.shape[0] > canvas_height or patch.shape[1] > canvas_width else cv2.INTER_LINEAR
    resized = cv2.resize(
        patch,
        (int(canvas_width), int(canvas_height)),
        interpolation=interpolation,
    )
    image = np.asarray(resized, dtype=np.float32) / 255.0
    return (
        image.astype(np.float32, copy=False),
        extent.astype(np.float32, copy=False),
        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2], dtype=np.float32),
        float(crop_size),
    )


def _orientation_source_to_uint8(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    if np.issubdtype(source.dtype, np.floating):
        finite_source = np.nan_to_num(source.astype(np.float32), nan=1.0, posinf=1.0, neginf=0.0)
        return np.clip(np.rint(finite_source * 255.0), 0, 255).astype(np.uint8)
    return np.clip(source, 0, 255).astype(np.uint8)


def _foreground_extent_xyxy(foreground_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(foreground_mask) > 0.5
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("cannot build orientation crop from empty foreground mask")
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _extract_square_patch_with_padding(
    image: np.ndarray,
    *,
    crop_x1: float,
    crop_y1: float,
    crop_size: float,
    fill_value: int,
) -> np.ndarray:
    source = np.asarray(image, dtype=np.uint8)
    source_h, source_w = int(source.shape[0]), int(source.shape[1])
    size = max(1, int(round(float(crop_size))))
    x1 = int(round(float(crop_x1)))
    y1 = int(round(float(crop_y1)))
    x2 = x1 + size
    y2 = y1 + size

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(source_w, x2)
    src_y2 = min(source_h, y2)

    patch = np.full((size, size), int(fill_value), dtype=np.uint8)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return patch

    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    patch[dst_y1:dst_y2, dst_x1:dst_x2] = source[src_y1:src_y2, src_x1:src_x2]
    return patch
