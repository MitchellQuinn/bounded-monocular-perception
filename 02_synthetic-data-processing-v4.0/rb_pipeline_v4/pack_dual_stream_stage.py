"""Stage 3 (v4): ROI grayscale + bbox metadata -> dual-stream NPZ shards."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Callable

import numpy as np
import pandas as pd

from .brightness_normalization import apply_brightness_normalization_v4
from .config import PackDualStreamStageConfigV4, StageSummaryV4
from .constants import (
    BBOX_FEATURE_SCHEMA,
    BRIGHTNESS_NORMALIZATION_COLUMNS,
    DETECT_STAGE_COLUMNS,
    FOREGROUND_ENHANCEMENT_COLUMNS,
    ORIENTATION_TARGET_COLUMNS,
    PACK_STAGE_COLUMNS,
    POSITION_TARGET_COLUMNS,
    SILHOUETTE_STAGE_COLUMNS,
    UNITY_REQUIRED_COLUMNS,
)
from .foreground_enhancement import apply_foreground_enhancement_v4
from .image_io import read_grayscale_uint8
from .logging_utils import StageLogger
from .manifest import (
    append_columns,
    copy_run_json,
    load_samples_csv,
    samples_csv_path,
    upsert_preprocessing_contract_v4,
    write_samples_csv,
)
from .paths import (
    ensure_run_dirs,
    input_run_paths,
    resolve_manifest_path,
    silhouette_run_paths,
    training_run_paths,
)
from .utils import selected_row_indices
from .validation import (
    PipelineValidationError,
    capture_success_mask,
    validate_dual_stream_npz_file,
    validate_required_columns,
    validate_run_structure,
)

_YAW_SOURCE_COLUMNS = ("yaw_deg", "final_rot_y_deg")



def run_pack_dual_stream_stage_v4(
    project_root: Path,
    run_name: str,
    config: PackDualStreamStageConfigV4,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV4:
    """Pack dual-stream training inputs into NPZ shards."""

    input_paths = input_run_paths(project_root, run_name)
    source_paths = silhouette_run_paths(project_root, run_name)
    output_paths = training_run_paths(project_root, run_name)

    validation_errors = validate_run_structure(source_paths, require_images=True)
    validation_errors.extend(validate_run_structure(input_paths, require_images=True))
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    source_samples_path = samples_csv_path(source_paths.manifests_dir)
    samples_df = load_samples_csv(source_samples_path)

    required_columns = UNITY_REQUIRED_COLUMNS + DETECT_STAGE_COLUMNS + SILHOUETTE_STAGE_COLUMNS + POSITION_TARGET_COLUMNS
    validation_errors.extend(validate_required_columns(samples_df, required_columns))
    validation_errors.extend(_validate_yaw_source_columns(samples_df))
    if validation_errors:
        raise PipelineValidationError("\n".join(validation_errors))

    append_columns(samples_df, PACK_STAGE_COLUMNS)
    for column in PACK_STAGE_COLUMNS:
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
    image_representation_mode = config.normalized_image_representation_mode()
    shard_size = config.normalized_shard_size()
    foreground_config = config.normalized_foreground_enhancement()
    foreground_contract = foreground_config.to_contract_dict()
    foreground_method = foreground_config.normalized_method()
    foreground_active = foreground_config.normalized_enabled() and foreground_method != "none"
    brightness_config = config.normalized_brightness_normalization()
    brightness_contract = brightness_config.to_contract_dict()
    brightness_method = brightness_config.normalized_method()
    brightness_active = brightness_config.normalized_enabled() and brightness_method != "none"
    stage_via_intermediate_npy = bool(config.use_intermediate_npy and not config.dry_run)
    delete_source_npy_after_pack = bool(config.delete_source_npy_after_pack)
    stage_npy_dir = (output_paths.arrays_dir or (output_paths.root / "arrays")) / "__pack_dual_stream_stage_tmp"

    ensure_run_dirs(output_paths, dry_run=config.dry_run)
    copy_run_json(source_paths.manifests_dir, output_paths.manifests_dir, dry_run=config.dry_run)

    upsert_preprocessing_contract_v4(
        output_paths.manifests_dir,
        stage_name="pack_dual_stream",
        stage_parameters={
            "CanvasWidth": int(canvas_w),
            "CanvasHeight": int(canvas_h),
            "ClipPolicy": clip_policy,
            "ImageRepresentationMode": _image_representation_mode_contract_value(image_representation_mode),
            "ForegroundEnhancement": foreground_contract,
            "IncludeV1CompatArrays": bool(config.include_v1_compat_arrays),
            "IncludeOptionalMetadataArrays": bool(config.include_optional_metadata_arrays),
            "UseIntermediateNpy": bool(config.use_intermediate_npy),
            "DeleteSourceNpyAfterPack": bool(config.delete_source_npy_after_pack),
            "Compress": bool(config.compress),
            "ShardSize": int(shard_size),
            "BrightnessNormalization": brightness_contract,
            "SampleOffset": config.normalized_sample_offset(),
            "SampleLimit": config.normalized_sample_limit(),
        },
        current_representation={
            "Kind": "dual_stream_npz",
            "StorageFormat": "npz",
            "ArrayKeys": [
                "silhouette_crop",
                "bbox_features",
                "y_position_3d",
                "y_distance_m",
                "y_yaw_deg",
                "y_yaw_sin",
                "y_yaw_cos",
            ],
            "TargetModes": ["position_3d", "scalar_distance", "orientation_yaw_deg_sincos"],
            "SilhouetteCropLayout": "N,C,H,W",
            "BBoxFeatureSchema": list(BBOX_FEATURE_SCHEMA),
            "BBoxFeatureDim": int(len(BBOX_FEATURE_SCHEMA)),
            "CanvasWidth": int(canvas_w),
            "CanvasHeight": int(canvas_h),
            "ImagePolarity": _image_polarity_contract_value(image_representation_mode),
            "ImageContent": _image_content_contract_value(
                image_representation_mode,
                foreground_active=foreground_active,
            ),
            "SilhouetteScaling": "disabled",
            "ForegroundEnhancement": foreground_contract,
            "BrightnessNormalization": brightness_contract,
        },
        dry_run=config.dry_run,
    )

    log_path = output_paths.manifests_dir / "pack_dual_stream_stage_log.txt"
    logger = StageLogger(
        stage_name="pack_dual_stream",
        run_name=run_name,
        log_path=log_path,
        dry_run=config.dry_run,
        sink=log_sink,
    )
    logger.log(f"Running v4 pack_dual_stream stage for run '{run_name}'")
    logger.log_parameters(config.to_log_dict())

    if foreground_active:
        append_columns(samples_df, FOREGROUND_ENHANCEMENT_COLUMNS)
        for column in FOREGROUND_ENHANCEMENT_COLUMNS:
            samples_df[column] = samples_df[column].astype("object")
        logger.log(
            "Foreground enhancement enabled: "
            f"method={foreground_method}, "
            f"target_median_darkness={foreground_config.normalized_target_median_darkness():.6g}, "
            f"gain=[{foreground_config.normalized_min_gain():.6g}, "
            f"{foreground_config.normalized_max_gain():.6g}], "
            f"empty_mask_policy={foreground_config.normalized_empty_mask_policy()}"
        )
    else:
        logger.log("Foreground enhancement disabled.")

    if brightness_active:
        append_columns(samples_df, BRIGHTNESS_NORMALIZATION_COLUMNS)
        for column in BRIGHTNESS_NORMALIZATION_COLUMNS:
            samples_df[column] = samples_df[column].astype("object")
        logger.log(
            "Brightness normalization enabled: "
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
            f"Found existing NPZ outputs for run '{run_name}'. Enable overwrite=True to replace them."
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
            "Intermediate NPY staging enabled for pack shard writes "
            f"(delete_after_pack={'yes' if delete_source_npy_after_pack else 'no'})"
        )
    elif bool(config.use_intermediate_npy) and config.dry_run:
        logger.log("Intermediate NPY staging requested but dry_run=True; using in-memory shard buffers.")
    else:
        logger.log("Intermediate NPY staging disabled; using in-memory shard buffers.")

    silhouette_buffer: list[np.ndarray] = []
    silhouette_npy_paths_buffer: list[Path] = []
    bbox_buffer: list[np.ndarray] = []
    y3_buffer: list[np.ndarray] = []
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
    logger.log(f"Selected rows for pack_dual_stream processing: {selected_total} / {len(samples_df)}")

    def _log_progress_if_needed() -> None:
        if (
            selected_total > 0
            and (
                processed_selected % progress_interval == 0
                or processed_selected == selected_total
            )
        ):
            logger.log(
                "Progress pack_dual_stream: "
                f"{processed_selected}/{selected_total} selected rows "
                f"(success={progress_success}, failed={progress_failed}, skipped={progress_skipped})"
            )

    def _buffered_rows_count() -> int:
        return int(len(silhouette_npy_paths_buffer) if stage_via_intermediate_npy else len(silhouette_buffer))

    def _reset_buffers() -> None:
        silhouette_buffer.clear()
        silhouette_npy_paths_buffer.clear()
        bbox_buffer.clear()
        y3_buffer.clear()
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
            samples_df.at[buffered_row_idx, "pack_dual_stream_stage_status"] = "failed"
            samples_df.at[buffered_row_idx, "pack_dual_stream_stage_error"] = error_message
            samples_df.at[buffered_row_idx, "npz_filename"] = ""
            samples_df.at[buffered_row_idx, "npz_row_index"] = pd.NA

    def _flush_shard() -> bool:
        nonlocal current_shard_idx

        if _buffered_rows_count() == 0:
            return True

        npz_name = _shard_filename(run_name, current_shard_idx, use_shards=(shard_size > 0))
        npz_path = output_paths.root / npz_name

        try:
            if stage_via_intermediate_npy:
                staged_arrays: list[np.ndarray] = []
                for npy_path in silhouette_npy_paths_buffer:
                    staged = np.load(npy_path, allow_pickle=False)
                    if staged.ndim != 3 or int(staged.shape[0]) != 1:
                        raise ValueError(
                            f"Unexpected staged silhouette array shape {staged.shape} for '{npy_path.name}'. "
                            "Expected (1, H, W)."
                        )
                    staged_arrays.append(np.asarray(staged, dtype=np.float32))
                silhouette_crop = np.stack(staged_arrays, axis=0).astype(np.float32)
            else:
                silhouette_crop = np.stack(silhouette_buffer, axis=0).astype(np.float32)
            bbox_features = np.stack(bbox_buffer, axis=0).astype(np.float32)
            y_position_3d = np.stack(y3_buffer, axis=0).astype(np.float32)
            y_distance_m = np.asarray(y_dist_buffer, dtype=np.float32)
            y_yaw_deg = np.asarray(y_yaw_deg_buffer, dtype=np.float32)
            y_yaw_sin = np.asarray(y_yaw_sin_buffer, dtype=np.float32)
            y_yaw_cos = np.asarray(y_yaw_cos_buffer, dtype=np.float32)
            sample_id = np.asarray(sample_id_buffer, dtype=str)
            image_filename = np.asarray(image_filename_buffer, dtype=str)
            npz_row_index = np.arange(silhouette_crop.shape[0], dtype=np.int64)

            payload: dict[str, np.ndarray] = {
                "silhouette_crop": silhouette_crop,
                "bbox_features": bbox_features,
                "y_position_3d": y_position_3d,
                "y_distance_m": y_distance_m,
                "y_yaw_deg": y_yaw_deg,
                "y_yaw_sin": y_yaw_sin,
                "y_yaw_cos": y_yaw_cos,
                "sample_id": sample_id,
                "image_filename": image_filename,
                "npz_row_index": npz_row_index,
            }

            if config.include_optional_metadata_arrays:
                n_rows = int(silhouette_crop.shape[0])
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
                payload["bbox_features_schema"] = np.asarray(BBOX_FEATURE_SCHEMA, dtype=str)
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
                validate_dual_stream_npz_file(
                    npz_path,
                    require_v1_compat_arrays=bool(config.include_v1_compat_arrays),
                )

            written_npz_paths.append(npz_path)

            for row_pos, buffered_row_idx in enumerate(row_indices_buffer):
                samples_df.at[buffered_row_idx, "pack_dual_stream_stage_status"] = "success"
                samples_df.at[buffered_row_idx, "pack_dual_stream_stage_error"] = ""
                samples_df.at[buffered_row_idx, "npz_filename"] = npz_name
                samples_df.at[buffered_row_idx, "npz_row_index"] = int(row_pos)

            deleted_npy_count = 0
            if stage_via_intermediate_npy and delete_source_npy_after_pack and not config.dry_run:
                for npy_path in silhouette_npy_paths_buffer:
                    try:
                        npy_path.unlink()
                        deleted_npy_count += 1
                    except FileNotFoundError:
                        continue
                    except Exception as exc:
                        logger.log(f"Could not delete staged NPY '{npy_path.name}': {exc}")

            logger.log(
                f"Wrote NPZ shard '{npz_name}' rows={silhouette_crop.shape[0]}, "
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
        samples_df.at[row_idx, "pack_dual_stream_stage_status"] = ""
        samples_df.at[row_idx, "pack_dual_stream_stage_error"] = ""
        samples_df.at[row_idx, "npz_filename"] = ""
        samples_df.at[row_idx, "npz_row_index"] = pd.NA
        samples_df.at[row_idx, "canvas_width_px"] = int(canvas_w)
        samples_df.at[row_idx, "canvas_height_px"] = int(canvas_h)
        if foreground_active:
            samples_df.at[row_idx, "foreground_enhancement_enabled"] = True
            samples_df.at[row_idx, "foreground_enhancement_method"] = foreground_method
            samples_df.at[row_idx, "foreground_enhancement_status"] = ""
            samples_df.at[row_idx, "foreground_enhancement_foreground_px"] = pd.NA
            samples_df.at[row_idx, "foreground_enhancement_current_median_darkness"] = pd.NA
            samples_df.at[row_idx, "foreground_enhancement_effective_median_darkness"] = pd.NA
            samples_df.at[row_idx, "foreground_enhancement_gain"] = pd.NA
        if brightness_active:
            samples_df.at[row_idx, "brightness_normalization_enabled"] = True
            samples_df.at[row_idx, "brightness_normalization_method"] = brightness_method
            samples_df.at[row_idx, "brightness_normalization_status"] = ""
            samples_df.at[row_idx, "brightness_normalization_foreground_px"] = pd.NA
            samples_df.at[row_idx, "brightness_normalization_current_median_darkness"] = pd.NA
            samples_df.at[row_idx, "brightness_normalization_effective_median_darkness"] = pd.NA
            samples_df.at[row_idx, "brightness_normalization_gain"] = pd.NA

        if row_idx not in selected_rows:
            samples_df.at[row_idx, "pack_dual_stream_stage_status"] = "skipped"
            samples_df.at[row_idx, "pack_dual_stream_stage_error"] = "outside selected subset"
            continue

        if not bool(capture_mask.loc[row_idx]):
            samples_df.at[row_idx, "pack_dual_stream_stage_status"] = "skipped"
            samples_df.at[row_idx, "pack_dual_stream_stage_error"] = "capture_success is false"
            processed_selected += 1
            progress_skipped += 1
            _log_progress_if_needed()
            continue

        detect_status = str(samples_df.at[row_idx, "detect_stage_status"]).strip().lower()
        silhouette_status = str(samples_df.at[row_idx, "silhouette_stage_status"]).strip().lower()
        if detect_status != "success" or silhouette_status != "success":
            samples_df.at[row_idx, "pack_dual_stream_stage_status"] = "skipped"
            samples_df.at[row_idx, "pack_dual_stream_stage_error"] = "upstream stage status is not success"
            processed_selected += 1
            progress_skipped += 1
            _log_progress_if_needed()
            continue

        try:
            roi_rel = str(samples_df.at[row_idx, "silhouette_roi_image_filename"]).strip()
            full_rel = str(samples_df.at[row_idx, "silhouette_image_filename"]).strip()
            if not roi_rel:
                raise ValueError("silhouette_roi_image_filename is empty")
            if not full_rel and config.include_v1_compat_arrays:
                raise ValueError("silhouette_image_filename is empty but include_v1_compat_arrays=True")

            roi_path = resolve_manifest_path(source_paths.root, "images", roi_rel)
            roi_gray = read_grayscale_uint8(roi_path)
            silhouette_background_mask = _silhouette_to_background_mask(roi_gray)

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
            roi_repr = _render_vehicle_detail_on_white(
                roi_source_gray,
                silhouette_background_mask,
                image_representation_mode=image_representation_mode,
            )
            if foreground_active:
                foreground_result = apply_foreground_enhancement_v4(
                    roi_repr,
                    silhouette_background_mask < 0.5,
                    foreground_config,
                )
                roi_repr = foreground_result.image
                samples_df.at[row_idx, "foreground_enhancement_status"] = foreground_result.status
                samples_df.at[row_idx, "foreground_enhancement_foreground_px"] = int(
                    foreground_result.foreground_pixel_count
                )
                samples_df.at[row_idx, "foreground_enhancement_current_median_darkness"] = float(
                    foreground_result.current_median_darkness
                )
                samples_df.at[row_idx, "foreground_enhancement_effective_median_darkness"] = float(
                    foreground_result.effective_median_darkness
                )
                samples_df.at[row_idx, "foreground_enhancement_gain"] = float(foreground_result.gain)
            if brightness_active:
                brightness_result = apply_brightness_normalization_v4(
                    roi_repr,
                    silhouette_background_mask < 0.5,
                    brightness_config,
                )
                roi_repr = brightness_result.image
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

            canvas, clipped = _place_image_on_canvas(
                roi_repr,
                canvas_height=canvas_h,
                canvas_width=canvas_w,
                clip_policy=clip_policy,
            )
            if clipped and clip_policy == "clip":
                samples_df.at[row_idx, "pack_dual_stream_stage_error"] = "clipped_to_canvas"

            bbox_features = _bbox_features_from_row(samples_df.loc[row_idx])
            _write_bbox_feature_columns(samples_df, row_idx=row_idx, bbox_features=bbox_features)

            y_position_3d = np.asarray(
                [
                    float(samples_df.at[row_idx, "final_pos_x_m"]),
                    float(samples_df.at[row_idx, "final_pos_y_m"]),
                    float(samples_df.at[row_idx, "final_pos_z_m"]),
                ],
                dtype=np.float32,
            )
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
            yaw_deg, yaw_sin, yaw_cos = _yaw_targets_from_row(samples_df.loc[row_idx])
            samples_df.at[row_idx, "yaw_deg"] = float(yaw_deg)
            samples_df.at[row_idx, "yaw_sin"] = float(yaw_sin)
            samples_df.at[row_idx, "yaw_cos"] = float(yaw_cos)
            compat_full_mask: np.ndarray | None = None
            if config.include_v1_compat_arrays:
                full_path = resolve_manifest_path(source_paths.root, "images", full_rel)
                full_gray = read_grayscale_uint8(full_path)
                compat_full_mask = _silhouette_to_background_mask(full_gray)

            if stage_via_intermediate_npy:
                stage_npy_path = stage_npy_dir / f"row_{int(row_idx):08d}.npy"
                stage_npy_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(stage_npy_path, canvas[None, ...].astype(np.float32))
                silhouette_npy_paths_buffer.append(stage_npy_path)
            else:
                silhouette_buffer.append(canvas[None, ...])
            bbox_buffer.append(bbox_features)
            y3_buffer.append(y_position_3d)
            y_dist_buffer.append(y_distance)
            y_yaw_deg_buffer.append(yaw_deg)
            y_yaw_sin_buffer.append(yaw_sin)
            y_yaw_cos_buffer.append(yaw_cos)
            sample_id_buffer.append(str(samples_df.at[row_idx, "sample_id"]))
            image_filename_buffer.append(str(samples_df.at[row_idx, "image_filename"]))
            detect_bbox_buffer.append(detect_bbox_xyxy)
            frame_wh_buffer.append(frame_wh)
            roi_request_buffer.append(roi_request_xyxy)
            roi_source_buffer.append(roi_source_xyxy)
            roi_canvas_insert_buffer.append(roi_canvas_insert_xyxy)
            roi_canvas_wh_buffer.append(np.asarray([int(canvas_w), int(canvas_h)], dtype=np.int32))
            roi_padding_buffer.append(np.float32(roi_padding))
            row_indices_buffer.append(int(row_idx))

            if compat_full_mask is not None:
                compat_x_buffer.append(compat_full_mask)

            samples_df.at[row_idx, "pack_dual_stream_stage_status"] = "pending"
            samples_df.at[row_idx, "pack_dual_stream_stage_error"] = ""

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
            samples_df.at[row_idx, "pack_dual_stream_stage_status"] = "failed"
            samples_df.at[row_idx, "pack_dual_stream_stage_error"] = str(exc)
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

    status_series = samples_df["pack_dual_stream_stage_status"].fillna("")
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
            # Keep non-empty staging dir on disk for debugging if any leftovers remain.
            pass
    logger.write()

    if aborted:
        raise RuntimeError("pack_dual_stream stage stopped after first row failure (continue_on_error=False).")

    return StageSummaryV4(
        run_name=run_name,
        stage_name="pack_dual_stream",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(output_ref),
        log_path=str(log_path),
        dry_run=config.dry_run,
    )



def _silhouette_to_background_mask(gray_image: np.ndarray) -> np.ndarray:
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {gray_image.shape}")
    white_count = int(np.count_nonzero(gray_image > 127))
    black_count = int(gray_image.size - white_count)

    # Model convention: white background (1.0), black silhouette (0.0).
    # If source appears white-on-black (white is minority), invert into model convention.
    if white_count <= black_count:
        return (gray_image < 128).astype(np.float32)
    return (gray_image > 127).astype(np.float32)



def _render_inverted_vehicle_detail_on_white(
    roi_source_gray: np.ndarray,
    background_mask: np.ndarray,
) -> np.ndarray:
    return _render_vehicle_detail_on_white(
        roi_source_gray,
        background_mask,
        image_representation_mode="inverted_vehicle_on_white",
    )


def _render_vehicle_detail_on_white(
    roi_source_gray: np.ndarray,
    background_mask: np.ndarray,
    *,
    image_representation_mode: str,
) -> np.ndarray:
    if roi_source_gray.ndim != 2:
        raise ValueError(f"Expected 2D grayscale ROI source image, got {roi_source_gray.shape}")
    if background_mask.ndim != 2:
        raise ValueError(f"Expected 2D background mask, got {background_mask.shape}")
    if roi_source_gray.shape != background_mask.shape:
        raise ValueError(
            "ROI source and silhouette mask shape mismatch: "
            f"source={roi_source_gray.shape}, mask={background_mask.shape}"
        )

    mode = str(image_representation_mode).strip().lower()
    if mode not in {"inverted_vehicle_on_white", "raw_grayscale_on_white"}:
        raise ValueError(f"Unsupported image representation mode: {image_representation_mode}")

    source_u8 = np.clip(roi_source_gray, 0, 255).astype(np.uint8)
    vehicle_detail = 255 - source_u8 if mode == "inverted_vehicle_on_white" else source_u8
    out_u8 = np.full(source_u8.shape, 255, dtype=np.uint8)
    vehicle_mask = background_mask < 0.5
    out_u8[vehicle_mask] = vehicle_detail[vehicle_mask]
    return out_u8.astype(np.float32) / 255.0


def _image_representation_mode_contract_value(image_representation_mode: str) -> str:
    mode = str(image_representation_mode).strip().lower()
    if mode == "inverted_vehicle_on_white":
        return "roi_grayscale_inverted_vehicle_on_white"
    if mode == "raw_grayscale_on_white":
        return "roi_raw_grayscale_vehicle_on_white"
    raise ValueError(f"Unsupported image representation mode: {image_representation_mode}")


def _image_polarity_contract_value(image_representation_mode: str) -> str:
    mode = str(image_representation_mode).strip().lower()
    if mode == "inverted_vehicle_on_white":
        return "dark_vehicle_detail_on_white_background"
    if mode == "raw_grayscale_on_white":
        return "source_grayscale_vehicle_detail_on_white_background"
    raise ValueError(f"Unsupported image representation mode: {image_representation_mode}")


def _image_content_contract_value(
    image_representation_mode: str,
    *,
    foreground_active: bool,
) -> str:
    mode = str(image_representation_mode).strip().lower()
    suffix = "_foreground_enhanced" if foreground_active else ""
    if mode == "inverted_vehicle_on_white":
        return f"inverted_grayscale_vehicle_detail_inside_silhouette{suffix}"
    if mode == "raw_grayscale_on_white":
        return f"raw_grayscale_vehicle_detail_inside_silhouette{suffix}"
    raise ValueError(f"Unsupported image representation mode: {image_representation_mode}")



def _place_image_on_canvas(
    image: np.ndarray,
    *,
    canvas_height: int,
    canvas_width: int,
    clip_policy: str,
) -> tuple[np.ndarray, bool]:
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    clipped = src_h > canvas_height or src_w > canvas_width
    if clipped and clip_policy == "fail":
        raise ValueError(
            f"ROI image {src_h}x{src_w} exceeds canvas {canvas_height}x{canvas_width}"
        )

    src_y0 = max(0, (src_h - canvas_height) // 2)
    src_x0 = max(0, (src_w - canvas_width) // 2)
    src_y1 = min(src_h, src_y0 + canvas_height)
    src_x1 = min(src_w, src_x0 + canvas_width)

    cropped = image[src_y0:src_y1, src_x0:src_x1]

    out = np.ones((canvas_height, canvas_width), dtype=np.float32)
    dst_y0 = max(0, (canvas_height - cropped.shape[0]) // 2)
    dst_x0 = max(0, (canvas_width - cropped.shape[1]) // 2)
    dst_y1 = dst_y0 + cropped.shape[0]
    dst_x1 = dst_x0 + cropped.shape[1]
    out[dst_y0:dst_y1, dst_x0:dst_x1] = np.asarray(cropped, dtype=np.float32)

    return out, clipped



def _reconstruct_roi_canvas_from_source(
    source_gray: np.ndarray,
    *,
    source_xyxy: np.ndarray,
    canvas_insert_xyxy: np.ndarray,
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    if source_gray.ndim != 2:
        raise ValueError(f"Expected 2D source grayscale image, got {source_gray.shape}")

    src_x1, src_y1, src_x2, src_y2 = _xyxy_to_int_bounds(source_xyxy, name="roi_source_xyxy_px")
    ins_x1, ins_y1, ins_x2, ins_y2 = _xyxy_to_int_bounds(
        canvas_insert_xyxy,
        name="roi_canvas_insert_xyxy_px",
    )

    source_h, source_w = int(source_gray.shape[0]), int(source_gray.shape[1])
    if src_x1 < 0 or src_y1 < 0 or src_x2 > source_w or src_y2 > source_h:
        raise ValueError(
            "roi_source_xyxy_px is outside source image bounds: "
            f"source={source_w}x{source_h}, bounds={[src_x1, src_y1, src_x2, src_y2]}"
        )

    canvas_w = int(canvas_width)
    canvas_h = int(canvas_height)
    if ins_x1 < 0 or ins_y1 < 0 or ins_x2 > canvas_w or ins_y2 > canvas_h:
        raise ValueError(
            "roi_canvas_insert_xyxy_px is outside ROI canvas bounds: "
            f"canvas={canvas_w}x{canvas_h}, bounds={[ins_x1, ins_y1, ins_x2, ins_y2]}"
        )

    source_patch = np.asarray(source_gray[src_y1:src_y2, src_x1:src_x2], dtype=np.uint8)
    expected_h = int(ins_y2 - ins_y1)
    expected_w = int(ins_x2 - ins_x1)
    if source_patch.shape != (expected_h, expected_w):
        raise ValueError(
            "ROI source patch shape does not match canvas insertion geometry: "
            f"source_patch={source_patch.shape}, expected={(expected_h, expected_w)}"
        )

    roi_canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    roi_canvas[ins_y1:ins_y2, ins_x1:ins_x2] = source_patch
    return roi_canvas



def _xyxy_to_int_bounds(bounds_xyxy: np.ndarray, *, name: str) -> tuple[int, int, int, int]:
    if bounds_xyxy.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {bounds_xyxy.shape}")

    x1 = int(round(float(bounds_xyxy[0])))
    y1 = int(round(float(bounds_xyxy[1])))
    x2 = int(round(float(bounds_xyxy[2])))
    y2 = int(round(float(bounds_xyxy[3])))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} is degenerate: {[x1, y1, x2, y2]}")
    return x1, y1, x2, y2



def _validate_yaw_source_columns(samples_df: pd.DataFrame) -> list[str]:
    for column in _YAW_SOURCE_COLUMNS:
        if column in samples_df.columns:
            return []
    return [
        "Missing yaw source column for orientation targets; expected one of: "
        + ", ".join(_YAW_SOURCE_COLUMNS)
    ]



def _yaw_targets_from_row(row: pd.Series) -> tuple[np.float32, np.float32, np.float32]:
    yaw_deg = None
    for column in _YAW_SOURCE_COLUMNS:
        yaw_deg = _safe_float(row.get(column))
        if yaw_deg is not None:
            break
    if yaw_deg is None:
        raise ValueError("missing yaw source value in row")

    yaw_rad = np.deg2rad(float(yaw_deg))
    yaw_sin = float(np.sin(yaw_rad))
    yaw_cos = float(np.cos(yaw_rad))
    return np.float32(float(yaw_deg)), np.float32(yaw_sin), np.float32(yaw_cos)



def _roi_geometry_from_row(
    row: pd.Series,
    *,
    canvas_width: int,
    canvas_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    req_x1 = _safe_float(row.get("silhouette_roi_request_x1_px"))
    req_y1 = _safe_float(row.get("silhouette_roi_request_y1_px"))
    req_x2 = _safe_float(row.get("silhouette_roi_request_x2_px"))
    req_y2 = _safe_float(row.get("silhouette_roi_request_y2_px"))

    src_x1 = _safe_float(row.get("silhouette_roi_source_x1_px"))
    src_y1 = _safe_float(row.get("silhouette_roi_source_y1_px"))
    src_x2 = _safe_float(row.get("silhouette_roi_source_x2_px"))
    src_y2 = _safe_float(row.get("silhouette_roi_source_y2_px"))

    ins_x1 = _safe_float(row.get("silhouette_roi_canvas_x1_px"))
    ins_y1 = _safe_float(row.get("silhouette_roi_canvas_y1_px"))
    ins_x2 = _safe_float(row.get("silhouette_roi_canvas_x2_px"))
    ins_y2 = _safe_float(row.get("silhouette_roi_canvas_y2_px"))

    if None in {req_x1, req_y1, req_x2, req_y2, src_x1, src_y1, src_x2, src_y2, ins_x1, ins_y1, ins_x2, ins_y2}:
        raise ValueError("missing silhouette ROI geometry columns for pack metadata")

    request_xyxy = np.asarray([req_x1, req_y1, req_x2, req_y2], dtype=np.float32)
    source_xyxy = np.asarray([src_x1, src_y1, src_x2, src_y2], dtype=np.float32)
    canvas_insert_xyxy = np.asarray([ins_x1, ins_y1, ins_x2, ins_y2], dtype=np.float32)

    if (
        int(round(req_x2 - req_x1)) != int(canvas_width)
        or int(round(req_y2 - req_y1)) != int(canvas_height)
    ):
        raise ValueError(
            "silhouette ROI request bounds do not match configured pack canvas: "
            f"request={request_xyxy.tolist()}, expected_canvas={canvas_width}x{canvas_height}"
        )

    roi_padding = _safe_float(row.get("silhouette_roi_padding_px"))
    if roi_padding is None:
        roi_padding = 0.0

    return request_xyxy, source_xyxy, canvas_insert_xyxy, np.float32(roi_padding)



def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return number



def _bbox_features_from_row(row: pd.Series) -> np.ndarray:
    x1 = float(row["detect_bbox_x1"])
    y1 = float(row["detect_bbox_y1"])
    x2 = float(row["detect_bbox_x2"])
    y2 = float(row["detect_bbox_y2"])

    frame_w = max(1.0, float(row["image_width_px"]))
    frame_h = max(1.0, float(row["image_height_px"]))

    w_px = max(1e-6, x2 - x1)
    h_px = max(1e-6, y2 - y1)
    cx_px = x1 + 0.5 * w_px
    cy_px = y1 + 0.5 * h_px

    cx_norm = cx_px / frame_w
    cy_norm = cy_px / frame_h
    w_norm = w_px / frame_w
    h_norm = h_px / frame_h
    aspect_ratio = w_px / h_px
    area_norm = (w_px * h_px) / (frame_w * frame_h)

    values = np.asarray(
        [
            cx_px,
            cy_px,
            w_px,
            h_px,
            cx_norm,
            cy_norm,
            w_norm,
            h_norm,
            aspect_ratio,
            area_norm,
        ],
        dtype=np.float32,
    )

    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("bbox feature vector contains NaN or Inf")

    return values



def _write_bbox_feature_columns(samples_df: pd.DataFrame, *, row_idx: int, bbox_features: np.ndarray) -> None:
    samples_df.at[row_idx, "bbox_feat_cx_px"] = float(bbox_features[0])
    samples_df.at[row_idx, "bbox_feat_cy_px"] = float(bbox_features[1])
    samples_df.at[row_idx, "bbox_feat_w_px"] = float(bbox_features[2])
    samples_df.at[row_idx, "bbox_feat_h_px"] = float(bbox_features[3])
    samples_df.at[row_idx, "bbox_feat_cx_norm"] = float(bbox_features[4])
    samples_df.at[row_idx, "bbox_feat_cy_norm"] = float(bbox_features[5])
    samples_df.at[row_idx, "bbox_feat_w_norm"] = float(bbox_features[6])
    samples_df.at[row_idx, "bbox_feat_h_norm"] = float(bbox_features[7])
    samples_df.at[row_idx, "bbox_feat_aspect_ratio"] = float(bbox_features[8])
    samples_df.at[row_idx, "bbox_feat_area_norm"] = float(bbox_features[9])



def _shard_filename(run_name: str, shard_idx: int, *, use_shards: bool) -> str:
    if use_shards:
        return f"{run_name}_shard_{shard_idx:05d}.npz"
    return f"{run_name}.npz"
