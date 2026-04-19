"""Stage 2: pack grayscale locator inputs plus geometry metadata into NPZ shards."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from .config import PackRoiFcnConfig
from .contracts import LOCATOR_GEOMETRY_SCHEMA, PACK_STAGE_COLUMNS, StageSummaryV01
from .external import ensure_external_paths
from .manifest import append_columns, load_samples_csv, upsert_preprocessing_contract, write_samples_csv
from .paths import SplitPaths, ensure_split_output_dirs, resolve_input_image_path
from .validation import (
    RoiFcnPreprocessingValidationError,
    ensure_pack_prerequisites,
    stage_summary_counts_from_samples,
    validate_capture_success_images,
    validate_input_split_structure,
    validate_roi_fcn_npz_file,
)

ensure_external_paths()

from rb_pipeline_v4.image_io import read_grayscale_uint8
from rb_pipeline_v4.logging_utils import StageLogger


_STATUS_COLUMN = "pack_roi_fcn_stage_status"
_ERROR_COLUMN = "pack_roi_fcn_stage_error"
_BOOTSTRAP_STATUS_COLUMN = "bootstrap_center_target_stage_status"
_PROGRESS_LOG_INTERVAL_ROWS = 500
_CHECKPOINT_WRITE_INTERVAL_ROWS = 2000
_MAX_IN_FLIGHT_MULTIPLIER = 4


@dataclass(frozen=True)
class _BufferedPackRow:
    row_idx: int
    locator_input_image: np.ndarray
    target_center_xy_original_px: np.ndarray
    target_center_xy_canvas_px: np.ndarray
    source_image_wh_px: np.ndarray
    resized_image_wh_px: np.ndarray
    padding_ltrb_px: np.ndarray
    resize_scale: np.float32
    sample_id: str
    image_filename: str
    bootstrap_bbox_xyxy_px: np.ndarray
    bootstrap_confidence: np.float32
    locator_canvas_width_px: int
    locator_canvas_height_px: int
    locator_resized_width_px: int
    locator_resized_height_px: int
    locator_pad_left_px: int
    locator_pad_right_px: int
    locator_pad_top_px: int
    locator_pad_bottom_px: int
    locator_center_x_px: float
    locator_center_y_px: float


@dataclass(frozen=True)
class _PackRowResult:
    sequence_idx: int
    row_idx: int
    status: str
    error_message: str
    packed: _BufferedPackRow | None = None


def run_pack_roi_fcn_stage(
    split_paths: SplitPaths,
    config: PackRoiFcnConfig,
    *,
    log_sink: Callable[[str], None] | None = None,
) -> StageSummaryV01:
    """Run stage 2 for one split."""

    validation_errors = validate_input_split_structure(split_paths)
    if validation_errors:
        raise RoiFcnPreprocessingValidationError("\n".join(validation_errors))
    if not split_paths.output_samples_csv_path.is_file():
        raise RoiFcnPreprocessingValidationError(
            f"Missing stage-1 samples.csv: {split_paths.output_samples_csv_path}"
        )
    if not split_paths.output_run_json_path.is_file():
        raise RoiFcnPreprocessingValidationError(
            f"Missing stage-1 run.json: {split_paths.output_run_json_path}"
        )

    source_check_df = load_samples_csv(split_paths.input_samples_csv_path)
    image_errors = validate_capture_success_images(source_check_df, split_paths)
    if image_errors:
        raise RoiFcnPreprocessingValidationError("\n".join(image_errors))

    samples_df = load_samples_csv(split_paths.output_samples_csv_path)
    ensure_pack_prerequisites(samples_df)

    append_columns(samples_df, PACK_STAGE_COLUMNS)
    for column in PACK_STAGE_COLUMNS:
        samples_df[column] = samples_df[column].astype("object")

    ensure_split_output_dirs(split_paths, dry_run=config.dry_run)
    _prepare_arrays_dir(split_paths.output_arrays_dir, overwrite=config.overwrite, dry_run=config.dry_run)
    write_samples_csv(samples_df, split_paths.output_samples_csv_path, dry_run=config.dry_run)

    canvas_width = config.normalized_canvas_width()
    canvas_height = config.normalized_canvas_height()
    shard_size = config.normalized_shard_size()
    requested_workers = config.normalized_num_workers()
    effective_workers = _effective_worker_count(config)

    log_path = split_paths.output_manifests_dir / "pack_roi_fcn_stage_log.txt"
    logger = StageLogger(
        stage_name="pack_roi_fcn",
        run_name=f"{split_paths.dataset_reference}/{split_paths.split_name}",
        log_path=log_path,
        dry_run=config.dry_run,
        sink=log_sink,
    )
    logger.log(
        "Running pack_roi_fcn for "
        f"dataset='{split_paths.dataset_reference}' split='{split_paths.split_name}'"
    )
    logger.log_parameters(config.to_log_dict())

    bootstrap_status_series = samples_df[_BOOTSTRAP_STATUS_COLUMN].fillna("").astype(str).str.strip().str.lower()
    eligible_rows = int((bootstrap_status_series == "success").sum())
    logger.log(f"Total input rows: {len(samples_df)}")
    logger.log(f"Rows eligible for packing from bootstrap success: {eligible_rows}")
    logger.log(f"Requested CPU workers: {requested_workers}; effective workers: {effective_workers}")
    if effective_workers != requested_workers:
        logger.log("continue_on_error=False forces pack_roi_fcn to run single-threaded for fail-fast behavior.")
    if effective_workers > 1:
        logger.log("OpenCV worker threads are capped at 1 per task to avoid CPU oversubscription.")
    logger.log(f"NPZ compression is {'enabled' if config.compress else 'disabled'}.")
    if shard_size > 0:
        logger.log(
            f"The first NPZ shard will appear after {shard_size} successful packed rows have been buffered."
        )
    else:
        logger.log("Shard size is 0; one NPZ file will be written at the end of the split.")
    logger.log(
        f"Progress will be logged every {_PROGRESS_LOG_INTERVAL_ROWS} rows; "
        f"samples.csv checkpoints every {_CHECKPOINT_WRITE_INTERVAL_ROWS} rows and after each shard write."
    )
    logger.write()

    buffer: list[_BufferedPackRow] = []
    written_npz_paths: list[Path] = []
    current_shard_idx = 0
    aborted = False
    processed_rows = 0
    progress_success = 0
    progress_failed = 0
    progress_skipped = 0
    started_at = time.perf_counter()

    def _log_progress(*, force: bool = False) -> None:
        if processed_rows <= 0:
            return
        if not force and processed_rows % _PROGRESS_LOG_INTERVAL_ROWS != 0:
            return
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        rows_per_sec = processed_rows / elapsed
        remaining_rows = max(len(samples_df) - processed_rows, 0)
        eta_seconds = (remaining_rows / rows_per_sec) if rows_per_sec > 0.0 else float("inf")
        logger.log(
            "Progress pack_roi_fcn: "
            f"{processed_rows}/{len(samples_df)} rows "
            f"(ready={progress_success}, failed={progress_failed}, skipped={progress_skipped}, "
            f"written={progress_success - len(buffer)}, buffered={len(buffer)}, "
            f"rate={rows_per_sec:.2f} rows/s, eta={_format_eta_seconds(eta_seconds)})"
        )
        logger.write()

    def _checkpoint_samples(*, force: bool = False, reason: str | None = None) -> None:
        if not force and processed_rows % _CHECKPOINT_WRITE_INTERVAL_ROWS != 0:
            return
        write_samples_csv(samples_df, split_paths.output_samples_csv_path, dry_run=config.dry_run)
        if reason is None:
            reason = f"{processed_rows}/{len(samples_df)} rows"
        logger.log(f"Checkpointed samples.csv at {reason}.")
        logger.write()

    def flush_buffer() -> bool:
        nonlocal current_shard_idx, progress_success, progress_failed
        if not buffer:
            return True

        buffered_count = len(buffer)
        npz_name = _shard_filename(
            split_paths.dataset_reference,
            split_paths.split_name,
            current_shard_idx,
            use_shards=(shard_size > 0),
        )
        npz_path = split_paths.output_arrays_dir / npz_name
        try:
            payload = {
                "locator_input_image": np.stack([item.locator_input_image for item in buffer], axis=0).astype(np.float32),
                "target_center_xy_original_px": np.stack(
                    [item.target_center_xy_original_px for item in buffer],
                    axis=0,
                ).astype(np.float32),
                "target_center_xy_canvas_px": np.stack(
                    [item.target_center_xy_canvas_px for item in buffer],
                    axis=0,
                ).astype(np.float32),
                "source_image_wh_px": np.stack([item.source_image_wh_px for item in buffer], axis=0).astype(np.int32),
                "resized_image_wh_px": np.stack([item.resized_image_wh_px for item in buffer], axis=0).astype(np.int32),
                "padding_ltrb_px": np.stack([item.padding_ltrb_px for item in buffer], axis=0).astype(np.int32),
                "resize_scale": np.asarray([item.resize_scale for item in buffer], dtype=np.float32),
                "sample_id": np.asarray([item.sample_id for item in buffer], dtype=np.str_),
                "image_filename": np.asarray([item.image_filename for item in buffer], dtype=np.str_),
                "npz_row_index": np.arange(len(buffer), dtype=np.int64),
                "bootstrap_bbox_xyxy_px": np.stack(
                    [item.bootstrap_bbox_xyxy_px for item in buffer],
                    axis=0,
                ).astype(np.float32),
                "bootstrap_confidence": np.asarray(
                    [item.bootstrap_confidence for item in buffer],
                    dtype=np.float32,
                ),
                "locator_geometry_schema": np.asarray(LOCATOR_GEOMETRY_SCHEMA, dtype=np.str_),
            }
            if config.dry_run:
                for local_row_index, item in enumerate(buffer):
                    _mark_row_success(samples_df, item, npz_name, local_row_index)
            else:
                if config.compress:
                    np.savez_compressed(npz_path, **payload)
                else:
                    np.savez(npz_path, **payload)
                validate_roi_fcn_npz_file(
                    npz_path,
                    expected_canvas_height=canvas_height,
                    expected_canvas_width=canvas_width,
                )
                written_npz_paths.append(npz_path)
                for local_row_index, item in enumerate(buffer):
                    _mark_row_success(samples_df, item, npz_name, local_row_index)
            logger.log(f"Wrote shard {npz_name} with {buffered_count} rows.")
            _checkpoint_samples(force=True, reason=f"shard write {npz_name}")
        except Exception as exc:
            if npz_path.exists() and not config.dry_run:
                npz_path.unlink()
            for item in buffer:
                _mark_row_failed(samples_df, item.row_idx, str(exc))
            progress_success -= buffered_count
            progress_failed += buffered_count
            logger.log(f"Failed to write shard {npz_name}: {exc}")
            logger.write()
            buffer.clear()
            return False

        current_shard_idx += 1
        buffer.clear()
        return True

    work_items: list[tuple[int, int, dict[str, object]]] = []
    for row_idx, row in samples_df.iterrows():
        bootstrap_status = str(row.get(_BOOTSTRAP_STATUS_COLUMN, "")).strip().lower()
        if bootstrap_status != "success":
            _mark_row_skipped(samples_df, row_idx)
            processed_rows += 1
            progress_skipped += 1
            continue
        work_items.append(
            (
                len(work_items),
                int(row_idx),
                {
                    "sample_id": row.get("sample_id", ""),
                    "image_filename": row.get("image_filename", ""),
                    "bootstrap_center_x_px": row.get("bootstrap_center_x_px"),
                    "bootstrap_center_y_px": row.get("bootstrap_center_y_px"),
                    "bootstrap_bbox_x1": row.get("bootstrap_bbox_x1"),
                    "bootstrap_bbox_y1": row.get("bootstrap_bbox_y1"),
                    "bootstrap_bbox_x2": row.get("bootstrap_bbox_x2"),
                    "bootstrap_bbox_y2": row.get("bootstrap_bbox_y2"),
                    "bootstrap_confidence": row.get("bootstrap_confidence"),
                },
            )
        )

    if processed_rows > 0:
        _log_progress(force=(processed_rows == len(samples_df)))
        _checkpoint_samples(force=(processed_rows == len(samples_df)))

    max_in_flight = max(1, effective_workers * _MAX_IN_FLIGHT_MULTIPLIER)
    next_commit_sequence = 0
    pending_results: dict[int, _PackRowResult] = {}

    if work_items and not aborted:
        with _opencv_single_thread_mode(enabled=(effective_workers > 1)):
            with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="roi-fcn-pack") as executor:
                work_iter = iter(work_items)
                in_flight: dict[object, int] = {}

                def submit_available() -> None:
                    while len(in_flight) < max_in_flight:
                        try:
                            sequence_idx, next_row_idx, row_payload = next(work_iter)
                        except StopIteration:
                            break
                        future = executor.submit(
                            _process_pack_row,
                            split_paths=split_paths,
                            sequence_idx=sequence_idx,
                            row_idx=next_row_idx,
                            row=row_payload,
                            canvas_width=canvas_width,
                            canvas_height=canvas_height,
                        )
                        in_flight[future] = next_row_idx

                submit_available()
                while in_flight:
                    done, _ = wait(tuple(in_flight.keys()), return_when=FIRST_COMPLETED)
                    for future in done:
                        row_idx = in_flight.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = _PackRowResult(
                                sequence_idx=-1,
                                row_idx=row_idx,
                                status="failed",
                                error_message=str(exc),
                            )
                        pending_results[result.sequence_idx] = result

                    while next_commit_sequence in pending_results:
                        result = pending_results.pop(next_commit_sequence)
                        next_commit_sequence += 1

                        if result.status == "success":
                            assert result.packed is not None
                            buffer.append(result.packed)
                            progress_success += 1
                            if shard_size > 0 and len(buffer) >= shard_size:
                                if not flush_buffer() and not config.continue_on_error:
                                    aborted = True
                        else:
                            _mark_row_failed(samples_df, result.row_idx, result.error_message)
                            progress_failed += 1
                            if not config.continue_on_error:
                                aborted = True
                                logger.log(f"Stopping after row {result.row_idx} failure: {result.error_message}")

                        processed_rows += 1
                        _log_progress(force=(processed_rows == len(samples_df)))
                        _checkpoint_samples(force=(processed_rows == len(samples_df) or aborted))
                        if aborted:
                            break

                    if aborted:
                        for future in in_flight:
                            future.cancel()
                        break
                    submit_available()

    if not aborted and buffer:
        if not flush_buffer() and not config.continue_on_error:
            aborted = True

    write_samples_csv(samples_df, split_paths.output_samples_csv_path, dry_run=config.dry_run)

    successful_rows, failed_rows, skipped_rows = stage_summary_counts_from_samples(samples_df, _STATUS_COLUMN)
    output_ref: Path = (
        written_npz_paths[0]
        if len(written_npz_paths) == 1
        else split_paths.output_arrays_dir
    )
    summary = StageSummaryV01(
        dataset_reference=split_paths.dataset_reference,
        split_name=split_paths.split_name,
        stage_name="pack_roi_fcn",
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=str(output_ref),
        log_path=str(log_path),
        dry_run=config.dry_run,
    )

    upsert_preprocessing_contract(
        split_paths.output_manifests_dir,
        stage_name="pack_roi_fcn",
        stage_parameters={
            "CanvasWidth": int(canvas_width),
            "CanvasHeight": int(canvas_height),
            "ShardSize": int(shard_size),
            "NormalizationMode": "zero_to_one_float32",
            "PadValue": 0,
            "TargetStorageMode": "point_only_with_geometry_metadata",
            "NumWorkers": int(effective_workers),
            "Compression": bool(config.compress),
        },
        current_representation={
            "Kind": "roi_fcn_locator_npz",
            "StorageFormat": "npz",
            "ArrayKeys": [
                "locator_input_image",
                "target_center_xy_original_px",
                "target_center_xy_canvas_px",
                "source_image_wh_px",
                "resized_image_wh_px",
                "padding_ltrb_px",
                "resize_scale",
                "sample_id",
                "image_filename",
                "npz_row_index",
                "bootstrap_bbox_xyxy_px",
                "bootstrap_confidence",
                "locator_geometry_schema",
            ],
            "ImageLayout": "N,C,H,W",
            "Channels": 1,
            "CanvasWidth": int(canvas_width),
            "CanvasHeight": int(canvas_height),
            "ImageKind": "full_frame_locator_canvas",
            "ImageColorMode": "grayscale",
            "NormalizationRange": [0.0, 1.0],
            "AspectRatioPolicy": "preserve_with_padding",
            "PadValue": 0,
            "TargetType": "crop_center_point",
            "TargetGeneration": "training_loader_gaussian_from_canvas_center",
            "TargetSource": "edge_roi_v1_bootstrap",
            "FixedROICropWidthPx": 300,
            "FixedROICropHeightPx": 300,
        },
        stage_summary=summary,
        dry_run=config.dry_run,
    )

    logger.log_summary(
        total_rows=len(samples_df),
        successful_rows=successful_rows,
        failed_rows=failed_rows,
        skipped_rows=skipped_rows,
        output_path=output_ref,
    )
    logger.write()

    if aborted:
        raise RuntimeError("pack_roi_fcn stopped after first row failure (continue_on_error=False).")

    return summary


def _effective_worker_count(config: PackRoiFcnConfig) -> int:
    requested = config.normalized_num_workers()
    if not config.continue_on_error:
        return 1
    return requested


def _format_eta_seconds(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "unknown"
    total_seconds = int(round(float(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _prepare_arrays_dir(arrays_dir: Path, *, overwrite: bool, dry_run: bool) -> None:
    existing_npz = sorted(arrays_dir.glob("*.npz")) if arrays_dir.exists() else []
    if existing_npz and not overwrite:
        raise FileExistsError(
            "Found existing NPZ outputs. Enable overwrite=True to replace them: "
            f"{arrays_dir}"
        )
    if overwrite and not dry_run:
        for npz_path in existing_npz:
            npz_path.unlink()


def _process_pack_row(
    *,
    split_paths: SplitPaths,
    sequence_idx: int,
    row_idx: int,
    row: dict[str, object],
    canvas_width: int,
    canvas_height: int,
) -> _PackRowResult:
    try:
        image_path = resolve_input_image_path(split_paths, row.get("image_filename"))
        gray = read_grayscale_uint8(image_path)
        packed = _pack_one_row(
            row_idx=row_idx,
            row=row,
            gray_image=gray,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )
        return _PackRowResult(
            sequence_idx=sequence_idx,
            row_idx=row_idx,
            status="success",
            error_message="",
            packed=packed,
        )
    except Exception as exc:
        return _PackRowResult(
            sequence_idx=sequence_idx,
            row_idx=row_idx,
            status="failed",
            error_message=str(exc),
            packed=None,
        )


def _pack_one_row(
    *,
    row_idx: int,
    row,
    gray_image: np.ndarray,
    canvas_width: int,
    canvas_height: int,
) -> _BufferedPackRow:
    src_h, src_w = int(gray_image.shape[0]), int(gray_image.shape[1])
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid source image shape: {gray_image.shape}")

    scale = min(float(canvas_width) / float(src_w), float(canvas_height) / float(src_h))
    resized_w = int(round(float(src_w) * scale))
    resized_h = int(round(float(src_h) * scale))
    if resized_w <= 0 or resized_h <= 0:
        raise ValueError(
            "Resized image dimensions must stay positive after aspect-preserving scale: "
            f"src={src_w}x{src_h}, scale={scale}"
        )

    pad_left = int((canvas_width - resized_w) // 2)
    pad_right = int(canvas_width - resized_w - pad_left)
    pad_top = int((canvas_height - resized_h) // 2)
    pad_bottom = int(canvas_height - resized_h - pad_top)
    if min(pad_left, pad_right, pad_top, pad_bottom) < 0:
        raise ValueError(
            "Computed negative padding; source image does not fit locator canvas: "
            f"src={src_w}x{src_h}, resized={resized_w}x{resized_h}, canvas={canvas_width}x{canvas_height}"
        )

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_gray = cv2.resize(gray_image, (resized_w, resized_h), interpolation=interpolation)
    locator_canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    locator_canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = (
        resized_gray.astype(np.float32) / 255.0
    )

    bootstrap_center_x = float(row["bootstrap_center_x_px"])
    bootstrap_center_y = float(row["bootstrap_center_y_px"])
    locator_center_x = (bootstrap_center_x * scale) + float(pad_left)
    locator_center_y = (bootstrap_center_y * scale) + float(pad_top)
    if not (0.0 <= locator_center_x < float(canvas_width)):
        raise ValueError(
            "locator_center_x_px falls outside locator canvas after transform: "
            f"center={locator_center_x}, canvas_width={canvas_width}"
        )
    if not (0.0 <= locator_center_y < float(canvas_height)):
        raise ValueError(
            "locator_center_y_px falls outside locator canvas after transform: "
            f"center={locator_center_y}, canvas_height={canvas_height}"
        )

    return _BufferedPackRow(
        row_idx=row_idx,
        locator_input_image=locator_canvas[None, :, :].astype(np.float32),
        target_center_xy_original_px=np.asarray([bootstrap_center_x, bootstrap_center_y], dtype=np.float32),
        target_center_xy_canvas_px=np.asarray([locator_center_x, locator_center_y], dtype=np.float32),
        source_image_wh_px=np.asarray([src_w, src_h], dtype=np.int32),
        resized_image_wh_px=np.asarray([resized_w, resized_h], dtype=np.int32),
        padding_ltrb_px=np.asarray([pad_left, pad_top, pad_right, pad_bottom], dtype=np.int32),
        resize_scale=np.float32(scale),
        sample_id=str(row.get("sample_id", "")),
        image_filename=str(row.get("image_filename", "")),
        bootstrap_bbox_xyxy_px=np.asarray(
            [
                float(row["bootstrap_bbox_x1"]),
                float(row["bootstrap_bbox_y1"]),
                float(row["bootstrap_bbox_x2"]),
                float(row["bootstrap_bbox_y2"]),
            ],
            dtype=np.float32,
        ),
        bootstrap_confidence=np.float32(float(row["bootstrap_confidence"])),
        locator_canvas_width_px=int(canvas_width),
        locator_canvas_height_px=int(canvas_height),
        locator_resized_width_px=int(resized_w),
        locator_resized_height_px=int(resized_h),
        locator_pad_left_px=int(pad_left),
        locator_pad_right_px=int(pad_right),
        locator_pad_top_px=int(pad_top),
        locator_pad_bottom_px=int(pad_bottom),
        locator_center_x_px=float(locator_center_x),
        locator_center_y_px=float(locator_center_y),
    )


def _mark_row_success(samples_df, item: _BufferedPackRow, npz_filename: str, npz_row_index: int) -> None:
    samples_df.at[item.row_idx, _STATUS_COLUMN] = "success"
    samples_df.at[item.row_idx, _ERROR_COLUMN] = ""
    samples_df.at[item.row_idx, "npz_filename"] = npz_filename
    samples_df.at[item.row_idx, "npz_row_index"] = int(npz_row_index)
    samples_df.at[item.row_idx, "locator_canvas_width_px"] = int(item.locator_canvas_width_px)
    samples_df.at[item.row_idx, "locator_canvas_height_px"] = int(item.locator_canvas_height_px)
    samples_df.at[item.row_idx, "locator_resize_scale"] = float(item.resize_scale)
    samples_df.at[item.row_idx, "locator_resized_width_px"] = int(item.locator_resized_width_px)
    samples_df.at[item.row_idx, "locator_resized_height_px"] = int(item.locator_resized_height_px)
    samples_df.at[item.row_idx, "locator_pad_left_px"] = int(item.locator_pad_left_px)
    samples_df.at[item.row_idx, "locator_pad_right_px"] = int(item.locator_pad_right_px)
    samples_df.at[item.row_idx, "locator_pad_top_px"] = int(item.locator_pad_top_px)
    samples_df.at[item.row_idx, "locator_pad_bottom_px"] = int(item.locator_pad_bottom_px)
    samples_df.at[item.row_idx, "locator_center_x_px"] = float(item.locator_center_x_px)
    samples_df.at[item.row_idx, "locator_center_y_px"] = float(item.locator_center_y_px)


def _mark_row_skipped(samples_df, row_idx: int) -> None:
    samples_df.at[row_idx, _STATUS_COLUMN] = "skipped"
    samples_df.at[row_idx, _ERROR_COLUMN] = ""
    samples_df.at[row_idx, "npz_filename"] = ""
    samples_df.at[row_idx, "npz_row_index"] = ""
    samples_df.at[row_idx, "locator_canvas_width_px"] = ""
    samples_df.at[row_idx, "locator_canvas_height_px"] = ""
    samples_df.at[row_idx, "locator_resize_scale"] = ""
    samples_df.at[row_idx, "locator_resized_width_px"] = ""
    samples_df.at[row_idx, "locator_resized_height_px"] = ""
    samples_df.at[row_idx, "locator_pad_left_px"] = ""
    samples_df.at[row_idx, "locator_pad_right_px"] = ""
    samples_df.at[row_idx, "locator_pad_top_px"] = ""
    samples_df.at[row_idx, "locator_pad_bottom_px"] = ""
    samples_df.at[row_idx, "locator_center_x_px"] = ""
    samples_df.at[row_idx, "locator_center_y_px"] = ""


def _mark_row_failed(samples_df, row_idx: int, error_message: str) -> None:
    samples_df.at[row_idx, _STATUS_COLUMN] = "failed"
    samples_df.at[row_idx, _ERROR_COLUMN] = str(error_message)
    samples_df.at[row_idx, "npz_filename"] = ""
    samples_df.at[row_idx, "npz_row_index"] = ""
    samples_df.at[row_idx, "locator_canvas_width_px"] = ""
    samples_df.at[row_idx, "locator_canvas_height_px"] = ""
    samples_df.at[row_idx, "locator_resize_scale"] = ""
    samples_df.at[row_idx, "locator_resized_width_px"] = ""
    samples_df.at[row_idx, "locator_resized_height_px"] = ""
    samples_df.at[row_idx, "locator_pad_left_px"] = ""
    samples_df.at[row_idx, "locator_pad_right_px"] = ""
    samples_df.at[row_idx, "locator_pad_top_px"] = ""
    samples_df.at[row_idx, "locator_pad_bottom_px"] = ""
    samples_df.at[row_idx, "locator_center_x_px"] = ""
    samples_df.at[row_idx, "locator_center_y_px"] = ""


def _shard_filename(dataset_reference: str, split_name: str, shard_idx: int, *, use_shards: bool) -> str:
    if use_shards:
        return f"{dataset_reference}__{split_name}__shard_{shard_idx:04d}.npz"
    return f"{dataset_reference}__{split_name}.npz"


@contextmanager
def _opencv_single_thread_mode(*, enabled: bool):
    if not enabled:
        yield
        return

    previous_threads: int | None = None
    try:
        previous_threads = int(cv2.getNumThreads())
    except Exception:
        previous_threads = None

    try:
        cv2.setNumThreads(1)
    except Exception:
        previous_threads = None

    try:
        yield
    finally:
        if previous_threads is not None:
            try:
                cv2.setNumThreads(previous_threads)
            except Exception:
                pass
