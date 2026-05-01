"""ipywidgets UI components for the v4 dual-stream pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

from .brightness_normalization import apply_brightness_normalization_v4
from .config import (
    BrightnessNormalizationConfigV4,
    DetectStageConfigV4,
    PackDualStreamStageConfigV4,
    SilhouetteStageConfigV4,
)
from .contracts import Detection
from .detector import EdgeRoiDetector, UltralyticsYoloDetector
from .image_io import read_image_unchanged, to_bgr_uint8, to_grayscale_uint8
from .manifest import load_samples_csv, samples_csv_path
from .paths import (
    detect_run_paths,
    find_project_root,
    input_run_paths,
    list_input_runs,
    resolve_manifest_path,
)
from .pipeline import STAGE_ORDER_V4, run_v4_stage_sequence_for_run
from .silhouette_algorithms import (
    ContourSilhouetteGeneratorV2,
    ConvexHullFallbackV1,
    FilledArtifactWriterV1,
    OutlineArtifactWriterV1,
)

WIDGETS_UI_BUILD = "2026-04-24-brightness-normalization-before-pack-v4"
DEFAULT_BRIGHTNESS_NORMALIZATION_METHOD = "masked_median_darkness_gain"


def _load_samples(manifests_dir: Path) -> pd.DataFrame | None:
    path = samples_csv_path(manifests_dir)
    if not path.is_file():
        return None
    return load_samples_csv(path)


def _safe_bgr_image(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    try:
        image = read_image_unchanged(path)
        return to_bgr_uint8(image)
    except Exception:
        return None


def _to_png_bytes(image: np.ndarray | None) -> bytes:
    if image is None:
        return b""
    if image.ndim not in {2, 3}:
        return b""

    if image.dtype.kind == "f":
        image = np.clip(image, 0.0, 1.0) * 255.0
        image_u8 = np.rint(image).astype(np.uint8)
    else:
        image_u8 = np.clip(image, 0, 255).astype(np.uint8)

    ok, encoded = cv2.imencode(".png", image_u8)
    if not ok:
        return b""
    return encoded.tobytes()


def _edge_foreground_mask_from_config(
    gray_image: np.ndarray,
    *,
    blur_kernel_size: int,
    canny_low_threshold: int,
    canny_high_threshold: int,
    close_kernel_size: int,
    foreground_threshold: int,
) -> np.ndarray:
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {gray_image.shape}")

    processed = gray_image
    if int(blur_kernel_size) > 1:
        processed = cv2.GaussianBlur(processed, (int(blur_kernel_size), int(blur_kernel_size)), 0)

    edges = cv2.Canny(processed, int(canny_low_threshold), int(canny_high_threshold))
    if int(close_kernel_size) > 1:
        close_kernel = np.ones((int(close_kernel_size), int(close_kernel_size)), dtype=np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

    edge_black_on_white = np.full(gray_image.shape, 255, dtype=np.uint8)
    edge_black_on_white[edges > 0] = 0
    return edge_black_on_white < int(foreground_threshold)


def _draw_source_overlay(
    source_bgr: np.ndarray,
    *,
    edge_mask: np.ndarray | None,
    roi_bounds: tuple[int, int, int, int] | None,
) -> np.ndarray:
    out = source_bgr.copy()

    if edge_mask is not None and edge_mask.shape[:2] == out.shape[:2]:
        tint = np.zeros_like(out)
        tint[..., 1] = 200  # Green tint for detected edge region.
        tint[..., 2] = 40
        mask_u8 = (edge_mask.astype(np.uint8) * 255)
        overlay = cv2.addWeighted(out, 1.0, tint, 0.35, 0.0)
        out[mask_u8 > 0] = overlay[mask_u8 > 0]

    if roi_bounds is not None:
        x1, y1, x2, y2 = roi_bounds
        cv2.rectangle(out, (int(x1), int(y1)), (int(max(x1 + 1, x2)), int(max(y1 + 1, y2))), color=(40, 220, 40), thickness=2)

    return out


def _clean_manifest_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return None
        return number
    except Exception:
        return None


def _select_detection(
    detections: list[Detection],
    *,
    allowed_ids: set[int],
    allowed_names: set[str],
) -> Detection | None:
    if not detections:
        return None

    filtered: list[Detection] = []
    for detection in detections:
        class_name = str(detection.class_name).strip().lower()
        if allowed_ids or allowed_names:
            if detection.class_id in allowed_ids or class_name in allowed_names:
                filtered.append(detection)
        else:
            filtered.append(detection)

    if not filtered:
        return None
    return max(filtered, key=lambda item: float(item.confidence))


def _clamp_detection(detection: Detection, *, frame_width: int, frame_height: int) -> Detection:
    width = max(1, int(frame_width))
    height = max(1, int(frame_height))

    x1 = max(0.0, min(float(width - 1), float(detection.x1)))
    y1 = max(0.0, min(float(height - 1), float(detection.y1)))
    x2 = max(x1 + 1e-6, min(float(width), float(detection.x2)))
    y2 = max(y1 + 1e-6, min(float(height), float(detection.y2)))

    center_x = detection.center_x_px
    center_y = detection.center_y_px
    if center_x is not None:
        center_x = max(0.0, min(float(width - 1), float(center_x)))
    if center_y is not None:
        center_y = max(0.0, min(float(height - 1), float(center_y)))

    return Detection(
        class_id=int(detection.class_id),
        class_name=str(detection.class_name),
        confidence=float(detection.confidence),
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        center_x_px=center_x,
        center_y_px=center_y,
    )


def _roi_bounds_from_detection(
    detection: Detection,
    *,
    frame_width: int,
    frame_height: int,
    padding_px: int,
) -> tuple[int, int, int, int]:
    x1 = float(detection.x1)
    y1 = float(detection.y1)
    x2 = float(detection.x2)
    y2 = float(detection.y2)

    left = max(0, int(math.floor(x1)) - int(padding_px))
    top = max(0, int(math.floor(y1)) - int(padding_px))
    right = min(int(frame_width), int(math.ceil(x2)) + int(padding_px))
    bottom = min(int(frame_height), int(math.ceil(y2)) + int(padding_px))

    if right <= left:
        right = min(int(frame_width), left + 1)
    if bottom <= top:
        bottom = min(int(frame_height), top + 1)
    return left, top, right, bottom


def _extract_centered_canvas_from_detection(
    source_gray: np.ndarray,
    detection: Detection,
    *,
    canvas_width: int,
    canvas_height: int,
    padding_px: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Extract fixed-size ROI canvas centered on detection center without scaling."""
    if source_gray.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {source_gray.shape}")

    frame_height = int(source_gray.shape[0])
    frame_width = int(source_gray.shape[1])
    canvas_w = max(1, int(canvas_width))
    canvas_h = max(1, int(canvas_height))

    x1 = float(detection.x1) - float(padding_px)
    y1 = float(detection.y1) - float(padding_px)
    x2 = float(detection.x2) + float(padding_px)
    y2 = float(detection.y2) + float(padding_px)

    center_x = detection.center_x_px if detection.center_x_px is not None else ((x1 + x2) * 0.5)
    center_y = detection.center_y_px if detection.center_y_px is not None else ((y1 + y2) * 0.5)

    req_x1 = int(round(float(center_x) - (canvas_w / 2.0)))
    req_y1 = int(round(float(center_y) - (canvas_h / 2.0)))
    req_x2 = req_x1 + canvas_w
    req_y2 = req_y1 + canvas_h

    src_x1 = max(0, req_x1)
    src_y1 = max(0, req_y1)
    src_x2 = min(frame_width, req_x2)
    src_y2 = min(frame_height, req_y2)
    if src_x2 <= src_x1 or src_y2 <= src_y1:
        raise ValueError("empty ROI after centered canvas extraction")

    dst_x1 = src_x1 - req_x1
    dst_y1 = src_y1 - req_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = source_gray[src_y1:src_y2, src_x1:src_x2]

    return (
        canvas,
        (src_x1, src_y1, src_x2, src_y2),
        (dst_x1, dst_y1, dst_x2, dst_y2),
        (req_x1, req_y1, req_x2, req_y2),
    )


def _contour_break_reason(contour: np.ndarray | None) -> str:
    if contour is None:
        return "no_contour"
    if contour.ndim != 3 or contour.shape[0] < 3:
        return "degenerate_contour"
    area = float(abs(cv2.contourArea(contour)))
    if area <= 0.0:
        return "degenerate_contour_area"
    return ""


def _render_is_empty(gray_image: np.ndarray) -> bool:
    return gray_image.ndim != 2 or not bool(np.any(gray_image < 255))


def _background_mask_from_binary_silhouette(gray_image: np.ndarray) -> np.ndarray:
    """Return white-background mask (1.0=background) supporting both polarity conventions."""
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {gray_image.shape}")

    white_count = int(np.count_nonzero(gray_image > 127))
    black_count = int(gray_image.size - white_count)

    if white_count <= black_count:
        return (gray_image < 128).astype(np.float32)
    return (gray_image > 127).astype(np.float32)


def _render_inverted_vehicle_detail_on_white_for_preview(
    roi_source_gray: np.ndarray,
    background_mask: np.ndarray,
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

    source_u8 = np.clip(roi_source_gray, 0, 255).astype(np.uint8)
    inverted = 255 - source_u8
    out_u8 = np.full(source_u8.shape, 255, dtype=np.uint8)
    vehicle_mask = background_mask < 0.5
    out_u8[vehicle_mask] = inverted[vehicle_mask]
    return out_u8.astype(np.float32) / 255.0


def _place_image_on_canvas_for_preview(
    image: np.ndarray,
    *,
    canvas_height: int,
    canvas_width: int,
    clip_policy: str,
) -> tuple[np.ndarray, bool]:
    """Mirror pack-stage placement: no scaling, center on canvas, optional clipping."""
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    src_h, src_w = int(image.shape[0]), int(image.shape[1])
    clipped = src_h > canvas_height or src_w > canvas_width
    if clipped and str(clip_policy).strip().lower() == "fail":
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


def _preview_float_image_to_uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _ensure_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Expected 2D grayscale or 3-channel image, got {image.shape}")

    if gray.dtype.kind == "f":
        max_value = float(np.nanmax(gray)) if gray.size else 0.0
        scale = 255.0 if max_value <= 1.0 else 1.0
        return np.rint(np.clip(gray * scale, 0.0, 255.0)).astype(np.uint8)
    return np.clip(gray, 0, 255).astype(np.uint8)


def _silhouette_debug_strip(
    debug_images: dict[str, np.ndarray],
    *,
    tile_height: int,
    tile_width: int,
) -> np.ndarray:
    labels = [
        ("raw_edge", "raw_edge"),
        ("post_morph", "post_morph"),
        ("selected_component", "selected_component"),
    ]

    height = max(1, int(tile_height))
    width = max(1, int(tile_width))
    tiles: list[np.ndarray] = []
    for image_key, text_label in labels:
        image = debug_images.get(image_key)
        if image is None:
            gray = np.full((height, width), 255, dtype=np.uint8)
        else:
            gray = _ensure_grayscale_uint8(image)
            if gray.shape[:2] != (height, width):
                gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)

        tile = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            tile,
            text_label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (36, 36, 220),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    return cv2.hconcat(tiles)


def _clamp_interval_by_size_for_preview(start: int, size: int, *, limit: int) -> tuple[int, int]:
    total = max(1, int(limit))
    extent = max(1, int(size))
    if extent >= total:
        return 0, total

    s = int(start)
    e = s + extent
    if s < 0:
        s = 0
        e = extent
    if e > total:
        e = total
        s = max(0, total - extent)
    return s, e


def _draw_bbox_inclusive(canvas: np.ndarray, bbox: tuple[int, int, int, int], *, color: tuple[int, int, int], thickness: int) -> None:
    height = int(canvas.shape[0])
    width = int(canvas.shape[1])
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1, min(width - 1, int(x2)))
    y2 = max(y1, min(height - 1, int(y2)))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color=color, thickness=max(1, int(thickness)))


def _draw_bbox_exclusive(canvas: np.ndarray, bbox: tuple[int, int, int, int], *, color: tuple[int, int, int], thickness: int) -> None:
    x1, y1, x2, y2 = bbox
    if int(x2) <= int(x1) or int(y2) <= int(y1):
        return
    _draw_bbox_inclusive(
        canvas,
        (int(x1), int(y1), int(x2) - 1, int(y2) - 1),
        color=color,
        thickness=thickness,
    )


def _format_bbox_inclusive(bbox: tuple[int, int, int, int] | None) -> str:
    if bbox is None:
        return "n/a"
    x1, y1, x2, y2 = bbox
    return f"[{int(x1)},{int(y1)}] -> [{int(x2)},{int(y2)}]"


def _format_bbox_exclusive(bbox: tuple[int, int, int, int] | None) -> str:
    if bbox is None:
        return "n/a"
    x1, y1, x2, y2 = bbox
    return f"[{int(x1)},{int(y1)}] -> [{int(x2)},{int(y2)})"


def _edge_roi_selection_debug_strip(
    gray_image: np.ndarray,
    *,
    blur_kernel_size: int,
    canny_low_threshold: int,
    canny_high_threshold: int,
    close_kernel_size: int,
    foreground_threshold: int,
    padding_px: int,
    min_foreground_px: int,
    selected_detection: Detection | None,
) -> tuple[np.ndarray, dict[str, object]]:
    if gray_image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got {gray_image.shape}")

    processed = gray_image
    if int(blur_kernel_size) > 1:
        processed = cv2.GaussianBlur(processed, (int(blur_kernel_size), int(blur_kernel_size)), 0)

    raw_edges = cv2.Canny(processed, int(canny_low_threshold), int(canny_high_threshold))
    post_morph = raw_edges.copy()
    if int(close_kernel_size) > 1:
        close_kernel = np.ones((int(close_kernel_size), int(close_kernel_size)), dtype=np.uint8)
        post_morph = cv2.morphologyEx(post_morph, cv2.MORPH_CLOSE, close_kernel)

    raw_edge_bw = np.full(gray_image.shape, 255, dtype=np.uint8)
    raw_edge_bw[raw_edges > 0] = 0

    post_morph_bw = np.full(gray_image.shape, 255, dtype=np.uint8)
    post_morph_bw[post_morph > 0] = 0

    foreground_mask = post_morph_bw < int(foreground_threshold)
    foreground_bw = np.full(gray_image.shape, 255, dtype=np.uint8)
    foreground_bw[foreground_mask] = 0

    geometry = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        geometry,
        "yellow=raw bbox, orange=pre-clamp, green=live, magenta=used",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (32, 32, 220),
        1,
        cv2.LINE_AA,
    )

    summary: dict[str, object] = {
        "selection_status": "not_selected",
        "foreground_pixel_count": int(np.count_nonzero(foreground_mask)),
        "min_foreground_px": int(min_foreground_px),
        "centroid_xy": None,
        "raw_bbox_inclusive": None,
        "centered_pre_clamp_exclusive": None,
        "live_bbox_post_clamp_exclusive": None,
        "selected_bbox_exclusive": None,
        "selected_center_xy": None,
    }

    foreground_pixels = int(summary["foreground_pixel_count"])
    if foreground_pixels > 0:
        ys, xs = np.where(foreground_mask)
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        summary["centroid_xy"] = (center_x, center_y)
        cv2.circle(geometry, (int(round(center_x)), int(round(center_y))), 4, (0, 0, 255), -1, cv2.LINE_AA)

        raw_x1 = max(0, int(xs.min()) - int(padding_px))
        raw_y1 = max(0, int(ys.min()) - int(padding_px))
        raw_x2 = min(gray_image.shape[1] - 1, int(xs.max()) + int(padding_px))
        raw_y2 = min(gray_image.shape[0] - 1, int(ys.max()) + int(padding_px))
        raw_bbox = (raw_x1, raw_y1, raw_x2, raw_y2)
        summary["raw_bbox_inclusive"] = raw_bbox
        _draw_bbox_inclusive(geometry, raw_bbox, color=(0, 215, 255), thickness=2)

        roi_width = max(1, int(raw_x2 - raw_x1 + 1))
        roi_height = max(1, int(raw_y2 - raw_y1 + 1))
        centered_x1 = int(round(center_x - (roi_width / 2.0)))
        centered_y1 = int(round(center_y - (roi_height / 2.0)))
        centered_pre_clamp = (
            int(centered_x1),
            int(centered_y1),
            int(centered_x1 + roi_width),
            int(centered_y1 + roi_height),
        )
        summary["centered_pre_clamp_exclusive"] = centered_pre_clamp
        _draw_bbox_exclusive(geometry, centered_pre_clamp, color=(0, 140, 255), thickness=1)

        if foreground_pixels >= int(min_foreground_px):
            live_x1, live_x2 = _clamp_interval_by_size_for_preview(
                centered_x1,
                roi_width,
                limit=int(gray_image.shape[1]),
            )
            live_y1, live_y2 = _clamp_interval_by_size_for_preview(
                centered_y1,
                roi_height,
                limit=int(gray_image.shape[0]),
            )
            live_bbox = (int(live_x1), int(live_y1), int(live_x2), int(live_y2))
            summary["live_bbox_post_clamp_exclusive"] = live_bbox
            summary["selection_status"] = "live_detection_success"
            _draw_bbox_exclusive(geometry, live_bbox, color=(40, 220, 40), thickness=2)
        else:
            summary["selection_status"] = "foreground_below_min"
    else:
        summary["selection_status"] = "no_foreground_after_threshold"

    if selected_detection is not None:
        selected_x1 = int(round(float(selected_detection.x1)))
        selected_y1 = int(round(float(selected_detection.y1)))
        selected_x2 = int(round(float(selected_detection.x2)))
        selected_y2 = int(round(float(selected_detection.y2)))
        selected_bbox = (selected_x1, selected_y1, selected_x2, selected_y2)
        summary["selected_bbox_exclusive"] = selected_bbox
        _draw_bbox_exclusive(geometry, selected_bbox, color=(220, 40, 220), thickness=2)

        selected_center_x = float(selected_detection.center_x_px) if selected_detection.center_x_px is not None else (
            (float(selected_detection.x1) + float(selected_detection.x2)) * 0.5
        )
        selected_center_y = float(selected_detection.center_y_px) if selected_detection.center_y_px is not None else (
            (float(selected_detection.y1) + float(selected_detection.y2)) * 0.5
        )
        summary["selected_center_xy"] = (selected_center_x, selected_center_y)
        cv2.circle(
            geometry,
            (int(round(selected_center_x)), int(round(selected_center_y))),
            5,
            (220, 40, 220),
            1,
            cv2.LINE_AA,
        )

    def _tile(image: np.ndarray, label: str, *, interpolation: int) -> np.ndarray:
        tile = image.copy()
        if tile.ndim == 2:
            tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        target_height = max(120, min(260, int(round(gray_image.shape[0] * 0.20))))
        target_width = max(1, int(round(gray_image.shape[1] * (target_height / max(1, gray_image.shape[0])))))
        if tile.shape[0] != target_height or tile.shape[1] != target_width:
            tile = cv2.resize(tile, (target_width, target_height), interpolation=interpolation)

        cv2.putText(
            tile,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (36, 36, 220),
            1,
            cv2.LINE_AA,
        )
        return tile

    strip = cv2.hconcat(
        [
            _tile(raw_edge_bw, "raw_edge", interpolation=cv2.INTER_NEAREST),
            _tile(post_morph_bw, "post_morph", interpolation=cv2.INTER_NEAREST),
            _tile(foreground_bw, "foreground_mask", interpolation=cv2.INTER_NEAREST),
            _tile(geometry, "roi_geometry", interpolation=cv2.INTER_AREA),
        ]
    )
    return strip, summary


class PipelineLauncherV4:
    """Notebook launcher preserving stage workflow and visual preview inspection."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self._preview_detector: UltralyticsYoloDetector | EdgeRoiDetector | None = None
        self._preview_detector_key: tuple[object, ...] | None = None
        self._widget_root: widgets.Widget | None = None
        self._callbacks_registered = False
        self._execution_in_progress = False

        self.run_dropdown = widgets.Dropdown(description="Run:", options=[])
        self.stage_dropdown = widgets.Dropdown(
            description="Stage:",
            options=[("All", "all"), *[(stage, stage) for stage in STAGE_ORDER_V4]],
            value="all",
        )

        self.refresh_button = widgets.Button(description="Refresh Runs")
        self.execute_button = widgets.Button(description="Run Stage(s)", button_style="success")

        self.model_path_text = widgets.Text(
            description="YOLO weights:",
            placeholder="/abs/path/to/model.pt (optional)",
        )
        self.detector_backend_dropdown = widgets.Dropdown(
            description="Detect via:",
            options=[("Edge ROI (v1 style)", "edge"), ("YOLO", "yolo")],
            value="edge",
        )
        self.default_model_ref_text = widgets.Text(
            description="Fallback model:",
            value="yolov8n.pt",
            placeholder="Used when YOLO weights is blank",
        )
        self.detect_conf_slider = widgets.FloatSlider(
            description="Detect conf",
            value=0.01,
            min=0.00,
            max=1.00,
            step=0.01,
            readout_format=".2f",
            continuous_update=False,
        )
        self.detect_iou_slider = widgets.FloatSlider(
            description="Detect IoU",
            value=0.70,
            min=0.00,
            max=1.00,
            step=0.01,
            readout_format=".2f",
            continuous_update=False,
        )
        self.class_ids_text = widgets.Text(description="Class IDs:", placeholder="e.g. 2,7")
        self.class_names_text = widgets.Text(description="Class Names:", placeholder="e.g. defender")
        self.edge_blur_kernel_slider = widgets.IntSlider(
            description="Edge blur k",
            value=5,
            min=1,
            max=31,
            step=2,
            continuous_update=False,
        )
        self.edge_canny_low_slider = widgets.IntSlider(
            description="Edge low",
            value=50,
            min=0,
            max=255,
            step=1,
            continuous_update=False,
        )
        self.edge_canny_high_slider = widgets.IntSlider(
            description="Edge high",
            value=150,
            min=0,
            max=255,
            step=1,
            continuous_update=False,
        )
        self.edge_foreground_threshold_slider = widgets.IntSlider(
            description="FG thresh",
            value=250,
            min=0,
            max=255,
            step=1,
            continuous_update=False,
        )
        self.edge_padding_int = widgets.BoundedIntText(
            description="Edge pad:",
            value=0,
            min=0,
            max=1024,
        )
        self.edge_min_foreground_int = widgets.BoundedIntText(
            description="Min edge px:",
            value=16,
            min=1,
            max=1_000_000,
        )
        self.preview_ignore_filter_checkbox = widgets.Checkbox(
            value=True,
            description="Preview fallback if class-filter misses",
        )

        self.silhouette_mode = widgets.Dropdown(
            description="Silhouette:",
            options=[("Filled", "filled"), ("Outline", "outline")],
            value="filled",
        )
        self.threshold_low_slider = widgets.IntSlider(
            description="Threshold low",
            value=50,
            min=0,
            max=255,
            step=1,
            continuous_update=False,
        )
        self.threshold_high_slider = widgets.IntSlider(
            description="Threshold high",
            value=150,
            min=0,
            max=255,
            step=1,
            continuous_update=False,
        )
        self.min_area_input = widgets.BoundedIntText(description="Min area:", value=50, min=1, max=1_000_000)
        self.fill_holes_checkbox = widgets.Checkbox(value=True, description="Fill internal holes")
        self.close_kernel_slider = widgets.IntSlider(
            description="Close k",
            value=3,
            min=1,
            max=31,
            step=1,
            continuous_update=False,
        )
        self.outline_thickness_slider = widgets.IntSlider(
            description="Outline px",
            value=1,
            min=1,
            max=10,
            step=1,
            continuous_update=False,
        )
        self.roi_padding_int = widgets.IntText(description="ROI pad:", value=0)

        self.canvas_w_int = widgets.IntText(description="Canvas W:", value=300)
        self.canvas_h_int = widgets.IntText(description="Canvas H:", value=300)
        self.clip_policy_dropdown = widgets.Dropdown(
            description="Clip:",
            options=[("Fail", "fail"), ("Clip", "clip")],
            value="fail",
        )
        self.shard_size_int = widgets.IntText(description="Shard size:", value=8192)
        self.brightness_norm_enabled_checkbox = widgets.Checkbox(
            value=True,
            description="Brightness norm",
        )
        self.brightness_norm_method_dropdown = widgets.Dropdown(
            description="Method:",
            options=[
                ("Masked median darkness gain", "masked_median_darkness_gain"),
            ],
            value=DEFAULT_BRIGHTNESS_NORMALIZATION_METHOD,
        )
        self.brightness_norm_target_float = widgets.BoundedFloatText(
            description="Target dark:",
            value=0.55,
            min=0.05,
            max=0.95,
            step=0.01,
        )
        self.brightness_norm_min_gain_float = widgets.BoundedFloatText(
            description="Min gain:",
            value=0.5,
            min=0.000001,
            max=100.0,
            step=0.05,
        )
        self.brightness_norm_max_gain_float = widgets.BoundedFloatText(
            description="Max gain:",
            value=2.0,
            min=0.000001,
            max=100.0,
            step=0.05,
        )
        self.brightness_norm_epsilon_float = widgets.BoundedFloatText(
            description="Epsilon:",
            value=1e-6,
            min=1e-12,
            max=1.0,
            step=1e-6,
        )
        self.brightness_norm_empty_policy_dropdown = widgets.Dropdown(
            description="Empty mask:",
            options=[("Skip", "skip"), ("Fail", "fail")],
            value="skip",
        )
        self.brightness_norm_pane = widgets.VBox(
            [
                widgets.HTML("<b>Brightness Normalization</b>"),
                widgets.HBox([self.brightness_norm_enabled_checkbox, self.brightness_norm_method_dropdown]),
                widgets.HBox([self.brightness_norm_target_float, self.brightness_norm_empty_policy_dropdown]),
                widgets.HBox([self.brightness_norm_min_gain_float, self.brightness_norm_max_gain_float]),
                self.brightness_norm_epsilon_float,
            ],
            layout=widgets.Layout(
                border="1px solid #ddd",
                padding="8px",
                margin="8px 0 0 0",
            ),
        )

        self.sample_offset_input = widgets.BoundedIntText(description="Sample offset:", value=0, min=0, max=1_000_000)
        self.sample_limit_input = widgets.BoundedIntText(description="Sample limit:", value=0, min=0, max=1_000_000)

        self.log_output = widgets.Output(
            layout=widgets.Layout(border="1px solid #ddd", padding="8px", max_height="260px", overflow_y="auto")
        )

        self.preview_sample_offset_label = widgets.Label(
            value="Sample offset:",
            layout=widgets.Layout(width="120px", min_width="120px", flex="0 0 120px"),
        )
        self.preview_sample_offset_input = widgets.BoundedIntText(
            description="",
            value=0,
            min=0,
            max=1_000_000,
            layout=widgets.Layout(width="220px", min_width="220px", flex="0 0 220px"),
        )
        self.preview_refresh_button = widgets.Button(
            description="Preview",
            button_style="info",
            layout=widgets.Layout(width="120px", min_width="120px", flex="0 0 120px"),
        )
        self.preview_controls_row = widgets.HBox(
            [self.preview_sample_offset_label, self.preview_sample_offset_input, self.preview_refresh_button],
            layout=widgets.Layout(
                width="100%",
                display="flex",
                flex_flow="row nowrap",
                align_items="center",
                justify_content="flex-start",
                overflow="visible",
                margin="0 0 8px 0",
            ),
        )
        self.preview_status_html = widgets.HTML()

        image_layout = widgets.Layout(width="100%", border="1px solid #ddd")
        self.preview_source_overlay = widgets.Image(format="png", value=b"", layout=image_layout)
        self.preview_roi_selection_debug = widgets.Image(format="png", value=b"", layout=image_layout)
        self.preview_roi_selection_stats_html = widgets.HTML()
        self.preview_extracted_roi = widgets.Image(format="png", value=b"", layout=image_layout)
        self.preview_silhouette_debug = widgets.Image(format="png", value=b"", layout=image_layout)
        self.preview_silhouette_roi = widgets.Image(format="png", value=b"", layout=image_layout)
        self.preview_packed_canvas = widgets.Image(format="png", value=b"", layout=image_layout)
        self.preview_silhouette_full = widgets.Image(format="png", value=b"", layout=image_layout)

        self.refresh_button.on_click(self._on_refresh)
        self.execute_button.on_click(self._on_execute)
        self.preview_refresh_button.on_click(self._on_preview_refresh)
        self.run_dropdown.observe(self._on_run_change, names="value")
        self.sample_offset_input.observe(self._on_sample_offset_change, names="value")
        self.preview_sample_offset_input.observe(self._on_preview_sample_offset_change, names="value")
        self.brightness_norm_enabled_checkbox.observe(self._on_brightness_norm_enabled_change, names="value")
        self.brightness_norm_method_dropdown.observe(self._on_brightness_norm_method_change, names="value")
        self._callbacks_registered = True

        self._syncing_sample_offset = False
        self._suspend_run_change_preview = True

        try:
            self.refresh_runs()
        finally:
            self._suspend_run_change_preview = False

    @property
    def widget(self) -> widgets.Widget:
        if self._widget_root is None:
            self._widget_root = self._build_widget()
        return self._widget_root

    def _build_widget(self) -> widgets.Widget:
        controls = widgets.VBox(
            [
                widgets.HTML(f"<b>Pipeline Launcher (v4)</b> <code>{WIDGETS_UI_BUILD}</code>"),
                widgets.HBox([self.run_dropdown, self.stage_dropdown]),
                widgets.HBox([self.refresh_button, self.execute_button]),
                widgets.HTML("<hr><b>Detect</b>"),
                self.detector_backend_dropdown,
                self.model_path_text,
                self.default_model_ref_text,
                widgets.HBox([self.detect_conf_slider, self.detect_iou_slider]),
                self.class_ids_text,
                self.class_names_text,
                widgets.HBox([self.edge_blur_kernel_slider, self.edge_canny_low_slider, self.edge_canny_high_slider]),
                widgets.HBox([self.edge_foreground_threshold_slider, self.edge_padding_int, self.edge_min_foreground_int]),
                self.preview_ignore_filter_checkbox,
                widgets.HTML("<hr><b>Silhouette</b>"),
                self.silhouette_mode,
                self.threshold_low_slider,
                self.threshold_high_slider,
                widgets.HBox([self.min_area_input, self.roi_padding_int]),
                widgets.HBox([self.fill_holes_checkbox, self.close_kernel_slider, self.outline_thickness_slider]),
                widgets.HTML("<hr>"),
                self.brightness_norm_pane,
                widgets.HTML("<hr><b>Pack Dual Stream</b>"),
                widgets.HBox([self.canvas_w_int, self.canvas_h_int]),
                widgets.HBox([self.clip_policy_dropdown, self.shard_size_int]),
                widgets.HTML("<hr><b>Sampling</b>"),
                widgets.HBox([self.sample_offset_input, self.sample_limit_input]),
                widgets.HTML("<b>Logs</b>"),
                self.log_output,
            ],
            layout=widgets.Layout(width="52%"),
        )

        preview = widgets.VBox(
            [
                widgets.HTML("<b>Preview Panel</b>"),
                self.preview_controls_row,
                self.preview_status_html,
                widgets.HTML("<b>0) Source + Edge Region + ROI Box (context only; display-only)</b>"),
                self.preview_source_overlay,
                widgets.HTML("<b>1) ROI Selection Debug (edge detect path used for ROI geometry)</b>"),
                self.preview_roi_selection_debug,
                self.preview_roi_selection_stats_html,
                widgets.HTML("<b>2) ROI Input Canvas (silhouette input; matches Pack Canvas W/H, no scaling)</b>"),
                self.preview_extracted_roi,
                widgets.HTML("<b>3) Silhouette Debug Strip (raw edge | post morph | selected component)</b>"),
                self.preview_silhouette_debug,
                widgets.HTML("<b>4) ROI Detail Isolated By Silhouette (after optional brightness normalization)</b>"),
                self.preview_silhouette_roi,
                widgets.HTML("<b>5) Packed Canvas (exact npz['silhouette_crop'] geometry; dark detail on white)</b>"),
                self.preview_packed_canvas,
                widgets.HTML("<b>6) Full-Frame Silhouette (auxiliary view; not used by pack payload)</b>"),
                self.preview_silhouette_full,
            ],
            layout=widgets.Layout(width="48%", max_height="980px", overflow_y="auto", padding="0 0 0 8px"),
        )

        return widgets.HBox([controls, preview], layout=widgets.Layout(width="100%"))

    def close(self) -> None:
        """Release widget callbacks and front-end views for notebook reruns."""

        if self._callbacks_registered:
            self.refresh_button.on_click(self._on_refresh, remove=True)
            self.execute_button.on_click(self._on_execute, remove=True)
            self.preview_refresh_button.on_click(self._on_preview_refresh, remove=True)
            self.run_dropdown.unobserve(self._on_run_change, names="value")
            self.sample_offset_input.unobserve(self._on_sample_offset_change, names="value")
            self.preview_sample_offset_input.unobserve(self._on_preview_sample_offset_change, names="value")
            self.brightness_norm_enabled_checkbox.unobserve(self._on_brightness_norm_enabled_change, names="value")
            self.brightness_norm_method_dropdown.unobserve(self._on_brightness_norm_method_change, names="value")
            self._callbacks_registered = False

        if self._widget_root is not None:
            self._widget_root.close()
            self._widget_root = None

    def refresh_runs(self) -> None:
        runs = list_input_runs(self.project_root)
        self.run_dropdown.options = runs
        if runs and self.run_dropdown.value not in runs:
            self.run_dropdown.value = runs[0]
        self._sync_preview_row_bounds()

    def _sync_preview_row_bounds(self) -> None:
        run_name = self.run_dropdown.value
        if not run_name:
            self.sample_offset_input.max = 0
            self.preview_sample_offset_input.max = 0
            self._set_sample_offset_value(0)
            return

        input_paths = input_run_paths(self.project_root, str(run_name))
        samples_df = _load_samples(input_paths.manifests_dir)
        max_value = max(0, len(samples_df) - 1) if samples_df is not None else 0
        self.sample_offset_input.max = int(max_value)
        self.preview_sample_offset_input.max = int(max_value)
        if int(self.sample_offset_input.value) > int(max_value):
            self._set_sample_offset_value(int(max_value))

    def _on_refresh(self, _button: widgets.Button) -> None:
        self.refresh_runs()
        self.render_preview()

    def _on_run_change(self, _change: dict) -> None:
        self._sync_preview_row_bounds()
        if self._suspend_run_change_preview:
            return
        self.render_preview()

    def _set_sample_offset_value(self, value: int) -> None:
        normalized = max(0, int(value))
        if self._syncing_sample_offset:
            return
        self._syncing_sample_offset = True
        try:
            if int(self.sample_offset_input.value) != normalized:
                self.sample_offset_input.value = normalized
            if int(self.preview_sample_offset_input.value) != normalized:
                self.preview_sample_offset_input.value = normalized
        finally:
            self._syncing_sample_offset = False

    def _on_sample_offset_change(self, _change: dict) -> None:
        self._set_sample_offset_value(int(self.sample_offset_input.value))

    def _on_preview_sample_offset_change(self, _change: dict) -> None:
        self._set_sample_offset_value(int(self.preview_sample_offset_input.value))

    def _on_preview_refresh(self, _button: widgets.Button) -> None:
        self._set_sample_offset_value(int(self.preview_sample_offset_input.value))
        self.render_preview()

    def _on_brightness_norm_enabled_change(self, _change: dict) -> None:
        if (
            bool(self.brightness_norm_enabled_checkbox.value)
            and str(self.brightness_norm_method_dropdown.value) == "none"
        ):
            self.brightness_norm_method_dropdown.value = DEFAULT_BRIGHTNESS_NORMALIZATION_METHOD

    def _on_brightness_norm_method_change(self, _change: dict) -> None:
        method = str(self.brightness_norm_method_dropdown.value)
        enabled = bool(self.brightness_norm_enabled_checkbox.value)
        if method == "none" and enabled:
            self.brightness_norm_enabled_checkbox.value = False
        elif method != "none" and not enabled:
            self.brightness_norm_enabled_checkbox.value = True

    def _sample_offset(self) -> int:
        return int(self.sample_offset_input.value)

    def _sample_limit(self) -> int:
        return int(self.sample_limit_input.value)

    def _build_detect_config(self) -> DetectStageConfigV4:
        return DetectStageConfigV4(
            detector_backend=str(self.detector_backend_dropdown.value),
            model_path=str(self.model_path_text.value).strip(),
            default_model_ref=str(self.default_model_ref_text.value).strip(),
            conf_threshold=float(self.detect_conf_slider.value),
            iou_threshold=float(self.detect_iou_slider.value),
            defender_class_ids=_parse_int_tuple(self.class_ids_text.value),
            defender_class_names=_parse_str_tuple(self.class_names_text.value),
            edge_blur_kernel_size=int(self.edge_blur_kernel_slider.value),
            edge_canny_low_threshold=int(self.edge_canny_low_slider.value),
            edge_canny_high_threshold=int(self.edge_canny_high_slider.value),
            edge_foreground_threshold=int(self.edge_foreground_threshold_slider.value),
            edge_padding_px=int(self.edge_padding_int.value),
            edge_min_foreground_px=int(self.edge_min_foreground_int.value),
            sample_offset=self._sample_offset(),
            sample_limit=self._sample_limit(),
        )

    def _build_silhouette_config(self) -> SilhouetteStageConfigV4:
        return SilhouetteStageConfigV4(
            representation_mode=str(self.silhouette_mode.value),
            canny_low_threshold=int(self.threshold_low_slider.value),
            canny_high_threshold=int(self.threshold_high_slider.value),
            min_component_area_px=int(self.min_area_input.value),
            fill_holes=bool(self.fill_holes_checkbox.value),
            close_kernel_size=int(self.close_kernel_slider.value),
            outline_thickness=int(self.outline_thickness_slider.value),
            roi_padding_px=int(self.roi_padding_int.value),
            roi_canvas_width_px=int(self.canvas_w_int.value),
            roi_canvas_height_px=int(self.canvas_h_int.value),
            sample_offset=self._sample_offset(),
            sample_limit=self._sample_limit(),
        )

    def _build_pack_config(self) -> PackDualStreamStageConfigV4:
        brightness_enabled = bool(self.brightness_norm_enabled_checkbox.value)
        brightness_method = (
            str(self.brightness_norm_method_dropdown.value)
            if brightness_enabled
            else "none"
        )

        return PackDualStreamStageConfigV4(
            canvas_width_px=int(self.canvas_w_int.value),
            canvas_height_px=int(self.canvas_h_int.value),
            clip_policy=str(self.clip_policy_dropdown.value),
            include_v1_compat_arrays=False,
            brightness_normalization=BrightnessNormalizationConfigV4(
                enabled=brightness_enabled,
                method=brightness_method,
                target_median_darkness=float(self.brightness_norm_target_float.value),
                min_gain=float(self.brightness_norm_min_gain_float.value),
                max_gain=float(self.brightness_norm_max_gain_float.value),
                epsilon=float(self.brightness_norm_epsilon_float.value),
                empty_mask_policy=str(self.brightness_norm_empty_policy_dropdown.value),
            ),
            shard_size=int(self.shard_size_int.value),
            sample_offset=self._sample_offset(),
            sample_limit=self._sample_limit(),
        )

    def _reset_extra_preview_outputs(self) -> None:
        """Hook for launcher variants with additional preview streams."""

    def _render_extra_pack_preview(
        self,
        *,
        extracted_roi_gray: np.ndarray,
        background_mask: np.ndarray,
        pack_config: PackDualStreamStageConfigV4,
    ) -> None:
        """Hook for launcher variants with additional pack outputs."""
        del extracted_roi_gray, background_mask, pack_config

    def _saved_path_preview_text(self, pack_config: PackDualStreamStageConfigV4) -> str:
        return (
            "<b>Saved Path:</b> fixed ROI canvas -> silhouette mask isolation -> invert grayscale inside mask "
            "-> white outside mask -> optional foreground brightness normalization -> centered canvas "
            f"({pack_config.normalized_canvas_height_px()}x{pack_config.normalized_canvas_width_px()})"
            " -> npz['silhouette_crop']"
        )

    def _get_preview_detector(self, config: DetectStageConfigV4):
        backend = config.normalized_detector_backend()
        edge_ids = config.normalized_defender_class_ids()
        edge_names = config.normalized_defender_class_names()

        if backend == "edge":
            key = (
                backend,
                config.normalized_edge_blur_kernel_size(),
                config.normalized_edge_canny_low_threshold(),
                config.normalized_edge_canny_high_threshold(),
                config.normalized_edge_foreground_threshold(),
                config.normalized_edge_padding_px(),
                config.normalized_edge_min_foreground_px(),
                config.normalized_edge_close_kernel_size(),
                int(edge_ids[0]) if edge_ids else 0,
                str(edge_names[0]) if edge_names else "defender",
            )
            if self._preview_detector is not None and self._preview_detector_key == key:
                return self._preview_detector

            detector = EdgeRoiDetector(
                blur_kernel_size=config.normalized_edge_blur_kernel_size(),
                canny_low_threshold=config.normalized_edge_canny_low_threshold(),
                canny_high_threshold=config.normalized_edge_canny_high_threshold(),
                foreground_threshold=config.normalized_edge_foreground_threshold(),
                padding_px=config.normalized_edge_padding_px(),
                min_foreground_px=config.normalized_edge_min_foreground_px(),
                close_kernel_size=config.normalized_edge_close_kernel_size(),
                class_id=int(edge_ids[0]) if edge_ids else 0,
                class_name=str(edge_names[0]) if edge_names else "defender",
            )
            self._preview_detector = detector
            self._preview_detector_key = key
            return detector

        model_ref = self._resolve_model_ref_for_preview(config)
        if not model_ref:
            raise ValueError("No model configured for preview.")

        key = (
            backend,
            model_ref,
            config.normalized_conf_threshold(),
            config.normalized_iou_threshold(),
            config.normalized_imgsz(),
            config.normalized_max_det(),
            config.normalized_device(),
        )
        if self._preview_detector is not None and self._preview_detector_key == key:
            return self._preview_detector

        detector = UltralyticsYoloDetector(
            model_path=model_ref,
            conf_threshold=config.normalized_conf_threshold(),
            iou_threshold=config.normalized_iou_threshold(),
            imgsz=config.normalized_imgsz(),
            max_det=config.normalized_max_det(),
            device=config.normalized_device(),
        )
        self._preview_detector = detector
        self._preview_detector_key = key
        return detector

    def _resolve_model_ref_for_preview(self, config: DetectStageConfigV4) -> str:
        explicit = config.normalized_model_path()
        if explicit:
            return explicit

        ref = config.normalized_default_model_ref()
        if not ref:
            return ""

        ref_path = Path(ref)
        if ref_path.is_file():
            return str(ref_path)

        candidates = [
            self.project_root / ref,
            self.project_root / "rb_ui_v4" / ref,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)

        return ref

    def _on_execute(self, _button: widgets.Button) -> None:
        if self._execution_in_progress:
            return

        run_name = self.run_dropdown.value
        stage_name = self.stage_dropdown.value
        if not run_name:
            with self.log_output:
                print("No run selected.")
            return

        self._execution_in_progress = True
        self.execute_button.disabled = True

        try:
            self.log_output.clear_output(wait=False)
            with self.log_output:
                print(f"Running stage '{stage_name}' for run '{run_name}' ...")

            detect_config = self._build_detect_config()
            silhouette_config = self._build_silhouette_config()
            pack_config = self._build_pack_config()

            summaries = run_v4_stage_sequence_for_run(
                self.project_root,
                str(run_name),
                str(stage_name),
                detect_config=detect_config,
                silhouette_config=silhouette_config,
                pack_dual_stream_config=pack_config,
                log_sink=self._log_sink,
            )
            with self.log_output:
                print("Done.")
                for summary in summaries:
                    print(
                        f"- {summary.stage_name}: success={summary.successful_rows}, "
                        f"failed={summary.failed_rows}, skipped={summary.skipped_rows}"
                    )
        except Exception as exc:
            with self.log_output:
                print(f"Error: {exc}")
        finally:
            self._execution_in_progress = False
            self.execute_button.disabled = False
            self.render_preview()

    def _log_sink(self, message: str) -> None:
        with self.log_output:
            print(message)

    def render_preview(self) -> None:
        run_name = self.run_dropdown.value
        row_offset = int(self.sample_offset_input.value)

        self.preview_source_overlay.value = b""
        self.preview_roi_selection_debug.value = b""
        self.preview_roi_selection_stats_html.value = ""
        self.preview_extracted_roi.value = b""
        self.preview_silhouette_debug.value = b""
        self.preview_silhouette_roi.value = b""
        self.preview_packed_canvas.value = b""
        self.preview_silhouette_full.value = b""
        self.preview_status_html.value = ""
        self._reset_extra_preview_outputs()

        if not run_name:
            self.preview_status_html.value = "Select a run first."
            return

        detect_config = self._build_detect_config()
        silhouette_config = self._build_silhouette_config()
        pack_config = self._build_pack_config()

        input_paths = input_run_paths(self.project_root, str(run_name))
        detect_paths = detect_run_paths(self.project_root, str(run_name))

        input_df = _load_samples(input_paths.manifests_dir)
        detect_df = _load_samples(detect_paths.manifests_dir)

        if input_df is None or len(input_df) == 0:
            self.preview_status_html.value = "Input samples.csv is missing or empty."
            return

        if row_offset < 0 or row_offset >= len(input_df):
            self.preview_status_html.value = f"Sample offset {row_offset} is out of range for {len(input_df)} rows."
            return

        input_row = input_df.iloc[row_offset]
        detect_row = detect_df.iloc[row_offset] if detect_df is not None and row_offset < len(detect_df) else None

        source_filename = _clean_manifest_text(input_row.get("image_filename", ""))
        sample_id = _clean_manifest_text(input_row.get("sample_id", "")) or f"row_{row_offset}"

        source_bgr: np.ndarray | None = None
        source_gray: np.ndarray | None = None
        source_error = ""
        try:
            source_path = resolve_manifest_path(input_paths.root, "images", source_filename)
            source_bgr = _safe_bgr_image(source_path)
            if source_bgr is None:
                raise ValueError(f"unable to decode source image '{source_filename}'")
            source_gray = to_grayscale_uint8(source_bgr)
        except Exception as exc:
            source_bgr = None
            source_gray = None
            source_error = str(exc)

        if source_bgr is None or source_gray is None:
            self.preview_status_html.value = (
                f"Failed to load source image for row {row_offset} ({source_filename}). "
                f"{source_error}"
            )
            return

        detect_status = "not_run"
        detect_error = ""
        detection_source = "none"
        selected_detection: Detection | None = None
        live_candidate_summary = ""

        if detect_row is not None:
            detect_status = _clean_manifest_text(detect_row.get("detect_stage_status", "")) or "not_set"
            detect_error = _clean_manifest_text(detect_row.get("detect_stage_error", ""))

            if str(detect_status).lower() == "success":
                x1 = _safe_float(detect_row.get("detect_bbox_x1"))
                y1 = _safe_float(detect_row.get("detect_bbox_y1"))
                x2 = _safe_float(detect_row.get("detect_bbox_x2"))
                y2 = _safe_float(detect_row.get("detect_bbox_y2"))
                center_x = _safe_float(detect_row.get("detect_center_x_px"))
                center_y = _safe_float(detect_row.get("detect_center_y_px"))
                if None not in {x1, y1, x2, y2}:
                    selected_detection = Detection(
                        class_id=int(_safe_float(detect_row.get("detect_class_id")) or -1),
                        class_name=_clean_manifest_text(detect_row.get("detect_class_name")) or "detected",
                        confidence=float(_safe_float(detect_row.get("detect_confidence")) or 0.0),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        center_x_px=float(center_x) if center_x is not None else None,
                        center_y_px=float(center_y) if center_y is not None else None,
                    )
                    detection_source = "manifest"

        if selected_detection is None:
            try:
                detector = self._get_preview_detector(detect_config)
                candidates = detector.detect(source_bgr)
                sorted_candidates = sorted(candidates, key=lambda item: float(item.confidence), reverse=True)
                if sorted_candidates:
                    live_candidate_summary = ", ".join(
                        f"{str(item.class_name)}:{float(item.confidence):.3f}"
                        for item in sorted_candidates[:3]
                    )

                if detect_config.normalized_detector_backend() == "edge":
                    allowed_ids: set[int] = set()
                    allowed_names: set[str] = set()
                else:
                    allowed_ids = set(detect_config.normalized_defender_class_ids())
                    allowed_names = {name.lower() for name in detect_config.normalized_defender_class_names()}

                selected = _select_detection(
                    candidates,
                    allowed_ids=allowed_ids,
                    allowed_names=allowed_names,
                )
                selected_from_fallback = False
                fallback_error = ""
                if selected is None and bool(self.preview_ignore_filter_checkbox.value) and sorted_candidates:
                    selected = sorted_candidates[0]
                    selected_from_fallback = True
                    fallback_error = "no class-filter match; showing top-confidence candidate"

                if selected is not None:
                    selected_detection = _clamp_detection(
                        selected,
                        frame_width=source_bgr.shape[1],
                        frame_height=source_bgr.shape[0],
                    )
                    detection_source = "live"
                    detect_status = "preview_live_fallback_any_class" if selected_from_fallback else "preview_live_success"
                    detect_error = fallback_error
                elif str(detect_status).lower() not in {"success"}:
                    detect_status = "preview_live_no_detection"
                    detect_error = ""
            except Exception as exc:
                if str(detect_status).lower() not in {"success"}:
                    detect_status = "preview_live_error"
                    detect_error = str(exc)

        edge_mask: np.ndarray | None = None
        edge_error = ""
        if detect_config.normalized_detector_backend() == "edge":
            try:
                edge_mask = _edge_foreground_mask_from_config(
                    source_gray,
                    blur_kernel_size=detect_config.normalized_edge_blur_kernel_size(),
                    canny_low_threshold=detect_config.normalized_edge_canny_low_threshold(),
                    canny_high_threshold=detect_config.normalized_edge_canny_high_threshold(),
                    close_kernel_size=detect_config.normalized_edge_close_kernel_size(),
                    foreground_threshold=detect_config.normalized_edge_foreground_threshold(),
                )
            except Exception as exc:
                edge_error = str(exc)

        roi_selection_summary: dict[str, object] = {}
        roi_selection_debug_error = ""
        if detect_config.normalized_detector_backend() == "edge":
            try:
                roi_selection_debug, roi_selection_summary = _edge_roi_selection_debug_strip(
                    source_gray,
                    blur_kernel_size=detect_config.normalized_edge_blur_kernel_size(),
                    canny_low_threshold=detect_config.normalized_edge_canny_low_threshold(),
                    canny_high_threshold=detect_config.normalized_edge_canny_high_threshold(),
                    close_kernel_size=detect_config.normalized_edge_close_kernel_size(),
                    foreground_threshold=detect_config.normalized_edge_foreground_threshold(),
                    padding_px=detect_config.normalized_edge_padding_px(),
                    min_foreground_px=detect_config.normalized_edge_min_foreground_px(),
                    selected_detection=selected_detection,
                )
                self.preview_roi_selection_debug.value = _to_png_bytes(roi_selection_debug)
            except Exception as exc:
                roi_selection_debug_error = str(exc)

        roi_bounds: tuple[int, int, int, int] | None = None
        roi_source_bounds: tuple[int, int, int, int] | None = None
        roi_canvas_bounds: tuple[int, int, int, int] | None = None
        roi_requested_bounds: tuple[int, int, int, int] | None = None
        extracted_roi_gray: np.ndarray | None = None
        roi_center_note = ""
        roi_extract_error = ""
        if selected_detection is not None:
            try:
                extracted_roi_gray, roi_source_bounds, roi_canvas_bounds, roi_requested_bounds = _extract_centered_canvas_from_detection(
                    source_gray,
                    selected_detection,
                    canvas_width=pack_config.normalized_canvas_width_px(),
                    canvas_height=pack_config.normalized_canvas_height_px(),
                    padding_px=silhouette_config.normalized_roi_padding_px(),
                )
                roi_bounds = roi_source_bounds

                center_x = selected_detection.center_x_px
                center_y = selected_detection.center_y_px
                if center_x is None or center_y is None:
                    center_x = (float(selected_detection.x1) + float(selected_detection.x2)) * 0.5
                    center_y = (float(selected_detection.y1) + float(selected_detection.y2)) * 0.5
                req_x1, req_y1, req_x2, req_y2 = roi_requested_bounds
                roi_mid_x = float(req_x1) + (float(req_x2 - req_x1) * 0.5)
                roi_mid_y = float(req_y1) + (float(req_y2 - req_y1) * 0.5)
                roi_center_note = (
                    f"ROI center offset dx={float(center_x) - roi_mid_x:.2f}px, "
                    f"dy={float(center_y) - roi_mid_y:.2f}px"
                )
            except Exception as exc:
                roi_extract_error = str(exc)

        roi_selection_lines: list[str] = []
        if detect_config.normalized_detector_backend() == "edge":
            roi_selection_lines.append(f"<b>ROI source used for crop:</b> {detection_source}")
            roi_selection_lines.append(
                f"<b>Edge selection status:</b> {str(roi_selection_summary.get('selection_status', 'not_run'))}"
            )
            roi_selection_lines.append(
                "<b>Foreground pixels:</b> "
                f"{int(roi_selection_summary.get('foreground_pixel_count', 0))} "
                f"(min required {int(roi_selection_summary.get('min_foreground_px', 0))})"
            )

            centroid_xy = roi_selection_summary.get("centroid_xy")
            if isinstance(centroid_xy, tuple) and len(centroid_xy) == 2:
                centroid_x = float(centroid_xy[0])
                centroid_y = float(centroid_xy[1])
                roi_selection_lines.append(f"<b>Foreground centroid:</b> ({centroid_x:.2f}, {centroid_y:.2f})")
            else:
                roi_selection_lines.append("<b>Foreground centroid:</b> n/a")

            raw_bbox = roi_selection_summary.get("raw_bbox_inclusive")
            centered_pre = roi_selection_summary.get("centered_pre_clamp_exclusive")
            live_bbox = roi_selection_summary.get("live_bbox_post_clamp_exclusive")
            selected_bbox = roi_selection_summary.get("selected_bbox_exclusive")
            roi_selection_lines.append(f"<b>Raw bbox (+pad):</b> {_format_bbox_inclusive(raw_bbox if isinstance(raw_bbox, tuple) else None)}")
            roi_selection_lines.append(
                f"<b>Centered bbox (pre-clamp):</b> "
                f"{_format_bbox_exclusive(centered_pre if isinstance(centered_pre, tuple) else None)}"
            )
            roi_selection_lines.append(
                f"<b>Live edge bbox (post-clamp):</b> "
                f"{_format_bbox_exclusive(live_bbox if isinstance(live_bbox, tuple) else None)}"
            )
            roi_selection_lines.append(
                f"<b>BBox used for crop:</b> "
                f"{_format_bbox_exclusive(selected_bbox if isinstance(selected_bbox, tuple) else None)}"
            )
            if isinstance(live_bbox, tuple) and isinstance(selected_bbox, tuple):
                roi_selection_lines.append(f"<b>Live bbox matches selected bbox:</b> {live_bbox == selected_bbox}")
        else:
            roi_selection_lines.append(
                "<b>ROI selection debug:</b> unavailable for non-edge detector backend in this preview."
            )

        if roi_requested_bounds is not None:
            roi_selection_lines.append(
                f"<b>Fixed canvas request bounds:</b> {_format_bbox_exclusive(roi_requested_bounds)}"
            )
        if roi_source_bounds is not None:
            roi_selection_lines.append(
                f"<b>Source bounds copied into canvas:</b> {_format_bbox_exclusive(roi_source_bounds)}"
            )
        if roi_canvas_bounds is not None:
            roi_selection_lines.append(
                f"<b>Canvas insertion bounds:</b> {_format_bbox_exclusive(roi_canvas_bounds)}"
            )
        if roi_selection_debug_error:
            roi_selection_lines.append(f"<b>ROI debug error:</b> {roi_selection_debug_error}")
        if roi_extract_error:
            roi_selection_lines.append(f"<b>ROI extraction error:</b> {roi_extract_error}")

        self.preview_roi_selection_stats_html.value = "<br>".join(roi_selection_lines)

        source_overlay = _draw_source_overlay(source_bgr, edge_mask=edge_mask, roi_bounds=roi_bounds)
        self.preview_source_overlay.value = _to_png_bytes(source_overlay)
        if extracted_roi_gray is not None and extracted_roi_gray.size > 0:
            self.preview_extracted_roi.value = _to_png_bytes(extracted_roi_gray)

        silhouette_status = "not_run"
        silhouette_error = ""
        canvas_clipped = False
        brightness_note = ""
        if extracted_roi_gray is not None and extracted_roi_gray.size > 0:
            try:
                generator = ContourSilhouetteGeneratorV2()
                fallback = ConvexHullFallbackV1()
                writer = (
                    FilledArtifactWriterV1()
                    if silhouette_config.normalized_representation_mode() == "filled"
                    else OutlineArtifactWriterV1()
                )

                generated = generator.generate(
                    extracted_roi_gray,
                    blur_kernel_size=silhouette_config.normalized_blur_kernel_size(),
                    canny_low_threshold=int(silhouette_config.canny_low_threshold),
                    canny_high_threshold=int(silhouette_config.canny_high_threshold),
                    close_kernel_size=silhouette_config.normalized_close_kernel_size(),
                    dilate_kernel_size=silhouette_config.normalized_dilate_kernel_size(),
                    min_component_area_px=silhouette_config.normalized_min_component_area_px(),
                    fill_holes=bool(silhouette_config.fill_holes),
                )
                self.preview_silhouette_debug.value = _to_png_bytes(
                    _silhouette_debug_strip(
                        generated.debug_images,
                        tile_height=int(extracted_roi_gray.shape[0]),
                        tile_width=int(extracted_roi_gray.shape[1]),
                    )
                )

                contour = generated.contour
                primary_break = _contour_break_reason(contour)
                fallback_used = False
                if primary_break:
                    if not bool(silhouette_config.use_convex_hull_fallback):
                        reason = generated.primary_reason or primary_break
                        raise ValueError(f"primary contour failed: {reason}")
                    contour, recover_reason = fallback.recover(generated.fallback_mask)
                    fallback_used = True
                    if contour is None:
                        raise ValueError(f"fallback failed: {recover_reason}")

                roi_silhouette = writer.render(
                    extracted_roi_gray.shape,
                    contour,
                    line_thickness=silhouette_config.normalized_outline_thickness(),
                )
                if _render_is_empty(roi_silhouette):
                    if not fallback_used and bool(silhouette_config.use_convex_hull_fallback):
                        contour, recover_reason = fallback.recover(generated.fallback_mask)
                        if contour is None:
                            raise ValueError(f"fallback failed: {recover_reason}")
                        roi_silhouette = writer.render(
                            extracted_roi_gray.shape,
                            contour,
                            line_thickness=silhouette_config.normalized_outline_thickness(),
                        )
                    if _render_is_empty(roi_silhouette):
                        raise ValueError("rendered silhouette is empty")

                background_mask = _background_mask_from_binary_silhouette(roi_silhouette)
                roi_repr = _render_inverted_vehicle_detail_on_white_for_preview(
                    extracted_roi_gray,
                    background_mask,
                )
                brightness_config = pack_config.normalized_brightness_normalization()
                brightness_method = brightness_config.normalized_method()
                if brightness_config.normalized_enabled() and brightness_method != "none":
                    brightness_result = apply_brightness_normalization_v4(
                        roi_repr,
                        background_mask < 0.5,
                        brightness_config,
                    )
                    roi_repr = brightness_result.image
                    if brightness_result.status == "success":
                        brightness_note = (
                            "brightness normalization "
                            f"{brightness_result.method}: gain={brightness_result.gain:.4f}, "
                            f"median={brightness_result.current_median_darkness:.4f}"
                        )
                    else:
                        brightness_note = f"brightness normalization {brightness_result.status}"
                roi_canvas, canvas_clipped = _place_image_on_canvas_for_preview(
                    roi_repr,
                    canvas_height=pack_config.normalized_canvas_height_px(),
                    canvas_width=pack_config.normalized_canvas_width_px(),
                    clip_policy=pack_config.normalized_clip_policy(),
                )

                self.preview_silhouette_roi.value = _to_png_bytes(_preview_float_image_to_uint8(roi_repr))
                self.preview_packed_canvas.value = _to_png_bytes(_preview_float_image_to_uint8(roi_canvas))
                self._render_extra_pack_preview(
                    extracted_roi_gray=extracted_roi_gray,
                    background_mask=background_mask,
                    pack_config=pack_config,
                )

                if roi_source_bounds is not None and roi_canvas_bounds is not None:
                    src_x1, src_y1, src_x2, src_y2 = roi_source_bounds
                    roi_x1, roi_y1, roi_x2, roi_y2 = roi_canvas_bounds
                    full_silhouette = np.full(source_gray.shape, 255, dtype=np.uint8)
                    roi_target = full_silhouette[src_y1:src_y2, src_x1:src_x2]
                    roi_source_aligned = roi_silhouette[roi_y1:roi_y2, roi_x1:roi_x2]
                    roi_target[roi_source_aligned < 255] = 0
                    full_silhouette[src_y1:src_y2, src_x1:src_x2] = roi_target
                    self.preview_silhouette_full.value = _to_png_bytes(full_silhouette)

                silhouette_status = "preview_live_success"
                silhouette_error = ""
            except Exception as exc:
                silhouette_status = "preview_live_error"
                silhouette_error = str(exc)
        elif selected_detection is None:
            silhouette_status = "skipped_no_detection"
            silhouette_error = "no detection available for ROI extraction"
        else:
            silhouette_status = "preview_live_error"
            silhouette_error = roi_extract_error or "empty ROI after centered canvas extraction"

        notes: list[str] = []
        if detect_df is None:
            if detect_config.normalized_detector_backend() == "edge":
                notes.append("detect manifest not found; using live edge ROI detection for preview")
            else:
                notes.append("detect manifest not found; using live detector output for preview")
        if source_error:
            notes.append(f"source load error: {source_error}")
        if detection_source == "manifest":
            notes.append("ROI geometry sourced from detect manifest row")
        elif detection_source == "live":
            notes.append("ROI geometry generated live for preview")
        if live_candidate_summary:
            notes.append(f"top live candidates: {live_candidate_summary}")
        if edge_error:
            notes.append(f"edge overlay unavailable: {edge_error}")
        if roi_extract_error:
            notes.append(f"ROI extraction error: {roi_extract_error}")
        if canvas_clipped:
            notes.append("packed canvas clipped ROI to fit configured dimensions")
        if roi_center_note:
            notes.append(roi_center_note)
        notes.append("preview overlays are display-only copies; they never enter saved training arrays")

        status_lines = [
            f"<b>Sample:</b> {sample_id} | <b>Sample offset:</b> {int(row_offset)} | <b>Image:</b> {source_filename}",
            f"<b>Detect:</b> {detect_status} [source={detection_source}]" + (f" ({detect_error})" if detect_error else ""),
            f"<b>Silhouette:</b> {silhouette_status}" + (f" ({silhouette_error})" if silhouette_error else ""),
            f"<b>ROI Canvas (from Pack):</b> {pack_config.normalized_canvas_width_px()}x{pack_config.normalized_canvas_height_px()}",
            self._saved_path_preview_text(pack_config),
        ]
        if brightness_note:
            notes.append(brightness_note)
        if notes:
            status_lines.append(f"<b>Notes:</b> {'; '.join(notes)}")
        self.preview_status_html.value = "<br>".join(status_lines)


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    text = str(raw).strip()
    if not text:
        return ()

    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return tuple(values)


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    text = str(raw).strip()
    if not text:
        return ()

    values: list[str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(part)
    return tuple(values)


def display_pipeline_launcher_v4(start: Path | None = None) -> PipelineLauncherV4:
    """Locate project root and display the v4 pipeline launcher widget."""

    project_root = find_project_root(start)
    launcher = PipelineLauncherV4(project_root)
    display(launcher.widget)
    return launcher
