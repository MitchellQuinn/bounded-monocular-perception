"""Concrete tri-stream raw image preprocessor for live inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path
import sys
from typing import Any

import interfaces.contracts as contracts
from interfaces.contracts import InferenceRequest, PreparedInferenceInputs
from live_inference.masking import (
    BackgroundSnapshot,
    BackgroundState,
    FrameMaskSnapshot,
    FrameMaskState,
    apply_fill_to_mask,
    combine_ignore_masks,
    compute_background_removal_mask,
)
from live_inference.model_registry.model_manifest import (
    ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE,
    LiveModelManifest,
)

from .preprocessing_config import (
    BrightnessNormalizationRuntimeConfig,
    TriStreamPreprocessingConfig,
)
from .roi_locator import RoiLocation, RoiLocator


def _ensure_preprocessing_paths() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    for path in (repo_root / "02_synthetic-data-processing-v4.0",):
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


_ensure_preprocessing_paths()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rb_pipeline_v4.brightness_normalization import (  # noqa: E402
    BrightnessNormalizationResultV4,
    apply_brightness_normalization_v4,
)
from rb_pipeline_v4.image_io import to_grayscale_uint8  # noqa: E402
from rb_pipeline_v4.pack_dual_stream_stage import (  # noqa: E402
    _place_image_on_canvas,
    _reconstruct_roi_canvas_from_source,
    _render_inverted_vehicle_detail_on_white,
    _silhouette_to_background_mask,
)
from rb_pipeline_v4.pack_tri_stream_stage import (  # noqa: E402
    _render_orientation_image_scaled_by_foreground_extent,
)
from rb_pipeline_v4.silhouette_algorithms import (  # noqa: E402
    ContourSilhouetteGeneratorV2,
    ConvexHullFallbackV1,
    FilledArtifactWriterV1,
    OutlineArtifactWriterV1,
)

from .debug_artifacts import (  # noqa: E402
    ARTIFACT_ACCEPTED_RAW_FRAME,
    ARTIFACT_DISTANCE_IMAGE,
    ARTIFACT_LOCATOR_INPUT,
    ARTIFACT_ORIENTATION_IMAGE,
    ARTIFACT_ROI_CROP,
    DebugArtifactWriter,
    default_debug_output_dir,
)
from .roi_locator import build_roi_fcn_locator_input  # noqa: E402


@dataclass(frozen=True)
class _SilhouetteResult:
    roi_silhouette: np.ndarray
    full_silhouette: np.ndarray
    area_px: int
    bbox_inclusive_xyxy_px: tuple[int, int, int, int]
    feature_bbox_xyxy_px: np.ndarray
    fallback_used: bool
    primary_break_reason: str
    diagnostics: Mapping[str, Any]


class TriStreamLivePreprocessor:
    """Prepare live raw image bytes as tri-stream model inputs."""

    def __init__(
        self,
        *,
        roi_locator: RoiLocator,
        model_manifest: LiveModelManifest | None = None,
        config: TriStreamPreprocessingConfig | None = None,
        runtime_parameter_revision_getter: Callable[[], int | None] | None = None,
        mask_state: FrameMaskState | None = None,
        background_state: BackgroundState | None = None,
    ) -> None:
        if config is None:
            if model_manifest is None:
                raise ValueError(
                    "TriStreamLivePreprocessor requires a model_manifest or explicit config."
                )
            config = TriStreamPreprocessingConfig.from_manifest(model_manifest)
        config.validate()
        self._config = config
        self._roi_locator = roi_locator
        self._runtime_parameter_revision_getter = runtime_parameter_revision_getter
        self._mask_state = mask_state
        self._background_state = background_state

    @property
    def config(self) -> TriStreamPreprocessingConfig:
        return self._config

    def prepare_model_inputs(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
    ) -> PreparedInferenceInputs:
        """Decode raw bytes and reproduce the v0.4 tri-stream preprocessing contract."""
        decoded_source_gray = _decode_image_bytes_to_grayscale(image_bytes)
        source_gray, mask_metadata, roi_exclusion_mask = self._apply_preprocessing_masks(
            decoded_source_gray
        )
        source_h, source_w = int(source_gray.shape[0]), int(source_gray.shape[1])
        warnings = _hash_warnings(request, image_bytes)
        for warning_key in ("frame_mask_warning", "background_warning"):
            warning = mask_metadata.get(warning_key)
            if warning:
                warnings.append(str(warning))

        roi_location = self._locate_roi(
            source_gray,
            excluded_source_mask=roi_exclusion_mask,
        )
        center_x_px, center_y_px = _coerce_center_xy(roi_location)
        silhouette_w = int(self._config.silhouette_config.normalized_roi_canvas_width_px())
        silhouette_h = int(self._config.silhouette_config.normalized_roi_canvas_height_px())

        roi_gray, source_bounds, roi_bounds, request_bounds = _extract_centered_canvas(
            source_gray,
            center_x_px=center_x_px,
            center_y_px=center_y_px,
            canvas_width_px=silhouette_w,
            canvas_height_px=silhouette_h,
        )
        silhouette_result = self._render_silhouette(
            roi_gray=roi_gray,
            source_gray=source_gray,
            source_bounds=source_bounds,
            roi_bounds=roi_bounds,
        )

        background_mask = _silhouette_to_background_mask(silhouette_result.roi_silhouette)
        foreground_mask = background_mask < 0.5
        roi_source_gray = _reconstruct_roi_canvas_from_source(
            source_gray,
            source_xyxy=np.asarray(source_bounds, dtype=np.float32),
            canvas_insert_xyxy=np.asarray(roi_bounds, dtype=np.float32),
            canvas_width=silhouette_w,
            canvas_height=silhouette_h,
        )
        roi_repr = _render_inverted_vehicle_detail_on_white(
            roi_source_gray,
            background_mask,
        )
        inverted_orientation_repr = roi_repr
        distance_image_2d, brightness_payload, distance_clipped = self._build_distance_image(
            roi_repr=roi_repr,
            foreground_mask=foreground_mask,
        )
        (
            orientation_image_2d,
            orientation_source_extent_xyxy,
            orientation_crop_source_xyxy,
            orientation_crop_size_px,
        ) = self._build_orientation_image(
            roi_source_gray=roi_source_gray,
            inverted_orientation_repr=inverted_orientation_repr,
            foreground_mask=foreground_mask,
        )
        geometry = _bbox_features_from_xyxy(
            silhouette_result.feature_bbox_xyxy_px,
            image_width_px=source_w,
            image_height_px=source_h,
        )
        runtime_revision = self._runtime_parameter_revision()
        input_image_hash = _accepted_input_image_hash(request, image_bytes)

        metadata = {
            "preprocessing_contract_name": self._config.preprocessing_contract_name,
            "preprocessing_contract_version": self._config.preprocessing_contract_version,
            "input_mode": contracts.TRI_STREAM_INPUT_MODE,
            "input_keys": contracts.TRI_STREAM_INPUT_KEYS,
            "representation_kind": self._config.representation_kind,
            "geometry_schema": self._config.geometry_schema,
            "geometry_dim": int(self._config.geometry_dim),
            "input_image_hash": input_image_hash.value,
            "input_image_hash_algorithm": input_image_hash.algorithm,
            "source_image_width_px": source_w,
            "source_image_height_px": source_h,
            "distance_canvas_width_px": int(self._config.distance_canvas_size[0]),
            "distance_canvas_height_px": int(self._config.distance_canvas_size[1]),
            "orientation_canvas_width_px": int(self._config.orientation_canvas_size[0]),
            "orientation_canvas_height_px": int(self._config.orientation_canvas_size[1]),
            "orientation_source_mode": self._config.orientation_source_mode,
            "roi_request_xyxy_px": _array_xyxy_to_tuple(request_bounds),
            "roi_source_xyxy_px": _array_xyxy_to_tuple(source_bounds),
            "roi_canvas_insert_xyxy_px": _array_xyxy_to_tuple(roi_bounds),
            "roi_locator_bounds_xyxy_px": roi_location.roi_bounds_xyxy_px,
            "roi_locator_metadata": dict(roi_location.metadata),
            "predicted_roi_center_xy_px": (float(center_x_px), float(center_y_px)),
            "silhouette_bbox_xyxy_px": _array_xyxy_to_tuple(
                silhouette_result.feature_bbox_xyxy_px
            ),
            "silhouette_bbox_inclusive_xyxy_px": silhouette_result.bbox_inclusive_xyxy_px,
            "silhouette_area_px": int(silhouette_result.area_px),
            "silhouette_fallback_used": bool(silhouette_result.fallback_used),
            "silhouette_primary_break_reason": silhouette_result.primary_break_reason,
            "silhouette_diagnostics": dict(silhouette_result.diagnostics),
            "brightness_normalization": brightness_payload,
            "distance_clipped": bool(distance_clipped),
            "orientation_context_scale": float(self._config.orientation_context_scale),
            "orientation_source_extent_xyxy_px": _array_xyxy_to_tuple(
                orientation_source_extent_xyxy
            ),
            "orientation_crop_source_xyxy_px": _array_xyxy_to_tuple(
                orientation_crop_source_xyxy
            ),
            "orientation_crop_size_px": float(orientation_crop_size_px),
            "runtime_parameter_revision": runtime_revision,
            "warnings": tuple(warnings),
        }
        metadata.update(mask_metadata)
        debug_paths = self._write_debug_artifacts(
            request=request,
            input_image_hash=input_image_hash,
            preprocessing_parameter_revision=runtime_revision,
            source_gray=source_gray,
            roi_location=roi_location,
            roi_crop=roi_source_gray,
            distance_image=distance_image_2d,
            orientation_image=orientation_image_2d,
            metadata=metadata,
        )
        if debug_paths:
            metadata = dict(metadata)
            metadata["debug_paths"] = {
                str(kind): str(path) for kind, path in debug_paths.items()
            }

        return PreparedInferenceInputs(
            request_id=request.request_id,
            input_mode=contracts.InferenceInputMode.TRI_STREAM_V0_4,
            input_keys=contracts.TRI_STREAM_INPUT_KEYS,
            model_inputs={
                contracts.TRI_STREAM_DISTANCE_IMAGE_KEY: distance_image_2d[
                    None, ...
                ].astype(np.float32, copy=False),
                contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY: orientation_image_2d[
                    None, ...
                ].astype(np.float32, copy=False),
                contracts.TRI_STREAM_GEOMETRY_KEY: geometry.astype(np.float32, copy=False),
            },
            source_frame=request.frame,
            preprocessing_metadata=metadata,
        )

    def _locate_roi(
        self,
        source_gray: np.ndarray,
        *,
        excluded_source_mask: np.ndarray | None = None,
    ) -> RoiLocation:
        if excluded_source_mask is not None and _locator_accepts_exclusion_mask(
            self._roi_locator
        ):
            location = self._roi_locator.locate(
                source_gray,
                excluded_source_mask=excluded_source_mask,
            )
        else:
            location = self._roi_locator.locate(source_gray)
        if not isinstance(location, RoiLocation):
            raise TypeError(
                "ROI locator must return live_inference.preprocessing.RoiLocation; "
                f"got {type(location).__name__}."
            )
        return location

    def _apply_preprocessing_masks(
        self,
        source_gray: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any], np.ndarray | None]:
        source_h, source_w = int(source_gray.shape[0]), int(source_gray.shape[1])
        frame_snapshot = (
            self._mask_state.get_snapshot() if self._mask_state is not None else None
        )
        background_snapshot = (
            self._background_state.get_snapshot()
            if self._background_state is not None
            else None
        )
        fill_value = int(frame_snapshot.fill_value) if frame_snapshot is not None else 255
        metadata = _frame_mask_metadata(
            snapshot=frame_snapshot,
            source_width_px=source_w,
            source_height_px=source_h,
            applied=False,
            fill_value=fill_value,
        )
        metadata.update(
            _background_metadata(
                snapshot=background_snapshot,
                applied=False,
                remove_pixel_count=0,
                warning=None,
            )
        )

        manual_mask: np.ndarray | None = None
        if (
            frame_snapshot is not None
            and frame_snapshot.enabled
            and frame_snapshot.has_geometry
            and frame_snapshot.pixel_count > 0
        ):
            if not frame_snapshot.dimensions_match(source_w, source_h):
                warning = (
                    "frame mask skipped: mask size "
                    f"{(frame_snapshot.width_px, frame_snapshot.height_px)} "
                    f"does not match source image size {(source_w, source_h)}."
                )
                metadata["frame_mask_warning"] = warning
            else:
                manual_mask = frame_snapshot.mask
                metadata["frame_mask_applied"] = True

        background_result = compute_background_removal_mask(source_gray, background_snapshot)
        if background_result.warning:
            metadata["background_warning"] = background_result.warning
        metadata.update(
            _background_metadata(
                snapshot=background_snapshot,
                applied=background_result.applied,
                remove_pixel_count=background_result.pixel_count,
                warning=background_result.warning,
            )
        )

        combined_ignore_mask = combine_ignore_masks(
            shape=(source_h, source_w),
            manual_mask=manual_mask,
            background_mask=background_result.mask,
        )
        combined_count = (
            int(np.count_nonzero(combined_ignore_mask))
            if combined_ignore_mask is not None
            else 0
        )
        metadata["combined_ignore_pixel_count"] = combined_count
        metadata["combined_ignore_excluded_from_roi_locator"] = False
        if combined_ignore_mask is None:
            return source_gray, metadata, None

        masked = apply_fill_to_mask(
            source_gray,
            combined_ignore_mask,
            fill_value=fill_value,
        )
        excluded = _locator_accepts_exclusion_mask(self._roi_locator)
        metadata["frame_mask_excluded_from_roi_locator"] = excluded
        metadata["combined_ignore_excluded_from_roi_locator"] = excluded
        return masked, metadata, combined_ignore_mask

    def _render_silhouette(
        self,
        *,
        roi_gray: np.ndarray,
        source_gray: np.ndarray,
        source_bounds: np.ndarray,
        roi_bounds: np.ndarray,
    ) -> _SilhouetteResult:
        silhouette_config = self._config.silhouette_config
        generator, fallback, writer = _select_silhouette_components(silhouette_config)
        generated = generator.generate(
            roi_gray,
            blur_kernel_size=silhouette_config.normalized_blur_kernel_size(),
            canny_low_threshold=int(silhouette_config.canny_low_threshold),
            canny_high_threshold=int(silhouette_config.canny_high_threshold),
            close_kernel_size=silhouette_config.normalized_close_kernel_size(),
            dilate_kernel_size=silhouette_config.normalized_dilate_kernel_size(),
            min_component_area_px=silhouette_config.normalized_min_component_area_px(),
            fill_holes=bool(silhouette_config.fill_holes),
        )

        contour = generated.contour
        primary_break_reason = _contour_break_reason(contour)
        fallback_used = False
        if primary_break_reason:
            if not bool(silhouette_config.use_convex_hull_fallback):
                break_reason = generated.primary_reason or primary_break_reason
                raise ValueError(
                    f"Primary contour failed ({break_reason}) and fallback is disabled"
                )
            contour, recovery_reason = fallback.recover(generated.fallback_mask)
            fallback_used = True
            if contour is None:
                raise ValueError(f"Fallback failed: {recovery_reason}")

        roi_silhouette = writer.render(
            roi_gray.shape,
            contour,
            line_thickness=silhouette_config.normalized_outline_thickness(),
        )
        if _render_is_empty(roi_silhouette):
            if not fallback_used and bool(silhouette_config.use_convex_hull_fallback):
                contour, recovery_reason = fallback.recover(generated.fallback_mask)
                fallback_used = True
                if contour is None:
                    raise ValueError(f"Fallback failed: {recovery_reason}")
                roi_silhouette = writer.render(
                    roi_gray.shape,
                    contour,
                    line_thickness=silhouette_config.normalized_outline_thickness(),
                )
            if _render_is_empty(roi_silhouette):
                raise ValueError("Rendered silhouette is empty after fallback")

        src_x1, src_y1, src_x2, src_y2 = [int(value) for value in source_bounds.tolist()]
        roi_x1, roi_y1, roi_x2, roi_y2 = [int(value) for value in roi_bounds.tolist()]
        full_silhouette = np.full(source_gray.shape, 255, dtype=np.uint8)
        roi_target = full_silhouette[src_y1:src_y2, src_x1:src_x2]
        roi_source_aligned = roi_silhouette[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_target[roi_source_aligned < 255] = 0
        full_silhouette[src_y1:src_y2, src_x1:src_x2] = roi_target

        area_px, bbox = _mask_geometry(full_silhouette < 255)
        if area_px > 0:
            feature_bbox_xyxy = np.asarray(
                [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(min(int(source_gray.shape[1]), bbox[2] + 1)),
                    float(min(int(source_gray.shape[0]), bbox[3] + 1)),
                ],
                dtype=np.float32,
            )
        else:
            feature_bbox_xyxy = np.asarray(source_bounds, dtype=np.float32)

        return _SilhouetteResult(
            roi_silhouette=roi_silhouette,
            full_silhouette=full_silhouette,
            area_px=area_px,
            bbox_inclusive_xyxy_px=bbox,
            feature_bbox_xyxy_px=feature_bbox_xyxy,
            fallback_used=fallback_used,
            primary_break_reason=primary_break_reason,
            diagnostics=getattr(generated, "diagnostics", {}),
        )

    def _build_distance_image(
        self,
        *,
        roi_repr: np.ndarray,
        foreground_mask: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any], bool]:
        runtime = self._config.brightness_runtime
        canvas_w, canvas_h = self._config.distance_canvas_size
        if runtime.active():
            expected_canvas_shape = (int(canvas_h), int(canvas_w))
            image_shape = tuple(roi_repr.shape)
            mask_shape = tuple(foreground_mask.shape)
            if image_shape != expected_canvas_shape or mask_shape != expected_canvas_shape:
                raise ValueError(
                    "Distance/yaw model expects brightness normalization, but the "
                    "reconstructed foreground mask is not aligned with the regressor "
                    f"canvas: image={image_shape}, mask={mask_shape}, "
                    f"expected_canvas={expected_canvas_shape}."
                )
            brightness_result = apply_brightness_normalization_v4(
                roi_repr,
                foreground_mask.astype(bool, copy=False),
                runtime.config,
            )
            roi_repr = brightness_result.image
            brightness_payload = _brightness_result_payload(runtime, brightness_result)
        else:
            brightness_payload = _disabled_brightness_payload(runtime, foreground_mask)

        canvas, clipped = _place_image_on_canvas(
            roi_repr,
            canvas_height=int(canvas_h),
            canvas_width=int(canvas_w),
            clip_policy=str(self._config.clip_policy),
        )
        return canvas.astype(np.float32, copy=False), brightness_payload, bool(clipped)

    def _build_orientation_image(
        self,
        *,
        roi_source_gray: np.ndarray,
        inverted_orientation_repr: np.ndarray,
        foreground_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        orientation_source_mode = self._config.orientation_source_mode
        if orientation_source_mode == ORIENTATION_SOURCE_RAW_GRAYSCALE:
            orientation_source_image = roi_source_gray
        elif orientation_source_mode == ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE:
            orientation_source_image = inverted_orientation_repr
        else:
            raise ValueError(
                "Unsupported resolved tri-stream orientation source mode: "
                f"{orientation_source_mode!r}."
            )

        canvas_w, canvas_h = self._config.orientation_canvas_size
        return _render_orientation_image_scaled_by_foreground_extent(
            orientation_source_image,
            foreground_mask.astype(np.float32, copy=False),
            canvas_height=int(canvas_h),
            canvas_width=int(canvas_w),
            context_scale=float(self._config.orientation_context_scale),
        )

    def _runtime_parameter_revision(self) -> int | None:
        if self._runtime_parameter_revision_getter is None:
            return None
        revision = self._runtime_parameter_revision_getter()
        return int(revision) if revision is not None else None

    def _write_debug_artifacts(
        self,
        *,
        request: InferenceRequest,
        input_image_hash: contracts.FrameHash,
        preprocessing_parameter_revision: int | None,
        source_gray: np.ndarray,
        roi_location: RoiLocation,
        roi_crop: np.ndarray,
        distance_image: np.ndarray,
        orientation_image: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> dict[str, Path]:
        if not bool(request.save_debug_images):
            return {}
        output_dir = (
            Path(request.debug_output_dir)
            if request.debug_output_dir is not None
            else default_debug_output_dir()
        )
        writer = DebugArtifactWriter(enabled=True, output_dir=output_dir)
        return writer.write_preprocessing_artifacts(
            request_id=request.request_id,
            input_image_hash=input_image_hash,
            preprocessing_parameter_revision=preprocessing_parameter_revision,
            image_artifacts={
                ARTIFACT_ACCEPTED_RAW_FRAME: source_gray,
                ARTIFACT_ROI_CROP: roi_crop,
                ARTIFACT_LOCATOR_INPUT: _debug_locator_input(source_gray, roi_location),
                ARTIFACT_DISTANCE_IMAGE: distance_image,
                ARTIFACT_ORIENTATION_IMAGE: orientation_image,
            },
            metadata=metadata,
        )


def _decode_image_bytes_to_grayscale(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Could not decode image bytes: payload is empty.")
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Could not decode image bytes as a supported image.")
    return to_grayscale_uint8(decoded)


def _frame_mask_metadata(
    *,
    snapshot: FrameMaskSnapshot | None,
    source_width_px: int,
    source_height_px: int,
    applied: bool,
    fill_value: int | None = None,
) -> dict[str, Any]:
    if snapshot is None:
        return {
            "frame_mask_applied": False,
            "frame_mask_revision": None,
            "frame_mask_width_px": None,
            "frame_mask_height_px": None,
            "frame_mask_pixel_count": 0,
            "frame_mask_fill_value": fill_value,
            "frame_mask_excluded_from_roi_locator": False,
        }
    return {
        "frame_mask_applied": bool(applied),
        "frame_mask_revision": int(snapshot.revision),
        "frame_mask_width_px": int(snapshot.width_px),
        "frame_mask_height_px": int(snapshot.height_px),
        "frame_mask_pixel_count": int(snapshot.pixel_count),
        "frame_mask_fill_value": int(snapshot.fill_value),
        "frame_mask_source_width_px": int(source_width_px),
        "frame_mask_source_height_px": int(source_height_px),
        "frame_mask_excluded_from_roi_locator": False,
    }


def _background_metadata(
    *,
    snapshot: BackgroundSnapshot | None,
    applied: bool,
    remove_pixel_count: int,
    warning: str | None,
) -> dict[str, Any]:
    if snapshot is None:
        return {
            "background_captured": False,
            "background_removal_enabled": False,
            "background_removal_applied": False,
            "background_revision": None,
            "background_threshold": None,
            "background_remove_pixel_count": int(remove_pixel_count),
            "background_warning": warning,
        }
    return {
        "background_captured": bool(snapshot.captured),
        "background_removal_enabled": bool(snapshot.enabled),
        "background_removal_applied": bool(applied),
        "background_revision": int(snapshot.revision) if snapshot.captured else None,
        "background_threshold": int(snapshot.threshold),
        "background_remove_pixel_count": int(remove_pixel_count),
        "background_warning": warning,
    }


def _locator_accepts_exclusion_mask(locator: RoiLocator) -> bool:
    locate = getattr(locator, "locate", None)
    if locate is None:
        return False
    try:
        signature = inspect.signature(locate)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "excluded_source_mask":
            return True
    return False


def _extract_centered_canvas(
    source_gray: np.ndarray,
    *,
    center_x_px: float,
    center_y_px: float,
    canvas_width_px: int,
    canvas_height_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract one fixed-size ROI canvas centred on the predicted ROI-FCN point."""
    if source_gray.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got {source_gray.shape}")

    frame_height = int(source_gray.shape[0])
    frame_width = int(source_gray.shape[1])
    canvas_w = max(1, int(canvas_width_px))
    canvas_h = max(1, int(canvas_height_px))

    req_x1 = int(round(float(center_x_px) - (canvas_w / 2.0)))
    req_y1 = int(round(float(center_y_px) - (canvas_h / 2.0)))
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
        np.asarray([src_x1, src_y1, src_x2, src_y2], dtype=np.float32),
        np.asarray([dst_x1, dst_y1, dst_x2, dst_y2], dtype=np.float32),
        np.asarray([req_x1, req_y1, req_x2, req_y2], dtype=np.float32),
    )


def _select_silhouette_components(
    silhouette_config: Any,
) -> tuple[
    ContourSilhouetteGeneratorV2,
    ConvexHullFallbackV1,
    FilledArtifactWriterV1 | OutlineArtifactWriterV1,
]:
    if silhouette_config.normalized_generator_id() != "silhouette.contour_v2":
        raise ValueError("Only generator_id='silhouette.contour_v2' is supported")
    if silhouette_config.normalized_fallback_id() != "fallback.convex_hull_v1":
        raise ValueError("Only fallback_id='fallback.convex_hull_v1' is supported")
    mode = silhouette_config.normalized_representation_mode()
    writer = FilledArtifactWriterV1() if mode == "filled" else OutlineArtifactWriterV1()
    return ContourSilhouetteGeneratorV2(), ConvexHullFallbackV1(), writer


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


def _mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, (0, 0, 0, 0)
    return int(xs.size), (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _bbox_features_from_xyxy(
    bbox_xyxy: np.ndarray,
    *,
    image_width_px: int,
    image_height_px: int,
) -> np.ndarray:
    x1 = float(bbox_xyxy[0])
    y1 = float(bbox_xyxy[1])
    x2 = float(bbox_xyxy[2])
    y2 = float(bbox_xyxy[3])

    frame_w = max(1.0, float(image_width_px))
    frame_h = max(1.0, float(image_height_px))
    w_px = max(1e-6, x2 - x1)
    h_px = max(1e-6, y2 - y1)
    cx_px = x1 + (0.5 * w_px)
    cy_px = y1 + (0.5 * h_px)

    values = np.asarray(
        [
            cx_px,
            cy_px,
            w_px,
            h_px,
            cx_px / frame_w,
            cy_px / frame_h,
            w_px / frame_w,
            h_px / frame_h,
            w_px / h_px,
            (w_px * h_px) / (frame_w * frame_h),
        ],
        dtype=np.float32,
    )
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("bbox feature vector contains NaN or Inf")
    return values


def _brightness_result_payload(
    runtime: BrightnessNormalizationRuntimeConfig,
    result: BrightnessNormalizationResultV4,
) -> dict[str, Any]:
    payload = runtime.to_log_dict()
    payload.update(
        {
            "Status": result.status,
            "ForegroundPixelCount": int(result.foreground_pixel_count),
            "CurrentMedianDarkness": float(result.current_median_darkness),
            "EffectiveMedianDarkness": float(result.effective_median_darkness),
            "Gain": float(result.gain),
        }
    )
    return payload


def _disabled_brightness_payload(
    runtime: BrightnessNormalizationRuntimeConfig,
    foreground_mask: np.ndarray,
) -> dict[str, Any]:
    result = BrightnessNormalizationResultV4(
        image=np.empty((0, 0), dtype=np.float32),
        status="disabled",
        method=runtime.config.normalized_method(),
        foreground_pixel_count=int(np.count_nonzero(foreground_mask)),
        current_median_darkness=float("nan"),
        effective_median_darkness=float("nan"),
        gain=1.0,
    )
    return _brightness_result_payload(runtime, result)


def _coerce_center_xy(location: RoiLocation) -> tuple[float, float]:
    try:
        center_x, center_y = location.center_xy_px
    except Exception as exc:
        raise ValueError("ROI location must include center_xy_px=(x, y).") from exc
    center = (float(center_x), float(center_y))
    if not all(math.isfinite(value) for value in center):
        raise ValueError(f"ROI location center must be finite; got {center!r}.")
    return center


def _debug_locator_input(
    source_gray: np.ndarray,
    roi_location: RoiLocation,
) -> np.ndarray | None:
    metadata = roi_location.metadata
    canvas_width = _optional_positive_int(metadata.get("locator_canvas_width_px"))
    canvas_height = _optional_positive_int(metadata.get("locator_canvas_height_px"))
    if canvas_width is None or canvas_height is None:
        return None
    return build_roi_fcn_locator_input(
        source_gray,
        canvas_width_px=canvas_width,
        canvas_height_px=canvas_height,
    ).locator_image


def _optional_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _array_xyxy_to_tuple(values: np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float32).reshape(4)
    return (
        float(array[0]),
        float(array[1]),
        float(array[2]),
        float(array[3]),
    )


def _accepted_input_image_hash(
    request: InferenceRequest,
    image_bytes: bytes,
) -> contracts.FrameHash:
    frame_hash = request.frame.frame_hash
    algorithm = (
        frame_hash.algorithm
        if frame_hash is not None
        else contracts.DEFAULT_FRAME_HASH_ALGORITHM
    )
    digest_size = (
        int(frame_hash.digest_size_bytes)
        if frame_hash is not None
        else contracts.DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES
    )
    if not str(algorithm).startswith("blake2b"):
        if frame_hash is not None:
            return frame_hash
        algorithm = contracts.DEFAULT_FRAME_HASH_ALGORITHM
        digest_size = contracts.DEFAULT_FRAME_HASH_DIGEST_SIZE_BYTES
    digest = hashlib.blake2b(image_bytes, digest_size=int(digest_size)).hexdigest()
    return contracts.FrameHash(
        value=digest,
        algorithm=str(algorithm),
        digest_size_bytes=int(digest_size),
    )


def _hash_warnings(request: InferenceRequest, image_bytes: bytes) -> list[str]:
    frame_hash = request.frame.frame_hash
    if frame_hash is None:
        return []
    if frame_hash.algorithm != contracts.DEFAULT_FRAME_HASH_ALGORITHM:
        return []
    digest = hashlib.blake2b(
        image_bytes,
        digest_size=int(frame_hash.digest_size_bytes),
    ).hexdigest()
    if digest == frame_hash.value:
        return []
    return [
        "request.frame.frame_hash does not match the supplied image bytes: "
        f"expected {frame_hash.value!r}, computed {digest!r}."
    ]


__all__ = [
    "TriStreamLivePreprocessor",
    "_bbox_features_from_xyxy",
    "_brightness_result_payload",
    "_contour_break_reason",
    "_disabled_brightness_payload",
    "_extract_centered_canvas",
    "_mask_geometry",
    "_render_is_empty",
]
