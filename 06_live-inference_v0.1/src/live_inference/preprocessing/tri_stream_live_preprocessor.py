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
    compute_background_removal_mask_from_arrays,
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


@dataclass(frozen=True)
class _MaskPreparation:
    source_gray: np.ndarray
    metadata: dict[str, Any]
    roi_exclusion_mask: np.ndarray | None
    background_snapshot: BackgroundSnapshot | None
    fill_value: int


@dataclass(frozen=True)
class _RoiBackgroundRemoval:
    preview_gray: np.ndarray
    removal_mask: np.ndarray | None
    metadata: dict[str, Any]


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
        mask_preparation = self._prepare_source_masks(decoded_source_gray)
        source_gray = mask_preparation.source_gray
        mask_metadata = mask_preparation.metadata
        source_h, source_w = int(source_gray.shape[0]), int(source_gray.shape[1])
        warnings = _hash_warnings(request, image_bytes)
        for warning_key in (
            "frame_mask_warning",
            contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING,
        ):
            warning = mask_metadata.get(warning_key)
            if warning:
                warnings.append(str(warning))

        roi_location = self._locate_roi(
            source_gray,
            excluded_source_mask=mask_preparation.roi_exclusion_mask,
            background_snapshot=mask_preparation.background_snapshot,
            background_fill_value=mask_preparation.fill_value,
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
        roi_background = _apply_background_to_roi_canvas(
            roi_gray,
            background_snapshot=mask_preparation.background_snapshot,
            source_bounds=source_bounds,
            roi_bounds=roi_bounds,
        )
        mask_metadata.update(
            _localized_background_metadata(
                base=mask_metadata,
                roi_location=roi_location,
                roi_background_metadata=roi_background.metadata,
            )
        )
        silhouette_result = self._render_silhouette(
            roi_gray=roi_gray,
            source_gray=source_gray,
            source_bounds=source_bounds,
            roi_bounds=roi_bounds,
        )

        silhouette_background_mask = _silhouette_to_background_mask(
            silhouette_result.roi_silhouette
        )
        foreground_mask = silhouette_background_mask < 0.5
        foreground_mask, foreground_background_metadata = (
            _foreground_mask_after_background_removal(
                foreground_mask,
                roi_background.removal_mask,
            )
        )
        mask_metadata.update(foreground_background_metadata)
        model_background_mask = _background_mask_from_foreground(foreground_mask)
        roi_repr = _render_inverted_vehicle_detail_on_white(
            roi_gray,
            model_background_mask,
        )
        inverted_orientation_repr = roi_repr
        raw_orientation_source_gray = _raw_orientation_source_after_background_removal(
            roi_gray,
            roi_background.removal_mask,
        )
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
            roi_source_gray=raw_orientation_source_gray,
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
            contracts.PREPROCESSING_METADATA_GEOMETRY_SCHEMA: self._config.geometry_schema,
            "geometry_dim": int(self._config.geometry_dim),
            contracts.PREPROCESSING_METADATA_INPUT_IMAGE_HASH: input_image_hash.value,
            "input_image_hash_algorithm": input_image_hash.algorithm,
            contracts.PREPROCESSING_METADATA_SOURCE_IMAGE_WIDTH_PX: source_w,
            contracts.PREPROCESSING_METADATA_SOURCE_IMAGE_HEIGHT_PX: source_h,
            contracts.PREPROCESSING_METADATA_DISTANCE_CANVAS_WIDTH_PX: int(
                self._config.distance_canvas_size[0]
            ),
            contracts.PREPROCESSING_METADATA_DISTANCE_CANVAS_HEIGHT_PX: int(
                self._config.distance_canvas_size[1]
            ),
            contracts.PREPROCESSING_METADATA_ORIENTATION_CANVAS_WIDTH_PX: int(
                self._config.orientation_canvas_size[0]
            ),
            contracts.PREPROCESSING_METADATA_ORIENTATION_CANVAS_HEIGHT_PX: int(
                self._config.orientation_canvas_size[1]
            ),
            "orientation_source_mode": self._config.orientation_source_mode,
            contracts.PREPROCESSING_METADATA_ROI_REQUEST_XYXY_PX: (
                _array_xyxy_to_tuple(request_bounds)
            ),
            contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX: (
                _array_xyxy_to_tuple(source_bounds)
            ),
            contracts.PREPROCESSING_METADATA_ROI_CANVAS_INSERT_XYXY_PX: (
                _array_xyxy_to_tuple(roi_bounds)
            ),
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_BOUNDS_XYXY_PX: (
                roi_location.roi_bounds_xyxy_px
            ),
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA: dict(roi_location.metadata),
            contracts.PREPROCESSING_METADATA_PREDICTED_ROI_CENTER_XY_PX: (
                float(center_x_px),
                float(center_y_px),
            ),
            contracts.PREPROCESSING_METADATA_SILHOUETTE_BBOX_XYXY_PX: _array_xyxy_to_tuple(
                silhouette_result.feature_bbox_xyxy_px
            ),
            contracts.PREPROCESSING_METADATA_SILHOUETTE_BBOX_INCLUSIVE_XYXY_PX: (
                silhouette_result.bbox_inclusive_xyxy_px
            ),
            contracts.PREPROCESSING_METADATA_SILHOUETTE_AREA_PX: int(silhouette_result.area_px),
            contracts.PREPROCESSING_METADATA_SILHOUETTE_FALLBACK_USED: bool(
                silhouette_result.fallback_used
            ),
            contracts.PREPROCESSING_METADATA_SILHOUETTE_PRIMARY_BREAK_REASON: (
                silhouette_result.primary_break_reason
            ),
            "silhouette_diagnostics": dict(silhouette_result.diagnostics),
            "brightness_normalization": brightness_payload,
            "distance_clipped": bool(distance_clipped),
            "orientation_context_scale": float(self._config.orientation_context_scale),
            contracts.PREPROCESSING_METADATA_ORIENTATION_SOURCE_EXTENT_XYXY_PX: (
                _array_xyxy_to_tuple(orientation_source_extent_xyxy)
            ),
            contracts.PREPROCESSING_METADATA_ORIENTATION_CROP_SOURCE_XYXY_PX: _array_xyxy_to_tuple(
                orientation_crop_source_xyxy
            ),
            contracts.PREPROCESSING_METADATA_ORIENTATION_CROP_SIZE_PX: float(
                orientation_crop_size_px
            ),
            contracts.PREPROCESSING_METADATA_RUNTIME_PARAMETER_REVISION: runtime_revision,
            contracts.PREPROCESSING_METADATA_WARNINGS: tuple(warnings),
        }
        metadata.update(mask_metadata)
        debug_paths = self._write_debug_artifacts(
            request=request,
            input_image_hash=input_image_hash,
            preprocessing_parameter_revision=runtime_revision,
            source_gray=source_gray,
            roi_location=roi_location,
            roi_crop=roi_background.preview_gray,
            distance_image=distance_image_2d,
            orientation_image=orientation_image_2d,
            metadata=metadata,
            background_snapshot=mask_preparation.background_snapshot,
            fill_value=mask_preparation.fill_value,
        )
        if debug_paths:
            metadata = dict(metadata)
            metadata[contracts.PREPROCESSING_METADATA_DEBUG_PATHS] = {
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
        background_snapshot: BackgroundSnapshot | None = None,
        background_fill_value: int = 255,
    ) -> RoiLocation:
        kwargs: dict[str, Any] = {}
        if excluded_source_mask is not None and _locator_accepts_parameter(
            self._roi_locator,
            "excluded_source_mask",
        ):
            kwargs["excluded_source_mask"] = excluded_source_mask
        if background_snapshot is not None and _locator_accepts_parameter(
            self._roi_locator,
            "background_snapshot",
        ):
            kwargs["background_snapshot"] = background_snapshot
            if _locator_accepts_parameter(self._roi_locator, "background_fill_value"):
                kwargs["background_fill_value"] = background_fill_value
        location = self._roi_locator.locate(source_gray, **kwargs)
        if not isinstance(location, RoiLocation):
            raise TypeError(
                "ROI locator must return live_inference.preprocessing.RoiLocation; "
                f"got {type(location).__name__}."
            )
        return location

    def _prepare_source_masks(
        self,
        source_gray: np.ndarray,
    ) -> _MaskPreparation:
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
        masked_source = source_gray
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
                masked_source = apply_fill_to_mask(
                    source_gray,
                    manual_mask,
                    fill_value=fill_value,
                )

        background_warning = _background_source_warning(
            background_snapshot,
            source_width_px=source_w,
            source_height_px=source_h,
        )
        if background_warning is not None:
            metadata[contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING] = (
                background_warning
            )

        manual_count = (
            int(np.count_nonzero(manual_mask)) if manual_mask is not None else 0
        )
        metadata["combined_ignore_pixel_count"] = manual_count
        metadata["combined_ignore_excluded_from_roi_locator"] = False
        if manual_mask is not None:
            excluded = _locator_accepts_parameter(
                self._roi_locator,
                "excluded_source_mask",
            )
            metadata["frame_mask_excluded_from_roi_locator"] = excluded
            metadata["combined_ignore_excluded_from_roi_locator"] = excluded

        return _MaskPreparation(
            source_gray=masked_source,
            metadata=metadata,
            roi_exclusion_mask=manual_mask,
            background_snapshot=background_snapshot if background_warning is None else None,
            fill_value=fill_value,
        )

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
        background_snapshot: BackgroundSnapshot | None,
        fill_value: int,
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
                ARTIFACT_LOCATOR_INPUT: _debug_locator_input(
                    source_gray,
                    roi_location,
                    background_snapshot=background_snapshot,
                    fill_value=fill_value,
                ),
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
            contracts.PREPROCESSING_METADATA_BACKGROUND_CAPTURED: False,
            contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_ENABLED: False,
            contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED: False,
            contracts.PREPROCESSING_METADATA_BACKGROUND_REVISION: None,
            contracts.PREPROCESSING_METADATA_BACKGROUND_THRESHOLD: None,
            contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVE_PIXEL_COUNT: int(remove_pixel_count),
            contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING: warning,
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED: False,
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_REMOVE_PIXEL_COUNT: 0,
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_FCN_APPLIED: False,
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_FCN_REMOVE_PIXEL_COUNT: 0,
        }
    return {
        contracts.PREPROCESSING_METADATA_BACKGROUND_CAPTURED: bool(snapshot.captured),
        contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_ENABLED: bool(snapshot.enabled),
        contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED: bool(applied),
        contracts.PREPROCESSING_METADATA_BACKGROUND_REVISION: (
            int(snapshot.revision) if snapshot.captured else None
        ),
        contracts.PREPROCESSING_METADATA_BACKGROUND_THRESHOLD: int(snapshot.threshold),
        contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVE_PIXEL_COUNT: int(remove_pixel_count),
        contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING: warning,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED: False,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_REMOVE_PIXEL_COUNT: 0,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_FCN_APPLIED: False,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_FCN_REMOVE_PIXEL_COUNT: 0,
    }


def _background_source_warning(
    snapshot: BackgroundSnapshot | None,
    *,
    source_width_px: int,
    source_height_px: int,
) -> str | None:
    if snapshot is None or not snapshot.captured or not snapshot.enabled:
        return None
    if snapshot.dimensions_match(source_width_px, source_height_px):
        return None
    return (
        "background removal skipped: background size "
        f"{(snapshot.width_px, snapshot.height_px)} does not match source image "
        f"size {(int(source_width_px), int(source_height_px))}."
    )


def _locator_accepts_parameter(locator: RoiLocator, parameter_name: str) -> bool:
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
        if parameter.name == parameter_name:
            return True
    return False


def _apply_background_to_roi_canvas(
    roi_gray: np.ndarray,
    *,
    background_snapshot: BackgroundSnapshot | None,
    source_bounds: np.ndarray,
    roi_bounds: np.ndarray,
) -> _RoiBackgroundRemoval:
    metadata: dict[str, Any] = {
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED: False,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_REMOVE_PIXEL_COUNT: 0,
        contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING: None,
    }
    snapshot = background_snapshot
    if snapshot is None or not snapshot.captured or not snapshot.enabled:
        return _RoiBackgroundRemoval(
            preview_gray=roi_gray,
            removal_mask=None,
            metadata=metadata,
        )
    source_h, source_w = int(snapshot.height_px), int(snapshot.width_px)
    background = snapshot.grayscale_background
    if background.shape != (source_h, source_w):
        metadata[contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING] = (
            "background removal skipped for ROI crop: background shape "
            f"{background.shape} does not match {(source_h, source_w)}."
        )
        return _RoiBackgroundRemoval(
            preview_gray=roi_gray,
            removal_mask=None,
            metadata=metadata,
        )
    background_roi = _reconstruct_roi_canvas_from_source(
        background,
        source_xyxy=np.asarray(source_bounds, dtype=np.float32),
        canvas_insert_xyxy=np.asarray(roi_bounds, dtype=np.float32),
        canvas_width=int(roi_gray.shape[1]),
        canvas_height=int(roi_gray.shape[0]),
    )
    mask = compute_background_removal_mask_from_arrays(
        roi_gray,
        background_roi,
        threshold=snapshot.threshold,
    )
    metadata[contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED] = True
    metadata[
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_REMOVE_PIXEL_COUNT
    ] = int(np.count_nonzero(mask))
    return _RoiBackgroundRemoval(
        preview_gray=apply_fill_to_mask(roi_gray, mask, fill_value=255),
        removal_mask=mask,
        metadata=metadata,
    )


def _foreground_mask_after_background_removal(
    foreground_mask: np.ndarray,
    removal_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_EXCLUDED_FROM_FOREGROUND: False,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_FOREGROUND_REMOVE_PIXEL_COUNT: 0,
    }
    if removal_mask is None:
        return foreground_mask, metadata

    removable_foreground = np.asarray(foreground_mask, dtype=bool) & np.asarray(
        removal_mask,
        dtype=bool,
    )
    remove_count = int(np.count_nonzero(removable_foreground))
    metadata[
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_FOREGROUND_REMOVE_PIXEL_COUNT
    ] = remove_count
    if remove_count <= 0:
        return foreground_mask, metadata

    adjusted = np.asarray(foreground_mask, dtype=bool) & ~np.asarray(
        removal_mask,
        dtype=bool,
    )
    if not bool(np.any(adjusted)):
        metadata[contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING] = (
            "background removal mask would empty ROI foreground; keeping silhouette "
            "foreground for distance/orientation inputs."
        )
        return foreground_mask, metadata

    metadata[
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_EXCLUDED_FROM_FOREGROUND
    ] = True
    return adjusted, metadata


def _background_mask_from_foreground(foreground_mask: np.ndarray) -> np.ndarray:
    foreground = np.asarray(foreground_mask, dtype=bool)
    background_mask = np.ones(foreground.shape, dtype=np.float32)
    background_mask[foreground] = 0.0
    return background_mask


def _raw_orientation_source_after_background_removal(
    roi_gray: np.ndarray,
    removal_mask: np.ndarray | None,
) -> np.ndarray:
    if removal_mask is None:
        return roi_gray
    return apply_fill_to_mask(roi_gray, removal_mask, fill_value=255)


def _localized_background_metadata(
    *,
    base: Mapping[str, Any],
    roi_location: RoiLocation,
    roi_background_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    locator_metadata = dict(roi_location.metadata)
    roi_fcn_applied = bool(
        locator_metadata.get(
            contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_REMOVAL_APPLIED
        )
    )
    roi_fcn_count = int(
        locator_metadata.get(
            contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_REMOVE_PIXEL_COUNT
        )
        or 0
    )
    roi_crop_applied = bool(
        roi_background_metadata.get(
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED
        )
    )
    roi_crop_count = int(
        roi_background_metadata.get(
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_REMOVE_PIXEL_COUNT
        )
        or 0
    )
    warning = (
        roi_background_metadata.get(contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING)
        or locator_metadata.get(
            contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_WARNING
        )
        or base.get(contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING)
    )
    return {
        contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED: (
            roi_fcn_applied or roi_crop_applied
        ),
        contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVE_PIXEL_COUNT: roi_crop_count,
        contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING: warning,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED: roi_crop_applied,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_REMOVE_PIXEL_COUNT: (
            roi_crop_count
        ),
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_FCN_APPLIED: roi_fcn_applied,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_FCN_REMOVE_PIXEL_COUNT: (
            roi_fcn_count
        ),
        contracts.PREPROCESSING_METADATA_BACKGROUND_APPLICATION_SPACE: (
            contracts.BACKGROUND_APPLICATION_SPACE_ROI_FCN_INPUT_AND_ROI_CROP
            if bool(base.get(contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_ENABLED))
            else None
        ),
    }


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
    *,
    background_snapshot: BackgroundSnapshot | None,
    fill_value: int,
) -> np.ndarray | None:
    metadata = roi_location.metadata
    canvas_width = _optional_positive_int(
        metadata.get(contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_WIDTH_PX)
    )
    canvas_height = _optional_positive_int(
        metadata.get(contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_HEIGHT_PX)
    )
    if canvas_width is None or canvas_height is None:
        return None
    locator_input = build_roi_fcn_locator_input(
        source_gray,
        canvas_width_px=canvas_width,
        canvas_height_px=canvas_height,
    )
    snapshot = background_snapshot
    if snapshot is None or not snapshot.captured or not snapshot.enabled:
        return locator_input.locator_image
    source_w, source_h = (int(value) for value in locator_input.source_image_wh_px.tolist())
    if not snapshot.dimensions_match(source_w, source_h):
        return locator_input.locator_image
    background_input = build_roi_fcn_locator_input(
        snapshot.grayscale_background,
        canvas_width_px=canvas_width,
        canvas_height_px=canvas_height,
    )
    current_u8 = _locator_input_uint8(locator_input.locator_image)
    background_u8 = _locator_input_uint8(background_input.locator_image)
    mask = compute_background_removal_mask_from_arrays(
        current_u8,
        background_u8,
        threshold=snapshot.threshold,
    )
    debug_input = np.array(locator_input.locator_image, dtype=np.float32, copy=True)
    debug_input[0][mask] = float(_coerce_fill_value(fill_value)) / 255.0
    return debug_input


def _locator_input_uint8(locator_image: np.ndarray) -> np.ndarray:
    image = np.asarray(locator_image, dtype=np.float32)
    if image.ndim == 3 and int(image.shape[0]) == 1:
        image = image[0]
    return np.ascontiguousarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8))


def _coerce_fill_value(fill_value: int) -> int:
    value = int(fill_value)
    if value not in {0, 255}:
        raise ValueError(f"fill value must be 0 or 255; got {fill_value!r}.")
    return value


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
