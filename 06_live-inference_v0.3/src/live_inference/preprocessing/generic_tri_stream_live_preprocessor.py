"""v0.3 generic-locator tri-stream live preprocessor."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

import interfaces.contracts as contracts
from interfaces.contracts import InferenceRequest, PreparedInferenceInputs
from live_inference.masking import (
    BackgroundSnapshot,
    BackgroundState,
    FrameMaskSnapshot,
    FrameMaskState,
    apply_fill_to_mask,
)
from live_inference.model_registry.model_manifest import (
    ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE,
    ORIENTATION_SOURCE_RAW_GRAYSCALE_ON_WHITE,
    LiveModelManifest,
)

from .debug_artifacts import (
    ARTIFACT_ACCEPTED_RAW_FRAME,
    ARTIFACT_BACKGROUND_REMOVAL_MASK,
    ARTIFACT_BACKGROUND_SNAPSHOT,
    ARTIFACT_DISTANCE_IMAGE,
    ARTIFACT_FOREGROUND_MASK,
    ARTIFACT_FOREGROUND_MASK_BEFORE_COMPONENT_CLEANUP,
    ARTIFACT_GRAYSCALE_FRAME,
    ARTIFACT_MANUAL_MASK,
    ARTIFACT_ORIENTATION_IMAGE,
    ARTIFACT_PREPROCESSOR_SOURCE_AFTER_REGRESSOR_MASKS,
    ARTIFACT_PREPROCESSOR_SOURCE_BEFORE_REGRESSOR_MASKS,
    ARTIFACT_ROI_CROP,
    ARTIFACT_ROI_OVERLAY_METADATA,
    DebugArtifactWriter,
    default_debug_output_dir,
)
from .locators import FixedCenterRoiLocator, LocatorRuntimeParameterState
from .foreground_policy import (
    ForegroundExtractionPolicySnapshot,
    ForegroundExtractionPolicyState,
)
from .stage_policy import (
    StageTransformPolicySnapshot,
    StageTransformPolicyState,
)
from .camera_intrinsics import (
    CameraIntrinsicsFrameTransformer,
    CameraIntrinsicsTransformState,
)
from .preprocessing_config import TriStreamPreprocessingConfig
from .tri_stream_live_preprocessor import (
    PreprocessingDebugError,
    RoiRejectedError,
    _accepted_input_image_hash,
    _apply_background_to_roi_canvas,
    _array_xyxy_to_tuple,
    _background_mask_from_foreground,
    _bbox_features_from_xyxy,
    _brightness_result_payload,
    _contour_break_reason,
    _decode_image_bytes_to_grayscale,
    _disabled_brightness_payload,
    _extract_centered_canvas,
    _foreground_enhancement_payload,
    _foreground_mask_after_background_removal,
    _hash_warnings,
    _mask_geometry,
    _raw_orientation_source_after_background_removal,
    _render_is_empty,
    _select_silhouette_components,
)


from rb_pipeline_v4.brightness_normalization import apply_brightness_normalization_v4
from rb_pipeline_v4.foreground_enhancement import apply_foreground_enhancement_v4
from rb_pipeline_v4.pack_dual_stream_stage import (
    _place_image_on_canvas,
    _render_vehicle_detail_on_white,
)
from rb_pipeline_v4.pack_tri_stream_stage import (
    _render_orientation_image_scaled_by_foreground_extent,
)


@dataclass(frozen=True)
class _ForegroundExtractionResult:
    roi_silhouette: np.ndarray
    full_silhouette: np.ndarray
    roi_foreground_mask: np.ndarray
    full_foreground_mask: np.ndarray
    area_px: int
    bbox_inclusive_xyxy_px: tuple[int, int, int, int]
    feature_bbox_xyxy_px: np.ndarray
    extraction_mode: str
    fallback_used: bool
    primary_break_reason: str
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class LocatorDiagnosticResult:
    """Output from exact-frame locator-only diagnostics."""

    request_id: str
    input_image_hash: contracts.FrameHash
    locator_input_image: np.ndarray
    preprocessing_metadata: Mapping[str, Any]
    locator_result: contracts.LocatorResult | None = None
    debug_paths: Mapping[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class _FrameMaskPreparation:
    original_source_gray: np.ndarray
    regressor_source_gray: np.ndarray
    locator_ignore_mask: np.ndarray | None
    manual_mask: np.ndarray | None
    metadata: Mapping[str, Any]
    warnings: tuple[str, ...]
    fill_value: int


@dataclass(frozen=True)
class _PreparedSourceFrame:
    source_gray: np.ndarray
    locator_image_bytes: bytes
    metadata: Mapping[str, Any]


class TriStreamLivePreprocessor:
    """Prepare live raw image bytes as v0.3 generic-locator tri-stream inputs."""

    def __init__(
        self,
        *,
        locator: contracts.RoiLocator | None = None,
        roi_locator: contracts.RoiLocator | None = None,
        model_manifest: LiveModelManifest | None = None,
        config: TriStreamPreprocessingConfig | None = None,
        runtime_parameter_revision_getter: Callable[[], int | None] | None = None,
        background_state: BackgroundState | None = None,
        mask_state: FrameMaskState | None = None,
        locator_parameter_state: LocatorRuntimeParameterState | None = None,
        foreground_extraction_policy_state: ForegroundExtractionPolicyState | None = None,
        stage_policy_state: StageTransformPolicyState | None = None,
        stage_policy: StageTransformPolicySnapshot | None = None,
        camera_intrinsics_state: CameraIntrinsicsTransformState | None = None,
        camera_intrinsics_transformer: CameraIntrinsicsFrameTransformer | None = None,
        **_legacy_kwargs: Any,
    ) -> None:
        if config is None:
            if model_manifest is None:
                raise ValueError(
                    "TriStreamLivePreprocessor requires a model_manifest or explicit config."
                )
            config = TriStreamPreprocessingConfig.from_manifest(model_manifest)
        config.validate()
        self._config = config
        self._locator = locator or roi_locator or FixedCenterRoiLocator(
            roi_wh_px=self._roi_canvas_size()
        )
        self._runtime_parameter_revision_getter = runtime_parameter_revision_getter
        self._background_state = background_state
        self._mask_state = mask_state
        self._locator_parameter_state = locator_parameter_state
        self._foreground_extraction_policy_state = (
            foreground_extraction_policy_state or ForegroundExtractionPolicyState()
        )
        self._stage_policy_state = stage_policy_state or StageTransformPolicyState(
            stage_policy
        )
        self._camera_intrinsics_transformer = (
            camera_intrinsics_transformer
            if camera_intrinsics_transformer is not None
            else (
                CameraIntrinsicsFrameTransformer(camera_intrinsics_state)
                if camera_intrinsics_state is not None
                else None
            )
        )

    @property
    def config(self) -> TriStreamPreprocessingConfig:
        return self._config

    @property
    def locator(self) -> contracts.RoiLocator:
        return self._locator

    @property
    def foreground_extraction_policy_state(self) -> ForegroundExtractionPolicyState:
        return self._foreground_extraction_policy_state

    @property
    def stage_policy_state(self) -> StageTransformPolicyState:
        return self._stage_policy_state

    def preview_locator_input(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
    ) -> LocatorDiagnosticResult:
        """Compatibility alias: v0.3 locator diagnostics are locator-only runs."""
        return self.run_locator_only(request, image_bytes)

    def run_roi_locator_only(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
    ) -> LocatorDiagnosticResult:
        """Compatibility alias for retained single-frame runner code."""
        return self.run_locator_only(request, image_bytes)

    def run_locator_only(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
    ) -> LocatorDiagnosticResult:
        """Run only the configured locator and ROI guard for an exact frame."""
        prepared_source = self._prepare_source_frame(image_bytes)
        source_gray = prepared_source.source_gray
        source_h, source_w = int(source_gray.shape[0]), int(source_gray.shape[1])
        input_image_hash = _accepted_input_image_hash(request, image_bytes)
        runtime_revision = self._runtime_parameter_revision()
        stage_policy = self._stage_policy_state.get_snapshot()
        background_snapshot, background_warning = self._background_snapshot_and_warning(
            source_w,
            source_h,
        )
        mask_preparation = self._prepare_frame_mask(
            source_gray,
            apply_to_locator=bool(stage_policy.apply_manual_mask_to_roi_locator),
            apply_to_regressor=False,
        )
        warnings = _hash_warnings(request, image_bytes)
        warnings.extend(mask_preparation.warnings)
        if background_warning and bool(stage_policy.apply_background_removal_to_roi_locator):
            warnings.append(background_warning)
        locator_result = self._locate(
            request,
            prepared_source.locator_image_bytes,
            source_wh=(source_w, source_h),
            runtime_revision=runtime_revision,
            mask_preparation=mask_preparation,
            apply_background_removal_to_locator=bool(
                stage_policy.apply_background_removal_to_roi_locator
            ),
        )
        metadata = self._base_metadata(
            request=request,
            input_image_hash=input_image_hash,
            runtime_revision=runtime_revision,
            source_gray=source_gray,
            locator_result=locator_result,
            warnings=warnings,
            regressor_reached=False,
        )
        metadata.update(mask_preparation.metadata)
        metadata.update(stage_policy.to_metadata())
        metadata.update(
            self._background_metadata(
                snapshot=background_snapshot,
                source_wh=(source_w, source_h),
                warning=background_warning
                if bool(stage_policy.apply_background_removal_to_roi_locator)
                else None,
            )
        )
        metadata.update(
            _background_application_metadata(
                stage_policy=stage_policy,
                locator_result=locator_result,
                roi_background_metadata={},
            )
        )
        metadata.update(prepared_source.metadata)
        roi_crop = None
        if locator_result.center_xy_px is not None:
            roi_crop, _source_bounds, _roi_bounds, _request_bounds = _extract_centered_canvas(
                source_gray,
                center_x_px=locator_result.center_xy_px[0],
                center_y_px=locator_result.center_xy_px[1],
                canvas_width_px=self._roi_canvas_size()[0],
                canvas_height_px=self._roi_canvas_size()[1],
            )
        debug_paths = self._write_debug_artifacts(
            request=request,
            input_image_hash=input_image_hash,
            runtime_revision=runtime_revision,
            source_gray=source_gray,
            preprocessor_source_gray=mask_preparation.regressor_source_gray,
            manual_mask=mask_preparation.manual_mask,
            roi_crop=roi_crop,
            foreground_mask_before_component_cleanup=None,
            foreground_mask=None,
            distance_image=None,
            orientation_image=None,
            metadata=metadata,
            locator_result=locator_result,
            background_snapshot=background_snapshot,
            background_removal_mask=None,
        )
        if debug_paths:
            metadata = {**metadata, contracts.PREPROCESSING_METADATA_DEBUG_PATHS: _path_map(debug_paths)}
        display = _locator_display_image(source_gray, locator_result)
        return LocatorDiagnosticResult(
            request_id=request.request_id,
            input_image_hash=input_image_hash,
            locator_input_image=display,
            preprocessing_metadata=metadata,
            locator_result=locator_result,
            debug_paths=debug_paths,
        )

    def prepare_model_inputs(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
    ) -> PreparedInferenceInputs:
        """Decode raw bytes and reproduce the selected v4 tri-stream contract."""
        start = perf_counter()
        prepared_source = self._prepare_source_frame(image_bytes)
        source_gray = prepared_source.source_gray
        source_h, source_w = int(source_gray.shape[0]), int(source_gray.shape[1])
        input_image_hash = _accepted_input_image_hash(request, image_bytes)
        runtime_revision = self._runtime_parameter_revision()
        stage_policy = self._stage_policy_state.get_snapshot()
        background_snapshot, background_warning = self._background_snapshot_and_warning(
            source_w,
            source_h,
        )
        warnings = _hash_warnings(request, image_bytes)
        mask_preparation = self._prepare_frame_mask(
            source_gray,
            apply_to_locator=bool(stage_policy.apply_manual_mask_to_roi_locator),
            apply_to_regressor=bool(
                stage_policy.apply_manual_mask_to_regressor_preprocessing
            ),
        )
        warnings.extend(mask_preparation.warnings)
        if background_warning and (
            bool(stage_policy.apply_background_removal_to_roi_locator)
            or bool(stage_policy.apply_background_removal_to_regressor_preprocessing)
        ):
            warnings.append(background_warning)
        locator_result = self._locate(
            request,
            prepared_source.locator_image_bytes,
            source_wh=(source_w, source_h),
            runtime_revision=runtime_revision,
            mask_preparation=mask_preparation,
            apply_background_removal_to_locator=bool(
                stage_policy.apply_background_removal_to_roi_locator
            ),
        )
        warnings.extend(str(warning) for warning in locator_result.warnings)

        if not locator_result.accepted or locator_result.center_xy_px is None:
            metadata = self._base_metadata(
                request=request,
                input_image_hash=input_image_hash,
                runtime_revision=runtime_revision,
                source_gray=source_gray,
                locator_result=locator_result,
                warnings=warnings,
                regressor_reached=False,
            )
            metadata.update(mask_preparation.metadata)
            metadata.update(stage_policy.to_metadata())
            metadata.update(
                self._background_metadata(
                    snapshot=background_snapshot,
                    source_wh=(source_w, source_h),
                    warning=background_warning
                    if (
                        bool(stage_policy.apply_background_removal_to_roi_locator)
                        or bool(
                            stage_policy.apply_background_removal_to_regressor_preprocessing
                        )
                    )
                    else None,
                )
            )
            metadata.update(
                _background_application_metadata(
                    stage_policy=stage_policy,
                    locator_result=locator_result,
                    roi_background_metadata={},
                )
            )
            metadata.update(prepared_source.metadata)
            debug_paths = self._write_debug_artifacts(
                request=request,
                input_image_hash=input_image_hash,
                runtime_revision=runtime_revision,
                source_gray=source_gray,
                preprocessor_source_gray=mask_preparation.regressor_source_gray,
                manual_mask=mask_preparation.manual_mask,
                roi_crop=None,
                foreground_mask_before_component_cleanup=None,
                foreground_mask=None,
                distance_image=None,
                orientation_image=None,
                metadata=metadata,
                locator_result=locator_result,
                background_snapshot=background_snapshot,
                background_removal_mask=None,
            )
            if debug_paths:
                metadata = {**metadata, contracts.PREPROCESSING_METADATA_DEBUG_PATHS: _path_map(debug_paths)}
            details = _failure_details(
                request=request,
                input_image_hash=input_image_hash,
                metadata=metadata,
            )
            raise RoiRejectedError(
                "ROI rejected during preprocessing: "
                + (_reason_text(locator_result.roi_rejection_reasons) or "locator rejected frame"),
                details=details,
                preprocessing_metadata=metadata,
                debug_paths=debug_paths,
            )

        center_x, center_y = locator_result.center_xy_px
        roi_gray, source_bounds, roi_bounds, request_bounds = _extract_centered_canvas(
            mask_preparation.regressor_source_gray,
            center_x_px=center_x,
            center_y_px=center_y,
            canvas_width_px=self._roi_canvas_size()[0],
            canvas_height_px=self._roi_canvas_size()[1],
        )
        regressor_background_snapshot = _background_for_stage(
            background_snapshot,
            warning=background_warning,
            apply_to_stage=bool(
                stage_policy.apply_background_removal_to_regressor_preprocessing
            ),
        )
        roi_background = _apply_background_to_roi_canvas(
            roi_gray,
            background_snapshot=regressor_background_snapshot,
            source_bounds=source_bounds,
            roi_bounds=roi_bounds,
        )
        if roi_background.metadata.get(contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING):
            warnings.append(str(roi_background.metadata[contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING]))

        foreground_mask: np.ndarray | None = None
        foreground_mask_before_component_cleanup: np.ndarray | None = None
        distance_image_2d: np.ndarray | None = None
        orientation_image_2d: np.ndarray | None = None
        foreground_extraction_result: _ForegroundExtractionResult | None = None
        final_foreground_area_px: int | None = None
        final_foreground_bbox_inclusive_xyxy_px: tuple[int, int, int, int] | None = None
        final_feature_bbox_xyxy_px: np.ndarray | None = None
        foreground_extraction_policy = (
            self._foreground_extraction_policy_state.snapshot()
        )
        mask_metadata = dict(mask_preparation.metadata)
        mask_metadata.update(stage_policy.to_metadata())
        mask_metadata.update(foreground_extraction_policy.to_metadata())
        mask_metadata.update(
            self._background_metadata(
                snapshot=background_snapshot,
                source_wh=(source_w, source_h),
                warning=background_warning
                if (
                    bool(stage_policy.apply_background_removal_to_roi_locator)
                    or bool(stage_policy.apply_background_removal_to_regressor_preprocessing)
                )
                else None,
            )
        )
        mask_metadata.update(roi_background.metadata)
        mask_metadata.update(
            _background_application_metadata(
                stage_policy=stage_policy,
                locator_result=locator_result,
                roi_background_metadata=roi_background.metadata,
            )
        )
        try:
            foreground_extraction_result = self._extract_foreground(
                roi_gray=roi_background.preview_gray,
                source_gray=mask_preparation.regressor_source_gray,
                source_bounds=source_bounds,
                roi_bounds=roi_bounds,
                policy=foreground_extraction_policy,
            )
            foreground_mask = foreground_extraction_result.roi_foreground_mask.astype(
                bool,
                copy=True,
            )
            consistency_metadata, _consistency_error = (
                _foreground_locator_consistency_check(
                    foreground_result=foreground_extraction_result,
                    locator_result=locator_result,
                )
            )
            mask_metadata.update(consistency_metadata)
            foreground_mask, foreground_background_metadata = (
                _foreground_mask_after_background_removal(
                    foreground_mask,
                    roi_background.removal_mask,
                )
            )
            mask_metadata.update(foreground_background_metadata)
            foreground_mask_before_component_cleanup = np.array(
                foreground_mask,
                dtype=bool,
                copy=True,
            )
            foreground_mask, component_cleanup_metadata = (
                _foreground_mask_component_cleanup(
                    foreground_mask,
                    roi_gray=roi_background.preview_gray,
                )
            )
            mask_metadata.update(component_cleanup_metadata)
            (
                _final_full_foreground_mask,
                final_foreground_area_px,
                final_foreground_bbox_inclusive_xyxy_px,
                final_feature_bbox_xyxy_px,
            ) = _foreground_geometry_from_roi_mask(
                foreground_mask,
                source_gray=mask_preparation.regressor_source_gray,
                source_bounds=source_bounds,
                roi_bounds=roi_bounds,
            )
            model_background_mask = _background_mask_from_foreground(foreground_mask)
            roi_repr = _render_vehicle_detail_on_white(
                roi_background.preview_gray,
                model_background_mask,
                image_representation_mode=self._config.image_representation_mode,
            )
            foreground_enhancement_result = None
            if self._config.foreground_runtime.active():
                foreground_enhancement_result = apply_foreground_enhancement_v4(
                    roi_repr,
                    foreground_mask.astype(bool, copy=False),
                    self._config.foreground_runtime.config,
                )
                roi_repr = foreground_enhancement_result.image
            orientation_repr = roi_repr
            raw_orientation_source_gray = _raw_orientation_source_after_background_removal(
                roi_gray,
                roi_background.removal_mask,
            )
            distance_image_2d, brightness_payload, distance_clipped = (
                self._build_distance_image(
                    roi_repr=roi_repr,
                    foreground_mask=foreground_mask,
                )
            )
            (
                orientation_image_2d,
                orientation_source_extent_xyxy,
                orientation_crop_source_xyxy,
                orientation_crop_size_px,
            ) = self._build_orientation_image(
                roi_source_gray=raw_orientation_source_gray,
                representation_source=orientation_repr,
                foreground_mask=foreground_mask,
            )
            geometry = _bbox_features_from_xyxy(
                final_feature_bbox_xyxy_px,
                image_width_px=source_w,
                image_height_px=source_h,
            )
        except Exception as exc:
            metadata = self._base_metadata(
                request=request,
                input_image_hash=input_image_hash,
                runtime_revision=runtime_revision,
                source_gray=source_gray,
                locator_result=locator_result,
                warnings=warnings,
                regressor_reached=False,
            )
            metadata.update(mask_metadata)
            metadata.update(prepared_source.metadata)
            metadata.update(
                {
                    "preprocessing_failure_type": type(exc).__name__,
                    "preprocessing_failure_message": str(exc),
                    "roi_crop_available": True,
                    "foreground_mask_empty": (
                        None if foreground_mask is None else not bool(np.any(foreground_mask))
                    ),
                    "foreground_pixel_count": (
                        None if foreground_mask is None else int(np.count_nonzero(foreground_mask))
                    ),
                    "silhouette_diagnostics": (
                        None if foreground_extraction_result is None else dict(foreground_extraction_result.diagnostics)
                    ),
                    "foreground_extraction_diagnostics": (
                        None
                        if foreground_extraction_result is None
                        else dict(foreground_extraction_result.diagnostics)
                    ),
                }
            )
            debug_paths = self._write_debug_artifacts(
                request=request,
                input_image_hash=input_image_hash,
                runtime_revision=runtime_revision,
                source_gray=source_gray,
                preprocessor_source_gray=mask_preparation.regressor_source_gray,
                manual_mask=mask_preparation.manual_mask,
                roi_crop=roi_background.preview_gray,
                foreground_mask_before_component_cleanup=(
                    foreground_mask_before_component_cleanup
                ),
                foreground_mask=foreground_mask,
                distance_image=distance_image_2d,
                orientation_image=orientation_image_2d,
                metadata=metadata,
                locator_result=locator_result,
                background_snapshot=background_snapshot,
                background_removal_mask=roi_background.removal_mask,
            )
            if debug_paths:
                metadata = {**metadata, contracts.PREPROCESSING_METADATA_DEBUG_PATHS: _path_map(debug_paths)}
            raise PreprocessingDebugError(
                f"Preprocessing failed after locator result: {exc}",
                details=_failure_details(
                    request=request,
                    input_image_hash=input_image_hash,
                    metadata=metadata,
                ),
                preprocessing_metadata=metadata,
                debug_paths=debug_paths,
            ) from exc

        metadata = self._base_metadata(
            request=request,
            input_image_hash=input_image_hash,
            runtime_revision=runtime_revision,
            source_gray=source_gray,
            locator_result=locator_result,
            warnings=warnings,
            regressor_reached=True,
        )
        metadata.update(mask_metadata)
        metadata.update(prepared_source.metadata)
        metadata.update(
            {
                contracts.PREPROCESSING_METADATA_FOREGROUND_EXTRACTION_MODE: (
                    foreground_extraction_result.extraction_mode
                ),
                contracts.PREPROCESSING_METADATA_FOREGROUND_EXTRACTION_REVISION: (
                    int(foreground_extraction_policy.revision)
                ),
                contracts.PREPROCESSING_METADATA_FOREGROUND_BBOX_XYXY_PX: (
                    _array_xyxy_to_tuple(final_feature_bbox_xyxy_px)
                ),
                contracts.PREPROCESSING_METADATA_FOREGROUND_BBOX_INCLUSIVE_XYXY_PX: (
                    final_foreground_bbox_inclusive_xyxy_px
                ),
                contracts.PREPROCESSING_METADATA_FOREGROUND_AREA_PX: int(
                    final_foreground_area_px
                ),
                contracts.PREPROCESSING_METADATA_SILHOUETTE_BBOX_XYXY_PX: (
                    _array_xyxy_to_tuple(final_feature_bbox_xyxy_px)
                ),
                contracts.PREPROCESSING_METADATA_SILHOUETTE_BBOX_INCLUSIVE_XYXY_PX: (
                    final_foreground_bbox_inclusive_xyxy_px
                ),
                contracts.PREPROCESSING_METADATA_SILHOUETTE_AREA_PX: int(
                    final_foreground_area_px
                ),
                contracts.PREPROCESSING_METADATA_SILHOUETTE_FALLBACK_USED: bool(
                    foreground_extraction_result.fallback_used
                ),
                contracts.PREPROCESSING_METADATA_SILHOUETTE_PRIMARY_BREAK_REASON: (
                    foreground_extraction_result.primary_break_reason
                ),
                "silhouette_diagnostics": dict(foreground_extraction_result.diagnostics),
                "foreground_extraction_diagnostics": dict(
                    foreground_extraction_result.diagnostics
                ),
                "foreground_mask_empty": not bool(np.any(foreground_mask)),
                "foreground_pixel_count": int(np.count_nonzero(foreground_mask)),
                "brightness_normalization": brightness_payload,
                "foreground_enhancement": _foreground_enhancement_payload(
                    self._config.foreground_runtime,
                    foreground_enhancement_result,
                    foreground_mask,
                ),
                "distance_clipped": bool(distance_clipped),
                "orientation_context_scale": float(self._config.orientation_context_scale),
                contracts.PREPROCESSING_METADATA_ORIENTATION_SOURCE_EXTENT_XYXY_PX: (
                    _array_xyxy_to_tuple(orientation_source_extent_xyxy)
                ),
                contracts.PREPROCESSING_METADATA_ORIENTATION_CROP_SOURCE_XYXY_PX: (
                    _array_xyxy_to_tuple(orientation_crop_source_xyxy)
                ),
                contracts.PREPROCESSING_METADATA_ORIENTATION_CROP_SIZE_PX: float(
                    orientation_crop_size_px
                ),
                "preprocessing_time_ms": (perf_counter() - start) * 1000.0,
            }
        )
        debug_paths = self._write_debug_artifacts(
            request=request,
            input_image_hash=input_image_hash,
            runtime_revision=runtime_revision,
            source_gray=source_gray,
            preprocessor_source_gray=mask_preparation.regressor_source_gray,
            manual_mask=mask_preparation.manual_mask,
            roi_crop=roi_background.preview_gray,
            foreground_mask_before_component_cleanup=(
                foreground_mask_before_component_cleanup
            ),
            foreground_mask=foreground_mask,
            distance_image=distance_image_2d,
            orientation_image=orientation_image_2d,
            metadata=metadata,
            locator_result=locator_result,
            background_snapshot=background_snapshot,
            background_removal_mask=roi_background.removal_mask,
        )
        if debug_paths:
            metadata = {**metadata, contracts.PREPROCESSING_METADATA_DEBUG_PATHS: _path_map(debug_paths)}

        return PreparedInferenceInputs(
            request_id=request.request_id,
            input_mode=contracts.InferenceInputMode.TRI_STREAM_V0_4,
            input_keys=contracts.TRI_STREAM_INPUT_KEYS,
            model_inputs={
                contracts.TRI_STREAM_DISTANCE_IMAGE_KEY: distance_image_2d[None, ...].astype(
                    np.float32,
                    copy=False,
                ),
                contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY: orientation_image_2d[None, ...].astype(
                    np.float32,
                    copy=False,
                ),
                contracts.TRI_STREAM_GEOMETRY_KEY: geometry.astype(np.float32, copy=False),
            },
            source_frame=request.frame,
            preprocessing_metadata=metadata,
        )

    def _locate(
        self,
        request: InferenceRequest,
        image_bytes: bytes,
        *,
        source_wh: tuple[int, int],
        runtime_revision: int | None,
        mask_preparation: _FrameMaskPreparation | None = None,
        apply_background_removal_to_locator: bool = False,
    ) -> contracts.LocatorResult:
        snapshot = self._background_state.get_snapshot() if self._background_state is not None else None
        locator_kind = getattr(self._locator, "locator_kind", contracts.LocatorKind.BACKGROUND_EDGE_V1)
        extras: dict[str, Any] = {
            **dict(request.extras),
            "locator_parameters": {
                "roi_width_px": self._roi_canvas_size()[0],
                "roi_height_px": self._roi_canvas_size()[1],
            },
            contracts.PREPROCESSING_METADATA_APPLY_BACKGROUND_REMOVAL_TO_ROI_LOCATOR: bool(
                apply_background_removal_to_locator
            ),
        }
        if mask_preparation is not None and mask_preparation.locator_ignore_mask is not None:
            extras.update(
                {
                    "manual_ignore_mask": mask_preparation.locator_ignore_mask,
                    "manual_ignore_mask_revision": mask_preparation.metadata.get(
                        "frame_mask_revision"
                    ),
                    "manual_ignore_mask_pixel_count": mask_preparation.metadata.get(
                        "frame_mask_pixel_count"
                    ),
                    "manual_ignore_mask_fill_value": mask_preparation.fill_value,
                }
            )
        locator_request = contracts.LocatorRequest(
            request_id=request.request_id,
            frame=request.frame,
            requested_at_utc=request.requested_at_utc,
            locator_kind=locator_kind,
            source_image_wh_px=source_wh,
            background_revision=(
                int(snapshot.revision) if snapshot is not None and snapshot.captured else None
            ),
            runtime_parameter_revision=runtime_revision,
            save_debug_images=bool(request.save_debug_images),
            debug_output_dir=request.debug_output_dir,
            extras=extras,
        )
        result = self._locator.locate(locator_request, image_bytes)
        if not isinstance(result, contracts.LocatorResult):
            raise TypeError(
                "ROI locator must return interfaces.contracts.LocatorResult; "
                f"got {type(result).__name__}."
            )
        if tuple(result.source_image_wh_px) != tuple(source_wh):
            raise ValueError(
                "Locator result source size mismatch: "
                f"result={result.source_image_wh_px}, decoded={source_wh}."
            )
        return result

    def _prepare_source_frame(self, image_bytes: bytes) -> _PreparedSourceFrame:
        transformer = self._camera_intrinsics_transformer
        if transformer is None:
            return _PreparedSourceFrame(
                source_gray=_decode_image_bytes_to_grayscale(image_bytes),
                locator_image_bytes=image_bytes,
                metadata={},
            )
        result = transformer.transform_image_bytes(image_bytes, grayscale=True)
        return _PreparedSourceFrame(
            source_gray=np.asarray(result.image, dtype=np.uint8),
            locator_image_bytes=result.image_bytes,
            metadata=dict(result.metadata),
        )

    def _base_metadata(
        self,
        *,
        request: InferenceRequest,
        input_image_hash: contracts.FrameHash,
        runtime_revision: int | None,
        source_gray: np.ndarray,
        locator_result: contracts.LocatorResult,
        warnings: list[str],
        regressor_reached: bool,
    ) -> dict[str, Any]:
        source_h, source_w = int(source_gray.shape[0]), int(source_gray.shape[1])
        clip_amount = dict(locator_result.roi_clip_amount_px)
        clip_max = max((int(value) for value in clip_amount.values()), default=0)
        accepted = bool(locator_result.accepted)
        reason = _reason_text(locator_result.roi_rejection_reasons)
        locator_metadata = {
            "locator_result": locator_result.to_dict(),
            "locator_debug_artifacts": locator_result.debug_artifacts.to_dict(),
            **dict(locator_result.extras),
        }
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
            contracts.PREPROCESSING_METADATA_SOURCE_IMAGE_WH_PX: (source_w, source_h),
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
            contracts.PREPROCESSING_METADATA_LOCATOR_KIND: locator_result.locator_kind.value,
            contracts.PREPROCESSING_METADATA_LOCATOR_METADATA: locator_metadata,
            contracts.PREPROCESSING_METADATA_LOCATOR_RESULT_ACCEPTED: accepted,
            contracts.PREPROCESSING_METADATA_LOCATOR_CONFIDENCE: locator_result.confidence,
            contracts.PREPROCESSING_METADATA_LOCATOR_CANDIDATE_COUNT: len(locator_result.candidates),
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA: locator_metadata,
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_CONFIDENCE: locator_result.confidence,
            contracts.PREPROCESSING_METADATA_ROI_CONFIDENCE: locator_result.confidence,
            contracts.PREPROCESSING_METADATA_PREDICTED_ROI_CENTER_XY_PX: (
                locator_result.center_xy_px
            ),
            contracts.PREPROCESSING_METADATA_ROI_CENTER_XY_PX: locator_result.center_xy_px,
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_CENTER_SOURCE_XY_PX: (
                locator_result.center_xy_px
            ),
            contracts.PREPROCESSING_METADATA_ROI_LOCATOR_BOUNDS_XYXY_PX: (
                locator_result.bbox_xyxy_px
            ),
            contracts.PREPROCESSING_METADATA_ROI_REQUEST_XYXY_PX: (
                locator_result.roi_requested_xyxy_px
            ),
            contracts.PREPROCESSING_METADATA_ROI_REQUESTED_XYXY_PX: (
                locator_result.roi_requested_xyxy_px
            ),
            "roi_pre_clip_bounds_xyxy_px": locator_result.roi_requested_xyxy_px,
            contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX: (
                locator_result.roi_source_xyxy_px
            ),
            "roi_clipped_bounds_xyxy_px": locator_result.roi_source_xyxy_px,
            contracts.PREPROCESSING_METADATA_ROI_CANVAS_INSERT_XYXY_PX: (
                locator_result.roi_canvas_insert_xyxy_px
            ),
            contracts.PREPROCESSING_METADATA_ROI_CLIPPED: bool(locator_result.roi_clipped),
            contracts.PREPROCESSING_METADATA_ROI_CLIP_LEFT_PX: int(clip_amount.get("left", 0)),
            contracts.PREPROCESSING_METADATA_ROI_CLIP_RIGHT_PX: int(clip_amount.get("right", 0)),
            contracts.PREPROCESSING_METADATA_ROI_CLIP_TOP_PX: int(clip_amount.get("top", 0)),
            contracts.PREPROCESSING_METADATA_ROI_CLIP_BOTTOM_PX: int(clip_amount.get("bottom", 0)),
            contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX: int(clip_max),
            contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERANCE_PX: (
                _locator_parameter(locator_result, "roi_clip_tolerance_px")
            ),
            contracts.PREPROCESSING_METADATA_ROI_CLIP_TOLERATED: False,
            contracts.PREPROCESSING_METADATA_ROI_ACCEPTED: accepted,
            contracts.PREPROCESSING_METADATA_ROI_REJECTED: not accepted,
            contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASON: reason,
            contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASONS: (
                locator_result.roi_rejection_reasons
            ),
            contracts.PREPROCESSING_METADATA_ROI_CONTENT_FRACTION: (
                locator_result.extras.get("roi_content_fraction")
            ),
            contracts.PREPROCESSING_METADATA_RUNTIME_PARAMETER_REVISION: runtime_revision,
            contracts.PREPROCESSING_METADATA_WARNINGS: tuple(warnings),
            contracts.PREPROCESSING_METADATA_DEBUG_PATHS: _path_map(
                locator_result.debug_artifacts.paths
            ),
            "request_id": request.request_id,
            "distance_orientation_regressor_reached": bool(regressor_reached),
        }
        metadata.update(self._foreground_extraction_policy_state.snapshot().to_metadata())
        return metadata

    def _background_metadata(
        self,
        *,
        snapshot: BackgroundSnapshot | None,
        source_wh: tuple[int, int],
        warning: str | None,
    ) -> dict[str, Any]:
        captured = bool(snapshot is not None and snapshot.captured)
        enabled = bool(snapshot is not None and snapshot.enabled)
        revision = int(snapshot.revision) if captured else None
        threshold = int(snapshot.threshold) if snapshot is not None else None
        return {
            contracts.PREPROCESSING_METADATA_BACKGROUND_CAPTURED: captured,
            contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_ENABLED: enabled,
            contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED: False,
            contracts.PREPROCESSING_METADATA_BACKGROUND_REVISION: revision,
            contracts.PREPROCESSING_METADATA_BACKGROUND_THRESHOLD: threshold,
            contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVE_PIXEL_COUNT: 0,
            contracts.PREPROCESSING_METADATA_BACKGROUND_WARNING: warning,
            contracts.PREPROCESSING_METADATA_BACKGROUND_APPLICATION_SPACE: None,
            "background_source_wh_px": source_wh,
        }

    def _background_snapshot_and_warning(
        self,
        source_w: int,
        source_h: int,
    ) -> tuple[BackgroundSnapshot | None, str | None]:
        if self._background_state is None:
            return None, None
        snapshot = self._background_state.get_snapshot()
        if snapshot is None or not snapshot.captured or not snapshot.enabled:
            return snapshot, None
        if snapshot.dimensions_match(source_w, source_h):
            return snapshot, None
        return snapshot, (
            "background removal skipped: background size "
            f"{(snapshot.width_px, snapshot.height_px)} does not match source image "
            f"size {(source_w, source_h)}."
        )

    def _prepare_frame_mask(
        self,
        source_gray: np.ndarray,
        *,
        apply_to_locator: bool,
        apply_to_regressor: bool,
    ) -> _FrameMaskPreparation:
        source_h, source_w = int(source_gray.shape[0]), int(source_gray.shape[1])
        snapshot = self._mask_state.get_snapshot() if self._mask_state is not None else None
        fill_value = int(snapshot.fill_value) if snapshot is not None else 255
        metadata = _frame_mask_metadata(
            snapshot=snapshot,
            source_width_px=source_w,
            source_height_px=source_h,
            applied=False,
            fill_value=fill_value,
        )
        warnings: list[str] = []
        manual_mask: np.ndarray | None = None
        manual_mask_valid = False

        if (
            snapshot is not None
            and snapshot.enabled
            and snapshot.has_geometry
            and snapshot.pixel_count > 0
        ):
            if not snapshot.dimensions_match(source_w, source_h):
                warning = (
                    "frame mask skipped: mask size "
                    f"{(snapshot.width_px, snapshot.height_px)} does not match "
                    f"source image size {(source_w, source_h)}."
                )
                metadata["frame_mask_warning"] = warning
                warnings.append(warning)
            else:
                manual_mask = np.array(snapshot.mask, dtype=bool, copy=True)
                manual_mask_valid = True

        manual_to_locator = bool(manual_mask_valid and apply_to_locator)
        manual_to_regressor = bool(manual_mask_valid and apply_to_regressor)
        regressor_source_gray = (
            apply_fill_to_mask(source_gray, manual_mask, fill_value=fill_value)
            if manual_to_regressor
            else np.array(source_gray, dtype=np.uint8, copy=True)
        )
        manual_count = int(np.count_nonzero(manual_mask)) if manual_mask is not None else 0
        metadata.update(
            {
                "frame_mask_applied": bool(manual_to_locator or manual_to_regressor),
                "frame_mask_application_space": _mask_application_space(
                    locator=manual_to_locator,
                    regressor=manual_to_regressor,
                ),
                "manual_mask_available": bool(manual_mask_valid),
                "apply_manual_mask_to_roi_locator": bool(apply_to_locator),
                "apply_manual_mask_to_regressor_preprocessing": bool(
                    apply_to_regressor
                ),
                "manual_mask_applied_to_roi_locator": bool(manual_to_locator),
                "manual_mask_applied_to_regressor_preprocessing": bool(
                    manual_to_regressor
                ),
                "frame_mask_excluded_from_roi_locator": bool(manual_to_locator),
                "combined_ignore_excluded_from_roi_locator": bool(manual_to_locator),
                "combined_ignore_pixel_count": manual_count
                if (manual_to_locator or manual_to_regressor)
                else 0,
            }
        )
        return _FrameMaskPreparation(
            original_source_gray=np.array(source_gray, dtype=np.uint8, copy=True),
            regressor_source_gray=regressor_source_gray,
            locator_ignore_mask=(
                np.array(manual_mask, dtype=bool, copy=True)
                if manual_to_locator and manual_mask is not None
                else None
            ),
            manual_mask=manual_mask,
            metadata=metadata,
            warnings=tuple(warnings),
            fill_value=fill_value,
        )

    def _extract_foreground(
        self,
        *,
        roi_gray: np.ndarray,
        source_gray: np.ndarray,
        source_bounds: np.ndarray,
        roi_bounds: np.ndarray,
        policy: ForegroundExtractionPolicySnapshot,
    ) -> _ForegroundExtractionResult:
        mode = str(policy.foreground_extraction_mode)
        if mode == contracts.ForegroundExtractionMode.THRESHOLD_FOREGROUND_V1.value:
            return self._extract_threshold_foreground(
                roi_gray=roi_gray,
                source_gray=source_gray,
                source_bounds=source_bounds,
                roi_bounds=roi_bounds,
                policy=policy,
            )
        if mode == contracts.ForegroundExtractionMode.SILHOUETTE_CONTOUR_V2.value:
            return self._render_silhouette(
                roi_gray=roi_gray,
                source_gray=source_gray,
                source_bounds=source_bounds,
                roi_bounds=roi_bounds,
            )
        raise ValueError(f"Unsupported foreground extraction mode: {mode!r}.")

    def _extract_threshold_foreground(
        self,
        *,
        roi_gray: np.ndarray,
        source_gray: np.ndarray,
        source_bounds: np.ndarray,
        roi_bounds: np.ndarray,
        policy: ForegroundExtractionPolicySnapshot,
    ) -> _ForegroundExtractionResult:
        foreground_mask, diagnostics = _threshold_foreground_mask(
            roi_gray,
            policy=policy,
        )
        if not bool(np.any(foreground_mask)):
            raise ValueError("Threshold foreground extraction produced an empty mask")

        roi_silhouette = np.full(roi_gray.shape, 255, dtype=np.uint8)
        roi_silhouette[foreground_mask] = 0
        src_x1, src_y1, src_x2, src_y2 = [
            int(value) for value in source_bounds.tolist()
        ]
        roi_x1, roi_y1, roi_x2, roi_y2 = [int(value) for value in roi_bounds.tolist()]
        full_foreground_mask = np.zeros(source_gray.shape, dtype=bool)
        full_target = full_foreground_mask[src_y1:src_y2, src_x1:src_x2]
        full_target[:, :] = foreground_mask[roi_y1:roi_y2, roi_x1:roi_x2]
        full_foreground_mask[src_y1:src_y2, src_x1:src_x2] = full_target
        full_silhouette = np.full(source_gray.shape, 255, dtype=np.uint8)
        full_silhouette[full_foreground_mask] = 0
        area_px, bbox = _mask_geometry(full_foreground_mask)
        feature_bbox_xyxy = _feature_bbox_from_geometry(
            bbox,
            area_px=area_px,
            fallback_bounds=source_bounds,
            source_shape=source_gray.shape,
        )
        return _ForegroundExtractionResult(
            roi_silhouette=roi_silhouette,
            full_silhouette=full_silhouette,
            roi_foreground_mask=foreground_mask.astype(bool, copy=False),
            full_foreground_mask=full_foreground_mask,
            area_px=area_px,
            bbox_inclusive_xyxy_px=bbox,
            feature_bbox_xyxy_px=feature_bbox_xyxy,
            extraction_mode=(
                contracts.ForegroundExtractionMode.THRESHOLD_FOREGROUND_V1.value
            ),
            fallback_used=False,
            primary_break_reason="",
            diagnostics=diagnostics,
        )

    def _render_silhouette(
        self,
        *,
        roi_gray: np.ndarray,
        source_gray: np.ndarray,
        source_bounds: np.ndarray,
        roi_bounds: np.ndarray,
    ) -> _ForegroundExtractionResult:
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
                raise ValueError(f"Primary contour failed ({primary_break_reason}) and fallback is disabled")
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

        src_x1, src_y1, src_x2, src_y2 = [
            int(value) for value in source_bounds.tolist()
        ]
        roi_x1, roi_y1, roi_x2, roi_y2 = [int(value) for value in roi_bounds.tolist()]
        full_silhouette = np.full(source_gray.shape, 255, dtype=np.uint8)
        roi_target = full_silhouette[src_y1:src_y2, src_x1:src_x2]
        roi_source_aligned = roi_silhouette[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_target[roi_source_aligned < 255] = 0
        full_silhouette[src_y1:src_y2, src_x1:src_x2] = roi_target
        full_foreground_mask = full_silhouette < 255
        roi_foreground_mask = roi_silhouette < 255
        area_px, bbox = _mask_geometry(full_foreground_mask)
        feature_bbox_xyxy = _feature_bbox_from_geometry(
            bbox,
            area_px=area_px,
            fallback_bounds=source_bounds,
            source_shape=source_gray.shape,
        )
        return _ForegroundExtractionResult(
            roi_silhouette=roi_silhouette,
            full_silhouette=full_silhouette,
            roi_foreground_mask=roi_foreground_mask,
            full_foreground_mask=full_foreground_mask,
            area_px=area_px,
            bbox_inclusive_xyxy_px=bbox,
            feature_bbox_xyxy_px=feature_bbox_xyxy,
            extraction_mode=(
                contracts.ForegroundExtractionMode.SILHOUETTE_CONTOUR_V2.value
            ),
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
            if tuple(roi_repr.shape) != expected_canvas_shape or tuple(foreground_mask.shape) != expected_canvas_shape:
                raise ValueError(
                    "Distance/yaw model expects brightness normalization, but the "
                    "foreground mask is not aligned with the regressor canvas."
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
        representation_source: np.ndarray,
        foreground_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        orientation_source_mode = self._config.orientation_source_mode
        if orientation_source_mode == ORIENTATION_SOURCE_RAW_GRAYSCALE:
            orientation_source_image = roi_source_gray
        elif orientation_source_mode in {
            ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
            ORIENTATION_SOURCE_RAW_GRAYSCALE_ON_WHITE,
        }:
            orientation_source_image = representation_source
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

    def _write_debug_artifacts(
        self,
        *,
        request: InferenceRequest,
        input_image_hash: contracts.FrameHash,
        runtime_revision: int | None,
        source_gray: np.ndarray,
        preprocessor_source_gray: np.ndarray,
        manual_mask: np.ndarray | None,
        roi_crop: np.ndarray | None,
        foreground_mask_before_component_cleanup: np.ndarray | None,
        foreground_mask: np.ndarray | None,
        distance_image: np.ndarray | None,
        orientation_image: np.ndarray | None,
        metadata: Mapping[str, Any],
        locator_result: contracts.LocatorResult,
        background_snapshot: BackgroundSnapshot | None = None,
        background_removal_mask: np.ndarray | None = None,
    ) -> dict[str, Path]:
        debug_paths = {str(key): Path(value) for key, value in locator_result.debug_artifacts.paths.items()}
        if not bool(request.save_debug_images):
            return debug_paths
        output_dir = (
            Path(request.debug_output_dir)
            if request.debug_output_dir is not None
            else default_debug_output_dir()
        )
        writer = DebugArtifactWriter(enabled=True, output_dir=output_dir)
        written = writer.write_preprocessing_artifacts(
            request_id=request.request_id,
            input_image_hash=input_image_hash,
            preprocessing_parameter_revision=runtime_revision,
            image_artifacts={
                ARTIFACT_ACCEPTED_RAW_FRAME: source_gray,
                ARTIFACT_GRAYSCALE_FRAME: source_gray,
                ARTIFACT_PREPROCESSOR_SOURCE_BEFORE_REGRESSOR_MASKS: source_gray,
                ARTIFACT_PREPROCESSOR_SOURCE_AFTER_REGRESSOR_MASKS: preprocessor_source_gray,
                ARTIFACT_MANUAL_MASK: manual_mask,
                ARTIFACT_BACKGROUND_SNAPSHOT: (
                    background_snapshot.grayscale_background
                    if background_snapshot is not None and background_snapshot.captured
                    else None
                ),
                ARTIFACT_BACKGROUND_REMOVAL_MASK: background_removal_mask,
                ARTIFACT_ROI_CROP: roi_crop,
                ARTIFACT_FOREGROUND_MASK_BEFORE_COMPONENT_CLEANUP: (
                    foreground_mask_before_component_cleanup
                ),
                ARTIFACT_FOREGROUND_MASK: foreground_mask,
                ARTIFACT_DISTANCE_IMAGE: distance_image,
                ARTIFACT_ORIENTATION_IMAGE: orientation_image,
            },
            metadata={**dict(metadata), contracts.PREPROCESSING_METADATA_DEBUG_PATHS: _path_map(debug_paths)},
        )
        debug_paths.update(written)
        return debug_paths

    def _runtime_parameter_revision(self) -> int | None:
        if self._runtime_parameter_revision_getter is not None:
            revision = self._runtime_parameter_revision_getter()
            return int(revision) if revision is not None else None
        if self._locator_parameter_state is not None:
            _config, revision = self._locator_parameter_state.snapshot()
            return int(revision)
        state = getattr(self._locator, "parameter_state", None)
        if state is not None and callable(getattr(state, "snapshot", None)):
            _config, revision = state.snapshot()
            return int(revision)
        return None

    def _roi_canvas_size(self) -> tuple[int, int]:
        return (
            int(self._config.silhouette_config.normalized_roi_canvas_width_px()),
            int(self._config.silhouette_config.normalized_roi_canvas_height_px()),
        )


def _threshold_foreground_mask(
    roi_gray: np.ndarray,
    *,
    policy: ForegroundExtractionPolicySnapshot,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray = np.asarray(roi_gray, dtype=np.uint8)
    if gray.ndim != 2:
        raise ValueError(
            "threshold foreground expects a 2D grayscale ROI; "
            f"got {gray.shape}."
        )
    background_white = _estimate_background_white(
        gray,
        policy.threshold_white_percentile,
    )
    relative_threshold = _clamped_uint8(
        int(round(background_white)) - int(policy.threshold_margin_px)
    )
    otsu_threshold, _otsu_image = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    otsu_threshold_i = _clamped_uint8(int(round(float(otsu_threshold))))
    otsu_mask = gray <= otsu_threshold_i
    otsu_fraction = _mask_fraction(otsu_mask)
    if (
        otsu_fraction >= float(policy.threshold_min_foreground_fraction)
        and otsu_fraction <= float(policy.threshold_max_foreground_fraction)
    ):
        selected_threshold = min(otsu_threshold_i, relative_threshold)
        threshold_source = "otsu_capped_by_background_white"
    else:
        selected_threshold = relative_threshold
        threshold_source = "background_white_relative"

    threshold_mask = gray <= int(selected_threshold)
    foreground_before_cleanup_px = int(np.count_nonzero(threshold_mask))
    closed_mask = _close_binary_mask(
        threshold_mask,
        kernel_size_px=int(policy.threshold_morphology_close_kernel_px),
    )
    component_mask, component_diagnostics = _select_foreground_components(
        closed_mask,
        gray=gray,
        selected_threshold=int(selected_threshold),
        background_white=float(background_white),
    )
    foreground_mask = component_mask
    if bool(policy.threshold_fill_holes):
        foreground_mask = _fill_binary_holes(foreground_mask)
    foreground_after_cleanup_px = int(np.count_nonzero(foreground_mask))
    diagnostics = {
        "foreground_extraction_algorithm": (
            contracts.ForegroundExtractionMode.THRESHOLD_FOREGROUND_V1.value
        ),
        "background_white_estimate": float(background_white),
        "background_white_percentile": float(policy.threshold_white_percentile),
        "relative_threshold": int(relative_threshold),
        "otsu_threshold": int(otsu_threshold_i),
        "otsu_foreground_fraction": float(otsu_fraction),
        "selected_threshold": int(selected_threshold),
        "selected_threshold_source": threshold_source,
        "foreground_pixel_count_before_cleanup": foreground_before_cleanup_px,
        "foreground_pixel_count_after_threshold_close": int(
            np.count_nonzero(closed_mask)
        ),
        "foreground_pixel_count_after_component_selection": int(
            np.count_nonzero(component_mask)
        ),
        "foreground_pixel_count_after_cleanup": foreground_after_cleanup_px,
        "morphology_close_kernel_px": int(
            _normalized_odd_kernel_size(policy.threshold_morphology_close_kernel_px)
        ),
        "fill_holes": bool(policy.threshold_fill_holes),
    }
    diagnostics.update(component_diagnostics)
    return foreground_mask.astype(bool, copy=False), diagnostics


def _select_foreground_components(
    mask: np.ndarray,
    *,
    gray: np.ndarray,
    selected_threshold: int,
    background_white: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate = np.asarray(mask, dtype=bool)
    roi_h, roi_w = int(candidate.shape[0]), int(candidate.shape[1])
    empty = np.zeros(candidate.shape, dtype=bool)
    if roi_h <= 0 or roi_w <= 0 or not bool(np.any(candidate)):
        return empty, {
            "component_selection_enabled": True,
            "component_selection_status": "empty_threshold_mask",
            "component_selection_source": "threshold_mask",
            "component_selection_component_count": 0,
            "component_selection_selected_label": None,
            "component_selection_selected_area_px": 0,
            "component_selection_selected_bbox_xywh_px": None,
            "component_selection_strict_threshold": _strict_component_threshold(
                selected_threshold,
                background_white,
            ),
        }

    selected, candidate_diag = _select_best_component(
        candidate,
        source="threshold_mask",
        reject_saturated=True,
    )
    strict_threshold = _strict_component_threshold(selected_threshold, background_white)
    strict_selected = empty
    strict_diag: dict[str, Any] | None = None
    if bool(candidate_diag.get("largest_component_saturated", False)):
        strict_mask = np.asarray(gray, dtype=np.uint8) <= int(strict_threshold)
        strict_mask = _close_binary_mask(strict_mask, kernel_size_px=3)
        strict_selected, strict_diag = _select_best_component(
            strict_mask,
            source="strict_threshold_mask",
            reject_saturated=False,
        )

    use_strict = (
        strict_diag is not None
        and int(strict_diag.get("selected_area_px") or 0)
        > max(0, int(candidate_diag.get("selected_area_px") or 0))
    )
    final = strict_selected if use_strict else selected
    final_diag = strict_diag if use_strict and strict_diag is not None else candidate_diag
    status = "selected_component"
    if use_strict:
        status = "selected_strict_component_after_saturated_candidate"
    elif int(candidate_diag.get("selected_area_px") or 0) <= 0:
        final = candidate
        status = "fallback_all_components"

    diagnostics = {
        "component_selection_enabled": True,
        "component_selection_status": status,
        "component_selection_source": final_diag.get("source"),
        "component_selection_component_count": int(
            final_diag.get("component_count") or 0
        ),
        "component_selection_selected_label": final_diag.get("selected_label"),
        "component_selection_selected_area_px": int(
            final_diag.get("selected_area_px") or 0
        ),
        "component_selection_selected_bbox_xywh_px": final_diag.get(
            "selected_bbox_xywh_px"
        ),
        "component_selection_selected_score": final_diag.get("selected_score"),
        "component_selection_largest_area_px": int(
            candidate_diag.get("largest_area_px") or 0
        ),
        "component_selection_largest_bbox_xywh_px": candidate_diag.get(
            "largest_bbox_xywh_px"
        ),
        "component_selection_largest_saturated": bool(
            candidate_diag.get("largest_component_saturated", False)
        ),
        "component_selection_saturated_rejected_count": int(
            candidate_diag.get("saturated_rejected_count") or 0
        ),
        "component_selection_strict_threshold": int(strict_threshold),
    }
    if strict_diag is not None:
        diagnostics.update(
            {
                "component_selection_strict_component_count": int(
                    strict_diag.get("component_count") or 0
                ),
                "component_selection_strict_selected_area_px": int(
                    strict_diag.get("selected_area_px") or 0
                ),
                "component_selection_strict_selected_bbox_xywh_px": strict_diag.get(
                    "selected_bbox_xywh_px"
                ),
            }
        )
    return final.astype(bool, copy=False), diagnostics


def _select_best_component(
    mask: np.ndarray,
    *,
    source: str,
    reject_saturated: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    component_mask = np.asarray(mask, dtype=np.uint8)
    roi_h, roi_w = int(component_mask.shape[0]), int(component_mask.shape[1])
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        component_mask,
        8,
    )
    selected_label: int | None = None
    selected_score = -1.0
    selected_area = 0
    selected_bbox: tuple[int, int, int, int] | None = None
    largest_area = 0
    largest_bbox: tuple[int, int, int, int] | None = None
    largest_saturated = False
    saturated_rejected_count = 0
    center_x = (float(roi_w) - 1.0) * 0.5
    center_y = (float(roi_h) - 1.0) * 0.5
    max_center_distance = max(1.0, math.hypot(center_x, center_y))

    for label in range(1, int(label_count)):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area <= 0:
            continue
        bbox = (x, y, w, h)
        saturated = _component_is_roi_saturated(
            bbox,
            area_px=area,
            roi_w=roi_w,
            roi_h=roi_h,
        )
        if area > largest_area:
            largest_area = area
            largest_bbox = bbox
            largest_saturated = saturated
        if reject_saturated and saturated:
            saturated_rejected_count += 1
            continue
        cx, cy = [float(value) for value in centroids[label]]
        center_distance = math.hypot(cx - center_x, cy - center_y) / max_center_distance
        center_weight = max(0.25, 1.0 - center_distance)
        border_touches = (
            int(x <= 0)
            + int(y <= 0)
            + int(x + w >= roi_w)
            + int(y + h >= roi_h)
        )
        border_weight = 1.0 / (1.0 + (0.35 * float(border_touches)))
        bbox_area_fraction = float(w * h) / max(1.0, float(roi_w * roi_h))
        extent_weight = max(0.15, 1.0 - max(0.0, bbox_area_fraction - 0.55))
        score = float(area) * center_weight * border_weight * extent_weight
        if score > selected_score:
            selected_score = score
            selected_label = label
            selected_area = area
            selected_bbox = bbox

    selected = labels == int(selected_label) if selected_label is not None else np.zeros(
        labels.shape,
        dtype=bool,
    )
    return selected.astype(bool, copy=False), {
        "source": source,
        "component_count": max(0, int(label_count) - 1),
        "selected_label": selected_label,
        "selected_score": float(selected_score) if selected_label is not None else None,
        "selected_area_px": int(selected_area),
        "selected_bbox_xywh_px": selected_bbox,
        "largest_area_px": int(largest_area),
        "largest_bbox_xywh_px": largest_bbox,
        "largest_component_saturated": bool(largest_saturated),
        "saturated_rejected_count": int(saturated_rejected_count),
    }


def _foreground_mask_component_cleanup(
    foreground_mask: np.ndarray,
    *,
    roi_gray: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = "foreground_mask_after_background_removal"
    mask = np.asarray(foreground_mask, dtype=bool)
    original_count = int(np.count_nonzero(mask))
    metadata: dict[str, Any] = {
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_STATUS: (
            "empty_mask" if original_count <= 0 else "single_component"
        ),
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_APPLIED: False,
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_COUNT: 0,
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_SOURCE: source,
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_KEPT_LABEL: None,
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_KEPT_AREA_PX: (
            original_count
        ),
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_REMOVED_AREA_PX: 0,
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_REMOVED_FRACTION: 0.0,
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_KEPT_ROI_BBOX_XYXY_PX: None,
        "foreground_mask_component_cleanup_selection_method": "largest_component",
        "foreground_mask_component_cleanup_kept_labels": (),
    }
    if original_count <= 0:
        return np.ascontiguousarray(mask), metadata

    selected, diagnostics = _select_dark_foreground_components(
        mask,
        roi_gray=roi_gray,
    )
    if selected is None:
        selected, diagnostics = _select_best_component(
            mask,
            source=source,
            reject_saturated=False,
        )
    component_count = int(diagnostics.get("component_count") or 0)
    selected_area = int(diagnostics.get("selected_area_px") or 0)
    removed_count = max(0, original_count - selected_area)
    selected_bbox = diagnostics.get("selected_bbox_xywh_px")
    metadata.update(
        {
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_COUNT: (
                component_count
            ),
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_KEPT_LABEL: (
                diagnostics.get("selected_label")
            ),
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_KEPT_AREA_PX: (
                selected_area
            ),
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_REMOVED_AREA_PX: (
                removed_count
            ),
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_REMOVED_FRACTION: (
                float(removed_count) / float(max(1, original_count))
            ),
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_KEPT_ROI_BBOX_XYXY_PX: (
                _xywh_to_xyxy_tuple(selected_bbox)
            ),
            "foreground_mask_component_cleanup_selection_method": diagnostics.get(
                "source"
            ),
            "foreground_mask_component_cleanup_kept_labels": tuple(
                int(label)
                for label in diagnostics.get("selected_labels", ())
            ),
            "foreground_mask_component_cleanup_kept_dark_area_px": int(
                diagnostics.get("selected_dark_area_px") or 0
            ),
            "foreground_mask_component_cleanup_kept_dark_fraction": (
                diagnostics.get("selected_dark_fraction")
            ),
            "foreground_mask_component_cleanup_kept_mean_gray": (
                diagnostics.get("selected_mean_gray")
            ),
        }
    )
    if component_count <= 1:
        metadata[
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_STATUS
        ] = "single_component"
        return np.ascontiguousarray(mask), metadata
    if selected_area <= 0 or not bool(np.any(selected)):
        metadata[
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_STATUS
        ] = str(diagnostics.get("status") or "no_selected_component")
        metadata[
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_KEPT_AREA_PX
        ] = original_count
        metadata[
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_REMOVED_AREA_PX
        ] = 0
        metadata[
            contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_REMOVED_FRACTION
        ] = 0.0
        return np.ascontiguousarray(mask), metadata

    metadata[
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_STATUS
    ] = str(diagnostics.get("status") or "kept_best_component")
    metadata[
        contracts.PREPROCESSING_METADATA_FOREGROUND_MASK_COMPONENT_CLEANUP_APPLIED
    ] = bool(removed_count > 0)
    return np.ascontiguousarray(selected.astype(bool, copy=False)), metadata


def _select_dark_foreground_components(
    mask: np.ndarray,
    *,
    roi_gray: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if roi_gray is None:
        return None, {}
    gray = np.asarray(roi_gray, dtype=np.uint8)
    candidate = np.asarray(mask, dtype=bool)
    if gray.shape != candidate.shape:
        return None, {
            "status": "dark_selection_shape_mismatch",
            "source": "dark_foreground_components",
        }

    label_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8),
        8,
    )
    components: list[dict[str, Any]] = []
    for label in range(1, int(label_count)):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area <= 0:
            continue
        pixels = gray[labels == label]
        dark_pixels = pixels <= 120
        dark_area = int(np.count_nonzero(dark_pixels))
        dark_fraction = float(dark_area) / float(max(1, area))
        mean_gray = float(np.mean(pixels))
        darkness_mass = float(np.sum(np.maximum(0, 180 - pixels.astype(np.int16))))
        components.append(
            {
                "label": int(label),
                "bbox": (x, y, w, h),
                "area": int(area),
                "dark_area": int(dark_area),
                "dark_fraction": float(dark_fraction),
                "mean_gray": float(mean_gray),
                "darkness_mass": float(darkness_mass),
            }
        )

    if not components:
        return np.zeros(candidate.shape, dtype=bool), {
            "status": "no_selected_component",
            "source": "dark_foreground_components",
            "component_count": 0,
            "selected_label": None,
            "selected_labels": (),
            "selected_area_px": 0,
            "selected_bbox_xywh_px": None,
            "selected_dark_area_px": 0,
            "selected_dark_fraction": None,
            "selected_mean_gray": None,
        }

    dark_components = [
        item
        for item in components
        if item["dark_fraction"] >= 0.25 and item["dark_area"] >= 64
    ]
    if not dark_components:
        return np.zeros(candidate.shape, dtype=bool), {
            "status": "no_dark_component_safety_skip",
            "source": "dark_foreground_components",
            "component_count": len(components),
            "selected_label": None,
            "selected_labels": (),
            "selected_area_px": 0,
            "selected_bbox_xywh_px": None,
            "selected_dark_area_px": 0,
            "selected_dark_fraction": None,
            "selected_mean_gray": None,
        }

    anchor = max(dark_components, key=lambda item: item["darkness_mass"])
    min_dark_mass = max(64.0, float(anchor["darkness_mass"]) * 0.03)
    kept_components = [
        item
        for item in dark_components
        if float(item["darkness_mass"]) >= min_dark_mass
    ]
    selected_labels = tuple(int(item["label"]) for item in kept_components)
    selected = np.isin(labels, selected_labels)
    selected_area = int(np.count_nonzero(selected))
    bbox = _mask_bbox_xywh(selected)
    selected_pixels = gray[selected]
    selected_dark_area = int(np.count_nonzero(selected_pixels <= 120))
    return selected.astype(bool, copy=False), {
        "status": "kept_dark_components",
        "source": "dark_foreground_components",
        "component_count": len(components),
        "selected_label": int(anchor["label"]),
        "selected_labels": selected_labels,
        "selected_area_px": selected_area,
        "selected_bbox_xywh_px": bbox,
        "selected_dark_area_px": selected_dark_area,
        "selected_dark_fraction": (
            float(selected_dark_area) / float(max(1, selected_area))
        ),
        "selected_mean_gray": float(np.mean(selected_pixels))
        if selected_pixels.size
        else None,
    }


def _foreground_geometry_from_roi_mask(
    foreground_mask: np.ndarray,
    *,
    source_gray: np.ndarray,
    source_bounds: np.ndarray,
    roi_bounds: np.ndarray,
) -> tuple[np.ndarray, int, tuple[int, int, int, int], np.ndarray]:
    src_x1, src_y1, src_x2, src_y2 = [
        int(value) for value in source_bounds.tolist()
    ]
    roi_x1, roi_y1, roi_x2, roi_y2 = [int(value) for value in roi_bounds.tolist()]
    full_foreground_mask = np.zeros(source_gray.shape, dtype=bool)
    full_target = full_foreground_mask[src_y1:src_y2, src_x1:src_x2]
    full_target[:, :] = np.asarray(foreground_mask, dtype=bool)[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ]
    full_foreground_mask[src_y1:src_y2, src_x1:src_x2] = full_target
    area_px, bbox = _mask_geometry(full_foreground_mask)
    feature_bbox_xyxy = _feature_bbox_from_geometry(
        bbox,
        area_px=area_px,
        fallback_bounds=source_bounds,
        source_shape=source_gray.shape,
    )
    return full_foreground_mask, area_px, bbox, feature_bbox_xyxy


def _xywh_to_xyxy_tuple(
    bbox_xywh: object,
) -> tuple[int, int, int, int] | None:
    if bbox_xywh is None:
        return None
    x, y, w, h = [int(value) for value in bbox_xywh]
    return (x, y, x + w, y + h)


def _mask_bbox_xywh(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return (x1, y1, x2 - x1, y2 - y1)


def _component_is_roi_saturated(
    bbox_xywh: tuple[int, int, int, int],
    *,
    area_px: int,
    roi_w: int,
    roi_h: int,
) -> bool:
    x, y, w, h = bbox_xywh
    roi_area = max(1.0, float(roi_w * roi_h))
    bbox_area_fraction = float(w * h) / roi_area
    area_fraction = float(area_px) / roi_area
    width_fraction = float(w) / max(1.0, float(roi_w))
    height_fraction = float(h) / max(1.0, float(roi_h))
    border_touches = (
        int(x <= 0)
        + int(y <= 0)
        + int(x + w >= roi_w)
        + int(y + h >= roi_h)
    )
    return (
        (width_fraction >= 0.95 and height_fraction >= 0.95)
        or (border_touches >= 3 and bbox_area_fraction >= 0.70)
        or (border_touches >= 2 and bbox_area_fraction >= 0.80 and area_fraction >= 0.30)
    )


def _strict_component_threshold(
    selected_threshold: int,
    background_white: float,
) -> int:
    return _clamped_uint8(
        min(
            int(selected_threshold) - 55,
            int(round(float(background_white))) - 105,
        )
    )


def _estimate_background_white(gray: np.ndarray, percentile: float) -> float:
    flat = np.asarray(gray, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return 255.0
    saturated_fraction = float(np.mean(flat >= 250))
    unsaturated = flat[flat < 250]
    if saturated_fraction < 0.50 and unsaturated.size >= max(
        32,
        int(flat.size * 0.05),
    ):
        sample = unsaturated
    else:
        sample = flat
    return float(np.percentile(sample.astype(np.float32), float(percentile)))


def _close_binary_mask(mask: np.ndarray, *, kernel_size_px: int) -> np.ndarray:
    kernel_size = _normalized_odd_kernel_size(kernel_size_px)
    if kernel_size <= 1:
        return mask.astype(bool, copy=True)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    closed = cv2.morphologyEx(
        mask.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        kernel,
    )
    return closed > 0


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    source = mask.astype(np.uint8)
    padded = np.pad(source, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = (flood == 0) & (padded == 0)
    filled = padded.astype(bool) | holes
    return filled[1:-1, 1:-1]


def _normalized_odd_kernel_size(value: int) -> int:
    size = max(0, int(value))
    if size <= 1:
        return 0
    return size if size % 2 == 1 else size + 1


def _mask_fraction(mask: np.ndarray) -> float:
    total = int(mask.size)
    if total <= 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(total)


def _feature_bbox_from_geometry(
    bbox: tuple[int, int, int, int],
    *,
    area_px: int,
    fallback_bounds: np.ndarray,
    source_shape: tuple[int, ...],
) -> np.ndarray:
    if area_px <= 0:
        return np.asarray(fallback_bounds, dtype=np.float32)
    source_h, source_w = int(source_shape[0]), int(source_shape[1])
    return np.asarray(
        [
            float(bbox[0]),
            float(bbox[1]),
            float(min(source_w, bbox[2] + 1)),
            float(min(source_h, bbox[3] + 1)),
        ],
        dtype=np.float32,
    )


def _clamped_uint8(value: int) -> int:
    return max(0, min(255, int(value)))


def _locator_display_image(
    source_gray: np.ndarray,
    locator_result: contracts.LocatorResult,
) -> np.ndarray:
    paths = locator_result.debug_artifacts.paths
    for key in (
        contracts.DISPLAY_ARTIFACT_CHOSEN_CONTOUR,
        contracts.DISPLAY_ARTIFACT_LOCATOR_OVERLAY,
        contracts.DISPLAY_ARTIFACT_CANDIDATE_CONTOURS,
        contracts.DISPLAY_ARTIFACT_EDGE_MAP,
    ):
        path = paths.get(key)
        if path is None:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image
    return source_gray


def _locator_parameter(
    locator_result: contracts.LocatorResult,
    key: str,
) -> object | None:
    params = locator_result.extras.get("locator_parameters")
    if isinstance(params, Mapping):
        return params.get(key)
    return None


def _reason_text(reasons: tuple[str, ...]) -> str:
    return ";".join(str(reason) for reason in reasons if str(reason).strip())


def _foreground_locator_consistency_check(
    *,
    foreground_result: _ForegroundExtractionResult,
    locator_result: contracts.LocatorResult,
) -> tuple[dict[str, Any], str | None]:
    locator_bbox = locator_result.bbox_xyxy_px
    if locator_bbox is None:
        return (
            {
                "foreground_locator_consistency_status": "skipped_no_locator_bbox",
                "foreground_locator_consistency_reason": "locator_result_missing_bbox",
            },
            None,
        )
    lx1, ly1, lx2, ly2 = [float(value) for value in locator_bbox]
    sx1, sy1, sx2, sy2 = [
        float(value) for value in foreground_result.feature_bbox_xyxy_px
    ]
    locator_w = max(1e-6, lx2 - lx1)
    locator_h = max(1e-6, ly2 - ly1)
    foreground_w = max(0.0, sx2 - sx1)
    foreground_h = max(0.0, sy2 - sy1)
    locator_area = locator_w * locator_h
    foreground_bbox_area = foreground_w * foreground_h
    foreground_pixel_count = int(foreground_result.area_px)
    width_ratio = foreground_w / locator_w
    height_ratio = foreground_h / locator_h
    bbox_area_ratio = foreground_bbox_area / locator_area
    pixel_area_ratio = float(foreground_pixel_count) / locator_area
    metadata: dict[str, Any] = {
        "foreground_locator_consistency_status": "ok",
        "foreground_locator_consistency_reason": None,
        "foreground_locator_bbox_xyxy_px": tuple(float(value) for value in locator_bbox),
        "foreground_locator_foreground_bbox_xyxy_px": tuple(
            float(value) for value in foreground_result.feature_bbox_xyxy_px
        ),
        "foreground_locator_locator_width_px": float(locator_w),
        "foreground_locator_locator_height_px": float(locator_h),
        "foreground_locator_locator_area_px": float(locator_area),
        "foreground_locator_foreground_width_px": float(foreground_w),
        "foreground_locator_foreground_height_px": float(foreground_h),
        "foreground_locator_foreground_bbox_area_px": float(foreground_bbox_area),
        "foreground_locator_foreground_pixel_count": int(foreground_pixel_count),
        "foreground_locator_width_ratio": float(width_ratio),
        "foreground_locator_height_ratio": float(height_ratio),
        "foreground_locator_bbox_area_ratio": float(bbox_area_ratio),
        "foreground_locator_pixel_area_ratio": float(pixel_area_ratio),
        "foreground_locator_min_guard_locator_area_px": 5_000.0,
        "foreground_locator_small_min_area_ratio": 0.02,
        "foreground_locator_small_min_width_ratio": 0.15,
        "foreground_locator_small_min_height_ratio": 0.15,
        "foreground_locator_expanded_max_bbox_area_ratio": 4.0,
        "foreground_locator_expanded_max_width_ratio": 1.75,
        "foreground_locator_expanded_max_height_ratio": 1.75,
    }
    if locator_area < 5_000.0:
        metadata["foreground_locator_consistency_status"] = "skipped_small_locator"
        metadata["foreground_locator_consistency_reason"] = (
            "locator_area_below_guard_minimum"
        )
        return metadata, None

    if bbox_area_ratio < 0.02 and width_ratio < 0.15 and height_ratio < 0.15:
        metadata["foreground_locator_consistency_status"] = (
            "diagnostic_small_foreground"
        )
        metadata["foreground_locator_consistency_reason"] = (
            "foreground_bbox_implausibly_small_relative_to_locator"
        )
        metadata["foreground_locator_consistency_warning"] = (
            "foreground bbox is implausibly small relative to accepted locator bbox: "
            f"foreground=({foreground_w:.1f}x{foreground_h:.1f}), "
            f"locator=({locator_w:.1f}x{locator_h:.1f}), "
            f"area_ratio={bbox_area_ratio:.4f}"
        )
        return metadata, None

    if (
        bbox_area_ratio > 4.0
        and width_ratio > 1.75
        and height_ratio > 1.75
    ):
        metadata["foreground_locator_consistency_status"] = (
            "diagnostic_expanded_foreground"
        )
        metadata["foreground_locator_consistency_reason"] = (
            "foreground_bbox_implausibly_large_relative_to_locator"
        )
        metadata["foreground_locator_consistency_warning"] = (
            "foreground bbox is implausibly large relative to accepted locator bbox: "
            f"foreground=({foreground_w:.1f}x{foreground_h:.1f}), "
            f"locator=({locator_w:.1f}x{locator_h:.1f}), "
            f"width_ratio={width_ratio:.3f}, "
            f"height_ratio={height_ratio:.3f}, "
            f"bbox_area_ratio={bbox_area_ratio:.3f}, "
            f"pixel_area_ratio={pixel_area_ratio:.3f}"
        )
        return metadata, None

    return metadata, None


def _background_for_stage(
    snapshot: BackgroundSnapshot | None,
    *,
    warning: str | None,
    apply_to_stage: bool,
) -> BackgroundSnapshot | None:
    if not bool(apply_to_stage) or warning is not None:
        return None
    if snapshot is None or not snapshot.captured or not snapshot.enabled:
        return None
    return snapshot


def _background_application_metadata(
    *,
    stage_policy: StageTransformPolicySnapshot,
    locator_result: contracts.LocatorResult,
    roi_background_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    locator_applied = bool(
        stage_policy.apply_background_removal_to_roi_locator
        and locator_result.extras.get("background_revision") is not None
    )
    regressor_applied = bool(
        roi_background_metadata.get(
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED
        )
    )
    regressor_count = int(
        roi_background_metadata.get(
            contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_REMOVE_PIXEL_COUNT
        )
        or 0
    )
    spaces: list[str] = []
    if locator_applied:
        spaces.append("roi_locator_input")
    if regressor_applied:
        spaces.append("regressor_preprocessing_input")
    return {
        contracts.PREPROCESSING_METADATA_APPLY_BACKGROUND_REMOVAL_TO_ROI_LOCATOR: bool(
            stage_policy.apply_background_removal_to_roi_locator
        ),
        contracts.PREPROCESSING_METADATA_APPLY_BACKGROUND_REMOVAL_TO_REGRESSOR_PREPROCESSING: bool(
            stage_policy.apply_background_removal_to_regressor_preprocessing
        ),
        contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED_TO_ROI_LOCATOR: locator_applied,
        contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED_TO_REGRESSOR_PREPROCESSING: regressor_applied,
        contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED: bool(
            locator_applied or regressor_applied
        ),
        contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVE_PIXEL_COUNT: regressor_count,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_APPLIED: regressor_applied,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_CROP_REMOVE_PIXEL_COUNT: regressor_count,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_FCN_APPLIED: False,
        contracts.PREPROCESSING_METADATA_BACKGROUND_ROI_FCN_REMOVE_PIXEL_COUNT: 0,
        contracts.PREPROCESSING_METADATA_BACKGROUND_APPLICATION_SPACE: (
            "+".join(spaces) if spaces else None
        ),
    }


def _path_map(paths: Mapping[str, Path]) -> dict[str, str]:
    return {str(key): str(value) for key, value in paths.items()}


def _mask_application_space(*, locator: bool, regressor: bool) -> str | None:
    spaces: list[str] = []
    if locator:
        spaces.append("locator")
    if regressor:
        spaces.append("regressor_preprocessing")
    return ",".join(spaces) if spaces else None


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
            "frame_mask_source_width_px": int(source_width_px),
            "frame_mask_source_height_px": int(source_height_px),
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


def _failure_details(
    *,
    request: InferenceRequest,
    input_image_hash: contracts.FrameHash,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    keys = (
        "request_id",
        contracts.PREPROCESSING_METADATA_LOCATOR_KIND,
        contracts.PREPROCESSING_METADATA_LOCATOR_RESULT_ACCEPTED,
        contracts.PREPROCESSING_METADATA_LOCATOR_CONFIDENCE,
        contracts.PREPROCESSING_METADATA_LOCATOR_CANDIDATE_COUNT,
        contracts.PREPROCESSING_METADATA_ROI_ACCEPTED,
        contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASON,
        contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASONS,
        contracts.PREPROCESSING_METADATA_ROI_REQUESTED_XYXY_PX,
        contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX,
        contracts.PREPROCESSING_METADATA_ROI_CLIPPED,
        contracts.PREPROCESSING_METADATA_ROI_CLIP_MAX_PX,
        "frame_mask_warning",
        "frame_mask_revision",
        "frame_mask_pixel_count",
        "manual_mask_applied_to_regressor_preprocessing",
        contracts.PREPROCESSING_METADATA_DEBUG_PATHS,
    )
    details = {key: metadata.get(key) for key in keys if key in metadata}
    details["request_id"] = request.request_id
    details["frame_hash"] = input_image_hash.value
    details["mark_frame_processed"] = True
    return details


__all__ = [
    "LocatorDiagnosticResult",
    "PreprocessingDebugError",
    "RoiRejectedError",
    "TriStreamLivePreprocessor",
]
