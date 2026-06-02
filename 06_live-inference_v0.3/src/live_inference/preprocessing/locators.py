"""Generic ROI locator implementations for live inference v0.3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

import interfaces.contracts as contracts
from live_inference.masking import BackgroundSnapshot, BackgroundState


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class BackgroundEdgeLocatorConfig:
    """Small runtime parameter set for the deterministic background/edge locator."""

    roi_width_px: int = 300
    roi_height_px: int = 300
    background_threshold: int = 25
    min_foreground_area_px: int = 250
    canny_low_threshold: int = 40
    canny_high_threshold: int = 120
    morphology_close_kernel_px: int = 5
    dilation_kernel_px: int = 3
    roi_clip_tolerance_px: int = 0
    min_candidate_score: float = 0.05
    min_roi_content_fraction: float = 0.0005

    def normalized(self) -> "BackgroundEdgeLocatorConfig":
        low = max(0, min(255, int(self.canny_low_threshold)))
        high = max(low + 1, min(255, int(self.canny_high_threshold)))
        return replace(
            self,
            roi_width_px=max(1, int(self.roi_width_px)),
            roi_height_px=max(1, int(self.roi_height_px)),
            background_threshold=max(0, min(255, int(self.background_threshold))),
            min_foreground_area_px=max(0, int(self.min_foreground_area_px)),
            canny_low_threshold=low,
            canny_high_threshold=high,
            morphology_close_kernel_px=max(0, int(self.morphology_close_kernel_px)),
            dilation_kernel_px=max(0, int(self.dilation_kernel_px)),
            roi_clip_tolerance_px=max(0, int(self.roi_clip_tolerance_px)),
            min_candidate_score=max(0.0, min(1.0, float(self.min_candidate_score))),
            min_roi_content_fraction=max(0.0, float(self.min_roi_content_fraction)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_width_px": int(self.roi_width_px),
            "roi_height_px": int(self.roi_height_px),
            "background_threshold": int(self.background_threshold),
            "min_foreground_area_px": int(self.min_foreground_area_px),
            "canny_low_threshold": int(self.canny_low_threshold),
            "canny_high_threshold": int(self.canny_high_threshold),
            "morphology_close_kernel_px": int(self.morphology_close_kernel_px),
            "dilation_kernel_px": int(self.dilation_kernel_px),
            "roi_clip_tolerance_px": int(self.roi_clip_tolerance_px),
            "min_candidate_score": float(self.min_candidate_score),
            "min_roi_content_fraction": float(self.min_roi_content_fraction),
        }


class LocatorRuntimeParameterState:
    """Thread-light parameter holder used by the GUI and locator."""

    def __init__(self, config: BackgroundEdgeLocatorConfig | None = None) -> None:
        self._config = (config or BackgroundEdgeLocatorConfig()).normalized()
        self._revision = 0

    def snapshot(self) -> tuple[BackgroundEdgeLocatorConfig, int]:
        return self._config, int(self._revision)

    def update(self, **updates: Any) -> tuple[BackgroundEdgeLocatorConfig, int]:
        allowed = set(BackgroundEdgeLocatorConfig.__dataclass_fields__)
        payload = {
            key: value
            for key, value in updates.items()
            if key in allowed and value is not None
        }
        if not payload:
            return self.snapshot()
        next_config = replace(self._config, **payload).normalized()
        if next_config == self._config:
            return self.snapshot()
        self._config = next_config
        self._revision += 1
        return self.snapshot()


class BackgroundEdgeLocator:
    """Inspectable deterministic locator for real camera demos."""

    def __init__(
        self,
        *,
        background_state: BackgroundState | None = None,
        parameter_state: LocatorRuntimeParameterState | None = None,
        config: BackgroundEdgeLocatorConfig | None = None,
    ) -> None:
        self._background_state = background_state
        self._parameter_state = parameter_state or LocatorRuntimeParameterState(config)

    @property
    def locator_kind(self) -> contracts.LocatorKind:
        return contracts.LocatorKind.BACKGROUND_EDGE_V1

    @property
    def parameter_state(self) -> LocatorRuntimeParameterState:
        return self._parameter_state

    def locate(
        self,
        request: contracts.LocatorRequest,
        image_bytes: bytes,
    ) -> contracts.LocatorResult:
        config, revision = self._parameter_state.snapshot()
        config = _config_from_request(request, config).normalized()
        gray = _decode_image_bytes_to_grayscale(image_bytes)
        source_h, source_w = int(gray.shape[0]), int(gray.shape[1])
        warnings: list[str] = []
        rejection_reasons: list[str] = []
        background_removal_required = _background_removal_explicitly_requested(request)
        background = self._background_snapshot(config, source_w, source_h, warnings, request)
        background_applied = bool(background is not None and background.captured and background.enabled)
        manual_ignore_mask = _manual_ignore_mask_from_request(
            request,
            source_w=source_w,
            source_h=source_h,
            warnings=warnings,
        )

        if background_removal_required and not background_applied:
            warnings.append(
                "Background removal was requested for ROI locator, but no enabled "
                "matching background is available; refusing dark-on-light fallback."
            )
            diff = None
            foreground_mask = np.zeros(gray.shape, dtype=bool)
        else:
            diff, foreground_mask = _foreground_mask(gray, background, config, warnings)
        if manual_ignore_mask is not None:
            foreground_mask = np.array(foreground_mask, dtype=bool, copy=True)
            foreground_mask[manual_ignore_mask] = False
        if not bool(np.any(foreground_mask)):
            rejection_reasons.append(contracts.LocatorFailureReason.NO_FOREGROUND.value)

        foreground_mask = _morphology_mask(foreground_mask, config)
        edge_map = _edge_map(gray, foreground_mask, config)
        if bool(np.any(foreground_mask)) and not bool(np.any(edge_map)):
            warnings.append("No Canny edges found; falling back to foreground-mask contours.")

        candidates = _build_candidates(
            edge_map=edge_map,
            foreground_mask=foreground_mask,
            source_wh=(source_w, source_h),
            config=config,
            prefer_components=background_applied,
        )
        accepted_candidates = tuple(candidate for candidate in candidates if not candidate.rejection_reason)
        if not accepted_candidates:
            if bool(np.any(foreground_mask)):
                rejection_reasons.append(contracts.LocatorFailureReason.NO_CANDIDATES.value)
            chosen = None
        else:
            eligible_candidates = tuple(
                candidate
                for candidate in accepted_candidates
                if not _candidate_roi_rejection_reasons(
                    candidate,
                    foreground_mask=foreground_mask,
                    source_wh=(source_w, source_h),
                    config=config,
                )
            )
            chosen = max(
                eligible_candidates or accepted_candidates,
                key=lambda item: item.score,
            )

        if chosen is not None:
            roi_geometry = _roi_geometry(
                center_xy=chosen.center_xy_px,
                source_wh=(source_w, source_h),
                roi_wh=(config.roi_width_px, config.roi_height_px),
            )
            clip_max = max(roi_geometry["clip_amount"].values())
            content_fraction = _roi_content_fraction(
                foreground_mask,
                roi_geometry["source_xyxy"],
            )
            rejection_reasons.extend(
                _candidate_roi_rejection_reasons(
                    chosen,
                    foreground_mask=foreground_mask,
                    source_wh=(source_w, source_h),
                    config=config,
                )
            )
            accepted = not rejection_reasons
            confidence = float(chosen.score)
            bbox = chosen.bbox_xyxy_px
            center = chosen.center_xy_px
            roi_requested = roi_geometry["requested_xyxy"]
            roi_source = roi_geometry["source_xyxy"]
            roi_insert = roi_geometry["insert_xyxy"]
            roi_clipped = bool(clip_max > 0)
            roi_clip_amount = roi_geometry["clip_amount"]
        else:
            accepted = False
            confidence = 0.0
            bbox = None
            center = None
            roi_requested = None
            roi_source = None
            roi_insert = None
            roi_clipped = False
            roi_clip_amount = {}
            content_fraction = 0.0

        overlays = _locator_overlays(
            gray=gray,
            candidates=candidates,
            chosen=chosen,
            roi_requested_xyxy=roi_requested,
        )
        artifacts = contracts.LocatorDebugArtifacts(
            paths=_write_locator_artifacts(
                request=request,
                image_bytes=image_bytes,
                gray=gray,
                background=background,
                diff=diff,
                foreground_mask=foreground_mask,
                edge_map=edge_map,
                candidate_overlay=overlays["candidate_overlay"],
                chosen_overlay=overlays["chosen_overlay"],
                roi_crop=_roi_crop(gray, roi_source, roi_insert, config),
                metadata={
                    "locator_kind": self.locator_kind.value,
                    "locator_parameters": config.to_dict(),
                    "runtime_parameter_revision": int(revision),
                    "candidate_count": len(candidates),
                    "accepted_candidate_count": len(accepted_candidates),
                    "roi_content_fraction": float(content_fraction),
                    contracts.PREPROCESSING_METADATA_APPLY_BACKGROUND_REMOVAL_TO_ROI_LOCATOR: (
                        _apply_background_removal_to_locator(request)
                    ),
                    contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED_TO_ROI_LOCATOR: background_applied,
                    "manual_ignore_mask_applied": manual_ignore_mask is not None,
                    "manual_ignore_mask_pixel_count": (
                        int(np.count_nonzero(manual_ignore_mask))
                        if manual_ignore_mask is not None
                        else 0
                    ),
                    "manual_ignore_mask_revision": request.extras.get(
                        "manual_ignore_mask_revision"
                    ),
                    "warnings": tuple(warnings),
                    "rejection_reasons": tuple(rejection_reasons),
                },
            ),
            metadata={
                "locator_kind": self.locator_kind.value,
                "locator_parameters": config.to_dict(),
                "runtime_parameter_revision": int(revision),
                "candidate_count": len(candidates),
                "accepted_candidate_count": len(accepted_candidates),
                "roi_content_fraction": float(content_fraction),
                contracts.PREPROCESSING_METADATA_APPLY_BACKGROUND_REMOVAL_TO_ROI_LOCATOR: (
                    _apply_background_removal_to_locator(request)
                ),
                contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED_TO_ROI_LOCATOR: background_applied,
                "manual_ignore_mask_applied": manual_ignore_mask is not None,
                "manual_ignore_mask_pixel_count": (
                    int(np.count_nonzero(manual_ignore_mask))
                    if manual_ignore_mask is not None
                    else 0
                ),
                "manual_ignore_mask_revision": request.extras.get(
                    "manual_ignore_mask_revision"
                ),
            },
        )
        result = contracts.LocatorResult(
            request_id=request.request_id,
            locator_kind=self.locator_kind,
            accepted=bool(accepted),
            confidence=confidence,
            source_image_wh_px=(source_w, source_h),
            chosen_candidate=chosen,
            candidates=tuple(candidates),
            bbox_xyxy_px=bbox,
            center_xy_px=center,
            roi_requested_xyxy_px=roi_requested,
            roi_source_xyxy_px=roi_source,
            roi_canvas_insert_xyxy_px=roi_insert,
            roi_clipped=roi_clipped,
            roi_clip_amount_px=roi_clip_amount,
            roi_rejection_reasons=tuple(rejection_reasons),
            debug_artifacts=artifacts,
            warnings=tuple(warnings),
            extras={
                "locator_parameters": config.to_dict(),
                "runtime_parameter_revision": int(revision),
                "background_revision": _snapshot_revision(background),
                contracts.PREPROCESSING_METADATA_APPLY_BACKGROUND_REMOVAL_TO_ROI_LOCATOR: (
                    _apply_background_removal_to_locator(request)
                ),
                contracts.PREPROCESSING_METADATA_BACKGROUND_REMOVAL_APPLIED_TO_ROI_LOCATOR: background_applied,
                "roi_content_fraction": float(content_fraction),
                "manual_ignore_mask_applied": manual_ignore_mask is not None,
                "manual_ignore_mask_pixel_count": (
                    int(np.count_nonzero(manual_ignore_mask))
                    if manual_ignore_mask is not None
                    else 0
                ),
                "manual_ignore_mask_revision": request.extras.get(
                    "manual_ignore_mask_revision"
                ),
            },
        )
        return _write_locator_result_json(request, result)

    def _background_snapshot(
        self,
        config: BackgroundEdgeLocatorConfig,
        source_w: int,
        source_h: int,
        warnings: list[str],
        request: contracts.LocatorRequest,
    ) -> BackgroundSnapshot | None:
        if not _apply_background_removal_to_locator(request):
            return None
        if self._background_state is None:
            return None
        snapshot = self._background_state.get_snapshot()
        if snapshot is None or not snapshot.captured or not snapshot.enabled:
            return None
        if not snapshot.dimensions_match(source_w, source_h):
            warnings.append(
                "background removal skipped for locator: background size "
                f"{(snapshot.width_px, snapshot.height_px)} does not match source "
                f"image size {(source_w, source_h)}."
            )
            return None
        if int(snapshot.threshold) != int(config.background_threshold):
            self._background_state.set_threshold(config.background_threshold)
            snapshot = self._background_state.get_snapshot()
        return snapshot


class FixedCenterRoiLocator:
    """Smoke-test fallback locator that always chooses the frame centre."""

    def __init__(self, *, roi_wh_px: tuple[int, int] = (300, 300)) -> None:
        self._roi_wh_px = (max(1, int(roi_wh_px[0])), max(1, int(roi_wh_px[1])))

    @property
    def locator_kind(self) -> contracts.LocatorKind:
        return contracts.LocatorKind.FIXED_CENTER_ROI

    def locate(
        self,
        request: contracts.LocatorRequest,
        image_bytes: bytes,
    ) -> contracts.LocatorResult:
        gray = _decode_image_bytes_to_grayscale(image_bytes)
        source_h, source_w = int(gray.shape[0]), int(gray.shape[1])
        center = (float(source_w) / 2.0, float(source_h) / 2.0)
        return _fixed_result(
            request=request,
            gray=gray,
            center_xy=center,
            roi_wh=self._roi_wh_px,
            locator_kind=self.locator_kind,
            label="fixed center ROI",
        )


class ManualFixedRoiLocator:
    """Emergency fallback locator using a supplied fixed bbox or centre."""

    def __init__(
        self,
        *,
        bbox_xyxy_px: tuple[float, float, float, float] | None = None,
        center_xy_px: tuple[float, float] | None = None,
        roi_wh_px: tuple[int, int] = (300, 300),
    ) -> None:
        self._bbox_xyxy_px = bbox_xyxy_px
        self._center_xy_px = center_xy_px
        self._roi_wh_px = (max(1, int(roi_wh_px[0])), max(1, int(roi_wh_px[1])))

    @property
    def locator_kind(self) -> contracts.LocatorKind:
        return contracts.LocatorKind.MANUAL_FIXED_ROI

    def locate(
        self,
        request: contracts.LocatorRequest,
        image_bytes: bytes,
    ) -> contracts.LocatorResult:
        gray = _decode_image_bytes_to_grayscale(image_bytes)
        source_h, source_w = int(gray.shape[0]), int(gray.shape[1])
        if self._center_xy_px is not None:
            center = (float(self._center_xy_px[0]), float(self._center_xy_px[1]))
        elif self._bbox_xyxy_px is not None:
            assert self._bbox_xyxy_px is not None
            x1, y1, x2, y2 = self._bbox_xyxy_px
            center = ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
        else:
            center = (float(source_w) / 2.0, float(source_h) / 2.0)
        return _fixed_result(
            request=request,
            gray=gray,
            center_xy=center,
            roi_wh=self._roi_wh_px,
            locator_kind=self.locator_kind,
            label="manual fixed ROI",
            bbox_xyxy_px=self._bbox_xyxy_px,
        )


class RoiFcnLegacyLocatorAdapter:
    """Generic contract adapter around the retained legacy ROI-FCN locator."""

    def __init__(self, legacy_locator: Any) -> None:
        self._legacy_locator = legacy_locator

    @property
    def locator_kind(self) -> contracts.LocatorKind:
        return contracts.LocatorKind.ROI_FCN_LEGACY

    def locate(
        self,
        request: contracts.LocatorRequest,
        image_bytes: bytes,
    ) -> contracts.LocatorResult:
        gray = _decode_image_bytes_to_grayscale(image_bytes)
        location = self._legacy_locator.locate(gray)
        center = tuple(float(value) for value in location.center_xy_px)
        metadata = dict(getattr(location, "metadata", {}) or {})
        roi_wh = _roi_wh_from_request(request, default=(300, 300))
        source_h, source_w = int(gray.shape[0]), int(gray.shape[1])
        roi_geometry = _roi_geometry(
            center_xy=center,
            source_wh=(source_w, source_h),
            roi_wh=roi_wh,
        )
        bbox = _xyxy_or_none(getattr(location, "roi_bounds_xyxy_px", None))
        confidence = _float_or_none(
            metadata.get("heatmap_peak_confidence")
            or metadata.get("confidence")
            or _mapping(metadata.get("decoded_heatmap")).get("confidence")
        )
        candidate = contracts.RoiCandidate(
            candidate_id="roi_fcn_legacy_0",
            bbox_xyxy_px=bbox or roi_geometry["requested_xyxy"],
            center_xy_px=center,
            area_px=float(roi_wh[0] * roi_wh[1]),
            contour_area_px=0.0,
            bbox_area_px=float(roi_wh[0] * roi_wh[1]),
            aspect_ratio=float(roi_wh[0]) / float(max(1, roi_wh[1])),
            score=float(confidence if confidence is not None else 1.0),
            extras=metadata,
        )
        result = contracts.LocatorResult(
            request_id=request.request_id,
            locator_kind=self.locator_kind,
            accepted=True,
            confidence=candidate.score,
            source_image_wh_px=(source_w, source_h),
            chosen_candidate=candidate,
            candidates=(candidate,),
            bbox_xyxy_px=bbox,
            center_xy_px=center,
            roi_requested_xyxy_px=roi_geometry["requested_xyxy"],
            roi_source_xyxy_px=roi_geometry["source_xyxy"],
            roi_canvas_insert_xyxy_px=roi_geometry["insert_xyxy"],
            roi_clipped=max(roi_geometry["clip_amount"].values()) > 0,
            roi_clip_amount_px=roi_geometry["clip_amount"],
            debug_artifacts=contracts.LocatorDebugArtifacts(metadata=metadata),
            extras={"legacy_roi_fcn_metadata": metadata},
        )
        return _write_locator_result_json(request, result)


def build_locator(
    locator_kind: contracts.LocatorKind | str,
    *,
    background_state: BackgroundState | None = None,
    parameter_state: LocatorRuntimeParameterState | None = None,
    roi_wh_px: tuple[int, int] = (300, 300),
    legacy_locator: Any | None = None,
) -> contracts.RoiLocator:
    """Construct a supported locator implementation."""
    kind = (
        locator_kind
        if isinstance(locator_kind, contracts.LocatorKind)
        else contracts.LocatorKind(str(locator_kind))
    )
    if kind == contracts.LocatorKind.BACKGROUND_EDGE_V1:
        config = BackgroundEdgeLocatorConfig(
            roi_width_px=int(roi_wh_px[0]),
            roi_height_px=int(roi_wh_px[1]),
        )
        return BackgroundEdgeLocator(
            background_state=background_state,
            parameter_state=parameter_state,
            config=config,
        )
    if kind == contracts.LocatorKind.FIXED_CENTER_ROI:
        return FixedCenterRoiLocator(roi_wh_px=roi_wh_px)
    if kind == contracts.LocatorKind.MANUAL_FIXED_ROI:
        return ManualFixedRoiLocator(roi_wh_px=roi_wh_px)
    if kind == contracts.LocatorKind.ROI_FCN_LEGACY and legacy_locator is not None:
        return RoiFcnLegacyLocatorAdapter(legacy_locator)
    raise ValueError(f"Cannot build locator kind {kind.value!r}.")


def _apply_background_removal_to_locator(request: contracts.LocatorRequest) -> bool:
    return bool(
        request.extras.get(
            contracts.PREPROCESSING_METADATA_APPLY_BACKGROUND_REMOVAL_TO_ROI_LOCATOR,
            True,
        )
    )


def _background_removal_explicitly_requested(
    request: contracts.LocatorRequest,
) -> bool:
    return (
        request.extras.get(
            contracts.PREPROCESSING_METADATA_APPLY_BACKGROUND_REMOVAL_TO_ROI_LOCATOR
        )
        is True
    )


def _config_from_request(
    request: contracts.LocatorRequest,
    config: BackgroundEdgeLocatorConfig,
) -> BackgroundEdgeLocatorConfig:
    extras = _mapping(request.extras)
    params = _mapping(extras.get("locator_parameters"))
    if not params:
        return config
    allowed = set(BackgroundEdgeLocatorConfig.__dataclass_fields__)
    updates = {key: params[key] for key in params if key in allowed}
    return replace(config, **updates)


def _manual_ignore_mask_from_request(
    request: contracts.LocatorRequest,
    *,
    source_w: int,
    source_h: int,
    warnings: list[str],
) -> np.ndarray | None:
    raw_mask = request.extras.get("manual_ignore_mask")
    if raw_mask is None:
        return None
    mask = np.asarray(raw_mask, dtype=bool)
    expected_shape = (int(source_h), int(source_w))
    if mask.shape != expected_shape:
        warnings.append(
            "manual frame mask skipped for locator: mask shape "
            f"{mask.shape} does not match source image shape {expected_shape}."
        )
        return None
    if not bool(np.any(mask)):
        return None
    return np.array(mask, dtype=bool, copy=True)


def _foreground_mask(
    gray: np.ndarray,
    background: BackgroundSnapshot | None,
    config: BackgroundEdgeLocatorConfig,
    warnings: list[str],
) -> tuple[np.ndarray | None, np.ndarray]:
    if background is not None and background.captured and background.enabled:
        diff = cv2.absdiff(gray, background.grayscale_background)
        mask = diff >= int(background.threshold)
        return diff, mask

    warnings.append(
        "No enabled matching background is available; using dark-on-light frame heuristic."
    )
    diff = None
    mask = gray < 245
    return diff, mask


def _morphology_mask(mask: np.ndarray, config: BackgroundEdgeLocatorConfig) -> np.ndarray:
    output = np.asarray(mask, dtype=np.uint8)
    close_kernel = _odd_kernel(config.morphology_close_kernel_px)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    dilate_kernel = _odd_kernel(config.dilation_kernel_px)
    if dilate_kernel > 1:
        kernel = np.ones((dilate_kernel, dilate_kernel), dtype=np.uint8)
        output = cv2.dilate(output, kernel, iterations=1)
    return output.astype(bool)


def _edge_map(
    gray: np.ndarray,
    foreground_mask: np.ndarray,
    config: BackgroundEdgeLocatorConfig,
) -> np.ndarray:
    masked = np.full(gray.shape, 255, dtype=np.uint8)
    masked[foreground_mask] = gray[foreground_mask]
    edges = cv2.Canny(
        masked,
        int(config.canny_low_threshold),
        int(config.canny_high_threshold),
    )
    edges[~foreground_mask] = 0
    return np.ascontiguousarray(edges)


def _build_candidates(
    *,
    edge_map: np.ndarray,
    foreground_mask: np.ndarray,
    source_wh: tuple[int, int],
    config: BackgroundEdgeLocatorConfig,
    prefer_components: bool = True,
) -> tuple[contracts.RoiCandidate, ...]:
    component_candidates = _build_foreground_component_candidates(
        edge_map=edge_map,
        foreground_mask=foreground_mask,
        source_wh=source_wh,
        config=config,
    )
    accepted_components = tuple(
        candidate for candidate in component_candidates if not candidate.rejection_reason
    )
    if prefer_components and accepted_components:
        return tuple(
            sorted(component_candidates, key=lambda item: item.score, reverse=True)
        )

    edge_candidates = _build_edge_contour_candidates(
        edge_map=edge_map,
        foreground_mask=foreground_mask,
        source_wh=source_wh,
        config=config,
    )
    return tuple(
        sorted(
            (*component_candidates, *edge_candidates),
            key=lambda item: item.score,
            reverse=True,
        )
    )


def _build_foreground_component_candidates(
    *,
    edge_map: np.ndarray,
    foreground_mask: np.ndarray,
    source_wh: tuple[int, int],
    config: BackgroundEdgeLocatorConfig,
) -> tuple[contracts.RoiCandidate, ...]:
    frame_w, frame_h = source_wh
    frame_area = float(max(1, frame_w * frame_h))
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        np.asarray(foreground_mask, dtype=np.uint8),
        8,
    )
    candidates: list[contracts.RoiCandidate] = []
    for label in range(1, int(component_count)):
        x, y, w, h, area = [int(value) for value in stats[label]]
        bbox_area = float(max(0, w * h))
        area_px = float(max(0, area))
        aspect = float(w) / float(max(1, h))
        edge_density = float(np.count_nonzero(edge_map[y : y + h, x : x + w])) / float(
            max(1.0, bbox_area)
        )
        rejection_reason = _candidate_rejection_reason(
            area_px=area_px,
            bbox_area=bbox_area,
            aspect=aspect,
            x=x,
            y=y,
            w=w,
            h=h,
            frame_w=frame_w,
            frame_h=frame_h,
            config=config,
        )
        if rejection_reason is None:
            rejection_reason = _diffuse_component_rejection_reason(
                bbox_area=bbox_area,
                frame_area=frame_area,
                edge_density=edge_density,
            )
        score = _candidate_score(
            area_px=area_px,
            bbox_area=bbox_area,
            frame_area=frame_area,
            edge_density=edge_density,
            config=config,
        )
        if rejection_reason is None and score < float(config.min_candidate_score):
            rejection_reason = (
                f"{contracts.LocatorFailureReason.LOW_CONFIDENCE.value}:{score:.3f}"
            )
        candidates.append(
            contracts.RoiCandidate(
                candidate_id=f"component_{label:03d}",
                bbox_xyxy_px=(float(x), float(y), float(x + w), float(y + h)),
                center_xy_px=(float(x) + (float(w) / 2.0), float(y) + (float(h) / 2.0)),
                area_px=area_px,
                contour_area_px=area_px,
                bbox_area_px=bbox_area,
                aspect_ratio=aspect,
                score=score,
                rejection_reason=rejection_reason,
                extras={
                    contracts.ROI_CANDIDATE_SOURCE_FIELD: (
                        contracts.ROI_CANDIDATE_SOURCE_FOREGROUND_COMPONENT
                    ),
                    "edge_density": edge_density,
                },
            )
        )
    return tuple(candidates)


def _build_edge_contour_candidates(
    *,
    edge_map: np.ndarray,
    foreground_mask: np.ndarray,
    source_wh: tuple[int, int],
    config: BackgroundEdgeLocatorConfig,
) -> tuple[contracts.RoiCandidate, ...]:
    if not bool(np.any(edge_map)):
        return ()
    contour_source = edge_map
    contours, _hierarchy = cv2.findContours(
        contour_source,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates: list[contracts.RoiCandidate] = []
    frame_w, frame_h = source_wh
    frame_area = float(max(1, frame_w * frame_h))
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = float(max(0, w * h))
        contour_area = float(abs(cv2.contourArea(contour)))
        mask_area = int(np.count_nonzero(foreground_mask[y : y + h, x : x + w]))
        area_px = float(max(mask_area, contour_area, bbox_area))
        aspect = float(w) / float(max(1, h))
        edge_density = float(np.count_nonzero(edge_map[y : y + h, x : x + w])) / float(
            max(1.0, bbox_area)
        )
        rejection_reason = _candidate_rejection_reason(
            area_px=area_px,
            bbox_area=bbox_area,
            aspect=aspect,
            x=x,
            y=y,
            w=w,
            h=h,
            frame_w=frame_w,
            frame_h=frame_h,
            config=config,
        )
        score = _candidate_score(
            area_px=area_px,
            bbox_area=bbox_area,
            frame_area=frame_area,
            edge_density=edge_density,
            config=config,
        )
        if rejection_reason is None and score < float(config.min_candidate_score):
            rejection_reason = (
                f"{contracts.LocatorFailureReason.LOW_CONFIDENCE.value}:{score:.3f}"
            )
        candidates.append(
            contracts.RoiCandidate(
                candidate_id=f"edge_{index:03d}",
                bbox_xyxy_px=(float(x), float(y), float(x + w), float(y + h)),
                center_xy_px=(float(x) + (float(w) / 2.0), float(y) + (float(h) / 2.0)),
                area_px=area_px,
                contour_area_px=contour_area,
                bbox_area_px=bbox_area,
                aspect_ratio=aspect,
                score=score,
                rejection_reason=rejection_reason,
                extras={
                    contracts.ROI_CANDIDATE_SOURCE_FIELD: (
                        contracts.ROI_CANDIDATE_SOURCE_EDGE_CONTOUR
                    ),
                    "edge_density": edge_density,
                },
            )
        )
    return tuple(candidates)


def _candidate_rejection_reason(
    *,
    area_px: float,
    bbox_area: float,
    aspect: float,
    x: int,
    y: int,
    w: int,
    h: int,
    frame_w: int,
    frame_h: int,
    config: BackgroundEdgeLocatorConfig,
) -> str | None:
    if area_px < float(config.min_foreground_area_px):
        return (
            f"{contracts.LocatorFailureReason.NO_CANDIDATES.value}:"
            f"area<{int(config.min_foreground_area_px)}"
        )
    if aspect < 0.15 or aspect > 8.0:
        return f"implausible_aspect_ratio:{aspect:.3f}"

    frame_area = float(max(1, int(frame_w) * int(frame_h)))
    bbox_fraction = float(bbox_area) / frame_area
    touches_left = int(x) <= 0
    touches_top = int(y) <= 0
    touches_right = int(x + w) >= int(frame_w)
    touches_bottom = int(y + h) >= int(frame_h)
    touched_borders = sum(
        int(value)
        for value in (touches_left, touches_top, touches_right, touches_bottom)
    )
    if bbox_fraction > 0.45:
        return f"implausibly_large_candidate:{bbox_fraction:.3f}"
    if touched_borders >= 2 and bbox_fraction > 0.20:
        return f"border_saturated_candidate:{bbox_fraction:.3f}"
    return None


def _diffuse_component_rejection_reason(
    *,
    bbox_area: float,
    frame_area: float,
    edge_density: float,
) -> str | None:
    bbox_fraction = float(bbox_area) / float(max(1.0, frame_area))
    if bbox_fraction > 0.15 and float(edge_density) < 0.01:
        return (
            "diffuse_large_component:"
            f"{bbox_fraction:.3f},edge_density:{float(edge_density):.4f}"
        )
    return None


def _candidate_roi_rejection_reasons(
    candidate: contracts.RoiCandidate,
    *,
    foreground_mask: np.ndarray,
    source_wh: tuple[int, int],
    config: BackgroundEdgeLocatorConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    roi_geometry = _roi_geometry(
        center_xy=candidate.center_xy_px,
        source_wh=source_wh,
        roi_wh=(config.roi_width_px, config.roi_height_px),
    )
    clip_max = max(
        (int(value) for value in roi_geometry["clip_amount"].values()),
        default=0,
    )
    content_fraction = _roi_content_fraction(
        foreground_mask,
        roi_geometry["source_xyxy"],
    )
    if candidate.score < config.min_candidate_score:
        reasons.append(
            f"{contracts.LocatorFailureReason.LOW_CONFIDENCE.value}:"
            f"{float(candidate.score):.3f}"
        )
    if clip_max > int(config.roi_clip_tolerance_px):
        reasons.append(contracts.LocatorFailureReason.ROI_CLIPPED.value)
    if content_fraction < float(config.min_roi_content_fraction):
        reasons.append(
            f"{contracts.LocatorFailureReason.ROI_CONTENT_TOO_LOW.value}:"
            f"{float(content_fraction):.4f}"
        )
    return tuple(reasons)


def _candidate_score(
    *,
    area_px: float,
    bbox_area: float,
    frame_area: float,
    edge_density: float,
    config: BackgroundEdgeLocatorConfig,
) -> float:
    area_score = min(
        1.0,
        float(area_px) / max(float(config.min_foreground_area_px) * 8.0, 1.0),
    )
    extent_score = min(1.0, (float(bbox_area) / max(1.0, frame_area)) * 80.0)
    density_score = min(1.0, float(edge_density) * 10.0)
    return max(
        0.0,
        min(1.0, 0.65 * area_score + 0.25 * extent_score + 0.10 * density_score),
    )


def _roi_geometry(
    *,
    center_xy: tuple[float, float],
    source_wh: tuple[int, int],
    roi_wh: tuple[int, int],
) -> dict[str, Any]:
    source_w, source_h = int(source_wh[0]), int(source_wh[1])
    roi_w, roi_h = int(roi_wh[0]), int(roi_wh[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    req_x1 = int(round(cx - (float(roi_w) / 2.0)))
    req_y1 = int(round(cy - (float(roi_h) / 2.0)))
    req_x2 = req_x1 + roi_w
    req_y2 = req_y1 + roi_h
    src_x1 = max(0, req_x1)
    src_y1 = max(0, req_y1)
    src_x2 = min(source_w, req_x2)
    src_y2 = min(source_h, req_y2)
    dst_x1 = src_x1 - req_x1
    dst_y1 = src_y1 - req_y1
    dst_x2 = dst_x1 + max(0, src_x2 - src_x1)
    dst_y2 = dst_y1 + max(0, src_y2 - src_y1)
    return {
        "requested_xyxy": (float(req_x1), float(req_y1), float(req_x2), float(req_y2)),
        "source_xyxy": (float(src_x1), float(src_y1), float(src_x2), float(src_y2)),
        "insert_xyxy": (float(dst_x1), float(dst_y1), float(dst_x2), float(dst_y2)),
        "clip_amount": {
            "left": max(0, src_x1 - req_x1),
            "top": max(0, src_y1 - req_y1),
            "right": max(0, req_x2 - src_x2),
            "bottom": max(0, req_y2 - src_y2),
        },
    }


def _roi_content_fraction(
    mask: np.ndarray,
    roi_source_xyxy: tuple[float, float, float, float],
) -> float:
    x1, y1, x2, y2 = [int(round(value)) for value in roi_source_xyxy]
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    return float(np.count_nonzero(roi)) / float(max(1, roi.size))


def _locator_overlays(
    *,
    gray: np.ndarray,
    candidates: tuple[contracts.RoiCandidate, ...],
    chosen: contracts.RoiCandidate | None,
    roi_requested_xyxy: tuple[float, float, float, float] | None,
) -> dict[str, np.ndarray]:
    candidate_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for candidate in candidates:
        color = (32, 180, 32) if not candidate.rejection_reason else (60, 150, 255)
        _draw_xyxy(candidate_overlay, candidate.bbox_xyxy_px, color=color, thickness=1)
    chosen_overlay = np.array(candidate_overlay, copy=True)
    if chosen is not None:
        _draw_xyxy(chosen_overlay, chosen.bbox_xyxy_px, color=(255, 90, 30), thickness=2)
        cv2.circle(
            chosen_overlay,
            (int(round(chosen.center_xy_px[0])), int(round(chosen.center_xy_px[1]))),
            5,
            (255, 90, 30),
            thickness=-1,
        )
    if roi_requested_xyxy is not None:
        _draw_xyxy(chosen_overlay, roi_requested_xyxy, color=(255, 255, 30), thickness=2)
    return {"candidate_overlay": candidate_overlay, "chosen_overlay": chosen_overlay}


def _roi_crop(
    gray: np.ndarray,
    roi_source_xyxy: tuple[float, float, float, float] | None,
    roi_insert_xyxy: tuple[float, float, float, float] | None,
    config: BackgroundEdgeLocatorConfig,
) -> np.ndarray | None:
    if roi_source_xyxy is None or roi_insert_xyxy is None:
        return None
    canvas = np.full((int(config.roi_height_px), int(config.roi_width_px)), 255, dtype=np.uint8)
    sx1, sy1, sx2, sy2 = [int(round(value)) for value in roi_source_xyxy]
    dx1, dy1, dx2, dy2 = [int(round(value)) for value in roi_insert_xyxy]
    if sx2 <= sx1 or sy2 <= sy1 or dx2 <= dx1 or dy2 <= dy1:
        return canvas
    canvas[dy1:dy2, dx1:dx2] = gray[sy1:sy2, sx1:sx2]
    return canvas


def _fixed_result(
    *,
    request: contracts.LocatorRequest,
    gray: np.ndarray,
    center_xy: tuple[float, float],
    roi_wh: tuple[int, int],
    locator_kind: contracts.LocatorKind,
    label: str,
    bbox_xyxy_px: tuple[float, float, float, float] | None = None,
) -> contracts.LocatorResult:
    source_h, source_w = int(gray.shape[0]), int(gray.shape[1])
    roi_geometry = _roi_geometry(center_xy=center_xy, source_wh=(source_w, source_h), roi_wh=roi_wh)
    bbox = bbox_xyxy_px or roi_geometry["requested_xyxy"]
    candidate = contracts.RoiCandidate(
        candidate_id="fixed_000",
        bbox_xyxy_px=bbox,
        center_xy_px=center_xy,
        area_px=float(roi_wh[0] * roi_wh[1]),
        contour_area_px=0.0,
        bbox_area_px=float(roi_wh[0] * roi_wh[1]),
        aspect_ratio=float(roi_wh[0]) / float(max(1, roi_wh[1])),
        score=1.0,
        extras={"label": label},
    )
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    _draw_xyxy(overlay, roi_geometry["requested_xyxy"], color=(255, 255, 30), thickness=2)
    artifacts = contracts.LocatorDebugArtifacts(
        paths=_write_locator_artifacts(
            request=request,
            image_bytes=b"",
            gray=gray,
            background=None,
            diff=None,
            foreground_mask=None,
            edge_map=None,
            candidate_overlay=overlay,
            chosen_overlay=overlay,
            roi_crop=_roi_crop(
                gray,
                roi_geometry["source_xyxy"],
                roi_geometry["insert_xyxy"],
                BackgroundEdgeLocatorConfig(roi_width_px=roi_wh[0], roi_height_px=roi_wh[1]),
            ),
            metadata={"locator_kind": locator_kind.value, "label": label},
        ),
        metadata={"label": label},
    )
    return _write_locator_result_json(
        request,
        contracts.LocatorResult(
            request_id=request.request_id,
            locator_kind=locator_kind,
            accepted=True,
            confidence=1.0,
            source_image_wh_px=(source_w, source_h),
            chosen_candidate=candidate,
            candidates=(candidate,),
            bbox_xyxy_px=bbox,
            center_xy_px=center_xy,
            roi_requested_xyxy_px=roi_geometry["requested_xyxy"],
            roi_source_xyxy_px=roi_geometry["source_xyxy"],
            roi_canvas_insert_xyxy_px=roi_geometry["insert_xyxy"],
            roi_clipped=max(roi_geometry["clip_amount"].values()) > 0,
            roi_clip_amount_px=roi_geometry["clip_amount"],
            debug_artifacts=artifacts,
        ),
    )


def _write_locator_artifacts(
    *,
    request: contracts.LocatorRequest,
    image_bytes: bytes,
    gray: np.ndarray,
    background: BackgroundSnapshot | None,
    diff: np.ndarray | None,
    foreground_mask: np.ndarray | None,
    edge_map: np.ndarray | None,
    candidate_overlay: np.ndarray | None,
    chosen_overlay: np.ndarray | None,
    roi_crop: np.ndarray | None,
    metadata: Mapping[str, Any],
) -> dict[str, Path]:
    if not bool(request.save_debug_images):
        return {}
    output_dir = Path(request.debug_output_dir or contracts.DEFAULT_DEBUG_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _artifact_prefix(request)
    paths: dict[str, Path] = {}
    artifacts: dict[str, Any] = {
        contracts.DISPLAY_ARTIFACT_ACCEPTED_RAW_FRAME: gray,
        contracts.DISPLAY_ARTIFACT_GRAYSCALE_FRAME: gray,
        contracts.DISPLAY_ARTIFACT_BACKGROUND_FRAME: (
            background.grayscale_background
            if background is not None and background.captured
            else None
        ),
        contracts.DISPLAY_ARTIFACT_BACKGROUND_DIFF: diff,
        contracts.DISPLAY_ARTIFACT_FOREGROUND_MASK: foreground_mask,
        contracts.DISPLAY_ARTIFACT_EDGE_MAP: edge_map,
        contracts.DISPLAY_ARTIFACT_CANDIDATE_CONTOURS: candidate_overlay,
        contracts.DISPLAY_ARTIFACT_CHOSEN_CONTOUR: chosen_overlay,
        contracts.DISPLAY_ARTIFACT_LOCATOR_OVERLAY: chosen_overlay,
        contracts.DISPLAY_ARTIFACT_ROI_CROP: roi_crop,
    }
    for kind, image in artifacts.items():
        if image is None:
            continue
        path = output_dir / f"{prefix}__{_safe_filename(kind)}.png"
        _write_image(path, image)
        paths[str(kind)] = path
    metadata_path = output_dir / f"{prefix}__{contracts.DISPLAY_ARTIFACT_ROI_OVERLAY_METADATA}.json"
    _write_json(metadata_path, {"created_at_utc": _utc_now_iso(), **dict(metadata)})
    paths[contracts.DISPLAY_ARTIFACT_ROI_OVERLAY_METADATA] = metadata_path
    if image_bytes:
        raw_path = output_dir / f"{prefix}__accepted_raw_frame_bytes.bin"
        raw_path.write_bytes(bytes(image_bytes))
    return paths


def _write_locator_result_json(
    request: contracts.LocatorRequest,
    result: contracts.LocatorResult,
) -> contracts.LocatorResult:
    if not bool(request.save_debug_images):
        return result
    output_dir = Path(request.debug_output_dir or contracts.DEFAULT_DEBUG_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_artifact_prefix(request)}__locator_result.json"
    payload = result.to_dict()
    _write_json(path, payload)
    paths = dict(result.debug_artifacts.paths)
    paths["locator_result"] = path
    artifacts = replace(result.debug_artifacts, paths=paths)
    return replace(result, debug_artifacts=artifacts)


def _artifact_prefix(request: contracts.LocatorRequest) -> str:
    frame_hash = request.frame.frame_hash.value if request.frame.frame_hash is not None else "nohash"
    return f"{_safe_filename(request.request_id)}__{_safe_filename(frame_hash[:12])}"


def _decode_image_bytes_to_grayscale(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("Could not decode image bytes: payload is empty.")
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Could not decode image bytes as a supported image.")
    if decoded.ndim == 2:
        gray = decoded
    elif decoded.ndim == 3 and int(decoded.shape[2]) == 4:
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGRA2GRAY)
    elif decoded.ndim == 3:
        gray = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported decoded image shape: {decoded.shape}")
    return np.ascontiguousarray(np.asarray(gray, dtype=np.uint8))


def _write_image(path: Path, image: Any) -> None:
    array = _image_uint8(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Failed to write locator debug image: {path}")


def _image_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array.astype(np.uint8) * 255)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    numeric = np.asarray(array, dtype=np.float32)
    finite_max = float(np.nanmax(numeric)) if numeric.size else 0.0
    if finite_max <= 1.0:
        numeric *= 255.0
    return np.ascontiguousarray(np.clip(np.nan_to_num(numeric), 0, 255).astype(np.uint8))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _json_safe(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _safe_filename(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip(".-") or "artifact"


def _draw_xyxy(
    image: np.ndarray,
    xyxy: tuple[float, float, float, float],
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=thickness)


def _odd_kernel(value: int) -> int:
    size = max(0, int(value))
    if size <= 1:
        return 0
    return size if size % 2 == 1 else size + 1


def _snapshot_revision(snapshot: BackgroundSnapshot | None) -> int | None:
    if snapshot is None or not snapshot.captured:
        return None
    return int(snapshot.revision)


def _roi_wh_from_request(
    request: contracts.LocatorRequest,
    *,
    default: tuple[int, int],
) -> tuple[int, int]:
    params = _mapping(_mapping(request.extras).get("locator_parameters"))
    width = int(params.get("roi_width_px", default[0]))
    height = int(params.get("roi_height_px", default[1]))
    return max(1, width), max(1, height)


def _xyxy_or_none(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "BackgroundEdgeLocator",
    "BackgroundEdgeLocatorConfig",
    "FixedCenterRoiLocator",
    "LocatorRuntimeParameterState",
    "ManualFixedRoiLocator",
    "RoiFcnLegacyLocatorAdapter",
    "build_locator",
]
