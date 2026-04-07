"""Contour and convex-hull implementations for v2 silhouette generation."""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import GeneratorOutput


class ContourSilhouetteGeneratorV1:
    """Legacy contour-first extractor kept intact for comparisons and backward compatibility."""

    generator_id = "silhouette.contour_v1"

    def generate(
        self,
        source_gray: np.ndarray,
        *,
        blur_kernel_size: int,
        canny_low_threshold: int,
        canny_high_threshold: int,
        close_kernel_size: int,
        dilate_kernel_size: int,
        min_component_area_px: int,
        fill_holes: bool,
        source_bgr: np.ndarray | None = None,
        experimental_params: dict[str, object] | None = None,
    ) -> GeneratorOutput:
        del source_bgr, experimental_params
        del fill_holes

        if source_gray.ndim != 2:
            raise ValueError("source_gray must be 2D grayscale")

        processed = source_gray
        if blur_kernel_size > 1:
            processed = cv2.GaussianBlur(processed, (blur_kernel_size, blur_kernel_size), 0)

        edges = cv2.Canny(processed, int(canny_low_threshold), int(canny_high_threshold))
        raw_binary = np.where(edges > 0, 255, 0).astype(np.uint8)
        cleaned = raw_binary.copy()

        if close_kernel_size > 1:
            close_kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

        if dilate_kernel_size > 1:
            dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
            cleaned = cv2.dilate(cleaned, dilate_kernel, iterations=1)

        component_count, labels, stats = _component_stats(cleaned)
        total_components = max(0, component_count - 1)
        passing_labels = [
            label
            for label in range(1, component_count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_component_area_px)
        ]

        component_union = np.zeros_like(cleaned, dtype=np.uint8)
        for label in passing_labels:
            component_union[labels == label] = 255

        component_mask = _largest_valid_component(cleaned, min_component_area_px)
        if component_mask is None:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=cleaned,
                contour=None,
                primary_reason="no_valid_component",
                debug_images={
                    "raw_edge": _black_on_white(raw_binary),
                    "post_morph": _black_on_white(cleaned),
                    "components_before_selection": _black_on_white(component_union),
                },
                diagnostics={
                    "num_components_total": total_components,
                    "num_components_after_filter": len(passing_labels),
                },
            )

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=component_mask,
                contour=None,
                primary_reason="no_contours",
                debug_images={
                    "raw_edge": _black_on_white(raw_binary),
                    "post_morph": _black_on_white(cleaned),
                    "components_before_selection": _black_on_white(component_union),
                    "selected_component": _black_on_white(component_mask),
                },
                diagnostics={
                    "num_components_total": total_components,
                    "num_components_after_filter": len(passing_labels),
                },
            )

        contour = max(contours, key=lambda value: float(abs(cv2.contourArea(value))))
        if contour is None or contour.size == 0:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=component_mask,
                contour=None,
                primary_reason="empty_contour",
                debug_images={
                    "raw_edge": _black_on_white(raw_binary),
                    "post_morph": _black_on_white(cleaned),
                    "components_before_selection": _black_on_white(component_union),
                    "selected_component": _black_on_white(component_mask),
                },
                diagnostics={
                    "num_components_total": total_components,
                    "num_components_after_filter": len(passing_labels),
                },
            )

        if contour.shape[0] < 3:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=component_mask,
                contour=None,
                primary_reason="degenerate_contour",
                debug_images={
                    "raw_edge": _black_on_white(raw_binary),
                    "post_morph": _black_on_white(cleaned),
                    "components_before_selection": _black_on_white(component_union),
                    "selected_component": _black_on_white(component_mask),
                },
                diagnostics={
                    "num_components_total": total_components,
                    "num_components_after_filter": len(passing_labels),
                },
            )

        area = float(abs(cv2.contourArea(contour)))
        if area < 1.0:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=component_mask,
                contour=None,
                primary_reason="degenerate_contour_area",
                debug_images={
                    "raw_edge": _black_on_white(raw_binary),
                    "post_morph": _black_on_white(cleaned),
                    "components_before_selection": _black_on_white(component_union),
                    "selected_component": _black_on_white(component_mask),
                },
                diagnostics={
                    "num_components_total": total_components,
                    "num_components_after_filter": len(passing_labels),
                },
            )

        selected_area = int(np.count_nonzero(component_mask))
        selected_bbox = _bbox_for_mask(component_mask)
        contour_bbox = _bbox_for_contour(contour)
        filled_mask = _filled_mask_from_contour(component_mask.shape, contour)

        return GeneratorOutput(
            edge_binary=cleaned,
            fallback_mask=component_mask,
            contour=contour,
            primary_reason="",
            debug_images={
                "raw_edge": _black_on_white(raw_binary),
                "post_morph": _black_on_white(cleaned),
                "components_before_selection": _black_on_white(component_union),
                "selected_component": _black_on_white(component_mask),
                "external_contour": _render_contour_outline(component_mask.shape, contour),
                "final_filled": _black_on_white(filled_mask),
            },
            diagnostics={
                "num_components_total": total_components,
                "num_components_after_filter": len(passing_labels),
                "selected_component_area": selected_area,
                "selected_component_bbox": selected_bbox,
                "contour_area": int(round(area)),
                "contour_bbox": contour_bbox,
            },
        )


class ContourSilhouetteGeneratorV2:
    """Main-blob + external-contour-only generator biased toward stable outer silhouettes."""

    generator_id = "silhouette.contour_v2"

    def generate(
        self,
        source_gray: np.ndarray,
        *,
        blur_kernel_size: int,
        canny_low_threshold: int,
        canny_high_threshold: int,
        close_kernel_size: int,
        dilate_kernel_size: int,
        min_component_area_px: int,
        fill_holes: bool,
        source_bgr: np.ndarray | None = None,
        experimental_params: dict[str, object] | None = None,
    ) -> GeneratorOutput:
        del source_bgr, experimental_params
        if source_gray.ndim != 2:
            raise ValueError("source_gray must be 2D grayscale")

        processed = source_gray
        if blur_kernel_size > 1:
            processed = cv2.GaussianBlur(processed, (blur_kernel_size, blur_kernel_size), 0)

        edges = cv2.Canny(processed, int(canny_low_threshold), int(canny_high_threshold))
        raw_binary = np.where(edges > 0, 255, 0).astype(np.uint8)
        post_morph = raw_binary.copy()

        if close_kernel_size > 1:
            close_kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
            post_morph = cv2.morphologyEx(post_morph, cv2.MORPH_CLOSE, close_kernel)

        if dilate_kernel_size > 1:
            dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
            post_morph = cv2.dilate(post_morph, dilate_kernel, iterations=1)

        component_count, labels, stats = _component_stats(post_morph)
        passing_labels = [
            label
            for label in range(1, component_count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_component_area_px)
        ]

        components_union = np.zeros_like(post_morph, dtype=np.uint8)
        for label in passing_labels:
            components_union[labels == label] = 255

        diagnostics: dict[str, object] = {
            "num_components_total": max(0, component_count - 1),
            "num_components_after_filter": len(passing_labels),
            "selected_component_area": "",
            "selected_component_bbox": "",
            "contour_area": "",
            "contour_bbox": "",
        }

        debug_images: dict[str, np.ndarray] = {
            "raw_edge": _black_on_white(raw_binary),
            "post_morph": _black_on_white(post_morph),
            "components_before_selection": _black_on_white(components_union),
        }

        if not passing_labels:
            return GeneratorOutput(
                edge_binary=post_morph,
                fallback_mask=post_morph,
                contour=None,
                primary_reason="no_valid_component",
                debug_images=debug_images,
                quality_flags=("no_component_after_filter",),
                diagnostics=diagnostics,
            )

        largest_label = max(
            passing_labels,
            key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        )
        selected_mask = np.zeros_like(post_morph, dtype=np.uint8)
        selected_mask[labels == largest_label] = 255

        selected_area = int(stats[largest_label, cv2.CC_STAT_AREA])
        selected_bbox = _bbox_for_component_stats(stats[largest_label, :])
        diagnostics["selected_component_area"] = selected_area
        diagnostics["selected_component_bbox"] = selected_bbox
        debug_images["selected_component"] = _black_on_white(selected_mask)

        contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GeneratorOutput(
                edge_binary=post_morph,
                fallback_mask=selected_mask,
                contour=None,
                primary_reason="no_external_contours",
                debug_images=debug_images,
                quality_flags=("no_external_contours",),
                diagnostics=diagnostics,
            )

        contour = max(contours, key=lambda value: float(abs(cv2.contourArea(value))))
        quality_flags, contour_area, contour_bbox, filled_area = _evaluate_contour_quality(
            contour,
            image_shape=source_gray.shape,
            selected_component_area=selected_area,
            min_component_area_px=int(min_component_area_px),
        )

        diagnostics["contour_area"] = contour_area
        diagnostics["contour_bbox"] = contour_bbox

        debug_images["external_contour"] = _render_contour_outline(source_gray.shape, contour)

        if fill_holes:
            final_mask = _filled_mask_from_contour(source_gray.shape, contour)
            recontours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if recontours:
                contour = max(recontours, key=lambda value: float(abs(cv2.contourArea(value))))
                contour_area = int(round(abs(float(cv2.contourArea(contour)))))
                contour_bbox = _bbox_for_contour(contour)
                diagnostics["contour_area"] = contour_area
                diagnostics["contour_bbox"] = contour_bbox
        else:
            final_mask = _filled_mask_from_contour(source_gray.shape, contour)

        debug_images["final_filled"] = _black_on_white(final_mask)

        if quality_flags:
            return GeneratorOutput(
                edge_binary=post_morph,
                fallback_mask=selected_mask,
                contour=None,
                primary_reason=f"poor_contour_{quality_flags[0]}",
                debug_images=debug_images,
                quality_flags=tuple(quality_flags),
                diagnostics=diagnostics
                | {
                    "filled_area": filled_area,
                    "filled_to_component_ratio": _safe_ratio(filled_area, selected_area),
                },
            )

        return GeneratorOutput(
            edge_binary=post_morph,
            fallback_mask=selected_mask,
            contour=contour,
            primary_reason="",
            debug_images=debug_images,
            diagnostics=diagnostics
            | {
                "filled_area": filled_area,
                "filled_to_component_ratio": _safe_ratio(filled_area, selected_area),
            },
        )


class BlobBackgroundSilhouetteGeneratorV1:
    """Experimental controlled-demo generator using foreground blob isolation."""

    generator_id = "silhouette.blob_bg_v1"

    def generate(
        self,
        source_gray: np.ndarray,
        *,
        blur_kernel_size: int,
        canny_low_threshold: int,
        canny_high_threshold: int,
        close_kernel_size: int,
        dilate_kernel_size: int,
        min_component_area_px: int,
        fill_holes: bool,
        source_bgr: np.ndarray | None = None,
        experimental_params: dict[str, object] | None = None,
    ) -> GeneratorOutput:
        del canny_low_threshold, canny_high_threshold

        if source_gray.ndim != 2:
            raise ValueError("source_gray must be 2D grayscale")

        params = dict(experimental_params or {})
        border_fraction = _safe_float(params.get("blob_border_fraction"), default=0.05)
        hue_delta = _safe_float(params.get("blob_hue_delta"), default=16.0)
        sat_min = _safe_float(params.get("blob_sat_min"), default=42.0)
        val_min = _safe_float(params.get("blob_val_min"), default=50.0)
        bright_val_min = _safe_float(params.get("blob_bright_val_min"), default=95.0)
        bright_sat_min = _safe_float(params.get("blob_bright_sat_min"), default=35.0)
        use_bright_rescue = _safe_bool(params.get("blob_use_bright_rescue"), default=True)
        reject_large_border_components = _safe_bool(
            params.get("blob_reject_large_border_components"),
            default=True,
        )
        large_border_component_ratio = _safe_float(
            params.get("blob_large_border_component_ratio"),
            default=0.01,
        )

        if source_bgr is not None and source_bgr.ndim == 3 and source_bgr.shape[2] >= 3:
            bgr = source_bgr[:, :, :3].copy()
        else:
            bgr = cv2.cvtColor(source_gray, cv2.COLOR_GRAY2BGR)

        if blur_kernel_size > 1:
            bgr = cv2.GaussianBlur(bgr, (blur_kernel_size, blur_kernel_size), 0)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)

        border_px = max(2, int(round(min(source_gray.shape) * max(0.01, border_fraction))))
        border_mask = np.zeros(source_gray.shape, dtype=bool)
        border_mask[:border_px, :] = True
        border_mask[-border_px:, :] = True
        border_mask[:, :border_px] = True
        border_mask[:, -border_px:] = True

        background_hue = int(np.median(h_channel[border_mask])) if np.any(border_mask) else 110
        hue_distance = _hue_distance(h_channel, background_hue)

        foreground_primary = (
            (hue_distance >= hue_delta)
            & (s_channel >= sat_min)
            & (v_channel >= val_min)
        )
        foreground_mask = foreground_primary.astype(np.uint8) * 255

        if use_bright_rescue:
            bright_rescue = ((v_channel >= bright_val_min) & (s_channel >= bright_sat_min)).astype(np.uint8) * 255
            foreground_mask = np.where((foreground_mask > 0) | (bright_rescue > 0), 255, 0).astype(np.uint8)

        post_morph = foreground_mask.copy()
        if close_kernel_size > 1:
            close_kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
            post_morph = cv2.morphologyEx(post_morph, cv2.MORPH_CLOSE, close_kernel)
        if dilate_kernel_size > 1:
            dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
            post_morph = cv2.dilate(post_morph, dilate_kernel, iterations=1)

        component_count, labels, stats = _component_stats(post_morph)
        passing_labels: list[int] = []
        image_area = int(source_gray.shape[0] * source_gray.shape[1])
        large_border_area_threshold = int(round(max(0.0, large_border_component_ratio) * image_area))

        for label in range(1, component_count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < int(min_component_area_px):
                continue

            if reject_large_border_components and area >= large_border_area_threshold:
                x = int(stats[label, cv2.CC_STAT_LEFT])
                y = int(stats[label, cv2.CC_STAT_TOP])
                w = int(stats[label, cv2.CC_STAT_WIDTH])
                h = int(stats[label, cv2.CC_STAT_HEIGHT])
                touches_border = (
                    x <= 0
                    or y <= 0
                    or (x + w) >= source_gray.shape[1]
                    or (y + h) >= source_gray.shape[0]
                )
                if touches_border:
                    continue

            passing_labels.append(label)

        components_union = np.zeros_like(post_morph, dtype=np.uint8)
        for label in passing_labels:
            components_union[labels == label] = 255

        diagnostics: dict[str, object] = {
            "num_components_total": max(0, component_count - 1),
            "num_components_after_filter": len(passing_labels),
            "selected_component_area": "",
            "selected_component_bbox": "",
            "contour_area": "",
            "contour_bbox": "",
            "blob_background_hue": background_hue,
        }
        debug_images: dict[str, np.ndarray] = {
            "raw_edge": _black_on_white(foreground_mask),
            "post_morph": _black_on_white(post_morph),
            "components_before_selection": _black_on_white(components_union),
        }

        if not passing_labels:
            return GeneratorOutput(
                edge_binary=post_morph,
                fallback_mask=post_morph,
                contour=None,
                primary_reason="no_valid_blob_component",
                debug_images=debug_images,
                quality_flags=("no_component_after_filter",),
                diagnostics=diagnostics,
            )

        selected_label = max(
            passing_labels,
            key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
        )
        selected_mask = np.zeros_like(post_morph, dtype=np.uint8)
        selected_mask[labels == selected_label] = 255
        if fill_holes:
            selected_mask = _fill_holes_binary(selected_mask)

        selected_area = int(np.count_nonzero(selected_mask))
        selected_bbox = _bbox_for_mask(selected_mask)
        diagnostics["selected_component_area"] = selected_area
        diagnostics["selected_component_bbox"] = selected_bbox
        debug_images["selected_component"] = _black_on_white(selected_mask)

        contours, _ = cv2.findContours(selected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GeneratorOutput(
                edge_binary=post_morph,
                fallback_mask=selected_mask,
                contour=None,
                primary_reason="no_external_contours",
                debug_images=debug_images,
                quality_flags=("no_external_contours",),
                diagnostics=diagnostics,
            )

        contour = max(contours, key=lambda value: float(abs(cv2.contourArea(value))))
        quality_flags, contour_area, contour_bbox, filled_area = _evaluate_contour_quality(
            contour,
            image_shape=source_gray.shape,
            selected_component_area=selected_area,
            min_component_area_px=int(min_component_area_px),
        )
        diagnostics["contour_area"] = contour_area
        diagnostics["contour_bbox"] = contour_bbox
        debug_images["external_contour"] = _render_contour_outline(source_gray.shape, contour)

        final_mask = _filled_mask_from_contour(source_gray.shape, contour)
        debug_images["final_filled"] = _black_on_white(final_mask)

        if quality_flags:
            return GeneratorOutput(
                edge_binary=post_morph,
                fallback_mask=selected_mask,
                contour=None,
                primary_reason=f"poor_blob_contour_{quality_flags[0]}",
                debug_images=debug_images,
                quality_flags=tuple(quality_flags),
                diagnostics=diagnostics
                | {
                    "filled_area": filled_area,
                    "filled_to_component_ratio": _safe_ratio(filled_area, selected_area),
                },
            )

        return GeneratorOutput(
            edge_binary=post_morph,
            fallback_mask=selected_mask,
            contour=contour,
            primary_reason="",
            debug_images=debug_images,
            diagnostics=diagnostics
            | {
                "filled_area": filled_area,
                "filled_to_component_ratio": _safe_ratio(filled_area, selected_area),
            },
        )


class ConvexHullFallbackV1:
    """Fallback that recovers one contour as a convex hull over edge points."""

    fallback_id = "fallback.convex_hull_v1"

    def recover(self, fallback_mask: np.ndarray) -> tuple[np.ndarray | None, str]:
        if fallback_mask.ndim != 2:
            return None, "invalid_fallback_mask"

        ys, xs = np.where(fallback_mask > 0)
        if len(xs) < 3:
            return None, "insufficient_points_for_hull"

        points = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(points)

        if hull is None or hull.size == 0:
            return None, "convex_hull_empty"

        if hull.shape[0] < 3:
            return None, "convex_hull_degenerate"

        area = float(abs(cv2.contourArea(hull)))
        if area < 1.0:
            return None, "convex_hull_zero_area"

        return hull, ""


class OutlineArtifactWriterV1:
    """Render contour outline on a white canvas."""

    writer_id = "artifact.outline_v1"
    representation_mode = "outline"

    def render(
        self,
        image_shape: tuple[int, int],
        contour: np.ndarray,
        *,
        line_thickness: int,
    ) -> np.ndarray:
        canvas = np.full(image_shape, 255, dtype=np.uint8)
        cv2.drawContours(
            canvas,
            [contour],
            contourIdx=-1,
            color=0,
            thickness=max(1, int(line_thickness)),
            lineType=cv2.LINE_AA,
        )
        return canvas


class FilledArtifactWriterV1:
    """Render filled contour on a white canvas."""

    writer_id = "artifact.filled_v1"
    representation_mode = "filled"

    def render(
        self,
        image_shape: tuple[int, int],
        contour: np.ndarray,
        *,
        line_thickness: int,
    ) -> np.ndarray:
        del line_thickness

        canvas = np.full(image_shape, 255, dtype=np.uint8)
        cv2.drawContours(
            canvas,
            [contour],
            contourIdx=-1,
            color=0,
            thickness=cv2.FILLED,
            lineType=cv2.LINE_8,
        )
        return canvas


def _largest_valid_component(binary_mask: np.ndarray, min_component_area_px: int) -> np.ndarray | None:
    labels_source = (binary_mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(labels_source, connectivity=8)
    if count <= 1:
        return None

    best_label = -1
    best_area = -1

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_component_area_px):
            continue
        if area > best_area:
            best_label = label
            best_area = area

    if best_label < 0:
        return None

    output = np.zeros_like(binary_mask, dtype=np.uint8)
    output[labels == best_label] = 255
    return output


def _component_stats(binary_mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    labels_source = (binary_mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(labels_source, connectivity=8)
    return int(count), labels, stats


def _evaluate_contour_quality(
    contour: np.ndarray | None,
    *,
    image_shape: tuple[int, int],
    selected_component_area: int,
    min_component_area_px: int,
) -> tuple[list[str], int, tuple[int, int, int, int], int]:
    if contour is None or contour.size == 0:
        return ["contour_missing"], 0, (0, 0, 0, 0), 0

    flags: list[str] = []
    contour_points = int(contour.shape[0]) if contour.ndim >= 2 else 0
    if contour_points < 3:
        flags.append("contour_too_few_points")

    contour_area = int(round(abs(float(cv2.contourArea(contour)))))
    if contour_area < int(min_component_area_px):
        flags.append("contour_area_below_min")

    x, y, w, h = cv2.boundingRect(contour)
    contour_bbox = (int(x), int(y), int(x + max(0, w - 1)), int(y + max(0, h - 1)))
    if w <= 1 or h <= 1:
        flags.append("contour_bbox_degenerate")

    filled_mask = _filled_mask_from_contour(image_shape, contour)
    filled_area = int(np.count_nonzero(filled_mask))
    filled_ratio = _safe_ratio(filled_area, selected_component_area)
    if filled_ratio < 0.20:
        flags.append("contour_fill_ratio_small")

    return flags, contour_area, contour_bbox, filled_area


def _filled_mask_from_contour(image_shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    filled = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    return filled


def _render_contour_outline(image_shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    canvas = np.full(image_shape, 255, dtype=np.uint8)
    cv2.drawContours(
        canvas,
        [contour],
        contourIdx=-1,
        color=0,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return canvas


def _black_on_white(mask: np.ndarray) -> np.ndarray:
    canvas = np.full(mask.shape, 255, dtype=np.uint8)
    canvas[mask > 0] = 0
    return canvas


def _bbox_for_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _bbox_for_contour(contour: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour)
    return (int(x), int(y), int(x + max(0, w - 1)), int(y + max(0, h - 1)))


def _bbox_for_component_stats(stats_row: np.ndarray) -> tuple[int, int, int, int]:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    return (x, y, x + max(0, w - 1), y + max(0, h - 1))


def _hue_distance(h_channel: np.ndarray, reference_hue: int) -> np.ndarray:
    distance = np.abs(h_channel.astype(np.int16) - int(reference_hue))
    return np.minimum(distance, 180 - distance)


def _fill_holes_binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        return mask

    flood = mask.copy()
    h, w = flood.shape
    flood_canvas = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_canvas, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, flood_inv)


def _safe_float(value: object, *, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    denom = float(denominator)
    if denom <= 0.0:
        return 0.0
    return float(numerator) / denom
