"""Silhouette contour generation and rendering primitives for v4."""

from __future__ import annotations

import cv2
import numpy as np

from .contracts import GeneratorOutput


class ContourSilhouetteGeneratorV2:
    """Main-blob + external-contour generator tuned for stable outer silhouettes."""

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
    ) -> GeneratorOutput:
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

        largest_label = max(passing_labels, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
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


class ConvexHullFallbackV1:
    """Fallback that recovers one contour as a convex hull over edge points."""

    fallback_id = "fallback.convex_hull_v1"

    def recover(self, fallback_mask: np.ndarray) -> tuple[np.ndarray | None, str]:
        if fallback_mask.ndim != 2:
            return None, "invalid_fallback_mask"

        foreground = (fallback_mask > 0).astype(np.uint8)
        ys, xs = np.where(foreground > 0)
        if len(xs) < 3:
            return None, "insufficient_points_for_hull"

        points = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        hull, reason = self._build_valid_hull(points)
        if hull is not None:
            return hull, ""

        if reason in {"convex_hull_degenerate", "convex_hull_zero_area"}:
            # Collinear edge points can collapse to a line; thicken before retrying hull recovery.
            kernel = np.ones((3, 3), dtype=np.uint8)
            dilated = cv2.dilate(foreground, kernel, iterations=1)
            ys_d, xs_d = np.where(dilated > 0)
            if len(xs_d) >= 3:
                points_d = np.stack([xs_d, ys_d], axis=1).astype(np.int32).reshape(-1, 1, 2)
                hull_d, reason_d = self._build_valid_hull(points_d)
                if hull_d is not None:
                    return hull_d, ""
                reason = reason_d

            box_contour = self._thin_box_fallback(xs, ys, image_shape=fallback_mask.shape)
            if box_contour is not None:
                return box_contour, ""

        return None, reason

    @staticmethod
    def _build_valid_hull(points: np.ndarray) -> tuple[np.ndarray | None, str]:
        hull = cv2.convexHull(points)
        if hull is None or hull.size == 0:
            return None, "convex_hull_empty"
        if hull.shape[0] < 3:
            return None, "convex_hull_degenerate"
        area = float(abs(cv2.contourArea(hull)))
        if area < 1.0:
            return None, "convex_hull_zero_area"
        return hull, ""

    @staticmethod
    def _thin_box_fallback(
        xs: np.ndarray,
        ys: np.ndarray,
        *,
        image_shape: tuple[int, int],
    ) -> np.ndarray | None:
        if xs.size == 0 or ys.size == 0:
            return None

        image_h = int(image_shape[0])
        image_w = int(image_shape[1])
        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())

        if x1 == x2:
            if image_w <= 1:
                return None
            if x2 < image_w - 1:
                x2 += 1
            elif x1 > 0:
                x1 -= 1
            else:
                return None

        if y1 == y2:
            if image_h <= 1:
                return None
            if y2 < image_h - 1:
                y2 += 1
            elif y1 > 0:
                y1 -= 1
            else:
                return None

        if x2 <= x1 or y2 <= y1:
            return None

        contour = np.asarray(
            [
                [[x1, y1]],
                [[x2, y1]],
                [[x2, y2]],
                [[x1, y2]],
            ],
            dtype=np.int32,
        )
        if float(abs(cv2.contourArea(contour))) < 1.0:
            return None
        return contour


class OutlineArtifactWriterV1:
    """Render contour outline as black foreground on white background."""

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
    """Render filled contour as black foreground on white background."""

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
    if _safe_ratio(filled_area, selected_component_area) < 0.20:
        flags.append("contour_fill_ratio_small")

    return flags, contour_area, contour_bbox, filled_area



def _filled_mask_from_contour(image_shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    filled = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(filled, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    return filled



def _render_contour_outline(image_shape: tuple[int, int], contour: np.ndarray) -> np.ndarray:
    canvas = np.full(image_shape, 255, dtype=np.uint8)
    cv2.drawContours(canvas, [contour], contourIdx=-1, color=0, thickness=1, lineType=cv2.LINE_AA)
    return canvas



def _black_on_white(mask: np.ndarray) -> np.ndarray:
    canvas = np.full(mask.shape, 255, dtype=np.uint8)
    canvas[mask > 0] = 0
    return canvas



def _bbox_for_contour(contour: np.ndarray) -> tuple[int, int, int, int]:
    x, y, w, h = cv2.boundingRect(contour)
    return (int(x), int(y), int(x + max(0, w - 1)), int(y + max(0, h - 1)))



def _bbox_for_component_stats(stats_row: np.ndarray) -> tuple[int, int, int, int]:
    x = int(stats_row[cv2.CC_STAT_LEFT])
    y = int(stats_row[cv2.CC_STAT_TOP])
    w = int(stats_row[cv2.CC_STAT_WIDTH])
    h = int(stats_row[cv2.CC_STAT_HEIGHT])
    return (x, y, x + max(0, w - 1), y + max(0, h - 1))



def _safe_ratio(numerator: int, denominator: int) -> float:
    if int(denominator) <= 0:
        return 0.0
    return float(numerator) / float(denominator)
