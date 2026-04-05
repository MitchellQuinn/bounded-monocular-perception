"""Pass-1 contour and convex-hull implementations for silhouette generation."""

from __future__ import annotations

import cv2
import numpy as np

from ..contracts import GeneratorOutput


class ContourSilhouetteGeneratorV1:
    """Primary contour-first extractor using edge cleanup and largest component selection."""

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
    ) -> GeneratorOutput:
        if source_gray.ndim != 2:
            raise ValueError("source_gray must be 2D grayscale")

        processed = source_gray
        if blur_kernel_size > 1:
            processed = cv2.GaussianBlur(processed, (blur_kernel_size, blur_kernel_size), 0)

        edges = cv2.Canny(processed, int(canny_low_threshold), int(canny_high_threshold))
        cleaned = np.where(edges > 0, 255, 0).astype(np.uint8)

        if close_kernel_size > 1:
            close_kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

        if dilate_kernel_size > 1:
            dilate_kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
            cleaned = cv2.dilate(cleaned, dilate_kernel, iterations=1)

        component_mask = _largest_valid_component(cleaned, min_component_area_px)
        if component_mask is None:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=cleaned,
                contour=None,
                primary_reason="no_valid_component",
            )

        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=component_mask,
                contour=None,
                primary_reason="no_contours",
            )

        contour = max(contours, key=lambda value: float(abs(cv2.contourArea(value))))
        if contour is None or contour.size == 0:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=component_mask,
                contour=None,
                primary_reason="empty_contour",
            )

        if contour.shape[0] < 3:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=component_mask,
                contour=None,
                primary_reason="degenerate_contour",
            )

        area = float(abs(cv2.contourArea(contour)))
        if area < 1.0:
            return GeneratorOutput(
                edge_binary=cleaned,
                fallback_mask=component_mask,
                contour=None,
                primary_reason="degenerate_contour_area",
            )

        return GeneratorOutput(
            edge_binary=cleaned,
            fallback_mask=component_mask,
            contour=contour,
            primary_reason="",
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
