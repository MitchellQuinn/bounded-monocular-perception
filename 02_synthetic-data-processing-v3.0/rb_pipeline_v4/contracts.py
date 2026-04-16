"""Protocol interfaces and small data contracts for v4 pipeline components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detector prediction in pixel-space xyxy format."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    center_x_px: float | None = None
    center_y_px: float | None = None


@dataclass(frozen=True)
class GeneratorOutput:
    """Output from silhouette contour generation before rendering."""

    edge_binary: np.ndarray
    fallback_mask: np.ndarray
    contour: np.ndarray | None
    primary_reason: str
    debug_images: dict[str, np.ndarray] | None = None
    quality_flags: tuple[str, ...] = ()
    diagnostics: dict[str, object] | None = None


class ObjectDetector(Protocol):
    """Detector capable of returning per-image candidate boxes."""

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        ...


class RepresentationGenerator(Protocol):
    """Generates one primary contour candidate from a grayscale ROI."""

    generator_id: str

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
        ...


class FallbackStrategy(Protocol):
    """Recovers a contour when primary generation fails."""

    fallback_id: str

    def recover(self, fallback_mask: np.ndarray) -> tuple[np.ndarray | None, str]:
        ...


class ArtifactWriter(Protocol):
    """Renders a contour into an artifact image."""

    writer_id: str
    representation_mode: str

    def render(
        self,
        image_shape: tuple[int, int],
        contour: np.ndarray,
        *,
        line_thickness: int,
    ) -> np.ndarray:
        ...
