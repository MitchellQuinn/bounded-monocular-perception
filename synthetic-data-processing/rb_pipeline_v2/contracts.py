"""Protocol interfaces and dataclasses for v2 pluggable components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


ContourArray = np.ndarray


@dataclass(frozen=True)
class GeneratorOutput:
    """Output from a representation generator before rendering."""

    edge_binary: np.ndarray
    fallback_mask: np.ndarray
    contour: ContourArray | None
    primary_reason: str
    debug_images: dict[str, np.ndarray] | None = None
    quality_flags: tuple[str, ...] = ()
    diagnostics: dict[str, object] | None = None


class RepresentationGenerator(Protocol):
    """Build a primary contour candidate from one grayscale source image."""

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
        source_bgr: np.ndarray | None = None,
        experimental_params: dict[str, object] | None = None,
    ) -> GeneratorOutput:
        ...


class FallbackStrategy(Protocol):
    """Recover a contour when primary extraction is clearly broken."""

    fallback_id: str

    def recover(self, fallback_mask: np.ndarray) -> tuple[ContourArray | None, str]:
        ...


class ArtifactWriter(Protocol):
    """Render one representation artifact from a contour."""

    writer_id: str
    representation_mode: str

    def render(
        self,
        image_shape: tuple[int, int],
        contour: ContourArray,
        *,
        line_thickness: int,
    ) -> np.ndarray:
        ...


class ArrayExporter(Protocol):
    """Convert a grayscale representation artifact into a training array."""

    exporter_id: str

    def export(
        self,
        gray_image: np.ndarray,
        *,
        normalize: bool,
        invert: bool,
        output_dtype: str,
    ) -> np.ndarray:
        ...
