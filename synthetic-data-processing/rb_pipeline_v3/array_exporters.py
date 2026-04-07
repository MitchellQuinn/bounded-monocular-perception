"""Array exporters for v3 threshold artifacts."""

from __future__ import annotations

import numpy as np


class GrayscaleArrayExporterV1:
    """Convert grayscale threshold PNG to a model-ready 2D array."""

    exporter_id = "array.grayscale_v1"

    def export(
        self,
        gray_image: np.ndarray,
        *,
        normalize: bool,
        invert: bool,
        output_dtype: str,
    ) -> np.ndarray:
        if gray_image.ndim != 2:
            raise ValueError("gray_image must be 2D grayscale")

        arr = gray_image.astype(np.float32)

        if normalize:
            arr = arr / 255.0

        if invert:
            arr = 1.0 - arr

        return _coerce_output_dtype(arr, output_dtype=output_dtype, normalize=normalize)



def _coerce_output_dtype(array: np.ndarray, *, output_dtype: str, normalize: bool) -> np.ndarray:
    dtype_name = str(output_dtype).strip().lower()

    if dtype_name == "float32":
        return array.astype(np.float32, copy=False)

    if dtype_name == "float16":
        return array.astype(np.float16, copy=False)

    if dtype_name == "uint8":
        if normalize:
            scaled = array * 255.0
        else:
            scaled = array
        return np.clip(np.rint(scaled), 0.0, 255.0).astype(np.uint8, copy=False)

    raise ValueError(f"Unsupported output_dtype '{output_dtype}'")
