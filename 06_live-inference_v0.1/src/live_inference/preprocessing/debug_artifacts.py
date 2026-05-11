"""Debug artifact writing for live preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

import interfaces.contracts as contracts
from interfaces.contracts import FrameHash


ARTIFACT_ACCEPTED_RAW_FRAME = contracts.DISPLAY_ARTIFACT_ACCEPTED_RAW_FRAME
ARTIFACT_ROI_CROP = contracts.DISPLAY_ARTIFACT_ROI_CROP
ARTIFACT_LOCATOR_INPUT = contracts.DISPLAY_ARTIFACT_LOCATOR_INPUT
ARTIFACT_DISTANCE_IMAGE = contracts.TRI_STREAM_DISTANCE_IMAGE_KEY
ARTIFACT_ORIENTATION_IMAGE = contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY
ARTIFACT_ROI_OVERLAY_METADATA = contracts.DISPLAY_ARTIFACT_ROI_OVERLAY_METADATA


def default_debug_output_dir() -> Path:
    """Return the live-local default debug artifact directory."""
    return Path(__file__).resolve().parents[3] / "live_debug"


class DebugArtifactWriter:
    """Write hash-matched preprocessing debug artifacts as files."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Path | str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir) if output_dir is not None else default_debug_output_dir()

    def write_preprocessing_artifacts(
        self,
        *,
        request_id: str,
        input_image_hash: FrameHash | str,
        preprocessing_parameter_revision: int | None,
        image_artifacts: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> dict[str, Path]:
        """Write image artifacts and metadata JSON, returning artifact paths."""
        if not self.enabled:
            return {}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self._filename_prefix(request_id, input_image_hash)
        image_paths: dict[str, Path] = {}
        for kind, image in image_artifacts.items():
            if image is None:
                continue
            path = self.output_dir / f"{prefix}__{_safe_filename(kind)}.png"
            _write_image(path, image)
            image_paths[str(kind)] = path

        metadata_path = (
            self.output_dir / f"{prefix}__{ARTIFACT_ROI_OVERLAY_METADATA}.json"
        )
        debug_paths = dict(image_paths)
        debug_paths[ARTIFACT_ROI_OVERLAY_METADATA] = metadata_path
        _write_json(
            metadata_path,
            _metadata_payload(
                request_id=request_id,
                input_image_hash=input_image_hash,
                preprocessing_parameter_revision=preprocessing_parameter_revision,
                metadata=metadata,
                debug_paths=debug_paths,
            ),
        )
        return debug_paths

    def _filename_prefix(self, request_id: str, input_image_hash: FrameHash | str) -> str:
        return (
            f"{_safe_filename(request_id)}__"
            f"{_safe_filename(_short_frame_hash(input_image_hash))}"
        )


def _write_image(path: Path, image: Any) -> None:
    array = _image_array_uint8(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), array):
        raise OSError(f"Failed to write debug image: {path}")


def _image_array_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]
    elif array.ndim == 3 and int(array.shape[0]) in (3, 4) and int(array.shape[-1]) not in (3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim not in (2, 3):
        raise ValueError(f"Debug image must be 2D or 3D; got shape {array.shape}.")

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if array.dtype == np.bool_:
        return np.ascontiguousarray(array.astype(np.uint8) * 255)

    numeric = np.asarray(array, dtype=np.float32)
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=255.0, neginf=0.0)
    finite_max = float(np.max(numeric)) if numeric.size else 0.0
    if finite_max <= 1.0:
        numeric = numeric * 255.0
    return np.ascontiguousarray(np.clip(numeric, 0.0, 255.0).astype(np.uint8))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _metadata_payload(
    *,
    request_id: str,
    input_image_hash: FrameHash | str,
    preprocessing_parameter_revision: int | None,
    metadata: Mapping[str, Any],
    debug_paths: Mapping[str, Path],
) -> dict[str, Any]:
    payload = dict(metadata)
    payload.update(
        {
            "request_id": str(request_id),
            "input_image_hash": _hash_value(input_image_hash),
            "preprocessing_parameter_revision": preprocessing_parameter_revision,
            "debug_paths": {str(key): str(path) for key, path in debug_paths.items()},
        }
    )
    return payload


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


def _short_frame_hash(input_image_hash: FrameHash | str) -> str:
    return _hash_value(input_image_hash)[:12] or "nohash"


def _hash_value(input_image_hash: FrameHash | str) -> str:
    if isinstance(input_image_hash, FrameHash):
        return str(input_image_hash.value)
    return str(input_image_hash)


__all__ = [
    "ARTIFACT_ACCEPTED_RAW_FRAME",
    "ARTIFACT_DISTANCE_IMAGE",
    "ARTIFACT_LOCATOR_INPUT",
    "ARTIFACT_ORIENTATION_IMAGE",
    "ARTIFACT_ROI_CROP",
    "ARTIFACT_ROI_OVERLAY_METADATA",
    "DebugArtifactWriter",
    "default_debug_output_dir",
]
