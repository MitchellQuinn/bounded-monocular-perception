"""Post-foreground model representation transforms for live inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any

import cv2
import numpy as np

import interfaces.contracts as contracts


MODEL_REPRESENTATION_TRANSFORM_STAGE = "post_foreground_pre_pack"
MODEL_REPRESENTATION_ANCHOR_ROI_CENTER = "roi_center"
MODEL_REPRESENTATION_ANCHOR_FOREGROUND_BBOX_CENTER = "foreground_bbox_center"
MODEL_REPRESENTATION_ANCHOR_EXPLICIT_POINT = "explicit_point"
SUPPORTED_MODEL_REPRESENTATION_ANCHORS = (
    MODEL_REPRESENTATION_ANCHOR_ROI_CENTER,
    MODEL_REPRESENTATION_ANCHOR_FOREGROUND_BBOX_CENTER,
    MODEL_REPRESENTATION_ANCHOR_EXPLICIT_POINT,
)
MODEL_REPRESENTATION_GEOMETRY_NORMALIZATION_SOURCE_IMAGE = "source_image"
SUPPORTED_MODEL_REPRESENTATION_GEOMETRY_NORMALIZATION_SPACES = (
    MODEL_REPRESENTATION_GEOMETRY_NORMALIZATION_SOURCE_IMAGE,
)


@dataclass(frozen=True)
class ModelRepresentationTransformConfig:
    """TOML-backed model-facing ROI transform configuration."""

    enabled: bool = False
    space_name: str | None = None
    stage: str = MODEL_REPRESENTATION_TRANSFORM_STAGE
    scale_x: float | None = None
    scale_y: float | None = None
    anchor: str = MODEL_REPRESENTATION_ANCHOR_FOREGROUND_BBOX_CENTER
    anchor_x_px: float | None = None
    anchor_y_px: float | None = None
    translate_x_px: float = 0.0
    translate_y_px: float = 0.0
    output_width_px: int | None = None
    output_height_px: int | None = None
    image_interpolation: str = "linear"
    mask_interpolation: str = "nearest"
    image_fill_value: int = 255
    mask_fill_value: bool = False
    recompute_geometry_from_transformed_mask: bool = True
    normalization_space: str = MODEL_REPRESENTATION_GEOMETRY_NORMALIZATION_SOURCE_IMAGE

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(
            self,
            "stage",
            _non_empty_text(self.stage, "stage"),
        )
        object.__setattr__(
            self,
            "anchor",
            normalize_model_representation_anchor(self.anchor),
        )
        object.__setattr__(
            self,
            "image_interpolation",
            normalize_interpolation_name(self.image_interpolation),
        )
        object.__setattr__(
            self,
            "mask_interpolation",
            normalize_interpolation_name(self.mask_interpolation),
        )
        object.__setattr__(
            self,
            "image_fill_value",
            _uint8_value(self.image_fill_value, "image_fill_value"),
        )
        object.__setattr__(self, "mask_fill_value", bool(self.mask_fill_value))
        object.__setattr__(
            self,
            "normalization_space",
            normalize_geometry_normalization_space(self.normalization_space),
        )
        object.__setattr__(
            self,
            "recompute_geometry_from_transformed_mask",
            bool(self.recompute_geometry_from_transformed_mask),
        )
        object.__setattr__(
            self,
            "output_width_px",
            _optional_positive_int(self.output_width_px, "output_width_px"),
        )
        object.__setattr__(
            self,
            "output_height_px",
            _optional_positive_int(self.output_height_px, "output_height_px"),
        )
        object.__setattr__(
            self,
            "translate_x_px",
            _finite_float(self.translate_x_px, "translate_x_px"),
        )
        object.__setattr__(
            self,
            "translate_y_px",
            _finite_float(self.translate_y_px, "translate_y_px"),
        )
        if not bool(self.enabled):
            return

        object.__setattr__(
            self,
            "space_name",
            _non_empty_text(self.space_name, "space_name"),
        )
        object.__setattr__(
            self,
            "scale_x",
            _positive_finite_float(self.scale_x, "scale_x"),
        )
        object.__setattr__(
            self,
            "scale_y",
            _positive_finite_float(self.scale_y, "scale_y"),
        )
        if self.anchor == MODEL_REPRESENTATION_ANCHOR_EXPLICIT_POINT:
            object.__setattr__(
                self,
                "anchor_x_px",
                _finite_float(self.anchor_x_px, "anchor_x_px"),
            )
            object.__setattr__(
                self,
                "anchor_y_px",
                _finite_float(self.anchor_y_px, "anchor_y_px"),
            )

    def metadata_base(self) -> dict[str, Any]:
        """Return serializable transform configuration metadata."""
        return {
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_ENABLED: bool(self.enabled),
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_SPACE_NAME: self.space_name,
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_STAGE: self.stage,
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_SCALE_X: self.scale_x,
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_SCALE_Y: self.scale_y,
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_ANCHOR: self.anchor,
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_ANCHOR_XY_PX: (
                None
                if self.anchor_x_px is None or self.anchor_y_px is None
                else (float(self.anchor_x_px), float(self.anchor_y_px))
            ),
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_TRANSLATE_XY_PX: (
                float(self.translate_x_px),
                float(self.translate_y_px),
            ),
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_OUTPUT_WH_PX: (
                None
                if self.output_width_px is None or self.output_height_px is None
                else (int(self.output_width_px), int(self.output_height_px))
            ),
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_IMAGE_INTERPOLATION: (
                self.image_interpolation
            ),
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_MASK_INTERPOLATION: (
                self.mask_interpolation
            ),
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_IMAGE_FILL_VALUE: int(
                self.image_fill_value
            ),
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_MASK_FILL_VALUE: bool(
                self.mask_fill_value
            ),
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_GEOMETRY_NORMALIZATION_SPACE: (
                self.normalization_space
            ),
            contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_RECOMPUTE_GEOMETRY: (
                bool(self.recompute_geometry_from_transformed_mask)
            ),
        }


@dataclass(frozen=True)
class ModelRepresentationTransformResult:
    """Transformed model-space ROI representation and geometry."""

    roi_repr: np.ndarray
    orientation_source_gray: np.ndarray
    foreground_mask: np.ndarray
    model_full_foreground_mask: np.ndarray
    model_foreground_area_px: int
    model_foreground_bbox_inclusive_xyxy_px: tuple[int, int, int, int]
    model_feature_bbox_xyxy_px: np.ndarray
    metadata: Mapping[str, Any]
    debug_images: Mapping[str, np.ndarray]


class ModelRepresentationTransformer:
    """Apply a configurable ROI-space transform at the model packing boundary."""

    def __init__(
        self,
        config: ModelRepresentationTransformConfig | None = None,
    ) -> None:
        self._config = config or ModelRepresentationTransformConfig()
        self._image_interpolation = interpolation_flag(
            self._config.image_interpolation
        )
        self._mask_interpolation = interpolation_flag(self._config.mask_interpolation)

    @property
    def config(self) -> ModelRepresentationTransformConfig:
        return self._config

    def transform(
        self,
        *,
        roi_repr: np.ndarray,
        orientation_source_gray: np.ndarray,
        foreground_mask: np.ndarray,
        source_gray_shape: tuple[int, ...],
        source_bounds: np.ndarray,
        roi_bounds: np.ndarray,
    ) -> ModelRepresentationTransformResult:
        """Transform aligned ROI representation inputs into model space."""
        roi_image = _ensure_2d_image(roi_repr, "roi_repr")
        orientation_image = _ensure_2d_image(
            orientation_source_gray,
            "orientation_source_gray",
        )
        mask = _ensure_2d_bool(foreground_mask, "foreground_mask")
        _require_same_shape(
            roi_image,
            orientation_image,
            "roi_repr",
            "orientation_source_gray",
        )
        _require_same_shape(roi_image, mask, "roi_repr", "foreground_mask")

        input_h, input_w = int(roi_image.shape[0]), int(roi_image.shape[1])
        output_w = self._config.output_width_px or input_w
        output_h = self._config.output_height_px or input_h
        if (int(output_w), int(output_h)) != (input_w, input_h):
            raise ValueError(
                "ModelRepresentationTransformer currently requires output_width_px "
                "and output_height_px to match the ROI canvas size so source-image "
                "geometry remains compatible with the tri-stream contract."
            )

        if not bool(self._config.enabled):
            return self._result(
                roi_repr=roi_image,
                orientation_source_gray=orientation_image,
                foreground_mask=mask,
                source_gray_shape=source_gray_shape,
                source_bounds=source_bounds,
                roi_bounds=roi_bounds,
                applied=False,
                affine_matrix=None,
                anchor_xy_px=None,
                debug_images={},
            )

        anchor_xy = _anchor_xy(mask, self._config)
        matrix = _scale_translate_matrix(
            anchor_xy=anchor_xy,
            scale_x=float(self._config.scale_x),
            scale_y=float(self._config.scale_y),
            translate_x_px=float(self._config.translate_x_px),
            translate_y_px=float(self._config.translate_y_px),
        )
        dsize = (input_w, input_h)
        transformed_roi_repr = cv2.warpAffine(
            roi_image,
            matrix,
            dsize,
            flags=self._image_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_image_border_value(
                roi_image,
                self._config.image_fill_value,
            ),
        )
        transformed_orientation = cv2.warpAffine(
            orientation_image,
            matrix,
            dsize,
            flags=self._image_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=_image_border_value(
                orientation_image,
                self._config.image_fill_value,
            ),
        )
        transformed_mask_u8 = cv2.warpAffine(
            mask.astype(np.uint8) * 255,
            matrix,
            dsize,
            flags=self._mask_interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255 if bool(self._config.mask_fill_value) else 0,
        )
        transformed_mask = transformed_mask_u8 > 0
        if not bool(np.any(transformed_mask)):
            raise ValueError(
                "Model representation transform produced an empty foreground mask."
            )

        return self._result(
            roi_repr=np.ascontiguousarray(transformed_roi_repr),
            orientation_source_gray=np.ascontiguousarray(transformed_orientation),
            foreground_mask=transformed_mask,
            source_gray_shape=source_gray_shape,
            source_bounds=source_bounds,
            roi_bounds=roi_bounds,
            applied=True,
            affine_matrix=matrix,
            anchor_xy_px=anchor_xy,
            debug_images={
                "model_roi_repr_before_transform": roi_image,
                "model_roi_repr_after_transform": transformed_roi_repr,
                "model_foreground_mask_before_transform": mask,
                "model_foreground_mask_after_transform": transformed_mask,
                "model_orientation_source_before_transform": orientation_image,
                "model_orientation_source_after_transform": transformed_orientation,
            },
        )

    def _result(
        self,
        *,
        roi_repr: np.ndarray,
        orientation_source_gray: np.ndarray,
        foreground_mask: np.ndarray,
        source_gray_shape: tuple[int, ...],
        source_bounds: np.ndarray,
        roi_bounds: np.ndarray,
        applied: bool,
        affine_matrix: np.ndarray | None,
        anchor_xy_px: tuple[float, float] | None,
        debug_images: Mapping[str, np.ndarray],
    ) -> ModelRepresentationTransformResult:
        (
            full_foreground_mask,
            foreground_area_px,
            foreground_bbox,
            feature_bbox_xyxy,
        ) = _foreground_geometry_from_roi_mask(
            foreground_mask,
            source_shape=source_gray_shape,
            source_bounds=source_bounds,
            roi_bounds=roi_bounds,
        )
        metadata = self._config.metadata_base()
        metadata.update(
            {
                contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_APPLIED: bool(applied),
                contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_ANCHOR_RESOLVED_XY_PX: (
                    None
                    if anchor_xy_px is None
                    else (float(anchor_xy_px[0]), float(anchor_xy_px[1]))
                ),
                contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_AFFINE_MATRIX_2X3: (
                    None if affine_matrix is None else affine_matrix.tolist()
                ),
                contracts.PREPROCESSING_METADATA_MODEL_FOREGROUND_BBOX_XYXY_PX: _array_xyxy_to_tuple(
                    feature_bbox_xyxy
                ),
                contracts.PREPROCESSING_METADATA_MODEL_FOREGROUND_BBOX_INCLUSIVE_XYXY_PX: foreground_bbox,
                contracts.PREPROCESSING_METADATA_MODEL_FOREGROUND_AREA_PX: int(foreground_area_px),
            }
        )
        return ModelRepresentationTransformResult(
            roi_repr=np.ascontiguousarray(roi_repr),
            orientation_source_gray=np.ascontiguousarray(orientation_source_gray),
            foreground_mask=np.ascontiguousarray(foreground_mask, dtype=bool),
            model_full_foreground_mask=full_foreground_mask,
            model_foreground_area_px=int(foreground_area_px),
            model_foreground_bbox_inclusive_xyxy_px=foreground_bbox,
            model_feature_bbox_xyxy_px=feature_bbox_xyxy,
            metadata=metadata,
            debug_images=dict(debug_images),
        )


def load_model_representation_transform_config(
    path: Path | str,
) -> ModelRepresentationTransformConfig:
    """Load a model representation transform config from a TOML file."""
    resolved = Path(path).expanduser().resolve(strict=False)
    payload = tomllib.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Model representation transform TOML must contain tables.")
    section = payload.get("model_representation_transform")
    if section is None:
        return ModelRepresentationTransformConfig()
    if not isinstance(section, Mapping):
        raise ValueError("[model_representation_transform] must be a table.")

    enabled = _optional_bool(section, "enabled", default=False)
    if not enabled:
        return ModelRepresentationTransformConfig(
            enabled=False,
            space_name=_optional_text(section, "space_name", default=None),
            stage=_optional_text(
                section,
                "stage",
                default=MODEL_REPRESENTATION_TRANSFORM_STAGE,
            ),
        )

    affine = _required_table(section, "affine")
    resampling = _required_table(section, "resampling")
    geometry = _required_table(section, "geometry")
    output = _optional_table(section, "output")
    return ModelRepresentationTransformConfig(
        enabled=True,
        space_name=_required_text(section, "space_name"),
        stage=_required_text(section, "stage"),
        scale_x=_required_float(affine, "scale_x"),
        scale_y=_required_float(affine, "scale_y"),
        anchor=_required_text(affine, "anchor"),
        anchor_x_px=_optional_float(affine, "anchor_x_px"),
        anchor_y_px=_optional_float(affine, "anchor_y_px"),
        translate_x_px=_required_float(affine, "translate_x_px"),
        translate_y_px=_required_float(affine, "translate_y_px"),
        output_width_px=_optional_int(output, "width_px"),
        output_height_px=_optional_int(output, "height_px"),
        image_interpolation=_required_text(resampling, "image_interpolation"),
        mask_interpolation=_required_text(resampling, "mask_interpolation"),
        image_fill_value=_required_int(resampling, "image_fill_value"),
        mask_fill_value=_required_bool(resampling, "mask_fill_value"),
        recompute_geometry_from_transformed_mask=_required_bool(
            geometry,
            "recompute_from_transformed_mask",
        ),
        normalization_space=_required_text(geometry, "normalization_space"),
    )


def normalize_model_representation_anchor(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "roi_center": MODEL_REPRESENTATION_ANCHOR_ROI_CENTER,
        "center": MODEL_REPRESENTATION_ANCHOR_ROI_CENTER,
        "foreground_bbox_center": MODEL_REPRESENTATION_ANCHOR_FOREGROUND_BBOX_CENTER,
        "foreground_center": MODEL_REPRESENTATION_ANCHOR_FOREGROUND_BBOX_CENTER,
        "bbox_center": MODEL_REPRESENTATION_ANCHOR_FOREGROUND_BBOX_CENTER,
        "explicit_point": MODEL_REPRESENTATION_ANCHOR_EXPLICIT_POINT,
        "explicit": MODEL_REPRESENTATION_ANCHOR_EXPLICIT_POINT,
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(
        "model representation transform anchor must be one of "
        f"{SUPPORTED_MODEL_REPRESENTATION_ANCHORS!r}; got {value!r}."
    )


def normalize_geometry_normalization_space(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"source_image", "source"}:
        return MODEL_REPRESENTATION_GEOMETRY_NORMALIZATION_SOURCE_IMAGE
    raise ValueError(
        "model representation transform geometry normalization_space must be one "
        f"of {SUPPORTED_MODEL_REPRESENTATION_GEOMETRY_NORMALIZATION_SPACES!r}; "
        f"got {value!r}."
    )


def normalize_interpolation_name(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "nearest": "nearest",
        "nearest_neighbor": "nearest",
        "linear": "linear",
        "bilinear": "linear",
        "area": "area",
        "cubic": "cubic",
    }
    if text in aliases:
        return aliases[text]
    raise ValueError(
        "interpolation must be one of ('nearest', 'linear', 'area', 'cubic'); "
        f"got {value!r}."
    )


def interpolation_flag(value: Any) -> int:
    name = normalize_interpolation_name(value)
    if name == "nearest":
        return int(cv2.INTER_NEAREST)
    if name == "linear":
        return int(cv2.INTER_LINEAR)
    if name == "area":
        return int(cv2.INTER_AREA)
    if name == "cubic":
        return int(cv2.INTER_CUBIC)
    raise ValueError(f"Unsupported interpolation: {value!r}.")


def _foreground_geometry_from_roi_mask(
    foreground_mask: np.ndarray,
    *,
    source_shape: tuple[int, ...],
    source_bounds: np.ndarray,
    roi_bounds: np.ndarray,
) -> tuple[np.ndarray, int, tuple[int, int, int, int], np.ndarray]:
    source_h, source_w = int(source_shape[0]), int(source_shape[1])
    src_x1, src_y1, src_x2, src_y2 = [int(value) for value in source_bounds.tolist()]
    roi_x1, roi_y1, roi_x2, roi_y2 = [int(value) for value in roi_bounds.tolist()]
    full_foreground_mask = np.zeros((source_h, source_w), dtype=bool)
    source_target = full_foreground_mask[src_y1:src_y2, src_x1:src_x2]
    roi_source = np.asarray(foreground_mask, dtype=bool)[
        roi_y1:roi_y2,
        roi_x1:roi_x2,
    ]
    if tuple(source_target.shape) != tuple(roi_source.shape):
        raise ValueError(
            "Transformed foreground mask ROI slice does not match source insert "
            f"shape: roi={roi_source.shape!r}, source={source_target.shape!r}."
        )
    source_target[:, :] = roi_source
    full_foreground_mask[src_y1:src_y2, src_x1:src_x2] = source_target
    area_px, bbox = _mask_geometry(full_foreground_mask)
    feature_bbox_xyxy = _feature_bbox_from_geometry(
        bbox,
        area_px=area_px,
        fallback_bounds=source_bounds,
        source_shape=full_foreground_mask.shape,
    )
    return full_foreground_mask, area_px, bbox, feature_bbox_xyxy


def _mask_geometry(mask: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return 0, (0, 0, 0, 0)
    return int(xs.size), (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


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


def _anchor_xy(
    foreground_mask: np.ndarray,
    config: ModelRepresentationTransformConfig,
) -> tuple[float, float]:
    mask_h, mask_w = int(foreground_mask.shape[0]), int(foreground_mask.shape[1])
    if config.anchor == MODEL_REPRESENTATION_ANCHOR_ROI_CENTER:
        return float(mask_w) * 0.5, float(mask_h) * 0.5
    if config.anchor == MODEL_REPRESENTATION_ANCHOR_EXPLICIT_POINT:
        return float(config.anchor_x_px), float(config.anchor_y_px)
    if config.anchor == MODEL_REPRESENTATION_ANCHOR_FOREGROUND_BBOX_CENTER:
        bbox = _mask_bbox_xyxy(foreground_mask)
        if bbox is None:
            return float(mask_w) * 0.5, float(mask_h) * 0.5
        x1, y1, x2, y2 = bbox
        return x1 + (0.5 * (x2 - x1)), y1 + (0.5 * (y2 - y1))
    raise ValueError(f"Unsupported transform anchor: {config.anchor!r}.")


def _mask_bbox_xyxy(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return (
        float(xs.min()),
        float(ys.min()),
        float(xs.max() + 1),
        float(ys.max() + 1),
    )


def _scale_translate_matrix(
    *,
    anchor_xy: tuple[float, float],
    scale_x: float,
    scale_y: float,
    translate_x_px: float,
    translate_y_px: float,
) -> np.ndarray:
    anchor_x, anchor_y = float(anchor_xy[0]), float(anchor_xy[1])
    return np.asarray(
        [
            [scale_x, 0.0, anchor_x - (scale_x * anchor_x) + translate_x_px],
            [0.0, scale_y, anchor_y - (scale_y * anchor_y) + translate_y_px],
        ],
        dtype=np.float32,
    )


def _image_border_value(image: np.ndarray, fill_value: int) -> float | int:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0 and int(fill_value) > 1:
            return float(fill_value) / 255.0
        return float(fill_value)
    return int(fill_value)


def _array_xyxy_to_tuple(value: np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(value, dtype=np.float32).reshape(4)
    return tuple(float(item) for item in array.tolist())


def _ensure_2d_image(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array; got shape {array.shape!r}.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label} must be numeric; got dtype {array.dtype!r}.")
    return np.ascontiguousarray(array)


def _ensure_2d_bool(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array; got shape {array.shape!r}.")
    return np.ascontiguousarray(array.astype(bool, copy=False))


def _require_same_shape(
    left: np.ndarray,
    right: np.ndarray,
    left_label: str,
    right_label: str,
) -> None:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(
            f"{left_label} and {right_label} must have the same shape; got "
            f"{left.shape!r} and {right.shape!r}."
        )


def _required_table(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"[model_representation_transform.{key}] is required.")
    return value


def _optional_table(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"[model_representation_transform.{key}] must be a table.")
    return value


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    if key not in payload:
        raise ValueError(f"model representation transform {key!r} is required.")
    return _non_empty_text(payload[key], key)


def _optional_text(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: str | None,
) -> str | None:
    if key not in payload:
        return default
    return _non_empty_text(payload[key], key)


def _required_float(payload: Mapping[str, Any], key: str) -> float:
    if key not in payload:
        raise ValueError(f"model representation transform {key!r} is required.")
    return _finite_float(payload[key], key)


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    if key not in payload:
        return None
    return _finite_float(payload[key], key)


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"model representation transform {key!r} is required.")
    if isinstance(payload[key], bool) or not isinstance(payload[key], int):
        raise ValueError(f"{key} must be an int; got {payload[key]!r}.")
    return int(payload[key])


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    if key not in payload:
        return None
    if isinstance(payload[key], bool) or not isinstance(payload[key], int):
        raise ValueError(f"{key} must be an int; got {payload[key]!r}.")
    return int(payload[key])


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    if key not in payload:
        raise ValueError(f"model representation transform {key!r} is required.")
    if not isinstance(payload[key], bool):
        raise ValueError(f"{key} must be a bool; got {payload[key]!r}.")
    return bool(payload[key])


def _optional_bool(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    if key not in payload:
        return default
    if not isinstance(payload[key], bool):
        raise ValueError(f"{key} must be a bool; got {payload[key]!r}.")
    return bool(payload[key])


def _non_empty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string; got {value!r}.")
    return value.strip()


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{label} must be a finite number; got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; got {value!r}.")
    return number


def _positive_finite_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be > 0; got {value!r}.")
    return number


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a positive int; got {value!r}.")
    number = int(value)
    if number <= 0:
        raise ValueError(f"{label} must be > 0; got {value!r}.")
    return number


def _uint8_value(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an int in [0, 255]; got {value!r}.")
    number = int(value)
    if number < 0 or number > 255:
        raise ValueError(f"{label} must be in [0, 255]; got {value!r}.")
    return number


__all__ = [
    "MODEL_REPRESENTATION_ANCHOR_EXPLICIT_POINT",
    "MODEL_REPRESENTATION_ANCHOR_FOREGROUND_BBOX_CENTER",
    "MODEL_REPRESENTATION_ANCHOR_ROI_CENTER",
    "MODEL_REPRESENTATION_GEOMETRY_NORMALIZATION_SOURCE_IMAGE",
    "MODEL_REPRESENTATION_TRANSFORM_STAGE",
    "SUPPORTED_MODEL_REPRESENTATION_ANCHORS",
    "SUPPORTED_MODEL_REPRESENTATION_GEOMETRY_NORMALIZATION_SPACES",
    "ModelRepresentationTransformConfig",
    "ModelRepresentationTransformResult",
    "ModelRepresentationTransformer",
    "interpolation_flag",
    "load_model_representation_transform_config",
    "normalize_geometry_normalization_space",
    "normalize_interpolation_name",
    "normalize_model_representation_anchor",
]
