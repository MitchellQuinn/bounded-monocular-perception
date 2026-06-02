from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import interfaces.contracts as contracts  # noqa: E402
from interfaces import FrameReference, InferenceRequest  # noqa: E402
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402
from live_inference.model_registry import load_live_model_manifest  # noqa: E402
from live_inference.preprocessing import (  # noqa: E402
    FixedCenterRoiLocator,
    ModelRepresentationTransformConfig,
    ModelRepresentationTransformer,
    TriStreamLivePreprocessor,
    load_model_representation_transform_config,
)


class ModelRepresentationTransformTests(unittest.TestCase):
    def test_loader_rejects_enabled_transform_without_scale_y(self) -> None:
        path = _temp_toml(
            """
            [model_representation_transform]
            enabled = true
            space_name = "test_space"
            stage = "post_foreground_pre_pack"

            [model_representation_transform.affine]
            scale_x = 0.5
            anchor = "foreground_bbox_center"
            translate_x_px = 0.0
            translate_y_px = 0.0

            [model_representation_transform.resampling]
            image_interpolation = "linear"
            mask_interpolation = "nearest"
            image_fill_value = 255
            mask_fill_value = false

            [model_representation_transform.geometry]
            recompute_from_transformed_mask = true
            normalization_space = "source_image"
            """
        )

        with self.assertRaisesRegex(ValueError, "scale_y"):
            load_model_representation_transform_config(path)

    def test_anisotropic_scale_changes_mask_width_and_height_independently(self) -> None:
        roi_repr, orientation_source, foreground_mask = _roi_fixture()
        transformer = ModelRepresentationTransformer(
            _enabled_config(scale_x=0.5, scale_y=0.25)
        )

        result = transformer.transform(
            roi_repr=roi_repr,
            orientation_source_gray=orientation_source,
            foreground_mask=foreground_mask,
            source_gray_shape=roi_repr.shape,
            source_bounds=np.asarray([0, 0, roi_repr.shape[1], roi_repr.shape[0]]),
            roi_bounds=np.asarray([0, 0, roi_repr.shape[1], roi_repr.shape[0]]),
        )

        x1, y1, x2, y2 = result.model_feature_bbox_xyxy_px.tolist()
        width_px = x2 - x1
        height_px = y2 - y1
        self.assertLess(width_px, 20.0)
        self.assertLess(height_px, 20.0)
        self.assertGreater(width_px, height_px)
        self.assertTrue(result.metadata[contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_APPLIED])
        self.assertEqual(
            result.metadata[contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_SCALE_X],
            0.5,
        )
        self.assertEqual(
            result.metadata[contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_SCALE_Y],
            0.25,
        )

    def test_preprocessor_recomputes_geometry_from_transformed_mask(self) -> None:
        image_bytes = _fixture_image_bytes()
        prepared = TriStreamLivePreprocessor(
            model_manifest=_manifest(),
            locator=FixedCenterRoiLocator(roi_wh_px=(320, 320)),
            model_representation_transform_config=_enabled_config(
                scale_x=0.5,
                scale_y=0.5,
            ),
        ).prepare_model_inputs(_request(image_bytes), image_bytes)

        metadata = prepared.preprocessing_metadata
        geometry = prepared.model_inputs[contracts.TRI_STREAM_GEOMETRY_KEY]
        raw_x1, _raw_y1, raw_x2, _raw_y2 = metadata[contracts.PREPROCESSING_METADATA_RAW_FOREGROUND_BBOX_XYXY_PX]
        model_x1, _model_y1, model_x2, _model_y2 = metadata[
            contracts.PREPROCESSING_METADATA_FOREGROUND_BBOX_XYXY_PX
        ]

        self.assertTrue(metadata[contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_ENABLED])
        self.assertTrue(metadata[contracts.PREPROCESSING_METADATA_MODEL_REPRESENTATION_TRANSFORM_APPLIED])
        self.assertLess(model_x2 - model_x1, raw_x2 - raw_x1)
        self.assertAlmostEqual(float(geometry[2]), model_x2 - model_x1, places=4)
        self.assertEqual(
            metadata[contracts.PREPROCESSING_METADATA_MODEL_FOREGROUND_BBOX_XYXY_PX],
            metadata[contracts.PREPROCESSING_METADATA_FOREGROUND_BBOX_XYXY_PX],
        )


def _enabled_config(
    *,
    scale_x: float,
    scale_y: float,
) -> ModelRepresentationTransformConfig:
    return ModelRepresentationTransformConfig(
        enabled=True,
        space_name="test_model_space",
        scale_x=scale_x,
        scale_y=scale_y,
        anchor="foreground_bbox_center",
        translate_x_px=0.0,
        translate_y_px=0.0,
        image_interpolation="linear",
        mask_interpolation="nearest",
        image_fill_value=255,
        mask_fill_value=False,
        recompute_geometry_from_transformed_mask=True,
        normalization_space="source_image",
    )


def _roi_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roi_repr = np.full((80, 100), 255, dtype=np.uint8)
    foreground_mask = np.zeros((80, 100), dtype=bool)
    foreground_mask[30:50, 40:60] = True
    roi_repr[foreground_mask] = 40
    orientation_source = roi_repr.copy()
    return roi_repr, orientation_source, foreground_mask


def _fixture_image_bytes() -> bytes:
    image = np.full((600, 960), 255, dtype=np.uint8)
    cv2.rectangle(image, (430, 270), (530, 330), 80, thickness=-1)
    cv2.line(image, (430, 270), (530, 330), 40, thickness=2)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AssertionError("Could not encode fixture image")
    return encoded.tobytes()


def _request(image_bytes: bytes) -> InferenceRequest:
    return InferenceRequest(
        request_id="req",
        frame=FrameReference(
            image_path=Path("fixture.png"),
            frame_hash=compute_frame_hash(image_bytes),
        ),
        requested_at_utc="2026-06-01T00:00:00Z",
    )


def _manifest() -> object:
    return load_live_model_manifest(
        PROJECT_ROOT / "models/distance-orientation/260515-1301_ts-2d-cnn"
    )


def _temp_toml(contents: str) -> Path:
    path = Path(tempfile.mkdtemp()) / "transform.toml"
    path.write_text(contents, encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
