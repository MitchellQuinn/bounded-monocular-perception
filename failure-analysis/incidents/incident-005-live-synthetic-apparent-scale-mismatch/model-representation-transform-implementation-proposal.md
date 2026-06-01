# Model Representation Transform Implementation Proposal

**Incident:** `incident-005-live-synthetic-apparent-scale-mismatch`  
**System:** bounded monocular perception, live inference v0.3  
**Status:** Proposed implementation plan  
**Date:** 2026-06-01

## 1. Purpose

Incident 005 shows that accepted live frames can pass the locator and foreground
pipeline while still presenting the Defender at the wrong apparent scale for the
direct distance/yaw model. The observed live target is larger than the matched
synthetic target, which makes the synthetic-trained model predict the live target
too close.

This proposal defines a configurable mitigation that corrects the model-facing
representation without forcing the raw camera preprocessing algorithms to operate
on transformed images.

The main design position is:

```text
raw camera preprocessing should remain in raw camera space
real-to-model representation correction should happen at the model packing boundary
```

## 2. Design Goals

- Keep locator, background removal, manual masks, ROI extraction, and foreground
  extraction operating on raw camera imagery.
- Add a configurable model representation transform after foreground extraction
  and before final model inputs are emitted.
- Configure all camera-specific values from TOML.
- Support independent horizontal and vertical correction through separate
  `scale_x` and `scale_y` settings.
- Avoid hard-coded camera, calibration, scale, interpolation, and fill-value
  parameters in Python code.
- Preserve both raw-space and model-space metadata in traces.
- Make the transform reusable for different real camera systems and future
  synthetic camera contracts.

## 3. Pipeline Placement

The current v0.3 generic preprocessor reaches the model packing boundary in:

```text
06_live-inference_v0.3/src/live_inference/preprocessing/generic_tri_stream_live_preprocessor.py
```

The relevant local sequence is:

```text
foreground_mask -> component cleanup
roi_repr = _render_vehicle_detail_on_white(...)
foreground_enhancement
_build_distance_image(...)
_build_orientation_image(...)
geometry = _bbox_features_from_xyxy(...)
```

The proposed transform should sit after foreground mask cleanup and ROI
representation rendering, but before the final distance image, orientation image,
and geometry vector are created.

Recommended revised sequence:

```text
raw camera space:
  decode
  manual mask
  background removal
  locator
  ROI crop
  foreground extraction
  foreground component cleanup
  raw foreground metadata

model representation space:
  render ROI representation
  apply model representation transform
  foreground enhancement / brightness normalization as model-contract stages
  build x_distance_image
  build x_orientation_image
  recompute x_geometry from transformed mask
```

This means the existing frame-level `CameraIntrinsicsFrameTransformer` is not the
right implementation point for Incident 005. It transforms the whole frame before
the locator. Incident 005 needs a post-foreground model representation transform.

## 4. Proposed Module

Add a new module:

```text
06_live-inference_v0.3/src/live_inference/preprocessing/model_representation_transform.py
```

Suggested public API:

```python
@dataclass(frozen=True)
class ModelRepresentationTransformConfig:
    enabled: bool
    space_name: str
    scale_x: float
    scale_y: float
    anchor: str
    translate_x_px: float
    translate_y_px: float
    output_width_px: int | None
    output_height_px: int | None
    image_interpolation: str
    mask_interpolation: str
    image_fill_value: int
    mask_fill_value: bool
    recompute_geometry_from_transformed_mask: bool
    normalization_space: str


@dataclass(frozen=True)
class ModelRepresentationTransformResult:
    roi_repr: np.ndarray
    orientation_source_gray: np.ndarray
    foreground_mask: np.ndarray
    model_full_foreground_mask: np.ndarray
    model_foreground_area_px: int
    model_foreground_bbox_inclusive_xyxy_px: tuple[int, int, int, int]
    model_feature_bbox_xyxy_px: np.ndarray
    metadata: Mapping[str, Any]


class ModelRepresentationTransformer:
    def transform(
        self,
        *,
        roi_repr: np.ndarray,
        orientation_source_gray: np.ndarray,
        foreground_mask: np.ndarray,
        source_gray_shape: tuple[int, int],
        source_bounds: np.ndarray,
        roi_bounds: np.ndarray,
    ) -> ModelRepresentationTransformResult:
        ...
```

The transformer should use OpenCV affine warping internally. Images and masks must
be transformed with the same affine matrix, but with independently configurable
resampling policies.

## 5. TOML Configuration

The transform should be configured from a camera/runtime profile TOML, not from the
model-selection TOML. Model selection should remain responsible for artifact roots
and device policy. Camera profiles should describe how a real camera runtime maps
into the selected model representation contract.

Suggested profile shape:

```toml
[model_representation_transform]
enabled = true
space_name = "arducam_ar0234_to_synthetic_ts2dcnn_v1"
stage = "post_foreground_pre_pack"

[model_representation_transform.affine]
scale_x = 1.0
scale_y = 1.0
anchor = "foreground_bbox_center"
translate_x_px = 0.0
translate_y_px = 0.0

[model_representation_transform.output]
width_px = 300
height_px = 300

[model_representation_transform.resampling]
image_interpolation = "linear"
mask_interpolation = "nearest"
image_fill_value = 255
mask_fill_value = false

[model_representation_transform.geometry]
recompute_from_transformed_mask = true
normalization_space = "source_image"
```

Important requirements:

- `scale_x` and `scale_y` are separate required fields when the transform is
  enabled.
- `scale_x` must not be silently reused for `scale_y`.
- `anchor` should be configurable. Initial supported values can be:
  - `roi_center`
  - `foreground_bbox_center`
  - `explicit_point`
- If `anchor = "explicit_point"`, require `anchor_x_px` and `anchor_y_px`.
- The TOML loader should reject enabled transforms with missing, zero, negative,
  NaN, or infinite scale values.
- Paths, calibration references, scale factors, interpolation names, fill values,
  and output dimensions should all come from TOML or from the active model
  preprocessing contract, not hard-coded constants.

## 6. Coordinate Spaces

The implementation should explicitly distinguish these spaces:

| Space | Meaning | Examples |
| --- | --- | --- |
| Raw source space | Original accepted camera frame | locator result, manual mask, background snapshot |
| Raw ROI space | Fixed ROI canvas extracted from raw source | `roi_gray`, raw foreground mask |
| Model ROI space | ROI canvas after apparent-scale correction | transformed representation and mask |
| Model input space | Final tensors emitted to the model | `x_distance_image`, `x_orientation_image`, `x_geometry` |

For compatibility with the current geometry schema:

```text
cx_px, cy_px, w_px, h_px, cx_norm, cy_norm, w_norm, h_norm, aspect_ratio, area_norm
```

the transformed foreground mask should be placed back into a source-sized canvas
using the same ROI source bounds. Geometry should then be recomputed from that
model-space full foreground mask.

This preserves the current source-image normalization contract while allowing the
model-facing bbox width and height to reflect the calibrated apparent-scale
correction.

## 7. Integration Details

In `TriStreamLivePreprocessor.__init__`, accept either:

```python
model_representation_transform_config: ModelRepresentationTransformConfig | None
model_representation_transformer: ModelRepresentationTransformer | None
```

In `prepare_model_inputs`, keep the existing raw preprocessing flow through:

```text
_foreground_mask_after_background_removal
_foreground_mask_component_cleanup
_foreground_geometry_from_roi_mask
_render_vehicle_detail_on_white
```

Then apply the new transform before packing:

```python
raw_geometry_payload = _foreground_geometry_from_roi_mask(...)
roi_repr = _render_vehicle_detail_on_white(...)
raw_orientation_source_gray = _raw_orientation_source_after_background_removal(...)

model_repr = self._model_representation_transformer.transform(
    roi_repr=roi_repr,
    orientation_source_gray=raw_orientation_source_gray,
    foreground_mask=foreground_mask,
    source_gray_shape=mask_preparation.regressor_source_gray.shape,
    source_bounds=source_bounds,
    roi_bounds=roi_bounds,
)

foreground_mask = model_repr.foreground_mask
roi_repr = model_repr.roi_repr
raw_orientation_source_gray = model_repr.orientation_source_gray
final_feature_bbox_xyxy_px = model_repr.model_feature_bbox_xyxy_px
```

Then build the existing model inputs from the transformed values:

```python
distance_image_2d = self._build_distance_image(
    roi_repr=roi_repr,
    foreground_mask=foreground_mask,
)

orientation_image_2d = self._build_orientation_image(
    roi_source_gray=raw_orientation_source_gray,
    representation_source=roi_repr,
    foreground_mask=foreground_mask,
)

geometry = _bbox_features_from_xyxy(
    final_feature_bbox_xyxy_px,
    image_width_px=source_w,
    image_height_px=source_h,
)
```

## 8. Metadata and Trace Contract

The current metadata keys for foreground bbox and area should continue to describe
the model-facing representation used for inference. Add explicit raw-space keys so
the trace remains auditable:

```text
raw_foreground_bbox_xyxy_px
raw_foreground_bbox_inclusive_xyxy_px
raw_foreground_area_px
model_foreground_bbox_xyxy_px
model_foreground_bbox_inclusive_xyxy_px
model_foreground_area_px
model_representation_transform_enabled
model_representation_transform_space_name
model_representation_transform_stage
model_representation_transform_scale_x
model_representation_transform_scale_y
model_representation_transform_anchor
model_representation_transform_translate_xy_px
model_representation_transform_affine_matrix_2x3
model_representation_transform_output_wh_px
model_representation_transform_geometry_normalization_space
```

The existing foreground metadata keys can be set to the model-space values after
the transform:

```text
foreground_bbox_xyxy_px
foreground_bbox_inclusive_xyxy_px
foreground_area_px
silhouette_bbox_xyxy_px
silhouette_bbox_inclusive_xyxy_px
silhouette_area_px
```

This keeps the inference result contract aligned with what the model actually
received while preserving raw diagnostic evidence.

## 9. Debug Artifacts

Add optional debug artifacts when `save_debug_images` is enabled:

```text
model_roi_repr_before_transform.png
model_roi_repr_after_transform.png
model_foreground_mask_before_transform.png
model_foreground_mask_after_transform.png
model_orientation_source_before_transform.png
model_orientation_source_after_transform.png
```

The GUI preview should remain raw by default. Debug artifact views can expose
model-space images explicitly.

Changing this transform should not automatically clear raw manual masks or raw
background snapshots unless the underlying raw source dimensions change. The raw
operators still belong to raw camera space.

## 10. Calibration Tooling

Add a small derivation tool:

```text
06_live-inference_v0.3/tools/derive_model_representation_transform.py
```

Inputs:

- matched synthetic/live bbox table
- nominal distances
- orientation labels
- target transform convention

Outputs:

- width scale summary
- height scale summary
- suggested `scale_x`
- suggested `scale_y`
- implied apparent-distance offsets before and after correction
- TOML snippet for the selected camera profile

This tool should use the Incident 005 scale-pair method as a reproducible
calculation instead of relying on hand-copied arithmetic.

## 11. Tests

Add focused tests under:

```text
06_live-inference_v0.3/tests/
```

Recommended coverage:

1. TOML loader accepts a complete disabled config.
2. TOML loader rejects an enabled transform with missing `scale_x`.
3. TOML loader rejects an enabled transform with missing `scale_y`.
4. TOML loader rejects zero or negative scale values.
5. Anisotropic scaling changes width and height independently.
6. Image and mask transforms remain spatially aligned.
7. Nearest-neighbor mask interpolation preserves boolean mask semantics.
8. `x_geometry` is recomputed from the transformed mask.
9. Raw foreground metadata remains unchanged.
10. Model foreground metadata matches the transformed representation.
11. A scale-pair fixture shows apparent-distance offset moving toward zero after
    applying the derived transform.

Run with the repository venv:

```bash
PYTHONPATH=06_live-inference_v0.3/src ./.venv/bin/python \
  -m unittest discover -s 06_live-inference_v0.3/tests -v
```

## 12. Rollout Plan

### Phase 1: Configuration and Pure Transform

- Add TOML config dataclasses and loader.
- Add pure ROI/mask affine transform implementation.
- Add unit tests for validation and anisotropic scaling.

### Phase 2: Preprocessor Integration

- Inject the transform into `TriStreamLivePreprocessor`.
- Apply it after foreground cleanup and before model packing.
- Recompute model-space geometry from the transformed mask.
- Add raw-space and model-space metadata.
- Add debug artifacts.

### Phase 3: Runtime Profile Wiring

- Add a camera/runtime profile argument to the GUI app.
- Load the model representation transform from TOML.
- Pass the config into the preprocessor.
- Keep existing camera capture settings configurable from the same profile or a
  clearly linked camera TOML.

### Phase 4: Calibration and Evidence Loop

- Add the derivation tool for scale-pair data.
- Generate a first Arducam-to-model profile from Incident 005 measurements.
- Re-run matched synthetic/live scale checks.
- Re-run the live distance sweep at the same measured reference positions.

### Phase 5: Acceptance Gate

Accept the mitigation only when:

- The live/synthetic apparent-scale offset materially moves toward zero.
- Clean live signed error no longer clusters around `-0.35 m` to `-0.40 m`.
- Trace bundles show both raw-space and model-space evidence.
- Camera-specific transform values are present in TOML and absent from Python
  constants.

## 13. Open Questions

- Should foreground enhancement happen before or after the model representation
  transform? The recommended default is after transform because it is part of the
  model-facing representation contract, but this should be verified against the
  training preprocessing contract.
- Should geometry normalization remain `source_image`, or should a future model
  contract define ROI-local geometry? The current model expects source-normalized
  geometry, so the first implementation should preserve it.
- Should the initial transform be a simple affine scale/translate only, or should
  future profiles support lens-model or homography terms? The Incident 005
  mitigation should start with affine scale/translate because the observed failure
  is an apparent-size mismatch.

## 14. Summary

The mitigation should not warp the whole live frame before the raw preprocessing
pipeline. Instead, it should add a configurable model representation transform
after foreground extraction and before final tri-stream packing.

That preserves the strengths of the v0.3 locator and foreground path while making
the model-facing representation explicit, traceable, and camera-profile driven.
Most importantly for Incident 005, horizontal and vertical apparent-scale
corrections are independent TOML settings, so different camera systems can be
calibrated without changing Python code.
