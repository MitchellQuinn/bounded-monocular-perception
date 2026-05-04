# Current Live-Local Model Selection

Generated: 2026-05-04

This selection stages the currently recommended metadata-compatible model pair
into the live inference model tree. The selection file selects artifact roots
only; compatibility must still be checked by the `model_registry` layer before
runtime use.

## Selection

- Selection file: `06_live-inference_v0.1/models/selections/current.toml`
- Distance/orientation live-local path: `06_live-inference_v0.1/models/distance-orientation/260504-1100_ts-2d-cnn__run_0001`
- ROI-FCN live-local path: `06_live-inference_v0.1/models/roi-fcn/260420-1219_roi-fcn-tiny__run_0003`
- Device requests: distance/orientation `cuda`; ROI-FCN `cuda`
- Artifact staging: copied directories; no symlinks

## Compatibility Result

- Checker: `load_live_model_manifest()` with `roi_locator_root`, followed by
  `check_live_model_compatibility()`
- Result: pass
- Errors: none
- Warnings: none

## Checkpoints

- Distance/orientation selected checkpoint: `best.pt`
- ROI-FCN selected checkpoint candidate present: `best.pt`
- Checkpoints were discovered by filename metadata only; no checkpoint was
  loaded with PyTorch.

## Topology And Preprocessing Contract

- Topology id: `distance_regressor_tri_stream_yaw`
- Topology variant: `tri_stream_yaw_v0_1`
- Topology contract version: `rb-topology-output-reporting-v1`
- Preprocessing contract name: `rb-preprocess-v4-tri-stream-orientation-v1`
- Input mode: `tri_stream_distance_orientation_geometry`
- Representation kind: `tri_stream_npz`
- Input keys: `x_distance_image`, `x_orientation_image`, `x_geometry`,
  `y_distance_m`, `y_yaw_deg`, `y_yaw_sin`, `y_yaw_cos`
- Geometry schema: `cx_px`, `cy_px`, `w_px`, `h_px`, `cx_norm`, `cy_norm`,
  `w_norm`, `h_norm`, `aspect_ratio`, `area_norm`
- Distance canvas size: `300x300`
- Orientation canvas size: `300x300`
- Output keys: `distance_m`, `yaw_sin_cos`
- Output widths: distance `1`; yaw `2`

## ROI Crop And Canvas Compatibility

- ROI-FCN crop size: `300x300`
- ROI-FCN locator canvas size: `480x300`
- Compatibility: ROI crop size matches the distance canvas and fits within the
  locator canvas.

## Notes

- This report validates metadata compatibility only.
- The pending preprocessing contract update for the representation correction is
  not implemented here; real preprocessor implementation should wait until that
  contract lands.
