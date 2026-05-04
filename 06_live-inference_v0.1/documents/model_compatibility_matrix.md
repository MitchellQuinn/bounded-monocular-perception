# Live Model Compatibility Matrix

Generated: 2026-05-04

## Summary

- Metadata-only scan; no checkpoints were loaded and no model runtimes were imported.
- Called `load_live_model_manifest()` and `check_live_model_compatibility()` from `06_live-inference_v0.1/src/live_inference/model_registry/` for distance/orientation compatibility.
- Distance/orientation parent directories with `run_register.json` are included as registry-root loader checks; the actual loader-readable artifacts are their `runs/run_*` directories.
- ROI-FCN is treated as an independently selected locator/preprocessing dependency, not as an input head for the distance/orientation regressor.
- Distance/orientation artifacts scanned: 5 registered run artifact(s); 4 parent registry root(s) inspected.
- ROI-FCN artifacts scanned: 3.
- Compatible live tri-stream distance/orientation models: 3.
- Likely current deployment candidate: `05_inference-v0.4-ts/models/distance-orientation/260504-1100_ts-2d-cnn/runs/run_0001`.
- ROI-FCN candidate: `05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0003`.
- Blockers: No live-tree distance/orientation artifacts were found; No live-tree ROI-FCN artifacts were found; Parent registry roots contain run_register.json but are not directly loader-readable model bundles; use/copy a run artifact root or add manifest metadata at the selected root.

Expected live distance/orientation contract:

- topology contract version: `rb-topology-output-reporting-v1`
- preprocessing contract name: `rb-preprocess-v4-tri-stream-orientation-v1`
- input mode: `tri_stream_distance_orientation_geometry`
- representation kind: `tri_stream_npz`
- required input keys: `x_distance_image, x_orientation_image, x_geometry`
- required output keys: `distance_m`, `yaw_sin_cos`

## Distance/Orientation Model Matrix

| artifact root | location | compatible | likely family | checkpoint selected | topology id | topology variant | topology contract version | preprocessing contract name | input mode | representation kind | input keys discovered | geometry schema status | output keys discovered | distance output width | yaw output width | compatibility errors | compatibility warnings | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 05_inference-v0.4-ts/models/distance-orientation/260415-1146_ds-2d-cnn | old-tree | no | ds | not found | metadata missing | metadata missing | metadata missing | metadata missing | metadata missing | metadata missing | none | metadata missing | none | metadata missing | metadata missing | metadata missing: no loader-recognized model metadata files or checkpoint at the requested registry root; loader discovery gap: root contains run_register.json; actual artifacts are under 05_inference-v0.4-ts/models/distance-orientation/260415-1146_ds-2d-cnn/runs/run_0001 | none | Requested registry root; the current loader does not descend into runs/. |
| 05_inference-v0.4-ts/models/distance-orientation/260425-1025_ds-2d-cnn | old-tree | no | ds | not found | metadata missing | metadata missing | metadata missing | metadata missing | metadata missing | metadata missing | none | metadata missing | none | metadata missing | metadata missing | metadata missing: no loader-recognized model metadata files or checkpoint at the requested registry root; loader discovery gap: root contains run_register.json; actual artifacts are under 05_inference-v0.4-ts/models/distance-orientation/260425-1025_ds-2d-cnn/runs/run_0001 | none | Requested registry root; the current loader does not descend into runs/. |
| 05_inference-v0.4-ts/models/distance-orientation/260430-1023_ts-2d-cnn | old-tree | no | ts | not found | metadata missing | metadata missing | metadata missing | metadata missing | metadata missing | metadata missing | none | metadata missing | none | metadata missing | metadata missing | metadata missing: no loader-recognized model metadata files or checkpoint at the requested registry root; loader discovery gap: root contains run_register.json; actual artifacts are under 05_inference-v0.4-ts/models/distance-orientation/260430-1023_ts-2d-cnn/runs/run_0001, 05_inference-v0.4-ts/models/distance-orientation/260430-1023_ts-2d-cnn/runs/run_0002 | none | Requested registry root; the current loader does not descend into runs/. |
| 05_inference-v0.4-ts/models/distance-orientation/260504-1100_ts-2d-cnn | old-tree | no | ts | not found | metadata missing | metadata missing | metadata missing | metadata missing | metadata missing | metadata missing | none | metadata missing | none | metadata missing | metadata missing | metadata missing: no loader-recognized model metadata files or checkpoint at the requested registry root; loader discovery gap: root contains run_register.json; actual artifacts are under 05_inference-v0.4-ts/models/distance-orientation/260504-1100_ts-2d-cnn/runs/run_0001 | none | Requested registry root; the current loader does not descend into runs/. |
| 05_inference-v0.4-ts/models/distance-orientation/260415-1146_ds-2d-cnn/runs/run_0001 | old-tree | no | ds | best.pt | distance_regressor_dual_stream_yaw | dual_stream_yaw_v0_1 | rb-topology-output-reporting-v1 | rb-preprocess-v4-dual-stream-orientation-v1 | dual_stream_image_bbox_features | dual_stream_npz | silhouette_crop, bbox_features, y_position_3d, y_distance_m, y_yaw_deg, y_yaw_sin, y_yaw_cos | legacy dual-stream / no x_geometry schema | distance_m, yaw_sin_cos | 1 | 2 | legacy dual-stream / wrong input contract: input mode mismatch; preprocessing contract mismatch; representation kind mismatch; missing required tri-stream input keys x_distance_image, x_orientation_image, x_geometry; missing tri-stream geometry schema/dimension metadata | none | legacy dual-stream artifact; consumes silhouette/bbox-style inputs, not x_distance_image + x_orientation_image + x_geometry |
| 05_inference-v0.4-ts/models/distance-orientation/260425-1025_ds-2d-cnn/runs/run_0001 | old-tree | no | ds | best.pt | distance_regressor_dual_stream_yaw | dual_stream_yaw_v0_1 | rb-topology-output-reporting-v1 | rb-preprocess-v4-dual-stream-orientation-brightness-v1 | dual_stream_image_bbox_features | dual_stream_npz | silhouette_crop, bbox_features, y_position_3d, y_distance_m, y_yaw_deg, y_yaw_sin, y_yaw_cos | legacy dual-stream / no x_geometry schema | distance_m, yaw_sin_cos | 1 | 2 | legacy dual-stream / wrong input contract: input mode mismatch; preprocessing contract mismatch; representation kind mismatch; missing required tri-stream input keys x_distance_image, x_orientation_image, x_geometry; missing tri-stream geometry schema/dimension metadata | none | legacy dual-stream artifact; consumes silhouette/bbox-style inputs, not x_distance_image + x_orientation_image + x_geometry |
| 05_inference-v0.4-ts/models/distance-orientation/260430-1023_ts-2d-cnn/runs/run_0001 | old-tree | yes | ts | best.pt | distance_regressor_tri_stream_yaw | tri_stream_yaw_v0_1 | rb-topology-output-reporting-v1 | rb-preprocess-v4-tri-stream-orientation-v1 | tri_stream_distance_orientation_geometry | tri_stream_npz | x_distance_image, x_orientation_image, x_geometry, y_distance_m, y_yaw_deg, y_yaw_sin, y_yaw_cos | ok: 10-field tri-stream geometry schema | distance_m, yaw_sin_cos | 1 | 2 | none | none | passes current live tri-stream compatibility checker |
| 05_inference-v0.4-ts/models/distance-orientation/260430-1023_ts-2d-cnn/runs/run_0002 | old-tree | yes | ts | best.pt | distance_regressor_tri_stream_yaw | tri_stream_yaw_v0_1 | rb-topology-output-reporting-v1 | rb-preprocess-v4-tri-stream-orientation-v1 | tri_stream_distance_orientation_geometry | tri_stream_npz | x_distance_image, x_orientation_image, x_geometry, y_distance_m, y_yaw_deg, y_yaw_sin, y_yaw_cos | ok: 10-field tri-stream geometry schema | distance_m, yaw_sin_cos | 1 | 2 | none | none | passes current live tri-stream compatibility checker |
| 05_inference-v0.4-ts/models/distance-orientation/260504-1100_ts-2d-cnn/runs/run_0001 | old-tree | yes | ts | best.pt | distance_regressor_tri_stream_yaw | tri_stream_yaw_v0_1 | rb-topology-output-reporting-v1 | rb-preprocess-v4-tri-stream-orientation-v1 | tri_stream_distance_orientation_geometry | tri_stream_npz | x_distance_image, x_orientation_image, x_geometry, y_distance_m, y_yaw_deg, y_yaw_sin, y_yaw_cos | ok: 10-field tri-stream geometry schema | distance_m, yaw_sin_cos | 1 | 2 | none | none | passes current live tri-stream compatibility checker; most recent compatible distance/orientation artifact found |

## ROI-FCN Artifact Inventory

| artifact root | location | checkpoint selected | run config found | dataset contract found | locator canvas size if discoverable | ROI crop size if discoverable | artifact contract/version if discoverable | notes/errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0001 | old-tree | best.pt | yes | yes | 480x300 | 300x300 | rb-roi-fcn-topology-v0_1, rb-roi-fcn-training-v0_1, rb-preprocess-roi-fcn-v0_1 | metadata declares ROI crop 300x300 on locator canvas 480x300; summary.json not found |
| 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0002 | old-tree | best.pt | yes | yes | 480x300 | 300x300 | rb-roi-fcn-topology-v0_1, rb-roi-fcn-training-v0_1, rb-preprocess-roi-fcn-v0_1 | metadata declares ROI crop 300x300 on locator canvas 480x300; resumed from run_0001; summary.json not found |
| 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0003 | old-tree | best.pt | yes | yes | 480x300 | 300x300 | rb-roi-fcn-topology-v0_1, rb-roi-fcn-training-v0_1, rb-preprocess-roi-fcn-v0_1 | metadata declares ROI crop 300x300 on locator canvas 480x300; resumed from run_0002; validation mean center error 3.176px |

## Pairing Notes

The distance/orientation model consumes `x_distance_image`, `x_orientation_image`, and `x_geometry`. It does not consume ROI-FCN heatmaps or logits directly. Pairing below only checks metadata-discoverable locator crop/canvas compatibility.

| compatible distance/orientation model | ROI-FCN artifacts that appear pairable | metadata warnings | notes |
| --- | --- | --- | --- |
| 05_inference-v0.4-ts/models/distance-orientation/260430-1023_ts-2d-cnn/runs/run_0001 | 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0001, 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0002, 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0003 | none | metadata pairable: ROI crop size matches the distance canvas and fits inside the locator canvas; runtime behavior not checked |
| 05_inference-v0.4-ts/models/distance-orientation/260430-1023_ts-2d-cnn/runs/run_0002 | 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0001, 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0002, 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0003 | none | metadata pairable: ROI crop size matches the distance canvas and fits inside the locator canvas; runtime behavior not checked |
| 05_inference-v0.4-ts/models/distance-orientation/260504-1100_ts-2d-cnn/runs/run_0001 | 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0001, 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0002, 05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0003 | none | metadata pairable: ROI crop size matches the distance canvas and fits inside the locator canvas; runtime behavior not checked |

## Recommended Current Selection

- `distance_orientation`: `05_inference-v0.4-ts/models/distance-orientation/260504-1100_ts-2d-cnn/runs/run_0001`
- `roi_fcn`: `05_inference-v0.4-ts/models/roi-fcn/260420-1219_roi-fcn-tiny/runs/run_0003`
- Notes: Recommended roots are loader-readable run artifacts. No live-tree artifact copy was found in this scan.

Do not treat this as runtime validation. It proves only that the available metadata matches the live tri-stream and locator pairing checks exposed by the current lightweight loader/checker.

## Machine-Readable Output

- JSON: `06_live-inference_v0.1/documents/model_compatibility_matrix.json`
