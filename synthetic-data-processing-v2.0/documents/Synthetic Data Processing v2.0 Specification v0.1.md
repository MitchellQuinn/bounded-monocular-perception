# Synthetic Data Processing v2.0 Specification v0.1

## 1. Purpose

Define a clean, low-debt preprocessing pipeline (`synthetic-data-processing-v2.0`) that produces training artifacts for the dual-stream distance regressor described in:

- `rb-training-v2.0/src/topologies/topology_dual_stream_v0.1.py`
- `rb-training-v2.0/documents/Distance Regressor Dual Stream Definition v0.1`

This pipeline introduces YOLO-based defender localization and retains silhouette extraction from `synthetic-data-processing-v1.0` after detection.

## 2. Scope

In scope:

- New pipeline architecture and stage contracts.
- New manifest and `PreprocessingContract` schema.
- New NPZ shard schema for dual-stream model input.
- Backward-compatible outputs to support fair v1 vs v2 model comparison from the same upstream frames.

Out of scope:

- Training loop implementation changes in `rb-training-v2.0`.
- Hyperparameter tuning beyond required defaults.

## 3. Non-Negotiable Output Requirements

Pipeline outputs MUST satisfy dual-stream model input requirements:

- `silhouette_crop`: float32, shape `(N, C_in, H_canvas, W_canvas)`, values in `[0, 1]`.
- `bbox_features`: float32, shape `(N, 10)`, exact feature order in section 7.
- Targets:
  - `y_position_3d`: float32, shape `(N, 3)`
  - `y_distance_m`: float32, shape `(N,)`

Additional requirement from model definition section 10:

- Emit full-frame silhouette artifacts for v1 comparison harness.

## 4. High-Level Pipeline Design

Stage order:

1. `detect` (YOLO defender bbox extraction)
2. `silhouette` (reuse v1.0 silhouette extraction behavior on YOLO ROI)
3. `pack_dual_stream` (feature assembly, canvas crop generation, shard writing)
4. `shuffle` (optional, same purpose as v1.0)

Authoritative stage statuses are row-wise in `samples.csv` and run-wise in `run.json` (`PreprocessingContract`).

## 5. Input Contract

Required input root per run:

- `input-images/<run_name>/images/*.png`
- `input-images/<run_name>/manifests/samples.csv`
- `input-images/<run_name>/manifests/run.json`

Required input manifest columns (minimum):

- Existing unity columns: `run_id`, `sample_id`, `frame_index`, `image_filename`, `distance_m`, `image_width_px`, `image_height_px`, `capture_success`
- Position target columns for 3D mode: `final_pos_x_m`, `final_pos_y_m`, `final_pos_z_m`

Rows with `capture_success != true` are marked `skipped` for all downstream stages.

## 6. Stage 1: `detect` (YOLO)

### 6.1 Dependency

Use installed YOLO package from `.venv` (Ultralytics API). Model weights must be pinned by path and hash in config.

### 6.2 Detection rules

- Infer on full frame.
- Filter detections to configured defender class ids/names.
- If multiple defender boxes remain, select highest confidence.
- Convert bbox to clamped image-space `xyxy` float coordinates.

Failure handling:

- No valid defender detection -> `detect_stage_status=failed`, with reason.
- Downstream stages for that row -> `skipped` with upstream failure reason.

### 6.3 Detect stage outputs (new columns)

- `detect_stage_status`, `detect_stage_error`
- `detect_model_path`, `detect_model_sha256`
- `detect_class_id`, `detect_class_name`, `detect_confidence`
- `detect_bbox_x1`, `detect_bbox_y1`, `detect_bbox_x2`, `detect_bbox_y2`
- `detect_bbox_w_px`, `detect_bbox_h_px`
- `detect_candidates_total`
- `detect_debug_image_filename` (optional overlay)

## 7. `bbox_features` Contract (Exact Order)

For each successful detection, compute float32 features in this exact order:

1. `cx_px`
2. `cy_px`
3. `w_px`
4. `h_px`
5. `cx_norm`
6. `cy_norm`
7. `w_norm`
8. `h_norm`
9. `aspect_ratio`
10. `area_norm`

Definitions:

- `w_px = max(1e-6, x2 - x1)`
- `h_px = max(1e-6, y2 - y1)`
- `cx_px = x1 + 0.5 * w_px`
- `cy_px = y1 + 0.5 * h_px`
- `cx_norm = cx_px / frame_width_px`
- `cy_norm = cy_px / frame_height_px`
- `w_norm = w_px / frame_width_px`
- `h_norm = h_px / frame_height_px`
- `aspect_ratio = w_px / h_px`
- `area_norm = (w_px * h_px) / (frame_width_px * frame_height_px)`

No hand-crafted depth feature is added.

## 8. Stage 2: `silhouette`

### 8.1 Reuse requirement

Keep silhouette processing behavior from `synthetic-data-processing-v1.0` (`rb_pipeline_v2` algorithms):

- generator: `silhouette.contour_v2`
- fallback: `fallback.convex_hull_v1`
- writers: existing outline/filled artifact writers

### 8.2 Ordering and ROI behavior

- YOLO detection runs first.
- Silhouette extraction runs on YOLO ROI (with configurable padding).
- Result is persisted both as:
  - ROI silhouette artifact (for inspection)
  - Full-frame silhouette artifact with ROI result pasted back into source coordinates (for v1 comparison compatibility)

### 8.3 Silhouette stage outputs

Reuse existing silhouette diagnostics where possible, plus ROI-aware fields:

- `silhouette_stage_status`, `silhouette_stage_error`
- `silhouette_image_filename` (full-frame)
- `silhouette_roi_image_filename` (new)
- Existing contour/fallback/quality/debug columns from v1.0 silhouette stage

## 9. Stage 3: `pack_dual_stream`

### 9.1 Canvas sizing policy

`H_canvas`/`W_canvas` must be selected empirically:

1. Collect successful detection bbox widths/heights on calibration corpus.
2. Take 99th percentile of width and height.
3. Round each up to nearest multiple of 32.
4. Store selected values in run contract and shard metadata.

Default placeholder when calibration is unavailable: `224 x 224`.

### 9.2 Silhouette crop generation (no rescaling)

For each row:

1. Start from binary silhouette in YOLO ROI coordinates.
2. Convert to model channel tensor with **white background `1.0`** and **black silhouette `0.0`**.
3. Place on white background canvas without any geometric scaling.
4. Center placement on canvas (integer floor when odd offsets).

If ROI silhouette exceeds canvas in either dimension:

- default policy: fail row (`pack_dual_stream_stage_status=failed`)
- optional policy flag: allow clipping for debug-only runs (must set quality flag and be disabled for production corpora)

### 9.3 Target assembly

- `y_position_3d[i] = [final_pos_x_m, final_pos_y_m, final_pos_z_m]`
- `y_distance_m[i] = distance_m`

### 9.4 NPZ shard schema (required keys)

Each shard MUST include:

- `silhouette_crop` -> `(N, C_in, H_canvas, W_canvas)`, `float32`
- `bbox_features` -> `(N, 10)`, `float32`
- `y_position_3d` -> `(N, 3)`, `float32`
- `y_distance_m` -> `(N,)`, `float32`
- `sample_id` -> `(N,)`, string
- `image_filename` -> `(N,)`, string
- `npz_row_index` -> `(N,)`, `int64`, contiguous `0..N-1`

Loader-side target rule:

- If training `output_mode=position_3d`, collate uses `y_position_3d` as `target`.
- If training `output_mode=scalar_distance`, collate uses `y_distance_m` as `target`.

Strongly recommended metadata arrays:

- `detect_bbox_xyxy_px` -> `(N, 4)`, `float32`
- `frame_wh_px` -> `(N, 2)`, `int32`
- `bbox_features_schema` -> scalar string or `(10,)` string array documenting feature order

### 9.5 Compatibility arrays (for v1 comparison)

To support section 10 of dual-stream definition, optionally write:

- `X` -> full-frame silhouette tensor `(N, H_frame, W_frame)`
- `y` -> alias of `y_distance_m`

These arrays keep existing scalar-distance baselines runnable from the same generated corpus.

## 10. Manifest and Contract Versioning

## 10.1 `samples.csv` required pack columns

- `pack_dual_stream_stage_status`, `pack_dual_stream_stage_error`
- `npz_filename`, `npz_row_index`
- `canvas_width_px`, `canvas_height_px`
- Optional flattened bbox feature columns for audit:
  - `bbox_feat_cx_px`, `bbox_feat_cy_px`, `bbox_feat_w_px`, `bbox_feat_h_px`
  - `bbox_feat_cx_norm`, `bbox_feat_cy_norm`, `bbox_feat_w_norm`, `bbox_feat_h_norm`
  - `bbox_feat_aspect_ratio`, `bbox_feat_area_norm`

## 10.2 `run.json` PreprocessingContract

Use new contract version:

- `ContractVersion: "rb-preprocess-v4-dual-stream"`
- `CompletedStages: ["detect", "silhouette", "pack_dual_stream"]`

`CurrentRepresentation` for completed pack stage must include:

- `Kind: "dual_stream_npz"`
- `StorageFormat: "npz"`
- `ArrayKeys: ["silhouette_crop", "bbox_features", "y_position_3d", "y_distance_m"]`
- `TargetModes: ["position_3d", "scalar_distance"]`
- `SilhouetteCropLayout: "N,C,H,W"`
- `BBoxFeatureDim: 10`
- `CanvasHeight`, `CanvasWidth`
- `SilhouetteScaling: "disabled"`

## 11. Directory Layout (v2.0)

Recommended top-level structure:

- `synthetic-data-processing-v2.0/`
  - `rb_pipeline_v4/`
  - `input-images/`
  - `detect-images-v4/`
  - `silhouette-images-v4/`
  - `training-data-v4/`
  - `tests/`
  - `documents/`

Per run output:

- `<stage-root>/<run_name>/images/`
- `<stage-root>/<run_name>/manifests/run.json`
- `<stage-root>/<run_name>/manifests/samples.csv`
- `training-data-v4/<run_name>/*_shard_XXXXX.npz`

## 12. Validation and Quality Gates

Minimum hard checks before marking a corpus production-ready:

1. Schema checks
   - Required manifest columns present.
   - Required NPZ keys present.
   - Dtypes and shapes match section 9.4.
2. Geometry checks
   - `bbox_features` finite (no NaN/Inf).
   - Normalized features in expected ranges.
3. Canvas checks
   - No clipped samples (unless debug clipping mode explicitly enabled).
4. Alignment checks
   - `sample_id`, `image_filename`, `npz_row_index` aligned between CSV and NPZ.
5. Contract checks
   - Single, consistent `PreprocessingContract` across all corpora used together.

## 13. Test Plan (Required)

Unit tests:

- YOLO detection selection logic (single, multiple, none).
- `bbox_features` numeric correctness and ordering.
- No-rescale canvas placement behavior.
- Silhouette stage fallback behavior parity with v1.0.

Integration tests:

- End-to-end run from input images to NPZ shards.
- Dual-stream key presence and shape validation.
- Optional compatibility arrays `X`/`y` generation.

Regression tests:

- Compare silhouette diagnostics on a fixed fixture set between v1.0 silhouette stage and v2.0 silhouette stage (post-YOLO ROI) to detect unintended algorithm drift.

## 14. Implementation Notes for Debt Reduction

- Keep stage modules small and single-purpose (`detect_stage.py`, `silhouette_stage.py`, `pack_dual_stream_stage.py`).
- Centralize schema constants in one module to avoid column drift.
- Keep strict unknown-config-key errors (same philosophy as topology strict params).
- Avoid hidden defaults for model-critical behavior (canvas size, class filter, clipping policy).

## 15. Acceptance Criteria

The v2.0 specification is considered satisfied when:

1. A generated corpus can provide `silhouette_crop` and `bbox_features` exactly matching dual-stream topology requirements.
2. `bbox_features` are exactly dimension 10 in the documented order.
3. Silhouette crops are not geometrically rescaled.
4. Full-frame silhouette outputs are still available for fair v1 vs v2 comparisons.
5. `PreprocessingContract` captures enough detail to reproduce preprocessing at inference time.
