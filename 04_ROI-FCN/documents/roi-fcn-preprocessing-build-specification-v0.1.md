# ROI FCN Preprocessing Build Specification v0.1

## 1. Purpose

Define the implementation contract for `04_ROI-FCN/01_preprocessing`.

This preprocessing subsystem must turn already-generated full-frame image corpora into packed ROI-FCN training corpora that preserve:

- the full-frame locator input
- the authoritative crop-centre target
- the metadata required to map between original-image space and locator-canvas space

This document is implementation-grade. It is intended to guide the build of the notebook launcher, helper modules, stage contracts, manifest updates, and packed shard outputs.

It sits downstream of:

- `04_ROI-FCN/documents/roi-fcn-functional-specification-v0.1.md`

and must follow the repository generation discipline defined in:

- `documents/generation-standards-v0.1.md`

---

## 2. Governing References

### 2.1 Functional authority

The functional authority for ROI-FCN preprocessing is:

- `04_ROI-FCN/documents/roi-fcn-functional-specification-v0.1.md`

That document defines the narrow responsibility boundary:

- input: full-frame image
- output: crop centre in original full-frame coordinates
- non-goals: bbox prediction, distance regression, orientation regression, downstream policy

### 2.2 Repository generation authority

This build must explicitly follow `documents/generation-standards-v0.1.md`, especially the following rules:

- explicit contracts over implied behavior
- one authority per concern
- fail loudly on invalid state
- extend by composition, not duplication
- keep notebook UI separate from reusable logic
- version contracts when semantics change

### 2.3 Existing code that must be reused

The ROI-FCN preprocessing build must reuse the current edge ROI path rather than reimplementing it locally.

Authoritative reuse points are:

- `02_synthetic-data-processing-v3.0/rb_pipeline_v4/detector.py`
  - `EdgeRoiDetector`
- `02_synthetic-data-processing-v3.0/rb_pipeline_v4/config.py`
  - `DetectStageConfigV4`
- `02_synthetic-data-processing-v3.0/rb_pipeline_v4/contracts.py`
  - `Detection`
- `02_synthetic-data-processing-v3.0/rb_pipeline_v4/manifest.py`
  - existing `PreprocessingContract` update pattern
- `02_synthetic-data-processing-v3.0/rb_pipeline_v4/widgets.py`
- `02_synthetic-data-processing-v3.0/rb_pipeline_v4/widgets_v05.py`
  - launcher interaction pattern and edge-only control behavior
- `05_inference-v0.1/src/inference_v0_1/pipeline.py`
  - repository example of reusing preprocessing-side semantics instead of forking them

The important rule is: ROI target bootstrapping must use the same edge ROI v1 behavior that is already trusted in `02_synthetic-data-processing-v3.0` and reused in `05_inference-v0.1`.

---

## 3. Scope

This build specification covers:

- dataset discovery under `04_ROI-FCN/01_preprocessing/input`
- notebook launcher behavior for one preprocessing run
- the new `train` / `validate` split convention for ROI-FCN input datasets
- strict split handling for `train` then `validate`
- bootstrapping crop-centre targets from the current edge ROI algorithm
- packing locator-ready FCN corpora into shards
- split-level manifest and contract updates
- tests required to keep the preprocessing contract explicit and stable

This build does not cover:

- ROI-FCN topology implementation
- ROI-FCN training loop code
- Gaussian-heatmap loss implementation inside the training code
- live inference with the trained ROI-FCN
- downstream `300 x 300` ROI extraction for distance/orientation regression

---

## 4. Repository-Alignment Requirements

### 4.1 Separation of concerns is strict

Following `documents/generation-standards-v0.1.md`, the notebook is a control surface only.

The notebook may own:

- widget creation
- widget layout
- wiring button handlers to helper calls
- status display
- final split summaries and verdict

The helper modules under `01_preprocessing/src/` must own:

- dataset discovery
- path resolution
- validation
- manifest mutation
- stage execution
- shard writing
- reusable transform logic
- contract serialization

No canonical preprocessing logic may live only inside the notebook.

### 4.2 Reuse, do not fork

The current edge ROI behavior is already implemented and should remain single-authority.

ROI-FCN preprocessing must therefore:

- reuse `EdgeRoiDetector`
- reuse the detector-side parameter semantics already established in `DetectStageConfigV4`
- wrap those imports behind ROI-FCN-local helper code rather than scattering direct imports throughout the notebook

It must not:

- create a second local implementation of edge ROI v1
- redefine detector parameter meanings
- introduce a subtly different crop-centre rule for training-target bootstrapping

### 4.3 Fail loudly

The build must reject invalid state early and explicitly, including:

- missing `train` or `validate` split
- malformed split structure
- missing `run.json` or `samples.csv`
- missing required manifest columns
- unreadable source images
- unsupported detector backend
- existing output collision when overwrite is not explicitly enabled

### 4.4 Version and contract visibility

The packed output is a new preprocessing representation and must therefore define its own explicit contract version and array-key contract.

No silent reuse of the dual-stream preprocessing contract version is allowed.

---

## 5. Input Dataset Contract

### 5.1 New split convention

This document defines a new ROI-FCN preprocessing input convention.

The selectable unit in the notebook is a dataset reference directory under:

- `04_ROI-FCN/01_preprocessing/input/`

Each selectable dataset reference must contain:

- `train/`
- `validate/`

Example:

```text
04_ROI-FCN/01_preprocessing/input/26-04-11_v022/
  train/
  validate/
```

The dataset reference is the name of that directory:

- `26-04-11_v022`

### 5.2 Required split structure

Each split must satisfy the raw-image corpus structure already used elsewhere in the repository:

```text
04_ROI-FCN/01_preprocessing/input/<dataset-reference>/<split>/
  images/
  manifests/
    run.json
    samples.csv
```

Where `<split>` is exactly one of:

- `train`
- `validate`

These `images/` directories contain full-frame source images. They are not pre-cropped `300 x 300` ROI inputs.

### 5.3 Required manifest assumptions

At minimum, each input `samples.csv` must contain the same authoritative raw-capture fields expected by the existing preprocessing path:

- `run_id`
- `sample_id`
- `frame_index`
- `image_filename`
- `distance_m`
- `image_width_px`
- `image_height_px`
- `capture_success`

Rows with `capture_success != true` are not processed into packed locator shards. They must be marked `skipped` at stage level.

### 5.4 Discovery behavior

The notebook launcher must:

- inspect `01_preprocessing/input/`
- list only dataset references that contain both valid `train` and `validate` split corpora
- refuse silent fallback if multiple datasets exist
- require explicit user selection

This follows the repository rule that corpus selection must never be ambiguous.

---

## 6. Output Dataset Contract

### 6.1 Output root

For a selected dataset reference `<dataset-reference>`, the preprocessing run must write to:

- `04_ROI-FCN/01_preprocessing/output/<dataset-reference>/`

### 6.2 Split output structure

The output must contain one processed split root for `train` and one for `validate`:

```text
04_ROI-FCN/01_preprocessing/output/<dataset-reference>/
  train/
    arrays/
    manifests/
      run.json
      samples.csv
      bootstrap_center_target_stage_log.txt
      pack_roi_fcn_stage_log.txt
  validate/
    arrays/
    manifests/
      run.json
      samples.csv
      bootstrap_center_target_stage_log.txt
      pack_roi_fcn_stage_log.txt
```

### 6.3 Authority rule

The authoritative split outputs are:

- `manifests/run.json`
- `manifests/samples.csv`
- `arrays/*.npz`

No extra parent-level manifest is required in v0.1.

This avoids creating a second authority layer above the already-canonical split manifests.

### 6.4 Naming rule

NPZ shard names must be deterministic and traceable.

Required pattern:

- `<dataset-reference>__train__shard_0000.npz`
- `<dataset-reference>__train__shard_0001.npz`
- `<dataset-reference>__validate__shard_0000.npz`
- etc.

If `shard_size == 0`, the split may write exactly one file:

- `<dataset-reference>__train.npz`
- `<dataset-reference>__validate.npz`

---

## 7. Notebook Launcher Contract

### 7.1 Notebook file

The launcher notebook should be created at:

- `04_ROI-FCN/01_preprocessing/notebooks/00_roi_fcn_preprocessing_launcher_v0.1.ipynb`

### 7.2 Interaction model

The notebook should follow the same broad interaction style as `00_pipeline_launcher_v05.ipynb`:

- explicit selection control
- explicit visible parameters
- refresh action
- run action
- clear log/status area
- edge-only first-pass behavior

Unlike the synthetic preprocessing launcher, this ROI-FCN notebook does not need a stage selector in v0.1.
It always runs the full split sequence:

1. `bootstrap_center_target`
2. `pack_roi_fcn`

for:

1. `train`
2. `validate`

If `train` fails at any stage, the dataset run aborts and `validate` is not started.

### 7.3 Required launcher controls

The notebook must expose the following controls.

#### Dataset selection

- dataset selection widget populated from `01_preprocessing/input/`
- refresh datasets button
- derived output path display

#### Detection controls

- `detector_backend`
  - widget type: dropdown
  - default visible value: `edge_roi_v1`
  - v0.1 should expose only the edge ROI v1 path
- `edge_blur_k`
  - default: `5`
- `edge_low`
  - default: `50`
- `edge_high`
  - default: `150`
- `fg_threshold`
  - default: `250`
- `edge_pad`
  - default: `0`
- `min_edge_pixels`
  - default: `16`

#### Packing controls

- `canvas_width`
  - default: `300`
- `canvas_height`
  - default: `300`
- `shard_size`
  - default: `8192`

#### Actions and feedback

- run preprocessing button
- scrolling log/status output area
- final verdict/status block

### 7.4 Controls intentionally not exposed

To keep v0.1 scope narrow and aligned with the repository style, the notebook must not expose:

- YOLO controls
- clip policy controls
- multi-backend detector selection beyond edge ROI v1
- notebook-only business logic hidden in callback cells

### 7.5 Backend-value mapping

The user-facing backend label is:

- `edge_roi_v1`

Internally, the helper layer may translate that to the reused `rb_pipeline_v4` backend value:

- `edge`

That translation must live in helper code, not in notebook cells.

---

## 8. Required Source Layout

The preprocessing package should live under:

- `04_ROI-FCN/01_preprocessing/src/roi_fcn_preprocessing_v0_1/`

Recommended module split:

- `__init__.py`
- `config.py`
  - ROI-FCN-local config dataclasses and normalization
- `contracts.py`
  - split metadata, stage summary, packed-corpus contracts
- `paths.py`
  - root discovery and split-path resolution
- `discovery.py`
  - dataset-reference discovery from `input/`
- `validation.py`
  - input schema and split-structure validation
- `manifest.py`
  - `run.json` / `samples.csv` helpers and contract upserts
- `edge_roi_adapter.py`
  - centralized wrapper around reused `rb_pipeline_v4` detector/config behavior
- `bootstrap_center_target_stage.py`
  - stage 1 target bootstrapping
- `pack_roi_fcn_stage.py`
  - stage 2 locator packing
- `pipeline.py`
  - split sequencing and dataset-run orchestration
- `widgets_v01.py`
  - notebook widget assembly and event wiring helpers only

This mirrors the separation-of-concerns pattern already used in `02_synthetic-data-processing-v3.0` and `05_inference-v0.1`.

---

## 9. Stage 1: `bootstrap_center_target`

### 9.1 Purpose

Produce the authoritative crop-centre training target for each successful row by reusing the current edge ROI v1 algorithm.

This stage exists because the ROI-FCN functional spec explicitly allows the first-pass training corpus to be bootstrapped from the existing edge-based crop-placement system.

### 9.2 Input

For each split:

- source image from `images/`
- source row from `manifests/samples.csv`
- detector controls from the notebook

### 9.3 Reuse rule

This stage must reuse the current edge detector implementation.

Required behavior:

- instantiate `EdgeRoiDetector` via ROI-FCN-local adapter code
- map notebook controls onto the same semantics already used by `DetectStageConfigV4`
- preserve the existing meaning of blur, thresholds, padding, and minimum foreground pixels

### 9.4 Row processing rules

For each row:

1. if `capture_success != true`, mark the row `skipped`
2. load source image
3. run edge ROI v1 detection
4. if no valid detection is returned, mark the row `failed`
5. if one valid detection is returned, the authoritative crop-centre target for v0.1 is exactly the centre reported by the reused edge ROI implementation; bbox midpoint is only a fallback when explicit centre coordinates are absent from the reused detection contract
6. persist stage metadata into the split `samples.csv`

### 9.5 Stage outputs in `samples.csv`

This stage must write, at minimum:

- `bootstrap_center_target_stage_status`
- `bootstrap_center_target_stage_error`
- `bootstrap_target_algorithm`
- `bootstrap_confidence`
- `bootstrap_bbox_x1`
- `bootstrap_bbox_y1`
- `bootstrap_bbox_x2`
- `bootstrap_bbox_y2`
- `bootstrap_bbox_w_px`
- `bootstrap_bbox_h_px`
- `bootstrap_center_x_px`
- `bootstrap_center_y_px`

Required algorithm value:

- `bootstrap_target_algorithm = "edge_roi_v1"`

`bootstrap_debug_image_filename` is optional only. If present, it must be a split-root-relative path to a persisted full-frame diagnostic image for that row, and that image must show the reused detection bbox plus the authoritative target centre overlay.

### 9.6 Stage contract update

This stage must update the output split `run.json` under `PreprocessingContract` with:

- `CurrentStage: "bootstrap_center_target"`
- stage parameters for the selected detector controls
- the stage summary counts defined in section 13.5

---

## 10. Stage 2: `pack_roi_fcn`

### 10.1 Purpose

Convert successful full-frame source images plus bootstrapped centre targets into packed locator-model input corpora.

### 10.2 Input transform rules

For each successful stage-1 row:

1. load the original full-frame source image
2. convert to grayscale
3. resize while preserving aspect ratio
4. place the resized image into a fixed locator canvas
5. pad unused area
6. normalize to float32 in `[0.0, 1.0]`
7. map the bootstrapped target centre into locator-canvas coordinates using the exact same scale and padding metadata

The grayscale conversion applies to the image tensor only. Target-centre coordinates are derived from the original image-space bootstrap target and then transformed numerically; they must not be re-inferred from the grayscale or canvas-transformed image.

### 10.3 Required transform formulas

For a source image with:

- `src_w`
- `src_h`

and a locator canvas with:

- `canvas_width`
- `canvas_height`

the required mapping is:

```text
scale = min(canvas_width / src_w, canvas_height / src_h)
resized_w = round(src_w * scale)
resized_h = round(src_h * scale)

pad_left = (canvas_width - resized_w) // 2
pad_right = canvas_width - resized_w - pad_left
pad_top = (canvas_height - resized_h) // 2
pad_bottom = canvas_height - resized_h - pad_top

locator_center_x = (bootstrap_center_x_px * scale) + pad_left
locator_center_y = (bootstrap_center_y_px * scale) + pad_top
```

Target-centre coordinates are derived from the original image-space bootstrap target and then transformed numerically using the recorded scale and padding values; they are not re-inferred from the grayscale or locator-canvas image.

If the transformed locator centre falls outside the locator canvas for a stage-1-success row, the row must be marked failed rather than silently clipped.

These values must be persisted per row so the mapping is invertible and auditable.

### 10.4 Required packing behavior

The pack stage must:

- process only rows where stage 1 succeeded
- write deterministic NPZ shards into `arrays/`
- append packed-row metadata to `samples.csv`
- record canvas size, resize scale, and padding metadata per row

### 10.5 Packed row metadata in `samples.csv`

This stage must write, at minimum:

- `pack_roi_fcn_stage_status`
- `pack_roi_fcn_stage_error`
- `npz_filename`
- `npz_row_index`
- `locator_canvas_width_px`
- `locator_canvas_height_px`
- `locator_resize_scale`
- `locator_resized_width_px`
- `locator_resized_height_px`
- `locator_pad_left_px`
- `locator_pad_right_px`
- `locator_pad_top_px`
- `locator_pad_bottom_px`
- `locator_center_x_px`
- `locator_center_y_px`

### 10.6 Stage contract update

This stage must update the output split `run.json` under `PreprocessingContract` with:

- `CurrentStage: "pack_roi_fcn"`
- `CompletedStages: ["bootstrap_center_target", "pack_roi_fcn"]`
- the final packed representation contract
- the stage summary counts defined in section 13.5

---

## 11. Gaussian-Heatmap Ownership Rule

The ROI-FCN functional specification recommends Gaussian heatmap supervision in model output space.

However, preprocessing v0.1 must not hard-code topology-specific output heatmaps into the canonical packed corpus unless the topology output-space contract is already fixed and versioned.

Following `documents/generation-standards-v0.1.md`:

- preprocessing owns authoritative centre-point targets and locator-geometry metadata
- training owns topology-dependent target construction

Therefore, in preprocessing v0.1:

- the packed corpus must store the authoritative centre point in original-image space
- the packed corpus must store the corresponding centre point in locator-canvas space
- the packed corpus must store the resize/padding metadata needed for deterministic reconstruction

The training-side data loader can then generate Gaussian heatmaps in model output space without revisiting the raw images.

This keeps ownership clean and avoids contract drift between preprocessing and training.

---

## 12. NPZ Shard Contract

### 12.1 Required arrays

Each packed shard must contain the following required keys:

- `locator_input_image`
  - shape: `(N, 1, H, W)`
  - dtype: `float32`
  - values: `[0.0, 1.0]`
- `target_center_xy_original_px`
  - shape: `(N, 2)`
  - dtype: `float32`
- `target_center_xy_canvas_px`
  - shape: `(N, 2)`
  - dtype: `float32`
- `source_image_wh_px`
  - shape: `(N, 2)`
  - dtype: `int32`
- `resized_image_wh_px`
  - shape: `(N, 2)`
  - dtype: `int32`
- `padding_ltrb_px`
  - shape: `(N, 4)`
  - dtype: `int32`
- `resize_scale`
  - shape: `(N,)`
  - dtype: `float32`
- `sample_id`
  - shape: `(N,)`
  - dtype: string
- `image_filename`
  - shape: `(N,)`
  - dtype: string
- `npz_row_index`
  - shape: `(N,)`
  - dtype: `int64`

### 12.2 Strongly recommended traceability arrays

The following should also be written in v0.1 unless there is a strong implementation reason not to:

- `bootstrap_bbox_xyxy_px`
  - shape: `(N, 4)`
  - dtype: `float32`
- `bootstrap_confidence`
  - shape: `(N,)`
  - dtype: `float32`
- `locator_geometry_schema`
  - string array documenting:
    - `target_center_xy_original_px`
    - `target_center_xy_canvas_px`
    - `source_image_wh_px`
    - `resized_image_wh_px`
    - `padding_ltrb_px`
    - `resize_scale`

### 12.3 Packing rules

- `N` is the row count for that shard
- `H` and `W` are the configured locator canvas height and width
- `npz_row_index` must be contiguous within each shard: `0..N-1`
- shard contents must be validate-able without consulting notebook state

---

## 13. `run.json` PreprocessingContract

### 13.1 Contract version

The split output `run.json` must write a new preprocessing contract version:

- `ContractVersion: "rb-preprocess-roi-fcn-v0_1"`

### 13.2 Completed stage order

The canonical stage order for ROI-FCN preprocessing v0.1 is:

1. `bootstrap_center_target`
2. `pack_roi_fcn`

### 13.3 Required current representation for completed pack stage

When `pack_roi_fcn` completes, `CurrentRepresentation` must include:

- `Kind: "roi_fcn_locator_npz"`
- `StorageFormat: "npz"`
- `ArrayKeys`
  - must include all required arrays from section 12.1
- `ImageLayout: "N,C,H,W"`
- `Channels: 1`
- `CanvasWidth`
- `CanvasHeight`
- `ImageKind: "full_frame_locator_canvas"`
- `ImageColorMode: "grayscale"`
- `NormalizationRange: [0.0, 1.0]`
- `AspectRatioPolicy: "preserve_with_padding"`
- `PadValue: 0`
- `TargetType: "crop_center_point"`
- `TargetGeneration: "training_loader_gaussian_from_canvas_center"`
- `TargetSource: "edge_roi_v1_bootstrap"`
- `FixedROICropWidthPx: 300`
- `FixedROICropHeightPx: 300`

### 13.4 Required stage parameters

`Stages["bootstrap_center_target"]` must include:

- `DetectorBackend`
- `EdgeBlurKernelSize`
- `EdgeCannyLowThreshold`
- `EdgeCannyHighThreshold`
- `EdgeForegroundThreshold`
- `EdgePaddingPx`
- `EdgeMinForegroundPx`
- `EdgeCloseKernelSize`

In v0.1, `EdgeCloseKernelSize` should default to `1` internally and does not need a notebook control.

`Stages["pack_roi_fcn"]` must include:

- `CanvasWidth`
- `CanvasHeight`
- `ShardSize`
- `NormalizationMode`
- `PadValue`
- `TargetStorageMode`

Required values:

- `NormalizationMode = "zero_to_one_float32"`
- `TargetStorageMode = "point_only_with_geometry_metadata"`

### 13.5 Required stage summary counts

`PreprocessingContract["StageSummaries"][stage_name]` must exist for each completed stage.

For each stage, record:

- `TotalRowsSeen`
- `SkippedRows`
- `FailedRows`
- `SucceededRows`

These counts must be final stage totals written to `run.json`, and they must satisfy:

```text
TotalRowsSeen = SkippedRows + FailedRows + SucceededRows
```

---

## 14. Tests and Verification

The build must add tests under:

- `04_ROI-FCN/01_preprocessing/tests/`

Required coverage:

- dataset discovery only lists valid dataset references containing both `train` and `validate`
- malformed split structures fail loudly
- edge ROI adapter preserves the current edge ROI parameter meanings
- `bootstrap_center_target` writes the expected centre and bbox metadata for fixture images
- `pack_roi_fcn` preserves deterministic resize/pad mapping into locator-canvas coordinates
- for a known fixture row, stored `target_center_xy_canvas_px` matches the numerically transformed original target centre under the recorded resize/padding metadata
- shard writer emits the required NPZ keys, dtypes, and shapes
- end-to-end integration test processes a tiny two-split fixture dataset in `train -> validate` order
- if `train` fails at any stage, `validate` is not started

If a notebook smoke test is added, it is supplementary only. Canonical behavior must remain covered by Python tests.

---

## 15. Acceptance Criteria

The ROI-FCN preprocessing build is only complete when all of the following are true:

1. The notebook can discover and explicitly select a dataset reference from `01_preprocessing/input/`.
2. The selected dataset is rejected unless both `train` and `validate` split corpora are valid.
3. The preprocessing run executes `train` first and `validate` second.
4. If `train` fails at any stage, the dataset run aborts and `validate` is not started.
5. Stage 1 bootstraps crop-centre targets using the existing edge ROI v1 behavior rather than a local reimplementation.
6. Stage 2 packs grayscale full-frame locator inputs with explicit resize and padding metadata.
7. For a known fixture row, stored `target_center_xy_canvas_px` matches the numerically transformed original target centre under the recorded resize/padding metadata.
8. The output is written under `01_preprocessing/output/<dataset-reference>/` with split-level manifests and NPZ shards.
9. `run.json` records per-stage summary counts for total, skipped, failed, and succeeded rows.
10. `run.json` and `samples.csv` remain the canonical split-level authority files.
11. The packed representation contract is explicit, versioned, and test-covered.
12. The notebook remains a UI layer rather than becoming the home of canonical preprocessing logic.

---

## 16. Bottom Line

ROI-FCN preprocessing v0.1 is a two-stage, edge-bootstrapped corpus builder.

It must:

- discover one dataset reference containing `train` and `validate`
- reuse the current edge ROI v1 algorithm to generate authoritative crop-centre targets
- pack full-frame grayscale locator inputs plus deterministic geometry metadata into NPZ shards
- abort the dataset run before `validate` if `train` fails at any stage
- write explicit split-level contracts that future ROI-FCN training and inference code can trust

It must not:

- invent a second ROI algorithm
- hide preprocessing rules in notebook cells
- blur the boundary between preprocessing and topology-specific training behavior
