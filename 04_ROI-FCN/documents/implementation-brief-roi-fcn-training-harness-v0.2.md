# Implementation brief - ROI FCN training harness v0.2

## 1. Objective

Implement a training harness for an ROI FCN whose sole task is to predict the centre point around which a fixed ROI should be extracted from a full-frame image.

The model must output **crop-centre location only**. It must **not** predict box size, class, orientation, silhouettes, or downstream distance/orientation targets.

The operational output contract of this stage is:

* `center_x`
* `center_y`

in original full-frame image coordinates.

---

## 2. Scope boundary

This harness is for:

* training
* validation
* evaluation
* artifact generation
* notebook control of the ROI FCN stage only

It must **consume the already-implemented preprocessing pipeline outputs and contract**. It must not re-implement preprocessing logic inside the training repo.

It must not implement:

* preprocessing-stage edge ROI bootstrapping
* alternative crop-target generation logic
* downstream distance/orientation regression
* bbox-size prediction
* orientation regression
* segmentation training
* multi-object handling
* missing-object detection
* multi-scale detector machinery

---

## 3. Packed training data contract the harness must support

The harness must train from the packed ROI FCN preprocessing output, not from raw images.

### 3.1 Expected locator input representation

The training harness must consume the locator input representation **as recorded in the packed training corpora**, rather than anchoring to a hard-coded canvas size, resize scale, or padding pattern.

For each loaded corpus, the harness must treat the preprocessing output as authoritative and must read and validate the packed locator geometry contract from the corpus metadata and arrays, including at minimum:

- `locator_canvas_width_px`
- `locator_canvas_height_px`
- `locator_resize_scale`
- `locator_resized_width_px`
- `locator_resized_height_px`
- `locator_pad_left_px`
- `locator_pad_right_px`
- `locator_pad_top_px`
- `locator_pad_bottom_px`

The harness must support whatever locator canvas dimensions are defined by the training corpora, provided the corpus contract is internally consistent and matches the ROI FCN preprocessing representation expected by the harness.

The harness must **not** silently assume a specific source-image size, a specific resize factor, a specific padding pattern, or a specific locator canvas size. Instead, it must validate the corpus contract and then use the recorded locator geometry as the single authority for:

- input tensor shape
- target construction in model-space
- decode back to original-image coordinates
- evaluation and reporting

If the corpus geometry is missing, malformed, internally inconsistent, or incompatible with the expected ROI FCN preprocessing contract, the harness must fail loudly rather than guessing.

---

### 3.2 Required packed arrays

The harness must load and validate the packed NPZ contract produced by preprocessing, including at minimum:

- `locator_input_image`
- `target_center_xy_original_px`
- `target_center_xy_canvas_px`
- `source_image_wh_px`
- `resized_image_wh_px`
- `padding_ltrb_px`
- `resize_scale`
- `sample_id`
- `image_filename`
- `npz_row_index`

Where present, the harness should also load and use:

- `bootstrap_bbox_xyxy_px`
- `bootstrap_confidence`

The harness must treat the packed locator tensor shape in `locator_input_image` as authoritative for the model input contract of that corpus, subject to explicit validation against the recorded geometry metadata.

---

### 3.3 Required per-row metadata support

The harness must be compatible with the split manifest metadata already produced by preprocessing, including fields such as:

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
- `bootstrap_bbox_x1`
- `bootstrap_bbox_y1`
- `bootstrap_bbox_x2`
- `bootstrap_bbox_y2`

These fields must be treated as authoritative corpus metadata for geometry reconstruction and evaluation, not as advisory annotations.

---

### 3.4 Contract validation

The harness must fail loudly if the packed corpus does not match the expected ROI FCN preprocessing contract.

At minimum validate:

- required NPZ keys
- expected dtypes
- expected image layout
- expected channel count
- consistency between `locator_input_image` shape and recorded locator canvas metadata
- consistency between `target_center_xy_canvas_px` and the recorded resize/padding metadata where applicable
- presence of required geometry metadata

The harness may support multiple locator canvas sizes across different corpora, but it must not silently mix incompatible corpus contracts inside a single training run unless that behavior is explicitly designed, validated, and surfaced.

---

### 4. Model family and topology support

Implement a new ROI FCN topology family in the training repo as a **tiny fully convolutional spatial localiser**.

#### 4.1 Required first-pass model shape

The required first-pass shape is:

- single-stream
- single-channel input
- single-head
- heatmap-producing
- centre-only

v0.1 must stay deliberately narrow:

- no anchors
- no class head
- no box head
- no segmentation branch
- no orientation head
- no multi-task coupling

#### 4.2 Primary model output

The raw model output is a **single centre-likelihood heatmap**.

This heatmap is the primary model output artifact during training and evaluation.

#### 4.3 Output-space ownership

The topology definition must expose enough information for target generation and decode to operate correctly in model output space.

The harness must not guess output-space geometry.

For v0.1, this may be done by either:

- exposing the model output stride / output shape contract explicitly from the topology, or
- deriving output shape deterministically from a forward pass on a dummy tensor matching the validated corpus input shape

The target generator and decoder must use the same output-space contract.

The topology must operate over the validated locator input shape defined by the loaded corpus contract rather than assuming a fixed hard-coded canvas size in the training harness.

---

### 5. Target generation support

The harness must implement **Gaussian heatmap supervision** as a required part of the learning contract.

For each training sample, it must:

1. take the intended crop centre from the packed corpus
2. use the stored **canvas-space centre**
3. map that centre into model output space
4. generate a Gaussian target heatmap in output space

This must be used instead of a single hard positive pixel.

#### 5.1 Target semantics

The supervision target is the **authoritative crop centre** from the preprocessing corpus.

It is the intended centre for useful ROI placement.

Deliberate jitter is **out of scope** for ROI FCN supervision in v0.1.

#### 5.2 ROI size does not redefine the FCN target

Notebook-supplied ROI width and ROI height must **not** redefine the FCN supervision target in v0.1.

The FCN remains a centre-only predictor.

ROI width and height are for:

- derived ROI reporting
- ROI usefulness evaluation
- operational inspection

not for changing the underlying centre target semantics.

#### 5.3 Geometry authority for target construction

Target construction must use the validated corpus geometry as authority.

The harness must not assume a fixed resize factor, fixed padding pattern, or fixed locator canvas size when mapping packed centre targets into model output space.

---

## 6. Loss support

Use a simple first-pass loss consistent with the Gaussian heatmap target contract.

### Required v0.1 loss

* **MSE loss** between predicted heatmap and Gaussian target heatmap

This is the default required first-pass loss unless you discover a clear implementation blocker.

Loss configuration should be recorded in the run config.

---

### 7. Decode and post-processing support

The harness must include deterministic decode logic that turns the predicted heatmap into:

- `center_x`
- `center_y`

in original full-frame image coordinates.

#### 7.1 Required decode path

Decode must map:

1. model output heatmap space
2. back through output stride / output geometry
3. back into packed locator canvas space
4. back through resize scale and padding metadata
5. into original full-frame coordinates

#### 7.2 Peak decode rule

For v0.1, decode using:

- **argmax over the predicted heatmap**

Optional confidence may later be surfaced as:

- maximum predicted heatmap value

#### 7.3 Derived ROI bounds

The harness must derive ROI bounds **outside the model** using notebook-supplied ROI width and ROI height.

The FCN remains centre-only.

Derived ROI bounds are a downstream operational convenience and evaluation construct, not a second model head.

#### 7.4 Geometry authority for decode

Decode must use the recorded corpus geometry metadata for each sample rather than relying on hard-coded assumptions about locator canvas size, resize scale, source-image size, or padding pattern.

---

## 8. Notebook control surface

Add a dedicated notebook control surface for ROI FCN training, aligned with the repo’s existing pattern where notebooks are thin operator-facing launch surfaces and implementation logic lives in `src`.

The notebook must expose at minimum:

* training dataset selection
* validation dataset selection
* topology / variant selection
* run/output location controls
* train/validation execution controls
* ROI width input
* ROI height input

### Important constraint

ROI width and ROI height inputs must drive:

* derived ROI reporting
* ROI usefulness evaluation
* any visual overlays or summaries that depend on the fixed crop size

They must **not** change the FCN output contract from centre-only to bbox prediction.

---

### 9. Evaluation and reporting requirements

The harness must report both:

- raw optimisation behaviour
- operational localisation usefulness

This stage exists to place a useful fixed crop, so evaluation must reflect ROI placement usefulness, not just abstract heatmap loss.

#### 9.1 Required evaluation outputs

At minimum, evaluation output must include:

- train loss history
- validation loss history
- decoded `center_x` / `center_y` predictions
- prediction-vs-target exports
- per-run artifacts written into a normal run directory
- visual outputs that make heatmap behaviour inspectable
- visual outputs that make decoded centre behaviour inspectable
- metrics reflecting centre-point error in original full-frame coordinates
- metrics reflecting whether the decoded centre produces a useful ROI for downstream use

#### 9.2 Required v0.1 localisation metrics

At minimum include:

- mean centre error in original-image pixels
- median centre error in original-image pixels
- p95 centre error in original-image pixels

#### 9.3 Required v0.1 ROI usefulness metric

Because the preprocessing corpus includes bootstrapped bbox metadata, include at least one operational crop-usefulness metric.

Required first-pass metric:

- **ROI full-containment success rate**:
    - derive the ROI from decoded centre plus notebook ROI width/height
    - measure whether that ROI fully contains the bootstrapped bbox for the sample

Where bbox metadata is unavailable for a sample, report that clearly rather than silently excluding it.

This metric should be aggregated at least as:

- overall success rate
- optional split by train / validation

#### 9.4 Geometry-aware evaluation requirement

All decode, error, and ROI-usefulness evaluation must be computed using the validated per-sample corpus geometry metadata rather than assuming a fixed locator canvas size or fixed resize pattern across all corpora.

---

## 10. Run artifact requirements

The training harness should produce a normal structured run artifact set consistent with the repo’s established experiment style, including:

* run configuration
* metrics/history
* model weights/checkpoints
* prediction exports
* plots
* summary / model-card-style metadata for the ROI FCN family

The run outputs must make both levels of output inspectable:

### Raw model output

* centre-likelihood heatmap

### Usable pipeline output

* decoded crop centre in original full-frame coordinates

That distinction is part of the explicit output contract and must remain visible in artifacts.

---

## 11. Output contract the harness must enforce

The harness must treat the operational model output as:

* `center_x`
* `center_y`

in original full-frame camera pixel coordinates

Optional confidence may be surfaced later, but is not required for v0.1.

The harness must **not** redefine success around:

* silhouette centroid
* generic object box centre unrelated to the packed target definition
* orientation
* bbox size

Success remains defined around useful centre prediction for fixed-ROI placement.

---

## 12. Explicit non-goals for this implementation

Do not implement, in this v0.1 harness:

* bounding-box size prediction
* class prediction
* orientation regression
* multi-object handling
* missing-object detection
* segmentation training
* multi-scale detector machinery
* fancy preprocessing branches
* alternative target bootstrapping logic
* downstream distance/orientation regression coupling

Those are outside the defined role of the ROI FCN training stage.

---

### 13. Bottom line

This harness must train a tiny FCN that consumes the packed full-frame locator inputs already produced by preprocessing, learns the authoritative crop-centre target via Gaussian heatmap supervision, decodes predictions back into original full-frame coordinates using the recorded corpus geometry, and evaluates success in terms of whether the resulting fixed ROI would actually be useful downstream.


