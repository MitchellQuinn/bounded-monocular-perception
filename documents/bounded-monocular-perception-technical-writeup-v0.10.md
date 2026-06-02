# Bounded Monocular Perception System - Technical Writeup v0.10

**Current as of:** 2026-06-02  

## 1. Project Overview

This repository is a bounded computer-vision and applied-machine-learning
workspace for estimating vehicle distance and yaw from a fixed monocular camera
view under controlled conditions.

The system is deliberately scoped around one known vehicle family, one fixed
camera geometry, a constrained movement plane, synthetic supervision,
controlled full-frame captures, and live-local runtime testing under controlled
physical conditions. The engineering question is narrow:

> Can a fixed-camera system observing a known vehicle in a constrained scene
> estimate useful vehicle state from image-based geometric cues, and what breaks
> when the system moves from offline synthetic validation into composed live
> inference?

That framing is central to the repository. It should be read as a bounded
research-engineering workspace rather than a packaged product, benchmark
system, or broad general-purpose vision model. The value is the engineering
record: applied ML engineering, computer-vision system construction,
synthetic-data generation, preprocessing contracts, runtime composition, trace
capture, calibration support, failure analysis, and measuring degradation
between offline validation and composed live inference.

The main change in v0.10 is that Incident 005 is now a first-class part of the
technical narrative. After locator and foreground failures were made more
inspectable, the project found strong evidence that the remaining live
underprediction is largely explained by live/synthetic apparent-scale mismatch.
The current live runtime therefore includes a configurable post-foreground model
representation transform. A first follow-up sweep improved all-row mean signed
error to `-0.113 m` and mean absolute error to `0.118 m`; a later
three-distance sweep improved further to mean signed error `-0.033 m` and mean
absolute error `0.080 m`. That latest sweep is good at `2.20 m` and `2.90 m`,
but the `1.60 m` close-range rows still average `-0.145 m`, so the calibrated
live-distance claim remains bounded until the calibrated scale-validation loop
is complete.

## 2. Problem Scope

The core task is intentionally constrained:

* **Input:** a monocular full-frame image from a fixed camera.
* **Primary output:** vehicle distance in metres.
* **Secondary output:** vehicle yaw/orientation.
* **Runtime support task:** locate and construct the crop/foreground
  representation required by the selected model.

The current system assumes:

* one known vehicle family
* one fixed camera geometry
* one constrained movement plane
* synthetic training and validation data
* controlled full-frame live captures
* a bounded live-local camera setup rather than open-world deployment

These assumptions keep the problem falsifiable. They also make it possible to
separate model performance from failures introduced by localisation,
preprocessing, camera geometry, foreground extraction, apparent-scale alignment,
and live runtime composition.

## 3. Repository Architecture

The repository is organised as a versioned multi-project workspace:

* `01_rb_synthetic-data_3`: Unity/C# synthetic image generation, including
  distance/yaw targets and experimental Defender amodal keypoint labels.
* `02_synthetic-data-processing-v4.0`: OpenCV and NumPy preprocessing,
  detection, silhouette generation, foreground enhancement, and dual-stream /
  tri-stream packing.
* `03_rb-training-v2.0`: PyTorch training, topology registry, model evaluation,
  resume support, reporting, and experimental amodal keypoint pose topology
  work.
* `04_ROI-FCN`: preprocessing and training for crop-centre heatmap
  localisation.
* `05_inference-v0.3-ds`: raw-image inference using ROI-FCN plus dual-stream
  distance/yaw models.
* `05_inference-v0.4-ts`: tri-stream-facing inference work and
  brightness-analysis tooling.
* `06_live-inference_v0.1`: first live-local runtime with camera input, frame
  handoff, model registry, preprocessing, workers, and GUI.
* `06_live-inference_v0.2`: richer live diagnostics, trace capture, background
  handling, and ROI-FCN visualisation work.
* `06_live-inference_v0.3`: current live-local runtime with generic locator
  interfaces, deterministic background/edge localisation, manual masks,
  selectable foreground extraction, component-aware threshold foreground
  selection, camera-intrinsics preprocessing, post-foreground model
  representation transforms, trace evidence, and GUI controls.
* `charuco-calibration`: PySide6/OpenCV ChArUco calibration tool for capturing
  pose-diverse calibration frames, solving camera intrinsics, and exporting
  JSON/YAML calibration artifacts.
* `failure-analysis`: failure-analysis framework, model-evaluation reports,
  live-runtime incident investigations, and supporting evidence.

This layout reflects a research-engineering codebase moving from offline
experiments toward runtime composition. It is not packaged as a finished
product, but it contains real integration surfaces, tests, artifacts, runtime
contracts, and incident-analysis material.

## 4. Synthetic Data Generation

The Unity generator creates full-frame synthetic images with structured run
metadata and sample manifests. It is designed to produce controlled labelled
data with controlled coverage for the fixed-camera perception task.

Key generator components include:

* `CaptureService.cs` for render-texture capture.
* `RunControllerBehaviour.cs` for batch orchestration, cancellation, manifest
  flushing, and attempt-budget handling.
* `ManifestWriter.cs`, `ManifestRowMapper.cs`, and `RunMetadataWriter.cs` for
  traceable run outputs.
* `FileNamingStrategy.cs` for deterministic sample naming.
* `DistanceCalculator.cs` for explicit target derivation.
* `StratifiedPlacementPlanner.cs` for camera-footprint-aware placement.
* `VehicleProjectionValidator.cs` for image-space feasibility checks.
* `DefenderAmodalKeypointPoseTargetBuilder.cs` for experimental camera-space
  Defender centre, amodal keypoint, and visibility target generation.

The placement strategy is an important design choice. Rather than sampling
arbitrary world positions, the generator projects the camera footprint onto the
movement plane, divides the usable footprint into depth and lateral cells,
validates projected vehicle bounds in image space, and redistributes quota when
cells exhaust their attempt budget.

The generator writes:

* `images/*.png`
* `manifests/run.json`
* `manifests/samples.csv`
* `runlog.txt`

The sample manifest also carries Defender keypoint schema metadata,
camera-space centre labels, ten fixed camera-space 3D keypoints, and
per-keypoint visibility labels. This gives downstream stages explicit lineage
and typed target metadata rather than relying on filenames alone.

Incident 005 adds an important caveat to synthetic generation and rendering:
synthetic images must not merely contain plausible labelled targets. For a direct
monocular distance regressor, synthetic and live captures must also agree on
what apparent image scale corresponds to the same physical reference distance.
That apparent-scale contract is now an explicit validation boundary.

## 5. Preprocessing and Representation Design

The preprocessing layer is contract-driven and stage-based. The v4 pipeline
supports:

* `detect`: edge-based or detector-style vehicle localisation metadata.
* `silhouette`: ROI silhouette generation with contour processing and
  convex-hull fallback.
* `pack_dual_stream`: distance/yaw regression inputs with geometry features.
* `pack_tri_stream`: separate distance image, orientation image, and geometry
  streams.
* foreground enhancement and brightness-normalisation options.
* corpus shuffle and notebook control surfaces for repeated training workflows.

The strongest representation choices are:

* fixed ROI canvases for distance inference.
* no rescaling in the distance stream, preserving apparent object size as a
  depth cue.
* a 10-element geometry vector: `cx_px`, `cy_px`, `w_px`, `h_px`, `cx_norm`,
  `cy_norm`, `w_norm`, `h_norm`, `aspect_ratio`, `area_norm`.
* circular yaw targets represented as `sin/cos`.
* optional foreground-only brightness normalisation.
* foreground enhancement for grayscale-on-white model representations.
* a tri-stream contract that separates distance evidence from orientation
  evidence.

The current live-selected direct distance/yaw model uses the
`rb-preprocess-v4-tri-stream-grayscale-white-v1` contract. It keeps the
tri-stream input keys and uses `320 x 320` canvases with a grayscale
vehicle-on-white representation:

* `x_distance_image`: fixed unscaled ROI canvas.
* `x_orientation_image`: target-centred image scaled by foreground extent.
* `x_geometry`: 10-field foreground bounding-box vector.
* `y_distance_m`: scalar distance target.
* `y_yaw_deg`, `y_yaw_sin`, `y_yaw_cos`: orientation targets.

The split remains technically meaningful. Distance benefits from preserving
apparent scale, while yaw benefits from a target-centred orientation view.

The same design choice also creates a strict live/synthetic alignment
requirement. If a live target appears `1.20x` to `1.24x` larger than its
synthetic counterpart at the same nominal distance, the direct distance regressor
can coherently interpret the live target as closer. Incident 005 is the current
evidence for that failure mode.

## 6. Model Training and Evaluation

The training code is organised around reusable Python modules rather than
notebook-only logic. It includes:

* topology contracts and a topology registry.
* dataset summaries and preprocessing-contract validation.
* shard-based NPZ loading.
* RAM-aware shard caching.
* split-overlap checks.
* checkpointing and resume-state support.
* model cards, run manifests, plots, metrics, and sample predictions.
* task-aware reporting for scalar distance and multitask distance-plus-yaw
  outputs.

Model families represented in implemented code include:

* baseline full-frame CNN distance regression.
* dual-stream crop-plus-geometry distance regression.
* dual-stream distance-plus-yaw regression.
* tri-stream distance-plus-yaw regression.
* ROI-FCN crop-centre localisation.
* experimental Defender amodal keypoint pose regression.

Yaw is modelled through circular regression using `sin/cos` targets rather than
direct angle regression. The training runtime resolves prediction heads and
target heads from topology contracts, allowing the loss and reporting paths to
work across scalar and multitask models.

The current direct distance/yaw training family includes `tri_stream_yaw_v0_5`.
This variant keeps the existing tri-stream input contract but changes the
distance/yaw coupling:

* a camera trunk predicts the base distance from distance-image and geometry
  features.
* a yaw trunk predicts orientation from geometry, distance features, camera
  features, and orientation-image features.
* a bounded pose-conditioned distance residual uses detached yaw context and
  normalized geometry derivatives.
* the residual is limited with `tanh`, with a default residual limit of
  `0.35 m`.

That design was an attempt to protect distance prediction from unstable yaw
features while still allowing pose-conditioned corrections. Live incident
evidence shows that it is a useful baseline, but not a complete solution to
pose-linked or projection-linked distance bias.

The first experimental keypoint topology is registered as
`defender_amodal_keypoint_pose` with variant
`defender_amodal_keypoint_pose_v0_1`. It reuses the tri-stream input family and
adds heads for distance, yaw, Defender centre, flattened 3D keypoints, and
visibility logits. The training task runtime supports the corresponding
distance, orientation, centre, keypoint, and visibility losses and metrics. This
is an implementation milestone, not yet a selected live model artifact or a
real-camera accuracy result.

## 7. ROI-FCN and Runtime Localisation

The ROI-FCN subsystem turns crop placement into a learned task. It is trained to
predict the centre of the fixed ROI required by the downstream
distance/orientation model.

The ROI-FCN preprocessing path preserves:

* full-frame grayscale locator input.
* target centre in original-image coordinates.
* target centre in locator-canvas coordinates.
* resize scale.
* padding offsets.
* optional bootstrap box metadata.

The validated ROI-FCN artifact `260420-1219_roi-fcn-tiny__run_0003` uses:

* topology id: `roi_fcn_tiny`.
* topology variant: `tiny_v1`.
* locator canvas: `480 x 300`.
* downstream ROI crop: `300 x 300`.
* supervision: Gaussian heatmap.
* decode: deterministic argmax back into source-image coordinates.
* training split: `100,000` samples.
* validation split: `20,000` samples.

Its validation metrics are:

| Metric | Value |
| --- | ---: |
| Mean centre error | `3.1757 px` |
| Median centre error | `2.4354 px` |
| p95 centre error | `7.7098 px` |
| ROI full-containment success | `0.9891` |

The live locator history is important. Live inference v0.2 used ROI-FCN as
the active live ROI locator: it resized the grayscale camera frame into a fixed
locator canvas, ran the learned heatmap model, decoded the argmax to one
source-image centre, and extracted a fixed ROI around that point.

Live inference v0.3 changed the default locator to `background_edge_v1`, an
inspectable deterministic geometric locator built for the controlled
fixed-camera path. ROI-FCN is retained as `roi_fcn_legacy`, an explicit
comparison/fallback route, and the current model selection file references
`260516-1714_roi-fcn-tiny__run_0002` for that route.

Incident 004 records the engineering justification for this pivot. The issue was
not simply that ROI-FCN could miss a centre point. The deeper boundary mismatch
was that ROI-FCN compressed live ROI selection into a learned heatmap peak, while
the runtime needed an auditable apparent-scale measurement path: foreground mask,
edge map, candidate contours, chosen bbox, ROI crop bounds, rejection reasons,
and traceable artifacts. This matters because downstream distance depends on
crop boundaries, foreground quality, apparent target scale, and geometry, not
only on whether a centre point looks plausible.

## 8. Raw-Image Inference

The repository contains raw-image inference paths that compose separately
trained components:

1. a crop or ROI locator
2. a distance/yaw regression model
3. preprocessing logic that reconstructs the input representation expected by
   the selected model
4. JSON and image artifacts for inspection and failure analysis

The v0.3 dual-stream path runs the ROI-FCN localiser, extracts a fixed ROI,
generates the dual-stream model input, derives geometry features, and writes
per-sample predictions with actual distance/yaw values and deltas.

The v0.4 tri-stream-facing path extends the same runtime family toward the
tri-stream model contract, including separate distance and orientation inputs,
orientation source-mode handling, brightness analysis, and foreground
representation work.

This stage exposed an important system-level gap: some offline training metrics
are much stronger than composed raw-image runtime metrics. That gap is useful
engineering evidence. The project does not stop at preprocessed validation
performance; it measures degradation introduced by crop localisation, runtime
preprocessing, camera alignment, model representation construction, and model
composition.

## 9. Calibration, Intrinsics, and Representation Alignment

The `charuco-calibration` project provides a standalone PySide6/OpenCV workflow
for camera calibration. It captures pose-diverse ChArUco frames, scores capture
quality, tracks pose diversity, solves camera intrinsics, reports per-view
reprojection errors, and exports JSON/YAML calibration artifacts.

The calibration project is intentionally separate from inference. It does not
use ROI-FCN, distance/yaw models, synthetic-data preprocessing, background
removal, masking, or live inference logic. Its boundary layer is
`rb_camera_calibration/contracts.py`, which defines serialisable dataclasses,
enums, and protocols for board config, camera config, detections, quality
metrics, capture decisions, calibration results, worker state, and exported
artifacts.

The live v0.3 runtime contains calibration-backed camera intrinsics transforms.
Supported modes are:

* `disabled`
* `real_to_unity_intrinsics_remap`
* `real_undistort_only`

The live project includes:

* a real AR0234 ChArUco calibration artifact under
  `06_live-inference_v0.3/config/calibration/260519-1501_calibio_charuco_30mm_a4`.
* an analytic Unity AR0234 pinhole target calibration under
  `06_live-inference_v0.3/config/calibration/260520-1130_unity_ar0234_pinhole_1920x1200`.

The camera-intrinsics transform can undistort the real camera frame or remap it
toward the Unity-trained camera model before normal localisation, preprocessing,
geometry extraction, and model inference. Incident 002 showed that this improves
part of the runtime alignment problem, but does not by itself remove
pose-dependent live distance error.

Incident 005 adds a second alignment layer: the **model representation
transform**. This is not the same as the camera-intrinsics transform.

```text
camera intrinsics transform:
  full frame -> transformed full frame before locator and preprocessing

model representation transform:
  post-foreground ROI representation -> model-facing ROI representation
  before x_distance_image, x_orientation_image, and x_geometry are packed
```

The first implementation is an affine ROI-space transform with independent
horizontal and vertical scale factors. It can be configured by TOML, applies the
same transform to the model ROI representation, orientation source, and
foreground mask, then recomputes model-space foreground geometry from the
transformed mask.

The current Incident 005 AR0234 profile uses the inverse of the measured
live/synthetic apparent-scale ratios:

| Scale source | Value |
| --- | ---: |
| Measured mean live/synthetic width scale | `1.238x` |
| Measured mean live/synthetic height scale | `1.210x` |
| Transform `scale_x` | `0.8077544426494346` |
| Transform `scale_y` | `0.8264462809917356` |

The real OpenCV/V4L2 camera path currently loads that Incident 005 profile by
default unless a transform config override is supplied. Trace metadata records
whether the transform was enabled and applied, the scale factors, the anchor,
the affine matrix, raw foreground geometry, and model-space foreground geometry.

This is an initial mitigation and observability hook. The latest three-distance
follow-up sweep improved all-row mean absolute error to `0.080 m`, but it should
not be described as a validated live accuracy fix until the synthetic/live scale
check, repeat trace-backed sweeps, and calibrated distance-reference checks are
complete.

## 10. Live-Local Inference Runtime

The `06_live-inference_v0.3` project is the current live-local runtime and demo
stabilisation layer. It includes:

* model selection files for live-local artifacts.
* metadata-only model manifest loading.
* compatibility checks between selected artifacts and runtime contracts.
* device policy handling for `auto`, `cuda`, and `cpu`.
* atomic latest-frame file handoff.
* duplicate-frame detection through frame hashes.
* synthetic camera publisher for deterministic GUI smoke tests.
* OpenCV/V4L2 camera source targeting `/dev/video0`.
* deterministic `background_edge_v1` locator.
* generic `LocatorResult` metadata with accepted state, candidates, chosen bbox,
  ROI bounds, clipping/content metadata, warnings, and rejection reasons.
* retained ROI-FCN legacy locator path.
* manual fixed ROI and fixed-centre fallback locator paths.
* static background capture and clear controls.
* manual mask drawing, erasing, application, and clearing.
* camera intrinsics mode selection and preview/background transform handling.
* foreground extraction policy state with `threshold_foreground_v1` as default
  and `silhouette_contour_v2` retained as a legacy selectable path.
* component-aware threshold foreground selection with diagnostic metadata for
  foreground/locator disagreement.
* post-foreground model representation transform support.
* raw foreground and model-space foreground geometry metadata.
* diagnostic foreground ROI border-touch metadata.
* tri-stream live preprocessor.
* PyTorch tri-stream inference engine.
* camera and inference workers.
* PySide6 GUI with camera/inference controls, preview panes, status readouts,
  prediction outputs, timing, logs, and debug artifact display.

The current v0.3 distance/orientation selection is:

* distance/orientation: `260521-1029_ts-2d-cnn`
* topology id: `distance_regressor_tri_stream_yaw`
* topology variant: `tri_stream_yaw_v0_5`
* preprocessing contract: `rb-preprocess-v4-tri-stream-grayscale-white-v1`
* input mode: `tri_stream_distance_orientation_geometry`
* representation kind: `tri_stream_npz`
* input shape: `[1, 320, 320]`
* output keys: `distance_m`, `yaw_sin_cos`

The selected model's offline validation metrics are:

| Metric | Value |
| --- | ---: |
| Training samples | `300,000` |
| Validation samples | `60,000` |
| Distance MAE | `0.015856 m` |
| Distance RMSE | `0.026030 m` |
| Distance within `0.10 m` | `0.998483` |
| Distance within `0.25 m` | `0.999750` |
| Distance within `0.50 m` | `0.999917` |
| Yaw mean error | `1.503031 deg` |
| Yaw median error | `1.148117 deg` |
| Yaw p95 error | `3.740643 deg` |
| Yaw within `5 deg` | `0.985433` |

These are offline synthetic validation metrics for the selected model artifact.
They should not be interpreted as real-camera deployment accuracy.

## 11. Incident 001: Live Distance Spike Investigation

The `failure-analysis/incidents/incident-001-live-distance-regression-spike`
directory records a live-inference failure, reconstructs the pipeline state from
artifacts, identifies the root cause, implements remediation, and records
post-remediation traces.

The incident began with two near-identical live captures of a stationary vehicle
model producing sharply different distance estimates:

| Signal | Failing trace | Passing trace |
| --- | ---: | ---: |
| Measured physical reference distance | `~1.33 m` | `~1.33 m` |
| Predicted distance | `5.3009257 m` | `1.5157189 m` |
| Predicted yaw | `32.1212 deg` | `114.8822 deg` |
| ROI locator bbox | `[793, 847, 1043, 1149]` | `[794, 847, 1043, 1160]` |
| ROI locator confidence | `0.9151` | `0.9135` |
| ROI crop size | `320 x 320 px` | `320 x 320 px` |
| Silhouette area | `123 px` | `45,435 px` |
| Silhouette bbox | `[956, 946, 970, 966]` | `[793, 848, 1042, 1160]` |
| Geometry width / height | `14 x 20 px` | `249 x 312 px` |
| Geometry area norm | `0.0001215` | `0.0337187` |

The trace evidence showed that the model was not the primary source of the
spike. The accepted camera frame and ROI crop contained the vehicle, but the
downstream silhouette recovery reduced the failing trace to a tiny fragment.
That fragment then drove `x_distance_image`, `x_orientation_image`, and
`x_geometry`.

The root cause was the intensity-based silhouette recovery policy. When the
valid vehicle component touched the bottom border of the ROI, the recovery
heuristic preferred a tiny non-border component and returned early. A much
larger border-touching vehicle component was present but not selected.

The remediation had several parts:

* The contour/silhouette recovery selector was changed so border-touching
  components are scored together with non-border candidates.
* The live default foreground path was changed to `threshold_foreground_v1`.
* The old contour/silhouette path remains selectable as `silhouette_contour_v2`
  but is no longer the default live path.
* A locator-relative consistency check was added and later evolved into
  diagnostic metadata rather than a hard rejection path.
* Regression tests were added for the v4 silhouette algorithm and the live
  preprocessor using incident artifacts.

Post-remediation live traces from 2026-05-18 showed the new foreground path
producing vehicle-sized masks and plausible distance predictions:

| Frame hash prefix | Predicted distance | Foreground extraction | Foreground pixels | Foreground bbox | Locator confidence |
| --- | ---: | --- | ---: | ---: | ---: |
| `86683b78` | `2.0420 m` | `threshold_foreground_v1` | `16,742 px` | `126 x 175 px` | `0.8967` |
| `cc24cd51` | `1.7722 m` | `threshold_foreground_v1` | `29,830 px` | `199 x 231 px` | `0.9059` |
| `1d2d137e` | `1.5386 m` | `threshold_foreground_v1` | `49,486 px` | `292 x 225 px` | `0.9307` |

The incident demonstrates why trace bundles matter. The failure was localised
precisely:

```text
camera frame: good
ROI locator: good
ROI crop: good
silhouette recovery: failed
model input: corrupted
model output: explainable from corrupted input
```

## 12. Incident 002: Pose-Dependent Distance Bias

The `failure-analysis/incidents/incident-002-pose-dependent-distance-bias`
directory records a repeatable live-camera distance regression error in the
direct tri-stream distance/yaw model family.

The observed failure mode was pose-linked distance instability. At fixed
measured floor positions, predicted distance varied systematically with vehicle
pose. Front-facing views often predicted farther than side or rear views at the
same floor mark, rear-facing views often predicted closer, and side-facing views
were usually intermediate or closest to the measured reference distance.

Four measured positions were tested:

| Mark | Measured distance |
| ---: | ---: |
| 1 | `1.59 m` |
| 2 | `1.77 m` |
| 3 | `1.97 m` |
| 4 | `2.18 m` |

An input-space camera-model correction was applied using AR0234 calibration data
and an equivalent Unity camera model. The correction modestly improved aggregate
distance error, but did not reduce the pose-dependent spread:

| Metric | Baseline A | Baseline B | Camera-corrected C |
| --- | ---: | ---: | ---: |
| Mean absolute error | `0.1275 m` | `0.1267 m` | `0.1058 m` |
| RMSE | `0.1552 m` | `0.1567 m` | `0.1394 m` |
| Median absolute error | `0.1000 m` | `0.1200 m` | `0.0750 m` |
| Average pose spread | `0.2275 m` | `0.1825 m` | `0.2425 m` |

The conclusion is that camera-model mismatch contributed to live error, but was
not the dominant remaining cause.

A trace-backed sweep using `tri_stream_yaw_v0_4` with camera intrinsics applied
showed usable but insufficient live distance accuracy:

| Metric | Value |
| --- | ---: |
| Mean absolute error | `0.1105 m` |
| RMSE | `0.1317 m` |
| Mean signed error | `+0.0198 m` |
| Median absolute error | `0.0825 m` |
| Maximum absolute error | `0.2680 m` |
| Samples within `10 cm` | `7 / 12` |
| Samples within `5 cm` | `1 / 12` |

A later trace-backed rerun used the current `260521-1029_ts-2d-cnn` /
`tri_stream_yaw_v0_5` artifact. It did not cleanly solve the live failure mode:

| Metric | Value |
| --- | ---: |
| Mean absolute error | `0.1074 m` |
| RMSE | `0.1341 m` |
| Mean signed error | `-0.1008 m` |
| Median absolute error | `0.0837 m` |
| Maximum absolute error | `0.2895 m` |
| Samples within `10 cm` | `6 / 12` |
| Samples within `5 cm` | `3 / 12` |

The incident outcome is an architectural finding. The direct distance/yaw
tri-stream family remains useful as a baseline and live-runtime integration
path, but it is no longer the preferred path for the next major improvement
cycle. The remaining problem needs a representation that exposes inferred
geometry, not only final scalar outputs.

Incident 005 later narrows one part of the remaining problem: live/synthetic
apparent-scale mismatch can produce a systematic signed distance bias even when
the locator and foreground path are plausible.

## 13. Incident 003: Foreground Mask Contamination Underestimate

The `failure-analysis/incidents/incident-003-foreground-mask-contamination-distance-underestimate`
directory records a live preprocessing failure where foreground extraction
expanded the apparent vehicle extent and drove a distance underestimate.

The primary trace was captured on 2026-05-26. The system predicted the Defender
at `1.325526 m`, while nearby live behaviour suggested that this was a clear
"too close" outlier. The ROI locator found a plausible compact target, but the
threshold foreground path merged the vehicle with dark sheet folds and shadow on
the support surface. The model then received a foreground mask, distance image,
orientation image, and geometry vector describing a much larger object than the
Defender itself.

The key trace signals were:

| Signal | Value |
| --- | ---: |
| Predicted distance | `1.325526 m` |
| Predicted yaw | `29.3355 deg` |
| Locator bbox | `[1029, 521, 1179, 638]` |
| Locator bbox size | `150 x 117 px` |
| Locator confidence | `0.838026` |
| Foreground bbox | `[944, 441, 1248, 700]` |
| Foreground bbox size | `304 x 259 px` |
| Foreground pixel count | `44,792 px` |
| Foreground bbox area / locator bbox area | `4.49 x` |
| Foreground pixels / locator bbox area | `2.55 x` |
| Geometry `area_norm` | `0.034174` |

This incident is the mirror image of Incident 001. Incident 001 collapsed the
foreground to a tiny fragment and drove a distance overestimate. Incident 003
expanded the foreground into support-surface texture and drove a distance
underestimate. In both cases, the model output was coherent with corrupted model
inputs.

The first attempted remediation was a hard foreground-vs-locator rejection gate.
That change was backed out because the locator bbox is primarily a ROI-centre
cue, not a reliable object-extent contract, and hard rejection was too brittle
for live use. The retained remediation is diagnostic and corrective rather than
strictly rejecting:

* foreground-vs-locator disagreement is recorded as trace metadata and warning
  fields.
* `threshold_foreground_v1` performs connected-component selection before
  rendering model inputs.
* ROI-saturating threshold candidates can trigger a stricter threshold retry.
* disconnected or ROI-scale support-surface components are less likely to become
  `x_distance_image`, `x_orientation_image`, and `x_geometry`.

The current implementation does not claim to solve every support-texture case.
If contamination remains physically connected to the vehicle in the threshold
mask, the system is expected to preserve suspicious foreground/locator metadata
for trace analysis. The recommended follow-up work is broader trace replay, a
locator-anchored fallback extractor, a better background-removal workflow, and
eventually a stronger foreground model.

## 14. Incident 004: ROI-FCN to Geometric Locator Retrospective

The `failure-analysis/incidents/incident-004-roi-fcn-to-geometric-locator-retrospective`
directory records the architectural justification for moving the live default ROI
locator from ROI-FCN in v0.2 to the deterministic `background_edge_v1` locator in
v0.3.

The retrospective conclusion is not that ROI-FCN was useless or that a geometric
locator is a general object detector. ROI-FCN remains a valid learned crop-centre
localiser and a retained legacy comparison path. The problem was that ROI-FCN was
a poor operational boundary for the live system. It solved:

```text
choose one centre point from a learned heatmap
```

but the live runtime needed:

```text
choose, explain, reject, and repair a crop and apparent-scale representation
```

The v0.2 traces support that distinction. The checked-in trace population
contains:

| Trace population signal | Count / value |
| --- | ---: |
| Trace directories with preprocessing metadata | `33` |
| Inference traces | `21` |
| Failure traces | `8` |
| Locator-only traces | `4` |
| Clipped ROI failures | `6 / 8` |
| Low-confidence failures | `2 / 8` |
| Accepted inference confidence median | `0.4886` |
| Failure confidence median | `0.8479` |

The confidence signal was therefore not a reliable live health measure. Several
high-confidence ROI-FCN failures were rejected only after the centre-derived ROI
was found to be clipped. Accepted ROI-FCN traces also showed that accepting a
crop was not enough: one v0.2 trace accepted an ROI with confidence `0.359`, the
downstream foreground collapsed to `119` pixels, and the model predicted
`5.1837 m`. That pattern matches the later formal Incident 001 mechanism: the
locator can be acceptable while the model input is corrupted by foreground
collapse.

The v0.3 geometric locator exposes the evidence needed for live diagnosis:
foreground mask, edge map, contour candidates, chosen bbox, ROI request/source
bounds, clipping and content metadata, explicit rejection reasons, and debug
artifacts. That makes ROI selection inspectable, rejectable, tunable, and
comparable to downstream foreground geometry.

Incident 004 does not claim a quantified locator accuracy improvement. The
repository does not contain a controlled replay where every v0.2 trace is run
through both ROI-FCN and `background_edge_v1` against hand-labelled ROI centres.
The claim is narrower and better supported: for this bounded fixed-camera live
system, the geometric locator is a better engineering interface because it turns
ROI selection into an auditable apparent-scale measurement path.

## 15. Incident 005: Live/Synthetic Apparent-Scale Mismatch

The `failure-analysis/incidents/incident-005-live-synthetic-apparent-scale-mismatch`
directory records the current leading explanation for a post-ROI-fix live
distance underprediction and follow-up sweeps after apparent-scale mitigation.

After the live ROI path had moved to the geometric locator, accepted live
predictions were still consistently too close. The post-ROI-fix live sweep
recorded six accepted distance readings. Five clean or clean-ish readings
underpredicted measured distance by approximately `0.35 m` to `0.40 m`. One
additional `2.9 m -> 2.008 m` reading was treated as contaminated and excluded
from the clean bias estimate.

The clean trace summary was:

| Metric | Value |
| --- | ---: |
| Included readings | `5` |
| Mean signed error | `-0.364 m` |
| Median signed error | `-0.363 m` |
| Signed error range | `-0.345 m` to `-0.399 m` |
| Mean absolute error | `0.364 m` |

An independent synthetic/live image-pair analysis then compared the apparent
size of the Defender in nominally matched synthetic and live captures. Across
eight front/side image pairs, the live vehicle appeared consistently larger than
the synthetic vehicle at the same nominal lens distance. Using inverse-scale
geometry, that visual-scale mismatch predicts a mean apparent-distance offset of
`-0.336 m`, with a median of `-0.331 m` and a range from `-0.283 m` to
`-0.406 m`.

| Evidence source | Key result | Interpretation |
| --- | ---: | --- |
| Original clean live sweep | mean signed error `-0.364 m` | Model predicts target too close |
| Synthetic/live scale comparison | mean apparent-distance offset `-0.336 m` | Live target appears larger than synthetic equivalent |
| Difference between original means | `0.028 m` | Independent evidence paths converge |
| First follow-up live sweep | mean signed error `-0.113 m`; MAE `0.118 m` | Mitigation materially improved the live bias, but residual underprediction remained |
| Latest three-distance sweep | mean signed error `-0.033 m`; MAE `0.080 m` | Further improvement, with remaining close-range underprediction at `1.60 m` |

The incident therefore strongly supports the hypothesis that the live model
input presents the Defender as visually larger, and therefore apparently closer,
than the synthetic training representation. It does not prove one exact
low-level cause. The mismatch could still be split between Unity camera
parameters, real-to-Unity intrinsics mapping, viewport/capture handling, lens
model mismatch, synthetic object scale, or physical measurement reference
differences.

The engineering conclusion is narrower and stronger: once locator and foreground
failures are controlled, live/synthetic apparent-scale alignment becomes a
primary remaining distance-risk boundary.

The immediate implementation response is a configurable post-foreground model
representation transform:

```text
accepted camera frame
  -> optional camera intrinsics transform
  -> locator and ROI crop
  -> foreground extraction and component cleanup
  -> model representation transform
  -> recompute model-space x_geometry
  -> pack x_distance_image, x_orientation_image, x_geometry
  -> direct distance/yaw model
```

The transform is traceable and configurable. The first follow-up sweep after
apparent-scale mitigation recorded these live readings:

| Measured reference | Orientation | Predicted distance | Signed error | Note |
| ---: | --- | ---: | ---: | --- |
| `1.59 m` | `0 deg / front` | `1.41 m` | `-0.18 m` | slight ROI clipping contamination |
| `1.59 m` | `90 deg / side` | `1.35 m` | `-0.24 m` | underpredicting |
| `1.77 m` | `0 deg / front` | `1.65 m` | `-0.12 m` | improved |
| `1.77 m` | `90 deg / side` | `1.56 m` | `-0.21 m` | underpredicting |
| `1.97 m` | `0 deg / front` | `1.99 m` | `+0.02 m` | good |
| `1.97 m` | `90 deg / side` | `1.93 m` | `-0.04 m` | good |
| `2.18 m` | `0 deg / front` | `2.13 m` | `-0.05 m` | good |
| `2.18 m` | `90 deg / side` | `2.10 m` | `-0.08 m` | good |

The follow-up summary is:

| Population | Mean signed error | Mean absolute error |
| --- | ---: | ---: |
| All rows | `-0.113 m` | `0.118 m` |
| Excluding slightly clipped `1.59 m` front row | `-0.103 m` | `0.109 m` |

That first follow-up was a material improvement over the original clean-sweep
mean signed error and MAE of `-0.364 m` / `0.364 m`, but it was not closure. Four
of the eight follow-up rows remained outside the `0.10 m` distance threshold,
and the near-range side rows still underpredicted.

A later three-distance sweep recorded these live readings:

| Measured reference | Orientation | Predicted distance | Signed error | Note |
| ---: | --- | ---: | ---: | --- |
| `1.60 m` | `0 deg / front` | `1.48 m` | `-0.12 m` | close-range underprediction |
| `1.60 m` | `90 deg / side` | `1.43 m` | `-0.17 m` | reading when not locked onto foot |
| `2.20 m` | `0 deg / front` | `2.29 m` | `+0.09 m` | slight overprediction |
| `2.20 m` | `90 deg / side` | `2.24 m` | `+0.04 m` | good |
| `2.90 m` | `0 deg / front` | `2.91 m` | `+0.01 m` | very good |
| `2.90 m` | `90 deg / side` | `2.85 m` | `-0.05 m` | good |

The latest sweep summary is:

| Population | Mean signed error | Mean absolute error |
| --- | ---: | ---: |
| All rows | `-0.033 m` | `0.080 m` |

Distance-band signed-error summary:

| Distance band | Mean signed error | Interpretation |
| --- | ---: | --- |
| `1.60 m` | `-0.145 m` | close-range underprediction remains |
| `2.20 m` | `+0.065 m` | slight overprediction |
| `2.90 m` | `-0.020 m` | good far-range alignment |

The latest sweep further reduces the aggregate bias and improves MAE relative to
the first follow-up. It is still not closure: the close-range `1.60 m` rows
remain outside the `0.10 m` distance threshold, and one side reading is explicitly
noted as not locked onto the foot. The direct distance/yaw model should therefore
still be framed as:

```text
traceable live-runtime integration and failure-analysis evidence
not a calibrated live distance-estimation claim
```

The checked-in Incident 005 evidence includes the polished report and eight
compact scale-pair summary comparison images. The evidence manifest also
describes heavier raw image pairs and live-inference traces that remain in the
local incident workspace and can be copied later if the repository should
preserve the full artifact set.

## 16. Experimental Amodal Keypoint Topology

The repository contains both the keypoint topology design documents and a first
experimental implementation:

* `documents/keypoint-regression-topology-v0.4.md`
* `documents/keypoint-regression-topology-v0.4-technical-summary.md`
* `03_rb-training-v2.0/src/topologies/topology_defender_amodal_keypoint_pose.py`
* `03_rb-training-v2.0/src/topologies/topology_defender_amodal_keypoint_pose_v0_1.py`
* `03_rb-training-v2.0/schemas/defender_keypoint_schema.json`

The implemented topology is registered as `defender_amodal_keypoint_pose`, with
default variant `defender_amodal_keypoint_pose_v0_1`. Its metadata marks it as
experimental. It is not the currently selected live distance/yaw model, and this
writeup does not claim live keypoint accuracy.

The model emits a structured object-state hypothesis:

```text
tri-stream image-derived inputs
  -> vehicle centre in camera-space coordinates
  -> all fixed semantic external vehicle keypoints, including occluded keypoints
  -> keypoint visibility / in-frame state
  -> direct distance and yaw heads for compatibility
```

The current implemented outputs are:

* `distance_m`
* `yaw_sin_cos`
* `defender_center_3d`
* `defender_keypoints_3d_flat`
* `defender_keypoints_visible_logits`

The key design choice remains to predict all ten fixed external keypoints, not
only the visible ones. Hidden keypoints are treated as amodal inferred targets
derived from known object geometry, camera setup, visible evidence, ROI geometry,
and the synthetic training distribution. Visibility is a separate target and
diagnostic signal; it does not mask the amodal 3D supervision.

The first implementation milestone has concrete code support for:

* a registered topology family and selectable variant.
* a versioned `defender_keypoint_schema.json` with schema-hash validation.
* synthetic manifest labels for centre, ten 3D keypoints, visibility, and schema
  metadata.
* model outputs for distance, yaw, centre, flattened 3D keypoints, and
  visibility logits.
* distance, yaw, centre, keypoint, and visibility losses.
* centre/keypoint metrics plus visible-vs-hidden keypoint metrics.
* clear failures for missing labels or schema metadata.
* a `geometry_only` ablation mode.

This topology follows directly from Incidents 002, 003, 004, and 005. Direct
distance/yaw regression can report that a prediction is wrong, but it cannot
expose enough intermediate geometric state to determine whether the model
misunderstood scale, pose, extent, visibility, lighting, foreground shape, ROI
selection, synthetic/live projection, or a combination of those factors. A
keypoint-based representation gives the system an inspectable object hypothesis
that can be compared against known rigid geometry.

The remaining caution is important: until trained keypoint artifacts and
geometry-only ablations are evaluated, the keypoint topology should be described
as an implemented experimental direction rather than an externally validated
accuracy improvement.

## 17. Representative Results

The table below separates offline preprocessed evaluation, raw-image composed
inference, live-local artifact selection, live incident evidence, and current
mitigation work. These are different evidence types and should not be collapsed
into one headline number.

| Artifact / Run | Evidence Type | Train / Validation Samples | Distance MAE | Distance RMSE | Distance within `0.10 m` | Yaw Mean Error | Yaw within `5 deg` | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `260415-1146_ds-2d-cnn/run_0001` | offline preprocessed dual-stream distance+yaw validation | `250,000 / 50,000` | `0.01007 m` | `0.01297 m` | `0.99996` | `1.49987 deg` | `0.97482` | Strong recorded offline preprocessed dual-stream result. |
| `260515-1301_ts-2d-cnn` | prior selected tri-stream v0.4 offline validation artifact | `300,000 / 60,000` | `0.01485 m` | `0.02412 m` | `0.99907` | `1.03246 deg` | `0.99793` | Strong offline synthetic artifact, later superseded as the selected live direct-regression model. |
| `260521-1029_ts-2d-cnn` | current selected tri-stream v0.5 offline validation artifact | `300,000 / 60,000` | `0.01586 m` | `0.02603 m` | `0.99848` | `1.50303 deg` | `0.98543` | Current direct-regression baseline; strong offline metrics but unresolved live transfer risk. |
| `260504-1100_ts-2d-cnn__run_0001` | earlier live-local tri-stream selected artifact | `226,971 / 47,929` | `0.09756 m` | `0.12627 m` | `0.62826` | `3.87700 deg` | `0.74675` | Metadata-compatible earlier live candidate, weaker distance than later selected artifacts. |
| `260415-1146_ds-2d-cnn` raw-image output | composed ROI-FCN plus dual-stream inference on `49,999` raw validation rows | `n/a / 49,999` | `0.11117 m` | `0.43346 m` | `0.92948` | `12.33194 deg` | `0.58491` | Shows runtime degradation and a crop-boundary distance tail. |
| `260425-1025_ds-2d-cnn` raw-image output | composed ROI-FCN plus brightness-normalised dual-stream inference on `49,999` raw validation rows | `n/a / 49,999` | `0.04784 m` | `0.10853 m` | `0.95312` | `16.48438 deg` | `0.38421` | Better bulk distance, but broad yaw underperformance. |
| `260420-1219_roi-fcn-tiny__run_0003` | ROI-FCN locator validation | `100,000 / 20,000` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | Mean centre error `3.1757 px`, p95 `7.7098 px`, full-containment success `0.9891`. |
| Incident 004 ROI-FCN retrospective | live locator architecture evidence | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | v0.2 traces show high-confidence clipped ROI failures and accepted crops with downstream foreground collapse; v0.3 pivoted to inspectable geometric ROI selection. |
| `tri_stream_yaw_v0_4` live sweep | trace-backed live incident evidence | `n/a` | `0.1105 m` | `0.1317 m` | `7 / 12` | `n/a` | `n/a` | Pose-dependent live distance bias persisted with intrinsics applied. |
| `tri_stream_yaw_v0_5` live sweep | trace-backed live incident evidence | `n/a` | `0.1074 m` | `0.1341 m` | `6 / 12` | `n/a` | `n/a` | Current direct-regression model remained pose-sensitive and shifted signed error negative. |
| Incident 003 primary trace | live preprocessing failure evidence | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | Foreground-mask contamination expanded apparent scale and drove a distance underestimate; hard rejection was backed out in favour of diagnostic/component-selection remediation. |
| Incident 005 clean live sweep | live incident evidence | `n/a` | `0.364 m` | `n/a` | `0 / 5` | `n/a` | `n/a` | Five clean-ish accepted readings underpredicted by mean signed error `-0.364 m`; apparent-scale mismatch is the leading explanation. |
| Incident 005 synthetic/live scale pairs | paired image-scale evidence | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | Eight front/side pairs predict mean apparent-distance offset `-0.336 m`, close to the clean live bias. |
| Incident 005 first follow-up sweep | live mitigation evidence | `n/a` | `0.118 m` | `n/a` | `4 / 8` | `n/a` | `n/a` | Mean signed error improved to `-0.113 m`; excluding one slightly clipped row, MAE was `0.109 m`, but near-range side rows still underpredict. |
| Incident 005 latest three-distance sweep | live mitigation evidence | `n/a` | `0.080 m` | `n/a` | `4 / 6` | `n/a` | `n/a` | Mean signed error improved to `-0.033 m`; `1.60 m` rows still average `-0.145 m`, while `2.20 m` and `2.90 m` are close. |
| Incident 005 model representation transform | runtime mitigation and traceability hook | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | Implemented post-foreground affine scale correction with raw/model foreground metadata; follow-up sweeps improved error but repeat trace-backed validation remains required. |

Five conclusions follow from these results. First, the repository contains
strong offline evidence for the bounded preprocessed task. Second, end-to-end
raw-image and live inference are harder than the offline task. Third, foreground
quality is a first-class operational risk for the direct tri-stream family,
because corrupted apparent scale can drive confident but wrong distance outputs.
Fourth, the live ROI boundary needs inspectable geometric evidence, not only a
learned centre-point confidence. Fifth, once locator and foreground failures are
controlled, synthetic/live apparent-scale alignment itself becomes a first-class
validation gate.

## 18. Failure Analysis and Engineering Learnings

The failure-analysis framework uses a primary threshold of:

* distance failure: absolute error greater than `0.10 m`.
* yaw failure: absolute error greater than `5 deg`.
* clean success: distance error at most `0.05 m` and yaw error at most
  `2.5 deg`.

For the `260415-1146_ds-2d-cnn` raw-image run on `49,999` validation rows:

* joint success was `56.73%`.
* clean success was `17.44%`.
* distance within `10 cm` was `92.95%`.
* yaw within `5 deg` was `58.49%`.
* the most severe distance failures were strongly associated with fixed ROI
  requests extending beyond the source image boundary.
* yaw failures were dominated by a heavy tail, including near-180-degree
  orientation confusions.

For the `260425-1025_ds-2d-cnn` raw-image run:

* joint success was `38.31%`.
* clean success was `17.36%`.
* distance within `10 cm` was `95.31%`.
* yaw within `5 deg` was `38.42%`.
* distance improved in the bulk, but yaw degraded across the main distribution.

Incident 001 extends the same engineering pattern into live preprocessing. It
demonstrates that operational failures can arise after the locator and before
the model, and that model-input artifact capture is essential for debugging
composed ML systems.

Incident 002 shows that a failure can remain after foreground collapse is fixed
and camera intrinsics are applied. The direct model family can still encode
pose-dependent distance bias, which motivates a more inspectable intermediate
representation.

Incident 003 shows the opposite foreground failure from Incident 001: instead of
collapsing to a tiny fragment, foreground extraction expanded into support
surface texture and shadow. The initial hard rejection gate was backed out; the
retained strategy is connected-component foreground selection plus diagnostic
foreground/locator metadata.

Incident 004 reframes the ROI-FCN-to-geometric-locator pivot as an operational
boundary decision. ROI-FCN predicts a centre point, but live inference needs an
auditable apparent-scale path with candidates, bbox geometry, rejection reasons,
and artifacts that can be compared to downstream foreground geometry.

Incident 005 reframes the remaining live underprediction as a synthetic/live
projection and apparent-scale boundary. The model can receive a coherent,
vehicle-shaped input and still produce a systematically wrong distance if the
live model representation is scaled differently from the synthetic training
representation. The follow-up sweeps show that apparent-scale mitigation can
materially reduce the bias, with the latest three-distance sweep reaching mean
signed error `-0.033 m` and MAE `0.080 m`. They also show why the calibrated live
claim must remain bounded until repeated trace-backed sweeps close the residual
close-range error.

The main learning is that model metrics alone are insufficient. In a multi-stage
perception system, accuracy depends on the contracts and failure modes of every
stage: camera capture, calibration, background handling, ROI selection,
foreground extraction, representation alignment, geometry construction, model
input rendering, and output decoding.

## 19. Testing and Engineering Discipline

The repository includes focused tests across multiple layers:

* v4 preprocessing integration, silhouette algorithms, foreground handling, and
  brightness normalisation.
* topology contracts and task-runtime reporting, including keypoint/visibility
  heads for the experimental amodal topology.
* tri-stream yaw variants, including v0.5's pose-conditioned bounded residual
  structure.
* resume features and epoch summaries.
* ROI-FCN preprocessing, geometry, targets, data contracts, and training smoke
  tests.
* raw-image inference sample execution and brightness analysis.
* live inference model manifests, compatibility checks, frame handoff, frame
  selection, device policy, runtime parameters, ROI locators, tri-stream
  preprocessing, inference core, PyTorch engine, workers, GUI bridge, GUI main
  window, synthetic camera, and OpenCV/V4L2 camera source.
* live camera intrinsics modes, preview/background transform handling, and
  metadata propagation.
* v0.3-specific tests for `background_edge_v1`, generic tri-stream
  preprocessing, component-aware foreground selection, foreground policy
  selection, manual mask application, trace artifact contents, GUI app wiring,
  incident-001 preprocessing regression, incident-003 diagnostic behaviour, and
  generic locator compatibility.
* model representation transform tests covering enabled-config validation,
  independent `scale_x`/`scale_y` behaviour, spatial image/mask alignment,
  preprocessor integration, and recomputing `x_geometry` from the transformed
  foreground mask.
* ChArUco calibration contracts, config loading, dictionary probing, capture
  quality, capture decisions, session storage, pose diversity, reprojection, and
  artifact export.

The test coverage is strongest around contract boundaries, data-shape
assumptions, and runtime glue. That is the appropriate emphasis for a
multi-stage perception codebase where silent interface drift would be expensive.

The incident-specific tests are particularly valuable:

* `test_v4_silhouette_algorithms.py` verifies that the saved incident ROI crop
  now selects the large border-touching vehicle component rather than the tiny
  fragment.
* `test_generic_preprocessor.py` verifies that the live preprocessor produces a
  large foreground mask, large geometry, and non-empty vehicle representation
  from the Incident 001 frame and locator bbox.
* `test_generic_preprocessor.py` also verifies that incident-shaped foreground
  over-expansion is diagnostic-only and that disconnected threshold contaminants
  are excluded by component selection.
* `test_model_representation_transform.py` verifies that the Incident 005 style
  post-foreground transform changes mask width and height independently and
  updates model geometry from the transformed mask.
* The Incident 004 retrospective identifies v0.2 trace replay as useful
  follow-up work, but does not require it for the architectural justification.
* Incident 005 records follow-up live sweeps and identifies staged scale-pair
  fixtures plus repeat trace-backed sweeps as required follow-up work before
  stronger live distance claims.

## 20. Technically Distinctive Features

The project demonstrates:

* end-to-end ownership across simulation, preprocessing, training, evaluation,
  inference, live GUI runtime, calibration, and incident analysis.
* camera-footprint-aware synthetic placement rather than naive world-space
  sampling.
* explicit data and model contracts across preprocessing, topology, runtime,
  and artifact selection.
* preservation of geometric depth cues through fixed unscaled distance crops.
* circular yaw regression through `sin/cos` targets.
* learned ROI-FCN crop-centre localisation.
* deterministic background/edge live localisation for inspectable demo
  operation.
* geometric ROI selection that exposes foreground masks, edge maps, contour
  candidates, chosen bboxes, ROI bounds, and rejection reasons.
* manual masks and static background handling in the live runtime.
* component-aware threshold foreground extraction with foreground/locator
  diagnostic metadata.
* calibration-backed live camera intrinsics transforms.
* post-foreground model representation transforms for apparent-scale alignment.
* raw-vs-model foreground geometry metadata for trace analysis.
* a tri-stream model contract separating distance image, orientation image, and
  geometry features.
* runtime compatibility checking before model pairing.
* frame-handoff and worker contracts for a live-local application.
* trace bundles that capture accepted frames, locator outputs, preprocessing
  artifacts, model inputs, metadata, and inference outputs.
* failure analysis that distinguishes offline validation quality from composed
  runtime quality.
* an experimental amodal keypoint topology motivated by measured limitations of
  direct scalar regression, foreground-dependent apparent-scale failures, and
  synthetic/live projection mismatch.

These are not presented as novel research contributions. They are practical
engineering capabilities in applied ML and perception-system development.

## 21. Established Engineering Patterns Demonstrated

The repository implements established engineering patterns relevant to applied
ML systems:

* PyTorch training loops with checkpoints, schedulers, resumes, metrics, plots,
  and model cards.
* NPZ shard packing and schema validation.
* OpenCV contour processing, foreground extraction, silhouette generation,
  ChArUco detection, affine warping, and camera calibration.
* heatmap-based localisation and argmax decode.
* deterministic geometric locator contracts with candidate scoring and explicit
  rejection metadata.
* notebook control surfaces backed by importable Python modules.
* JSON, YAML, and TOML artifacts for traceability and audit.
* GUI-worker separation through payload contracts.
* device policy management for CPU and CUDA execution paths.
* artifact-backed incident analysis and regression tests.
* multitask topology contracts spanning scalar regression, circular yaw, 3D
  centre regression, keypoint regression, and visibility classification.
* configuration-backed runtime transforms with debug artifacts and metadata.

## 22. Scope and Current Limits

This repository is not presented as a packaged product or broad general-purpose
vision model. It is a bounded research-engineering workspace for testing whether
a fixed-camera system can estimate useful vehicle state under controlled
conditions, and for making the offline-to-runtime failure modes inspectable.

The current evidence should be read with the following constraints in mind:

* The task is limited to one known vehicle family, fixed camera geometry, a
  constrained movement plane, controlled full-frame captures, and synthetic
  training and validation data.
* Synthetic training and validation remain the strongest evidence base.
* Offline synthetic validation metrics are not real-camera accuracy;
  preprocessed validation, raw-image composed inference, and live-local
  inference are separate evidence types.
* The live-local runtime works, but real-camera accuracy is still under
  investigation.
* The current direct scalar distance/yaw model shows pose-linked and
  projection-linked distance risk in live testing.
* Camera calibration improves part of the runtime alignment problem but does
  not, by itself, solve pose-dependent or apparent-scale error.
* The live sweeps use practical measured floor marks, not calibrated metrology
  ground truth.
* The distance reference convention still needs to be made explicit across
  synthetic labels, physical measurement, and scale-pair analysis.
* `background_edge_v1` is deterministic and inspectable, and is designed for
  the controlled fixed-camera live-local path. It is not a general detector.
* ROI-FCN targets are bootstrapped from an existing crop heuristic, so the
  localiser initially learns that crop-centre definition rather than
  independently curated ground truth.
* Incident 004 is a retrospective engineering justification, not a benchmark
  that quantifies geometric-locator centre accuracy against ROI-FCN on
  hand-labelled live frames.
* Incident 005 strongly supports the apparent-scale mismatch hypothesis, but
  does not isolate one exact low-level geometry cause.
* The Incident 005 model representation transform is implemented, configurable,
  and test-covered. Follow-up sweeps improved error materially, with the latest
  three-distance sweep reaching MAE `0.080 m`, but repeat trace-backed sweeps and
  scripted scale fixtures remain required before claiming calibrated live
  distance accuracy.
* The current next architectural direction is a more inspectable
  keypoint/topology-based representation.
* The keypoint topology has a first experimental registered implementation,
  schema, labels, losses, metrics, and tests, but it is not yet a selected live
  model artifact or externally validated accuracy improvement.
* Foreground extraction remains a live-runtime risk. The current strategy
  favours component selection plus diagnostic metadata over brittle hard
  rejection.
* The codebase is a research workspace with versioned subprojects,
  compatibility shims, and evolving runtime paths.

These caveats are part of the technical value of the project. They keep the
claims bounded and make the results easier to evaluate honestly.

## 23. Current Version Focus

This standalone v0.10 writeup captures the current repository-level narrative in these ways:

* Adds Incident 005 as a first-class incident and evidence source.
* Separates live/synthetic apparent-scale mismatch from prior locator,
  foreground, and pose-bias failures.
* Adds the model representation transform as a current live-runtime capability.
* Distinguishes camera-intrinsics transforms from post-foreground model-space
  transforms.
* Adds raw-vs-model foreground metadata and debug artifacts to the live runtime
  description.
* Adds Incident 005 to representative results and current limitations, including
  follow-up live sweeps after apparent-scale mitigation.
* Tightens the claim boundary for direct distance/yaw regression: useful
  integration baseline and evidence path, not yet a calibrated live distance
  claim.

## 24. Short Technical Summary

This repository is a bounded computer-vision project for fixed-camera vehicle
distance and yaw estimation. It combines Unity synthetic data generation,
OpenCV preprocessing, PyTorch model training, learned and deterministic ROI
localisation, raw-image inference, camera calibration, post-foreground model
representation alignment, and a live PySide6 runtime.

The project is intentionally narrow: one known vehicle family, one fixed camera
geometry, synthetic labelled data, and a constrained operating plane. Within
that scope, it demonstrates the engineering work required to move from offline
model training toward composed runtime inference, including data contracts,
artifact compatibility checks, runtime preprocessing, camera-intrinsics
handling, apparent-scale alignment, trace capture, and failure analysis.

The most valuable result is not a single accuracy number. The repository shows
strong offline synthetic performance, measurable degradation in composed
raw-image inference, live preprocessing incidents that were traced and
remediated or partially remediated, a ROI-FCN-to-geometric-locator pivot toward
an inspectable apparent-scale measurement path, a pose-dependent distance-bias
incident that motivates a more inspectable model family, a live/synthetic
apparent-scale incident that motivates explicit representation alignment,
follow-up sweeps showing material but incomplete mitigation, and a first
experimental amodal keypoint topology implementation.

That makes the repository useful evidence of applied ML engineering, computer
vision, evaluation discipline, runtime integration capability, and careful
failure analysis under bounded claims.
