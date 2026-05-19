# Raccoon Ball: Bounded Monocular Perception System

## 1. Project Overview

Raccoon Ball is a bounded computer-vision and applied-machine-learning project for estimating vehicle distance and yaw from a fixed monocular camera view under controlled conditions.

The repository demonstrates end-to-end ownership of a perception pipeline spanning synthetic data generation, preprocessing contracts, PyTorch model training, learned and deterministic ROI localisation, raw-image inference, live local runtime integration, trace capture, and failure analysis.

The system is deliberately scoped. It is not intended to solve general autonomous driving, open-world object detection, multi-object tracking, or unconstrained real-world scene understanding. Its purpose is to investigate a narrower engineering question:

> Can a fixed-camera system observing a known vehicle in a constrained scene estimate useful vehicle state from image-based geometric cues, and how does performance change as the system moves from offline validation into composed runtime inference?

That framing is important. The repository is most useful as evidence of applied ML engineering, computer vision, data-pipeline design, experimental discipline, runtime integration, and operational debugging.

## 2. v0.7 Update Scope

This v0.7 writeup updates the previous v0.6.2 technical summary to reflect the substantial live-inference and failure-analysis work now present in the repository.

The main changes since v0.6.2 are:

* `06_live-inference_v0.2` and `06_live-inference_v0.3` extend the live-local runtime beyond the earlier v0.1 prototype.
* The live runtime now includes richer trace capture, diagnostic views, static background capture/removal, manual mask drawing, runtime locator controls, and a more explicit foreground-extraction policy.
* The current v0.3 demo-stabilisation path defaults to an inspectable deterministic locator, `background_edge_v1`, rather than the ROI-FCN locator. ROI-FCN remains available as a legacy comparison path.
* The selected v0.3 distance/orientation artifact is `260515-1301_ts-2d-cnn`, a `tri_stream_yaw_v0_4` model using the `rb-preprocess-v4-tri-stream-grayscale-white-v1` contract and `320 x 320` model canvases.
* The repository now contains `failure-analysis/incident_1`, a live-runtime failure investigation that traces a large distance-prediction spike to deterministic preprocessing, applies remediation, and adds fixture-backed regression coverage.

The most important change is not only that the live application has more controls. The project now contains a complete example of diagnosing a real live-inference failure from captured artifacts, changing the preprocessing design, and validating the fix against saved traces.

## 3. Problem Scope

The central task remains intentionally narrow:

* **Input:** a monocular full-frame image from a fixed camera
* **Primary output:** vehicle distance in metres
* **Secondary output:** vehicle yaw/orientation
* **Runtime support task:** locate the crop region required to construct the model input representation

The current system assumes:

* one known vehicle family
* one fixed camera geometry
* one constrained movement plane
* synthetic training and validation data
* controlled full-frame captures rather than arbitrary real-world scenes

Those constraints are a design choice. They keep the task falsifiable and allow the project to focus on the engineering boundary between preprocessed benchmark performance and composed runtime behaviour.

## 4. Repository Architecture

The repository is organised as a versioned multi-project workspace:

* `01_rb_synthetic-data_3`: Unity/C# synthetic image generation
* `02_synthetic-data-processing-v4.0`: OpenCV and NumPy preprocessing, detection, silhouette generation, foreground enhancement, and dual-stream / tri-stream packing
* `03_rb-training-v2.0`: PyTorch training, topology registry, model evaluation, resume support, and reporting
* `04_ROI-FCN`: preprocessing and training for crop-centre heatmap localisation
* `05_inference-v0.3-ds`: raw-image inference using ROI-FCN plus dual-stream distance/yaw models
* `05_inference-v0.4-ts`: tri-stream-facing inference work and brightness-analysis tooling
* `06_live-inference_v0.1`: first live-local runtime with camera input, frame handoff, model registry, preprocessing, workers, and GUI
* `06_live-inference_v0.2`: richer live diagnostics, trace capture, background handling, and ROI-FCN visualisation work
* `06_live-inference_v0.3`: current demo-stabilisation runtime using generic locator interfaces, deterministic background/edge localisation, manual masks, selectable foreground extraction, and improved trace evidence
* `failure-analysis/incident_1`: live-runtime incident investigation, remediation record, and post-remediation evidence

This layout reflects a research-engineering codebase moving from offline experiments toward runtime composition. It is not packaged as a finished product, but it contains real integration surfaces, tests, artifacts, runtime contracts, and incident-analysis material.

## 5. Synthetic Data Generation

The Unity generator creates full-frame synthetic images with structured run metadata and sample manifests. It is designed to produce controlled labelled data for the fixed-camera perception task rather than arbitrary visual variety.

Key generator components include:

* `CaptureService.cs` for render-texture capture
* `RunControllerBehaviour.cs` for batch orchestration, cancellation, manifest flushing, and attempt-budget handling
* `ManifestWriter.cs`, `ManifestRowMapper.cs`, and `RunMetadataWriter.cs` for traceable run outputs
* `FileNamingStrategy.cs` for deterministic sample naming
* `DistanceCalculator.cs` for explicit target derivation
* `StratifiedPlacementPlanner.cs` for camera-footprint-aware placement
* `VehicleProjectionValidator.cs` for image-space feasibility checks

The placement strategy is a central design choice. Rather than sampling arbitrary world positions, the generator projects the camera footprint onto the movement plane, divides the usable footprint into depth and lateral cells, validates projected vehicle bounds in image space, and redistributes quota when cells exhaust their attempt budget.

The generator writes:

* `images/*.png`
* `manifests/run.json`
* `manifests/samples.csv`
* `runlog.txt`

This gives downstream stages explicit lineage rather than relying on filenames alone.

## 6. Preprocessing and Representation Design

The preprocessing layer is contract-driven and stage-based. The v4 pipeline supports:

* `detect`: edge-based or detector-style vehicle localisation metadata
* `silhouette`: ROI silhouette generation with contour processing and convex-hull fallback
* `pack_dual_stream`: distance/yaw regression inputs with geometry features
* `pack_tri_stream`: separate distance image, orientation image, and geometry streams
* foreground enhancement and brightness-normalisation options
* corpus shuffle and notebook control surfaces for repeated training workflows

The strongest representation choices are:

* fixed ROI canvases for distance inference
* no rescaling in the distance stream, preserving apparent object size as a depth cue
* a 10-element geometry vector: `cx_px`, `cy_px`, `w_px`, `h_px`, `cx_norm`, `cy_norm`, `w_norm`, `h_norm`, `aspect_ratio`, `area_norm`
* circular yaw targets represented as `sin/cos`
* optional foreground-only brightness normalisation
* foreground enhancement for grayscale-on-white model representations
* a tri-stream contract that separates distance evidence from orientation evidence

The earlier tri-stream contract, `rb-preprocess-v4-tri-stream-orientation-v1`, writes:

* `x_distance_image`
* `x_orientation_image`
* `x_geometry`
* `y_distance_m`
* `y_yaw_deg`
* `y_yaw_sin`
* `y_yaw_cos`

The current v0.3 live-selected distance/orientation model uses the related `rb-preprocess-v4-tri-stream-grayscale-white-v1` contract. It keeps the same tri-stream input keys, but uses `320 x 320` canvases and a grayscale vehicle-on-white representation:

* distance image: fixed unscaled ROI canvas
* orientation image: target-centred image scaled by foreground extent
* geometry: 10-field foreground bounding-box vector
* foreground enhancement: masked median-darkness gain when enabled
* orientation context scale: `1.25`

The split remains technically meaningful. Distance benefits from preserving apparent scale, while yaw benefits from a target-centred orientation view.

## 7. Model Training and Evaluation

The training code is organised around reusable Python modules rather than notebook-only logic. It includes:

* topology contracts and a topology registry
* dataset summaries and preprocessing-contract validation
* shard-based NPZ loading
* RAM-aware shard caching
* split-overlap checks
* checkpointing and resume-state support
* model cards, run manifests, plots, metrics, and sample predictions
* task-aware reporting for scalar distance and multitask distance-plus-yaw outputs

Model families represented in the repository include:

* baseline full-frame CNN distance regression
* dual-stream crop-plus-geometry distance regression
* dual-stream distance-plus-yaw regression
* tri-stream distance-plus-yaw regression
* ROI-FCN crop-centre localisation

Yaw is modelled through circular regression using `sin/cos` targets rather than direct angle regression. The training runtime resolves prediction heads and target heads from topology contracts, allowing the loss and reporting paths to work across scalar and multitask models.

## 8. ROI-FCN and Runtime Localisation

The ROI-FCN subsystem turns crop placement into a learned task. It is trained to predict the centre of the fixed ROI required by the downstream distance/orientation model.

The ROI-FCN preprocessing path preserves:

* full-frame grayscale locator input
* target centre in original-image coordinates
* target centre in locator-canvas coordinates
* resize scale
* padding offsets
* optional bootstrap box metadata

The validated ROI-FCN artifact `260420-1219_roi-fcn-tiny__run_0003` uses:

* topology id: `roi_fcn_tiny`
* topology variant: `tiny_v1`
* locator canvas: `480 x 300`
* downstream ROI crop: `300 x 300`
* supervision: Gaussian heatmap
* decode: deterministic argmax back into source-image coordinates
* training split: `100,000` samples
* validation split: `20,000` samples

Its validation metrics are:

| Metric | Value |
| --- | ---: |
| Mean centre error | `3.1757 px` |
| Median centre error | `2.4354 px` |
| p95 centre error | `7.7098 px` |
| ROI full-containment success | `0.9891` |

In v0.7, it is important to separate the learned ROI-FCN path from the current live demo path. The v0.3 runtime defaults to `background_edge_v1`, an inspectable deterministic locator built for real-camera demo stabilisation. ROI-FCN is retained as an explicit legacy comparison path, and the current model selection file also references `260516-1714_roi-fcn-tiny__run_0002` for that legacy route. The default runtime claim should therefore be read as deterministic locator plus tri-stream regressor, not ROI-FCN plus regressor.

## 9. Raw-Image Inference

The repository contains raw-image inference paths that compose separately trained components:

1. a crop or ROI locator
2. a distance/yaw regression model
3. preprocessing logic that reconstructs the input representation expected by the selected model
4. JSON and image artifacts for inspection and failure analysis

The v0.3 dual-stream path runs the ROI-FCN localiser, extracts a fixed ROI, generates the dual-stream model input, derives geometry features, and writes per-sample predictions with actual distance/yaw values and deltas.

The v0.4 tri-stream-facing path extends the same runtime family toward the tri-stream model contract, including separate distance and orientation inputs, orientation source-mode handling, brightness analysis, and foreground representation work.

This stage exposed an important system-level gap: some offline training metrics are much stronger than composed raw-image runtime metrics. That gap is useful engineering evidence. The project does not stop at preprocessed validation performance; it measures degradation introduced by crop localisation, runtime preprocessing, and model composition.

## 10. Live-Local Inference Runtime

The `06_live-inference_v0.3` project is the current live-local runtime and demo-stabilisation layer. It includes:

* model selection files for live-local artifacts
* metadata-only model manifest loading
* compatibility checks between selected artifacts and runtime contracts
* device policy handling for `auto`, `cuda`, and `cpu`
* atomic latest-frame file handoff
* duplicate-frame detection through frame hashes
* synthetic camera publisher for deterministic GUI smoke tests
* OpenCV/V4L2 camera source targeting `/dev/video0`
* deterministic `background_edge_v1` locator
* retained ROI-FCN legacy locator path
* manual fixed ROI and fixed-centre fallback locator paths
* static background capture and clear controls
* manual mask drawing, erasing, application, and clearing
* foreground extraction policy state with `threshold_foreground_v1` as default and `silhouette_contour_v2` retained as a legacy selectable path
* tri-stream live preprocessor
* PyTorch tri-stream inference engine
* camera and inference workers
* PySide6 GUI with camera/inference controls, preview panes, status readouts, prediction outputs, timing, logs, and debug artifact display

The current v0.3 distance/orientation selection is:

* distance/orientation: `260515-1301_ts-2d-cnn`
* topology id: `distance_regressor_tri_stream_yaw`
* topology variant: `tri_stream_yaw_v0_4`
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
| Distance MAE | `0.014845 m` |
| Distance RMSE | `0.024120 m` |
| Distance within `0.10 m` | `0.999067` |
| Distance within `0.25 m` | `0.999783` |
| Distance within `0.50 m` | `0.999917` |
| Yaw mean error | `1.032462 deg` |
| Yaw median error | `0.780045 deg` |
| Yaw p95 error | `2.436653 deg` |
| Yaw within `5 deg` | `0.997933` |

These are offline synthetic validation metrics for the selected model artifact. They should not be interpreted as real-camera deployment accuracy.

## 11. `incident_1`: Live Distance Spike Investigation

The `failure-analysis/incident_1` directory is an important part of the repository's technical evidence. It records a live-inference failure, reconstructs the pipeline state from artifacts, identifies the root cause, implements remediation, and records post-remediation traces.

The incident began with two near-identical live captures of a stationary Defender model producing sharply different distance estimates:

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

The physical measurement was not an OpenCV calibration validation; the camera had not yet been calibrated in that sense. It was still a useful scene reference: a vehicle filling a large part of the image did not support a roughly `5 m` prediction.

The trace evidence showed that the model was not the primary source of the spike. The accepted camera frame and ROI crop contained the vehicle, but the downstream silhouette recovery reduced the failing trace to a tiny `14 x 20 px` fragment. That fragment then drove:

* `x_distance_image`
* `x_orientation_image`
* `x_geometry`

Given those corrupted inputs, the model's large-distance output was explainable. The failure was deterministic preprocessing, not an unexplained neural-network regression.

The root cause was the intensity-based silhouette recovery policy. When the valid vehicle component touched the bottom border of the ROI, the recovery heuristic preferred a tiny non-border component and returned early. A much larger border-touching vehicle component was present but not selected.

The remediation had several parts:

* The contour/silhouette recovery selector was changed so border-touching components are scored together with non-border candidates. Border contact remains a weak penalty, but it can no longer make a tiny interior fragment beat a much larger plausible vehicle component.
* The live v0.3 default foreground path was changed to `threshold_foreground_v1`, which estimates the effective white background from the ROI, applies capped Otsu thresholding, performs cleanup, and feeds the resulting foreground mask into the tri-stream representation.
* The old contour/silhouette path remains selectable as `silhouette_contour_v2`, but it is no longer the default live path.
* A locator-relative consistency guard was added before inference. If a large accepted locator box collapses to an implausibly small foreground representation, the system reports a preprocessing failure instead of feeding corrupted inputs to the regressor.
* Regression tests were added for both the v4 silhouette algorithm and the v0.3 live preprocessor using the incident artifacts.

Post-remediation live traces from 2026-05-18 show the new foreground path producing vehicle-sized masks and plausible distance predictions:

| Frame hash prefix | Predicted distance | Foreground extraction | Foreground pixels | Foreground bbox | Locator confidence |
| --- | ---: | --- | ---: | ---: | ---: |
| `86683b78` | `2.0420 m` | `threshold_foreground_v1` | `16,742 px` | `126 x 175 px` | `0.8967` |
| `cc24cd51` | `1.7722 m` | `threshold_foreground_v1` | `29,830 px` | `199 x 231 px` | `0.9059` |
| `1d2d137e` | `1.5386 m` | `threshold_foreground_v1` | `49,486 px` | `292 x 225 px` | `0.9307` |

These traces are not a controlled repeatability study of one fixed frame; they are three accepted frame hashes from live operation. Their significance is narrower and still important: the previously observed failure mode no longer collapses the foreground representation into tiny geometry.

The incident is a strong example of applied ML systems debugging. The trace artifacts allowed the failure to be localised precisely:

```text
camera frame: good
ROI locator: good
ROI crop: good
silhouette recovery: failed
model input: corrupted
model output: explainable from corrupted input
```

That is the kind of evidence needed in a composed perception system. The project does not treat a bad prediction as an opaque model failure; it reconstructs the data path and turns the finding into a testable engineering change.

## 12. Representative Results

The table below separates offline preprocessed evaluation, raw-image composed inference, and live-local artifact selection. These are different evidence types and should not be collapsed into a single headline number.

| Artifact / Run | Evidence Type | Train / Validation Samples | Distance MAE | Distance RMSE | Distance within `0.10 m` | Yaw Mean Error | Yaw within `5 deg` | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `260415-1146_ds-2d-cnn/run_0001` | offline preprocessed dual-stream distance+yaw validation | `250,000 / 50,000` | `0.01007 m` | `0.01297 m` | `0.99996` | `1.49987 deg` | `0.97482` | Strong recorded offline preprocessed dual-stream result. |
| `260515-1301_ts-2d-cnn` | current v0.3 selected tri-stream offline validation artifact | `300,000 / 60,000` | `0.01485 m` | `0.02412 m` | `0.99907` | `1.03246 deg` | `0.99793` | Strong selected v0.3 model artifact under synthetic preprocessed validation. |
| `260504-1100_ts-2d-cnn__run_0001` | earlier live-local tri-stream selected artifact | `226,971 / 47,929` | `0.09756 m` | `0.12627 m` | `0.62826` | `3.87700 deg` | `0.74675` | Metadata-compatible earlier live candidate, weaker distance than later selected artifact. |
| `260415-1146_ds-2d-cnn` raw-image output | composed ROI-FCN plus dual-stream inference on `49,999` raw validation rows | `n/a / 49,999` | `0.11117 m` | `0.43346 m` | `0.92948` | `12.33194 deg` | `0.58491` | Shows runtime degradation and a crop-boundary distance tail. |
| `260425-1025_ds-2d-cnn` raw-image output | composed ROI-FCN plus brightness-normalised dual-stream inference on `49,999` raw validation rows | `n/a / 49,999` | `0.04784 m` | `0.10853 m` | `0.95312` | `16.48438 deg` | `0.38421` | Better bulk distance, but broad yaw underperformance. |
| `260420-1219_roi-fcn-tiny__run_0003` | ROI-FCN locator validation | `100,000 / 20,000` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | Mean centre error `3.1757 px`, p95 `7.7098 px`, full-containment success `0.9891`. |

Two conclusions follow from these results.

First, the repository contains strong offline evidence for the bounded preprocessed task. Second, end-to-end raw-image and live inference are harder than the offline task, and the repository measures and investigates that gap rather than hiding it.

## 13. Failure Analysis and Engineering Learnings

The raw-image inference outputs support operational failure analysis using a primary threshold of:

* distance failure: absolute error greater than `0.10 m`
* yaw failure: absolute error greater than `5 deg`
* clean success: distance error at most `0.05 m` and yaw error at most `2.5 deg`

For the `260415-1146_ds-2d-cnn` raw-image run on `49,999` validation rows:

* joint success was `56.73%`
* clean success was `17.44%`
* distance within `10 cm` was `92.95%`
* yaw within `5 deg` was `58.49%`
* the most severe distance failures were strongly associated with fixed ROI requests extending beyond the source image boundary
* yaw failures were dominated by a heavy tail, including near-180-degree orientation confusions

For the `260425-1025_ds-2d-cnn` raw-image run:

* joint success was `38.31%`
* clean success was `17.36%`
* distance within `10 cm` was `95.31%`
* yaw within `5 deg` was `38.42%`
* distance improved in the bulk, but yaw degraded across the main distribution

`incident_1` extends the same engineering pattern into live operation. It demonstrates that operational failures can arise after the locator and before the model, and that model-input artifact capture is essential for debugging composed ML systems.

The main learning is that model metrics alone are insufficient. In a multi-stage perception system, accuracy depends on the contracts and failure modes of every stage: camera capture, background handling, ROI selection, foreground extraction, geometry construction, model input rendering, and output decoding.

## 14. Testing and Engineering Discipline

The repository includes focused tests across multiple layers:

* v4 preprocessing integration, silhouette algorithms, foreground handling, and brightness normalisation
* topology contracts and task-runtime reporting
* resume features and epoch summaries
* ROI-FCN preprocessing, geometry, targets, data contracts, and training smoke tests
* raw-image inference sample execution and brightness analysis
* live inference model manifests, compatibility checks, frame handoff, frame selection, device policy, runtime parameters, ROI locators, tri-stream preprocessing, inference core, PyTorch engine, workers, GUI bridge, GUI main window, synthetic camera, and OpenCV/V4L2 camera source
* v0.3-specific tests for `background_edge_v1`, generic tri-stream preprocessing, foreground policy selection, manual mask application, trace artifact contents, GUI app wiring, and the `incident_1` preprocessing regression

The test coverage is strongest around contract boundaries, data-shape assumptions, and runtime glue. That is the appropriate emphasis for a multi-stage perception codebase where silent interface drift would be expensive.

The incident-specific tests are particularly valuable:

* `test_v4_silhouette_algorithms.py` verifies that the saved incident ROI crop now selects the large border-touching vehicle component rather than the tiny fragment.
* `test_generic_preprocessor.py` verifies that the v0.3 live preprocessor produces a large foreground mask, large geometry, and non-empty vehicle representation from the incident frame and locator bbox.

## 15. Technically Distinctive Features

The project demonstrates:

* end-to-end ownership across simulation, preprocessing, training, evaluation, inference, live GUI runtime, and incident analysis
* camera-footprint-aware synthetic placement rather than naive world-space sampling
* explicit data and model contracts across preprocessing, topology, runtime, and artifact selection
* preservation of geometric depth cues through fixed unscaled distance crops
* circular yaw regression through `sin/cos` targets
* learned ROI-FCN crop-centre localisation
* deterministic background/edge live localisation for inspectable demo operation
* manual masks and static background handling in the live runtime
* a tri-stream model contract separating distance image, orientation image, and geometry features
* runtime compatibility checking before model pairing
* frame-handoff and worker contracts for a live-local application
* trace bundles that capture accepted frames, locator outputs, preprocessing artifacts, model inputs, metadata, and inference outputs
* failure analysis that distinguishes offline validation quality from composed runtime quality

These are not presented as novel research contributions. They are practical engineering capabilities in applied ML and perception-system development.

## 16. Established Engineering Patterns Demonstrated

The repository implements established engineering patterns that are directly relevant to applied ML systems:

* PyTorch training loops with checkpoints, schedulers, resumes, metrics, plots, and model cards
* NPZ shard packing and schema validation
* OpenCV contour processing, foreground extraction, silhouette generation, and convex-hull fallback logic
* heatmap-based localisation and argmax decode
* notebook control surfaces backed by importable Python modules
* JSON and TOML artifacts for traceability and audit
* GUI-worker separation through payload contracts
* device policy management for CPU and CUDA execution paths
* artifact-backed incident analysis and regression tests

## 17. Limitations and Caveats

The current evidence should be read with the following constraints in mind:

* All recorded model training and validation evidence is synthetic.
* The task is limited to one known vehicle family, one camera setup, and a constrained operating geometry.
* The system is not a general detector, tracker, or scene-understanding model.
* The strongest offline result and the raw-image runtime results are materially different.
* The current live v0.3 model artifact has strong offline synthetic validation metrics, but the repository does not yet contain a full-corpus live-runtime evaluation with calibrated real-camera ground truth.
* The physical distance references in `incident_1` are useful diagnostic context, not a calibrated camera-validation result.
* `background_edge_v1` is deterministic and inspectable, but it is not a general object detector.
* ROI-FCN targets are bootstrapped from an existing crop heuristic, so the localiser initially learns that crop-centre definition rather than independently curated ground truth.
* The codebase is a research workspace with versioned subprojects, compatibility shims, and evolving runtime paths, not a polished packaged product.

These caveats are part of the technical value of the project. They keep the claims bounded and make the results easier to evaluate honestly.

## 18. Short Technical Summary

Raccoon Ball is a bounded computer-vision project for fixed-camera vehicle distance and yaw estimation. It combines Unity synthetic data generation, OpenCV preprocessing, PyTorch model training, learned and deterministic ROI localisation, raw-image inference, and a live PySide6 runtime.

The project is intentionally narrow: one known vehicle family, one fixed camera geometry, synthetic labelled data, and a constrained operating plane. Within that scope, it demonstrates the engineering work required to move from offline model training toward composed runtime inference, including data contracts, artifact compatibility checks, runtime preprocessing, trace capture, and failure analysis.

The most valuable result is not a single accuracy number. The project shows strong offline synthetic performance, measurable degradation in composed raw-image inference, and a live incident investigation where a large distance-prediction spike was traced to foreground extraction, remediated in code, and covered by regression tests. That makes the repository useful evidence of applied ML engineering, computer vision, evaluation discipline, and runtime integration capability.
