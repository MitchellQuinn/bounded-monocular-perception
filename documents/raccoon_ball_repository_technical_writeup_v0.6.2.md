# Raccoon Ball: Bounded Monocular Perception System

## 1. Project Overview

Raccoon Ball is a bounded computer-vision and applied-ML project for estimating vehicle distance and yaw from a fixed monocular camera view under controlled synthetic conditions.

The project demonstrates end-to-end ownership of a perception pipeline spanning synthetic data generation, preprocessing contracts, PyTorch model training, learned ROI localisation, raw-image inference, failure analysis, and a live local GUI runtime.

The system is deliberately scoped. It is not intended to solve general autonomous driving, open-world object detection, multi-object tracking, or unconstrained real-world scene understanding. Its purpose is to investigate a narrower engineering question:

> Can a fixed-camera system observing a known vehicle in a constrained scene estimate useful vehicle state from image-based geometric cues, and how does performance change as the system moves from offline validation into composed runtime inference?

This makes the repository most useful as evidence of applied machine-learning engineering, computer vision, data-pipeline design, experimental discipline, and runtime integration.

## 2. Problem Scope

The central task is intentionally narrow:

* **Input:** a monocular full-frame image from a fixed calibrated camera
* **Primary output:** vehicle distance in metres
* **Secondary output:** vehicle yaw/orientation
* **Runtime support task:** predict the crop centre required to extract a fixed `300 x 300` region of interest for downstream regression

The current system assumes:

* one known vehicle family
* one fixed camera geometry
* one constrained movement plane
* synthetic imagery and synthetic labels
* controlled full-frame captures rather than arbitrary real-world scenes

Those constraints are a design choice. They keep the task falsifiable and allow the project to focus on the engineering boundary between synthetic benchmark performance and runtime behaviour.

## 3. Repository Architecture

The repository is organised as a versioned multi-project workspace:

* `01_rb_synthetic-data_3`: Unity/C# synthetic image generation
* `02_synthetic-data-processing-v4.0`: OpenCV and NumPy preprocessing, detection, silhouette generation, and dual-stream / tri-stream packing
* `03_rb-training-v2.0`: PyTorch training, topology registry, model evaluation, resume support, and reporting
* `04_ROI-FCN`: preprocessing and training for a crop-centre heatmap localiser
* `05_inference-v0.3-ds`: raw-image inference using ROI-FCN plus dual-stream distance/yaw models
* `05_inference-v0.4-ts`: tri-stream-facing inference work and brightness-analysis tooling
* `06_live-inference_v0.1`: live-local inference runtime with camera input, frame handoff, model registry, preprocessing, inference engine, workers, and GUI

This structure reflects a research-engineering codebase moving from offline experiments toward runtime composition. It is not packaged as a finished product, but it contains real integration surfaces, tests, artifacts, and runtime contracts.

## 4. Synthetic Data Generation

The Unity generator creates full-frame synthetic images with structured run metadata and sample manifests. It is designed to produce controlled labelled data for the fixed-camera perception task rather than arbitrary visual variety.

Key generator components include:

* `CaptureService.cs` for render-texture capture
* `RunControllerBehaviour.cs` for batch orchestration, cancellation, manifest flushing, and attempt-budget handling
* `ManifestWriter.cs`, `ManifestRowMapper.cs`, and `RunMetadataWriter.cs` for traceable run outputs
* `FileNamingStrategy.cs` for deterministic sample naming
* `DistanceCalculator.cs` for explicit target derivation
* `StratifiedPlacementPlanner.cs` for camera-footprint-aware placement
* `VehicleProjectionValidator.cs` for image-space feasibility checks

The main design choice is the placement strategy. Instead of sampling arbitrary world positions, the generator projects the camera footprint onto the movement plane, divides the usable footprint into depth and lateral cells, validates projected vehicle bounds in image space, and redistributes quota when cells exhaust their attempt budget.

This gives better coverage for the fixed-camera perception problem than naive world-space randomisation.

The generator writes:

* `images/*.png`
* `manifests/run.json`
* `manifests/samples.csv`
* `runlog.txt`

This gives downstream stages explicit lineage rather than relying on filenames alone.

## 5. Preprocessing and Representation Design

The preprocessing layer is contract-driven and stage-based. The current v4 pipeline supports:

* `detect`: edge-based or detector-style Defender localisation metadata
* `silhouette`: ROI silhouette generation with contour processing and convex-hull fallback
* `pack_dual_stream`: `300 x 300` distance/yaw regression inputs with geometry features
* `pack_tri_stream`: separate distance image, orientation image, and geometry streams
* corpus shuffle and notebook control surfaces for repeated training workflows

The strongest representation choices are:

* fixed `300 x 300` ROI canvases for distance inference
* no rescaling in the distance stream, preserving apparent object size as a depth cue
* a 10-element geometry vector: `cx_px`, `cy_px`, `w_px`, `h_px`, `cx_norm`, `cy_norm`, `w_norm`, `h_norm`, `aspect_ratio`, `area_norm`
* circular yaw targets represented as `sin/cos`
* optional foreground-only brightness normalisation for the distance stream
* a tri-stream contract that separates distance evidence from orientation evidence

The current tri-stream preprocessing contract is `rb-preprocess-v4-tri-stream-orientation-v1`. It writes:

* `x_distance_image`
* `x_orientation_image`
* `x_geometry`
* `y_distance_m`
* `y_yaw_deg`
* `y_yaw_sin`
* `y_yaw_cos`

The distance stream uses a fixed unscaled ROI canvas so that apparent object size remains available as a depth cue. The orientation stream uses a target-centred image scaled by silhouette foreground extent so that yaw evidence is presented in a more normalised form.

That split is technically meaningful: distance benefits from preserving scale, while yaw benefits from a more normalised orientation view.

## 6. Model Training and Evaluation

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

Yaw is modelled through circular regression using `sin/cos` targets rather than direct naive angle regression. The training runtime resolves prediction heads and target heads from topology contracts, allowing the loss and reporting paths to work across scalar and multitask models.

## 7. ROI-FCN Crop-Centre Localisation

The ROI-FCN subsystem turns crop placement into a learned task. It is trained to predict the centre of the fixed ROI required by the downstream distance/orientation model.

The ROI-FCN preprocessing path preserves:

* full-frame grayscale locator input
* target centre in original-image coordinates
* target centre in locator-canvas coordinates
* resize scale
* padding offsets
* optional bootstrap box metadata

The checked-in ROI-FCN artifact `260420-1219_roi-fcn-tiny__run_0003` uses:

* topology id: `roi_fcn_tiny`
* topology variant: `tiny_v1`
* locator canvas: `480 x 300`
* downstream ROI crop: `300 x 300`
* supervision: Gaussian heatmap
* decode: deterministic argmax back into source-image coordinates
* training split: `100,000` samples
* validation split: `20,000` samples

Its validation metrics are:

| Metric                       |       Value |
| ---------------------------- | ----------: |
| Mean centre error            | `3.1757 px` |
| Median centre error          | `2.4354 px` |
| p95 centre error             | `7.7098 px` |
| ROI full-containment success |    `0.9891` |

These results show that the learned localiser performs well on its bounded synthetic localisation task.

## 8. Raw-Image Inference

The repository contains raw-image inference paths that compose separately trained components:

1. an ROI-FCN crop-centre localiser
2. a distance/yaw regression model
3. preprocessing logic that reconstructs the input representation expected by the selected model
4. JSON and ROI-image artifact writing for inspection and failure analysis

The v0.3 dual-stream path runs the ROI-FCN localiser, extracts a fixed ROI, generates the dual-stream model input, derives geometry features, and writes per-sample predictions with actual distance/yaw values and deltas.

The v0.4 tri-stream-facing path extends the same runtime family toward the tri-stream model contract, including separate distance and orientation inputs, orientation source-mode handling, and brightness-analysis support.

This stage exposed an important system-level gap: some offline training metrics are much stronger than the composed raw-image runtime metrics. That gap is useful engineering evidence. The project does not stop at preprocessed validation performance; it measures the degradation introduced by crop localisation, runtime preprocessing, and model composition.

## 9. Live-Local Inference Runtime

The `06_live-inference_v0.1` project moves the work from offline experiments toward a runnable local application. It includes:

* model selection files for live-local artifacts
* metadata-only model manifest loading
* compatibility checks between selected distance/yaw and ROI-FCN artifacts
* device policy handling for `auto`, `cuda`, and `cpu`
* atomic latest-frame file handoff
* duplicate-frame detection through frame hashes
* synthetic camera publisher for deterministic GUI smoke tests
* OpenCV/V4L2 camera source targeting `/dev/video0`
* ROI-FCN live locator adapter
* tri-stream live preprocessor
* PyTorch tri-stream inference engine
* camera and inference workers
* PySide6 GUI with camera/inference controls, preview, status, prediction, timing, and log readouts

The current live-local model selection pairs:

* distance/orientation: `260504-1100_ts-2d-cnn__run_0001`
* ROI-FCN: `260420-1219_roi-fcn-tiny__run_0003`

The selected distance/orientation artifact declares:

* topology id: `distance_regressor_tri_stream_yaw`
* topology variant: `tri_stream_yaw_v0_1`
* preprocessing contract: `rb-preprocess-v4-tri-stream-orientation-v1`
* input mode: `tri_stream_distance_orientation_geometry`
* representation kind: `tri_stream_npz`
* output keys: `distance_m`, `yaw_sin_cos`

The selected pairing passes metadata compatibility checks: the ROI-FCN `300 x 300` crop matches the distance canvas and fits within the `480 x 300` locator canvas. This is not the same as full runtime validation, but it demonstrates explicit integration discipline.

## 10. Representative Results

The table below separates offline preprocessed evaluation, raw-image composed inference, and live-local artifact selection. These are different evidence types and should not be collapsed into a single headline number.

| Artifact / Run                           | Evidence Type                                                                                     | Train / Validation Samples | Distance MAE | Distance RMSE | Distance within `0.10 m` | Yaw Mean Error | Yaw within `5 deg` | Interpretation                                                                                                         |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------: | -----------: | ------------: | -----------------------: | -------------: | -----------------: | ---------------------------------------------------------------------------------------------------------------------- |
| `260415-1146_ds-2d-cnn/run_0001`         | offline preprocessed dual-stream distance+yaw validation                                          |         `250,000 / 50,000` |  `0.01007 m` |   `0.01297 m` |                `0.99996` |  `1.49987 deg` |          `0.97482` | Strongest recorded offline preprocessed result.                                                                        |
| `260504-1100_ts-2d-cnn__run_0001`        | current live-local tri-stream selected artifact                                                   |         `226,971 / 47,929` |  `0.09756 m` |   `0.12627 m` |                `0.62826` |  `3.87700 deg` |          `0.74675` | Metadata-compatible live candidate; good yaw distribution, weaker distance than the strongest offline dual-stream run. |
| `260415-1146_ds-2d-cnn` raw-image output | composed ROI-FCN plus dual-stream inference on `49,999` raw validation rows                       |             `n/a / 49,999` |  `0.11117 m` |   `0.43346 m` |                `0.92948` | `12.33194 deg` |          `0.58491` | Shows runtime degradation and a crop-boundary distance tail.                                                           |
| `260425-1025_ds-2d-cnn` raw-image output | composed ROI-FCN plus brightness-normalised dual-stream inference on `49,999` raw validation rows |             `n/a / 49,999` |  `0.04784 m` |   `0.10853 m` |                `0.95312` | `16.48438 deg` |          `0.38421` | Better bulk distance, but broad yaw underperformance.                                                                  |
| `260420-1219_roi-fcn-tiny__run_0003`     | ROI-FCN locator validation                                                                        |         `100,000 / 20,000` |        `n/a` |         `n/a` |                    `n/a` |          `n/a` |              `n/a` | Mean centre error `3.1757 px`, p95 `7.7098 px`, full-containment success `0.9891`.                                     |

Two conclusions follow from these results.

First, the repository contains strong offline evidence for the bounded preprocessed task. Second, end-to-end raw-image inference is harder than the offline task, and the repository measures that gap rather than hiding it.

## 11. Failure Analysis and Engineering Learnings

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

The important point is not only the model score. The project identifies system-level failure modes rather than treating model metrics as isolated numbers. The main unresolved runtime issues are crop-boundary handling, close-range/lower-frame yaw robustness, orientation flip cases, and the distinction between preprocessed validation quality and composed raw-image runtime quality.

## 12. Testing and Engineering Discipline

The repository includes focused tests across multiple layers:

* v4 preprocessing integration, silhouette algorithms, and brightness normalisation
* topology contracts and task-runtime reporting
* resume features and epoch summaries
* ROI-FCN preprocessing, geometry, targets, data contracts, and training smoke tests
* raw-image inference sample execution and brightness analysis
* live inference model manifests, compatibility checks, frame handoff, frame selection, device policy, runtime parameters, ROI-FCN locator, tri-stream preprocessing, inference core, PyTorch engine, workers, GUI bridge, GUI main window, synthetic camera, and OpenCV/V4L2 camera source

The test coverage is strongest around contract boundaries, data-shape assumptions, and runtime glue. That is the appropriate emphasis for a multi-stage perception codebase where silent interface drift would be expensive.

## 13. Technically Distinctive Features

The project demonstrates:

* end-to-end ownership across simulation, preprocessing, training, evaluation, inference, and live GUI runtime
* camera-footprint-aware synthetic placement rather than naive world-space sampling
* explicit data and model contracts across preprocessing, topology, runtime, and artifact selection
* preservation of geometric depth cues through fixed unscaled distance crops
* circular yaw regression through `sin/cos` targets
* a separate learned ROI-FCN localiser for crop-centre prediction
* a tri-stream model contract separating distance image, orientation image, and geometry features
* runtime compatibility checking before model pairing
* frame-handoff and worker contracts for a live-local application
* failure analysis that distinguishes offline validation quality from composed runtime quality

## 14. Established Engineering Patterns Demonstrated

The repository also demonstrates practical implementation of established engineering patterns:

* PyTorch training loops with checkpoints, schedulers, resumes, metrics, plots, and model cards
* NPZ shard packing and schema validation
* OpenCV contour processing, silhouette generation, and convex-hull fallback logic
* heatmap-based localisation and argmax decode
* notebook control surfaces backed by importable Python modules
* JSON artifacts for traceability and audit
* GUI-worker separation through payload contracts
* device policy management for CPU and CUDA execution paths

These are not presented as novel research contributions. They are valuable engineering capabilities in applied ML and perception-system development.

## 15. Limitations and Caveats

The current evidence should be read with the following constraints in mind:

* All recorded model evidence is synthetic; no successful real-image transfer is demonstrated in the recorded results.
* The task is limited to one known vehicle family, one camera setup, and a constrained operating geometry.
* The system is not a general detector, tracker, or scene-understanding model.
* The strongest offline result and the raw-image runtime results are materially different.
* The current live-local tri-stream model selection is metadata-compatible, but the repository does not yet show a full-corpus live tri-stream runtime evaluation matching the available raw-image dual-stream artifacts.
* ROI-FCN targets are bootstrapped from an existing crop heuristic, so the localiser initially learns that crop-centre definition rather than an independently curated ground truth.
* The codebase is a research workspace with versioned subprojects, compatibility shims, and evolving runtime paths, not a polished packaged product.

These caveats are part of the technical value of the project. They keep the claims bounded and make the results easier to evaluate honestly.

## 16. Skills Demonstrated and Role Relevance

Raccoon Ball demonstrates applied machine-learning and research-engineering capability across the full lifecycle of a bounded perception system.

The strongest signal is the breadth of practical ownership. The project covers the work that sits between model training and usable systems: data generation, data lineage, preprocessing contracts, representation design, model training, artifact management, inference composition, runtime compatibility checking, GUI integration, and failure analysis.

This makes the repository relevant to roles involving:

* machine-learning engineering
* computer vision engineering
* applied research engineering
* robotics-adjacent perception tooling
* simulation-based data generation
* ML systems prototyping
* validation, diagnostics, and evaluation tooling for perception systems

The project is particularly relevant where teams need engineers who can connect experimental ML work to runnable systems, maintain explicit data and model contracts, and investigate why performance changes when a model leaves the controlled training/evaluation path.

## 17. CV-Ready Summary

Possible CV phrasing:

* Built a bounded monocular computer-vision stack for fixed-camera vehicle distance and yaw estimation, spanning Unity synthetic data generation, OpenCV preprocessing, PyTorch model training, ROI localisation, raw-image inference, and a live PySide6 runtime.
* Designed contract-driven preprocessing and model interfaces for dual-stream and tri-stream perception models, including fixed-canvas distance inputs, orientation-specific image streams, geometry features, and circular yaw regression via `sin/cos` targets.
* Implemented a Unity synthetic data generator with camera-footprint-aware stratified placement, projection validation, structured manifests, deterministic sample naming, and batch orchestration.
* Trained and evaluated multi-task PyTorch regression models and a heatmap-based ROI-FCN crop-centre localiser, with metrics, model cards, checkpoints, resume support, and sample prediction artifacts.
* Built a live-local inference prototype with model-selection metadata, compatibility checks, ROI-FCN localisation, tri-stream preprocessing, PyTorch inference, atomic frame handoff, synthetic and V4L2 camera sources, worker lifecycles, and GUI status/readout controls.
* Performed operational failure analysis separating distance, yaw, joint, crop-boundary, and orientation-flip failure modes, revealing the gap between offline preprocessed validation and composed raw-image runtime performance.

Technologies demonstrated: Python, PyTorch, NumPy, pandas, OpenCV, PySide6, Unity, C#, Jupyter, pytest, JSON/TOML configuration, and CUDA-aware local inference.

## 18. Short External Summary

Raccoon Ball is a bounded computer-vision project for fixed-camera vehicle distance and yaw estimation. It combines Unity synthetic data generation, OpenCV preprocessing, PyTorch model training, learned ROI localisation, raw-image inference, and a live PySide6 runtime.

The project is intentionally narrow: one known vehicle family, one fixed camera geometry, synthetic labelled data, and a constrained operating plane. Within that scope, it demonstrates the engineering work required to move from offline model training toward composed runtime inference, including data contracts, artifact compatibility checks, runtime preprocessing, and failure analysis.

The most valuable result is not a single accuracy number. The project shows both strong offline synthetic performance and measurable degradation in composed raw-image inference, then uses that gap to identify system-level failure modes such as crop-boundary issues and yaw/orientation confusions. That makes it useful as evidence of applied ML engineering, computer vision, evaluation discipline, and runtime integration capability.
