# Raccoon Ball Repository Technical Writeup v0.3

## Overview

This repository is a coherent end-to-end machine learning research system for monocular distance estimation in a tightly bounded synthetic environment. The implemented stack spans:

1. synthetic dataset generation in Unity (`rb_synthetic-data_3`)
2. preprocessing and representation packing (`synthetic-data-processing-v2.0`)
3. model training, evaluation, and experiment operations (`rb-training-v2.0`)

The technical objective is not generic object detection or unconstrained scene understanding. The task is a controlled regression problem: estimate vehicle distance, and optionally 3D position, for a known vehicle instance in a constrained scene, observed by a fixed calibrated camera over a constrained movement plane. The project is therefore best understood as a bounded perception system for falsifiable geometry-aware experiments rather than as a general-purpose vision product.

The repository evidence remains consistent with sole development of the full machine learning pipeline, including data generation, preprocessing contracts, model development, experiment tooling, and supporting technical documentation.

## Task Definition

At the training layer, the core target is `distance_m`, with support for an alternative `position_3d` mode. Two model families are implemented:

- a naive full-frame CNN baseline
- a dual-stream CNN/MLP regressor combining silhouette imagery with explicit bounding-box geometry features

The system preserves traceability from generated image through preprocessing contract to training artifact. This is visible in the consistent use of `run.json`, `samples.csv`, packed NPZ shards, split manifests, run-level config files, prediction dumps, and model cards.

## Benchmark Regime Evolution

The most important recent story is not only model iteration, but benchmark expansion.

Earlier corpus families used stepped motion along the `Z` axis with the vehicle staying close to the camera centre line:

- `26-04-01_v010-train` used `PosX` jitter of `+-0.5 m`
- `26-04-06_v014-train` and `26-04-06_v015-validate` reduced `PosX` jitter to `+-0.25 m`

Those corpora were still effectively centre-line datasets. Coarse manifest analysis of `26-04-06_v015-validate-shuffled-ds` showed that all `10,020` validation samples fell in the middle third of the image by detected `x` centre.

The next stage moved to full-frame coverage:

- `26-04-11_v018-train` and `26-04-11_v019-validate` use camera-footprint stratified placement
- `DepthBandCount = 12`
- `LateralBinCount = 10`
- `PosX` and `PosZ` jitter are both `+-0.2 m`
- `RotY` jitter is only `+-6.25°`, for a total range of `12.5°`

The latest corpus family keeps that broader spatial regime but makes pose substantially harder:

- `26-04-11_v020-train` and `26-04-11_v021-validate` retain the footprint-stratified full-frame setup
- the deliberate change is yaw range: `RotYMinDeg = -180°`, `RotYMaxDeg = 180°`
- this produces full `360°` yaw variation for the Defender rather than a narrow local perturbation

The resulting validation distribution remains full-frame. Coarse manifest analysis of `26-04-11_v021-validate-shuffled` shows:

- left third: `16,328` samples
- middle third: `17,379` samples
- right third: `16,293` samples

This means the latest benchmark is harder in two different ways at once:

- full image-plane coverage rather than centre-line concentration
- full-rotation viewpoint variation rather than near-constant orientation

## Repository Architecture

### 1. Unity Synthetic Dataset Generation (`rb_synthetic-data_3`)

The Unity generator is more structured than a simple scripted camera sweep.

Implemented elements include:

- manual camera capture through `RenderTexture` and `Texture2D` in `CaptureService.cs`
- manifest and run metadata writing via `ManifestWriter`, `ManifestRowMapper`, and `RunMetadataWriter`
- deterministic sample naming and path construction in `FileNamingStrategy.cs`
- explicit Euclidean target generation in `DistanceCalculator.cs`
- camera-footprint-aware placement planning in `StratifiedPlacementPlanner.cs`
- projection-based feasibility validation in `VehicleProjectionValidator.cs`
- batch-oriented orchestration, attempt budgeting, and quota redistribution in `RunControllerBehaviour.cs`

The technically distinctive part is the placement strategy. Rather than sweeping depth uniformly or sampling naively in world space, the generator:

- projects the camera footprint onto the movement plane
- partitions that footprint into depth bands and lateral bins
- probes cell feasibility under image-space constraints
- allocates samples across feasible cells
- redistributes shortfall when a cell exhausts its attempt budget

This directly addresses a common synthetic-data failure mode in which nominal world-space coverage does not translate into useful image-space coverage.

### 2. Preprocessing Pipeline (`synthetic-data-processing-v2.0`)

The preprocessing repository implements a staged v4 dual-stream pipeline with explicit contracts:

1. `detect`
2. `silhouette`
3. `pack_dual_stream`
4. optional `shuffle`

The architectural strength here is contract-driven preprocessing. Representation metadata is written back into `run.json` under `PreprocessingContract`, and each sample row carries stage statuses and stage-specific metadata in `samples.csv`. Downstream training can therefore validate not only that files exist, but that representation semantics are compatible.

Implemented preprocessing features include:

- a detector abstraction with both Ultralytics YOLO and edge-based ROI backends
- ROI silhouette extraction with contour generation and convex-hull fallback
- dual-stream NPZ packing with fixed feature schema and validation
- optional backward-compatible arrays for fair comparison with older baselines
- integration and algorithm tests

One particularly good design choice is the treatment of the silhouette crop. The crop is placed on a fixed canvas without geometric rescaling. This preserves apparent object size as a depth cue rather than normalizing it away.

The repository substantiates the main benchmark stages of the project in two ways:

- directly present centre-line dual-stream training corpus `26-04-06_v014-train-shuffled-ds`: `50,063` samples
- directly present centre-line dual-stream validation corpus `26-04-06_v015-validate-shuffled-ds`: `10,020` samples
- directly present full-frame full-yaw training corpus `26-04-11_v020-train-shuffled`: `250,000` samples
- directly present full-frame full-yaw validation corpus `26-04-11_v021-validate-shuffled`: `50,000` samples
- historically substantiated full-frame limited-yaw benchmark: `260412-1759_ds-2d-cnn/run_0002` records the earlier `250,000 / 50,000` full-frame run with `12.5°` total yaw range

Important accuracy note: the checked-in successful production corpora were built with the edge ROI detector backend, not YOLO. YOLO support is implemented and documented, but the current substantiated training outputs rely on the edge detector path.

**Detector note:** Early experiments with off-the-shelf Ultralytics YOLO did not reliably identify the Defender 90 and produced incorrect labels (for example, `keyboard`). The current checked-in successful results therefore rely on the edge ROI backend rather than a YOLO-backed production path. A custom detector, fine-tuned YOLO model, or alternative live ROI strategy is likely to be required for real-world deployment.

### 3. Training and Evaluation Infrastructure (`rb-training-v2.0`)

The training repository is substantially more than a single model script. It contains:

- topology registration and versioned topology definitions
- manifest-aware dataset loading and validation
- shard streaming and RAM caching
- overlap checks between train and validation corpora
- run directory creation and artifact writing
- resume-state persistence with topology and dataset compatibility checks
- evaluation outputs including prediction CSVs and plots
- notebook-driven operational control with helper modules for implementation logic

Implemented model families include:

- `distance_regressor_2d_cnn`
- `distance_regressor_dual_stream`
- `distance_regressor_global_pool_cnn` as an additional registered extension point

The dual-stream topology is the most substantial model design in the repository. The shape stream is a CNN over `silhouette_crop`; the geometry stream is an MLP over a 10-element bounding-box feature vector. The v0.2 revision replaced `BatchNorm2d` with `GroupNorm` and removed fusion dropout by default in response to observed instability in v0.1.

The later full-frame runs keep the same `dual_stream_v0_2` topology but move to a `300 x 300` silhouette canvas and larger footprint-stratified corpora. The latest `260413-1847_ds-2d-cnn/run_0002` also demonstrates that the resume workflow is used in normal operation rather than existing only as scaffolding: it explicitly resumes from `run_0001` and completes successfully.

## Separation of Concerns and Supporting Documentation

A notable repository-level strength is the deliberate split between notebook control surfaces and implementation code.

Examples:

- `rb-training-v2.0/notebooks/02_train_ds_2d_cnn_v0.7.ipynb` acts as a control surface and delegates logic to `src`
- `synthetic-data-processing-v2.0/rb_ui_v4/*.ipynb` act as thin launch surfaces over `rb_pipeline_v4`

This separation is reinforced by written technical documents in:

- `synthetic-data-processing-v2.0/documents`
- `rb-training-v2.0/documents`
- this repository-level `documents` directory

Representative documents include:

- `Synthetic Data Processing v2.0 Specification v0.1.md`
- `Distance Regressor Dual Stream Definition v0.2.md`
- `Training Session Control Panel Implementation Playbook v0.1.md`
- `Adding New Topology v2.0.md`
- `Raccoon Ball Training Repo Notebook Standard v0.1.1`

This is relevant to potential employers because it demonstrates specification-driven development, architectural discipline, and a deliberate attempt to control drift in an active research codebase without relying on heavyweight external process.

## What Has Been Implemented

### Implemented End-to-End Workflow

- synthetic sample generation in Unity with run metadata and per-sample manifests
- preprocessing stage contracts and row-level stage status tracking
- detector abstraction supporting both YOLO and edge ROI extraction
- silhouette generation with fallback recovery logic
- dual-stream shard packing with optional compatibility outputs
- strict training-side manifest and shard validation
- topology registry and versioned topology definitions
- naive full-frame CNN baseline training
- dual-stream CNN/MLP training
- resume-state persistence and guarded resume workflows
- evaluation artifacts: metrics, history, sample predictions, scatter plots, residual plots, run manifests, and model cards

### Testing and Verification

The Python test suites pass when run from the individual project roots:

- `rb-training-v2.0`: `14 passed`
- `synthetic-data-processing-v2.0`: `6 passed`

Practical caveat: invoking those suites from the monorepo root failed due project-local import-path assumptions. This does not invalidate the tests, but it does show that the repositories are currently intended to be run from their own roots rather than from the workspace top level.

## Results and Assessment

### Representative Completed Runs

| Run | Model / Regime | Train / Validation Samples | Validation MAE | Validation RMSE | Accuracy within 0.10 m | Assessment |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `260407-1756_2d-cnn/run_0002` | naive full-frame CNN on centre-line `+-0.25 m` regime | `50,063 / 10,020` | `0.05338 m` | `0.06929 m` | `0.87375` | Best substantiated single-stream baseline. |
| `260411-1104_ds-2d-cnn/run_0001` | dual-stream v0.2 on centre-line `+-0.25 m` regime | `50,063 / 10,020` | `0.01512 m` | `0.02137 m` | `0.99950` | Best result on the narrow benchmark. |
| `260412-1759_ds-2d-cnn/run_0002` | dual-stream v0.2 on full-frame, limited-yaw (`12.5°` total yaw range) regime | `250,000 / 50,000` | `0.02785 m` | `0.09972 m` | `0.97892` | First completed full-frame benchmark run. Strong median accuracy, but a pronounced right-edge tail. |
| `260413-1847_ds-2d-cnn/run_0002` | dual-stream v0.2 on full-frame, full-yaw (`360°`) regime | `250,000 / 50,000` | `0.04652 m` | `0.10753 m` | `0.93460` | First completed full-rotation benchmark run. Harder overall, with edge-of-frame failures still present but less one-sided. |

### Assessment of the Most Recent Run

The completed artifact for the newest model is `260413-1847_ds-2d-cnn/run_0002`, created on `2026-04-14` as a resumed continuation of `run_0001`.

Substantiated metrics from `metrics.json` and `history.csv` are:

- best epoch: `30`
- configured stop epoch: `32`
- validation loss: `0.005178`
- validation MAE: `0.046516 m`
- validation RMSE: `0.107527 m`
- validation accuracy within `0.10 m`: `0.93460`
- validation accuracy within `0.25 m`: `0.98956`
- validation accuracy within `0.50 m`: `0.99354`

An important operational detail is that this run reached the configured epoch budget rather than stopping early on patience. The best validation loss occurred at epoch `30`, and the final two epochs remained close to that optimum. This suggests the experiment was still productive late in training and may not yet be fully exhausted.

### Comparison with the Previous Full-Frame Limited-Yaw Run

Relative to `260412-1759_ds-2d-cnn/run_0002`, the latest full-rotation run shows a clear shift in error profile.

Headline metric changes are:

- validation MAE worsened from `0.02785 m` to `0.04652 m` (`+67.0%`)
- validation RMSE worsened from `0.09972 m` to `0.10753 m` (`+7.8%`)
- accuracy within `0.10 m` dropped from `0.97892` to `0.93460` (`-4.43` percentage points)
- accuracy within `0.25 m` improved slightly from `0.98586` to `0.98956`
- accuracy within `0.50 m` improved slightly from `0.99234` to `0.99354`

This is a nuanced result. Fine-grained accuracy became materially worse, but coarse distance-band correctness remained very high. The most defensible interpretation is that full `360°` yaw makes the regression problem broadly harder, while still leaving the model usually in the correct approximate distance range.

Quantile comparison supports that reading:

- previous full-frame limited-yaw run:
  - median absolute error: `0.0141 m`
  - `95th` percentile: `0.0556 m`
  - `99th` percentile: `0.3980 m`
  - `99.9th` percentile: `1.4704 m`
- latest full-frame full-yaw run:
  - median absolute error: `0.0311 m`
  - `95th` percentile: `0.1102 m`
  - `99th` percentile: `0.2624 m`
  - `99.9th` percentile: `1.4556 m`

The latest run therefore appears worse in ordinary-case precision, but not clearly worse in the most catastrophic tail. That is consistent with the benchmark becoming more uniformly difficult rather than merely triggering one narrow failure mode.

### Distance and Pose Assessment

Distance-band analysis for the latest run shows that the model still behaves sensibly across the depth range, though accuracy degrades as distance increases:

- `1.5-2.5 m`: MAE `0.0420 m`
- `2.5-3.5 m`: MAE `0.0387 m`
- `3.5-4.5 m`: MAE `0.0420 m`
- `4.5-5.5 m`: MAE `0.0439 m`
- `5.5-7.5 m`: MAE `0.0731 m`

The yaw quadrants are comparatively similar:

- `-180° to -90°`: MAE `0.0463 m`
- `-90° to 0°`: MAE `0.0468 m`
- `0° to 90°`: MAE `0.0474 m`
- `90° to 180°`: MAE `0.0455 m`

Large-error counts by yaw quadrant are also broadly distributed rather than collapsing into a single orientation range. This suggests that the new `360°` regime is not creating one dominant pose-specific failure mode. Instead, it raises difficulty across pose space more generally.

### Pathology Analysis: Does the Right-Hand-of-Frame Issue Still Exist?

Yes, but not in the same form.

The earlier full-frame limited-yaw run had a strongly asymmetric failure mode. Coarse analysis showed:

- middle third of image: MAE `0.0138 m`, with `0` errors greater than `0.50 m`
- left third of image: MAE `0.0255 m`, with `49` errors greater than `0.50 m`
- right third of image: MAE `0.0408 m`, with `334` errors greater than `0.50 m`

The newest full-yaw run still shows a clear edge-of-frame pathology, but it is less exclusively right-sided:

- middle third of image: MAE `0.0335 m`, with `0` errors greater than `0.50 m`
- left third of image: MAE `0.0521 m`, with `144` errors greater than `0.50 m`
- right third of image: MAE `0.0548 m`, with `179` errors greater than `0.50 m`

The strongest honest conclusion is therefore:

- the model still has a pronounced lateral-edge weakness
- that weakness is no longer overwhelmingly concentrated on the right-hand side
- it now appears on both frame extremes, with a mild right-side bias rather than a dominant right-only pathology

The normalized lateral-position analysis is even clearer:

- centre band (`|cx_norm - 0.5| < 0.1`): MAE `0.0333 m`, with `0` errors greater than `0.50 m`
- extreme-lateral band (`|cx_norm - 0.5|` between `0.3` and `0.5`): MAE `0.0688 m`
- `318` of the `323` errors greater than `0.50 m` occur in that extreme-lateral band
- `126` of the `127` errors greater than `1.00 m` also occur there

This means the pathology remains strongly tied to extreme frame-edge placements. Full `360°` yaw does not remove that weakness. Instead, it appears to add broader pose difficulty on top of it.

Cross-analysis of edge position and yaw quadrant shows that these failures occur at both frame edges across all yaw quadrants. The middle third of the frame contains no errors greater than `0.50 m` in any yaw quadrant. This supports the view that the dominant unresolved issue is still lateral-edge robustness rather than one narrow orientation bug.

The largest outliers remain systematic overestimates, often predicting approximately `5.9-6.5 m` for true distances around `3.2-4.1 m`, and they occur at both left and right frame extremes with mixed yaw values. That is consistent with a persistent edge-placement failure mode rather than a purely yaw-specific pathology.

## What Appears Technically Distinctive

The following aspects stand out as more than routine implementation:

- end-to-end ownership across synthetic generation, preprocessing, training, evaluation, and experiment operations
- camera-footprint stratified placement rather than naive world-space randomization
- projection-based sample validity checks before capture
- preservation of apparent size cues by centering ROI silhouettes on a fixed canvas without rescaling
- explicit preprocessing contracts carried from `run.json` into training artifacts
- fair-comparison support between older and newer representations
- topology versioning with compatibility signatures and guarded resume support
- deliberate benchmark design that first broadened spatial coverage and then broadened pose coverage to expose hidden failure modes
- written technical playbooks governing notebook/control-panel architecture and topology extension

## What Is Strong Competent Implementation of Known Ideas

The following work is less novel, but clearly well executed:

- modular PyTorch training loops with Huber loss, early stopping, scheduler support, and structured artifact output
- evaluation output generation with metrics, prediction dumps, and plots
- dataset manifest validation and split leakage checks
- streaming NPZ loaders with memory-aware shard caching
- notebook-driven operational tooling over Python helper modules
- tmux-based long-running training control
- unit and integration tests around preprocessing and training infrastructure

## Weaknesses, Limitations, and Caveats

This project is strong, but its claims should remain bounded.

- The results are entirely synthetic. There is no checked-in evidence of transfer to real imagery.
- The task is deliberately constrained: one known object family, one fixed calibrated camera, one movement plane, and limited environment diversity. High accuracy in this context should not be presented as broad scene understanding.
- Training and validation are separate corpora and the loaders check for overlap, but both are generated by the same synthetic pipeline family. Domain gap remains the major open question.
- The current production corpora use the edge ROI detector backend. YOLO support exists, but the checked-in successful training results do not yet substantiate a YOLO-backed production path.
- Some narrative artifacts lag the implementation. Generated model cards still describe the input as full-frame grayscale imagery, whereas the actual v4 representation is `silhouette_crop + bbox_features`.
- The latest full-yaw result suggests that the current representation and topology are not yet robust to the combination of extreme lateral placement and unrestricted yaw variation.

## Current Status

At the current repository snapshot:

- the end-to-end path from Unity generation to packed corpora to trained model is functioning
- the dual-stream v0.2 architecture is implemented and operational on narrow, full-frame, and full-rotation benchmarks
- the best narrow-benchmark result remains `260411-1104_ds-2d-cnn/run_0001`
- the best full-frame fine-accuracy result remains `260412-1759_ds-2d-cnn/run_0002`
- `260413-1847_ds-2d-cnn/run_0002` establishes the first completed full-rotation benchmark result
- the principal next technical problem is robustness under combined edge-of-frame placement and unrestricted yaw variation

This remains an active research system rather than a closed benchmark report.

## Concrete Skills and Tools Demonstrated

### Machine Learning and Data

- PyTorch model development
- distance regression and optional 3D position regression
- multi-input model design
- topology and experiment versioning
- dataset schema validation and artifact lineage
- streaming NPZ data loading and memory-aware caching
- evaluation metric design and residual analysis
- experiment resumption and run compatibility validation

### Computer Vision

- OpenCV-based preprocessing
- edge detection, contour extraction, connected components, and convex hull recovery
- bounding-box feature engineering
- object detector abstraction and Ultralytics YOLO integration
- representation design for monocular geometry cues

### Synthetic Data and Simulation

- Unity C# runtime scripting
- scripted camera capture via render textures
- procedural sample generation
- projection-based placement validation
- stratified spatial sampling over a camera-visible footprint
- synthetic dataset manifest design
- dataset regime design for benchmark falsification

### Tooling and Engineering

- Jupyter notebooks as operator control surfaces
- Python helper modules for notebook/system separation
- tmux-based training orchestration
- resume-state persistence and guarded resume workflows
- structured run artifact generation
- unit and integration testing
- specification-driven development through internal documents and playbooks

### Languages and Frameworks

- Python
- C#
- PyTorch
- NumPy
- pandas
- OpenCV
- Unity
- Jupyter / ipywidgets
- tmux

## Overall Assessment

This repository provides evidence of end-to-end ownership of a bounded applied machine-learning and computer-vision system. Its strongest value is not as proof of broad deployment readiness, but as a legible demonstration of technical judgement: synthetic data generation was structured rather than ad hoc, preprocessing was treated as a contract-backed representation problem, training and evaluation were instrumented carefully, and the benchmark was deliberately expanded when the earlier regime became too flattering.

The most persuasive aspect of the work is therefore not any single headline metric in isolation, but the presence of a complete experimental loop: generate data, define and preserve representation semantics, train against explicit corpora, inspect failure patterns, and tighten the benchmark when weaknesses become visible.

The current evidence supports a bounded claim: this repository demonstrates the ability to design, build, instrument, and iteratively stress-test a complete monocular perception pipeline under controlled conditions, while remaining candid about where the present evidence stops. It should not be read as evidence of broad real-world visual generalisation or deployment readiness.
