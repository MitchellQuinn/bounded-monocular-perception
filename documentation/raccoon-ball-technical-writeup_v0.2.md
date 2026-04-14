# Raccoon Ball Repository Technical Writeup v0.2

## Overview

This repository is a coherent end-to-end machine learning research system for monocular distance estimation in a tightly bounded synthetic environment. The implemented stack spans:

1. synthetic dataset generation in Unity (`rb_synthetic-data_3`)
2. preprocessing and representation packing (`synthetic-data-processing-v2.0`)
3. model training, evaluation, and experiment operations (`rb-training-v2.0`)

The technical objective is not generic object detection or unconstrained scene understanding. The task is a controlled regression problem: estimate vehicle distance, and optionally 3D position, for a known object class observed by a fixed calibrated camera over a constrained movement plane. The work is therefore best understood as a bounded perception system for falsifiable geometry-aware experiments rather than as a general-purpose vision product.

The repository evidence is consistent with sole development of the full machine learning pipeline, including data generation, preprocessing contracts, model development, experiment tooling, and supporting technical documentation.

## Task Definition

At the training layer, the core target is `distance_m`, with support for an alternative `position_3d` mode. Two model families are implemented:

- a naive full-frame CNN baseline
- a dual-stream CNN/MLP regressor combining silhouette imagery with explicit bounding-box geometry features

The system is designed to preserve traceability from generated image through preprocessing contract to training artifact. This is visible in the use of `run.json`, `samples.csv`, packed NPZ shards, training manifests, run-level config files, prediction dumps, and model cards.

## Benchmark Regime Evolution

An important recent change is that the benchmark itself has become materially harder.

Earlier corpus families used stepped motion along the `Z` axis with the vehicle remaining near the camera centre line:

- `26-04-01_v010-train` used `PosX` jitter of `+-0.5 m`
- `26-04-06_v014-train` and `26-04-06_v015-validate` reduced `PosX` jitter to `+-0.25 m`

Those corpora were still essentially centre-line datasets. Coarse manifest analysis of `26-04-06_v015-validate-shuffled-ds` shows that all `10,020` validation samples fall in the middle third of the image by detected `x` centre.

The newer corpus family used by `260412-1759_ds-2d-cnn` changes this substantially:

- `26-04-11_v018-train` and `26-04-11_v019-validate` use camera-footprint stratified placement
- `DepthBandCount = 12`
- `LateralBinCount = 10`
- `PosX` and `PosZ` jitter are both `+-0.2 m`
- the generator notes explicitly describe "Stratified camera-footprint placement over flat movement plane"

Coarse manifest analysis of `26-04-11_v019-validate-shuffled` shows the resulting validation set is spread across the image plane:

- left third: `18,255` samples
- middle third: `13,617` samples
- right third: `18,128` samples

This distinction matters when interpreting the results. `260411-1104_ds-2d-cnn` is the best result on the narrower centre-line benchmark. `260412-1759_ds-2d-cnn` is the first completed result on a substantially broader full-frame benchmark.

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

This is a non-trivial design choice. It directly addresses a common synthetic-data failure mode in which nominal world-space coverage does not translate into useful image-space coverage.

One caveat is that the current repository snapshot does not fully mirror the raw Unity image archive for the latest `v018` corpus. The checked-in Unity folder for `26-04-11_v018-train` currently contains `102` images, while the downstream shuffled training corpus used by the latest model contains `250,000` processed samples. The evidence therefore supports the generator design and downstream consumption, but not a complete in-repository raw image archive for that newest corpus family.

### 2. Preprocessing Pipeline (`synthetic-data-processing-v2.0`)

The preprocessing repository implements a staged v4 dual-stream pipeline with explicit contracts:

1. `detect`
2. `silhouette`
3. `pack_dual_stream`
4. optional `shuffle`

The architectural strength here is contract-driven preprocessing. Representation metadata is written back into `run.json` under `PreprocessingContract`, and each sample row carries stage statuses and stage-specific metadata in `samples.csv`. Downstream training code can therefore validate not only that files exist, but that representation semantics are compatible.

Implemented preprocessing features include:

- a detector abstraction with both Ultralytics YOLO and edge-based ROI backends
- ROI silhouette extraction with contour generation and convex-hull fallback
- dual-stream NPZ packing with fixed feature schema and validation
- optional backward-compatible arrays for fair comparison with earlier single-stream baselines
- integration and algorithm tests

One particularly good design choice is the treatment of the silhouette crop. The crop is placed on a fixed canvas without geometric rescaling. This preserves apparent object size as a depth cue rather than normalizing it away.

The checked-in successful packed corpora substantiate two stages of the project:

- centre-line dual-stream training corpus `26-04-06_v014-train-shuffled-ds`: `50,063` samples
- centre-line dual-stream validation corpus `26-04-06_v015-validate-shuffled-ds`: `10,020` samples
- full-frame dual-stream training corpus `26-04-11_v018-train-shuffled`: `250,000` samples
- full-frame dual-stream validation corpus `26-04-11_v019-validate-shuffled`: `50,000` samples

Important accuracy note: the checked-in successful production corpora were built with the edge ROI detector backend, not YOLO. YOLO support is implemented and documented, but the current substantiated training outputs rely on the edge detector path.

### 3. Training and Evaluation Infrastructure (`rb-training-v2.0`)

The training repository is considerably more than a single model script. It contains:

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

The dual-stream topology is the most substantial model design in the repository. The shape stream is a CNN over `silhouette_crop`; the geometry stream is an MLP over a 10-element bounding-box feature vector. The v0.2 revision replaced `BatchNorm2d` with `GroupNorm` and removed fusion dropout by default in response to observed instability in v0.1. The latest full-frame run keeps the same `dual_stream_v0_2` topology but moves from the earlier `224 x 224` representation to a `300 x 300` silhouette canvas to match the newer corpus family.

The loader and experiment tooling are also stronger than a minimal prototype:

- manifest authority is enforced
- shard schemas are inspected before training
- `samples.csv` to NPZ row alignment is verified
- overlap warnings are generated
- training can operate sequentially, by shard shuffle, or via an active-shard reservoir mode
- RAM-budgeted LRU caches are implemented for train and validation shards

The latest model run also demonstrates that the resume workflow is not merely theoretical. `260412-1759_ds-2d-cnn/run_0002` explicitly resumes from `run_0001` and completes successfully, which is a good operational signal for long-running experiments.

## Separation of Concerns and Supporting Documentation

A notable repository-level strength is the deliberate split between notebook control surfaces and implementation code.

Examples:

- `rb-training-v2.0/notebooks/02_train_ds_2d_cnn_v0.7.ipynb` acts as a control surface and delegates logic to `src`
- `synthetic-data-processing-v2.0/rb_ui_v4/*.ipynb` act as thin launch surfaces over `rb_pipeline_v4`

This separation is reinforced by written technical documents in:

- `synthetic-data-processing-v2.0/documents`
- `rb-training-v2.0/documents`

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
| `260410-1817_ds-2d-cnn/run_0001` | dual-stream v0.1 on centre-line `+-0.25 m` regime | `50,063 / 10,020` | `0.06880 m` | `0.08158 m` | `0.76307` | Validated the architecture direction, but training was unstable. |
| `260411-1104_ds-2d-cnn/run_0001` | dual-stream v0.2 on centre-line `+-0.25 m` regime | `50,063 / 10,020` | `0.01512 m` | `0.02137 m` | `0.99950` | Best result on the narrower benchmark. |
| `260412-1759_ds-2d-cnn/run_0002` | dual-stream v0.2 on footprint-stratified full-frame regime | `250,000 / 50,000` | `0.02785 m` | `0.09972 m` | `0.97892` | First completed full-frame benchmark run; typical error remains good, but the tail is much heavier. |

### Assessment of the New Full-Frame Run

The completed artifact for the new model is `260412-1759_ds-2d-cnn/run_0002`, created on `2026-04-13` as a resumed continuation of `run_0001`.

Substantiated metrics from `metrics.json` and `history.csv` are:

- best epoch: `21`
- early stop after epoch: `29`
- validation loss: `0.004400`
- validation MAE: `0.027850 m`
- validation RMSE: `0.099725 m`
- validation accuracy within `0.10 m`: `0.97892`
- validation accuracy within `0.25 m`: `0.98586`
- validation accuracy within `0.50 m`: `0.99234`

This is a respectable first result on a much broader benchmark. The median absolute error remains low at approximately `0.0141 m`, which indicates that the model is usually accurate. However, the upper tail is materially worse than in the centre-line benchmark:

- `95th` percentile absolute error: approximately `0.0556 m`
- `99th` percentile absolute error: approximately `0.3980 m`
- `99.9th` percentile absolute error: approximately `1.4704 m`
- samples with absolute error greater than `0.50 m`: `383 / 50,000`
- samples with absolute error greater than `1.00 m`: `104 / 50,000`

Distance-band analysis also shows that the run is not uniformly weak. The near and mid-range bins remain reasonably accurate:

- `1.5-2.5 m`: MAE `0.0186 m`
- `2.5-3.5 m`: MAE `0.0157 m`
- `3.5-4.5 m`: MAE `0.0195 m`
- `4.5-5.5 m`: MAE `0.0333 m`
- `5.5-7.5 m`: MAE `0.0847 m`

The main weakness is therefore not general collapse. It is a concentrated robustness problem in the harder parts of the new distribution.

### What the New Run Reveals

The most important technical finding is that the broader dataset has exposed a specific failure mode that the earlier benchmark did not exercise.

Coarse error analysis against validation manifests shows:

- middle third of image: MAE `0.0138 m`, with `0` errors greater than `0.50 m`
- left third of image: MAE `0.0255 m`, with `49` errors greater than `0.50 m`
- right third of image: MAE `0.0408 m`, with `334` errors greater than `0.50 m`

An even clearer view comes from normalized lateral position:

- for the most extreme lateral placements (`|cx_norm - 0.5|` between `0.3` and `0.5`), MAE rises to `0.0456 m`
- all `383` errors greater than `0.50 m` occur in that extreme-lateral band
- all `104` errors greater than `1.00 m` also occur in that extreme-lateral band

The largest outliers are systematic overestimates, often predicting approximately `6.3-6.8 m` for true distances around `3.4-4.8 m`. This suggests, as an inference from the artifacts rather than a proven theorem, a specific edge-of-frame failure mode rather than smooth degradation across the whole space.

This is valuable from a research perspective. The new pipeline did not merely produce a larger dataset. It produced a better falsification benchmark and surfaced a real weakness that the centre-line regime had largely hidden.

### Relationship to the Earlier Centre-Line Result

The new run should not be read as a simple regression from `260411-1104_ds-2d-cnn`. The benchmark changed materially.

The earlier centre-line validation set is much narrower:

- all `10,020` validation samples lie in the middle third of the image
- only `2` samples exceed `0.50 m` absolute error
- no sample exceeds `1.00 m` absolute error

That earlier result therefore remains the best measured result on the narrower benchmark. The new full-frame result is more demanding and more informative. In fact, the middle-third performance of the new model (`0.0138 m` MAE) is close to the earlier centre-line benchmark (`0.0151 m` MAE). Most of the degradation comes from newly introduced edge placements rather than from a wholesale loss of model competence.

## What Appears Technically Distinctive

The following aspects stand out as more than routine implementation:

- end-to-end ownership across synthetic generation, preprocessing, training, evaluation, and experiment operations
- camera-footprint stratified placement rather than naive world-space randomization
- projection-based sample validity checks before capture
- preservation of apparent size cues by centering ROI silhouettes on a fixed canvas without rescaling
- explicit preprocessing contracts carried from `run.json` into training artifacts
- fair-comparison support between older and newer representations
- topology versioning with compatibility signatures and guarded resume support
- use of the synthetic data generator not only to supply data, but to redesign the benchmark and expose hidden lateral-edge failure modes
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
- The task is deliberately constrained: one known object family, one fixed calibrated camera, one movement plane, and limited pose variation. High accuracy in this context should not be presented as broad scene understanding.
- Training and validation are separate corpora and the loaders check for overlap, but both are generated by the same synthetic pipeline family. Domain gap remains the major open question.
- The current production corpora use the edge ROI detector backend. YOLO support exists, but the checked-in successful training results do not yet substantiate a YOLO-backed production path.
- Some narrative artifacts lag the implementation. In particular, generated model cards still describe the input as full-frame grayscale imagery, whereas the actual v4 representation is `silhouette_crop + bbox_features`.
- The raw Unity archive for the latest `v018` corpus is only partially mirrored in the repository snapshot, even though the downstream packed corpora and training artifacts are complete.
- The new full-frame benchmark exposes a clear unresolved weakness at extreme lateral placements, especially near the right edge of the image.

## Current Status

At the current repository snapshot:

- the end-to-end path from Unity generation to packed corpora to trained model is functioning
- the dual-stream v0.2 architecture is implemented and operational on both the earlier centre-line benchmark and the newer full-frame benchmark
- the best narrow-benchmark result remains `260411-1104_ds-2d-cnn/run_0001`
- the first completed full-frame benchmark result is `260412-1759_ds-2d-cnn/run_0002`
- the principal next technical problem is improving robustness at extreme lateral placements rather than improving median central-case accuracy

Project status also indicates that a subsequent corpus is currently moving through the preprocessing pipeline, so this should be understood as an active research system rather than a closed benchmark report.

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

This is a technically serious repository with unusually strong end-to-end ownership for a single-developer research project. The most compelling achievement is not any single metric in isolation, but the construction of a complete experimental system: controllable synthetic data generation, contract-aware preprocessing, disciplined experiment operations, and a model iteration loop that can both improve results and reveal when an easier benchmark has been outgrown.

The strongest honest presentation to employers is:

- a rigorous bounded ML research system
- a strong example of sole end-to-end ML engineering ownership
- evidence of careful experimental design, instrumentation, and technical self-critique

It should not be presented as evidence of deployment readiness or broad visual generalization. The strongest substantiated claim is that the repository demonstrates the ability to design, build, instrument, and iteratively stress-test a complete perception pipeline under controlled conditions, while being candid about where the current evidence ends.
