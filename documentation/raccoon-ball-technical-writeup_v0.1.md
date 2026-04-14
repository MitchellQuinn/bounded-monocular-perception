# Raccoon Ball Repository Technical Writeup

## Overview

This repository presents a coherent end-to-end machine learning workflow for monocular distance estimation in a tightly bounded synthetic environment. The implemented stack spans:

1. synthetic dataset generation in Unity (`rb_synthetic-data_3`)
2. preprocessing and representation packing (`synthetic-data-processing-v2.0`)
3. model training, evaluation, and experiment operations (`rb-training-v2.0`)

The technical objective is not generic object detection or unconstrained scene understanding. The task is a controlled regression problem: estimate vehicle distance, and optionally 3D position, for a known object class observed by a fixed calibrated camera under a constrained movement plane. The work is therefore best understood as a bounded perception system designed for falsifiable geometry-aware experiments rather than as a general-purpose vision product.

The repository evidence is consistent with sole development of the full ML pipeline, including data generation, data contracts, preprocessing, model development, experiment tooling, and supporting technical documentation.

## Task Definition

At the training layer, the core prediction target is `distance_m`, with support for an alternative `position_3d` mode. Two model families are implemented:

- a naive full-frame CNN baseline
- a dual-stream CNN/MLP regressor combining silhouette imagery with explicit bounding-box geometry features

The system is designed to preserve traceability from generated image through preprocessing contract to model artifact. This is visible in the use of `run.json`, `samples.csv`, packed NPZ shards, training manifests, run-level config files, and model cards.

## Repository Architecture

### 1. Unity Synthetic Dataset Generation (`rb_synthetic-data_3`)

The Unity generator is more structured than a simple scripted camera sweep.

Key implemented elements include:

- manual camera capture through `RenderTexture` and `Texture2D` in `CaptureService.cs`
- manifest and run metadata writing via `ManifestWriter`, `ManifestRowMapper`, and `RunMetadataWriter`
- deterministic sample naming and path construction in `FileNamingStrategy.cs`
- explicit Euclidean target generation in `DistanceCalculator.cs`
- camera-footprint-aware placement planning in `StratifiedPlacementPlanner.cs`
- projection-based feasibility validation in `VehicleProjectionValidator.cs`
- batch-oriented run orchestration, attempt budgeting, and quota redistribution in `RunControllerBehaviour.cs`

The technically distinctive part is the placement strategy. Rather than sweeping depth uniformly or sampling naively in world space, the generator:

- projects the camera footprint onto the movement plane
- partitions that footprint into depth bands and lateral bins
- probes cell feasibility under image-space constraints
- allocates samples across feasible cells
- redistributes shortfall when a cell exhausts its attempt budget

This is a non-trivial design choice. It directly addresses a common synthetic-data failure mode in which nominal world-space coverage produces poor image-space coverage or invalid edge-clipped samples.

The checked-in Unity logs substantiate this design. For example:

- `26-04-11_v0.1_smoketest` planned 256 samples across 120 cells, dropped infeasible cells to 113/120, and completed successfully.
- `26-04-11_v018-train` planned 250,000 samples across 120 cells, reported 107/120 feasible cells, and the snapshot currently contains 102 images and 102 manifest rows, indicating an in-progress or incomplete next corpus rather than a finished run.

### 2. Preprocessing Pipeline (`synthetic-data-processing-v2.0`)

The preprocessing repository implements a staged v4 dual-stream pipeline with explicit contracts:

1. `detect`
2. `silhouette`
3. `pack_dual_stream`
4. optional `shuffle`

The important architectural feature is contract-driven preprocessing. The pipeline writes representation metadata back into `run.json` under `PreprocessingContract`, and each sample row carries stage statuses and stage-specific metadata in `samples.csv`. This allows downstream training code to verify not only that files exist, but that representation semantics are consistent.

Implemented preprocessing features include:

- a detector abstraction with both Ultralytics YOLO and edge-based ROI backends
- ROI silhouette extraction with contour generation and convex-hull fallback
- dual-stream NPZ packing with fixed feature schema and validation
- optional backward-compatible arrays for fair comparison with earlier single-stream baselines
- integration and algorithm tests

One particularly strong design choice is the treatment of the silhouette crop. The crop is placed on a fixed canvas without geometric rescaling. This preserves apparent object size as a depth cue rather than normalizing it away. That is a technically meaningful choice for monocular distance estimation and is explicitly documented in the preprocessing specification and in the dual-stream model definition.

Another noteworthy feature is the attempt to keep v1 and v2 comparisons fair. The pack stage can optionally emit compatibility arrays so that older full-frame baselines can be run against the same upstream corpus family.

The currently checked-in successful packed corpora are:

- `26-04-06_v014-train-shuffled-ds`
- `26-04-06_v015-validate-shuffled-ds`

The pack logs show:

- training corpus: 50,063 successful rows, 0 failed rows, 37 skipped rows
- validation corpus: 10,020 successful rows, 0 failed rows, 0 skipped rows

The 37 skipped training rows were caused by upstream stage subset selection, not pack-stage failure. The resulting production corpus used for training is therefore 50,063 train and 10,020 validation samples.

Important accuracy note: the checked-in successful v4 corpora were produced with the edge ROI detector backend, not YOLO. YOLO support is implemented in code and documented in the specification, but the current substantiated production outputs rely on the edge detector path.

### 3. Training and Evaluation Infrastructure (`rb-training-v2.0`)

The training repository is not just a model script. It contains:

- topology registration and versioned topology definitions
- manifest-aware dataset loading and validation
- shard streaming and RAM caching
- overlap checks between train and validation corpora
- run directory creation and artifact writing
- resume-state support with topology and dataset compatibility checks
- evaluation outputs including prediction CSVs and plots
- notebook-driven operational control with helper modules for system logic

Implemented model families include:

- `distance_regressor_2d_cnn`
- `distance_regressor_dual_stream`
- an additional registered `distance_regressor_global_pool_cnn` extension point

The dual-stream topology is particularly well structured. The shape stream is a CNN over `silhouette_crop`; the geometry stream is an MLP over a 10-element bounding-box feature vector. The v0.2 revision removed `BatchNorm2d`, replaced it with `GroupNorm`, and set fusion dropout to zero by default in response to observed validation instability in v0.1.

The training loader is also stronger than a minimal prototype:

- manifest authority is enforced
- shard schemas are inspected before training
- `samples.csv` to NPZ row alignment is verified
- training and validation overlap warnings are generated
- streaming can operate sequentially, by shard shuffling, or via an active-shard reservoir mode
- RAM-budgeted LRU shard caches are implemented for both training and validation

This is competent and disciplined ML systems engineering rather than ad hoc experimentation.

## Separation of Concerns

A notable repository-level strength is the deliberate split between notebook control surfaces and implementation code.

Examples:

- `rb-training-v2.0/notebooks/02_train_ds_2d_cnn_v0.7.ipynb` explicitly declares itself as a UI-only tmux control panel and imports helper logic from `src/v0.2/training_control_panel.py` and `src/resume/control_panel.py`.
- `synthetic-data-processing-v2.0/rb_ui_v4/*.ipynb` act as thin launch surfaces over `rb_pipeline_v4`.

This separation is not accidental. It is reinforced by written internal standards such as:

- `synthetic-data-processing-v2.0/documents/Synthetic Data Processing v2.0 Specification v0.1.md`
- `rb-training-v2.0/documents/Distance Regressor Dual Stream Definition v0.2.md`
- `rb-training-v2.0/documents/Training Session Control Panel Implementation Playbook v0.1.md`
- `rb-training-v2.0/documents/Adding New Topology v2.0.md`
- `rb-training-v2.0/documents/Raccoon Ball Training Repo Notebook Standard v0.1.1`

This documentation is itself relevant to employers. It demonstrates the use of written specifications, playbooks, and templates to control architectural drift in a single-developer research codebase without introducing heavyweight external infrastructure.

## What Has Been Implemented

### Implemented End-to-End Workflow

- synthetic sample generation in Unity with run metadata and per-sample manifests
- preprocessing stage contracts and row-level stage status tracking
- detector abstraction supporting both YOLO and edge ROI extraction
- silhouette generation with fallback recovery logic
- dual-stream shard packing with optional compatibility outputs
- strict training-side manifest and shard validation
- topology registry and versioned topology definitions
- full-frame CNN baseline training
- dual-stream CNN/MLP training
- resume-state persistence and guarded resume workflows
- evaluation artifacts: metrics, history, sample predictions, scatter plots, residual plots, run manifests, and model cards

### Testing and Verification

The Python test suites currently pass when run from the individual project roots:

- `rb-training-v2.0`: `14 passed`
- `synthetic-data-processing-v2.0`: `6 passed`

Practical caveat: invoking those suites from the monorepo root failed due project-local import-path assumptions (`src` and `rb_pipeline_v4` were not on `PYTHONPATH`). This does not invalidate the test coverage, but it does indicate that the repositories are designed to be run from their own roots rather than from the workspace top level.

## Results and Assessment

### Training Results

The checked-in training artifacts show a clear progression across experiment generations.

| Run | Date | Model | Validation MAE | Validation RMSE | Accuracy within 0.10 m | Assessment |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `260331-2016_2d-cnn/run_0001` | 2026-03-31 | full-frame CNN baseline | 0.0818 | 0.1045 | 0.6724 | Promising early baseline on an older corpus, but run artifact set is incomplete. |
| `260402-1233_2d-cnn/run_0001` | 2026-04-02 | full-frame CNN baseline | 0.8010 at last logged epoch | 0.8266 | 0.0000 | Poor generalization on a smaller corpus; likely dataset/regime mismatch. |
| `260407-1606_2d-cnn/run_0001` | 2026-04-07 | full-frame CNN baseline | 0.1869 at last logged epoch | 0.1990 | 0.1128 | Improved, but still unstable and materially weaker than later runs. |
| `260407-1756_2d-cnn/run_0002` | 2026-04-08 | full-frame CNN baseline | 0.0534 | 0.0693 | 0.8738 | Best substantiated full-frame baseline. |
| `260410-1817_ds-2d-cnn/run_0001` | 2026-04-10 | dual-stream v0.1 | 0.0688 | 0.0816 | 0.7631 | Validates the architecture direction, but training is unstable. |
| `260411-1104_ds-2d-cnn/run_0001` | 2026-04-11 | dual-stream v0.2 | 0.0151 | 0.0214 | 0.9995 | Best current result; large improvement over both baseline and dual-stream v0.1. |

### Interpretation of the Best Current Result

The strongest checked-in run is `260411-1104_ds-2d-cnn/run_0001`:

- validation MAE: `0.01512 m`
- validation RMSE: `0.02137 m`
- validation accuracy within `0.10 m`: `0.99950`
- validation sample count: `10,020`

This is materially better than the best full-frame CNN baseline (`0.05338 m` MAE, `0.06929 m` RMSE, `0.87375` within `0.10 m`) and materially more stable than dual-stream v0.1.

The sample prediction artifact also suggests that v0.2 is not merely winning at one distance band. Validation MAE is approximately flat across the observed range:

- `2.4-3.5 m`: `0.0146 m`
- `3.5-4.5 m`: `0.0174 m`
- `4.5-5.5 m`: `0.0142 m`
- `5.5-7.0 m`: `0.0144 m`

The `95th` percentile absolute error is approximately `0.0377 m`, and the `99th` percentile absolute error is approximately `0.0536 m`.

### What Appears Technically Distinctive

The following aspects stand out as more than routine implementation:

- end-to-end ownership across synthetic generation, preprocessing, training, evaluation, and experiment operations
- camera-footprint stratified placement rather than naive world-space randomization
- projection-based sample validity checks before capture
- preservation of apparent size cues by centering ROI silhouettes on a fixed canvas without rescaling
- explicit preprocessing contracts carried from `run.json` into training artifacts
- fair-comparison support between older and newer representations
- topology versioning with compatibility signatures and guarded resume support
- written technical playbooks guiding notebook/control-panel architecture and topology extension

### What Is Strong Competent Implementation of Known Ideas

The following work is less novel but clearly well executed:

- modular PyTorch training loop with Huber loss, early stopping, scheduler support, and structured artifact output
- evaluation output generation with metrics, prediction dumps, and plots
- unit and integration test coverage around topology registration, resume features, and preprocessing
- notebook-driven operational tooling over helper modules
- dataset manifest validation and split leakage checks
- tmux-based long-running training control in Jupyter workflows

### Weaknesses, Limitations, and Caveats

This project is strong, but its claims should remain bounded.

- The results are entirely synthetic. There is no checked-in evidence of transfer to real images or of mixed-domain training.
- The task itself is deliberately constrained: one known object family, one fixed calibrated camera, one movement plane, and limited pose variation. Extremely high validation accuracy therefore reflects a highly regular problem setting, not general scene understanding.
- Training and validation are separate corpora and the loader checks found no overlap, but both come from the same synthetic pipeline family. Domain gap remains the major open question.
- The current production dual-stream corpora use the edge ROI detector backend. YOLO integration is implemented, but the checked-in successful corpora do not yet demonstrate YOLO-backed production training.
- Some generated narrative artifacts lag the code. In particular, the dual-stream model cards still describe the input as full-frame grayscale imagery, whereas the actual v4 preprocessing contract and topology use `silhouette_crop` plus `bbox_features`.
- Earlier runs include aborted or incomplete experiments. This is normal for active research, but it should be presented as iteration history rather than as polished production.

## Current Status

At the current repository snapshot:

- the v4 preprocessing pipeline is implemented and has produced usable dual-stream train/validation corpora
- the full-frame baseline is implemented and has at least one strong completed run
- the dual-stream v0.2 model is implemented and currently delivers the best checked-in result
- the next Unity corpus appears active or incomplete rather than finished: `26-04-11_v018-train` is configured for `250,000` samples, but only `102` images and `102` manifest rows are currently present, and the run log does not contain a completion message

This is therefore an active research system with a functioning end-to-end path and a clear next step: push a larger next-generation corpus through the preprocessing and training stack, then test whether the v0.2 result survives a more ambitious dataset.

## Concrete Skills and Tools Demonstrated

### Machine Learning and Data

- PyTorch model development
- distance regression and optional 3D position regression
- multi-input model design
- topology/version management for experiments
- dataset schema validation and artifact lineage
- streaming NPZ data loading and memory-aware caching
- evaluation metric design and prediction analysis

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
- ipywidgets
- tmux

## Overall Assessment

This is a technically serious repository with unusually strong end-to-end ownership for a single-developer research project. The most compelling achievement is not any single metric in isolation, but the construction of a complete experimental system: controllable synthetic data generation, contract-aware preprocessing, disciplined experiment operations, and a model iteration loop that demonstrably improved from an unstable dual-stream v0.1 to a strong dual-stream v0.2 result.

The work should be presented to employers as:

- a rigorous bounded ML research system
- a strong example of end-to-end ML engineering ownership
- evidence of careful experimental thinking and software discipline

It should not be presented as evidence of real-world deployment readiness or broad visual generalization. The strongest honest claim is that the repository demonstrates the ability to design, build, instrument, and iteratively improve a complete perception pipeline under controlled conditions, with clear awareness of where the current evidence ends.
