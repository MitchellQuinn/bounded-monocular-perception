# Raccoon Ball Repository Technical Writeup v0.4

## Overview

This repository is a coherent end-to-end machine learning research system for bounded monocular perception in a tightly controlled synthetic environment. The implemented stack spans:

1. synthetic dataset generation in Unity (`rb_synthetic-data_3`)
2. preprocessing and representation packing (`synthetic-data-processing-v2.0`)
3. model training, evaluation, and experiment operations (`rb-training-v2.0`)

The technical objective is still not generic object detection or unconstrained scene understanding. The task remains a controlled regression problem over a known vehicle instance in a constrained scene, observed by a fixed calibrated camera over a constrained movement plane. What has changed since v0.3 is that the repository now substantiates not only distance regression, but strong joint distance-and-orientation regression on the hardest checked-in synthetic benchmark.

The repository evidence remains consistent with sole development of the full machine learning pipeline, including data generation, preprocessing contracts, model development, experiment tooling, and supporting technical documentation.

## Task Definition

At the training layer, the original core target remains `distance_m`, with support for `position_3d`. The latest checked-in work extends that target space to joint distance plus yaw via `yaw_sin` and `yaw_cos`.

Three model families are now substantively represented:

- a naive full-frame CNN baseline
- a dual-stream CNN/MLP regressor combining crop imagery with explicit bounding-box geometry features for scalar distance
- a newer dual-stream multitask regressor that predicts both distance and yaw from the same bounded dual-stream input

The system preserves traceability from generated image through preprocessing contract to training artifact. This remains visible in the consistent use of `run.json`, `samples.csv`, packed NPZ shards, split manifests, run-level config files, prediction dumps, and model cards.

## Benchmark Regime Evolution

The most important recent story is still benchmark expansion, but there is now a second story on top of that: representational and target-contract evolution on the hardest benchmark rather than retreat from it.

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

The next benchmark family kept that broader spatial regime but made pose substantially harder:

- `26-04-11_v020-train` and `26-04-11_v021-validate` retain the footprint-stratified full-frame setup
- the deliberate change is yaw range: `RotYMinDeg = -180°`, `RotYMaxDeg = 180°`
- this produces full `360°` yaw variation for the Defender rather than a narrow local perturbation

The resulting validation distribution remains full-frame. Manifest analysis of the currently checked-in `26-04-11_v022-validate-shuffled` preprocessing corpus still shows the same underlying spatial spread:

- left third: `16,328` samples
- middle third: `17,379` samples
- right third: `16,293` samples

The latest corpus family should therefore not be read as a softer benchmark. Instead, it is the same hardest generator regime under a richer preprocessing and target contract:

- training now uses `26-04-11_v022-train-shuffled`
- validation now uses `26-04-11_v022-validate-shuffled`
- the checked-in `run.json` lineage still points back to the underlying synthetic captures `26-04-11_v020-train` and `26-04-11_v021-validate`
- the preprocessing contract was upgraded from `rb-preprocess-v4-dual-stream` to `rb-preprocess-v4-dual-stream-orientation-v1`
- the image stream remains an unscaled `300 x 300` crop canvas, but its declared content is now `grayscale_vehicle_detail_inside_silhouette`
- the target arrays now include `y_yaw_deg`, `y_yaw_sin`, and `y_yaw_cos`

The strongest honest interpretation is therefore:

- the benchmark difficulty from full-frame placement and full `360°` yaw remains
- the newest work attacks that harder regime with a better representation and a multitask objective, rather than by narrowing the benchmark again

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

The technically distinctive part is still the placement strategy. Rather than sweeping depth uniformly or sampling naively in world space, the generator:

- projects the camera footprint onto the movement plane
- partitions that footprint into depth bands and lateral bins
- probes cell feasibility under image-space constraints
- allocates samples across feasible cells
- redistributes shortfall when a cell exhausts its attempt budget

This directly addresses a common synthetic-data failure mode in which nominal world-space coverage does not translate into useful image-space coverage.

The latest yaw work also shows that the generator metadata was already rich enough to support new targets without a generator redesign. The manifests carry the final vehicle rotation explicitly, which made it possible to promote yaw from incidental metadata into a first-class training target.

### 2. Preprocessing Pipeline (`synthetic-data-processing-v2.0`)

The preprocessing repository still implements a staged v4 dual-stream pipeline with explicit contracts:

1. `detect`
2. `silhouette`
3. `pack_dual_stream`
4. optional `shuffle`

The architectural strength remains contract-driven preprocessing. Representation metadata is written back into `run.json` under `PreprocessingContract`, and each sample row carries stage statuses and stage-specific metadata in `samples.csv`. Downstream training can therefore validate not only that files exist, but that representation semantics are compatible.

Implemented preprocessing features now include:

- a detector abstraction with both Ultralytics YOLO and edge-based ROI backends
- ROI silhouette extraction with contour generation and convex-hull fallback
- dual-stream NPZ packing for both distance-only and orientation-aware contracts
- optional metadata arrays including yaw degree and yaw sin/cos targets in the latest contract
- optional backward-compatible arrays for fair comparison with older baselines
- integration and algorithm tests

One particularly good design choice persists across versions: the crop is placed on a fixed canvas without geometric rescaling. This preserves apparent object size as a depth cue rather than normalizing it away.

The newer orientation-aware contract adds an important refinement on top of that. The image array key remains `silhouette_crop`, but the semantic content is no longer just a filled binary silhouette. The v022 contract declares:

- `ImageContent = grayscale_vehicle_detail_inside_silhouette`
- `ImagePolarity = dark_vehicle_detail_on_white_background`
- `ImageRepresentationMode = roi_grayscale_inverted_vehicle_on_white`

That is a meaningful change. It keeps the bounded ROI framing and suppresses most background clutter, but preserves interior grayscale detail that the earlier pure-shape representation discarded.

The repository now substantiates the main benchmark stages of the project in four forms:

- directly present centre-line dual-stream training corpus `26-04-06_v014-train-shuffled-ds`: `50,063` samples
- directly present centre-line dual-stream validation corpus `26-04-06_v015-validate-shuffled-ds`: `10,020` samples
- historically substantiated full-frame limited-yaw benchmark over `26-04-11_v018-train-shuffled` / `26-04-11_v019-validate-shuffled`: `250,000 / 50,000`
- directly present latest orientation-aware full-frame full-yaw corpora `26-04-11_v022-train-shuffled` / `26-04-11_v022-validate-shuffled`: `250,000 / 50,000`

Important accuracy note: the checked-in successful production corpora were built with the edge ROI detector backend, not YOLO. YOLO support is implemented and documented, but the current substantiated training outputs still rely on the edge detector path.

**Detector note:** Early experiments with off-the-shelf Ultralytics YOLO did not reliably identify the Defender 90 and produced incorrect labels (for example, `keyboard`). The current checked-in successful results therefore rely on the edge ROI backend rather than a YOLO-backed production path. A custom detector, fine-tuned YOLO model, or alternative live ROI strategy is likely to be required for real-world deployment.

### 3. Training and Evaluation Infrastructure (`rb-training-v2.0`)

The training repository remains substantially more than a single model script. It contains:

- topology registration and versioned topology definitions
- manifest-aware dataset loading and validation
- shard streaming and RAM caching
- overlap checks between train and validation corpora
- run directory creation and artifact writing
- resume-state persistence with topology and dataset compatibility checks
- evaluation outputs including prediction CSVs and plots
- notebook-driven operational control with helper modules for implementation logic
- task-runtime reporting that now handles multitask distance-plus-orientation evaluation

Implemented model families now include:

- `distance_regressor_2d_cnn`
- `distance_regressor_dual_stream`
- `distance_regressor_dual_stream_yaw`
- `distance_regressor_global_pool_cnn` as an additional registered extension point

The distance-only dual-stream topology remains an important part of the repository story. Its shape stream is a CNN over the crop image; its geometry stream is an MLP over a 10-element bounding-box feature vector. The v0.2 revision replaced `BatchNorm2d` with `GroupNorm` and removed fusion dropout by default in response to observed instability in v0.1.

The newest architectural step is the `distance_regressor_dual_stream_yaw` topology with variant `dual_stream_yaw_v0_1`. This is not a completely unrelated model. It is a direct extension of the v0.2 dual-stream architecture:

- the image stream remains single-channel `300 x 300`
- the geometry stream remains a 10-feature MLP
- the fused trunk is shared
- the output is now split into two heads: scalar distance and a 2-value yaw sin/cos head
- the training objective is weighted multitask Huber with `distance=1.0` and `orientation=1.0`

That continuity matters because it makes the latest result a meaningful architectural iteration rather than a benchmark reset with an unrelated model family.

The reporting layer has also improved. The newest run emits a `distance_orientation_multitask` reporting family with:

- `distance_loss`
- `orientation_loss`
- `yaw_mean_error_deg`
- `yaw_median_error_deg`
- `yaw_p95_error_deg`
- `yaw_acc@5deg`
- `yaw_acc@10deg`
- `yaw_acc@15deg`

This is a good example of the repository expanding its evaluation contract when the task expands, rather than continuing to report only the older scalar-distance metrics.

## Separation of Concerns and Supporting Documentation

A notable repository-level strength is still the deliberate split between notebook control surfaces and implementation code.

Examples:

- `rb-training-v2.0/notebooks/02_train_ds_2d_cnn_v0.8.ipynb` and later notebook revisions act as control surfaces and delegate logic to `src`
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

This remains relevant to potential employers because it demonstrates specification-driven development, architectural discipline, and a deliberate attempt to control drift in an active research codebase without relying on heavyweight external process.

## What Has Been Implemented

### Implemented End-to-End Workflow

- synthetic sample generation in Unity with run metadata and per-sample manifests
- preprocessing stage contracts and row-level stage status tracking
- detector abstraction supporting both YOLO and edge ROI extraction
- silhouette generation with fallback recovery logic
- dual-stream shard packing for distance-only and orientation-aware contracts
- strict training-side manifest and shard validation
- topology registry and versioned topology definitions
- naive full-frame CNN baseline training
- dual-stream CNN/MLP distance training
- dual-stream CNN/MLP distance-plus-yaw training
- resume-state persistence and guarded resume workflows
- evaluation artifacts: metrics, history, sample predictions, scatter plots, residual plots, run manifests, and model cards

### Testing and Verification

The repositories still contain targeted tests around preprocessing contracts, topology registration, epoch reporting, runtime reporting, and the new dual-stream-yaw topology. The latest training repository snapshot includes explicit coverage for files such as:

- `test_topology_dual_stream_yaw.py`
- `test_task_runtime_reporting.py`
- `test_epoch_summary.py`
- `test_topology_registry.py`

Important honesty note: I did not re-run the Python suites in this workspace snapshot because `pytest` is not installed in the current shell environment. Earlier repo-local verification had substantiated passing suites, but this revision should not claim a fresh green test run without that dependency present.

The older practical caveat likely still applies as well: these repositories are set up like separate project roots rather than as one fully integrated monorepo Python package, so repo-local execution remains the more natural operating mode.

## Results and Assessment

### Representative Completed Runs

| Run | Model / Regime | Train / Validation Samples | Validation MAE | Validation RMSE | Accuracy within 0.10 m | Assessment |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `260407-1756_2d-cnn/run_0002` | naive full-frame CNN on centre-line `+-0.25 m` regime | `50,063 / 10,020` | `0.05338 m` | `0.06929 m` | `0.87375` | Best substantiated single-stream baseline. |
| `260411-1104_ds-2d-cnn/run_0001` | dual-stream v0.2 on centre-line `+-0.25 m` regime | `50,063 / 10,020` | `0.01512 m` | `0.02137 m` | `0.99950` | Best result on the narrow benchmark. |
| `260412-1759_ds-2d-cnn/run_0002` | dual-stream v0.2 on full-frame, limited-yaw (`12.5°` total yaw range) regime | `250,000 / 50,000` | `0.02785 m` | `0.09972 m` | `0.97892` | First completed full-frame benchmark run. Strong median accuracy, but a pronounced right-edge tail. |
| `260413-1847_ds-2d-cnn/run_0002` | dual-stream v0.2 on full-frame, full-yaw (`360°`) regime | `250,000 / 50,000` | `0.04652 m` | `0.10753 m` | `0.93460` | First completed full-rotation benchmark run. Harder overall, with edge-of-frame failures still present. |
| `260415-1146_ds-2d-cnn/run_0001` | dual-stream yaw v0.1 on the same full-frame, full-yaw regime, using the v022 grayscale-detail contract and joint distance+yaw regression | `250,000 / 50,000` | `0.01007 m` | `0.01297 m` | `0.99996` | Best checked-in result on the hardest benchmark by a very large margin, while also producing strong yaw estimates. |

### Assessment of the Most Recent Run

The completed artifact for the newest model is `260415-1146_ds-2d-cnn/run_0001`, created on `2026-04-15`.

Substantiated metrics from `metrics.json`, `history.csv`, and `train.log` are:

- best epoch: `31`
- configured stop epoch: `32`
- validation loss: `0.001251`
- validation distance loss component: `0.000084`
- validation orientation loss component: `0.001167`
- validation MAE: `0.010068 m`
- validation RMSE: `0.012967 m`
- validation accuracy within `0.10 m`: `0.99996`
- validation accuracy within `0.25 m`: `1.00000`
- validation accuracy within `0.50 m`: `1.00000`
- validation mean angular error: `1.49987°`
- validation median angular error: `0.91449°`
- validation `p95` angular error: `3.34459°`
- validation yaw accuracy within `5°`: `0.97482`
- validation yaw accuracy within `10°`: `0.98698`
- validation yaw accuracy within `15°`: `0.99168`

An important operational detail is that this run also reached the configured epoch budget rather than stopping early on patience. The best validation result occurred at epoch `31`, with epoch `32` remaining very close. That suggests the run was well behaved late in training rather than finding a single lucky early minimum.

### Comparison with the Previous Full-Frame Full-Yaw Run

Relative to `260413-1847_ds-2d-cnn/run_0002`, the newest run is not a marginal improvement. It is a step change.

Headline metric changes are:

- validation MAE improved from `0.04652 m` to `0.01007 m` (`-78.4%`)
- validation RMSE improved from `0.10753 m` to `0.01297 m` (`-87.9%`)
- accuracy within `0.10 m` improved from `0.93460` to `0.99996` (`+6.54` percentage points)
- accuracy within `0.25 m` improved from `0.98956` to `1.00000` (`+1.04` percentage points)
- accuracy within `0.50 m` improved from `0.99354` to `1.00000` (`+0.65` percentage points)

This is too large a change to treat as noise, routine retraining variance, or scheduler luck.

The most defensible interpretation is:

- the newer representation is materially better suited to the hardest synthetic benchmark than the earlier pure filled-silhouette contract
- the auxiliary yaw objective is likely helping the shared trunk learn a more geometry-aware representation
- the repository has now demonstrated not only failure analysis on the hard benchmark, but a successful architectural response to that failure

At the same time, causality should remain bounded. Multiple things changed together:

- preprocessing contract
- image content
- target set
- topology definition
- reporting regime

So this result strongly supports the package of changes, but it does not yet isolate which single change contributed most.

### Error Profile Comparison

Quantile comparison makes the improvement even clearer.

- previous full-frame full-yaw run:
  - median absolute error: `0.0311 m`
  - `95th` percentile: `0.1102 m`
  - `99th` percentile: `0.2624 m`
  - `99.9th` percentile: `1.4556 m`
- latest full-frame full-yaw run:
  - median absolute error: `0.0083 m`
  - `95th` percentile: `0.0254 m`
  - `99th` percentile: `0.0364 m`
  - `99.9th` percentile: `0.0548 m`
  - maximum absolute error: `0.1037 m`

Only `2` of `50,000` validation samples exceed `0.10 m`, and none exceed `0.25 m`.

That is an especially important change in judgement terms: the heavy tail that dominated the previous full-yaw run is no longer the defining characteristic of the benchmark result.

### Distance and Pose Assessment

Distance-band analysis for the latest run shows that the model behaves sensibly across the full depth range, with only mild degradation at longer distance:

- `1.5-2.5 m`: MAE `0.0083 m`
- `2.5-3.5 m`: MAE `0.0101 m`
- `3.5-4.5 m`: MAE `0.0094 m`
- `4.5-5.5 m`: MAE `0.0105 m`
- `5.5-7.5 m`: MAE `0.0127 m`

The corresponding mean angular errors by distance band are:

- `1.5-2.5 m`: `1.58°`
- `2.5-3.5 m`: `1.17°`
- `3.5-4.5 m`: `1.08°`
- `4.5-5.5 m`: `1.20°`
- `5.5-7.5 m`: `2.82°`

So the farthest band is still harder, but not in a way that currently threatens distance usefulness on the synthetic benchmark.

Yaw quadrants are also broadly similar for distance:

- `-180° to -90°`: distance MAE `0.0100 m`, mean angular error `1.47°`
- `-90° to 0°`: distance MAE `0.0101 m`, mean angular error `1.59°`
- `0° to 90°`: distance MAE `0.0102 m`, mean angular error `1.45°`
- `90° to 180°`: distance MAE `0.0101 m`, mean angular error `1.48°`

This is a good sign. The newest model does not look like a benchmark-specific trick that works only for one narrow yaw slice.

### Pathology Analysis: Does the Right-Hand-of-Frame Issue Still Exist?

Not as a meaningful distance pathology.

The earlier full-frame runs had a clear edge-of-frame weakness, and in the limited-yaw case it was strongly right-sided. That specific failure pattern is no longer substantiated in the latest distance output:

- left third of image: MAE `0.0098 m`, `0` errors greater than `0.10 m`
- middle third of image: MAE `0.0096 m`, `1` error greater than `0.10 m`
- right third of image: MAE `0.0108 m`, `1` error greater than `0.10 m`
- no image third contains any error greater than `0.25 m`

That is a qualitatively different result from the previous full-yaw run. The old distance pathology has effectively collapsed.

However, a milder lateral-position effect does remain in orientation:

- left third mean angular error: `1.68°`, with `211` samples above `15°`
- middle third mean angular error: `1.13°`, with `11` samples above `15°`
- right third mean angular error: `1.72°`, with `194` samples above `15°`

The center-versus-edge comparison says the same thing:

- centre band (`|cx_norm - 0.5| < 0.1`): mean angular error `1.15°`
- extreme-lateral band (`|cx_norm - 0.5|` between `0.3` and `0.5`): mean angular error `2.23°`

So the honest current judgement is:

- the earlier edge-of-frame distance failure is no longer the dominant problem
- lateral placement still affects pose estimation more than central placements
- the residual edge sensitivity is now mostly a thin orientation tail rather than a catastrophic distance tail

This is also visible in the rare angular outliers. The orientation distribution is strong overall, but it still has a long and very thin tail:

- angular median: `0.91°`
- angular `95th` percentile: `3.34°`
- angular `99th` percentile: `12.86°`
- angular `99.9th` percentile: `55.23°`
- maximum angular error: `177.86°`

Those large angular failures are uncommon and often coexist with small distance error. That is consistent with a shared representation that is now very robust for distance, while the yaw head still occasionally makes a near-flip mistake on visually awkward cases.

Most of the largest angular outliers are also still lateral-edge cases rather than centre-frame cases, which reinforces the view that the remaining synthetic weakness is now chiefly a pose-at-the-edges problem rather than a distance-at-the-edges problem.

## What Appears Technically Distinctive

The following aspects stand out as more than routine implementation:

- end-to-end ownership across synthetic generation, preprocessing, training, evaluation, and experiment operations
- camera-footprint stratified placement rather than naive world-space randomization
- projection-based sample validity checks before capture
- preservation of apparent size cues by centering crops on a fixed canvas without rescaling
- evolution from a distance-only preprocessing contract to an orientation-aware contract without breaking dataset lineage
- promotion of yaw from stored metadata into a first-class multitask target with reporting support
- deliberate benchmark design that first broadened spatial coverage, then broadened pose coverage, and then improved the representation rather than narrowing the benchmark again
- written technical playbooks governing notebook/control-panel architecture and topology extension

## What Is Strong Competent Implementation of Known Ideas

The following work is less novel, but clearly well executed:

- modular PyTorch training loops with Huber loss, scheduler support, structured artifact output, and multitask reporting
- evaluation output generation with metrics, prediction dumps, and plots
- dataset manifest validation and split leakage checks
- streaming NPZ loaders with memory-aware shard caching
- notebook-driven operational tooling over Python helper modules
- tmux-based long-running training control
- unit-style and integration-style tests around preprocessing and training infrastructure

## Weaknesses, Limitations, and Caveats

This project is strong, but its claims should remain bounded.

- The results are still entirely synthetic. There is no checked-in evidence of transfer to real imagery.
- The task is still deliberately constrained: one known object family, one fixed calibrated camera, one movement plane, and limited environment diversity. High accuracy in this context should not be presented as broad scene understanding.
- Training and validation are separate corpora and the loaders check for overlap, but both are generated by the same synthetic pipeline family. Domain gap remains the major open question.
- The current production corpora still rely on the edge ROI detector backend. YOLO support exists, but the checked-in successful training results do not yet substantiate a YOLO-backed production path.
- The newest performance gain comes from a bundle of changes rather than a clean ablation. The repository now has a much stronger result, but not yet a single-factor causal proof.
- The array key name `silhouette_crop` now spans materially different semantic content across preprocessing contracts. The contracts are explicit enough to make this manageable, but downstream consumers must read the contract rather than infer semantics from the legacy key name alone.
- Narrative artifact quality has improved, but it still varies by run generation. Older model cards can lag the real representation; the newest yaw run is more accurate in its model card than some earlier runs.
- The remaining visible synthetic weakness is no longer catastrophic distance failure, but a thin orientation tail that is still somewhat edge-sensitive.

## Current Status

At the current repository snapshot:

- the end-to-end path from Unity generation to packed corpora to trained model is functioning
- the dual-stream distance-only architecture remains operational on narrow, full-frame, and full-rotation benchmarks
- the newer dual-stream yaw architecture is implemented and operational on the hardest checked-in full-frame full-yaw benchmark
- the best narrow-benchmark result remains `260411-1104_ds-2d-cnn/run_0001`
- the best checked-in result on the hardest full-frame full-yaw benchmark is now `260415-1146_ds-2d-cnn/run_0001`
- the repository now substantiates strong joint distance-and-yaw regression rather than only scalar distance on that benchmark
- the principal next technical questions are now less about basic synthetic distance robustness and more about ablation, residual orientation-tail analysis, and domain transfer

This remains an active research system rather than a closed benchmark report.

## Concrete Skills and Tools Demonstrated

### Machine Learning and Data

- PyTorch model development
- scalar distance regression and optional 3D position regression
- multitask distance-plus-orientation regression
- circular regression via yaw sin/cos targets
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
- contract-backed transition from pure silhouette shape to masked grayscale vehicle detail

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

This repository still provides evidence of end-to-end ownership of a bounded applied machine-learning and computer-vision system. The latest update strengthens that case materially.

In v0.3, the strongest positive judgement was about experimental honesty: the benchmark had been widened until a real weakness became visible. In v0.4, the strongest positive judgement is slightly different. The repository now shows the next step in that loop as well:

- expose a real weakness on the harder benchmark
- change the representation and objective rather than quietly narrowing the task again
- re-run the hard benchmark
- materially improve the result while keeping the claim bounded

That is a persuasive demonstration of technical judgement. It is not just “good metrics”; it is evidence of a functioning research loop with instrumentation, falsification, revision, and retest.

The current evidence therefore supports a stronger but still bounded claim than before: this repository demonstrates the ability to design, build, instrument, and iteratively improve a complete monocular perception pipeline under controlled conditions, including a successful response to a previously exposed benchmark weakness.

It still should not be read as evidence of broad real-world visual generalisation or deployment readiness. The domain-gap question remains open. But as a portfolio artifact, it now does a better job of showing not only system construction, but also the ability to diagnose failure modes and meaningfully improve them without losing experimental discipline.
