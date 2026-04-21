# Raccoon Ball Repository Technical Writeup v0.5

## 1. Overview

Raccoon Ball is now best understood as a bounded monocular perception research-engineering repository rather than only a model-training repository.

The current repo state substantiates an end-to-end stack spanning:

1. synthetic full-frame image generation in Unity
2. contract-driven preprocessing and representation packing
3. model training and evaluation for distance and joint distance-plus-yaw regression
4. a separate ROI-FCN crop-centre localisation subsystem
5. raw-image inference pipelines that compose those trained components into runnable runtime paths

The strongest honest framing remains deliberately bounded. The repository is not evidence of unconstrained scene understanding, generic object detection, or real-world deployment readiness. It is evidence of end-to-end ownership of a controlled synthetic perception problem around one known vehicle family, a fixed calibrated camera, a constrained movement plane, explicit data contracts, and a progressively more complete runtime surface.

## 2. What the Repository Now Does

At the current snapshot, the repository substantiates the following capabilities:

- generation of synthetic Defender full-frame imagery with run metadata and per-sample manifests
- preprocessing of those captures into dual-stream training corpora for distance and yaw regression
- preprocessing of raw full-frame corpora into ROI-FCN locator corpora with explicit geometry metadata
- training of baseline CNN, dual-stream distance, and dual-stream distance-plus-yaw models
- training of a tiny fully convolutional crop-centre localiser
- evaluation with metrics, plots, prediction dumps, run manifests, and model cards
- raw-image inference through two distinct pipelines:
  - a v0.1 path that reuses the existing edge-ROI and silhouette stages
  - a v0.2 path that uses a learned ROI-FCN localiser ahead of the distance-and-yaw model

That last point materially changes the employer-facing interpretation of the repository. This is no longer only a set of training experiments. It is now a bounded perception stack with an implemented inference surface, explicit model interfaces, and checked-in runtime artifacts.

## 3. Task Definition and System Scope

The central perception task is still tightly constrained:

- input: a monocular full-frame image from a fixed camera
- primary output: vehicle distance in metres
- expanded output: vehicle yaw/orientation
- auxiliary runtime task: predict the crop centre needed to extract a fixed `300 x 300` ROI for downstream regression

The bounded assumptions remain important:

- one known vehicle family
- one fixed calibrated camera geometry
- one constrained movement plane
- synthetic data only
- no evidence of general multi-object handling, scene understanding, or real-world transfer

This means the repository should be read as evidence of bounded applied machine learning, computer vision, and research engineering discipline, not as evidence of a general perception system.

## 4. Repository Architecture

### 4.1 Synthetic Data Generation (`01_rb_synthetic-data_3`)

The Unity project remains a substantial part of the technical story.

Checked-in generator components include:

- `CaptureService.cs` for manual render-texture capture
- `ManifestWriter.cs`, `ManifestRowMapper.cs`, and `RunMetadataWriter.cs` for manifest and run metadata writing
- `FileNamingStrategy.cs` for deterministic sample naming
- `DistanceCalculator.cs` for explicit Euclidean target derivation
- `StratifiedPlacementPlanner.cs` for camera-footprint-aware sample placement
- `VehicleProjectionValidator.cs` for projection-based feasibility checks
- `RunControllerBehaviour.cs` for batch orchestration and attempt-budget management

The technically distinctive generator choice is still the placement strategy. The generator does not merely randomise positions in world space. It projects the camera footprint onto the movement plane, partitions it into feasible regions, validates placements under image-space constraints, and redistributes quota when regions exhaust their attempt budget. That is a better fit for usable image-space coverage than naive synthetic sweeps.

### 4.2 Preprocessing and Representation Packing (`02_synthetic-data-processing-v3.0`)

The preprocessing layer remains contract-driven and stage-based. The main dual-stream path is built around:

1. `detect`
2. `silhouette`
3. `pack_dual_stream`
4. optional shuffle support

The repository still shows good discipline here:

- preprocessing contracts are written into `run.json`
- sample-level stage status is preserved in `samples.csv`
- downstream training validates compatibility against those contracts rather than guessing from filenames

The current strongest multitask training path uses preprocessing contract `rb-preprocess-v4-dual-stream-orientation-v1`. That contract is important because it changes both target semantics and image semantics:

- targets include `y_distance_m`, `y_yaw_deg`, `y_yaw_sin`, and `y_yaw_cos`
- the crop remains a fixed `300 x 300` canvas with scaling disabled
- the image content is no longer just a filled silhouette shape
- the declared representation is `grayscale_vehicle_detail_inside_silhouette` with the vehicle rendered dark on a white background

That fixed-canvas, no-rescaling choice is still a strong one. It preserves apparent object size as a depth cue instead of normalising it away.

### 4.3 Training and Evaluation (`03_rb-training-v2.0`)

The training repository is broader than a single model script. The current snapshot includes:

- topology contracts and a topology registry
- dataset summary and preprocessing-contract validation
- shard-based NPZ loading with RAM-aware caching
- split overlap checks
- run manifests, model cards, plots, and sample prediction dumps
- guarded resume-state support
- task-runtime reporting for both scalar distance and multitask distance-plus-orientation outputs

Model families currently represented in `src/topologies` include:

- `distance_regressor_2d_cnn`
- `distance_regressor_dual_stream` variants
- `distance_regressor_dual_stream_yaw`
- `distance_regressor_global_pool_cnn` as an additional extension point

The strongest checked-in multitask model remains a direct extension of the dual-stream distance architecture rather than an unrelated reset:

- image input: single-channel `300 x 300` crop
- geometry input: 10-element bounding-box feature vector
- shared fused trunk
- output heads for scalar distance and yaw `sin/cos`
- weighted multitask Huber training objective

That continuity matters because it shows architectural iteration under a stable benchmark family, not a change of task to recover good-looking numbers.

### 4.4 ROI-FCN Preprocessing and Training (`04_ROI-FCN`)

The most material architectural expansion beyond the earlier write-up is the dedicated ROI-FCN subsystem.

`04_ROI-FCN/01_preprocessing` implements a separate preprocessing path for crop-centre localisation. It bootstraps centre targets from the existing edge-ROI path and packs locator datasets that preserve:

- full-frame grayscale locator input
- target centre in original-image space
- target centre in locator-canvas space
- resize scale
- padding offsets
- optional bootstrap box metadata

The checked-in dataset contract for the trained ROI-FCN run shows:

- preprocessing contract version `rb-preprocess-roi-fcn-v0_1`
- locator canvas size `480 x 300`
- training split size `100,000`
- validation split size `20,000`

`04_ROI-FCN/02_training` then trains a separate model family for this task. The first-pass topology is intentionally narrow:

- topology id `roi_fcn_tiny`
- single-channel input
- single heatmap output
- Gaussian heatmap supervision
- deterministic argmax decode back into original-image coordinates

The checked-in run `260420-1219_roi-fcn-tiny/run_0003` substantiates that this is not just scaffold code. It includes `run_config.json`, `dataset_contract.json`, checkpoints, plots, predictions, and summary metrics.

### 4.5 Inference Pipelines (`05_inference-v0.1` and `05_inference-v0.2`)

The repository now contains two distinct raw-image inference paths.

#### `05_inference-v0.1`

This is the simpler runtime path. It:

- loads one trained distance-orientation model
- discovers a raw-image corpus
- builds a temporary single-sample input run
- reuses the sibling preprocessing project’s `detect` and `silhouette` stages
- reconstructs the regression input under the model’s declared preprocessing contract
- runs single-sample distance-and-yaw inference
- optionally saves ROI and JSON artifacts

This path is important because it proves the trained model can be driven from raw full-frame imagery through the already-existing preprocessing stages. It is best read as an inference-time reuse of the current preprocessing stack rather than a deployment-oriented runtime redesign.

#### `05_inference-v0.2`

This is the more significant runtime addition. It composes two separately trained models:

1. an ROI-FCN crop-centre localiser
2. a dual-stream distance-and-yaw regressor

The runtime path does the following:

- loads the distance-orientation model and its preprocessing contract
- loads the ROI-FCN model and its dataset geometry contract
- builds a full-frame locator canvas
- predicts the crop centre from a heatmap
- derives fixed ROI bounds
- extracts a centred canvas from the raw full-frame image
- regenerates the downstream crop representation expected by the distance model
- derives the 10-element bbox feature vector
- runs single-sample or multi-sample inference
- writes aggregated JSON results and optional ROI images

This path also performs explicit compatibility checks between the selected ROI-FCN model and the selected distance model, including crop-size agreement with the downstream silhouette canvas.

That relationship between training contract, preprocessing contract, model interface, and runtime path materially strengthens the repository story. It is now possible to point to a complete bounded perception stack rather than only to offline training artifacts.

## 5. Benchmark, Representation, and Model Evolution

The repository still shows a coherent benchmark progression rather than a grab-bag of disconnected experiments.

The visible progression is:

- centre-line distance regression on narrow spatial regimes
- full-frame coverage with limited yaw variation
- full-frame coverage with full `360°` yaw variation
- orientation-aware dual-stream training on the hardest full-frame regime
- inference-facing work around new `def90_synth_v023-*` corpora and full-frame runtime paths

The representation and model evolution is similarly coherent:

- naive full-frame CNN baseline
- dual-stream crop-plus-geometry distance model
- dual-stream multitask distance-plus-yaw model
- separate ROI-FCN crop-centre localiser to support a more self-contained runtime path

One especially important transition is that the strongest multitask path did not respond to the harder benchmark by narrowing the task again. Instead, it improved the representation and target contract:

- preserved bounded ROI framing
- kept no-rescale crop semantics
- upgraded the crop content from pure silhouette shape to masked grayscale vehicle detail
- promoted yaw from metadata into a first-class prediction target
- expanded reporting to include angular metrics and thresholded yaw accuracy

The newer `260416-1425_ds-2d-cnn` run also shows an inference-facing shift in corpus usage. It trains against `def90_synth_v023-test-shuffled` and validates against `def90_synth_v023-validation-shuffled`. That is useful evidence that the repository is moving toward corpora aligned with the runtime stack. It is not, however, the strongest overall result, especially on yaw.

## 6. Representative Completed Runs and Current Best Substantiated Results

### Distance and Distance-plus-Yaw Regression

| Run | Regime | Train / Validation Samples | Validation MAE | Validation RMSE | Accuracy within `0.10 m` | Yaw Mean Error | Yaw Accuracy within `5°` | Assessment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `260407-1756_2d-cnn/run_0002` | naive full-frame CNN on narrow centre-line regime | `50,100 / 10,020` | `0.05338 m` | `0.06929 m` | `0.87375` | `-` | `-` | Best checked-in single-stream baseline. |
| `260411-1104_ds-2d-cnn/run_0001` | dual-stream distance model on narrow centre-line regime | `50,063 / 10,020` | `0.01512 m` | `0.02137 m` | `0.99950` | `-` | `-` | Strong narrow-regime distance result. |
| `260412-1759_ds-2d-cnn/run_0002` | dual-stream distance model on full-frame limited-yaw regime | `250,000 / 50,000` | `0.02785 m` | `0.09972 m` | `0.97892` | `-` | `-` | First strong checked-in full-frame result, but still tail-heavy. |
| `260413-1847_ds-2d-cnn/run_0002` | dual-stream distance model on full-frame full-yaw regime | `250,000 / 50,000` | `0.04652 m` | `0.10753 m` | `0.93460` | `-` | `-` | Harder regime exposed a real failure tail. |
| `260415-1146_ds-2d-cnn/run_0001` | dual-stream multitask distance+yaw model on full-frame full-yaw regime | `250,000 / 50,000` | `0.01007 m` | `0.01297 m` | `0.99996` | `1.49987°` | `0.97482` | Strongest checked-in offline result. |
| `260416-1425_ds-2d-cnn/run_0001` | multitask run on `def90_synth_v023-*` corpora | `200,000 / 50,000` | `0.01965 m` | `0.02496 m` | `0.99900` | `15.18585°` | `0.42464` | Useful inference-facing corpus alignment, but materially weaker yaw result. |

The strongest checked-in offline model is therefore still `260415-1146_ds-2d-cnn/run_0001`, created on `2026-04-15`.

### ROI-FCN Crop-Centre Localisation

The checked-in ROI-FCN run is substantively evidenced as well:

| Run | Task | Train / Validation Samples | Validation Mean Centre Error | Validation Median Centre Error | Validation `p95` Centre Error | ROI Full Containment Success | Assessment |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `260420-1219_roi-fcn-tiny/run_0003` | full-frame crop-centre localisation on locator canvas | `100,000 / 20,000` | `3.1757 px` | `2.4354 px` | `7.7098 px` | `0.9891` | Strong checked-in evidence that the learned localiser works on its own task. |

## 7. What Is Technically Distinctive

- End-to-end ownership across synthetic generation, preprocessing, training, evaluation, ROI localisation, and raw-image inference.
- Explicit contract design across dataset lineage, preprocessing semantics, topology reporting, and runtime compatibility.
- Camera-footprint-aware synthetic placement rather than naive world-space randomisation.
- Preservation of geometric depth cues by using fixed-canvas crops without image rescaling.
- Promotion of yaw from stored metadata to a first-class multitask target with its own runtime reporting contract.
- Addition of a separate ROI-FCN subsystem that turns crop placement into a learned, contract-backed problem rather than leaving it as an implicit preprocessing assumption.
- A composed runtime path that bridges independently trained components with geometry-aware validation rather than silent glue code.
- Written specifications and playbooks that show a habit of controlling drift in an active research codebase.

## 8. What Is Strong Competent Implementation of Known Ideas

- Modular PyTorch training and evaluation loops with checkpoints, resumes, schedulers, metrics, plots, and model cards.
- NPZ shard packing, schema validation, and split-overlap checks.
- OpenCV-based ROI extraction, contour processing, and convex-hull fallback logic.
- Heatmap-based FCN localisation with deterministic decode.
- Notebook control surfaces backed by reusable Python modules rather than notebook-only logic.
- Integration and smoke tests across preprocessing, training, ROI-FCN training, and inference packages.
- Inference-time artifact writing with JSON payloads and saved ROI images for inspection and auditability.

## 9. Inference and Runtime Capability Now Demonstrated

This is where the repository has changed the most.

The current snapshot does substantiate runnable full-frame inference, but it also shows that the runtime story is more mixed than the best offline benchmark numbers alone would suggest.

The strongest checked-in offline multitask result, `260415-1146_ds-2d-cnn/run_0001`, looks excellent under the failure-analysis framework:

- distance within `10 cm`: `49,998 / 50,000`
- yaw within `5°`: `48,741 / 50,000`
- operational failures are overwhelmingly orientation-driven
- a checked-in failure analysis concludes that distance failure has nearly collapsed and the residual weakness is a sparse, edge-sensitive yaw tail

That is an important positive result, and it reflects the current best offline benchmark picture.

However, the checked-in end-to-end raw-image inference artifact under `05_inference-v0.2/output/260415-1146_ds-2d-cnn/` tells a more cautious runtime story. The saved inference JSON contains `2,048` predictions created on `2026-04-21` using:

- distance-orientation model `260415-1146_ds-2d-cnn/run_0001`
- ROI-FCN model `260420-1219_roi-fcn-tiny/run_0003`
- raw corpus `def90_synth_v023-validation-shuffled`

Across that runtime artifact:

- distance MAE is `0.10187 m`
- distance accuracy within `0.10 m` is `0.9360`
- yaw mean absolute error is `11.8651°`
- yaw accuracy within `5°` is `0.6089`
- joint success under the framework (`<= 10 cm` and `<= 5°`) is `0.5957`
- clean success (`<= 5 cm` and `<= 2.5°`) is `0.1650`

That gap matters. It suggests the repository has indeed crossed from offline modelling into implemented runtime perception, but the end-to-end stack is not yet carrying the offline benchmark quality all the way through raw-image inference. The remaining problem is no longer just model quality in isolation. It is now crop placement, runtime composition, and raw full-frame robustness.

That is still a strong employer-facing story, because it shows the harder and more realistic engineering step: turning trained components into a bounded system and then measuring where the system degrades.

## 10. Weaknesses, Limitations, and Caveats

- All checked-in results remain synthetic. There is no repo evidence of real-image transfer.
- The task remains tightly bounded to one known object family, one camera setup, and a constrained operating geometry.
- Training and validation corpora are separated and contract-checked, but they are still generated by related synthetic pipelines.
- The strongest offline result and the strongest checked-in runtime artifact are not the same thing. End-to-end runtime quality is materially weaker.
- The v0.2 runtime path composes independently trained models from different subsystems rather than a jointly trained end-to-end stack.
- The ROI-FCN target is bootstrapped from the existing edge-ROI path, so it initially learns that heuristic rather than a separately curated ground-truth centre definition.
- The v0.2 inference path currently requires CUDA; it is not presented as a lightweight deployment package.
- The checked-in `260416-1425` inference-facing multitask run strengthens the transition-to-runtime story, but it also shows that yaw quality on the newer corpus family is still unresolved.
- Some newer repository areas are clearly still in motion. For example, `05_inference-v0.3` exists as a directory but has no checked-in implementation content yet.

## 11. Concrete Skills and Tools Demonstrated

- Machine learning and data: PyTorch model development, multitask regression, circular regression via yaw `sin/cos`, dataset validation, shard-based loading, metrics design, and failure analysis.
- Computer vision: edge-based ROI extraction, contour and hull processing, silhouette-based masking, grayscale representation design, full-frame crop localisation, and heatmap decode.
- Synthetic data and simulation: Unity C#, render-texture capture, structured manifest writing, distance-target derivation, and camera-footprint-aware synthetic placement.
- Runtime engineering: model discovery, contract-aware checkpoint loading, compatibility checks across model families, raw-image inference, and artifact serialization.
- Tooling and engineering discipline: notebooks as control surfaces, reusable helper modules, tests, run manifests, resume support, and specification-driven internal documentation.
- Languages and frameworks: Python, C#, Unity, PyTorch, NumPy, pandas, OpenCV, Jupyter, and tmux.

## 12. Overall Employment-Facing Assessment

The current repository substantiates more than model training competence.

It shows the ability to build and evolve a bounded perception system across data generation, representation design, training, evaluation, failure analysis, and runtime integration. It also shows a useful level of engineering honesty: the strongest offline benchmark result is excellent, but the checked-in runtime artifact still exposes a substantial end-to-end gap, especially in yaw robustness.

That combination is credible and employment-relevant. It suggests someone who can do more than optimise a training loop: someone who can define contracts, trace data lineage, extend a research stack into runtime code, measure where the system fails, and keep the claims bounded to what the repository actually proves.
