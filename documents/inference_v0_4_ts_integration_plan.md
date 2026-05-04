# Inference v0.4-ts to Live Inference v0.1 Integration Plan

## Summary

This plan integrates the inference implementation from `./05_inference-v0.4-ts` into `./06_live-inference_v0.1` by adding concrete implementation adapters behind the existing live contracts. The generic live pipeline should continue to know only that it has:

- a `RawImagePreprocessor`
- an `InferenceEngine`
- `PreparedInferenceInputs`
- an `InferenceResult`

The generic pipeline must not know OpenCV preprocessing internals, PyTorch model topology internals, checkpoint formats, notebook state, fixed corpus paths, or GUI details.

No implementation code, contract code, file moves, GUI code, worker/thread code, or model-loading changes are part of this planning step. The plan below assumes the current contract boundary remains dependency-light and unchanged for the first integration slice.

One repository note: the shared live contracts now live at `./06_live-inference_v0.1/src/interfaces/contracts.py`. The canonical import path is `interfaces.contracts`.

## Current Live-Inference Foundation

The live side already has a clean separation between contracts and generic orchestration:

- `./06_live-inference_v0.1/src/interfaces/contracts.py`
  - Source of live contract constants, dataclasses, and protocols.
  - Defines `RawImagePreprocessor`, `InferenceEngine`, `PreparedInferenceInputs`, `InferenceResult`, `RoiMetadata`, `DebugImageReference`, runtime parameter contracts, frame handoff contracts, worker event contracts, and tri-stream constants.
  - Import hygiene is intentional: no PySide6, OpenCV, NumPy, PyTorch, camera, preprocessing, or model-runtime imports.

- `./06_live-inference_v0.1/src/live_inference/frame_handoff.py`
  - Generic atomic latest-frame file handoff.
  - Uses bytes and contract dataclasses only.

- `./06_live-inference_v0.1/src/live_inference/frame_selection.py`
  - Generic latest-frame selection and duplicate hash skip.
  - Produces `InferenceRequest` plus exact image bytes.

- `./06_live-inference_v0.1/src/live_inference/runtime_parameters.py`
  - Generic runtime parameter spec/update state manager.
  - Does not know specific preprocessing parameter names.

- `./06_live-inference_v0.1/src/live_inference/inference_core.py`
  - Synchronous one-frame core.
  - Calls `preprocessor.prepare_model_inputs(request, image_bytes)`.
  - Calls `engine.run_inference(prepared_inputs)`.
  - Normalizes request id, input hash, parameter revision, and debug image references.
  - Does not import heavy runtime libraries.

- `./06_live-inference_v0.1/src/cameras/synthetic_camera/synthetic_camera.py`
  - Synthetic frame publisher that writes image bytes through the generic handoff.
  - Useful for non-GUI end-to-end testing.

The existing foundation should be preserved. New integration work should add concrete adapters and compatibility helpers around it.

## Old Inference Implementation Inventory

### Files and Roles

| Path | Role | Source-of-truth status | Integration treatment |
|---|---|---|---|
| `05_inference-v0.4-ts/src/inference_v0_1/pipeline.py` | Main old inference implementation. Mixes discovery, model loading, ROI-FCN localization, preprocessing, batching, output decoding, result shaping, and artifact saving. | Source of current notebook-facing end-to-end behavior. Not cleanly separated. | Split into concrete preprocessor adapter, concrete engine adapter, model artifact loader, compatibility checks, and debug artifact writer. Do not port as one module. |
| `05_inference-v0.4-ts/src/inference_v0_1/discovery.py` | Discovers selectable model runs and raw-image corpora. | Source of old operator discovery behavior. | Reuse ideas for model artifact discovery. Corpus discovery is notebook/test-only for live inference because live input is frame bytes. |
| `05_inference-v0.4-ts/src/inference_v0_1/brightness_normalization.py` | Vendored deterministic brightness normalization v3. | Not true long-term source of truth; module comment says it is copied from `02_synthetic-data-processing-v4.0/rb_pipeline_v4/brightness_normalization.py`. | Use in concrete preprocessing implementation only. Prefer wrapping the shared preprocessing source if stable; otherwise lift with parity tests. |
| `05_inference-v0.4-ts/src/inference_v0_1/brightness_analysis.py` | Offline brightness sensitivity diagnostics. | Notebook/operator analysis surface, not live inference core. | Leave behind for now. Some helper ideas can inform tests, but do not integrate into live pipeline. |
| `05_inference-v0.4-ts/src/inference_v0_1/external.py` | Adds sibling project roots to `sys.path` so old code can import preprocessing/training/ROI modules. | Bootstrap workaround for notebooks/scripts. | Do not copy as-is into generic modules. Concrete implementation may need explicit import/path strategy, packaging, or documented dependency setup. |
| `05_inference-v0.4-ts/src/inference_v0_1/paths.py` | Old inference project path helpers and output paths. | Old app path conventions only. | Do not use for live generic modules. Model registry can reuse path sanitization ideas if needed. |
| `05_inference-v0.4-ts/notebooks/01_single_sample_inference_v0.1.ipynb` | Single-sample operator UI using ipywidgets/matplotlib. Calls `run_single_sample_inference(..., device='cuda')`. | Notebook control surface only. | Leave behind. Do not port notebook widgets, matplotlib display, hardcoded CUDA, or save controls into live inference. |
| `05_inference-v0.4-ts/notebooks/02_multi_sample_inference_v0.3.ipynb` | Multi-sample and brightness-analysis operator UI. | Notebook control surface only. | Leave behind. Useful for understanding operator expectations, not live architecture. |
| `05_inference-v0.4-ts/tests/test_single_sample_inference.py` | Smoke and unit tests for old pipeline behavior, including tri-stream keys, orientation image semantics, CUDA policy, and result saving. | Regression reference. | Use as source for adapter tests, especially tri-stream shape/key tests and orientation rendering parity. |
| `05_inference-v0.4-ts/tests/test_brightness_normalization.py` | Parity tests against preprocessing brightness normalization. | Regression reference. | Port/adapt to live preprocessor tests if brightness normalization is included. |
| `05_inference-v0.4-ts/tests/test_brightness_analysis.py` | Offline analysis tests. | Regression reference for diagnostic tooling only. | Mostly leave behind. |
| `05_inference-v0.4-ts/models/distance-orientation/...` | Distance/yaw model artifacts. Includes `config.json`, checkpoints, `dataset_summary.json`, `model_architecture.json`, sometimes `run_manifest.json`, model cards, metrics. | Source of model topology, task contract, preprocessing contract, and checkpoint. | Read through a concrete model artifact loader and normalize into a live manifest view. Do not hardcode selected run paths. |
| `05_inference-v0.4-ts/models/roi-fcn/...` | ROI-FCN locator artifacts. Includes `run_config.json`, checkpoints, `dataset_contract.json`, and metrics. | Source of old ROI locator model and crop geometry. | Treat as a concrete preprocessing dependency, not as the generic `InferenceEngine`. Validate against distance model preprocessing contract. |
| `05_inference-v0.4-ts/input/...` | Raw-image corpus with `images/`, `manifests/run.json`, `manifests/samples.csv`. | Old notebook input data. | Use as fixtures for tests only. Live preprocessor should accept raw bytes, not corpus paths. |

### Important Sibling Dependencies Used by the Old Implementation

`pipeline.py` imports behavior from sibling projects:

- `02_synthetic-data-processing-v4.0/rb_pipeline_v4`
  - `SilhouetteStageConfigV4`
  - `read_grayscale_uint8`, `write_grayscale_png`
  - `_place_image_on_canvas`
  - `_reconstruct_roi_canvas_from_source`
  - `_render_inverted_vehicle_detail_on_white`
  - `_silhouette_to_background_mask`
  - `_yaw_targets_from_row`
  - `_render_orientation_image_scaled_by_foreground_extent`
  - silhouette generator/fallback/writers

- `03_rb-training-v2.0/src`
  - `Batch`
  - `_load_model_from_run`
  - `batch_to_model_inputs`
  - `extract_prediction_heads`
  - `extract_target_heads`
  - `summarize_task_metrics`

- `04_ROI-FCN/02_training/src`
  - `decode_heatmap_argmax`
  - `derive_roi_bounds`
  - ROI-FCN topology resolution/model construction

These are not generic live dependencies. They belong only in concrete implementation modules or behind a packaging/dependency setup step.

### Constants and Contract-Like Values Found in the Old Implementation

The old implementation uses or discovers these values:

- `TRI_STREAM_INPUT_MODE = "tri_stream_distance_orientation_geometry"`
- `TRI_STREAM_ORIENTATION_IMAGE_KEY = "x_orientation_image"`
- `x_distance_image`, `x_orientation_image`, `x_geometry` via `task_runtime.batch_to_model_inputs` and model preprocessing artifacts.
- Preprocessing contract name in artifacts: `rb-preprocess-v4-tri-stream-orientation-v1`.
- Current representation kind in artifacts: `tri_stream_npz`.
- Tri-stream geometry schema:
  - `cx_px`
  - `cy_px`
  - `w_px`
  - `h_px`
  - `cx_norm`
  - `cy_norm`
  - `w_norm`
  - `h_norm`
  - `aspect_ratio`
  - `area_norm`
- Model output keys from task/topology contracts:
  - `distance_m`
  - `yaw_sin_cos`
- Target/debug columns:
  - `distance_m`
  - `yaw_deg`
  - `yaw_sin`
  - `yaw_cos`
- Tri-stream artifact contract fields include canvas size, image layout, geometry dimension, image keys, brightness normalization, orientation context scale, and orientation extent source.

The live contract module already defines matching stable names for the tri-stream input keys, geometry schema, preprocessing contract name, model output keys, and GUI-facing result fields.

## Current v0.4-ts Inference Flow

The old end-to-end flow in `pipeline.py` is:

1. Operator chooses a distance/yaw run, ROI-FCN run, raw corpus, and image through a notebook.
2. Corpus discovery resolves an image path from `images/` plus `manifests/samples.csv`.
3. Distance/yaw model loading:
   - Reads `config.json`.
   - Reads optional `run_manifest.json`.
   - Reads optional `dataset_summary.json`.
   - Resolves CUDA-only device through `resolve_inference_device`.
   - Calls training `_load_model_from_run`.
   - Extracts topology `task_contract`.
   - Resolves preprocessing contract from artifact metadata.
4. ROI-FCN model loading:
   - Reads `run_config.json`.
   - Reads `dataset_contract.json`.
   - Resolves CUDA-only device.
   - Builds topology through ROI-FCN training code.
   - Loads `best.pt` or `latest.pt`.
   - Extracts locator canvas and ROI crop sizes from dataset/run metadata.
5. Compatibility validation:
   - Resolves model `input_mode`.
   - Resolves pack stage from preprocessing contract.
   - Validates tri-stream representation kind.
   - Builds silhouette config from preprocessing contract.
   - Builds pack settings from preprocessing contract.
   - Resolves brightness normalization runtime config.
   - Checks ROI-FCN crop size against silhouette ROI canvas.
   - For tri-stream, checks pack canvas equals silhouette ROI canvas.
6. Image decode/loading:
   - Reads a grayscale `uint8` image from a file path.
   - Live integration must replace this with byte decode while preserving the same grayscale semantics.
7. ROI-FCN locator preprocessing:
   - Builds an aspect-preserving locator canvas from the source grayscale image.
   - Runs ROI-FCN to produce a heatmap.
   - Decodes argmax heatmap to source-image coordinates.
   - Derives requested ROI bounds.
8. ROI extraction:
   - Extracts a fixed-size centered ROI canvas around the predicted center.
   - Fills out-of-frame areas with white.
9. Silhouette extraction:
   - Runs contour silhouette generation with contract-derived thresholds and morphology settings.
   - Uses convex hull fallback if configured and needed.
   - Renders filled or outline silhouette.
   - Rejects empty silhouette.
10. Full-image bbox and geometry:
    - Projects the ROI silhouette into full source coordinates.
    - Computes bbox from foreground mask.
    - Builds the 10-value geometry vector with `_bbox_features_from_xyxy`.
11. Distance image creation:
    - Reconstructs source grayscale ROI canvas.
    - Builds background mask from silhouette.
    - Renders inverted vehicle detail on white.
    - Applies foreground-only brightness normalization when the model preprocessing contract requires it.
    - Places the result on the configured canvas.
    - Old single-sample internal shape is `C,H,W` as `(1, H, W)`; batching later adds `N`.
12. Orientation image creation:
    - Uses the un-brightness-normalized orientation source image.
    - Uses foreground extent from the silhouette mask.
    - Crops target-centered square context by `OrientationContextScale`.
    - Resizes to configured canvas.
    - Old shape is also `(1, H, W)`.
13. Batch/tensor formatting:
    - Builds a training `Batch`.
    - For tri-stream, `Batch.images` becomes distance image, `Batch.extra_inputs["x_orientation_image"]` becomes orientation image, and `Batch.geometry` becomes geometry vector.
    - `batch_to_model_inputs` converts that to:
      - `x_distance_image`
      - `x_orientation_image`
      - `x_geometry`
14. Inference call:
    - Calls model with the model input mapping under `torch.no_grad()`.
15. Output decoding:
    - `extract_prediction_heads` resolves output heads from task contract.
    - `summarize_task_metrics` decodes `yaw_sin_cos` with `atan2(sin, cos)` to yaw degrees modulo 360.
    - Old result building reads `prediction_distance_m`, `prediction_yaw_sin`, `prediction_yaw_cos`, and `prediction_yaw_deg` from the prediction row.
16. Debug/result artifacts:
    - Notebook displays in-memory ROI/model image arrays through matplotlib.
    - Optional save writes JSON and ROI PNG files.
    - Brightness analysis optionally writes its own diagnostic JSON.

### Boundary Split

- Preprocessing:
  - image byte/path decode
  - ROI location
  - ROI extraction
  - silhouette generation
  - distance image
  - orientation image
  - geometry vector
  - preprocessing metadata and optional debug images

- Model inference:
  - model artifact loading
  - prepared input validation
  - tensor conversion and device placement
  - model call
  - output head extraction

- Output decoding:
  - distance scalar extraction
  - yaw sin/cos extraction
  - yaw degree calculation
  - optional output vector validation/correction warnings

- Visualization/debugging:
  - display file generation only
  - no NumPy arrays, tensors, QImages, or OpenCV matrices cross to the GUI side

## Mapping to Live Contracts

| Old concept/file/function | New live component | Contract object/protocol | Notes/risks |
|---|---|---|---|
| Raw image path loaded by `read_grayscale_uint8` | `TriStreamLivePreprocessor.prepare_model_inputs(request, image_bytes)` decodes bytes internally | `RawImagePreprocessor`, `InferenceRequest` | Must preserve grayscale decode semantics while accepting bytes only. |
| Raw image bytes from live handoff | Frame selector passes exact bytes into preprocessor | `SelectedFrameForInference`, `InferenceRequest` | Hash must correspond to exact bytes decoded. Already handled by live selector/core. |
| `_build_roi_fcn_locator_input` | Concrete ROI locator dependency inside preprocessor | Implementation-local class/function | Heavy OpenCV/NumPy/Torch imports stay in preprocessing implementation. |
| `_predict_roi_center`, `decode_heatmap_argmax`, `derive_roi_bounds` | ROI-FCN locator adapter used by preprocessor | Implementation-local, not generic `InferenceEngine` | ROI-FCN is preprocessing/ROI extraction for current model family. Validate its crop size against distance model contract. |
| `_extract_centered_canvas` | Preprocessor helper | Implementation-local | Can be lifted with tests. |
| Silhouette config and generator/fallback/writer calls | Preprocessor helper modules | Implementation-local | Wrap existing preprocessing source or lift with parity tests. Private helper imports are fragile. |
| `_render_inverted_vehicle_detail_on_white`, `_place_image_on_canvas`, brightness normalization | Distance stream preparation | `PreparedInferenceInputs.model_inputs["x_distance_image"]` | Must match model preprocessing contract. Do not invent a new algorithm. |
| `_render_orientation_image_scaled_by_foreground_extent` | Orientation stream preparation | `PreparedInferenceInputs.model_inputs["x_orientation_image"]` | Must preserve raw-detail/no-brightness semantics from artifact contract. |
| `_bbox_features_from_xyxy` | Geometry vector preparation | `PreparedInferenceInputs.model_inputs["x_geometry"]` | Must validate schema equals `TRI_STREAM_GEOMETRY_SCHEMA`. |
| `PreprocessedSample` | Replaced by `PreparedInferenceInputs` plus metadata | `PreparedInferenceInputs` | Old truth fields and corpus paths should not be required for live inference. |
| Bbox/ROI metadata in `row_payload` and `PreprocessedSample` | `preprocessing_metadata` and later `RoiMetadata` | `PreparedInferenceInputs.preprocessing_metadata`, `RoiMetadata` | Store bbox, source image size, canvas sizes, crop/request/source bounds, fallback flags, warnings, brightness metadata. |
| `load_model_context` | Concrete model artifact loader for distance/yaw engine | Implementation-local, possibly `model_registry` | Not generic core. Must normalize existing artifacts to a manifest-like view. |
| `load_roi_fcn_model_context` | Concrete ROI locator artifact loader | Implementation-local preprocessing dependency | Not generic core. May live under `preprocessing/roi_locator.py` or artifact loader with clear role. |
| `_validate_model_compatibility` | Compatibility helper | Implementation-local or light `model_registry.compatibility` | Should avoid heavy imports if possible; compare mappings and contract constants. |
| `Batch` and `batch_to_model_inputs` | Engine-side tensor formatting or direct mapping conversion | `InferenceEngine.run_inference(inputs)` | Prefer direct conversion from `PreparedInferenceInputs.model_inputs` to tensors. Avoid requiring training `Batch` in live core. |
| Model output mapping keys `distance_m`, `yaw_sin_cos` | Engine output extraction | `InferenceResult` fields | Validate keys and shapes before decoding. |
| `_decode_yaw_deg` in `task_runtime` | `engines/output_decoding.py` | `InferenceResult.predicted_yaw_deg` | Use `atan2(sin, cos)` modulo 360. Include warnings for non-finite or corrected outputs. |
| Old `InferenceResult` dataclass | Live `InferenceResult` | `InferenceResult` | Live result has no actual/truth/delta fields. Keep notebook result separate. |
| Saved ROI PNG/JSON | Optional debug artifact writer | `InferenceResult.debug_paths`, `DebugImageReference` | GUI receives paths only. No arrays or tensors. |
| Old brightness sensitivity diagnostics | Offline analysis only | None for live v0.1 | Do not integrate into live path. |
| Notebook widgets and displays | Leave behind | None | No GUI code in this integration. |
| Runtime preprocessing knobs from artifact stages | Preprocessor-advertised parameter set | `RuntimeParameterSpec`, `RuntimeParameterSetSpec`, `RuntimeParameterUpdate` | GUI should discover names from specs and not hardcode them. |

## Proposed Module Structure

Keep generic modules light and unchanged:

```text
06_live-inference_v0.1/
  src/
    interfaces/
      contracts.py
    live_inference/
      __init__.py
      frame_handoff.py
      frame_selection.py
      inference_core.py
      runtime_parameters.py
```

Add concrete implementation modules under `src/live_inference/`:

```text
live_inference/
  preprocessing/
    __init__.py
    tri_stream_live_preprocessor.py
    preprocessing_config.py
    roi_locator.py
    debug_artifacts.py

  engines/
    __init__.py
    torch_tri_stream_engine.py
    model_artifact_loader.py
    output_decoding.py

  model_registry/
    __init__.py
    model_manifest.py
    compatibility.py
```

Recommended responsibilities:

- `preprocessing/tri_stream_live_preprocessor.py`
  - Implements `RawImagePreprocessor`.
  - Owns byte decode and tri-stream preparation.
  - May import OpenCV/NumPy and concrete preprocessing helpers.

- `preprocessing/roi_locator.py`
  - Encapsulates ROI-FCN locator behavior and artifacts.
  - May import PyTorch, OpenCV, NumPy, and ROI-FCN training modules.
  - Keeps ROI-FCN details out of the distance/yaw engine and generic core.

- `preprocessing/preprocessing_config.py`
  - Resolves silhouette, pack, brightness, and tunable preprocessing config from model preprocessing contracts.
  - Can be light if it only parses mappings; heavy validation can stay in `tri_stream_live_preprocessor.py`.

- `preprocessing/debug_artifacts.py`
  - Writes PNG/JPG debug artifacts and returns file paths.
  - May import OpenCV/NumPy.

- `engines/torch_tri_stream_engine.py`
  - Implements `InferenceEngine`.
  - Loads and runs the distance/yaw model.
  - Converts prepared arrays to tensors.

- `engines/model_artifact_loader.py`
  - Concrete distance/yaw model loader.
  - Wraps existing training topology/checkpoint loading.
  - May import PyTorch and training modules.

- `engines/output_decoding.py`
  - Decodes `distance_m` and `yaw_sin_cos` outputs into live result fields.
  - May be NumPy/PyTorch-light depending on chosen tensor conversion.

- `model_registry/model_manifest.py`
  - Normalizes existing model directories into an implementation-local manifest object.
  - Preferred future input: explicit `live_model_manifest.json`.
  - Backward-compatible input: existing `run_manifest.json`, `model_architecture.json`, `dataset_summary.json`, `config.json`, and checkpoint candidates.

- `model_registry/compatibility.py`
  - Compares model, preprocessing, input, geometry, and output contracts.
  - Should stay stdlib plus `interfaces.contracts` if possible.
  - Returns structured compatibility diagnostics or raises clear errors.

This structure fits the current repo because it keeps the existing core modules untouched while isolating heavy imports in adapter packages.

## Model Update and Compatibility Strategy

Model replacement should be a config/artifact operation, not a change to GUI, frame handoff, frame selector, inference core, or contracts.

### Manifest Normalization

Introduce an implementation-local normalized manifest view, even before adding a new artifact file. The loader should prefer a future explicit `live_model_manifest.json`, but initially derive the same information from existing artifacts:

- `config.json`
- `run_manifest.json`, when present
- `dataset_summary.json`
- `model_architecture.json`
- checkpoint candidates such as `best.pt`, `best_model.pt`, `latest.pt`
- ROI locator `run_config.json`
- ROI locator `dataset_contract.json`

The normalized manifest should include:

- model artifact id/label
- model run path
- checkpoint path
- checkpoint format expectation
- topology id and variant
- topology contract version
- task contract
- preprocessing contract name/version
- input mode
- expected input keys
- expected per-input layout and shape constraints
- expected canvas size
- geometry schema and dimension
- model output keys
- output decoding policy
- device policy
- optional debug metadata
- ROI locator artifact path and crop/canvas contract, if the preprocessor uses ROI-FCN

If any required field is missing from existing artifacts, the loader should fail with a clear compatibility error rather than guessing.

### Compatibility Checks

Before running inference, validate:

- live contract version is supported by the concrete adapters
- topology contract version equals the expected topology contract, currently `rb-topology-output-reporting-v1`
- task `input_mode` is `tri_stream_distance_orientation_geometry`
- preprocessing contract name/version equals `rb-preprocess-v4-tri-stream-orientation-v1`
- preprocessing current representation kind is `tri_stream_npz`
- expected input keys equal:
  - `x_distance_image`
  - `x_orientation_image`
  - `x_geometry`
- distance image key in artifact equals `x_distance_image`
- orientation image key in artifact equals `x_orientation_image`
- geometry key in artifact equals `x_geometry`
- geometry schema equals the live `TRI_STREAM_GEOMETRY_SCHEMA`
- geometry dimension is 10
- canvas dimensions are positive and match across silhouette and pack stages where the old implementation requires that
- distance image and orientation image layout is understood by the adapters
- model output keys include:
  - `distance_m`
  - `yaw_sin_cos`
- yaw output width is 2
- distance output width is 1
- output decoding policy is distance plus yaw sin/cos
- ROI locator crop size is compatible with the distance model silhouette ROI canvas
- selected device is available, or a configured CPU/test policy is active

### Failure Modes

The concrete loader/compatibility layer should reject clearly when:

- preprocessing contract does not match the expected model contract
- input keys differ
- geometry schema differs
- output keys differ
- model artifact cannot be loaded
- checkpoint format is not recognized
- model expects a different runtime representation
- ROI locator crop/canvas geometry is incompatible
- required artifact metadata is missing
- configured device cannot run the artifact

These failures should surface through the concrete adapter construction or through `InferenceEngine.run_inference`, and `InferenceProcessingCore` will package runtime failures as `WorkerError` without needing to know the details.

### Model Replacement Flow

The future live app should point to a model bundle/config entry, not hardcoded paths in generic code:

1. Select or configure a model artifact root.
2. Load/derive normalized manifest.
3. Construct `TriStreamLivePreprocessor` with the manifest-derived preprocessing config and ROI locator dependency.
4. Construct `TorchTriStreamInferenceEngine` with the manifest-derived model context.
5. Run compatibility checks before first frame.
6. Only if checks pass, hand the concrete instances to `InferenceProcessingCore`.

Replacing a compatible model should not require edits to GUI code, frame handoff, frame selection, inference core, or runtime parameter manager.

## Preprocessor Adapter Plan

Proposed class:

```text
TriStreamLivePreprocessor
```

It should implement:

```text
RawImagePreprocessor.prepare_model_inputs(request, image_bytes) -> PreparedInferenceInputs
```

### Responsibilities

- Accept image bytes, not file paths.
- Decode image bytes internally to grayscale `uint8`.
- Reproduce the old tri-stream preprocessing semantics.
- Use contract-derived preprocessing settings, not hardcoded canvas sizes or thresholds.
- Produce `PreparedInferenceInputs`.
- Populate `model_inputs` with:
  - `x_distance_image`
  - `x_orientation_image`
  - `x_geometry`
- Populate `preprocessing_metadata` with contract, ROI, geometry, timing, warnings, and parameter revision details.
- Optionally write debug/preprocessed images to files and expose paths.

### Old Code to Lift or Wrap

Lift with tests where small and self-contained:

- `_build_roi_fcn_locator_input`
- `_extract_centered_canvas`
- `_contour_break_reason`
- `_render_is_empty`
- `_mask_geometry`
- `_bbox_features_from_xyxy`
- `_brightness_result_payload`
- `_disabled_brightness_payload`

Wrap or call existing source where possible:

- `SilhouetteStageConfigV4`
- `ContourSilhouetteGeneratorV2`
- `ConvexHullFallbackV1`
- `FilledArtifactWriterV1`
- `OutlineArtifactWriterV1`
- `_silhouette_to_background_mask`
- `_reconstruct_roi_canvas_from_source`
- `_render_inverted_vehicle_detail_on_white`
- `_place_image_on_canvas`
- `_render_orientation_image_scaled_by_foreground_extent`
- brightness normalization source
- ROI-FCN heatmap decode and ROI bound derivation

Do not invent or tune a new preprocessing algorithm during integration. The first implementation should be behavior-preserving, then tests can expose live-domain weaknesses separately.

### Prepared Input Shape Convention

The old single-frame preprocessor produces channel-first arrays shaped `(1, H, W)` and the old batch builder adds the batch dimension. The live adapter should document and test one invariant:

- preprocessor emits per-frame arrays as `C,H,W`, usually `(1, H, W)`
- engine adds `N=1` when converting to tensors

This can be recorded in `preprocessing_metadata` as an implementation invariant. A contract change is not required for the first adapter pair because `PreparedInferenceInputs.model_inputs` is intentionally runtime-typed as `Any`; however, if multiple independently developed preprocessors and engines need interchangeability, an explicit shape-convention field may become a real cross-component contract gap.

### Metadata to Populate

Recommended `preprocessing_metadata` fields:

- `preprocessing_contract_name`
- `preprocessing_contract_version`
- `input_mode`
- `input_keys`
- `representation_kind`
- `geometry_schema`
- `geometry_dim`
- `source_image_width_px`
- `source_image_height_px`
- `distance_canvas_width_px`
- `distance_canvas_height_px`
- `orientation_canvas_width_px`
- `orientation_canvas_height_px`
- `roi_request_xyxy_px`
- `roi_source_xyxy_px`
- `roi_canvas_insert_xyxy_px`
- `predicted_roi_center_xy_px`
- `detect_bbox_xyxy_px`
- `silhouette_bbox_xyxy_px`
- `silhouette_area_px`
- `silhouette_fallback_used`
- `brightness_normalization`
- `orientation_context_scale`
- `runtime_parameter_revision`
- `warnings`
- `debug_paths`, if debug image saving is enabled

The engine should convert relevant metadata into `RoiMetadata` on the final `InferenceResult`.

## Inference Engine Adapter Plan

Proposed class:

```text
TorchTriStreamInferenceEngine
```

It should implement:

```text
InferenceEngine.run_inference(inputs: PreparedInferenceInputs) -> InferenceResult
```

### Responsibilities

- Load the distance/yaw model artifact through concrete loader logic.
- Validate model/preprocessing/input/output compatibility before running.
- Accept `PreparedInferenceInputs`.
- Validate required tri-stream keys.
- Convert arrays to tensors and add batch dimension according to the chosen adapter invariant.
- Place tensors on the configured device.
- Run the PyTorch model under no-grad/eval mode.
- Decode outputs into:
  - `predicted_distance_m`
  - `predicted_yaw_sin`
  - `predicted_yaw_cos`
  - `predicted_yaw_deg`
- Populate timing fields:
  - preprocessing time if passed through metadata
  - inference time
  - total time if available
- Preserve request id and input image hash from `inputs.source_frame`.
- Populate `RoiMetadata` from preprocessor metadata.
- Pass through debug artifact paths from metadata to `InferenceResult.debug_paths`.
- Include warnings for output correction/normalization or metadata inconsistencies.

### Old Code to Wrap

- `_load_model_from_run` from training `src.evaluate`
- topology resolution/building from training modules
- `extract_prediction_heads` from `src.task_runtime`, or a smaller live decoder that validates the known mapping output keys

### Output Decoding Source

Current yaw degree decoding lives in `03_rb-training-v2.0/src/task_runtime.py` inside `_decode_yaw_deg`, called by `summarize_task_metrics`. It uses:

- `atan2(sin, cos)`
- degrees
- modulo 360

The live engine should not call `summarize_task_metrics` for normal inference because live inference has no truth targets and does not need metrics tables. Instead, move the minimal output decoding into `engines/output_decoding.py` and test it against known sin/cos values.

### Device Policy

The old implementation currently requires CUDA for distance/yaw and ROI-FCN loading. The live adapter should make device policy explicit:

- production default may remain `cuda`
- CPU should be available for tests when a small test artifact or fake torch model is used
- incompatible device requests should fail clearly
- generic live modules should not know device details

If current real artifacts cannot run on CPU with the existing loader, tests should use a small compatible test artifact or a fake engine until a CPU-safe artifact is available.

## Debug and Display Artifact Plan

The GUI should receive file references only, never OpenCV images, NumPy arrays, QImages, or tensors.

Debug artifact writing should be optional and configured by `InferenceRequest.save_debug_images` and `InferenceRequest.debug_output_dir` or equivalent construction config.

Recommended debug image kinds:

- `accepted_raw_frame`
- `x_distance_image`
- `x_orientation_image`
- `roi_overlay`
- optionally `silhouette_mask`, `roi_crop`, or locator heatmap under implementation-specific names

Recommended filename scheme:

```text
{debug_output_dir}/{request_id}__{short_frame_hash}__{image_kind}.png
```

Rules:

- use request id for uniqueness and traceability
- include a short frame hash prefix for human debugging
- write files atomically where practical
- return paths through `InferenceResult.debug_paths`
- let `InferenceProcessingCore` convert those paths into `DebugImageReference`

Cleanup policy:

- First implementation can leave cleanup to the caller/test harness.
- Add an optional retention policy later, such as max age or max count, in a debug-artifact manager.
- Cleanup should not be in contracts or generic core.

## Runtime Tuning Parameter Plan

The GUI should discover tunable parameters from the preprocessor/inference side through `RuntimeParameterSetSpec`. It should not hardcode parameter names.

The preprocessor should expose a namespace such as:

```text
owner = WorkerName.INFERENCE
namespace = "preprocessing"
```

Attach the current parameter revision to:

- `PreparedInferenceInputs.preprocessing_metadata["runtime_parameter_revision"]`
- `InferenceResult.preprocessing_parameter_revision`

### Plausible Runtime-Tunable Parameters

| Parameter | Source in old artifacts/code | Type | Suggested widget | Default source | Notes |
|---|---|---:|---|---|---|
| `silhouette.blur_kernel_size` | `Stages.silhouette.BlurKernelSize` | int | int input or slider | model preprocessing contract | Kernel sizes should remain positive odd integers. |
| `silhouette.canny_low_threshold` | `Stages.silhouette.CannyLowThreshold` | int | slider | model preprocessing contract | Valid range likely 0-255. |
| `silhouette.canny_high_threshold` | `Stages.silhouette.CannyHighThreshold` | int | slider | model preprocessing contract | Must be >= low threshold. Cross-field validation is implementation-specific. |
| `silhouette.close_kernel_size` | `Stages.silhouette.CloseKernelSize` | int | int input | model preprocessing contract | Positive odd integer. |
| `silhouette.dilate_kernel_size` | `Stages.silhouette.DilateKernelSize` | int | int input | model preprocessing contract | Positive integer. |
| `silhouette.min_component_area_px` | `Stages.silhouette.MinComponentAreaPx` | int | int input | model preprocessing contract | Valid range depends on canvas size. |
| `silhouette.fill_holes` | `Stages.silhouette.FillHoles` | bool | checkbox | model preprocessing contract | Behavior-preserving default from artifact. |
| `silhouette.use_convex_hull_fallback` | `Stages.silhouette.UseConvexHullFallback` | bool | checkbox | model preprocessing contract | Useful for live failures, but changes failure/recovery behavior. |
| `tri_stream.orientation_context_scale` | `Stages.pack_tri_stream.OrientationContextScale` | float | slider or float input | model preprocessing contract | High risk: affects model input distribution. Tune only if explicitly exposed. |
| `brightness.enabled` | `BrightnessNormalization.Enabled` | bool | checkbox | model preprocessing contract | Only expose when the artifact declares compatible brightness normalization. |
| `brightness.target_median_darkness` | `BrightnessNormalization.TargetMedianDarkness` | float | slider | model preprocessing contract | Current tri-stream artifact example uses 0.55. |
| `brightness.min_gain` | `BrightnessNormalization.MinGain` | float | float input | model preprocessing contract | Must be > 0 and <= max gain. |
| `brightness.max_gain` | `BrightnessNormalization.MaxGain` | float | float input | model preprocessing contract | Must be >= min gain. |
| `brightness.empty_mask_policy` | `BrightnessNormalization.EmptyMaskPolicy` | enum | dropdown | model preprocessing contract | Choices discovered from implementation, currently `skip` or `fail` in old helper. |

### Should Not Be Runtime-Tuned Initially

- model artifact path
- checkpoint path
- topology id/variant
- preprocessing contract name/version
- input keys
- output keys
- geometry schema
- geometry dimension
- distance/orientation canvas dimensions
- ROI-FCN model path
- ROI-FCN crop size
- target/output decoding mode
- frame hash algorithm

Those values are compatibility boundaries. Changing them should require artifact reload/revalidation, not a live parameter update.

## Testing Strategy

All Python commands should use the repository virtual environment, for example:

```text
./.venv/bin/python -m unittest ...
```

If a dependency is missing in the venv, the test should fail or skip clearly depending on level.

### A. Contract and Compatibility Tests

- Generic contract/core import hygiene:
  - no PySide6/OpenCV/NumPy/PyTorch imports in `src/interfaces/contracts.py`, `frame_handoff.py`, `frame_selection.py`, `runtime_parameters.py`, or `inference_core.py`.
- Manifest compatibility accepts known compatible tri-stream artifact metadata.
- Manifest compatibility rejects:
  - wrong preprocessing contract
  - wrong input mode
  - missing input keys
  - geometry schema mismatch
  - output key mismatch
  - incompatible ROI locator crop size
  - missing checkpoint
- Compatibility checks should use small JSON fixtures and no heavy model load where possible.

### B. Preprocessor Adapter Tests

- Accepts valid image bytes.
- Rejects invalid image bytes with clear error.
- Returns `PreparedInferenceInputs`.
- Uses input mode `tri_stream_distance_orientation_geometry`.
- Contains all tri-stream model input keys.
- Distance and orientation streams have expected per-frame shape and dtype.
- Geometry vector has dimension 10 and schema metadata.
- Metadata includes preprocessing contract, bbox/ROI data, image/canvas sizes, warnings, and parameter revision.
- A small fixture image reproduces old orientation-image semantics, especially white background and non-white vehicle detail.
- Brightness normalization parity test against the old/shared implementation.
- ROI locator can be faked for deterministic tests so tests do not require loading ROI-FCN.

### C. Engine Adapter Tests

- Loads a known small compatible test artifact if available.
- If no real CPU-safe artifact exists, use a fake torch module plus manifest fixture for unit tests.
- Rejects incompatible artifact metadata before inference.
- Rejects missing input keys.
- Rejects bad tensor shapes.
- Decodes distance and yaw outputs correctly.
- Handles CPU for tests if CUDA is unavailable.
- Maps output to live `InferenceResult`.
- Preserves `request_id`, `input_image_hash`, `debug_paths`, and parameter revision.

### D. End-to-End Non-GUI Tests

- Synthetic camera publishes a frame.
- Frame selector selects it.
- Preprocessor prepares inputs.
- Engine runs inference.
- Inference core returns `InferenceResult`.
- Duplicate skip occurs only after successful inference.
- Preprocess failure returns `WorkerError` at `PREPROCESS`.
- Engine failure returns `WorkerError` at `INFERENCE`.
- No GUI, worker, or thread code involved.

### E. Import Hygiene Tests

- Generic modules continue to avoid heavy runtime libraries.
- Heavy imports are isolated to:
  - `live_inference.preprocessing.*`
  - `live_inference.engines.*`
  - possibly ROI/model registry implementation modules
- `model_registry.compatibility` should stay light if it only compares mappings.

## Migration Strategy

Implement in small safe slices:

1. Finalize this inventory and mapping report.
2. Add light model/preprocessing manifest normalization and compatibility helpers.
3. Add tests for compatibility helpers using JSON fixtures.
4. Implement `TriStreamLivePreprocessor` with a fake ROI locator first.
5. Add preprocessor tests with small image-byte fixtures.
6. Add concrete ROI-FCN locator adapter behind the preprocessor, with tests that can mock model output.
7. Implement model artifact loader for distance/yaw artifacts.
8. Implement `TorchTriStreamInferenceEngine`.
9. Add engine tests, including CPU-safe fake/small artifact tests.
10. Add non-GUI end-to-end test using synthetic camera, frame selector, preprocessor, engine, and inference core.
11. Only after synchronous non-GUI path is green, build an inference worker/thread wrapper.
12. Only after worker behavior is green, start PySide6 GUI integration.

## Contract Gap Assessment

No contract change is required for the first implementation.

Existing contracts already provide:

- `RawImagePreprocessor`
- `InferenceEngine`
- `PreparedInferenceInputs`
- `InferenceResult`
- `RoiMetadata`
- `DebugImageReference`
- `ModelContractReference`
- runtime parameter contracts
- tri-stream input keys
- geometry schema
- output field names

Potential future gap:

- `PreparedInferenceInputs` does not explicitly state whether runtime arrays are per-frame `C,H,W` or batched `N,C,H,W`.

For the first adapter pair, document and test the invariant in implementation metadata instead of changing contracts. If separately replaceable preprocessors and engines need to interoperate across packages, this becomes genuinely cross-component and may justify adding a dependency-light shape/layout field to the contract.

## Risks and Open Questions

- The old notebooks contain operator-only assumptions: ipywidgets, matplotlib display, corpus path selection, save toggles, and hardcoded `device='cuda'`. These should not move into live inference.
- The inspected live contract file is currently under `src/interfaces/contracts.py`, not `src/live_inference/contracts.py`. The canonical import path is `interfaces.contracts`.
- Current old model loading requires CUDA. A CPU test path needs either a small compatible artifact, a fake torch model, or a controlled test-only device policy.
- Existing distance/yaw artifacts have useful metadata, but a future explicit `live_model_manifest.json` would make compatibility less dependent on reconstructing intent from training files.
- Some current artifacts have `run_manifest.json`; at least one inspected tri-stream run has complete metadata there, while another relies more on `dataset_summary.json`/`model_architecture.json`. Loader behavior must handle missing optional files without guessing required fields.
- The old inference implementation imports private underscore helpers from sibling preprocessing modules. Short-term wrapping is acceptable with parity tests; long-term, stable public helpers would reduce fragility.
- ROI-FCN is a model used inside preprocessing. Its artifact lifecycle should be explicit so a distance/yaw model update cannot silently pair with an incompatible locator crop geometry.
- The old detector/ROI strategy was validated on synthetic raw images. Live real-camera imagery may require a separate domain-gap investigation; do not solve that by changing generic pipeline code.
- The current contour/silhouette strategy may fail differently on live frames. Runtime tuning can help, but model compatibility and behavior-preserving defaults should come first.
- Debug artifact cleanup policy is not defined yet. Initial implementation can leave files for inspection; retention should be a later implementation detail.
