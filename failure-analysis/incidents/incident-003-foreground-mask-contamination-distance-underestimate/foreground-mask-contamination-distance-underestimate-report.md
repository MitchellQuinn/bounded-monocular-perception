# Incident Report: Foreground Mask Contamination Causing Live Distance Underestimate

**Incident:** `incident-003-foreground-mask-contamination-distance-underestimate`  
**System:** bounded monocular perception, live inference v0.3  
**Date analysed:** 2026-05-26  
**Status:** Investigated; remediation proposed  
**Primary trace:** [`20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859)

## 1. Executive Summary

During live inference, the system predicted the Defender at `1.3255 m`. The operator reported that similar physical distances were being estimated approximately correctly, making this trace a clear outlier in the "too close" direction.

The trace shows that the ROI locator found a plausible vehicle-sized target. The accepted locator bbox was `150 x 117 px` with confidence `0.8380`. The downstream foreground extraction step then expanded the model's foreground bbox to `304 x 259 px`, with `44,792` foreground pixels. The generated distance image and geometry vector therefore described a much larger object than the Defender itself.

This is the mirror image of incident 001. Incident 001 collapsed the foreground to a tiny fragment and drove a distance overestimate. Incident 003 expands the foreground by merging the target with dark sheet folds and shadow, driving a distance underestimate.

The model output is therefore explainable from the corrupted input. The current tri-stream model receives:

```text
x_distance_image
x_orientation_image
x_geometry
```

All three streams were contaminated by the foreground mask. The model saw a large, dark, contiguous foreground region on a `320 x 320 px` canvas and inferred a closer object.

The recommended remediation is to add a post-foreground quality gate that compares foreground geometry against locator geometry, then either rejects the frame or falls back to locator-anchored foreground extraction. The medium-term fix is to make foreground extraction component-aware and locator-anchored, rather than accepting all thresholded dark pixels inside the ROI.

## 2. Incident Scope

This report covers one live trace captured on `2026-05-26T11:18:23Z`:

```text
06_live-inference_v0.3/live_traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859
```

A copy of the trace is stored with this report under [`evidence/traces`](evidence/traces).

The trace used:

| Field | Value |
| --- | --- |
| Distance/orientation model | `260521-1029_ts-2d-cnn` |
| Topology variant | `tri_stream_yaw_v0_5` |
| Checkpoint | `models/distance-orientation/260521-1029_ts-2d-cnn/best.pt` |
| Checkpoint SHA-256 | `0696f50e1365df4210d7fb5cc98eca8176232977469a869eb6e0fbea4e863911` |
| Model selection | `06_live-inference_v0.3/models/selections/current.toml` |
| Camera source | `opencv-v4l2`, `/dev/video0`, `1920 x 1200`, `YUYV`, `50 fps` |
| Camera encoding | `bmp` |
| Runtime device | `cpu` |
| Git commit in manifest | `97aac9c3c7f77494d9f59220634d6f950d1b3771` |
| Git dirty flag in manifest | `false` |

The trace does not encode a measured ground-truth distance. The failure claim is therefore relative: this capture produced a materially lower distance estimate than nearby live behaviour and is visually inconsistent with the model input quality expected from previous successful traces.

## 3. Expected Behaviour

At a similar physical distance, a small change in vehicle placement or pose should not cause a large apparent-scale change in the model input. The locator, foreground mask, distance image, and geometry vector should remain mutually consistent:

```text
locator bbox ~= foreground bbox ~= vehicle extent in x_distance_image
```

The expected failure mode, if preprocessing is uncertain, is no prediction or an explicit preprocessing warning. A plausible-looking scalar distance should not be emitted when model-input geometry is visibly inconsistent with the accepted locator result.

## 4. Evidence Summary

| Signal | Trace value |
| --- | ---: |
| Predicted distance | `1.325526 m` |
| Predicted yaw | `29.3355 deg` |
| Locator bbox | `[1029, 521, 1179, 638]` |
| Locator bbox size | `150 x 117 px` |
| Locator confidence | `0.838026` |
| Regressor ROI source | `[944, 420, 1264, 740]` |
| Foreground bbox | `[944, 441, 1248, 700]` |
| Foreground bbox size | `304 x 259 px` |
| Foreground pixel count | `44,792 px` |
| Foreground bbox area / locator bbox area | `4.49 x` |
| Foreground pixels / locator bbox area | `2.55 x` |
| Geometry `w_norm` | `0.158333` |
| Geometry `h_norm` | `0.215833` |
| Geometry `area_norm` | `0.034174` |

The decisive discrepancy is between the accepted locator bbox and the geometry actually sent to the model. The locator found a compact target. The foreground extraction sent a much larger target to the regressor.

The saved `x_geometry` for the trace is:

```text
[
  1096.0,
  570.5,
  304.0,
  259.0,
  0.5708333253860474,
  0.4754166603088379,
  0.15833333134651184,
  0.21583333611488342,
  1.1737451553344727,
  0.03417361155152321
]
```

## 5. Visual Evidence Summary

The trace artifacts show the failure plainly:

- [`locator_overlay.png`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/locator_overlay.png) shows the accepted locator near the Defender.
- [`roi_crop.png`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/roi_crop.png) contains the Defender plus textured sheet folds.
- [`foreground_mask.png`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/foreground_mask.png) shows that a large region of sheet/shadow was classified as foreground.
- [`x_distance_image.png`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/x_distance_image.png) shows the model's distance stream: the vehicle is merged into a broader dark foreground region.
- [`x_orientation_image.png`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/x_orientation_image.png) is also affected because the orientation crop is derived from the same foreground extent.

The accepted camera frame is not enough to diagnose this failure. The model-input artifacts are the critical evidence.

## 6. Pipeline Reconstruction

The relevant live path is:

1. Decode the accepted camera frame.
2. Apply the manual mask.
3. Run the ROI locator.
4. Extract a fixed `320 x 320 px` ROI around the locator center.
5. Estimate a foreground mask inside the ROI using `threshold_foreground_v1`.
6. Render `x_distance_image` from the foreground mask.
7. Render `x_orientation_image` from the foreground extent.
8. Build `x_geometry` from the foreground bbox.
9. Run the tri-stream distance/yaw model.

The failure occurs between steps 5 and 8. The ROI locator is not the primary failure. The accepted locator bbox is compact and plausible. The foreground extraction stage expands the target region, and that expanded region becomes all downstream model evidence.

The relevant runtime implementation is:

- [`generic_tri_stream_live_preprocessor.py`](../../../06_live-inference_v0.3/src/live_inference/preprocessing/generic_tri_stream_live_preprocessor.py): foreground extraction, distance/orientation rendering, and geometry construction
- [`topology_tri_stream_yaw_v0_5.py`](../../../03_rb-training-v2.0/src/topologies/topology_tri_stream_yaw_v0_5.py): v0.5 model topology, including geometry-conditioned residual features

## 7. Root Cause

The root cause is foreground-mask contamination from the support surface.

The live thresholding policy assumes the target foreground is materially darker than the local background. In this trace, the Defender is dark, but so are nearby sheet folds and shadow. Those support-surface pixels connect to the vehicle in the ROI and are accepted as foreground.

The foreground extraction diagnostics are:

| Diagnostic | Value |
| --- | ---: |
| `background_white_estimate` | `162` |
| `background_white_percentile` | `90` |
| `otsu_threshold` | `185` |
| `otsu_foreground_fraction` | `0.806006` |
| `relative_threshold` | `127` |
| `selected_threshold` | `127` |
| `selected_threshold_source` | `background_white_relative` |
| Foreground pixels before cleanup | `42,666` |
| Foreground pixels after cleanup | `44,792` |

The Otsu foreground fraction was just over the configured maximum of `0.80`, so the code selected the background-relative threshold. That choice avoided accepting almost the whole ROI, but it still accepted a large dark support-surface region. Morphological close and hole filling then produced a coherent foreground mask that was much larger than the vehicle.

Background removal was not active for regressor preprocessing in this trace:

```text
apply_background_removal_to_regressor_preprocessing = false
background_captured = false
```

Without background subtraction or a locator-relative foreground gate, the threshold path had no independent way to distinguish the Defender from dark sheet texture.

## 8. Why the Prediction Became Too Close

Monocular distance estimation is strongly tied to apparent scale. In this pipeline, apparent scale is represented twice:

- visually, through `x_distance_image`
- explicitly, through `x_geometry`

The v0.5 topology uses normalized geometry fields including `w_norm`, `h_norm`, `aspect_ratio`, and `area_norm` in the distance residual path. The relevant geometry features are constructed in the live preprocessor and consumed by the model topology.

The latest trace produced:

```text
w_norm = 0.158333
h_norm = 0.215833
area_norm = 0.034174
```

By comparison, the May 21 v0.5 trace-backed sweep recorded in incident 002 had foreground `area_norm` values roughly in the `0.0082` to `0.0235` range across accepted examples. The contaminated trace is therefore larger than the comparison set in the geometry feature most directly associated with apparent scale.

A local diagnostic replay over the saved model inputs reproduced the trace prediction. Replacing only `x_geometry` with locator-derived geometry raised the predicted distance from approximately `1.325 m` to approximately `1.568 m`. That replay is not a proposed production fix, because the distance and orientation images remain contaminated, but it demonstrates that the geometry expansion materially contributes to the underestimate.

## 9. Relationship to Earlier Incidents

This incident is technically distinct from incident 002's pose-dependent distance bias. Incident 002 is a broader model-representation and real/synthetic generalization issue. Incident 003 is a concrete preprocessing contamination failure in one live trace.

It is closely related to incident 001:

| Incident | Foreground failure | Distance effect |
| --- | --- | --- |
| Incident 001 | Foreground collapsed to a tiny fragment | Large overestimate |
| Incident 003 | Foreground expanded into sheet/shadow | Large underestimate |

Together, these incidents show that foreground quality is a primary operational risk for the direct tri-stream distance/yaw family. The model can be internally coherent while externally wrong if preprocessing gives it the wrong apparent scale.

## 10. Impact

The practical impact is a confident-looking live distance estimate that is too close. This is more subtle than a hard failure because the system still returns a plausible number and a plausible yaw.

The failure is most likely under these conditions:

- dark vehicle on a non-uniform light support surface
- visible folds, seams, or shadows near the target
- no background snapshot available for subtraction
- foreground extraction based primarily on grayscale thresholding
- ROI large enough to include adjacent support texture
- no post-foreground consistency check against locator geometry

The trace system limited the impact by preserving enough evidence to reconstruct the failure. Operationally, however, the runtime should prevent this class of corrupted input from reaching the model.

## 11. Proposed Remediation

### 11.1 Add a post-foreground quality gate

Add a guard after foreground extraction and before model inference. The guard should compare foreground geometry against locator geometry and reject or fall back when they disagree.

Suggested checks:

- foreground bbox width or height exceeds the locator bbox by more than a configured ratio
- foreground bbox area is several times larger than locator bbox area
- foreground centroid is displaced too far from locator center
- foreground pixel count is implausibly high relative to locator bbox area
- threshold diagnostics show high Otsu foreground fraction or weak background-white separation

For this trace, the guard would have fired:

```text
locator bbox:    150 x 117 px
foreground bbox: 304 x 259 px
foreground / locator bbox area ratio: 4.49 x
foreground pixels / locator bbox area ratio: 2.55 x
```

Failing closed is acceptable. A rejected frame with a clear preprocessing warning is better than a wrong distance estimate with no indication of degraded input quality.

### 11.2 Add a locator-anchored fallback path

If locator confidence is high but foreground extraction is over-expanded, the system can attempt a bounded fallback:

1. Define a padded locator gate, for example `1.25x` to `1.50x` around the accepted locator bbox.
2. Keep only thresholded connected components that intersect the locator gate.
3. Prefer the component or component group nearest the locator center.
4. Rebuild `x_distance_image`, `x_orientation_image`, and `x_geometry` from the gated mask.
5. Mark the result in metadata as a fallback, not as the primary path.

This would preserve the useful signal from thresholding while preventing distant support-surface texture from dominating the geometry.

### 11.3 Enable and operationalize background removal

The trace reports that background removal was not captured or applied. For a fixed-camera bounded-perception system, background subtraction is a natural control.

Recommended changes:

- make background capture a visible readiness state in the live GUI
- warn when running the threshold foreground path without a background snapshot
- allow regressor preprocessing to use background removal independently from the locator path
- capture validation traces with and without background removal for the same physical setup

Background removal will not solve every issue, but it directly targets support-surface contamination.

### 11.4 Improve foreground extraction beyond global thresholding

The current threshold path is simple and fast, but it is sensitive to support texture. A stronger foreground extractor should combine intensity with spatial and locator priors.

Candidate improvements:

- connected-component selection anchored to locator center
- shadow-tolerant foreground scoring using edge density and local contrast
- learned ROI segmentation trained on live-support surfaces
- explicit "support-surface contamination" synthetic augmentation
- a foreground confidence score reported alongside distance

The key design requirement is that foreground extraction should explain why a set of pixels is the vehicle, not merely why they are dark.

### 11.5 Continue the model-representation pivot

Incident 002 already argues for an amodal keypoint topology. Incident 003 reinforces that direction. A keypoint or amodal-geometry model can expose intermediate structure that is easier to validate than a scalar distance regressor.

This does not remove the need for clean preprocessing. It does reduce the chance that one contaminated foreground mask silently becomes the only explanation for distance.

## 12. Verification Plan

The remediation should be validated with fixture-backed tests and live traces.

### 12.1 Fixture tests

Use this trace as a regression fixture. A test should assert at least one of:

- the foreground quality gate rejects the saved trace
- the fallback path produces a foreground bbox that is close to the locator extent
- the trace does not produce an unqualified scalar prediction

The test should also assert that clean May 21 v0.5 traces from incident 002 are not rejected by the same guard.

### 12.2 Model-input replay

Record model-input-level replay tests for:

- original contaminated inputs
- locator-anchored fallback inputs
- background-removal-enabled inputs, once captured

The goal is not to tune the model to one trace. The goal is to prove that corrupted apparent-scale inputs are either corrected or rejected before inference.

### 12.3 Live validation

Run a controlled sweep after remediation:

- same support surface, same lighting, same camera setup
- background snapshot captured and not captured
- Defender at the suspect distance and at nearby distances
- front, side, and rear orientations

Acceptance criteria:

- no accepted trace has foreground bbox area more than a configured multiple of locator bbox area
- no accepted trace has foreground pixel count inconsistent with locator geometry
- rejected traces produce explicit preprocessing warnings
- distance predictions for accepted traces return to the expected tolerance band used in the failure-analysis framework

## 13. Recommended Implementation Order

| Priority | Work item | Rationale |
| --- | --- | --- |
| P0 | Add foreground-vs-locator quality gate | Low risk; prevents silent corrupted predictions |
| P0 | Add metadata and warning fields for foreground contamination | Makes future traces self-diagnosing |
| P1 | Add locator-anchored fallback extraction | Preserves predictions when contamination is recoverable |
| P1 | Add fixture tests using this trace | Prevents recurrence |
| P1 | Improve background-removal workflow in live GUI | Reduces support-surface contamination at source |
| P2 | Train or integrate a learned ROI foreground segmenter | More robust than global thresholding |
| P2 | Continue amodal keypoint topology evaluation | Improves interpretability of remaining distance errors |

## 14. Engineering Lessons

This incident is a useful example of disciplined ML systems debugging:

```text
camera frame: plausible
locator: plausible
ROI crop: plausible
foreground mask: contaminated
model inputs: corrupted
model output: coherent with corrupted inputs
```

The failure was not visible from the scalar prediction alone. It became clear because the live system preserved the accepted frame, ROI crop, foreground mask, model inputs, geometry vector, model output, and runtime metadata in one trace.

The operational lesson is that model-input validity must be treated as a first-class runtime contract. In a bounded monocular system, apparent scale is not just another feature. It is a core measurement channel. If preprocessing corrupts apparent scale, the model can be wrong for entirely deterministic reasons.

## 15. Appendix: Key Artifact Links

Primary trace:

- [`trace_manifest.json`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/trace_manifest.json)
- [`inference_result.json`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/inference_result.json)
- [`model_outputs.json`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/model_outputs.json)
- [`preprocessing_metadata.json`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/preprocessing_metadata.json)
- [`locator_result.json`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/locator_result.json)
- [`x_geometry.json`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/x_geometry.json)
- [`roi_crop.png`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/roi_crop.png)
- [`foreground_mask.png`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/foreground_mask.png)
- [`x_distance_image.png`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/x_distance_image.png)
- [`x_orientation_image.png`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859/x_orientation_image.png)

Related reports:

- [`incident-001-live-distance-regression-spike`](../incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md)
- [`incident-002-pose-dependent-distance-bias`](../incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md)
