# Retrospective Incident Report: ROI-FCN to Geometric Locator Pivot

**Incident:** `incident-004-roi-fcn-to-geometric-locator-retrospective`  
**System:** bounded monocular perception, live inference v0.2 to v0.3  
**Date analysed:** 2026-05-27  
**Status:** Retrospective; engineering justification supported by repository evidence

## 1. Executive Summary

Live inference v0.2 used ROI-FCN as the default live ROI locator. The locator converted a grayscale camera frame into a fixed locator canvas, ran a learned heatmap model, decoded the heatmap argmax to one source-image centre point, and extracted a fixed ROI crop around that point.

Live inference v0.3 changed the default locator to `background_edge_v1`, a deterministic geometric locator. ROI-FCN was retained only as `roi_fcn_legacy`, an explicit comparison/fallback path.

The repository state supports a solid engineering justification for that move. The reason was not simply that ROI-FCN was inaccurate. The deeper reason was that ROI-FCN produced an opaque single-point decision in a live system where downstream distance depends critically on apparent scale, crop boundaries, foreground quality, and support-surface contamination. When ROI-FCN failed, the runtime usually had only a heatmap confidence and a post-hoc crop guard. When downstream foreground extraction failed after an accepted ROI, ROI-FCN provided little additional structure to diagnose or recover.

The geometric locator is a better operational boundary for this fixed-camera system because it exposes the intermediate evidence the runtime needs:

- foreground mask
- edge map
- contour candidates
- chosen contour
- source-space bbox
- ROI crop bounds
- explicit rejection reasons
- traceable debug artifacts

This does not make the geometric locator a general object detector, and the v0.3 incidents show that foreground extraction remains a major risk. The engineering improvement is that v0.3 turns ROI selection into an inspectable, rejectable, and tunable step rather than a learned heatmap argmax whose failure evidence is hard to interpret live.

## 2. Evidence Base

This retrospective is based on repository artifacts only:

- v0.2 ROI-FCN implementation:
  - [`roi_locator.py`](../../../06_live-inference_v0.2/src/live_inference/preprocessing/roi_locator.py)
  - [`roi_fcn_locator.py`](../../../06_live-inference_v0.2/src/live_inference/preprocessing/roi_fcn_locator.py)
  - [`tri_stream_live_preprocessor.py`](../../../06_live-inference_v0.2/src/live_inference/preprocessing/tri_stream_live_preprocessor.py)
  - [`gui/app.py`](../../../06_live-inference_v0.2/src/live_inference/gui/app.py)
- v0.3 geometric locator implementation:
  - [`locators.py`](../../../06_live-inference_v0.3/src/live_inference/preprocessing/locators.py)
  - [`generic_tri_stream_live_preprocessor.py`](../../../06_live-inference_v0.3/src/live_inference/preprocessing/generic_tri_stream_live_preprocessor.py)
  - [`RUNTIME_NOTES.md`](../../../06_live-inference_v0.3/RUNTIME_NOTES.md)
- v0.2 live traces under [`06_live-inference_v0.2/live_traces`](../../../06_live-inference_v0.2/live_traces)
- v0.3 live traces under [`06_live-inference_v0.3/live_traces`](../../../06_live-inference_v0.3/live_traces)
- existing failure-analysis reports:
  - [`incident-001-live-distance-regression-spike`](../incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md)
  - [`incident-002-pose-dependent-distance-bias`](../incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md)
  - [`incident-003-foreground-mask-contamination-distance-underestimate`](../incident-003-foreground-mask-contamination-distance-underestimate/foreground-mask-contamination-distance-underestimate-report.md)
  - [`260415-1146_ds-2d-cnn`](../../model-evaluations/260415-1146_ds-2d-cnn.md)
  - [`260425-1025_ds-2d-cnn`](../../model-evaluations/260425-1025_ds-2d-cnn.md)

## 3. What v0.2 Did

v0.2 wired ROI-FCN into the app as the active locator. `gui/app.py` loads the selected ROI-FCN root, constructs `RoiFcnLocator`, and passes it into `TriStreamLivePreprocessor`.

The ROI-FCN pipeline is:

1. Decode the accepted frame to grayscale.
2. Apply runtime stage policy and optional masks.
3. Build the ROI-FCN locator input representation.
4. Resize the frame into a fixed locator canvas.
5. Run the learned ROI-FCN model.
6. Decode the heatmap argmax to a source-image centre.
7. Derive a fixed ROI box around that centre.
8. Extract a fixed canvas for downstream tri-stream preprocessing.
9. Reject only after the centre-derived ROI is checked for low confidence, clipping, or content.

The implementation exposes several live tuning points around the learned locator input:

- `as_is`
- `inverted`
- `sheet_dark_foreground`
- manual mask inclusion/exclusion
- background removal for ROI-FCN input

Those controls were useful for diagnostics, but they also reveal the central fragility: live ROI-FCN behavior depended on choosing an input representation that matched the learned model's assumptions.

## 4. What v0.3 Changed

v0.3 defaults to `background_edge_v1`. The runtime notes state that ROI-FCN is retained only as an explicit legacy comparison path.

The v0.3 locator pipeline is:

1. Decode the accepted frame to grayscale.
2. Optionally apply manual ignore mask.
3. Build a foreground mask from a captured background, or from a dark-on-light heuristic when no background is available.
4. Morphologically clean the foreground.
5. Run Canny edge detection inside foreground.
6. Build contour candidates.
7. Score candidates by area, extent, and edge density.
8. Choose the best accepted candidate.
9. Compute ROI geometry, clipping, content fraction, and explicit rejection reasons.
10. Write debug artifacts: grayscale frame, background diff, foreground mask, edge map, candidate overlay, chosen overlay, ROI crop, and `locator_result.json`.

The important architectural change is the contract. v0.3 uses a generic `LocatorResult` with locator kind, accepted flag, confidence, candidates, chosen bbox, ROI request/source/insert bounds, clipping metadata, warnings, and rejection reasons. ROI selection is no longer only a centre point plus heatmap metadata.

## 5. v0.2 Trace Findings

I parsed the checked-in v0.2 trace metadata as a population:

```text
trace directories with preprocessing metadata: 33
inference traces: 21
failure traces: 8
locator-only traces: 4
```

Failure breakdown:

```text
clipped ROI failures: 6 / 8
low-confidence failures: 2 / 8
```

The confidence signal was not a reliable operational health measure. In the checked-in v0.2 traces:

```text
all ROI-FCN confidence values: min 0.1868, median 0.5665, max 0.8862
accepted inference confidence: min 0.3255, median 0.4886, max 0.7818
failure confidence: min 0.1868, median 0.8479, max 0.8862
```

The rejected failures often had high confidence because the heatmap peak landed near or beyond the source-frame boundary. Examples:

| Trace prefix | Status | Mode | Confidence | Rejection |
| --- | --- | --- | ---: | --- |
| `20260513T114741Z` | failure | `inverted` | `0.8848` | `clipped_roi:110px>tolerance:0px` |
| `20260513T114744Z` | failure | `inverted` | `0.8848` | `clipped_roi:110px>tolerance:0px` |
| `20260513T114746Z` | failure | `inverted` | `0.8848` | `clipped_roi:110px>tolerance:0px` |
| `20260513T114810Z` | failure | `inverted` | `0.8862` | `clipped_roi:110px>tolerance:0px` |
| `20260513T114934Z` | failure | `as_is` | `0.8111` | `clipped_roi:118px>tolerance:0px` |
| `20260513T143009Z` | failure | `inverted` | `0.4880` | `clipped_roi:146px>tolerance:0px` |
| `20260517T113934Z` | failure | `as_is` | `0.1868` | `low_confidence:0.187<min:0.300` |
| `20260517T114146Z` | failure | `as_is` | `0.1968` | `low_confidence:0.197<min:0.300` |

The accepted traces also show why accepting the ROI was not sufficient. Among the 21 inference traces:

```text
foreground_pixel_count min/median/max: 119 / 17436 / 49907
predicted_distance_m min/median/max: 1.5922 / 1.9965 / 5.1837
```

The most obvious trace-level risk is `20260517T124435Z`, where ROI-FCN accepted the crop with confidence `0.359`, downstream foreground collapsed to `119` pixels, and the model predicted `5.1837 m`. That pattern matches the later formal incident-001 mechanism: the locator can be acceptable while the model input is corrupted by foreground collapse.

## 6. Failure-Analysis Linkage

The existing failure-analysis documents support the same conclusion from different angles.

The model-evaluation reports show that fixed ROI crop boundary handling was already a known high-severity risk:

- `260415-1146_ds-2d-cnn`: `3,009` crop-boundary samples; `1,067 / 1,068` samples with distance error `> 0.50 m` were crop-boundary cases.
- `260425-1025_ds-2d-cnn`: `3,009` crop-boundary samples; all `62` samples with distance error `> 0.50 m` were crop-boundary cases.

Incident 001 shows that a correct locator result is not enough. The accepted ROI contained the vehicle, but downstream silhouette recovery collapsed to a tiny fragment, producing a large distance overestimate. The report explicitly identifies that the acceptance guard was applied at the locator stage, not at model-input quality.

Incident 002 shows that the direct scalar distance/yaw model family remained pose-sensitive at fixed measured floor positions. That motivated a broader move toward more inspectable geometry, not merely more tuning of scalar regression.

Incident 003 shows the mirror-image foreground failure: the locator found a plausible vehicle-sized target, but foreground extraction expanded into support-surface texture and drove a distance underestimate. The current remediation is diagnostic/component-selection oriented rather than a brittle hard rejection. That depends on having locator bbox and foreground geometry available for comparison, which is exactly the kind of state ROI-FCN did not naturally expose.

## 7. Root Cause

The ROI-FCN locator was a poor operational boundary for live inference because it compressed ROI selection into a learned heatmap peak before the runtime had enough inspectable state to decide whether the crop was physically plausible.

The root cause is a boundary mismatch:

```text
ROI-FCN solves: choose one centre point from a learned heatmap
live runtime needs: choose, explain, reject, and repair a crop and apparent-scale representation
```

This mismatch created several concrete failure modes:

- High-confidence ROI-FCN peaks could create clipped crops.
- ROI-FCN confidence was not calibrated as a live health signal.
- The runtime had no candidate set to inspect when the selected centre was wrong.
- The locator did not provide object extent, only a crop centre.
- Foreground failures after ROI selection could corrupt `x_distance_image`, `x_orientation_image`, and `x_geometry` while the ROI-FCN result still looked accepted.
- Live behavior depended on input representation choices such as polarity and sheet-dark preprocessing.
- Compatibility metadata proved only artifact pairing, not live runtime behavior.

## 8. Why the Geometric Locator Was the Right Pivot

The geometric solution fits the bounded live setup better. This system has a fixed camera, a constrained scene, operator-controlled background capture, manual masks, and trace-driven diagnostics. Under those assumptions, deterministic image geometry is not a step backward from a learned locator; it is a better engineering interface.

The v0.3 locator directly exposes the evidence needed for live safety:

- no foreground can be rejected as `no_foreground`
- no viable contour can be rejected as `no_candidates`
- low candidate score can be rejected as `low_confidence`
- clipped ROI can be rejected as `roi_clipped`
- low ROI content can be rejected as `roi_content_too_low`
- chosen bbox can be compared to downstream foreground bbox
- candidate overlays can be inspected visually
- failures can be reproduced from trace artifacts

The move therefore reduces hidden learned state at the most safety-critical preprocessing boundary. The learned distance/yaw regressor still exists, but the crop-selection step becomes deterministic, inspectable, and easier to test.

## 9. Impact

The v0.2 ROI-FCN design made live failures harder to classify:

```text
preview frame: plausible
heatmap confidence: sometimes high
ROI crop: sometimes clipped or acceptable-looking
foreground/model input: may collapse or expand
distance output: plausible-looking but wrong
```

The v0.3 design does not eliminate all of those failures. It narrows the unknown part of the system. When the model input is corrupted, the trace now contains enough locator and foreground evidence to say why.

This is the main incident outcome:

```text
The project moved from learned ROI centre prediction to geometric ROI selection because the operational problem was not just "find a centre"; it was "produce an auditable apparent-scale measurement path."
```

## 10. Limitations

The repository does not contain a controlled replay where every v0.2 trace is run through both ROI-FCN and `background_edge_v1` against hand-labelled ROI centres. Therefore this report should not claim a quantified accuracy delta between the locators.

The repository does contain enough evidence for the engineering decision:

- v0.2 traces show high-confidence clipped ROI failures and accepted traces with severe downstream foreground collapse.
- v0.2 source shows ROI-FCN returns a centre-point location plus heatmap metadata.
- v0.3 source shows a generic locator contract with candidates, bbox, artifacts, and rejection reasons.
- failure-analysis reports show that crop boundaries, foreground collapse, foreground expansion, and opaque scalar regression were the recurring operational risks.

Additional experiments would be useful if the goal is a benchmark-style statement such as "geometric locator improves centre error by X pixels." They are not required to justify the architectural pivot.

## 11. Follow-Up Work

Recommended follow-up work:

| Priority | Work item | Rationale |
| --- | --- | --- |
| P0 | Keep ROI-FCN as explicit legacy comparison only | Preserves reproducibility without making it the default live boundary |
| P0 | Keep `background_edge_v1` trace artifacts mandatory for live diagnostics | These artifacts are the evidence that makes failures explainable |
| P0 | Preserve foreground-vs-locator consistency metadata | Incidents 001 and 003 show apparent-scale corruption is the main operational risk |
| P1 | Replay selected v0.2 traces through v0.3 locator | Useful for a quantified retrospective, not needed for the current decision |
| P1 | Add a trace-replay test set covering clipped ROI, foreground collapse, and foreground expansion | Converts the historical failures into regression fixtures |
| P1 | Improve background capture workflow and readiness warnings | Reduces support-surface contamination in the fixed-camera setup |
| P2 | Continue the inspectable geometry/keypoint model direction from incident 002 | Moves more of the model's inferred geometry into observable outputs |

## 12. Conclusion

The move from ROI-FCN to a geometric locator was justified by operational evidence, not by aesthetic preference.

ROI-FCN provided a learned heatmap centre and could work on many frames, but its live failure modes were hard to inspect and poorly aligned with the downstream risk: corrupted apparent scale. The geometric locator provides a better runtime contract for this bounded system because it makes the ROI decision observable, rejectable, tunable, and comparable to downstream foreground geometry.

No additional v0.2 experiment is required to write the retrospective incident report. Further replay experiments would improve the quantitative comparison, but the engineering justification is already present in the repository.
