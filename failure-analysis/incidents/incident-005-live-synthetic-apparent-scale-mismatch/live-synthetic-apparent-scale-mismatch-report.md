# Incident Report: Live/Synthetic Apparent-Scale Mismatch Causing Distance Underprediction

**Incident:** `incident-005-live-synthetic-apparent-scale-mismatch`  
**System:** bounded monocular perception, live inference v0.3  
**Date analysed:** 2026-06-01  
**Date updated:** 2026-06-02
**Status:** Investigated; apparent-scale mismatch hypothesis strongly supported; first follow-up sweep improved but residual underprediction remains

## 1. Executive Summary

After the live ROI path had moved to the geometric locator, a new distance error remained: accepted live predictions were consistently too close. The post-ROI-fix live sweep recorded six accepted distance readings. Five clean or clean-ish readings underpredicted measured distance by approximately `0.35 m` to `0.40 m`. One additional `2.9 m -> 2.008 m` reading was treated as contaminated and excluded from the clean bias estimate.

An independent synthetic/live image-pair analysis then compared the apparent size of the Defender in nominally matched synthetic and live captures. Across eight front/side image pairs, the live vehicle appeared consistently larger than the synthetic vehicle at the same nominal lens distance. Using simple inverse scale geometry, that visual-scale mismatch predicts a mean apparent-distance offset of `-0.336 m`, with a median of `-0.331 m` and a range from `-0.283 m` to `-0.406 m`.

The clean live sweep mean error is `-0.364 m`. The image-pair scale analysis predicts `-0.336 m`. The difference between those two independently derived values is only `0.028 m`.

The incident therefore strongly supports the hypothesis that the live model input presents the Defender as visually larger, and therefore apparently closer, than the synthetic training representation. This does not prove one exact low-level cause. The mismatch could still be split between Unity camera parameters, the real-to-Unity intrinsics mapping, viewport/capture handling, lens model mismatch, synthetic object scale, or physical measurement reference differences. The engineering conclusion is narrower and stronger: once locator and foreground failures are controlled, live/synthetic apparent-scale alignment becomes a primary remaining distance-risk boundary.

A first follow-up live sweep is now recorded in this report. Across eight measured front/side rows, mean signed error improved to `-0.113 m` and mean absolute error improved to `0.118 m`. Excluding the slightly clipped `1.59 m` front row, mean signed error was `-0.103 m` and mean absolute error was `0.109 m`. This is a material improvement from the original clean-sweep mean signed error of `-0.364 m`, but it is not a final calibrated live-accuracy claim: residual underprediction remains, especially in the near-range and side-view rows.

## 2. Incident Scope

This report covers three evidence sources captured during the Incident 005 investigation:

1. A post-ROI-fix live sweep summarised in the incident observation note.
2. An eight-pair synthetic/live apparent-scale analysis using manually measured bounding boxes.
3. A follow-up eight-row front/side live sweep recorded after the apparent-scale mitigation work.

The report is intentionally limited to distance bias from apparent-scale mismatch and the first follow-up distance results. It does not re-litigate earlier locator and foreground-mask incidents except where those incidents explain why this failure is different.

The staged repository output includes the eight scale-pair summary comparison images under [`evidence/scale-pairs`](evidence/scale-pairs). The local incident workspace also contains raw image pairs and live-inference trace artifacts captured on `2026-05-31`; those optional heavier artifacts are described in [`evidence/evidence-manifest.md`](evidence/evidence-manifest.md). The follow-up sweep summary is recorded in this report; no raw trace bundle for that follow-up sweep is staged yet.

## 3. Expected Behaviour

For a bounded fixed-camera perception system trained primarily on synthetic imagery, synthetic and live captures at the same nominal lens-to-target distance should produce comparable apparent target scale after preprocessing.

The practical expectation is not perfect metrology. Manual placement, camera calibration, physical scale, and lens modelling all introduce tolerance. However, the direction and magnitude of the observed error are too structured to treat as random measurement noise:

```text
expected: live apparent scale ~= synthetic apparent scale
observed: live apparent scale > synthetic apparent scale
effect:   model predicts the live target closer than the measured reference distance
```

The project's failure-analysis framework treats `0.10 m` as the primary useful distance boundary and `0.05 m` as a stricter clean-success boundary. A recurring `0.35 m` to `0.40 m` signed bias is therefore a material system-level failure, even if individual predictions look numerically plausible.

## 4. Evidence Base

The local incident record contains these source notes:

- `Incident Issue Observation.md`: six-trace live sweep summary.
- `Image Analysis Results.md`: eight-pair visual-scale calculation.
- `Outcome Evidence Statement.md`: combined interpretation and root-cause caution.
- Follow-up live sweep summary supplied on 2026-06-02: eight front/side readings after the apparent-scale mitigation work.

The relevant repository context is:

- Incident 001: live distance overestimate from foreground/silhouette collapse.
- Incident 002: pose-dependent distance bias in the direct distance/yaw model family.
- Incident 003: live distance underestimate from foreground-mask contamination.
- Incident 004: retrospective justification for replacing ROI-FCN with the geometric locator.

Incident 005 is downstream of those findings. It asks what remains after the ROI boundary is more inspectable and the immediate foreground-mask failure class is no longer the main explanation.

## 5. Live Sweep Findings

### 5.1 Original post-ROI-fix sweep

The live sweep summary recorded six accepted readings after the ROI fix:

| Sample | Measured reference | Predicted distance | Signed error | ROI source | Locator size | Inclusion |
| ---: | ---: | ---: | ---: | --- | ---: | --- |
| 1 | `1.6 m` | `1.236 m` | `-0.364 m` | `foreground_component` | `204 x 317 px` | clean-ish |
| 2 | `1.6 m` | `1.249 m` | `-0.351 m` | `foreground_component` | `352 x 250 px` | clean-ish |
| 3 | `2.0 m` | `1.637 m` | `-0.363 m` | `foreground_component` | `138 x 208 px` | clean-ish |
| 4 | `2.0 m` | `1.655 m` | `-0.345 m` | `foreground_component` | `231 x 159 px` | clean-ish |
| 5 | `2.9 m` | `2.501 m` | `-0.399 m` | `foreground_component` | `91 x 134 px` | clean-ish |
| 6 | `2.9 m` | `2.008 m` | `-0.892 m` | `foreground_component` | `181 x 191 px` | contaminated/outlier |

Clean or clean-ish trace summary, excluding the contaminated outlier:

| Metric | Value |
| --- | ---: |
| Included readings | `5` |
| Mean signed error | `-0.364 m` |
| Median signed error | `-0.363 m` |
| Signed error range | `-0.345 m` to `-0.399 m` |
| Mean absolute error | `0.364 m` |

Including the contaminated outlier would produce a mean signed error of `-0.452 m`, but that is not the right incident summary. The useful signal is the recurring clean-trace underprediction around `-0.35 m` to `-0.40 m`.

### 5.2 Follow-up live sweep

A later eight-row front/side sweep recorded a materially smaller negative bias:

| Measured reference | Orientation | Predicted distance | Signed error | Note |
| ---: | --- | ---: | ---: | --- |
| `1.59 m` | `0 deg / front` | `1.41 m` | `-0.18 m` | slight ROI clipping contamination |
| `1.59 m` | `90 deg / side` | `1.35 m` | `-0.24 m` | underpredicting |
| `1.77 m` | `0 deg / front` | `1.65 m` | `-0.12 m` | improved |
| `1.77 m` | `90 deg / side` | `1.56 m` | `-0.21 m` | underpredicting |
| `1.97 m` | `0 deg / front` | `1.99 m` | `+0.02 m` | good |
| `1.97 m` | `90 deg / side` | `1.93 m` | `-0.04 m` | good |
| `2.18 m` | `0 deg / front` | `2.13 m` | `-0.05 m` | good |
| `2.18 m` | `90 deg / side` | `2.10 m` | `-0.08 m` | good |

Follow-up summary:

| Population | Mean signed error | Mean absolute error |
| --- | ---: | ---: |
| All rows | `-0.113 m` | `0.118 m` |
| Excluding slightly clipped `1.59 m` front row | `-0.103 m` | `0.109 m` |

This is a material improvement over the original clean-sweep mean signed error of `-0.364 m` and mean absolute error of `0.364 m`. It does not fully close the live-distance issue: the remaining bias is still mostly negative, `4 / 8` rows remain outside the `0.10 m` distance threshold, and the near-range side readings are still underpredicting.

## 6. Synthetic/Live Scale Analysis

The second evidence source compared the apparent bounding-box size of the Defender in synthetic and live captures at nominally matched lens distances.

The calculation used:

```text
apparent distance = nominal lens distance / live-to-synthetic scale ratio
offset = apparent distance - nominal lens distance
```

A negative offset means the live image makes the target appear closer or larger than the matched synthetic image.

| Pair | Point | Nominal distance | Orientation | Synthetic bbox | Live bbox | Width scale | Height scale | Width-implied distance | Width offset | Height-implied distance | Height offset | Mean implied distance | Mean offset |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | `1.59 m` | Front | `113 x 234 px` | `147 x 277 px` | `1.301x` | `1.184x` | `1.222 m` | `-0.368 m` | `1.343 m` | `-0.247 m` | `1.283 m` | `-0.307 m` |
| 2 | 1 | `1.59 m` | Side | `227 x 144 px` | `282 x 191 px` | `1.242x` | `1.326x` | `1.280 m` | `-0.310 m` | `1.199 m` | `-0.391 m` | `1.239 m` | `-0.351 m` |
| 3 | 2 | `1.77 m` | Front | `99 x 196 px` | `130 x 225 px` | `1.313x` | `1.148x` | `1.348 m` | `-0.422 m` | `1.542 m` | `-0.228 m` | `1.445 m` | `-0.325 m` |
| 4 | 2 | `1.77 m` | Side | `202 x 127 px` | `241 x 160 px` | `1.193x` | `1.260x` | `1.484 m` | `-0.286 m` | `1.405 m` | `-0.365 m` | `1.444 m` | `-0.326 m` |
| 5 | 3 | `1.97 m` | Front | `90 x 169 px` | `110 x 189 px` | `1.222x` | `1.118x` | `1.612 m` | `-0.358 m` | `1.762 m` | `-0.208 m` | `1.687 m` | `-0.283 m` |
| 6 | 3 | `1.97 m` | Side | `184 x 115 px` | `215 x 143 px` | `1.168x` | `1.243x` | `1.686 m` | `-0.284 m` | `1.584 m` | `-0.386 m` | `1.635 m` | `-0.335 m` |
| 7 | 4 | `2.18 m` | Front | `82 x 149 px` | `107 x 173 px` | `1.305x` | `1.161x` | `1.671 m` | `-0.509 m` | `1.878 m` | `-0.302 m` | `1.774 m` | `-0.406 m` |
| 8 | 4 | `2.18 m` | Side | `167 x 102 px` | `193 x 126 px` | `1.156x` | `1.235x` | `1.886 m` | `-0.294 m` | `1.765 m` | `-0.415 m` | `1.826 m` | `-0.354 m` |

Aggregate scale summary:

| Metric | Value |
| --- | ---: |
| Mean width scale | `1.238x` |
| Mean height scale | `1.210x` |
| Mean width-derived offset | `-0.354 m` |
| Mean height-derived offset | `-0.318 m` |
| Mean combined offset | `-0.336 m` |
| Median combined offset | `-0.331 m` |
| Combined offset range | `-0.283 m` to `-0.406 m` |

The scale-analysis offset is not merely directionally consistent with the live sweep. It is numerically close enough to explain most of the clean live error.

| Evidence source | Key result | Interpretation |
| --- | ---: | --- |
| Original clean live sweep | mean signed error `-0.364 m` | Model predicts target too close |
| Follow-up live sweep | mean signed error `-0.113 m`; MAE `0.118 m` | Materially improved but residual underprediction remains |
| Synthetic/live scale comparison | mean apparent-distance offset `-0.336 m` | Live target appears larger than synthetic equivalent |
| Difference between means | `0.028 m` | Independent evidence paths converge |

### 6.1 Image Evidence

The staged evidence set includes one summary comparison image for each row of the scale table:

| Pair | Orientation | Image evidence |
| ---: | --- | --- |
| 1 | Front | [`pair1_front_summary_comparison.png`](evidence/scale-pairs/pair1_front_summary_comparison.png) |
| 2 | Side | [`pair2_side_summary_comparison.png`](evidence/scale-pairs/pair2_side_summary_comparison.png) |
| 3 | Front | [`pair3_front_summary_comparison.png`](evidence/scale-pairs/pair3_front_summary_comparison.png) |
| 4 | Side | [`pair4_side_summary_comparison.png`](evidence/scale-pairs/pair4_side_summary_comparison.png) |
| 5 | Front | [`pair5_front_summary_comparison.png`](evidence/scale-pairs/pair5_front_summary_comparison.png) |
| 6 | Side | [`pair6_side_summary_comparison.png`](evidence/scale-pairs/pair6_side_summary_comparison.png) |
| 7 | Front | [`pair7_front_summary_comparison.png`](evidence/scale-pairs/pair7_front_summary_comparison.png) |
| 8 | Side | [`pair8_side_summary_comparison.png`](evidence/scale-pairs/pair8_side_summary_comparison.png) |

## 7. Pipeline Reconstruction

The relevant live inference path is the v0.3 tri-stream pipeline:

1. Decode accepted camera frame.
2. Apply the manual mask and live preprocessing policy.
3. Locate a vehicle-centred ROI using the geometric locator, typically `background_edge_v1`.
4. Extract a fixed ROI crop.
5. Produce foreground-derived model inputs:
   - `x_distance_image`
   - `x_orientation_image`
   - `x_geometry`
6. Run the selected direct distance/yaw model, currently represented in the local trace artifacts by `260521-1029_ts-2d-cnn`.

The important feature of this incident is that the locator can be working in the operational sense while the apparent-scale contract is still wrong. A correct or plausible crop does not guarantee that the live target appears at the same scale as the synthetic training target for the same physical distance.

The direct distance regressor is expected to use apparent scale heavily. That is not a defect by itself. Monocular distance estimation in this bounded setup depends on image scale. The failure arises when the training domain and live domain disagree on what scale corresponds to what distance.

## 8. Root Cause

The proximate root cause is live/synthetic apparent-scale mismatch.

At nominally matched lens-to-target distances, the live Defender occupies a larger bounding box than the synthetic Defender. The model trained on synthetic imagery therefore receives live inputs that look closer than the measured reference distance. Its underprediction is coherent with the scale it sees.

The root-cause family is:

```text
synthetic/live camera and scene geometry are not aligned tightly enough
for direct scalar distance regression to transfer cleanly
```

The exact low-level cause is not fully isolated. Plausible contributors include:

- Unity camera field-of-view or projection parameters.
- Real-camera intrinsics or undistortion remap.
- Viewport, capture, or resize behaviour between Unity and live preprocessing.
- Lens model mismatch not captured by the current correction path.
- Synthetic Defender model scale.
- Physical distance reference point mismatch between lens, target front, target centre, and synthetic object origin.
- Manual placement and bbox measurement tolerances.

The incident should therefore be framed as "supports the apparent-scale mismatch hypothesis", not as proof of one particular camera calibration bug.

## 9. Why This Produces Distance Underprediction

For a fixed target, apparent image size is approximately inverse with distance. If the live target is `1.20x` to `1.24x` larger than the synthetic target at the same nominal distance, a synthetic-trained model can reasonably interpret the live target as closer.

The image-pair calculation makes that relationship explicit:

```text
nominal distance:   1.59 m
live/synthetic scale: 1.301x width, 1.184x height
mean implied distance: 1.283 m
mean offset: -0.307 m
```

That pattern repeats across all eight pairs. It is not tied to one pose or one measurement point. The combined offset range, `-0.283 m` to `-0.406 m`, closely overlaps the clean live sweep error range, `-0.345 m` to `-0.399 m`.

This explains why the failure remains after obvious ROI failures are addressed. The model can receive a coherent, vehicle-shaped input and still produce a systematically wrong distance if the live apparent scale is shifted relative to the synthetic training distribution.

## 10. Relationship to Earlier Incidents

Incident 001 showed a large live overestimate caused by foreground/silhouette collapse. The model was not wrong in isolation; it responded to a tiny corrupted target representation.

Incident 003 showed the mirror failure: foreground contamination expanded the target representation and caused a live underestimate.

Incident 004 justified moving from ROI-FCN to a geometric locator because live ROI selection needed inspectable geometry, candidate evidence, and explicit rejection reasons.

Incident 005 is different. It shows that even when the ROI and foreground path are plausible enough to produce accepted readings, the synthetic/live camera-scale contract can still bias the direct distance regressor. This is a cross-domain calibration failure, not simply a locator or foreground cleanup failure.

It also strengthens the architectural lesson from Incident 002: scalar distance/yaw regression is useful as a baseline and runtime integration path, but it hides too much geometry when the remaining error is domain alignment, apparent scale, pose, and projection.

## 11. Impact

The practical impact is a plausible-looking live distance output that is systematically too close.

This is a higher-risk class than a hard preprocessing failure because the runtime still returns a normal number:

```text
ROI: accepted
ROI source: foreground_component
prediction: plausible
signed error: consistently negative
```

Without the paired scale analysis, the failure could be misread as model weakness, pose sensitivity, or noisy manual measurement. The image-pair evidence narrows it: a large share of the observed distance error is explainable before considering model internals.

The incident does not invalidate the geometric locator pivot. It clarifies the next boundary. The locator can make the crop path auditable, but the system also needs a calibrated synthetic/live projection contract.

The follow-up sweep shows that the mitigation direction is useful: the large original negative bias is materially reduced. The impact is not eliminated, because several near-range readings still underpredict by more than the `0.10 m` failure threshold and one row is explicitly noted as slightly clipped.

## 12. Remediation Strategy

### 12.1 Treat apparent-scale calibration as a P0 validation gate

Before claiming improved live distance accuracy from the direct distance regressor, the project should require a synthetic/live scale check at measured reference positions.

Minimum gate:

- same camera, lens, resolution, and preprocessing path used by live inference
- same synthetic camera parameters and resize/crop path used by training or evaluation
- front and side orientations at each measured mark
- measured target bbox in both domains
- live/synthetic width scale, height scale, and implied distance offset reported

This gate should be stored as a small table or CSV, not only as screenshots.

### 12.2 Isolate the low-level geometry source

The next debugging pass should test likely contributors one at a time:

| Candidate cause | Test |
| --- | --- |
| Unity field-of-view mismatch | Render a calibration target at known synthetic distances and compare expected pixel size |
| Intrinsics/remap mismatch | Compare raw, undistorted, and Unity-equivalent captures against the same bbox metric |
| Viewport/capture resizing | Verify source resolution, aspect ratio, crop bounds, and any letterbox/pad behaviour |
| Synthetic object scale | Render the synthetic Defender against a known calibration object or object-dimension reference |
| Distance reference mismatch | Define whether distance is lens-to-front, lens-to-centre, or lens-to-object-origin and use one convention |

### 12.3 Add scale-ratio regression fixtures

Once the evidence images are copied into the repository, add a lightweight scale-ratio regression fixture:

```text
input: matched synthetic/live image pair
output: synthetic bbox, live bbox, width scale, height scale, implied distance offset
assertion: scale offset remains within chosen tolerance after calibration changes
```

The first version can be manual or script-assisted. The important point is to stop treating visual scale as an informal observation.

### 12.4 Repeat live sweeps after apparent-scale mitigation

A first follow-up sweep is now recorded in Section 5.2. It reduced all-row mean signed error to `-0.113 m` and mean absolute error to `0.118 m`; excluding the slightly clipped `1.59 m` front row gives mean signed error `-0.103 m` and mean absolute error `0.109 m`.

The next requirement is repeatability and trace-backed evidence, not merely one improved summary table. Future sweeps should preserve:

- the same measured marks, including `1.59 m`, `1.77 m`, `1.97 m`, and `2.18 m`
- front and side orientations
- trace capture enabled
- raw frame, locator result, ROI crop, foreground mask, `x_distance_image`, `x_orientation_image`, `x_geometry`, and model output retained
- explicit notes for clipping, support-surface contamination, or manual-mask changes

The reportable metric should include both continuous and thresholded results:

- mean signed error
- mean absolute error
- RMSE
- median absolute error
- maximum absolute error
- count within `0.10 m`
- count within `0.05 m`

The key acceptance question is now whether the residual negative bias is repeatable, whether it can be reduced below the `0.10 m` failure boundary across near and far marks, and whether the apparent-scale correction remains stable across front and side views.

### 12.5 Keep direct regression claims bounded

After the first follow-up sweep, direct distance/yaw regression should be framed as:

```text
traceable live-runtime integration with initial apparent-scale mitigation evidence
not yet a calibrated live distance-estimation claim
```

That is still valuable. The incident shows engineering discipline: the system can produce artifacts and follow-up measurements that identify why a plausible live output is wrong and whether a mitigation materially improves it.

## 13. Verification Plan

The recommended verification plan is:

| Priority | Work item | Success criterion |
| --- | --- | --- |
| P0 | Copy image-pair and trace evidence into the repository incident folder | Report links resolve and evidence is reviewable |
| P0 | Define one distance-reference convention | Measurements and synthetic labels use the same reference point |
| P0 | Script the scale comparison table | Manual bbox arithmetic is replaced by reproducible calculation |
| P0 | Re-render synthetic matched views after calibration changes | Mean apparent-distance offset moves materially toward zero |
| P0 | Record first follow-up live sweep | Done in this report: all-row mean signed error `-0.113 m`, MAE `0.118 m` |
| P0 | Repeat live distance sweep with trace capture | Residual signed error is stable, traceable, and no longer exceeds the `0.10 m` boundary across near and far marks |
| P1 | Add scale fixtures to tests or analysis scripts | Future camera/render changes cannot silently reintroduce the mismatch |
| P1 | Compare direct regressor against the amodal/keypoint direction | Remaining failures are evaluated with more inspectable geometry |

## 14. Limitations

This incident report is intentionally conservative.

The live sweep distances are practical measured references, not laboratory metrology. Manual placement, target orientation, and reference-point convention can move centimetre-level results. They do not plausibly explain the repeated `0.35 m` to `0.40 m` signed bias on their own.

The image-pair analysis uses 2D bbox measurements. Bounding boxes are a useful proxy for apparent scale, but they are not a full camera calibration. Pose, occlusion, perspective, and bbox measurement judgement can affect the exact width and height values.

The contaminated `2.9 m -> 2.008 m` live reading is excluded from the clean bias estimate. That exclusion is appropriate for estimating the recurring scale-linked bias, but the trace should still be preserved because it may represent another recoverable preprocessing or support-surface failure.

The follow-up sweep is currently recorded as a numerical summary rather than a staged trace bundle or CSV. One row is explicitly marked as slightly affected by ROI clipping contamination, and several near-range rows remain outside the `0.10 m` distance threshold. The follow-up therefore supports material improvement, not closure of the incident as a calibrated live accuracy claim.

The report does not claim that scale mismatch is the only remaining live issue. It claims that scale mismatch is now strongly evidenced and large enough to explain most of the clean underprediction observed in this incident.

## 15. Engineering Lessons

This incident is a useful example of why bounded ML systems need domain-contract checks, not only model metrics.

The distance regressor can behave coherently and still be wrong if the synthetic/live projection contract is wrong. The incident evidence follows a clean chain:

```text
original live sweep: recurring negative distance bias
image-pair analysis: live target appears larger than synthetic target
inverse-scale estimate: predicted offset almost matches live bias
follow-up sweep: bias materially reduced but residual underprediction remains
engineering outcome: validate apparent scale before making stronger live claims
```

The strongest part of the investigation is that the main explanatory calculation is simple. It does not depend on a complex post-hoc neural-network interpretation. A larger image of a fixed-size object implies a closer apparent distance; the measured live/synthetic size ratio predicts almost the same error the live model produced.

## 16. Conclusion

Incident 005 identifies a likely synthetic-to-live projection mismatch in the bounded monocular perception system.

The post-ROI-fix live sweep showed clean-trace underprediction around `-0.35 m` to `-0.40 m`. The independent synthetic/live bbox comparison predicted an apparent-distance offset of about `-0.34 m`. Those two paths agree closely enough to make apparent-scale mismatch the leading explanation.

The immediate outcome is not another locator patch. The first follow-up sweep suggests the apparent-scale mitigation direction is useful: all-row mean signed error improved to `-0.113 m` and mean absolute error to `0.118 m`; excluding the slightly clipped `1.59 m` front row gives `-0.103 m` mean signed error and `0.109 m` mean absolute error.

That improvement is material, but it is not closure. The next engineering step is a calibrated synthetic/live scale-validation loop with repeat trace-backed live sweeps, staged artifacts, and scripted scale fixtures. Until that is done, the direct distance/yaw model remains useful as a baseline and runtime evidence path, but not as a calibrated live distance claim.

## 17. Appendix: Key Artifact Links

Recommended evidence layout after repository copy:

- [`evidence/scale-pairs/`](evidence/scale-pairs/): eight synthetic/live pair summary comparison images.
- [`evidence/evidence-manifest.md`](evidence/evidence-manifest.md): included evidence list and optional full-artifact copy map, including the heavier live-inference traces that were not staged by default.
- Section 5.2 of this report: first follow-up live sweep numerical summary; raw trace bundle and CSV are not staged yet.

Suggested related reports:

- [`incident-001-live-distance-regression-spike`](../incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md)
- [`incident-002-pose-dependent-distance-bias`](../incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md)
- [`incident-003-foreground-mask-contamination-distance-underestimate`](../incident-003-foreground-mask-contamination-distance-underestimate/foreground-mask-contamination-distance-underestimate-report.md)
- [`incident-004-roi-fcn-to-geometric-locator-retrospective`](../incident-004-roi-fcn-to-geometric-locator-retrospective/roi-fcn-to-geometric-locator-retrospective-report.md)
