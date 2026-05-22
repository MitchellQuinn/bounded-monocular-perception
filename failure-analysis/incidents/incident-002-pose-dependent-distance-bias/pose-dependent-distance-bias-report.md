# Incident Report: Pose-Dependent Distance Bias in Live Monocular Distance Regression

## Summary

This incident investigated a repeatable live-camera distance regression error in Project Raccoon Ball, a bounded monocular perception system for estimating vehicle distance and yaw from a fixed camera view. The system is intentionally scoped around a known vehicle, constrained camera geometry, synthetic supervision, runtime preprocessing, live inference, trace capture, and failure analysis.

The failure mode was identified during real-camera testing of the current tri-stream distance/yaw model family. At fixed measured floor positions, predicted distance varied systematically with vehicle pose. Front-facing views often predicted farther than side or rear views at the same floor mark, rear-facing views often predicted closer, and side-facing views were usually intermediate or closest to the measured reference distance.

The initial investigation tested whether the issue was primarily caused by camera-model mismatch between Unity synthetic camera geometry and the real AR0234 camera. Applying an input-space camera-model correction modestly improved aggregate distance error, but did not resolve the pose-dependent bias.

A later trace-backed comparison between TriStream v0.4 and TriStream v0.5 showed that v0.5 did not cleanly solve the live failure mode. In the recorded rerun, mean absolute error changed only slightly, from `0.1105 m` for v0.4 to `0.1074 m` for v0.5. v0.5 improved the strict `5 cm` count from `1 / 12` to `3 / 12`, but RMSE, maximum error, and the `10 cm` count did not improve. The v0.5 rerun also shifted the overall signed error strongly negative.

The incident outcome remains an architectural pivot. The current direct distance/yaw tri-stream family remains useful as a baseline and live-runtime integration path, but it is no longer the primary route for improving the system. The next model-development direction is the already-defined amodal keypoint topology, which is documented separately. This report does not reproduce that topology; it records the incident evidence that motivates the pivot.

---

## 1. System Context

Project Raccoon Ball is a bounded applied ML and computer-vision project. Its central task is to estimate the distance and yaw of a known vehicle from fixed-camera imagery under controlled conditions. The repository is not intended to demonstrate general object detection, autonomous driving, open-world scene understanding, or unconstrained real-world perception.

The current live inference path uses a tri-stream representation:

```text
x_distance_image
x_orientation_image
x_geometry
```

The tri-stream design separates distance evidence, orientation evidence, and explicit geometry features. The distance stream preserves apparent scale, the orientation stream provides a target-centred orientation view, and the geometry vector carries bounding-box and foreground-shape metadata.

The live runtime includes model selection, compatibility checks, deterministic and learned localisation paths, foreground extraction, trace capture, debug artifact handling, and PySide6 GUI controls.

---

## 2. Incident Objective

The incident had three practical objectives:

1. Determine whether live distance errors were primarily caused by camera-model mismatch between Unity and the real AR0234 camera.

2. Compare the current direct distance/yaw tri-stream model family across available model versions.

3. Decide whether further work should continue within the current direct-regression model family or move to a more inspectable representation.

The central diagnostic question was:

> At the same measured floor position, does predicted distance remain stable across front, side, and rear vehicle poses?

Distance should be broadly pose-invariant. The vehicle's yaw changes, but its position relative to the camera does not.

---

## 3. Measurement Method

The Defender model was placed manually on measured floor marks on a white hardboard surface. Marks were measured with a tape measure.

Four usable measured positions were tested:

|Mark|Measured distance|
|--:|--:|
|1|1.59 m|
|2|1.77 m|
|3|1.97 m|
|4|2.18 m|

A fifth mark at `2.39 m` was excluded because the Defender clipped at the top of the frame.

At each mark, three orientations were tested:

```text
front-facing
side-facing
rear-facing
```

The measured distances should be treated as practical reference distances, not calibrated ground truth. Manual placement, vehicle footprint, and the difference between physical floor marks and the Unity object-position target introduce tolerance. This limitation affects absolute precision, but it does not explain repeated pose-linked distance differences at the same floor mark.

For the v0.5 trace-backed rerun, the trace artifacts themselves do not encode human mark and pose labels. The mark assignment follows chronological sweep order, and pose assignment was verified from the recorded ROI crops:

```text
1.59 m: front, side, rear
1.77 m: front, side, rear
1.97 m: front, side, rear
2.18 m: front, side, rear
```

---

## 4. Evaluation Criteria

The project's failure-analysis framework uses `10 cm` as the primary distance success boundary and `5 cm` as a stricter clean-success boundary. The same framework recommends reporting both continuous metrics and thresholded categories, rather than relying on one headline metric.

For this incident, the most relevant metrics are:

```text
mean absolute error
RMSE
median absolute error
maximum absolute error
samples within 10 cm
samples within 5 cm
pose spread at fixed measured distance
```

Pose spread is especially important. It measures the difference between the highest and lowest predicted distance at the same floor mark across front, side, and rear views.

---

## 5. Expected Behaviour

At each measured mark, predicted distance should remain approximately stable across vehicle orientation.

Expected pattern:

```text
front prediction ~= side prediction ~= rear prediction
```

Observed pattern:

```text
predictions remain pose-dependent at fixed measured distance
front-facing views are often highest
rear-facing views are often lowest
```

The repeated pose-linked spread is the core failure mode.

---

## 6. Camera-Model Correction Test

The first phase tested whether the live-camera distance error was primarily caused by camera-model mismatch.

A camera-model correction was applied upstream of inference using AR0234 calibration data and an equivalent Unity camera model. The correction was applied to the input frame before normal localisation, preprocessing, geometry extraction, and model inference.

Three sweeps were collected:

|Session|Description|
|---|---|
|A|Baseline before camera-model correction|
|B|Baseline repeat before camera-model correction|
|C|Sweep after camera-model correction|

### 6.1 Aggregate Results

|Metric|Baseline A|Baseline B|Camera-corrected C|
|---|--:|--:|--:|
|Mean absolute error|0.1275 m|0.1267 m|**0.1058 m**|
|RMSE|0.1552 m|0.1567 m|**0.1394 m**|
|Median absolute error|0.1000 m|0.1200 m|**0.0750 m**|
|Average pose spread|0.2275 m|0.1825 m|0.2425 m|

### 6.2 Interpretation

The camera-model correction improved aggregate distance error, but did not reduce the main pose-dependent spread. The corrected sweep retained consistent front/side/rear divergence.

This indicates that camera-model mismatch contributed to live error, but was not the dominant remaining cause.

---

## 7. TriStream v0.4 Trace-Backed Sweep

A later sweep was run against TriStream v0.4 with camera intrinsics applied. This sweep used single-frame inference and trace recording.

One `1.97 m / rear` reading was rejected because the ROI locator result was visibly unsuitable. The accepted replacement reading is used below.

|Sample|Mark|Pose|Predicted distance|Error|
|--:|--:|---|--:|--:|
|V0.4-T-001|1.59 m|Front|1.740 m|+0.150 m|
|V0.4-T-002|1.59 m|Side|1.681 m|+0.091 m|
|V0.4-T-003|1.59 m|Rear|1.664 m|+0.074 m|
|V0.4-T-004|1.77 m|Front|2.014 m|+0.244 m|
|V0.4-T-005|1.77 m|Side|1.823 m|+0.053 m|
|V0.4-T-006|1.77 m|Rear|1.840 m|+0.070 m|
|V0.4-T-007|1.97 m|Front|2.036 m|+0.066 m|
|V0.4-T-008|1.97 m|Side|2.004 m|+0.034 m|
|V0.4-T-009|1.97 m|Rear|1.913 m|-0.057 m|
|V0.4-T-010|2.18 m|Front|2.074 m|-0.106 m|
|V0.4-T-011|2.18 m|Side|2.067 m|-0.113 m|
|V0.4-T-012|2.18 m|Rear|1.912 m|-0.268 m|

### 7.1 v0.4 Metrics

|Metric|Value|
|---|--:|
|Mean absolute error|0.1105 m|
|RMSE|0.1317 m|
|Mean signed error|+0.0198 m|
|Median absolute error|0.0825 m|
|Maximum absolute error|0.2680 m|
|Samples within 10 cm|7 / 12|
|Samples within 5 cm|1 / 12|

### 7.2 v0.4 Pose Spread

|Mark|Front|Side|Rear|Spread|
|--:|--:|--:|--:|--:|
|1.59 m|1.740 m|1.681 m|1.664 m|0.076 m|
|1.77 m|2.014 m|1.823 m|1.840 m|0.191 m|
|1.97 m|2.036 m|2.004 m|1.913 m|0.123 m|
|2.18 m|2.074 m|2.067 m|1.912 m|0.162 m|

The v0.4 trace-backed sweep showed usable but insufficient distance accuracy. The two strongest failure points were:

|Mark|Pose|Prediction|Error|
|--:|---|--:|--:|
|1.77 m|Front|2.014 m|+0.244 m|
|2.18 m|Rear|1.912 m|-0.268 m|

---

## 8. TriStream v0.5 Trace-Backed Rerun

The original report treated the v0.5 sweep as observational because trace recording had been accidentally disabled. That evidence is now superseded by a recorded v0.5 rerun.

The trace-backed v0.5 sweep is stored under:

```text
06_live-inference_v0.3/live_traces/
```

The unarchived trace set contains 12 accepted trace directories captured on `2026-05-21` between `15:47:48Z` and `15:51:22Z`. All 12 traces use:

|Field|Value|
|---|---|
|Distance/orientation model|`260521-1029_ts-2d-cnn`|
|Topology variant|`tri_stream_yaw_v0_5`|
|Checkpoint file|`models/distance-orientation/260521-1029_ts-2d-cnn/best.pt`|
|Checkpoint SHA-256 prefix|`0696f50e1365`|
|Model selection file|`06_live-inference_v0.3/models/selections/current.toml`|
|Git commit recorded by manifest|`2f29d447134c70cfee7f76278445a9fab66fab73`|
|Git dirty flag recorded by manifest|`null`|
|Camera source|`opencv-v4l2`, `/dev/video0`, `1920x1200`, `YUYV`, `50 fps`|
|ROI locator polarity|`inverted`|
|Manual mask applied to ROI locator|`true`|
|Manual mask applied to regressor preprocessing|`true`|

### 8.1 v0.5 Trace-Backed Samples

|Sample|Trace directory|Mark|Pose|Predicted distance|Error|
|--:|---|--:|---|--:|--:|
|V0.5-T-001|`20260521T154748Z__064d3f11...__a67c87cf`|1.59 m|Front|1.622 m|+0.032 m|
|V0.5-T-002|`20260521T154810Z__9abf135d...__5bce6e7a`|1.59 m|Side|1.534 m|-0.056 m|
|V0.5-T-003|`20260521T154827Z__066fb400...__316db59a`|1.59 m|Rear|1.537 m|-0.053 m|
|V0.5-T-004|`20260521T154842Z__3408094c...__b4354b5e`|1.77 m|Front|1.723 m|-0.047 m|
|V0.5-T-005|`20260521T154856Z__a7554386...__636ef929`|1.77 m|Side|1.704 m|-0.066 m|
|V0.5-T-006|`20260521T154910Z__dd6e4a04...__b05164d8`|1.77 m|Rear|1.668 m|-0.102 m|
|V0.5-T-007|`20260521T154932Z__f2180ea4...__032644ec`|1.97 m|Front|1.977 m|+0.007 m|
|V0.5-T-008|`20260521T154958Z__7b68b180...__fc77d9ba`|1.97 m|Side|1.854 m|-0.116 m|
|V0.5-T-009|`20260521T155016Z__e73d2f81...__06949045`|1.97 m|Rear|1.769 m|-0.201 m|
|V0.5-T-010|`20260521T155043Z__868a6536...__47bd7384`|2.18 m|Front|1.891 m|-0.289 m|
|V0.5-T-011|`20260521T155106Z__e5cd3ed9...__c99827b0`|2.18 m|Side|2.062 m|-0.118 m|
|V0.5-T-012|`20260521T155122Z__dfe65dea...__53517f6c`|2.18 m|Rear|1.979 m|-0.201 m|

Full trace directory names, in sample order:

```text
20260521T154748Z__064d3f11-f849-451e-a675-2707b67c4cd9__a67c87cf
20260521T154810Z__9abf135d-b069-4885-84ce-d161a3fec966__5bce6e7a
20260521T154827Z__066fb400-d3e7-4020-8792-08970c39951d__316db59a
20260521T154842Z__3408094c-817d-48f1-9cbf-f2180937df0e__b4354b5e
20260521T154856Z__a7554386-96ad-47ce-9e16-186c50ae5005__636ef929
20260521T154910Z__dd6e4a04-8a21-4e0f-9cac-ec01d6ecc796__b05164d8
20260521T154932Z__f2180ea4-d9de-432f-8b12-e31e55ff8016__032644ec
20260521T154958Z__7b68b180-2d12-43b9-bdbc-c505aac0a975__fc77d9ba
20260521T155016Z__e73d2f81-f07b-4fe3-bf71-5f79f36ac9f5__06949045
20260521T155043Z__868a6536-128a-4ce6-af6c-a6b4a558557b__47bd7384
20260521T155106Z__e5cd3ed9-e576-453a-b4db-edef3ceea761__c99827b0
20260521T155122Z__dfe65dea-eb25-4685-9444-8df71b3054c7__53517f6c
```

### 8.2 v0.5 Metrics

|Metric|Value|
|---|--:|
|Mean absolute error|0.1074 m|
|RMSE|0.1341 m|
|Mean signed error|-0.1008 m|
|Median absolute error|0.0837 m|
|Maximum absolute error|0.2895 m|
|Samples within 10 cm|6 / 12|
|Samples within 5 cm|3 / 12|

### 8.3 v0.5 Pose Spread

|Mark|Front|Side|Rear|Spread|
|--:|--:|--:|--:|--:|
|1.59 m|1.622 m|1.534 m|1.537 m|0.088 m|
|1.77 m|1.723 m|1.704 m|1.668 m|0.054 m|
|1.97 m|1.977 m|1.854 m|1.769 m|0.208 m|
|2.18 m|1.891 m|2.062 m|1.979 m|0.171 m|

The v0.5 trace-backed rerun confirms that the model remains pose-sensitive at fixed measured distances. The pattern is not identical to the earlier untraced observational sweep: the rerun shows a strong overall under-prediction bias, especially at the longer marks.

The original front-high / rear-low signature is still visible in three of the four mark groups:

```text
1.59 m: front highest, side/rear lower
1.77 m: front highest, rear lowest
1.97 m: front highest, rear lowest
```

The `2.18 m / front` sample is the strongest v0.5 failure in this trace set, with a `-0.289 m` error. That sample prevents a simple "front always high" reading of the rerun, but it reinforces the main finding: the direct-regression output is not stable across pose at the same floor position.

---

## 9. Model Comparison

|Metric|TriStream v0.4 trace-backed|TriStream v0.5 trace-backed|Change|
|---|--:|--:|--:|
|Mean absolute error|0.1105 m|**0.1074 m**|-0.0031 m|
|RMSE|**0.1317 m**|0.1341 m|+0.0024 m|
|Mean signed error|+0.0198 m|-0.1008 m|-0.1206 m|
|Median absolute error|**0.0825 m**|0.0837 m|+0.0012 m|
|Maximum absolute error|**0.2680 m**|0.2895 m|+0.0215 m|
|Samples within 10 cm|**7 / 12**|6 / 12|-1|
|Samples within 5 cm|1 / 12|**3 / 12**|+2|
|Average pose spread|0.1380 m|**0.1304 m**|-0.0076 m|

The trace-backed v0.5 rerun is not the clean scalar improvement suggested by the earlier untraced observations. It is marginally better on MAE, stricter `5 cm` count, and average pose spread, but worse on RMSE, maximum error, and `10 cm` count.

Most importantly, both versions retain pose-linked errors at fixed measured floor positions.

The pose mean error comparison is:

|Pose|v0.4 mean error|v0.5 mean error|
|---|--:|--:|
|Front|+0.0885 m|-0.0744 m|
|Side|+0.0162 m|-0.0891 m|
|Rear|-0.0453 m|-0.1391 m|

The v0.5 trace-backed rerun shifted the overall bias negative. Rear-facing views remain the most under-predicted on average, and front-facing views are still not stable across marks.

---

## 10. Brightness and Specular Sensitivity

The earlier untraced v0.5 sweep included exploratory readings suggesting that windscreen reflection could move front-facing distance predictions by several centimetres. Those readings are not used in the updated trace-backed metrics.

The recorded v0.5 traces still show visible windscreen reflection in some front-facing ROI crops, especially at the longer marks. However, the accepted trace set does not contain a controlled trace-backed A/B pair where only reflection changes while pose and placement remain fixed.

The conservative conclusion is therefore:

```text
specular sensitivity remains plausible
the trace-backed rerun does not quantify it independently
the main evidenced failure remains pose-linked distance instability
```

This supports the broader finding that the remaining error is not explained by camera geometry alone. Pose, appearance, foreground representation, and lighting all appear relevant.

---

## 11. Findings

### 11.1 Camera-model mismatch was a contributor, not the root cause

The camera-model correction improved aggregate distance metrics, indicating that real/synthetic camera mismatch had some effect. It did not resolve the pose-dependent structure.

### 11.2 TriStream v0.5 did not resolve the live failure mode

The trace-backed v0.5 rerun replaces the earlier untraced observational v0.5 sweep as repository evidence.

Against v0.4, v0.5 is only marginally better on mean absolute error and strict `5 cm` count, while being worse on RMSE, maximum error, and `10 cm` count. It also introduces a strong negative signed bias in this live sweep.

### 11.3 Pose-dependent distance bias remains unresolved

Across sweeps and model versions, predicted distance remains sensitive to vehicle pose at fixed measured floor positions.

The exact ordering varies by mark and run, but the structural failure remains:

```text
pose changes produce distance changes that are too large
rear-facing views tend to be under-predicted
front-facing behaviour is unstable across marks
```

This is a structural failure mode rather than a random measurement issue.

### 11.4 Scalar distance/yaw outputs are too opaque for the remaining problem

The direct-regression model family can report that distance and yaw are wrong. It cannot expose enough intermediate geometric state to determine whether the model has misinterpreted scale, pose, extent, foreground shape, visibility, lighting, or some combination of those factors.

This is the key architectural lesson from the incident.

---

## 12. Engineering Outcome

The current direct distance/yaw tri-stream family remains valuable as:

```text
a baseline model family
a live-runtime integration path
a comparison point for future architectures
a useful demonstration of iterative model improvement
```

It is not the preferred path for the next major improvement cycle.

The incident provides sufficient evidence to shift primary model-development effort toward the new amodal keypoint topology defined in the separate topology document. This pivot is not a replacement for failure analysis; it is the result of it.

The direct-regression family remains useful, but the persistence of pose-dependent bias indicates that further tuning is unlikely to provide the diagnostic visibility needed to resolve the deeper issue. The system now needs a representation that exposes the model's inferred geometry, rather than only its final scalar estimate.

---

## 13. Development Direction

The next model-development phase should focus on the already-defined keypoint-based topology. This incident report does not define that topology; it records why the project is moving in that direction.

The goals of the next phase are:

```text
retain the current direct-regression models as baselines
implement the new topology as a separate model family
compare it against TriStream v0.5 using the same live sweep protocol
evaluate whether the new representation reduces pose-linked distance error
use its intermediate outputs to diagnose remaining failures
```

The live inference stack should continue to support the current distance/yaw interface while the new topology is trained and evaluated. The existing runtime architecture already emphasises contract-driven model selection, compatibility checks, trace capture, and artifact-backed debugging, which are directly useful for this transition.

---

## 14. Recommended Next Steps

### 14.1 Preserve this incident as the pivot record

Store the incident report and sweep data under the failure-analysis area of the repository.

Recommended contents:

```text
incident report
raw sweep tables
summary metrics
trace-backed v0.4 samples
trace-backed v0.5 samples
excluded / exploratory lighting notes
notes on measurement limitations
```

### 14.2 Use TriStream v0.5 as the current direct-regression baseline

TriStream v0.5 should still be retained as the current direct-regression comparator because it is the selected current model and the latest trace-backed direct-regression run.

Future reports should compare the new topology against the trace-backed v0.5 sweep, not the superseded untraced v0.5 observations.

### 14.3 Evaluate the new topology against the same failure mode

The key question for the new topology is not only whether aggregate distance accuracy improves.

It should be evaluated against the specific incident failure:

```text
Does predicted distance remain pose-sensitive at fixed measured floor positions?
```

If pose sensitivity remains, the new topology should at least expose more diagnostic information about why.

### 14.4 Continue reporting measurement limitations clearly

The live sweeps are practical engineering tests, not calibrated metrology. That should remain explicit.

The important signal is the repeated pose-dependent structure, not millimetre-level distance precision.

---

## 15. Conclusion

This incident began as a camera-model alignment investigation and developed into a model-representation finding.

Camera-model correction improved aggregate live distance error, but did not remove pose-dependent prediction bias. The trace-backed v0.5 rerun supersedes the earlier untraced v0.5 observations and shows that the current direct-regression model still produces pose-linked distance errors at fixed floor positions.

The updated trace-backed comparison is more conservative than the earlier observational one: v0.5 marginally improves MAE and the `5 cm` count over v0.4, but does not improve RMSE, maximum error, or the `10 cm` count, and it introduces a stronger negative signed bias in this sweep.

The evidence indicates that the current direct distance/yaw regression family can be useful, but is unlikely to provide the diagnostic visibility needed for the next stage of the project.

The outcome is an architectural pivot. The current family remains a useful baseline and runtime integration path. Primary model-development effort now moves to the separately specified amodal keypoint topology, with this incident providing the empirical justification for that direction.