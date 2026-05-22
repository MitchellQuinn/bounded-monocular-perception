# Incident Report: Pose-Dependent Distance Bias in Live Monocular Distance Regression

## Summary

This incident investigated a repeatable live-camera distance regression error in Project Raccoon Ball, a bounded monocular perception system for estimating vehicle distance and yaw from a fixed camera view. The system is intentionally scoped around a known vehicle, constrained camera geometry, synthetic supervision, runtime preprocessing, live inference, trace capture, and failure analysis.

The failure mode was identified during real-camera testing of the current tri-stream distance/yaw model family. At fixed measured floor positions, predicted distance varied systematically with vehicle pose. Front-facing views generally predicted farther away, rear-facing views generally predicted closer, and side-facing views were usually closest to the measured reference distance.

The initial investigation tested whether the issue was primarily caused by camera-model mismatch between Unity synthetic camera geometry and the real AR0234 camera. Applying an input-space camera-model correction modestly improved aggregate distance error, but did not resolve the pose-dependent bias.

A later comparison between TriStream v0.4 and TriStream v0.5 showed that v0.5 substantially improved aggregate distance accuracy. Mean absolute error fell from approximately `0.1105 m` to `0.0673 m` in the available sweep data. However, the structural pose-dependence remained.

The incident outcome is an architectural pivot. The current direct distance/yaw tri-stream family remains useful as a baseline and live-runtime integration path, but it is no longer the primary route for improving the system. The next model-development direction is the already-defined amodal keypoint topology, which is documented separately. This report does not reproduce that topology; it records the incident evidence that motivates the pivot.

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

Distance should be broadly pose-invariant. The vehicle’s yaw changes, but its position relative to the camera does not.

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

---

## 4. Evaluation Criteria

The project’s failure-analysis framework uses `10 cm` as the primary distance success boundary and `5 cm` as a stricter clean-success boundary. The same framework recommends reporting both continuous metrics and thresholded categories, rather than relying on one headline metric.

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
front prediction ≈ side prediction ≈ rear prediction
```

Observed pattern:

```text
front prediction tends high
side prediction tends intermediate or closest
rear prediction tends low
```

This repeated ordering is the core failure mode.

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

## 8. TriStream v0.5 Observational Sweep

A second sweep was run against TriStream v0.5 with camera intrinsics applied. Trace recording was accidentally disabled, so this sweep is treated as observational evidence rather than trace-backed repository evidence.

Several exploratory readings were taken while managing windscreen reflection. The table below contains the accepted sweep values only.

|Sample|Mark|Pose|Predicted distance|Error|
|--:|--:|---|--:|--:|
|V0.5-O-001|1.59 m|Front|1.624 m|+0.034 m|
|V0.5-O-002|1.59 m|Side|1.563 m|-0.027 m|
|V0.5-O-003|1.59 m|Rear|1.540 m|-0.050 m|
|V0.5-O-004|1.77 m|Front|1.913 m|+0.143 m|
|V0.5-O-005|1.77 m|Side|1.758 m|-0.012 m|
|V0.5-O-006|1.77 m|Rear|1.729 m|-0.041 m|
|V0.5-O-007|1.97 m|Front|2.083 m|+0.113 m|
|V0.5-O-008|1.97 m|Side|1.935 m|-0.035 m|
|V0.5-O-009|1.97 m|Rear|1.849 m|-0.121 m|
|V0.5-O-010|2.18 m|Front|2.132 m|-0.048 m|
|V0.5-O-011|2.18 m|Side|2.204 m|+0.024 m|
|V0.5-O-012|2.18 m|Rear|2.021 m|-0.159 m|

### 8.1 v0.5 Metrics

|Metric|Value|
|---|--:|
|Mean absolute error|0.0673 m|
|RMSE|0.0834 m|
|Mean signed error|-0.0149 m|
|Median absolute error|0.0445 m|
|Maximum absolute error|0.1590 m|
|Samples within 10 cm|8 / 12|
|Samples within 5 cm|8 / 12|

### 8.2 v0.5 Pose Spread

|Mark|Front|Side|Rear|Spread|
|--:|--:|--:|--:|--:|
|1.59 m|1.624 m|1.563 m|1.540 m|0.084 m|
|1.77 m|1.913 m|1.758 m|1.729 m|0.184 m|
|1.97 m|2.083 m|1.935 m|1.849 m|0.234 m|
|2.18 m|2.132 m|2.204 m|2.021 m|0.183 m|

TriStream v0.5 materially improved aggregate distance accuracy. However, it did not eliminate pose-dependent distance bias.

---

## 9. Model Comparison

|Metric|TriStream v0.4|TriStream v0.5|Change|
|---|--:|--:|--:|
|Mean absolute error|0.1105 m|**0.0673 m**|-0.0432 m|
|RMSE|0.1317 m|**0.0834 m**|-0.0483 m|
|Median absolute error|0.0825 m|**0.0445 m**|-0.0380 m|
|Maximum absolute error|0.2680 m|**0.1590 m**|-0.1090 m|
|Samples within 10 cm|7 / 12|**8 / 12**|+1|
|Samples within 5 cm|1 / 12|**8 / 12**|+7|

TriStream v0.5 is a clear improvement over v0.4 on scalar distance accuracy.

However, the pose mean error remains structured:

|Pose|v0.4 mean error|v0.5 mean error|
|---|--:|--:|
|Front|+0.0885 m|+0.0605 m|
|Side|+0.0162 m|-0.0125 m|
|Rear|-0.0453 m|-0.0928 m|

v0.5 reduced front-facing over-prediction but increased rear-facing under-prediction. The overall model improved, but the underlying pose-linked structure survived.

---

## 10. Brightness and Specular Sensitivity

During the v0.5 sweep, front-facing readings appeared sensitive to windscreen reflection.

Exploratory readings included:

|Mark|Pose|Condition|Prediction|
|--:|---|---|--:|
|1.97 m|Front|visible windscreen shine|2.142 m|
|1.97 m|Front|reduced shine / cleaner view|2.083 m|
|2.18 m|Front|visible windscreen shine|2.232 m|
|2.18 m|Front|reduced shine / slight off-centre adjustment|2.132 m|

These exploratory values were excluded from the main metrics, but they are diagnostically important. They suggest that specular highlights can shift predicted distance by several centimetres. Against a `10 cm` operational threshold and `5 cm` clean threshold, that is large enough to matter.

This supports the broader finding that the remaining error is not explained by camera geometry alone. Pose, appearance, foreground representation, and lighting all appear relevant.

---

## 11. Findings

### 11.1 Camera-model mismatch was a contributor, not the root cause

The camera-model correction improved aggregate distance metrics, indicating that real/synthetic camera mismatch had some effect. It did not resolve the pose-dependent structure.

### 11.2 TriStream v0.5 improved the current direct-regression family

TriStream v0.5 substantially improved scalar distance accuracy compared with v0.4. The improvement from `1 / 12` to `8 / 12` samples within `5 cm` is especially notable.

### 11.3 Pose-dependent distance bias remains unresolved

Across sweeps and model versions, the same pattern persisted:

```text
front-facing views tend to predict farther
rear-facing views tend to predict closer
side-facing views tend to be closest or intermediate
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

The direct-regression family improved when moved from v0.4 to v0.5, but the persistence of pose-dependent bias indicates that further tuning is unlikely to provide the diagnostic visibility needed to resolve the deeper issue. The system now needs a representation that exposes the model’s inferred geometry, rather than only its final scalar estimate.

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
observational v0.5 samples
excluded / exploratory lighting readings
notes on measurement limitations
```

### 14.2 Use TriStream v0.5 as the direct-regression baseline

TriStream v0.5 should be retained as the best current direct-regression comparator.

Future reports should compare the new topology against v0.5 rather than against earlier weaker runs.

### 14.2 Evaluate the new topology against the same failure mode

The key question for the new topology is not only whether aggregate distance accuracy improves.

It should be evaluated against the specific incident failure:

```text
Does predicted distance remain pose-sensitive at fixed measured floor positions?
```

If pose sensitivity remains, the new topology should at least expose more diagnostic information about why.

### 14.3 Continue reporting measurement limitations clearly

The live sweeps are practical engineering tests, not calibrated metrology. That should remain explicit.

The important signal is the repeated pose-dependent structure, not millimetre-level distance precision.

---

## 15. Conclusion

This incident began as a camera-model alignment investigation and developed into a model-representation finding.

Camera-model correction improved aggregate live distance error, but did not remove pose-dependent prediction bias. TriStream v0.5 substantially improved direct distance accuracy compared with TriStream v0.4, but retained the same structural pattern: front-facing views tended to predict farther away, rear-facing views tended to predict closer, and side-facing views were usually closest.

The evidence indicates that the current direct distance/yaw regression family can be improved, but is unlikely to provide the diagnostic visibility needed for the next stage of the project.

The outcome is an architectural pivot. The current family remains a useful baseline and runtime integration path. Primary model-development effort now moves to the separately specified amodal keypoint topology, with this incident providing the empirical justification for that direction.
