# Failure Analysis Report: `260415-1146_ds-2d-cnn`

## 1. Evaluation Scope

This report applies the failure-labelling scheme defined in **Failure Analysis Framework v0.2** to the validation output of `260415-1146_ds-2d-cnn/run_0001`.

The underlying benchmark is the `def90_synth_v023-validation-shuffled` corpus, representing the current full-frame, full-yaw synthetic validation regime.

The operational definitions used in this report are:

- **failure**: distance absolute error `> 0.10 m` **or** yaw absolute error `> 5°`
- **clean success**: distance absolute error `<= 0.05 m` **and** yaw absolute error `<= 2.5°`

These definitions support both continuous evaluation and operationally interpretable thresholded analysis.

---

## 2. Executive Summary

This validation run demonstrates that operational distance error has been reduced to an effectively negligible level across the benchmark. Residual failure mass is now overwhelmingly concentrated in orientation prediction, specifically in a sparse but important yaw-error tail.

The principal findings are:

- distance performance is operationally strong across the full benchmark
- the earlier edge-associated distance weakness is no longer the dominant limitation
- yaw performance is strong in the bulk distribution, but exhibits a materially heavier tail than distance
- the remaining weakness is concentrated in edge-sensitive pose estimation and rare catastrophic orientation failures

The overall interpretation is that the model has moved beyond broad geometric instability and into a more specific residual-error regime: **distance is substantially solved, while sparse orientation failures remain the primary technical issue.**

---

## 3. Continuous Performance Summary

### 3.1 Distance error

- validation MAE: `0.010068 m`
- validation RMSE: `0.012967 m`
- median absolute error: `0.0083 m`
- `p95`: `0.0254 m`
- `p99`: `0.0364 m`
- `p99.9`: `0.0548 m`
- maximum absolute error: `0.1037 m`

### 3.2 Orientation error

- mean angular error: `1.49987°`
- median angular error: `0.91449°`
- `p95` angular error: `3.34459°`
- `p99` angular error: `12.86°`
- `p99.9` angular error: `55.23°`
- maximum angular error: `177.86°`

These continuous metrics show a clear asymmetry between the two targets. Distance error is tightly bounded throughout the distribution, whereas yaw error remains well controlled in the bulk but develops a substantially heavier upper tail.

---

## 4. Thresholded Operational Outcomes

### 4.1 Primary threshold results

- distance accuracy within `10 cm`: `0.99996` = `49,998 / 50,000`
- distance accuracy within `25 cm`: `1.00000`
- distance accuracy within `50 cm`: `1.00000`
- yaw accuracy within `5°`: `0.97482` = `48,741 / 50,000`
- yaw accuracy within `10°`: `0.98698`
- yaw accuracy within `15°`: `0.99168`

### 4.2 Joint success under the framework

Only **2** samples exceed the `10 cm` distance threshold, while **1,259** samples exceed the `5°` yaw threshold. On that basis, the operational failure population is clearly orientation-dominated.

Because the exact overlap between those two failure sets is not directly recoverable from the available output snippets, the joint outcome is most accurately expressed as a narrow bounded range:

- **joint success**: between `48,739` and `48,741` samples (`97.478%` to `97.482%`)
- **total failures under the framework**: between `1,259` and `1,261` samples (`2.518%` to `2.522%`)

---

## 5. Failure-Category Decomposition

Using the framework’s primary categories:

- **F1: distance-only failure** — between `0` and `2` samples
- **F2: yaw-only failure** — between `1,257` and `1,259` samples
- **F3: joint failure** — between `0` and `2` samples

This decomposition establishes the central conclusion of the present failure pass:

> **The model’s residual operational weakness is not distance estimation, but a sparse yaw-failure tail.**

---

## 6. Residual Error Structure

### 6.1 Distance error has been reduced to near-zero operational significance

Distance prediction is now tightly controlled across the benchmark:

- only `2` samples exceed `0.10 m`
- no samples exceed `0.25 m`
- left third of image: MAE `0.0098 m`, `0` errors `> 0.10 m`
- middle third: MAE `0.0096 m`, `1` error `> 0.10 m`
- right third: MAE `0.0108 m`, `1` error `> 0.10 m`

This indicates that the earlier edge-of-frame distance pathology is no longer the defining limitation of the system.

### 6.2 Residual failure mass is primarily carried by yaw outliers

Yaw performance remains strong in the bulk distribution:

- median `0.91°`
- `p95` `3.34°`

However, the upper tail is substantially heavier:

- `p99` `12.86°`
- `p99.9` `55.23°`
- maximum `177.86°`

This profile is characteristic of a model that is broadly competent in pose estimation, but still susceptible to occasional large or near-flip errors on difficult views.

### 6.3 Yaw degradation is strongly associated with lateral image position

The remaining weakness is strongly correlated with lateral placement:

- left third mean angular error: `1.68°`, with `211` samples above `15°`
- middle third mean angular error: `1.13°`, with `11` samples above `15°`
- right third mean angular error: `1.72°`, with `194` samples above `15°`
- centre band mean angular error: `1.15°`
- extreme-lateral band mean angular error: `2.23°`

This is an important structural result. The dominant residual issue is not general pose instability, but pose estimation under laterally extreme viewpoints.

### 6.4 Longer range modestly increases yaw error

Distance performance remains robust across depth bands, but yaw error increases in the farthest range band:

- `1.5–2.5 m`: mean angular error `1.58°`
- `2.5–3.5 m`: `1.17°`
- `3.5–4.5 m`: `1.08°`
- `4.5–5.5 m`: `1.20°`
- `5.5–7.5 m`: `2.82°`

This is not the principal driver of failure, but it is a real secondary gradient that should be preserved in future auditing.

---

## 7. Representative Failure Cases

### 7.1 Joint failure example

- `defender90_f000003_z04.957_j207`
- distance absolute error: `0.1298065 m`
- orientation absolute error: `5.6159363°`
- ROI centre: `[320, 220]` px

This is one of the rare cases in which both operational thresholds are crossed simultaneously. It occurs toward the left side of frame and at relatively long range.

### 7.2 Severe yaw-only failure

- `defender90_f001349_z01.141_j154`
- actual orientation: `59.6795°`
- orientation absolute error: `115.1596°`
- distance absolute error: `0.0437307 m`

This is a clear example of catastrophic orientation failure with distance remaining accurate, consistent with a near-flip or otherwise unstable pose interpretation.

### 7.3 Severe yaw-only failure with accurate distance

- `defender90_f002042_z01.675_j251`
- actual orientation: `111.6929°`
- predicted orientation: `327.4845°`
- orientation absolute error: `144.2084°`
- distance absolute error: `0.0144224 m`
- ROI centre: `[1312, 644]` px

This case provides strong evidence that the residual weakness is not a shared geometric collapse, but a specifically pose-related failure mode on difficult views.

### 7.4 Moderate yaw-only failure

- `defender90_f001611_z01.662_j075`
- orientation absolute error: `13.3205°`
- distance absolute error: `0.0357668 m`

This is representative of the more typical non-catastrophic yaw failure: distance remains clean, while yaw crosses the operational boundary by a meaningful margin.

### 7.5 Boundary-case yaw failure

- `defender90_f002014_z03.589_j181`
- orientation absolute error: `6.0323°`
- distance absolute error: `0.0870583 m`
- ROI centre: `[1264, 320]` px

This case is useful because it sits just beyond the operational threshold. It is not a dramatic collapse, but it is still operationally relevant under the framework.

---

## 8. Interpretation

This run should be regarded as a strong benchmark result with a clearly bounded residual weakness.

### 8.1 What the results now support

- the benchmark remains the most demanding checked-in full-frame, full-yaw synthetic regime
- distance performance is operationally strong
- bulk yaw performance is also strong
- the remaining failure population is dominated by a sparse orientation tail rather than broad system instability

### 8.2 What remains unresolved

- the yaw tail is not yet fully explained or eliminated
- edge-sensitive pose estimation remains an active limitation
- rare catastrophic orientation failures mean the model should not yet be treated as fully mature

The development story has therefore changed in an important way. The relevant question is no longer whether the model can perform the task in a broadly stable way. It can. The question is now how to explain, localise, and reduce the remaining orientation tail without disturbing the already-strong distance behaviour.

---

## 9. Recommended Next Analysis Pass

The next failure-analysis pass should focus on the remaining yaw tail as the highest-value source of technical signal.

### 9.1 Review set construction

Build a manual review set containing:

- all samples with yaw error `> 15°`
- all samples with yaw error `> 45°`
- all samples with distance error `> 0.10 m`

### 9.2 Interpretive tagging

For each reviewed sample, assign interpretive labels where applicable:

- **F4** near-flip / orientation confusion
- **F5** ROI / crop issue
- **F6** edge-of-frame / positional extremity
- **F7** visually ambiguous / hard case

### 9.3 Priority

The highest immediate analytical value lies in the most extreme yaw outliers, as these now contain most of the remaining failure signal.

---

## 10. Conclusion

`260415-1146_ds-2d-cnn/run_0001` should no longer be described as a model with a broad distance problem or a broad edge-of-frame problem. Its current profile is more precise:

- **operational distance failure has been reduced to near-zero**
- **bulk yaw performance is strong**
- **residual weakness is concentrated in a sparse, edge-sensitive orientation tail**
- **rare near-flip pose failures remain on difficult views**

That is a materially stronger and more specific position than earlier stages of the work. The next phase is not basic stabilisation, but targeted refinement: understanding and reducing the residual yaw tail while preserving the current distance performance.