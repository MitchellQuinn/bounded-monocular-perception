# Bounded Monocular Perception

> - 🎥 **Live demo:** [https://www.youtube.com/watch?v=IOYiBk6UhAs](https://www.youtube.com/watch?v=IOYiBk6UhAs)
> - 📄 **Technical writeup:** [`documents/bounded-monocular-perception-technical-writeup-v0.10.md`](documents/bounded-monocular-perception-technical-writeup-v0.10.md)
> - 🔍 **Failure analysis:** [`failure-analysis/failure-analysis-index.md`](failure-analysis/failure-analysis-index.md)
> - 🧭 **Current model direction:** apparent-scale representation alignment and amodal semantic keypoint regression

This repository is a bounded computer-vision and applied-machine-learning workspace for estimating vehicle distance and yaw from a fixed monocular camera view under controlled conditions. This is a bounded perception investigation, not a benchmark system.

It is built end-to-end across Unity synthetic data generation, preprocessing contracts, PyTorch training, raw-image inference, live PySide6 runtime integration, camera calibration, representation alignment, trace capture, and failure analysis.

The project is deliberately narrow: one known vehicle family, fixed monocular camera geometry, a constrained movement plane, controlled full-frame captures, synthetic training and validation data, and live-local testing under controlled physical conditions.

It should be read as evidence of applied ML engineering and perception-system investigation: not just training a model, but measuring what breaks when an offline synthetic benchmark is composed into a live camera-driven runtime.

---

## What This Is / Is Not

This is a bounded research-engineering artifact for making a controlled monocular perception problem inspectable.

It is **not** presented as:

- a general object detector
- an open-world monocular 3D perception solution
- a production-ready real-world vision product
- a claim of robust real-camera transfer

The current value is the engineering record:

- synthetic-data generation with traceable metadata
- contract-driven preprocessing
- distance/yaw model training and evaluation
- raw-image and live-runtime composition
- calibration support
- apparent-scale representation alignment
- trace-backed failure analysis
- architectural pivots driven by measured failure modes

---

## What This Repository Demonstrates

| Capability | Evidence in this repo |
| --- | --- |
| End-to-end CV pipeline ownership | Unity generation → preprocessing → training → raw-image inference → live GUI runtime |
| Applied ML engineering discipline | explicit preprocessing contracts, model metadata, compatibility checks, deterministic evaluation artifacts |
| Runtime systems thinking | camera workers, frame handoff, GUI controls, locator selection, trace capture, model loading, device policy |
| Evaluation honesty | offline synthetic validation, raw-image composed inference, and live-camera behaviour are reported separately |
| Failure analysis | live incidents are traced through captured artifacts rather than collapsed into a single score |
| Representation iteration | direct scalar distance/yaw regression exposed pose-linked bias, motivating a more inspectable keypoint topology |
| Calibration-aware investigation | ChArUco calibration tooling and live intrinsics modes are integrated into the runtime path |
| Representation alignment | Incident 005 drives a post-foreground model representation transform for apparent-scale mitigation |

---

## Current Status

The current live-local runtime is [`06_live-inference_v0.3`](06_live-inference_v0.3).

The current deployed live path uses:

- deterministic geometric / foreground-based localisation for demo stabilisation
- tri-stream live preprocessing
- a PyTorch tri-stream distance/yaw inference engine
- PySide6 camera and inference workers
- manual masks, background capture, foreground extraction controls, and trace capture
- camera intrinsics modes for real-camera undistortion or real-to-Unity remapping
- an Incident 005 model representation transform for apparent-scale mitigation

The previous learned ROI-FCN localiser is now best understood as a historical / comparison path. It remains in the repository because it was part of the system’s development history and helped expose the runtime composition problem, but the active live direction has moved toward more inspectable geometric localisation.

The ROI-FCN/localisation pivot is recorded as Incident 004, and the newer Incident 005 report records the live/synthetic apparent-scale mismatch that remains after the locator and foreground path are more inspectable.

The current direct distance/yaw baseline is useful, but live trace-backed testing shows unresolved pose-linked and apparent-scale transfer risks. Those results motivate both the current representation-alignment work and the active amodal keypoint direction.

The amodal keypoint model implementation now exists, and training data processing is under way. First training/evaluation results are pending. Until those metrics are available, the keypoint branch should be treated as active roadmap work rather than a validated performance claim.

---

## Quick Reviewer Path

For a fast technical review, start here:

1. [`documents/bounded-monocular-perception-technical-writeup-v0.10.md`](documents/bounded-monocular-perception-technical-writeup-v0.10.md)
   Current repository-level technical overview, architecture, results, caveats, and engineering learnings.

2. [`documents/document-index.md`](documents/document-index.md)
   Routing layer for current and historical technical material.

3. [`failure-analysis/failure-analysis-index.md`](failure-analysis/failure-analysis-index.md)
   Failure-analysis reports and incident evidence.

4. [`failure-analysis/incidents/incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md`](failure-analysis/incidents/incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md)
   Remediated live preprocessing failure with trace-backed regression coverage.

5. [`failure-analysis/incidents/incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md`](failure-analysis/incidents/incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md)
   Live pose-linked distance-bias investigation and architectural pivot.

6. [`failure-analysis/incidents/incident-004-roi-fcn-to-geometric-locator-retrospective/roi-fcn-to-geometric-locator-retrospective-report.md`](failure-analysis/incidents/incident-004-roi-fcn-to-geometric-locator-retrospective/roi-fcn-to-geometric-locator-retrospective-report.md)
   ROI-FCN-to-geometric-locator retrospective and live ROI-boundary justification.

7. [`failure-analysis/incidents/incident-005-live-synthetic-apparent-scale-mismatch/live-synthetic-apparent-scale-mismatch-report.md`](failure-analysis/incidents/incident-005-live-synthetic-apparent-scale-mismatch/live-synthetic-apparent-scale-mismatch-report.md)
   Live/synthetic apparent-scale mismatch investigation and representation-transform mitigation.

8. [`documents/keypoint-regression-topology-v0.4-technical-summary.md`](documents/keypoint-regression-topology-v0.4-technical-summary.md)
   Short summary of the amodal semantic keypoint direction.

9. [`06_live-inference_v0.3/RUNTIME_NOTES.md`](06_live-inference_v0.3/RUNTIME_NOTES.md)
   Current live-local runtime notes and diagnostic flow.

---

## Representative Evidence

The repository separates evidence types rather than collapsing them into one headline metric.

| Area | Representative evidence | Interpretation |
| --- | --- | --- |
| Offline synthetic validation | Current `260521-1029_ts-2d-cnn` direct tri-stream artifact records distance MAE `0.015856 m`, distance RMSE `0.026030 m`, yaw mean error `1.503031 deg`, and yaw within `5 deg` of `0.985433` on synthetic validation. | Strong bounded synthetic baseline under the trained representation contract. |
| ROI localisation | `260420-1219_roi-fcn-tiny__run_0003` records mean centre error `3.1757 px`, p95 `7.7098 px`, and ROI full-containment success `0.9891` on synthetic validation. | Useful historical learned-localisation baseline, but no longer the preferred live path. |
| Raw-image composed inference | Raw-image reports show material degradation compared with preprocessed validation, including crop-boundary distance tails and yaw-heavy failure populations. | Composition introduces system-level failure modes not visible in preprocessed validation. |
| Live preprocessing failure analysis | Incident 001 traces a live distance spike to foreground/silhouette collapse, remediates the preprocessing path, and adds fixture-backed regression tests. | Failure was made observable and converted into a testable remediation. |
| Live pose-bias failure analysis | Incident 002 shows that camera intrinsics improve aggregate live error but do not remove pose-linked distance bias in the direct tri-stream model family. | Direct scalar regression is not inspectable enough for the observed live failure mode. |
| Live apparent-scale failure analysis | Incident 005 shows that clean accepted live readings underpredicted by about `0.35 m` to `0.40 m`, while paired synthetic/live scale analysis predicted a similar apparent-distance offset. | Synthetic/live representation alignment is now a first-class validation boundary. |
| Current architectural pivot | Amodal semantic keypoint regression is being developed to expose the model’s inferred object geometry directly. | The next model family is designed around inspectability and diagnostic value, not just lower scalar error. |

---

## Runtime Gap

The central engineering lesson of the project is that strong offline synthetic validation is not enough.

A composed perception runtime depends on:

- camera capture
- camera intrinsics
- background handling
- mask/fill policy
- ROI or foreground localisation
- crop selection
- foreground extraction
- apparent-scale and model-space alignment
- representation reconstruction
- geometry-vector construction
- model compatibility
- inference execution
- output decoding
- trace capture

Failures can enter at any of those boundaries.

This repository treats that gap as part of the work. The live runtime is not presented as a finished deployment. It is an instrumented system for exposing where offline-trained perception models fail when composed with real camera input and runtime preprocessing.

---

## Failure Analysis Highlights

### Incident 001 — Live Distance Regression Spike

Two near-identical live captures of a stationary vehicle produced sharply different distance predictions.

The trace evidence showed that the model was not the primary source of the spike. The accepted camera frame and ROI crop contained the vehicle, but downstream silhouette recovery collapsed the vehicle representation to a tiny fragment.

The failure was remediated by changing the foreground/silhouette recovery selection logic, changing the live default foreground path, and adding regression coverage.

See: [`failure-analysis/incidents/incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md`](failure-analysis/incidents/incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md)

### Incident 002 — Pose-Dependent Distance Bias

Live trace-backed sweeps showed that predicted distance varied systematically with vehicle yaw, even when the physical distance was held constant.

Camera intrinsics correction improved part of the alignment problem but did not remove the pose-linked error pattern. This exposed a limitation of direct scalar distance/yaw regression: the model gives an answer but not an inspectable object-state hypothesis.

That finding motivates the amodal semantic keypoint topology.

See: [`failure-analysis/incidents/incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md`](failure-analysis/incidents/incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md)

### Incident 004: ROI-FCN Localisation Retrospective

The ROI-FCN path was a useful intermediate stage: it turned crop-centre selection into a learned localisation task and allowed raw-image inference paths to be composed.

In live testing, however, the learned localiser proved less useful than more inspectable geometric / foreground-driven localisation for the controlled physical demo setup. The active runtime direction has therefore moved away from ROI-FCN as the preferred live localiser.

See: [`failure-analysis/incidents/incident-004-roi-fcn-to-geometric-locator-retrospective/roi-fcn-to-geometric-locator-retrospective-report.md`](failure-analysis/incidents/incident-004-roi-fcn-to-geometric-locator-retrospective/roi-fcn-to-geometric-locator-retrospective-report.md)

---

### Incident 005: Live/Synthetic Apparent-Scale Mismatch

After the locator and foreground path became more inspectable, accepted live readings still underpredicted distance by about `0.35 m` to `0.40 m`. Paired synthetic/live image analysis showed the live target appearing larger than the synthetic target at the same nominal distance, predicting a mean apparent-distance offset close to the measured live bias.

The live runtime now includes a post-foreground model representation transform for apparent-scale mitigation. A first follow-up sweep improved all-row mean signed error to `-0.113 m` and mean absolute error to `0.118 m`; a later three-distance sweep improved further to mean signed error `-0.033 m` and mean absolute error `0.080 m`. That is a material improvement, but `1.60 m` close-range rows still average `-0.145 m`, so calibrated live-distance claims remain bounded.

See: [`failure-analysis/incidents/incident-005-live-synthetic-apparent-scale-mismatch/live-synthetic-apparent-scale-mismatch-report.md`](failure-analysis/incidents/incident-005-live-synthetic-apparent-scale-mismatch/live-synthetic-apparent-scale-mismatch-report.md)

---

## Amodal Keypoint Direction

The current direct distance/yaw path predicts final scalar outputs from tri-stream image-derived inputs:

```text
x_distance_image + x_orientation_image + x_geometry -> distance + yaw
```

That path is compact and operationally useful, but it hides the intermediate geometric state the model has inferred.

The active next model direction is amodal semantic keypoint regression. Instead of predicting only final distance and yaw, the model emits a structured object-state hypothesis:

```text
tri-stream inputs
  -> Defender centre in camera/world coordinates
  -> fixed semantic external Defender keypoints
  -> keypoint visibility / in-frame state
  -> direct distance and yaw heads
  -> optional geometry-fit diagnostics
```

The key design choice is to predict a fixed ordered set of semantic keypoints, including occluded keypoints, for a known rigid object in a bounded scene.

The intended engineering value is diagnostic:

* visible and hidden keypoint errors can be reported separately
* impossible keypoint geometry can be detected
* direct distance/yaw outputs can be compared against keypoint-derived pose
* scalar failure can be traced to object-state misunderstanding rather than remaining opaque

This is not a claim of general 3D reconstruction or arbitrary hidden-geometry recovery. It is a bounded representation experiment for a known vehicle instance under controlled conditions.

Current status:

* implementation exists
* training data processing is in progress
* first model training/evaluation run is pending
* external performance claims are blocked until evaluation results exist
* geometry-only ablation remains required before claiming meaningful image-stream contribution

See:

* [`documents/keypoint-regression-topology-v0.4-technical-summary.md`](documents/keypoint-regression-topology-v0.4-technical-summary.md)
* [`documents/keypoint-regression-topology-v0.4.md`](documents/keypoint-regression-topology-v0.4.md)

---

## Repository Layout

| Path                                                                                                                                                       | Purpose                                                                                                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`01_rb_synthetic-data_3`](01_rb_synthetic-data_3)                                                                                                         | Unity/C# synthetic full-frame image generation.                                                                                                            |
| [`02_synthetic-data-processing-v4.0`](02_synthetic-data-processing-v4.0)                                                                                   | v4 preprocessing, detection metadata, silhouette/foreground handling, dual-stream and tri-stream packing.                                                  |
| [`03_rb-training-v2.0`](03_rb-training-v2.0)                                                                                                               | PyTorch training, topology registry, evaluation, resume support, and model reporting.                                                                      |
| [`04_ROI-FCN`](04_ROI-FCN)                                                                                                                                 | Historical ROI-FCN preprocessing/training for crop-centre heatmap localisation. Retained for evidence and comparison.                                      |
| [`05_inference-v0.3-ds`](05_inference-v0.3-ds)                                                                                                             | Raw-image ROI-FCN plus dual-stream distance/yaw inference.                                                                                                 |
| [`05_inference-v0.4-ts`](05_inference-v0.4-ts)                                                                                                             | Tri-stream-facing raw-image inference and brightness-analysis tooling.                                                                                     |
| [`06_live-inference_v0.1`](06_live-inference_v0.1), [`06_live-inference_v0.2`](06_live-inference_v0.2), [`06_live-inference_v0.3`](06_live-inference_v0.3) | Live-local runtime iterations; v0.3 is the current path.                                                                                                   |
| [`charuco-calibration`](charuco-calibration)                                                                                                               | PySide6/OpenCV ChArUco calibration capture, solve, and artifact export tooling.                                                                            |
| [`failure-analysis`](failure-analysis)                                                                                                                     | Failure-analysis framework, model-evaluation reports, incident investigations, and supporting evidence.                                                    |
| [`documents`](documents)                                                                                                                                   | Technical writeups, topology proposals, implementation notes, and specifications. Start with [`documents/document-index.md`](documents/document-index.md). |
| [`examples/defender-images`](examples/defender-images)                                                                                                     | Scaffold for a bounded example-image corpus and its notices.                                                                                               |
| [`scripts/run-tests.sh`](scripts/run-tests.sh)                                                                                                             | Repo-level focused test runner for the checked-in subprojects.                                                                                             |

---

## Validation

Use the repository virtual environment Python for checks:

```bash
./scripts/run-tests.sh
```

The runner executes focused tests from the current checked-in subprojects. A plain repo-root `pytest` run is not the intended entry point for this multi-project layout.

---

## Roadmap

### Near term

* Script the Incident 005 synthetic/live scale comparison table and add scale fixtures.
* Re-run the live distance sweep after apparent-scale correction with trace capture.
* Complete amodal keypoint training-data processing.
* Train the first amodal keypoint model.
* Report keypoint metrics separately from direct distance/yaw metrics.
* Add visible-vs-hidden keypoint evaluation.
* Add or preserve geometry-only ablation before making external claims about image-stream contribution.
* Add live/demo visualisation for keypoint overlays once the model produces meaningful outputs.

### Medium term

* Move from Python/PyTorch live inference toward a C++ inference implementation.
* Package inference in a containerised form for reviewer/employer testing where practical.
* Define a small downloadable test/demo bundle that does not require the full training corpus.
* Decide whether selected model weights can be distributed separately from the source repository.
* Add a minimal reproducible inference path if licensing, artifact size, and support burden allow it.

### Longer term

* Compare direct scalar distance/yaw outputs against keypoint-derived pose.
* Use rigid-geometry residuals as diagnostic signals.
* Extend real-camera validation with a small controlled measurement set.
* Continue separating bounded evidence from unsupported deployment claims.

---

## Distributed and Non-Distributed Material

This repository intentionally does not currently distribute:

* trained `.pt` model weights
* the full synthetic training and validation corpora
* the original Defender `.fbx` source asset
* large generated datasets
* local runtime traces that are not needed for review or incident evidence

Model metadata, reports, selected traces, and small evidence artifacts are kept where they support review, reproducibility, or failure analysis.

A future packaged inference/demo artifact may distribute selected runtime assets separately if the packaging, rights, artifact-size, and support boundaries are clear.

---

## Rights and Third-Party Material

Unless otherwise stated, repo-authored code and documentation are provided under ordinary copyright terms only. No open-source license is attached to repo-authored material at this time.

See:

* [`COPYRIGHT.md`](COPYRIGHT.md) for repo-authored material.
* [`THIRD_PARTY.md`](THIRD_PARTY.md) for third-party asset provenance.
* [`examples/defender-images/NOTICE.md`](examples/defender-images/NOTICE.md) for the bounded example-image corpus notice when that corpus is populated.