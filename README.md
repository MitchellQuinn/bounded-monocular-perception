# Bounded Monocular Perception

This repository is a bounded computer-vision and applied-machine-learning
workspace for estimating vehicle distance and yaw from a fixed monocular camera
view under controlled conditions.

The project is deliberately narrow: one known vehicle family, fixed or
controlled camera geometry, synthetic supervision, explicit preprocessing
contracts, raw-image inference paths, a live local runtime, calibration support,
trace capture, and failure analysis.

It should be read as evidence of applied ML engineering, computer vision,
runtime integration, and experimental discipline. It is not a claim of open-world
 object detection, multi-object
tracking, or unconstrained real-world scene understanding.

## Quick Reviewer Path

For a fast technical review, start here:

1. [`documents/bounded-monocular-perception-technical-writeup-v0.8.md`](documents/bounded-monocular-perception-technical-writeup-v0.8.md) - current technical overview, architecture, results, caveats, and engineering learnings.
2. [`documents/document-index.md`](documents/document-index.md) - document routing layer for current and historical technical material.
3. [`06_live-inference_v0.3/RUNTIME_NOTES.md`](06_live-inference_v0.3/RUNTIME_NOTES.md) - current live-local runtime notes and diagnostic flow.
4. [`failure-analysis/failure-analysis-index.md`](failure-analysis/failure-analysis-index.md) - failure-analysis index.
5. [`failure-analysis/incidents/incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md`](failure-analysis/incidents/incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md) - remediated live preprocessing failure with trace-backed regression coverage.
6. [`failure-analysis/incidents/incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md`](failure-analysis/incidents/incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md) - live pose-linked distance-bias investigation and architectural pivot.
7. [`documents/keypoint-regression-topology-v0.4-technical-summary.md`](documents/keypoint-regression-topology-v0.4-technical-summary.md) - proposed next model direction using amodal semantic keypoints.

## What This Repository Demonstrates

- Unity/C# synthetic data generation with run manifests and traceable sample metadata.
- OpenCV/NumPy preprocessing pipelines with explicit representation contracts.
- PyTorch training code for distance, distance-plus-yaw, dual-stream, tri-stream, and ROI-localisation model families.
- Circular yaw regression through `sin/cos` targets.
- Learned ROI-FCN crop-centre localisation and deterministic live localisation alternatives.
- Raw-image inference paths that compose localisation, preprocessing, model loading, and JSON/image artifacts.
- A PySide6 live-local inference runtime with camera workers, GUI controls, background capture, manual masks, trace recording, and model compatibility checks.
- ChArUco camera calibration tooling and calibration-backed live camera intrinsics transforms.
- Failure analysis that distinguishes offline synthetic validation, composed raw-image inference, and live-camera behaviour.

## Current Status

The current live-local runtime is [`06_live-inference_v0.3`](06_live-inference_v0.3).
It defaults to the inspectable deterministic `background_edge_v1` locator for
demo stabilisation, while retaining ROI-FCN as an explicit legacy comparison
path.

The current direct distance/yaw baseline is the `260521-1029_ts-2d-cnn`
tri-stream artifact using the `tri_stream_yaw_v0_5` topology variant and the
`rb-preprocess-v4-tri-stream-grayscale-white-v1` preprocessing contract.

Offline synthetic validation is strong, but live trace-backed testing still
shows unresolved pose-dependent distance bias. That limitation is recorded in
Incident 002 and motivates the proposed amodal keypoint topology. The keypoint
topology is documented as a proposal and development direction; it is not yet an
implemented registered training topology in this repository snapshot.

## Repository Layout

- [`01_rb_synthetic-data_3`](01_rb_synthetic-data_3): Unity/C# synthetic full-frame image generation.
- [`02_synthetic-data-processing-v4.0`](02_synthetic-data-processing-v4.0): v4 preprocessing, detection metadata, silhouette/foreground handling, dual-stream and tri-stream packing.
- [`03_rb-training-v2.0`](03_rb-training-v2.0): PyTorch training, topology registry, evaluation, resume support, and model reporting.
- [`04_ROI-FCN`](04_ROI-FCN): ROI-FCN preprocessing and training for crop-centre heatmap localisation.
- [`05_inference-v0.3-ds`](05_inference-v0.3-ds): raw-image ROI-FCN plus dual-stream distance/yaw inference.
- [`05_inference-v0.4-ts`](05_inference-v0.4-ts): tri-stream-facing raw-image inference and brightness-analysis tooling.
- [`06_live-inference_v0.1`](06_live-inference_v0.1), [`06_live-inference_v0.2`](06_live-inference_v0.2), [`06_live-inference_v0.3`](06_live-inference_v0.3): live-local runtime iterations; v0.3 is the current path.
- [`charuco-calibration`](charuco-calibration): PySide6/OpenCV ChArUco calibration capture, solve, and artifact export tooling.
- [`failure-analysis`](failure-analysis): failure-analysis framework, model-evaluation reports, incident investigations, and supporting evidence.
- [`documents`](documents): technical writeups, topology proposals, implementation notes, and specifications; start with [`documents/document-index.md`](documents/document-index.md).
- [`examples/defender-images`](examples/defender-images): scaffold for a bounded example-image corpus and its notices.
- [`scripts/run-tests.sh`](scripts/run-tests.sh): repo-level focused test runner for the checked-in subprojects.

## Representative Evidence

The repository separates evidence types rather than collapsing them into one
headline metric.

| Area | Representative Evidence |
| --- | --- |
| Offline synthetic model validation | Current `260521-1029_ts-2d-cnn` direct tri-stream artifact records distance MAE `0.015856 m`, distance RMSE `0.026030 m`, yaw mean error `1.503031 deg`, and yaw within `5 deg` of `0.985433` on synthetic validation. |
| ROI localisation | `260420-1219_roi-fcn-tiny__run_0003` records mean centre error `3.1757 px`, p95 `7.7098 px`, and ROI full-containment success `0.9891`. |
| Raw-image composed inference | Raw-image reports show material degradation compared with preprocessed validation, including crop-boundary distance tails and yaw-heavy failure populations. |
| Live preprocessing failure analysis | Incident 001 traces a live distance spike to foreground/silhouette collapse, remediates the preprocessing path, and adds fixture-backed regression tests. |
| Live pose-bias failure analysis | Incident 002 shows that camera intrinsics improve aggregate live error but do not remove pose-linked distance bias in the direct tri-stream model family. |

See the v0.8 technical writeup for the full results table and caveats.

## Validation

Use the repository virtual environment Python for checks:

```bash
./scripts/run-tests.sh
```

The runner executes focused tests from the current checked-in subprojects. A
plain repo-root `pytest` run is not the intended entry point for this
multi-project layout.

## Distributed and Non-Distributed Material

This repository intentionally does not distribute:

- trained `.pt` model weights.
- the full synthetic training and validation corpora.
- the original Defender `.fbx` source asset.

Model metadata, reports, selected traces, and small evidence artifacts are kept
where they support review, reproducibility, or failure analysis. Large generated
datasets, runtime outputs, checkpoints, and local camera traces are excluded by
default.

## Rights and Third-Party Material

Unless otherwise stated, repo-authored code and documentation are provided under
ordinary copyright terms only. No open-source license is attached to
repo-authored material at this time.

See:

- [`COPYRIGHT.md`](COPYRIGHT.md) for repo-authored material.
- [`THIRD_PARTY.md`](THIRD_PARTY.md) for third-party asset provenance.
- [`examples/defender-images/NOTICE.md`](examples/defender-images/NOTICE.md) for the bounded example-image corpus notice when that corpus is populated.
