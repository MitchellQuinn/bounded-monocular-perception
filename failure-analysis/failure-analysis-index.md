# Failure Analysis

This directory is the canonical home for failure-analysis material in the repository. It keeps polished analysis close to the underlying evidence while separating offline model evaluations from live-runtime incidents.

## Structure

- [`framework.md`](framework.md): shared operational thresholds and failure categories used by the reports
- [`model-evaluations/`](model-evaluations): offline or raw-image benchmark analyses for named model runs
- [`incidents/`](incidents): live-runtime or system-level investigations with supporting artifacts

## Incident Reports

| Report | Status | Focus |
| --- | --- | --- |
| [`incident-001-live-distance-regression-spike`](incidents/incident-001-live-distance-regression-spike/live-distance-regression-spike-report.md) | Remediated | Live distance spike traced to foreground/silhouette preprocessing collapse |
| [`incident-002-pose-dependent-distance-bias`](incidents/incident-002-pose-dependent-distance-bias/pose-dependent-distance-bias-report.md) | Investigated; architectural pivot | Pose-linked live distance bias in the direct distance/yaw tri-stream model family |

## Model Evaluation Reports

| Report | Focus |
| --- | --- |
| [`260415-1146_ds-2d-cnn`](model-evaluations/260415-1146_ds-2d-cnn.md) | Raw-image validation analysis with severe crop-boundary distance tail and yaw-heavy failure population |
| [`260425-1025_ds-2d-cnn`](model-evaluations/260425-1025_ds-2d-cnn.md) | Raw-image validation analysis with stronger bulk distance performance but broad yaw underperformance |

## Reading Order

For a quick technical review, start with the framework, then read the incident summaries. The model-evaluation reports are useful supporting evidence for how the project separates continuous metrics, thresholded outcomes, and interpretable failure modes.

The incident directories intentionally retain raw images, JSON metadata, and trace artifacts where available. Those artifacts are part of the evidence record, not generated clutter.
