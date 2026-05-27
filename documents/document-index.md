# Document Index

This directory contains current technical summaries, experimental model
directions, historical implementation notes, and planning records. For a first
review, use the current documents below rather than reading the directory
alphabetically.

## Current Reviewer Path

1. [`bounded-monocular-perception-technical-writeup-v0.9.md`](bounded-monocular-perception-technical-writeup-v0.9.md) - current repository-level technical walkthrough, evidence summary, caveats, and engineering learnings.
2. [`keypoint-regression-topology-v0.4-technical-summary.md`](keypoint-regression-topology-v0.4-technical-summary.md) - short employer-facing summary of the amodal keypoint direction.
3. [`keypoint-regression-topology-v0.4.md`](keypoint-regression-topology-v0.4.md) - detailed engineering specification for the amodal keypoint model family.
4. [`../failure-analysis/failure-analysis-index.md`](../failure-analysis/failure-analysis-index.md) - failure-analysis reports and incident evidence.

## Current Documents

| Document | Status | Use |
| --- | --- | --- |
| [`bounded-monocular-perception-technical-writeup-v0.9.md`](bounded-monocular-perception-technical-writeup-v0.9.md) | Current overview | Primary technical summary for reviewers. |
| [`keypoint-regression-topology-v0.4-technical-summary.md`](keypoint-regression-topology-v0.4-technical-summary.md) | Current topology summary | Short explanation of the keypoint-based direction and implementation milestone. |
| [`keypoint-regression-topology-v0.4.md`](keypoint-regression-topology-v0.4.md) | Current detailed topology spec | Detailed topology justification and implementation specification. |

## Supporting And Historical Documents

| Document | Status | Use |
| --- | --- | --- |
| [`generation-standards-v0.1.md`](generation-standards-v0.1.md) | Supporting | Repo-specific generation standard for keeping generated changes aligned with local architecture. |
| [`inference_v0_4_ts_integration_plan.md`](inference_v0_4_ts_integration_plan.md) | Historical planning record | Planning document for integrating the v0.4 tri-stream inference path into early live inference. |
| [`specifications/Live Inference Pipeline - Architecture Sketch v0.3.md`](specifications/Live%20Inference%20Pipeline%20-%20Architecture%20Sketch%20v0.3.md) | Historical architecture sketch | Early live-inference architecture decisions around PySide6, worker boundaries, and frame handoff. |

## Notes

- The current live runtime is `06_live-inference_v0.3`; older live documents are retained as project history.
- The keypoint topology now has a first experimental registered implementation, but it is not yet a selected live model artifact or externally validated accuracy improvement.
- Historical documents may mention older directory names or runtime versions. Treat the v0.9 writeup and this index as the current routing layer.
