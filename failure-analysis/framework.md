# Failure Analysis Framework v0.2

## Purpose

This document sets out a practical framework for analysing failure modes in the joint evaluation of distance and yaw prediction.

The central design choice is to avoid using `2.5°` as the first binary success/failure boundary. That is not because `2.5°` is meaningless, but because it is probably too tight to serve as the primary operational threshold when the goals are:

- tractable failure analysis
- a sane number of failures to inspect
- a definition that feels proportionate next to `10 cm` for distance

The recommended shape is therefore a **two-layer scheme**.

## 1. Binary Operational Definition

This is the definition used for primary success/failure labelling.

**Distance success:** `abs(distance error) <= 0.10 m`  
**Orientation success:** `abs(yaw error) <= 5°`

From that, define:

- **Joint success** = both pass
- **Distance-only failure** = distance fails, yaw passes
- **Yaw-only failure** = distance passes, yaw fails
- **Joint failure** = both fail

This yields a clean, legible evaluation grid.

## 2. Tighter “Excellent / Clean” Threshold

This is **not** the failure boundary. It is a stricter quality threshold used to mark particularly strong predictions.

For example:

- **distance excellent:** `<= 0.05 m`
- **yaw excellent:** `<= 2.5°`

That preserves a clear role for `2.5°`.
It becomes a “clean hit” threshold rather than a threshold that floods the analysis with nominal failures.

This structure preserves both signals:

- `5°` for tractable binary failure analysis
- `2.5°` as a meaningful stricter quality indicator

## Why This Shape Is Appropriate

The objective is not only to score a model, but also to demonstrate serious evaluation practice.

Serious evaluation usually benefits from reporting both:

- **continuous metrics** for the fuller performance picture
- **thresholded categories** for operational interpretation

Both should therefore be reported.

### Continuous Metrics

- distance MAE / RMSE / median absolute error
- yaw mean absolute error / median absolute error / possibly `p95`
- possible calibration-oriented notes later, if useful

### Thresholded Success Metrics

- `% within 10 cm`
- `% within 5°`
- `% joint success (10 cm AND 5°)`
- optional `% excellent (5 cm AND 2.5°)`

This reporting structure is highly legible and supports both technical analysis and external communication.

## Failure-Mode Analysis

Once a sample is labelled as a failure, it should be tagged by **failure type** rather than treated as part of a single undifferentiated failure bucket.

Suggested tags:

- **F1: distance-only failure**
- **F2: yaw-only failure**
- **F3: joint failure**
- **F4: likely orientation flip / near-180° confusion**
- **F5: likely ROI / crop issue**
- **F6: edge-of-frame / positional extremity**
- **F7: visually ambiguous / hard case**

The first three are metric-defined categories. The later tags are interpretive analysis tags.

## Recommended Formal Definition

For the first proper formal definition:

> **A sample is a failure if distance absolute error exceeds `0.10 m` or orientation absolute error exceeds `5°`.**

Separately define:

> **A clean success is one with distance absolute error ≤ `0.05 m` and orientation absolute error ≤ `2.5°`.**

This gives a balanced and explainable framework that is neither too lax nor too punitive.

## Summary

- **Binary failure line:** `>10 cm` or `>5°`
- **Clean hit line:** `<=5 cm` and `<=2.5°`
