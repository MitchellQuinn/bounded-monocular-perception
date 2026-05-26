# Incident 003: Foreground Mask Contamination Distance Underestimate

This incident records a live inference trace where the system predicted the Defender as materially closer than expected because the foreground extraction stage merged the vehicle with dark sheet folds and shadow.

## Contents

- [`foreground-mask-contamination-distance-underestimate-report.md`](foreground-mask-contamination-distance-underestimate-report.md): employer-facing incident report with evidence, root-cause analysis, implemented P0 guard, and follow-up remediation plan
- [`evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859): copied live trace used as the primary evidence record

## Status

Partially remediated. The P0 foreground-vs-locator quality gate and regression test are implemented. Locator-anchored fallback extraction, background-removal workflow improvements, and stronger foreground extraction remain planned follow-up work.
