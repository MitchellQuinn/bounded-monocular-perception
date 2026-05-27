# Incident 003: Foreground Mask Contamination Distance Underestimate

This incident records a live inference trace where the system predicted the Defender as materially closer than expected because the foreground extraction stage merged the vehicle with dark sheet folds and shadow.

## Contents

- [`foreground-mask-contamination-distance-underestimate-report.md`](foreground-mask-contamination-distance-underestimate-report.md): employer-facing incident report with evidence, root-cause analysis, the backed-out foreground rejection, the current diagnostic/component-selection remediation, and follow-up plan
- [`evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859`](evidence/traces/20260526T111823Z__600bc624-cee3-4aa5-a22f-3cdbda11963a__354ad859): copied live trace used as the primary evidence record

## Status

Partially remediated. The hard foreground-vs-locator rejection introduced after the incident was backed out because it was too brittle for live use. Current live preprocessing keeps the foreground-vs-locator check as diagnostic metadata, adds warnings for implausible foreground extents, and performs connected-component selection in the threshold foreground path. Locator-anchored fallback extraction, broader fixture replay, background-removal workflow improvements, and stronger foreground extraction remain planned follow-up work.
