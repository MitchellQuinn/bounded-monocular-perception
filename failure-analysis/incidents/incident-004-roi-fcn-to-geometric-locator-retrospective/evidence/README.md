# Incident 004 Evidence Traces

This directory contains a representative evidence bundle for the ROI-FCN to
geometric-locator retrospective.

The full source trace population remains in:

- `06_live-inference_v0.2/live_traces`
- `06_live-inference_v0.3/live_traces`

Only selected trace directories are copied here. Each copied directory preserves
the canonical non-prefixed trace artifacts: frame images, ROI/locator artifacts,
model-input images, JSON metadata, inference/failure results, and trace
manifests. Generated duplicate sidecars with frame-hash prefixes and raw byte
dumps are omitted to avoid duplicating bulk artifacts that do not change the
incident analysis.

Included traces:

| Trace | Source | Evidence role |
| --- | --- | --- |
| `20260513T114741Z__4c7f3a05-a4a2-4839-9125-59529662c151__ad073540` | v0.2 ROI-FCN | High-confidence clipped ROI failure. |
| `20260513T143009Z__91bba664-eb45-479c-950b-062bca12f646__da1b05a4` | v0.2 ROI-FCN | Clipped ROI failure with large clipped extent. |
| `20260517T113934Z__1930e4b2-d6a9-4e55-b890-594717411915__11c31935` | v0.2 ROI-FCN | Low-confidence ROI-FCN rejection. |
| `20260517T124435Z__b1d8121b-d1e7-4537-99dc-f57419d6d099__856b315d` | v0.2 ROI-FCN | Accepted crop with downstream foreground collapse and distance overestimate. |
| `20260521T155122Z__dfe65dea-eb25-4685-9444-8df71b3054c7__53517f6c` | v0.3 `background_edge_v1` | Geometric locator comparison with candidate/edge/chosen-contour artifacts. |
