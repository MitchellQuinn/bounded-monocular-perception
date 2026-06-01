# Incident 005 Evidence Manifest

This directory is the intended repository evidence home for `incident-005-live-synthetic-apparent-scale-mismatch`.

The staged output includes the compact image evidence needed to review the apparent-scale finding. The heavier raw image pairs and live-inference trace directories remain in the local incident workspace and can be copied later if the repository should preserve the full artifact set.

## Included Scale-Pair Evidence

These files are already staged under:

```text
failure-analysis/incidents/incident-005-live-synthetic-apparent-scale-mismatch/evidence/scale-pairs/
```

Included files:

```text
pair1_front_summary_comparison.png
pair2_side_summary_comparison.png
pair3_front_summary_comparison.png
pair4_side_summary_comparison.png
pair5_front_summary_comparison.png
pair6_side_summary_comparison.png
pair7_front_summary_comparison.png
pair8_side_summary_comparison.png
```

These eight PNGs are the review-facing evidence for the table in the report. They show the synthetic/live comparison for every measured pair used in the apparent-distance offset calculation.

## Optional Raw Scale-Pair Evidence

The local workspace also contains the raw synthetic/live pair images:

```text
Project Raccoon Ball/Failure Investigations/Incident-005/Image Analysis/
```

Optional raw files:

```text
p1-l.png
p1-s.png
p2-l.png
p2-s.png
p3-l.png
p3-s.png
p4-l.png
p4-s.png
p5-l.png
p5-s.png
p6-l.png
p6-s.png
p7-l.png
p7-s.png
p8-l.png
p8-s.png
```

These are not required for first-pass review because the included summary comparison images preserve the scale evidence in a more compact form.

## Optional Live-Inference Trace Evidence

Copy these directories from:

```text
Project Raccoon Ball/Failure Investigations/Incident-005/Image Analysis/live-inference/
```

to:

```text
failure-analysis/incidents/incident-005-live-synthetic-apparent-scale-mismatch/evidence/live-inference/
```

Trace directories present in the incident workspace:

```text
20260531T143045Z__366647f1-314a-4251-88ef-7b3ddad9eef2__c522bf62
20260531T143106Z__2b62f333-95c1-4f15-968a-2bf1805d0326__40af3733
20260531T143126Z__be495924-207e-488c-8f2a-af1a22ccdef0__befd9ac3
20260531T143143Z__40132dcf-580a-40e5-bf19-81a1fde9ad46__fc29e324
20260531T143201Z__3ae2216c-98bc-4980-afd2-6d5dff58ae15__17119b4d
20260531T143217Z__0d14ab78-3c1f-429a-8eda-4b9f5f55e105__d1f21197
20260531T143231Z__bcde42a8-db82-47b3-afdc-27b8074f2079__a116b652
20260531T143246Z__2b22b3a6-57ed-45c1-95ea-4466d5f55d1c__4e167d15
```

The report's six-reading post-ROI-fix live sweep is recorded in the local observation note. If the original six trace directories or a CSV summary are available later, add them under:

```text
failure-analysis/incidents/incident-005-live-synthetic-apparent-scale-mismatch/evidence/live-sweep/
```

and update the report appendix with direct links.
