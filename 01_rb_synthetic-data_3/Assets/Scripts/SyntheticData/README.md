# Unity Synthetic Data Pipeline v1

This bundle contains a first-pass Unity-only pipeline for generating synthetic training images plus:

- `runlog.txt`
- `run.json`
- `samples.csv`

It does **not** do any OpenCV, edge detection, YOLO, or bounding box work.

## Scene objects expected

- `SyntheticDataRunner`
  - `RunControllerBehaviour`
- `CaptureCamera`
  - `Camera`
  - `CameraRigBehaviour`
- `Defender90`
  - `VehicleSceneControllerBehaviour`

## Expected coordinate convention

Camera:
- Position: `[0, 1.5, 0]`
- Rotation: `[27.5, 0, 0]`

Defender 90 baseline:
- Position: sampled from stratified depth-band/lateral-bin cells over the camera-visible footprint on the movement plane.
- Rotation: base `[0, 180, 0]` plus configurable vehicle yaw jitter only.

## What the pipeline writes

Under:

`<OutputRoot>/<RunId>/`

it writes:

- `runlog.txt`
- `images/*.png`
- `manifests/run.json`
- `manifests/samples.csv`

## Recommended setup

1. Create a `RunConfigAsset` from the Create menu:
   - `Create > Raccoon Ball > Synthetic Data > Run Config Asset`
2. Assign it to `RunControllerBehaviour`.
3. Assign the Defender root object to `VehicleSceneControllerBehaviour`.
4. Assign the capture camera to `CameraRigBehaviour`.
5. Attach `RunControllerBehaviour` to an empty runner object.
6. Use the component context menu: `Run Generation`.

## Important note on capture camera

This code manually calls `Camera.Render()` on the assigned camera and captures into a render texture.
A dedicated capture camera is the safest option.

## Placement strategy (first pass)

The generator now uses uniform stratified coverage over the camera-visible footprint:

1. Intersect camera corner rays with the movement plane (`Sweep.MovementPlaneY`) to derive the visible footprint.
2. Convert footprint to camera-relative X/Z coordinates (lateral/depth on the plane).
3. Split usable depth into `Sweep.DepthBandCount` bands.
4. Split each band's lateral extent into `Sweep.LateralBinCount` bins.
5. Probe per-cell acceptance and allocate total samples with acceptance-aware weighting (`Sweep.AcceptanceProbeAttemptsPerCell`, `Sweep.MinSamplesPerFeasibleCell`).
6. Random-sample candidate positions inside each cell until each cell quota is met.
7. Reject candidates when projected Defender bounds clip frame, violate `Sweep.EdgeMarginPx`, or violate projected size constraints.
8. Cells that are infeasible under current constraints are skipped after probing (`Sweep.FeasibilityProbeAttemptsPerCell`).
9. If a cell later exceeds `Sweep.MaxConsecutiveFailuresPerCell` or `Sweep.MaxFailuresPerCell`, its remaining quota is redistributed to eligible cells.
10. At run end, `runlog.txt` includes a final per-cell metrics breakdown: generated samples, failures, failure streaks, final quota, redistribution, status, and last rejection reason.

`VehicleJitter` controls vehicle yaw jitter (`RotY`) after position selection.
`CameraJitter` controls camera vertical position jitter (`PosY`) and pitch jitter (`RotX`) as offsets from the configured camera pose.

## Fail-fast behavior

If image capture or file write fails, the run aborts immediately after logging detailed debug information.
