# Fresh v0.2 Trace Validation Protocol

Use the GUI `Guided Diagnostics` panel for the baseline validation pass. It shows
the active profile, the effective stage policy, the current readiness state, the
next recommended action, and the latest trace manifest status.

Recommended GUI flow:

1. Start the camera.
2. Stop continuous inference before diagnostics. Single-frame diagnostics should
   run from a frozen captured frame, not while the continuous worker is active.
3. Capture a frame.
4. Click `Apply Baseline Profile`.
5. Confirm the checklist shows camera ready, inference stopped, frame captured,
   and baseline active.
6. Enable `Record Trace` before the final single-frame inference run.
7. Preview locator input, then run ROI locator only.
8. If the locator is accepted, run single-frame inference with trace enabled.
9. Review the trace path, manifest v0.2 root status, regressor-reached status,
   artifacts, and prediction.

Checklist meanings:

- Camera: the camera controller reports running.
- Inference stopped: continuous inference is not running or requested.
- Frame captured: a frozen single frame is available for diagnostics.
- Baseline active: `baseline_inverted_masked_locator` is the active diagnostic
  profile.
- Trace enabled: `Record Trace` is on.
- Locator accepted: the last ROI locator result was accepted.
- Regressor reached: the last diagnostic result or trace manifest reported
  `distance_orientation_regressor_reached = true`.

A. Clear or isolate:

```bash
./06_live-inference_v0.2/live_traces
```

B. Start v0.2 app.

C. Apply baseline policy:

- `roi_locator_input_mode = inverted`
- `manual mask to ROI locator = true`
- `manual mask to model preprocessing = true`
- `background removal to ROI locator = false`
- `background removal to model preprocessing = false`

D. Capture fresh v0.2 traces.

E. Confirm trace manifests point at v0.2 app root/path.

F. Confirm `final_locator_input.png` is sparse/target-like.

G. Confirm ROI-FCN centre is near the Defender.

H. Confirm ROI is not clipped or is only tolerably clipped.

I. Confirm `distance_orientation_regressor_reached = true` for successful traces.
