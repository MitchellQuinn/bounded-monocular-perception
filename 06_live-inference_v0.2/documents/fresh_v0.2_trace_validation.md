# Fresh v0.2 Trace Validation Protocol

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
