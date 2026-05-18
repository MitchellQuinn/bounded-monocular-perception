# Live Inference v0.3 Runtime Notes

This directory is the demo-stabilisation rebuild of the live Defender inference app.
The default locator is `background_edge_v1`; ROI-FCN is retained only as an explicit
legacy comparison path.

## Launch

Synthetic camera smoke launch:

```bash
PYTHONPATH=06_live-inference_v0.3/src ./.venv/bin/python -m live_inference.gui.app
```

Real camera example:

```bash
PYTHONPATH=06_live-inference_v0.3/src ./.venv/bin/python \
  -m live_inference.gui.app \
  --camera-source opencv-v4l2 \
  --camera-device /dev/video0 \
  --camera-width 960 \
  --camera-height 600 \
  --camera-fps 80 \
  --device auto
```

## Single-Frame Diagnostic Flow

1. Start Camera.
2. Capture Background when the Defender is absent from the scene.
3. Capture Frame when the Defender is in view.
4. Run Locator.
5. Inspect ROI/bbox/foreground/edge/candidate/chosen debug views.
6. Run Single Inference, or Start Continuous Inference.
7. Enable Record Trace before locator/inference runs when an artifact bundle is needed.

Single-frame mode displays artifacts from the exact captured frame. If an overlay does
not match the displayed image dimensions, the GUI refuses the overlay and logs a warning.

## Draw Mask

Use `Draw Mask` on a loaded preview frame to paint pixels that should be ignored by
model preprocessing. `Erase` removes painted regions, `Apply` commits the edit, and
`Clear Mask` removes the committed mask. The brush size is in source pixels.

In v0.3 the committed mask is applied when a frame is processed: the locator treats
painted pixels as ignored source pixels, and regressor/model preprocessing fills them
using the selected inference fill value. The GUI renders the mask as an overlay and
leaves the preview pixels unchanged.

## Background Handling

`Capture Background` stores a grayscale background and enables background removal.
`Clear Background` removes it. The locator and ROI preprocessing require the background
dimensions to match the source frame; mismatches are skipped and recorded as warnings.

The visible locator parameters are intentionally small:

- background threshold
- minimum foreground area
- Canny low threshold
- Canny high threshold

More parameters can be wired later through the runtime parameter state without expanding
the operator surface.

## Traces

Traces are written under:

```text
06_live-inference_v0.3/live_traces/
```

Trace bundles include the accepted raw frame, grayscale frame, background diff when
available, foreground mask, edge map, candidate/chosen overlays, ROI crop, model input
images, `x_geometry.json`, `locator_result.json`, preprocessing/inference result JSON,
and `trace_manifest.json`.

## Focused Tests

```bash
PYTHONPATH=06_live-inference_v0.3/src ./.venv/bin/python \
  -m unittest discover -s 06_live-inference_v0.3/tests -v
```

## Known Limitations

- `background_edge_v1` is deterministic and inspectable, not a general detector.
- `manual_fixed_roi` and `fixed_center_roi` are emergency/smoke-test fallbacks.
- `roi_fcn_legacy` still requires the ROI-FCN artifact selected in `models/selections/current.toml`.
- Continuous inference processes the newest completed frame and skips duplicate hashes; it does not attempt to process every camera frame.
