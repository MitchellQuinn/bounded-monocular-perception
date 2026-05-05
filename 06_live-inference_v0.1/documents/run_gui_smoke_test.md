# Live GUI Smoke Test

## Setup Assumptions

- Run commands from the repository root.
- Use the repository virtual environment Python: `./.venv/bin/python`.
- Make `06_live-inference_v0.1/src` importable with `PYTHONPATH`.
- PySide6, Torch, OpenCV, NumPy, and the model runtime dependencies are installed in the venv.

## Model Artifacts

The GUI uses the current live-local model selection by default:

```bash
06_live-inference_v0.1/models/selections/current.toml
```

That selection should point at staged artifact directories under:

```bash
06_live-inference_v0.1/models/
```

The default launch uses the device values in `current.toml`. The supported
device policies are:

- `auto`: use CUDA when Torch reports CUDA is available, otherwise CPU.
- `cuda`: require CUDA and fail clearly when CUDA is unavailable.
- `cpu`: force CPU execution. This is useful on mini PCs, but inference may be
  slower than on the GPU path.

To intentionally override both model devices for a local smoke run, pass
`--device auto`, `--device cuda`, or `--device cpu`.

## Synthetic Camera Images

The default synthetic camera config is:

```bash
06_live-inference_v0.1/config/synthetic_camera.toml.example
```

It reads a small demo source directory:

```bash
06_live-inference_v0.1/demo/synthetic_camera_source/
```

The demo images are copied from:

```bash
05_inference-v0.4-ts/input/def90_synth_v023-validation-shuffled/images/
```

Do not use symlinks. Keep this directory small so manual GUI startup remains fast.

## Launch Command

From the repository root:

```bash
PYTHONPATH=06_live-inference_v0.1/src ./.venv/bin/python -m live_inference.gui.app
```

Optional auto-start smoke command:

```bash
PYTHONPATH=06_live-inference_v0.1/src ./.venv/bin/python -m live_inference.gui.app --auto-start-camera --auto-start-inference
```

Mini PC / CPU auto-start smoke command:

```bash
PYTHONPATH=06_live-inference_v0.1/src ./.venv/bin/python -m live_inference.gui.app --device cpu --auto-start-camera --auto-start-inference
```

Useful manual-test options:

```bash
--synthetic-camera-config 06_live-inference_v0.1/config/synthetic_camera.toml.example
--model-selection 06_live-inference_v0.1/models/selections/current.toml
--frame-interval-ms 250
--debug
```

## Real Camera Smoke Test

The OpenCV/V4L2 source targets `/dev/video0` by default and requests
960x600 at 80 fps with YUYV 4:2:2 input, encoded to PNG for the existing
latest-frame GUI preview.

Proof-of-life commands:

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video0 --all
```

Live GUI command:

```bash
PYTHONPATH=06_live-inference_v0.1/src ./.venv/bin/python \
  -m live_inference.gui.app \
  --camera-source opencv-v4l2 \
  --camera-device /dev/video0 \
  --camera-width 960 \
  --camera-height 600 \
  --camera-fps 80 \
  --device auto \
  --auto-start-camera \
  --auto-start-inference
```

Do not use high-bandwidth `ffplay` raw preview over RDP as a normal workflow.
Prefer headless capture checks or the app's compressed and scaled GUI preview.
Full resolution 1920x1200 at 50 fps is for later testing, not the default.

## Manual Steps

1. Launch the GUI with the command above.
2. Click `Start Camera`.
3. Confirm the frame preview updates.
4. Click `Start Inference`.
5. Confirm the distance, yaw, and timing labels update.
6. Click `Stop Inference`, then `Stop Camera`.
7. Start both again, then click `Stop All`.
8. Close the window and confirm the process exits cleanly.

## Expected Behaviour

- The synthetic camera repeatedly publishes the copied demo images into
  `06_live-inference_v0.1/live_frames/`.
- The preview shows the latest published frame.
- Inference reads the latest handoff frame and updates the prediction labels.
- Stop buttons request clean worker shutdown without terminating threads abruptly.

## Known Caveats

- The frame preview is a live latest-frame preview, not guaranteed to match the exact inference result frame.
- Inference inspection/debug views are not implemented yet.
- Parameter tuning UI is not implemented yet.
- Real camera support currently starts with OpenCV/V4L2 at 960x600 and 80 fps.
