# ChArUco Camera Calibration

Employer-facing PySide6/OpenCV calibration tool for Project Raccoon Ball.

This application captures pose-diverse ChArUco calibration frames, solves camera
intrinsics, and exports JSON/YAML calibration artifacts for later runtime use.
It is deliberately separate from the inference pipeline: it does not use ROI-FCN,
distance/yaw models, synthetic-data preprocessing, background removal, masking,
or live inference logic.

## Board Setup

The known measured board properties are:

```text
board_type = "charuco"
checker_size_m = 0.015
marker_size_m = 0.011
```

The calib.io print label says `15x10`, but OpenCV ChArUco board construction
requires square counts as `squares_x` and `squares_y`. Do not infer these from
portrait/landscape appearance alone. Set them explicitly in:

```text
<CHARUCO_CALIBRATION_DIR>/config/charuco_board.example.toml
```

The ArUco dictionary is also explicit by design. The example config contains
`<ARUCO_DICTIONARY>` and calibration will not proceed until that placeholder is
replaced with a real OpenCV predefined dictionary such as `DICT_5X5_100`.

## Launch GUI

```bash
PYTHONPATH=<CHARUCO_CALIBRATION_DIR>/src ./.venv/bin/python -m rb_camera_calibration.gui.app \
  --board-config <CHARUCO_CALIBRATION_DIR>/config/charuco_board.example.toml \
  --camera-config <CHARUCO_CALIBRATION_DIR>/config/camera.example.toml \
  --capture-policy <CHARUCO_CALIBRATION_DIR>/config/capture_policy.example.toml
```

From the repository root, that is:

```bash
PYTHONPATH=charuco-calibration/src ./.venv/bin/python -m rb_camera_calibration.gui.app \
  --board-config charuco-calibration/config/charuco_board.example.toml \
  --camera-config charuco-calibration/config/camera.example.toml \
  --capture-policy charuco-calibration/config/capture_policy.example.toml
```

Use the repository virtual environment Python (`./.venv/bin/python`) for all
commands.

## Camera Mode

The example camera config is set for the Arducam B0495 AR0234 at `1920x1200`.
Calibrate using the same camera resolution and sensor mode as the live runtime.
If live inference is launched with a different width/height, update
`config/camera.example.toml` before collecting calibration frames.

## Probe Dictionary First

Run the dictionary probe against a saved frame before a serious capture session:

```bash
PYTHONPATH=<CHARUCO_CALIBRATION_DIR>/src ./.venv/bin/python -m rb_camera_calibration.detection.dictionary_probe \
  --image path/to/frame.png
```

The probe reports marker counts, detected IDs, confidence, and the best
candidate among common OpenCV predefined dictionaries. It is diagnostic only;
the board config must still be edited to select the dictionary explicitly.

## Calibration Workflow

1. Start the configured OpenCV camera.
2. Probe the dictionary if the printed label is incomplete.
3. Confirm `aruco_dictionary`, `squares_x`, and `squares_y`.
4. Enable auto-capture and move the board through varied locations, scales, roll
   angles, and tilts.
5. Inspect rejected reasons and pose suggestions while collecting frames.
6. Run calibration.
7. Inspect per-view reprojection errors.
8. Remove poor frames if needed and recalibrate.
9. Export the calibration artifact.

## Resume Or Merge Runs

In the GUI, use **Merge All Runs** to copy accepted frames from every discovered
run into the current session. Stop the camera before merging so the manifest and
pose-coverage state stay consistent.

For an iterative session, relaunch with the same run directory:

```bash
PYTHONPATH=charuco-calibration/src ./.venv/bin/python -m rb_camera_calibration.gui.app \
  --board-config charuco-calibration/config/charuco_board.example.toml \
  --camera-config charuco-calibration/config/camera.example.toml \
  --capture-policy charuco-calibration/config/capture_policy.example.toml \
  --session-root charuco-calibration/calibration_runs/260519-1331_calibio_charuco_15mm_mdf
```

The GUI loads the existing `session_manifest.json`, shows the accepted frames,
and continues numbering new accepted images after the existing set.

To merge accepted frames from multiple runs into one new run:

```bash
PYTHONPATH=charuco-calibration/src ./.venv/bin/python -m rb_camera_calibration.capture.merge_sessions \
  --board-config charuco-calibration/config/charuco_board.example.toml \
  --camera-config charuco-calibration/config/camera.example.toml \
  --capture-policy charuco-calibration/config/capture_policy.example.toml \
  --output-session-root charuco-calibration/calibration_runs/merged_charuco_session \
  charuco-calibration/calibration_runs/run_a \
  charuco-calibration/calibration_runs/run_b
```

Then launch the GUI with `--session-root charuco-calibration/calibration_runs/merged_charuco_session`
to continue capture or calibrate/export from the merged accepted-frame set.

Session outputs are written under `<CHARUCO_CALIBRATION_DIR>/calibration_runs/`
by default, no matter where you launch the app from. You can override this with
`--session-root` if needed. For example:

```text
charuco-calibration/calibration_runs/
  260518-1430_calibio_charuco_15mm_mdf/
    session_manifest.json
    accepted/
    rejected_samples/
    calibration_result.json
    calibration_result.yaml
    per_view_reprojection_errors.csv
    artifact_manifest.json
```

## Architecture

`src/rb_camera_calibration/contracts.py` is the boundary layer between the GUI
and functional calibration logic. It avoids PySide6, OpenCV, NumPy, and camera
imports and defines serialisable dataclasses/enums/protocols for board config,
camera config, detections, quality metrics, capture decisions, calibration
requests/results, worker state, and exported artifacts.

OpenCV and NumPy details are kept inside camera, detection, capture-quality, and
calibration modules. Qt widgets orchestrate the workflow through signals/slots
and do not own calibration logic.

## Tests

```bash
PYTHONPATH=<CHARUCO_CALIBRATION_DIR>/src ./.venv/bin/python -m pytest <CHARUCO_CALIBRATION_DIR>/tests
```
