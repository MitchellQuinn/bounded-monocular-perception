# Live Inference Pipeline — Initial Architecture Sketch

## Core idea

Use a single application with:

1. One main GUI/orchestration thread
2. One camera-monitor worker thread
3. One inference worker thread

The system should be designed around explicit contracts between components rather than tightly coupling everything together.

The GUI should orchestrate startup/shutdown and display output. The camera monitor should only handle camera frames. The inference worker should only watch for completed images, run inference, and return structured results.

The chosen GUI framework is **PySide6 / Qt for Python**.

This decision is now part of the initial architecture. PySide6 fits the intended structure well because the application needs:

* a responsive desktop GUI
* explicit worker-thread boundaries
* signal-based communication between workers and the GUI
* live image display
* start/stop orchestration controls
* status/error reporting
* a path toward a fullscreen demonstration UI later

The system is deliberately simple, visible, and decoupled. It is demo architecture, not final production architecture.

---

## Resolved architecture decisions

The following decisions are now considered settled for the first implementation:

### GUI framework

Use:

* **PySide6 / Qt for Python**

Do not use, for the first live demo app:

* Tkinter
* Streamlit
* Gradio
* Kivy
* DearPyGui
* browser-based UI

Those may still be useful elsewhere, but the live camera/inference demo should use PySide6.

### Frame handoff

Use:

* **atomic latest-frame file handoff**

The camera worker writes a temporary image file first, then atomically replaces the live/latest frame file.

The inference worker only reads the completed live/latest frame file.

### Frame processing policy

The inference worker should process only the most recent valid completed frame.

The goal is current live output, not processing every captured frame.

Dropped intermediate frames are acceptable.

### Duplicate-frame prevention

The inference worker should compute a fast hash of the image bytes it is about to run through inference.

If the hash matches the last successfully processed frame hash, the inference worker should skip inference for that frame.

This protects against repeated reads of the same latest-frame file.

### Worker separation

The GUI must not directly perform:

* camera capture
* image writing
* image preprocessing
* model inference
* file watching/polling logic

The GUI starts/stops workers, receives signals, displays output, and logs status.

---

## High-level application structure

A single PySide6 application coordinates two background workers.

### Main GUI thread

Responsibilities:

* Start the application
* Provide controls to start/stop the camera monitor
* Provide controls to start/stop the inference worker
* Provide Stop All / Shutdown control
* Display the most recent camera frame or processed frame
* Display inference output
* Display worker status
* Log warnings/errors
* Remain responsive while workers run in the background

The GUI should not directly perform heavy inference or camera capture work.

### Camera worker thread

Responsibilities:

* Connect to the camera
* Pull frames from the camera stream
* Save the newest frame using atomic file handoff
* Avoid filling the directory with unbounded frame files
* Optionally rate-limit capture, though the camera’s FPS may already provide a natural limit
* Emit status/error messages back to the GUI

### Inference worker thread

Responsibilities:

* Monitor the frame directory
* Detect when a completed latest-frame file is available
* Read the candidate image bytes
* Compute a duplicate-detection hash
* Skip duplicate frames
* Decode the image from the same bytes that were hashed
* Run the raw-image-to-model-input preprocessing path
* Generate the required model inputs
* Run model inference
* Emit structured output back to the GUI
* Return to watching for the next frame
* Emit status/error messages back to the GUI

---

## PySide6 structure

The intended PySide6 shape is:

* `QApplication` starts the application
* `LiveDemoWindow(QMainWindow)` owns the visible UI
* `CameraWorker(QObject)` runs in a `QThread`
* `InferenceWorker(QObject)` runs in a separate `QThread`
* worker objects communicate with the GUI only through Qt signals
* the GUI starts/stops workers but does not perform worker tasks itself

The GUI should not subclass `QThread` for worker logic.

Prefer the standard Qt pattern:

* create a `QObject` worker
* create a `QThread`
* move the worker object to the thread
* connect thread and worker signals
* start the thread
* stop the worker cleanly on shutdown

---

## Thread roles

## 1. Main GUI / orchestration thread

Responsibilities:

* Start the application
* Provide controls to start/stop the camera monitor
* Provide controls to start/stop the inference worker
* Display the most recent camera frame or processed frame
* Display inference output
* Log warnings/errors
* Remain responsive while workers run in the background
* Own the visible UI state
* Own application shutdown orchestration

The GUI receives structured inference results from the inference worker.

It displays the fields it recognises and ignores unknown fields while logging a warning.

This allows the output contract to evolve without breaking the UI every time a new debug field or metric is added.

In the PySide6 implementation, the main GUI should probably be built around:

* `QMainWindow` for the primary application window
* `QPushButton` controls for worker start/stop actions
* `QLabel` or a custom widget for frame/image display
* `QLabel` or similar widgets for distance/yaw/latency values
* `QPlainTextEdit`, `QListWidget`, or similar for logs/status output
* Qt signals/slots for receiving worker updates safely from background threads

The GUI should remain a thin orchestration and display layer.

It should not own:

* camera capture logic
* preprocessing logic
* model inference logic
* frame handoff semantics
* file polling semantics

---

## 2. Camera monitor worker thread

Responsibilities:

* Connect to the camera
* Pull frames from the camera stream
* Save the newest frame to the watched directory
* Use atomic write/replace semantics
* Avoid filling the directory with unbounded frame files
* Optionally rate-limit capture, though the camera’s FPS may already provide a natural limit
* Emit status/error messages back to the GUI

The camera is expected to run at maximum resolution, which limits the stream to 50 FPS, roughly 1 frame every 20 ms.

This can be rate-limited later if needed.

The camera monitor should keep the file handoff simple and inspectable.

The first implementation should use a single latest-frame file rather than unique per-frame filenames.

The camera worker should communicate with the GUI using signals such as:

* `status_changed`
* `frame_written`
* `error_occurred`
* `warning_occurred`
* `stopped`

The camera worker should not know about GUI widgets directly.

---

## 3. Inference worker thread

Responsibilities:

* Monitor the frame directory
* Detect when a completed latest-frame file is available
* Read the newest available completed frame
* Compute a fast hash of the image bytes
* Skip the frame if the hash matches the most recent successfully processed frame
* Decode the image from the same bytes that were hashed
* Run the inference preprocessing path
* Generate the required model inputs
* Run model inference
* Emit structured output back to the GUI
* Return to watching for the next frame
* Emit status/error messages back to the GUI

The inference worker should be started once, then loop:

* look for the latest completed frame
* read candidate frame bytes
* hash candidate frame bytes
* compare hash with previous processed hash
* skip if duplicate
* decode image
* preprocess image
* run inference
* emit result
* return to watching

The worker should only process the most recent valid frame.

The important thing is current output, not processing every captured frame.

The inference worker should communicate with the GUI using signals such as:

* `status_changed`
* `result_ready`
* `debug_image_ready`
* `error_occurred`
* `warning_occurred`
* `stopped`

The inference worker should not know about GUI widgets directly.

---

## File-based handoff

For the demo architecture, a file-based handoff between camera and inference is acceptable and useful.

The camera monitor writes image files to a known directory.

The inference worker watches that directory and processes the newest completed image.

This is not a perfect production architecture, but it is good for this phase because it is:

* simple
* inspectable
* easy to debug
* decoupled
* restartable
* resistant to threading/event-loop complexity
* easy for Codex to implement incrementally

If something goes wrong, we can look directly at the directory and see what images are being produced.

This avoids tightly coupling camera capture directly into model inference before the pipeline is stable.

---

## Atomic frame handoff contract

The first implementation should use atomic latest-frame handoff.

### Directory

Default frame directory:

* `./live_frames`

### Files

The camera worker writes:

* temporary frame file: `latest_frame.tmp.png`
* live/latest frame file: `latest_frame.png`

The inference worker reads only:

* `latest_frame.png`

The inference worker ignores:

* temporary files
* partial files
* missing files
* unreadable files
* files whose bytes match the last successfully processed frame hash

### Camera write sequence

The camera worker should follow this sequence:

1. Capture frame from camera.
2. Encode/write frame to `latest_frame.tmp.png`.
3. Confirm the temporary write completed successfully.
4. Atomically replace `latest_frame.png` with `latest_frame.tmp.png`.
5. Emit a `frame_written` signal.
6. Return to capturing the next frame.

The preferred replacement operation is equivalent to:

* write temp file in the same directory
* call atomic replace/rename onto the final filename

The temp file must be on the same filesystem as the final file so the replacement can be atomic.

### Why this matters

The inference worker should never read a half-written image.

With atomic replacement, the inference worker sees either:

* the previous completed frame
* or the new completed frame

It should not see a partially written frame.

### Platform caveat

On some platforms, replacing a file that is currently open for reading may fail.

If the camera worker cannot replace `latest_frame.png` because the file is locked or temporarily unavailable, it should:

* log a warning
* keep or overwrite the temp file on the next attempt
* avoid deleting the current valid `latest_frame.png`
* retry on the next capture loop
* not block the GUI thread

This is a recoverable camera-worker issue.

### No unique-frame filenames initially

The first implementation should not create an unbounded sequence such as:

* `frame_000001.png`
* `frame_000002.png`
* `frame_000003.png`

That can be added later if trace capture becomes important.

The first demo needs current live output, not a permanent frame archive.

---

## Duplicate-frame detection contract

The inference worker should avoid running inference twice on the same completed frame.

### Hash source

The hash should be computed from the exact image bytes that will be decoded and passed into preprocessing.

Preferred sequence:

1. Open `latest_frame.png`.
2. Read its full contents into memory as bytes.
3. Compute hash from those bytes.
4. Compare hash with `last_processed_frame_hash`.
5. If duplicate, skip.
6. If new, decode those same bytes into an image.
7. Run preprocessing and inference.
8. Store the hash as `last_processed_frame_hash` only after inference completes successfully.

This avoids a subtle mismatch where the worker hashes one version of the file but decodes another version after the file has been replaced.

### Hash type

Use a fast, simple, built-in hash for the first implementation.

Recommended:

* `hashlib.blake2b` with a short digest, such as 16 bytes

This is not for cryptographic security.

It is only for duplicate-frame detection.

If hashing becomes a bottleneck later, this can be replaced with a faster non-cryptographic hash such as xxHash, but the first implementation should avoid unnecessary dependencies.

### Duplicate skip rule

If:

* `candidate_hash == last_processed_frame_hash`

then:

* do not run preprocessing
* do not run inference
* increment `frames_skipped_duplicate`
* emit or log duplicate-skip status only at a low/no-spam level
* return to watching for a new frame

### Important caveat

Hash-only duplicate detection cannot distinguish between:

* accidentally reading the same file twice
* two genuinely separate camera frames that are byte-identical

For this demo architecture, that is acceptable.

If this becomes a problem later, the camera worker can add a frame metadata sidecar or frame counter.

---

## Contracts

The most important design principle is that the system should be contract-driven.

---

## Camera-to-inference contract

The camera monitor provides a completed latest-frame image file.

The minimum frame handoff contract is:

* image file path
* completed image bytes available at that path
* atomic replacement semantics
* optional timestamp
* optional frame metadata

The inference worker should not need to know anything about:

* GUI internals
* camera widget state
* camera object internals
* camera capture loop implementation

It only needs a valid completed image matching the expected raw image input assumptions.

---

## Inference input contract

The inference pipeline takes a raw image and generates the model’s required inputs.

For the tri-stream model, that means:

* `x_distance_image`
* `x_orientation_image`
* `x_geometry`

The inference path must reproduce the same semantic preprocessing contract used during training.

### Distance image

Expected properties:

* fixed ROI / fixed canvas
* spatially unscaled
* preserves apparent target size
* brightness-normalised
* dark vehicle detail on white background

### Orientation image

Expected properties:

* target-centred
* scaled/normalised by target or foreground extent
* raw/detail-preserving
* no brightness normalisation

### Geometry

Expected properties:

* existing bbox / ROI / context vector
* same schema expected by the model

Inference starts from a raw image.

The file path may be used as metadata, but the preferred inference-worker behaviour is to read the image bytes once, hash those bytes, decode from those bytes, and pass the decoded image into preprocessing.

---

## Inference-to-GUI contract

The inference worker emits a structured result object.

Initial fields:

* `input_image_path`
* `input_image_hash`
* `timestamp_utc`
* `predicted_distance_m`
* `predicted_yaw_sin`
* `predicted_yaw_cos`
* `predicted_yaw_deg`
* `inference_time_ms`
* `preprocessing_time_ms`
* `total_time_ms`
* `roi_metadata`
* `debug_paths`
* `warnings`

The GUI displays the fields it understands.

Unknown fields should not crash the GUI.

Unknown fields should be ignored and logged as warnings.

To avoid log spam, the GUI may warn only once per unknown field name per session.

This lets the inference output evolve without constantly breaking the display layer.

A suitable first implementation would use a Python `dataclass` or plain dictionary for the inference result.

---

## Suggested result schema

The first inference result object should contain:

    InferenceResult:
      input_image_path: str
      input_image_hash: str
      timestamp_utc: str
      predicted_distance_m: float
      predicted_yaw_sin: float
      predicted_yaw_cos: float
      predicted_yaw_deg: float
      inference_time_ms: float
      preprocessing_time_ms: float | None
      total_time_ms: float | None
      roi_metadata: dict | None
      debug_paths: dict | None
      warnings: list[str]

### Field notes

`input_image_path`

Path to the frame file used as the source.

`input_image_hash`

Hash of the exact image bytes used for this inference run.

`timestamp_utc`

Timestamp generated by the inference worker when the frame is accepted for processing.

`predicted_distance_m`

Predicted distance in metres.

`predicted_yaw_sin`

Model output yaw sine.

`predicted_yaw_cos`

Model output yaw cosine.

`predicted_yaw_deg`

Decoded yaw angle in degrees.

`inference_time_ms`

Time spent in model inference.

`preprocessing_time_ms`

Time spent generating model inputs from the raw image.

`total_time_ms`

End-to-end inference-worker processing time for this frame.

`roi_metadata`

Optional dictionary containing ROI/crop/bbox/context metadata.

`debug_paths`

Optional dictionary containing paths to generated debug images, if debug image saving is enabled.

`warnings`

List of warnings generated during processing.

---

## Suggested worker status schema

Worker status messages should be structured.

    WorkerStatus:
      worker_name: "camera" | "inference"
      state: str
      message: str
      timestamp_utc: str

The GUI should use these to update visible status indicators.

---

## Suggested worker error schema

Worker error messages should be structured.

    WorkerError:
      worker_name: "camera" | "inference"
      error_type: str
      message: str
      recoverable: bool
      timestamp_utc: str

Recoverable errors should be logged without necessarily stopping the app.

Non-recoverable errors should place the relevant worker into `ERROR` state.

---

## Worker lifecycle and state model

Both workers should use the same basic state vocabulary.

Allowed states:

* `STOPPED`
* `STARTING`
* `RUNNING`
* `STOPPING`
* `ERROR`

### Camera worker state transitions

Expected camera worker transitions:

    STOPPED -> STARTING -> RUNNING
    RUNNING -> STOPPING -> STOPPED
    RUNNING -> ERROR
    ERROR -> STOPPED

### Inference worker state transitions

Expected inference worker transitions:

    STOPPED -> STARTING -> RUNNING
    RUNNING -> STOPPING -> STOPPED
    RUNNING -> ERROR
    ERROR -> STOPPED

### Start behaviour

When a worker is started:

* GUI requests worker start
* worker enters `STARTING`
* worker performs setup
* if setup succeeds, worker enters `RUNNING`
* if setup fails, worker enters `ERROR`

### Stop behaviour

When a worker is stopped:

* GUI requests worker stop
* worker enters `STOPPING`
* worker exits its loop safely
* worker releases resources
* worker emits `stopped`
* worker enters `STOPPED`

### Inference stop semantics

The inference worker should not be killed halfway through a model call.

If stop is requested while inference is already running:

* finish the current inference pass if possible
* emit the result or error as appropriate
* then stop

This is safer than interrupting model execution.

### Camera stop semantics

If stop is requested while the camera worker is running:

* exit the capture loop
* release the camera
* avoid leaving partial temp files if practical
* emit `stopped`

---

## Shutdown contract

The application should support:

* Stop Camera
* Stop Inference
* Stop All / Shutdown

### Stop Camera

Expected sequence:

1. Disable or update camera controls in the GUI.
2. Request camera worker stop.
3. Camera worker exits capture loop.
4. Camera worker releases camera.
5. Camera worker emits stopped.
6. GUI updates camera state to stopped.

### Stop Inference

Expected sequence:

1. Disable or update inference controls in the GUI.
2. Request inference worker stop.
3. Inference worker finishes current inference pass if one is already underway.
4. Inference worker exits watch loop.
5. Inference worker emits stopped.
6. GUI updates inference state to stopped.

### Stop All / Shutdown

Expected sequence:

1. Request both workers stop.
2. Let each worker shut down cleanly.
3. Release camera and inference resources.
4. Close the application only after workers have stopped, or after a clearly logged shutdown timeout.

The GUI should never silently abandon running worker threads.

---

## Configuration contract

The first implementation should avoid hardcoded paths and settings scattered throughout the code.

Use a central configuration object.

Suggested config:

    LiveInferenceConfig:
      frame_dir: str
      latest_frame_filename: str
      temp_frame_filename: str
      camera_index: int
      camera_width: int
      camera_height: int
      camera_fps: int | None
      inference_poll_interval_ms: int
      duplicate_hash_skip_enabled: bool
      model_path: str
      device: "cpu" | "cuda"
      save_debug_images: bool
      debug_output_dir: str | None

Suggested defaults:

    frame_dir: "./live_frames"
    latest_frame_filename: "latest_frame.png"
    temp_frame_filename: "latest_frame.tmp.png"
    camera_index: 0
    camera_width: use camera/default or configured value
    camera_height: use camera/default or configured value
    camera_fps: None initially
    inference_poll_interval_ms: 10
    duplicate_hash_skip_enabled: true
    device: "cuda"
    save_debug_images: false
    debug_output_dir: "./live_debug"

### Config principles

The config should define:

* where frames are written
* what filenames are used
* which camera to use
* how frequently inference polls
* whether duplicate hash skipping is enabled
* where the model artifact lives
* whether inference runs on CPU or CUDA
* whether debug images are saved

The GUI may expose some of these later.

For the first implementation, they can be loaded from a config file, command-line args, or a simple in-code config object.

---

## Observability and debug counters

The GUI should make the live system observable.

This is not just a visual demo surface.

It is also a diagnostic surface.

### Camera counters

Minimum camera status/counters:

* `state`
* `frames_captured`
* `frames_written`
* `frame_write_failures`
* `last_frame_write_time_utc`
* `last_frame_path`
* `last_error`

### Inference counters

Minimum inference status/counters:

* `state`
* `frames_seen`
* `frames_processed`
* `frames_skipped_duplicate`
* `frames_failed_read`
* `frames_failed_decode`
* `frames_failed_preprocess`
* `frames_failed_inference`
* `last_input_hash`
* `last_inference_time_ms`
* `last_preprocessing_time_ms`
* `last_total_time_ms`
* `last_result_time_utc`
* `last_error`

### GUI log levels

Use simple log levels:

* `INFO`
* `WARNING`
* `ERROR`

### Log destinations

First implementation:

* visible GUI log panel
* stdout/stderr for development

Later optional extension:

* structured log file
* session-level run log
* debug artifact directory

Do not overbuild structured disk logging in the first GUI pass.

---

## Error handling expectations

The application should distinguish recoverable per-frame errors from worker-fatal errors.

### Recoverable camera errors

Examples:

* temporary frame write failure
* atomic replace failure due to file lock
* a single failed capture
* temporary camera read glitch

Expected behaviour:

* log warning/error
* increment counter
* keep worker alive if practical
* try again on next loop

### Non-recoverable camera errors

Examples:

* camera cannot be opened
* camera disconnect cannot be recovered
* invalid camera configuration prevents startup

Expected behaviour:

* emit `WorkerError`
* set state to `ERROR`
* stop camera worker

### Recoverable inference errors

Examples:

* latest frame file missing
* latest frame temporarily unreadable
* duplicate frame detected
* image decode failure
* one-frame preprocessing failure

Expected behaviour:

* log warning/error
* increment counter
* skip that frame
* continue polling

### Non-recoverable inference errors

Examples:

* model artifact cannot be loaded
* incompatible model/preprocessing contract
* CUDA/device setup failure that cannot be recovered
* persistent inference engine failure

Expected behaviour:

* emit `WorkerError`
* set state to `ERROR`
* stop inference worker

---

## GUI framework decision

The selected GUI framework is:

**PySide6 / Qt for Python**

This is preferred over Tkinter, Streamlit, Gradio, Kivy, DearPyGui, or a browser-based UI for the initial live inference demo.

Reasons:

* mature desktop GUI framework
* cross-platform
* good support for worker-thread architecture
* signal/slot communication maps well to the contract-driven design
* suitable for live frame display
* suitable for status-heavy engineering tools
* can later be polished into a fullscreen demo UI
* likely to be implementable incrementally by Codex
* cleaner licensing position than PyQt for a potentially employer-facing or partner-facing project

PySide6 should be used to enforce boundaries rather than blur them.

The GUI should not become the place where camera, preprocessing, and inference logic get mixed together.

---

## GUI expectations

The GUI should be simple initially.

Minimum controls:

* Start Camera
* Stop Camera
* Start Inference
* Stop Inference
* Stop All / Shutdown

Minimum display:

* latest frame or processed frame
* predicted distance
* predicted yaw
* inference timing / latency
* camera worker status
* inference worker status
* warning/error log

The GUI does not need to be elegant at first.

It needs to force clean boundaries and make the live system observable.

### Likely first layout

A likely first layout:

* left side: latest raw or processed frame
* right side: numeric inference output and worker status
* bottom: logs/warnings/errors
* top or side: start/stop controls

### First display mode

The first display mode should show:

* latest raw frame or latest accepted frame
* predicted distance
* predicted yaw
* preprocessing time
* inference time
* total time
* camera state
* inference state

### Later display modes

Possible later display modes:

* raw camera frame only
* processed distance image
* processed orientation image
* side-by-side raw + distance + orientation
* raw frame with ROI/debug overlay
* latency and FPS counters
* simplified “demo mode” view with fewer engineering details

The later fullscreen demo layout should be added only after the pipeline is stable.

---

## Debug image policy

Debug images are useful but should not be mandatory in the first running system.

Initial default:

* `save_debug_images = false`

When enabled, the inference worker may save:

* generated distance image
* generated orientation image
* ROI/crop debug image
* overlay/debug visualization

Debug output directory:

* `./live_debug`

Suggested debug naming should include at least one of:

* timestamp
* frame hash
* short hash prefix

Example debug names:

* `20260430T120000Z_ab12cd34_distance.png`
* `20260430T120000Z_ab12cd34_orientation.png`
* `20260430T120000Z_ab12cd34_overlay.png`

Debug paths should be included in the `debug_paths` field of `InferenceResult`.

---

## Important architectural principle

The purpose of putting a GUI/application boundary in place is to force separation of concerns:

* camera capture is one component
* inference is one component
* display/orchestration is one component

Each part should be independently replaceable.

This prevents the current kind of pipeline friction where assumptions live in notebooks, scattered scripts, and implicit mental context.

The ideal future architecture would have model artifacts carry or reference their preprocessing/inference contract directly.

That is probably not a task for right now, but the live inference pipeline should move in that direction by making contracts explicit.

---

## First implementation boundary

The first implementation is not expected to be elegant or feature-complete.

It is expected to prove:

* PySide6 application starts cleanly
* GUI remains responsive
* camera worker can start and stop
* inference worker can start and stop
* camera worker writes frames using atomic latest-frame handoff
* inference worker reads only completed latest-frame files
* inference worker skips duplicate frame hashes
* inference worker emits structured results
* GUI displays recognised result fields
* GUI logs warnings/errors
* workers shut down cleanly

The first implementation should not try to solve:

* production deployment
* beautiful fullscreen UI
* permanent frame archiving
* multi-camera support
* full structured logging system
* cloud upload
* database storage
* model retraining
* automatic camera calibration
* all possible debug visualization modes

The first implementation should be a well-bounded engineering demo.

---

## Likely development sequence

1. Finish current synthetic corpus generation.
2. Continue with tri-stream inference pipeline v0.4 from raw image input.
3. Confirm offline single-image/small-corpus inference works.
4. Define the file-based camera-to-inference handoff.
5. Build a minimal camera monitor that writes frames to a watched directory.
6. Implement atomic latest-frame writing.
7. Build an inference worker that watches the latest-frame file.
8. Add duplicate-frame hash skipping in the inference worker.
9. Confirm inference worker processes only new completed frames.
10. Build a minimal PySide6 GUI/orchestrator that starts/stops both workers and displays results.
11. Add debug outputs and status display.
12. Add worker lifecycle cleanup and robust shutdown behaviour.
13. Add frame display modes and result-contract logging.
14. Only then refine layout/fullscreen demo behaviour.

---

## Open questions

Resolved:

* GUI framework: **PySide6 / Qt for Python**
* Frame handoff: **atomic latest-frame file**
* Inference should process only the most recent valid frame
* Inference should skip duplicate frames by hashing image bytes
* First implementation should not use unbounded unique frame files

Still open:

* What frame rate is realistic once preprocessing and inference are included?
* How much latency is acceptable for the demo?
* Should the first GUI display raw camera frames, processed frames, or both?
* Should the first GUI include debug image panels, or should those be added after the basic worker loop is stable?
* Should the first app log only to GUI/stdout, or also write structured logs to disk?
* Should the model artifact carry or reference its preprocessing contract directly?
* Should a later version add frame metadata sidecars or a frame counter?

---

## Current working architecture summary

A single PySide6 application coordinates two background workers.

Main GUI thread:

* starts/stops workers
* displays latest result
* logs status
* remains responsive
* does not perform camera capture or inference directly

Camera worker:

* runs in a `QThread`
* reads camera frames
* writes the latest frame using atomic handoff
* writes temp file first
* atomically replaces the live/latest frame file
* emits status/error signals

Inference worker:

* runs in a separate `QThread`
* watches the latest-frame file
* reads completed frame bytes
* computes a hash of those bytes
* skips duplicate hashes
* decodes the same bytes it hashed
* runs raw-image-to-tri-stream preprocessing
* runs model inference
* emits structured result to GUI

The system is deliberately simple, visible, and decoupled.

It is demo architecture, not final production architecture.

PySide6 is now the selected GUI framework because it fits the worker-thread, signal-driven, contract-based structure of the live inference demonstration.

Atomic latest-frame handoff is now the selected file-transfer pattern because it reduces the risk of the inference worker reading half-written images.

Hash-based duplicate-frame skipping is now part of the inference-worker contract because it prevents accidental repeated inference on the same completed frame.