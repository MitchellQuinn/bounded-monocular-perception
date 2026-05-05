"""Minimal runnable PySide6 application for the live inference GUI shell."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
import sys

from live_inference.runtime.device import normalize_torch_device_policy

DEFAULT_FRAME_INTERVAL_MS = 250
DEFAULT_INFERENCE_POLL_INTERVAL_MS = 10
DEFAULT_SYNTHETIC_SOURCE_DIR = Path("demo/synthetic_camera_source")
DEFAULT_SYNTHETIC_OUTPUT_DIR = Path("live_frames")
DEFAULT_CAMERA_SOURCE = "synthetic"
OPENCV_V4L2_CAMERA_SOURCE = "opencv-v4l2"
DEFAULT_CAMERA_DEVICE = "/dev/video0"
DEFAULT_CAMERA_WIDTH_PX = 960
DEFAULT_CAMERA_HEIGHT_PX = 600
DEFAULT_CAMERA_FPS = 80
DEFAULT_CAMERA_PIXEL_FORMAT = "YUYV"
DEFAULT_CAMERA_ENCODING = "png"
DEFAULT_CAMERA_NAME = "Arducam B0495 AR0234"


@dataclass(frozen=True)
class LiveInferenceGuiContext:
    """Composed worker controllers for the GUI shell."""

    camera_controller: object
    inference_controller: object
    model_selection_path: Path
    live_inference_config: object
    camera_source: str
    synthetic_camera_config_path: Path | None
    synthetic_camera_base_dir: Path
    synthetic_camera_config: object | None
    distance_orientation_device: str
    roi_fcn_device: str


@dataclass(frozen=True)
class _RuntimeDependencies:
    synthetic_camera_config_cls: type
    synthetic_camera_publisher_cls: type
    load_synthetic_camera_config: Callable[[Path], object]
    opencv_v4l2_camera_publisher_cls: type
    atomic_frame_handoff_writer_cls: type
    live_inference_config_cls: type
    torch_tri_stream_inference_engine_cls: type
    latest_frame_handoff_reader_cls: type
    inference_frame_selector_cls: type
    inference_processing_core_cls: type
    load_live_model_manifest: Callable[..., object]
    load_model_selection: Callable[[Path], object]
    roi_fcn_locator_cls: type
    tri_stream_live_preprocessor_cls: type
    camera_worker_cls: type
    inference_worker_cls: type
    worker_thread_controller_cls: type


def default_model_selection_path() -> Path:
    """Return the default current live-local model selection path."""
    return _live_project_root() / "models/selections/current.toml"


def default_synthetic_camera_config_path() -> Path:
    """Return the default synthetic camera smoke config path."""
    return _live_project_root() / "config/synthetic_camera.toml.example"


def build_live_inference_gui_context(
    *,
    camera_source: str = DEFAULT_CAMERA_SOURCE,
    model_selection_path: Path | None = None,
    synthetic_camera_config_path: Path | None = None,
    selection_path: Path | None = None,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    camera_device: str | int = DEFAULT_CAMERA_DEVICE,
    camera_width_px: int = DEFAULT_CAMERA_WIDTH_PX,
    camera_height_px: int = DEFAULT_CAMERA_HEIGHT_PX,
    camera_fps: int = DEFAULT_CAMERA_FPS,
    camera_pixel_format: str = DEFAULT_CAMERA_PIXEL_FORMAT,
    camera_encoding: str = DEFAULT_CAMERA_ENCODING,
    camera_name: str = DEFAULT_CAMERA_NAME,
    log_v4l2_controls_at_startup: bool = False,
    device: str | None = None,
    frame_interval_ms: int = DEFAULT_FRAME_INTERVAL_MS,
    inference_poll_interval_ms: int = DEFAULT_INFERENCE_POLL_INTERVAL_MS,
    dependency_loader: Callable[[], _RuntimeDependencies] | None = None,
) -> LiveInferenceGuiContext:
    """Build the selected camera to tri-stream inference pipeline."""
    deps = (dependency_loader or _load_runtime_dependencies)()

    resolved_camera_source = _camera_source(camera_source)
    project_root = _live_project_root()
    resolved_selection_path = _resolve_path(
        model_selection_path or selection_path or default_model_selection_path()
    )
    synthetic_base_dir = project_root
    resolved_camera_config_path: Path | None = None
    synthetic_config: object | None = None

    if resolved_camera_source == DEFAULT_CAMERA_SOURCE:
        resolved_camera_config_path = _resolve_synthetic_camera_config_path(
            synthetic_camera_config_path
        )
        synthetic_config = _synthetic_camera_config_from_path_or_default(
            deps,
            resolved_camera_config_path,
        )
        synthetic_config = _override_synthetic_camera_config(
            synthetic_config,
            source_dir=source_dir,
            output_dir=output_dir,
            frame_interval_ms=frame_interval_ms,
        )
        publisher = deps.synthetic_camera_publisher_cls(
            synthetic_config,
            base_dir=synthetic_base_dir,
        )
        live_config = deps.live_inference_config_cls(
            frame_dir=publisher.output_dir,
            latest_frame_filename=synthetic_config.latest_frame_filename,
            temp_frame_filename=synthetic_config.temp_frame_filename,
            inference_poll_interval_ms=inference_poll_interval_ms,
        )
    else:
        camera_frame_dir = _resolve_camera_frame_dir(project_root, output_dir)
        camera_extension = _frame_extension(camera_encoding)
        live_config = deps.live_inference_config_cls(
            frame_dir=camera_frame_dir,
            latest_frame_filename=f"latest_frame.{camera_extension}",
            temp_frame_filename=f"latest_frame.tmp.{camera_extension}",
            inference_poll_interval_ms=inference_poll_interval_ms,
        )
        writer = deps.atomic_frame_handoff_writer_cls(live_config)
        publisher = deps.opencv_v4l2_camera_publisher_cls(
            handoff_writer=writer,
            device=camera_device,
            width_px=camera_width_px,
            height_px=camera_height_px,
            fps=camera_fps,
            pixel_format=camera_pixel_format,
            encoding=camera_encoding,
            camera_name=camera_name,
            auto_open=False,
            log_v4l2_controls_at_startup=log_v4l2_controls_at_startup,
        )

    selection = deps.load_model_selection(resolved_selection_path)
    device_override = (
        normalize_torch_device_policy(device) if device is not None else None
    )
    distance_orientation_device = normalize_torch_device_policy(
        device_override or selection.distance_orientation_device
    )
    roi_fcn_device = normalize_torch_device_policy(
        device_override or selection.roi_fcn_device
    )
    manifest = deps.load_live_model_manifest(
        selection.distance_orientation_root,
        roi_locator_root=selection.roi_fcn_root,
    )

    reader = deps.latest_frame_handoff_reader_cls(live_config)
    selector = deps.inference_frame_selector_cls(
        reader,
        duplicate_hash_skip_enabled=live_config.duplicate_hash_skip_enabled,
    )
    roi_locator = deps.roi_fcn_locator_cls(selection.roi_fcn_root, device=roi_fcn_device)
    preprocessor = deps.tri_stream_live_preprocessor_cls(
        model_manifest=manifest,
        roi_locator=roi_locator,
    )
    engine = deps.torch_tri_stream_inference_engine_cls(
        model_root=selection.distance_orientation_root,
        model_manifest=manifest,
        device=distance_orientation_device,
    )
    core = deps.inference_processing_core_cls(selector, preprocessor, engine)

    camera_worker = deps.camera_worker_cls(publisher)
    inference_worker = deps.inference_worker_cls(
        core,
        poll_interval_ms=live_config.inference_poll_interval_ms,
    )
    return LiveInferenceGuiContext(
        camera_controller=deps.worker_thread_controller_cls(camera_worker),
        inference_controller=deps.worker_thread_controller_cls(inference_worker),
        model_selection_path=resolved_selection_path,
        live_inference_config=live_config,
        camera_source=resolved_camera_source,
        synthetic_camera_config_path=resolved_camera_config_path,
        synthetic_camera_base_dir=synthetic_base_dir,
        synthetic_camera_config=synthetic_config,
        distance_orientation_device=distance_orientation_device,
        roi_fcn_device=roi_fcn_device,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)

    try:
        from PySide6.QtCore import QTimer  # noqa: PLC0415
        from PySide6.QtWidgets import QApplication  # noqa: PLC0415

        from live_inference.gui.main_window import LiveInferenceMainWindow  # noqa: PLC0415

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv[:1])

        context = build_live_inference_gui_context(
            camera_source=args.camera_source,
            model_selection_path=args.model_selection,
            synthetic_camera_config_path=args.synthetic_camera_config,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            camera_device=args.camera_device,
            camera_width_px=args.camera_width,
            camera_height_px=args.camera_height,
            camera_fps=args.camera_fps,
            camera_pixel_format=args.camera_pixel_format,
            camera_encoding=args.camera_encoding,
            camera_name=args.camera_name,
            log_v4l2_controls_at_startup=args.log_v4l2_controls_at_startup,
            device=args.device,
            frame_interval_ms=args.frame_interval_ms,
            inference_poll_interval_ms=args.inference_poll_interval_ms,
        )
        if args.debug:
            _print_launch_debug(context)

        window = LiveInferenceMainWindow(
            camera_controller=context.camera_controller,
            inference_controller=context.inference_controller,
        )
        app.aboutToQuit.connect(window.stop_all)
        window.resize(960, 600)
        window.show()

        if args.auto_start_camera:
            QTimer.singleShot(0, window.start_camera)
        if args.auto_start_inference:
            QTimer.singleShot(0, window.start_inference)

        try:
            return int(app.exec())
        finally:
            _stop_context(context)
    except Exception as exc:
        if args.debug:
            raise
        print(f"Live inference GUI launch failed: {exc}", file=sys.stderr)
        return 1


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-source",
        choices=(DEFAULT_CAMERA_SOURCE, OPENCV_V4L2_CAMERA_SOURCE),
        default=DEFAULT_CAMERA_SOURCE,
        help="Camera source implementation.",
    )
    parser.add_argument(
        "--synthetic-camera-config",
        type=Path,
        default=default_synthetic_camera_config_path(),
        help="Synthetic camera TOML config path.",
    )
    parser.add_argument(
        "--model-selection",
        type=Path,
        default=default_model_selection_path(),
        help="Live-local model selection TOML path.",
    )
    parser.add_argument(
        "--auto-start-camera",
        action="store_true",
        help="Start the synthetic camera after the window opens.",
    )
    parser.add_argument(
        "--auto-start-inference",
        action="store_true",
        help="Start inference after the window opens.",
    )
    parser.add_argument(
        "--frame-interval-ms",
        type=_positive_int,
        default=DEFAULT_FRAME_INTERVAL_MS,
        help="Synthetic camera frame interval in milliseconds.",
    )
    parser.add_argument(
        "--camera-device",
        default=DEFAULT_CAMERA_DEVICE,
        help="OpenCV/V4L2 camera device path or index.",
    )
    parser.add_argument(
        "--camera-width",
        type=_positive_int,
        default=DEFAULT_CAMERA_WIDTH_PX,
        help="OpenCV/V4L2 requested capture width in pixels.",
    )
    parser.add_argument(
        "--camera-height",
        type=_positive_int,
        default=DEFAULT_CAMERA_HEIGHT_PX,
        help="OpenCV/V4L2 requested capture height in pixels.",
    )
    parser.add_argument(
        "--camera-fps",
        type=_positive_int,
        default=DEFAULT_CAMERA_FPS,
        help="OpenCV/V4L2 requested capture frame rate.",
    )
    parser.add_argument(
        "--camera-pixel-format",
        default=DEFAULT_CAMERA_PIXEL_FORMAT,
        help="OpenCV/V4L2 requested FOURCC pixel format.",
    )
    parser.add_argument(
        "--camera-encoding",
        default=DEFAULT_CAMERA_ENCODING,
        help="Encoded handoff image format.",
    )
    parser.add_argument(
        "--camera-name",
        default=DEFAULT_CAMERA_NAME,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--log-v4l2-controls-at-startup",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print launch details and allow startup exceptions to show tracebacks.",
    )
    parser.add_argument(
        "--device",
        type=_device_policy,
        default=None,
        metavar="{auto,cuda,cpu}",
        help="Optional runtime device override for both selected models.",
    )
    parser.add_argument(
        "--inference-poll-interval-ms",
        type=_non_negative_int,
        default=DEFAULT_INFERENCE_POLL_INTERVAL_MS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--selection-path",
        dest="model_selection",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--source-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def _load_runtime_dependencies() -> _RuntimeDependencies:
    from cameras.synthetic_camera import (  # noqa: PLC0415
        SyntheticCameraConfig,
        SyntheticCameraPublisher,
        load_synthetic_camera_config,
    )
    from interfaces import LiveInferenceConfig  # noqa: PLC0415
    from live_inference.cameras import OpenCvV4L2CameraPublisher  # noqa: PLC0415
    from live_inference.engines import TorchTriStreamInferenceEngine  # noqa: PLC0415
    from live_inference.frame_handoff import (  # noqa: PLC0415
        AtomicFrameHandoffWriter,
        LatestFrameHandoffReader,
    )
    from live_inference.frame_selection import InferenceFrameSelector  # noqa: PLC0415
    from live_inference.gui.qt_worker_bridge import WorkerThreadController  # noqa: PLC0415
    from live_inference.inference_core import InferenceProcessingCore  # noqa: PLC0415
    from live_inference.model_registry import (  # noqa: PLC0415
        load_live_model_manifest,
        load_model_selection,
    )
    from live_inference.preprocessing import (  # noqa: PLC0415
        RoiFcnLocator,
        TriStreamLivePreprocessor,
    )
    from live_inference.workers import CameraWorker, InferenceWorker  # noqa: PLC0415

    return _RuntimeDependencies(
        synthetic_camera_config_cls=SyntheticCameraConfig,
        synthetic_camera_publisher_cls=SyntheticCameraPublisher,
        load_synthetic_camera_config=load_synthetic_camera_config,
        opencv_v4l2_camera_publisher_cls=OpenCvV4L2CameraPublisher,
        atomic_frame_handoff_writer_cls=AtomicFrameHandoffWriter,
        live_inference_config_cls=LiveInferenceConfig,
        torch_tri_stream_inference_engine_cls=TorchTriStreamInferenceEngine,
        latest_frame_handoff_reader_cls=LatestFrameHandoffReader,
        inference_frame_selector_cls=InferenceFrameSelector,
        inference_processing_core_cls=InferenceProcessingCore,
        load_live_model_manifest=load_live_model_manifest,
        load_model_selection=load_model_selection,
        roi_fcn_locator_cls=RoiFcnLocator,
        tri_stream_live_preprocessor_cls=TriStreamLivePreprocessor,
        camera_worker_cls=CameraWorker,
        inference_worker_cls=InferenceWorker,
        worker_thread_controller_cls=WorkerThreadController,
    )


def _synthetic_camera_config_from_path_or_default(
    deps: _RuntimeDependencies,
    config_path: Path | None,
) -> object:
    if config_path is not None and config_path.is_file():
        return deps.load_synthetic_camera_config(config_path)
    return deps.synthetic_camera_config_cls(
        source_dir=DEFAULT_SYNTHETIC_SOURCE_DIR,
        output_dir=DEFAULT_SYNTHETIC_OUTPUT_DIR,
        frame_interval_ms=DEFAULT_FRAME_INTERVAL_MS,
        loop=True,
    )


def _override_synthetic_camera_config(
    config: object,
    *,
    source_dir: Path | None,
    output_dir: Path | None,
    frame_interval_ms: int,
) -> object:
    updates: dict[str, object] = {"frame_interval_ms": int(frame_interval_ms)}
    if source_dir is not None:
        updates["source_dir"] = source_dir
    if output_dir is not None:
        updates["output_dir"] = output_dir
    return replace(config, **updates)


def _camera_source(value: str) -> str:
    text = str(value).strip()
    if text in {DEFAULT_CAMERA_SOURCE, OPENCV_V4L2_CAMERA_SOURCE}:
        return text
    raise ValueError(
        "camera_source must be one of "
        f"{DEFAULT_CAMERA_SOURCE!r}, {OPENCV_V4L2_CAMERA_SOURCE!r}; got {value!r}."
    )


def _resolve_camera_frame_dir(project_root: Path, output_dir: Path | None) -> Path:
    frame_dir = Path(DEFAULT_SYNTHETIC_OUTPUT_DIR if output_dir is None else output_dir)
    if frame_dir.is_absolute():
        return frame_dir
    return project_root / frame_dir


def _frame_extension(encoding: str) -> str:
    text = str(encoding).strip().lower().lstrip(".")
    if text == "jpeg":
        return "jpg"
    if not text:
        raise ValueError("camera_encoding must not be empty.")
    return text


def _resolve_path(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _resolve_synthetic_camera_config_path(path: Path | None) -> Path | None:
    if path is None:
        default_path = default_synthetic_camera_config_path()
        return default_path if default_path.is_file() else None

    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Synthetic camera config does not exist: {resolved}")
    return resolved


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return number


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return number


def _device_policy(value: str) -> str:
    try:
        return normalize_torch_device_policy(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _print_launch_debug(context: LiveInferenceGuiContext) -> None:
    synthetic_config = context.synthetic_camera_config
    print("Live inference GUI launch context:", file=sys.stderr)
    print(f"  camera_source = {context.camera_source}", file=sys.stderr)
    print(f"  model_selection = {context.model_selection_path}", file=sys.stderr)
    print(
        f"  distance_orientation_device = {context.distance_orientation_device}",
        file=sys.stderr,
    )
    print(f"  roi_fcn_device = {context.roi_fcn_device}", file=sys.stderr)
    print(
        f"  synthetic_camera_config = {context.synthetic_camera_config_path}",
        file=sys.stderr,
    )
    print(
        f"  synthetic_camera_base_dir = {context.synthetic_camera_base_dir}",
        file=sys.stderr,
    )
    print(
        f"  camera_frame_dir = {getattr(context.live_inference_config, 'frame_dir', 'n/a')}",
        file=sys.stderr,
    )
    print(
        f"  synthetic_camera_source_dir = {getattr(synthetic_config, 'source_dir', 'n/a')}",
        file=sys.stderr,
    )
    print(
        f"  synthetic_camera_output_dir = {getattr(synthetic_config, 'output_dir', 'n/a')}",
        file=sys.stderr,
    )


def _stop_context(context: LiveInferenceGuiContext, timeout_ms: int = 2000) -> None:
    for controller in (context.camera_controller, context.inference_controller):
        request_stop = getattr(controller, "request_stop", None)
        if callable(request_stop):
            request_stop()
    for controller in (context.camera_controller, context.inference_controller):
        wait = getattr(controller, "wait", None)
        if callable(wait):
            wait(timeout_ms)


def _live_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_FRAME_INTERVAL_MS",
    "DEFAULT_INFERENCE_POLL_INTERVAL_MS",
    "DEFAULT_CAMERA_DEVICE",
    "DEFAULT_CAMERA_ENCODING",
    "DEFAULT_CAMERA_FPS",
    "DEFAULT_CAMERA_HEIGHT_PX",
    "DEFAULT_CAMERA_NAME",
    "DEFAULT_CAMERA_PIXEL_FORMAT",
    "DEFAULT_CAMERA_SOURCE",
    "DEFAULT_CAMERA_WIDTH_PX",
    "DEFAULT_SYNTHETIC_OUTPUT_DIR",
    "DEFAULT_SYNTHETIC_SOURCE_DIR",
    "LiveInferenceGuiContext",
    "OPENCV_V4L2_CAMERA_SOURCE",
    "build_live_inference_gui_context",
    "default_model_selection_path",
    "default_synthetic_camera_config_path",
    "main",
]
