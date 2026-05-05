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


@dataclass(frozen=True)
class LiveInferenceGuiContext:
    """Composed worker controllers for the GUI shell."""

    camera_controller: object
    inference_controller: object
    model_selection_path: Path
    synthetic_camera_config_path: Path | None
    synthetic_camera_base_dir: Path
    synthetic_camera_config: object
    distance_orientation_device: str
    roi_fcn_device: str


@dataclass(frozen=True)
class _RuntimeDependencies:
    synthetic_camera_config_cls: type
    synthetic_camera_publisher_cls: type
    load_synthetic_camera_config: Callable[[Path], object]
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
    model_selection_path: Path | None = None,
    synthetic_camera_config_path: Path | None = None,
    selection_path: Path | None = None,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    device: str | None = None,
    frame_interval_ms: int = DEFAULT_FRAME_INTERVAL_MS,
    inference_poll_interval_ms: int = DEFAULT_INFERENCE_POLL_INTERVAL_MS,
    dependency_loader: Callable[[], _RuntimeDependencies] | None = None,
) -> LiveInferenceGuiContext:
    """Build the current synthetic-camera to tri-stream inference pipeline."""
    deps = (dependency_loader or _load_runtime_dependencies)()

    project_root = _live_project_root()
    resolved_selection_path = _resolve_path(
        model_selection_path or selection_path or default_model_selection_path()
    )
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
    synthetic_base_dir = project_root
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
            model_selection_path=args.model_selection,
            synthetic_camera_config_path=args.synthetic_camera_config,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
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
    from live_inference.engines import TorchTriStreamInferenceEngine  # noqa: PLC0415
    from live_inference.frame_handoff import LatestFrameHandoffReader  # noqa: PLC0415
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
    "DEFAULT_SYNTHETIC_OUTPUT_DIR",
    "DEFAULT_SYNTHETIC_SOURCE_DIR",
    "LiveInferenceGuiContext",
    "build_live_inference_gui_context",
    "default_model_selection_path",
    "default_synthetic_camera_config_path",
    "main",
]
