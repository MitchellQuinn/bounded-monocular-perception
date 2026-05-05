"""Minimal runnable PySide6 application for the live inference GUI shell."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication

from live_inference.gui.main_window import LiveInferenceMainWindow
from live_inference.gui.qt_worker_bridge import WorkerThreadController


DEFAULT_SOURCE_DIR = Path(
    "05_inference-v0.4-ts/input/def90_synth_v023-validation-shuffled/images"
)
DEFAULT_OUTPUT_DIR = Path("06_live-inference_v0.1/live_frames")


@dataclass(frozen=True)
class LiveInferenceGuiContext:
    """Composed worker controllers for the GUI shell."""

    camera_controller: WorkerThreadController
    inference_controller: WorkerThreadController


def build_live_inference_gui_context(
    *,
    selection_path: Path | None = None,
    source_dir: Path | None = None,
    output_dir: Path | None = None,
    device: str | None = None,
    frame_interval_ms: int = 100,
    inference_poll_interval_ms: int = 10,
) -> LiveInferenceGuiContext:
    """Build the current synthetic-camera to tri-stream inference pipeline."""
    from cameras.synthetic_camera import (  # noqa: PLC0415
        SyntheticCameraConfig,
        SyntheticCameraPublisher,
    )
    from interfaces import LiveInferenceConfig  # noqa: PLC0415
    from live_inference.engines import TorchTriStreamInferenceEngine  # noqa: PLC0415
    from live_inference.frame_handoff import LatestFrameHandoffReader  # noqa: PLC0415
    from live_inference.frame_selection import InferenceFrameSelector  # noqa: PLC0415
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

    project_root = _live_project_root()
    repo_root = project_root.parent
    resolved_selection_path = selection_path or (
        project_root / "models/selections/current.toml"
    )
    synthetic_config = SyntheticCameraConfig(
        source_dir=source_dir or DEFAULT_SOURCE_DIR,
        output_dir=output_dir or DEFAULT_OUTPUT_DIR,
        frame_interval_ms=frame_interval_ms,
        loop=True,
    )
    publisher = SyntheticCameraPublisher(synthetic_config, base_dir=repo_root)

    live_config = LiveInferenceConfig(
        frame_dir=publisher.output_dir,
        latest_frame_filename=synthetic_config.latest_frame_filename,
        temp_frame_filename=synthetic_config.temp_frame_filename,
        inference_poll_interval_ms=inference_poll_interval_ms,
    )
    selection = load_model_selection(resolved_selection_path)
    distance_orientation_device = device or selection.distance_orientation_device
    roi_fcn_device = device or selection.roi_fcn_device
    manifest = load_live_model_manifest(
        selection.distance_orientation_root,
        roi_locator_root=selection.roi_fcn_root,
    )

    reader = LatestFrameHandoffReader(live_config)
    selector = InferenceFrameSelector(
        reader,
        duplicate_hash_skip_enabled=live_config.duplicate_hash_skip_enabled,
    )
    roi_locator = RoiFcnLocator(selection.roi_fcn_root, device=roi_fcn_device)
    preprocessor = TriStreamLivePreprocessor(
        model_manifest=manifest,
        roi_locator=roi_locator,
    )
    engine = TorchTriStreamInferenceEngine(
        model_root=selection.distance_orientation_root,
        model_manifest=manifest,
        device=distance_orientation_device,
    )
    core = InferenceProcessingCore(selector, preprocessor, engine)

    camera_worker = CameraWorker(publisher)
    inference_worker = InferenceWorker(
        core,
        poll_interval_ms=live_config.inference_poll_interval_ms,
    )
    return LiveInferenceGuiContext(
        camera_controller=WorkerThreadController(camera_worker),
        inference_controller=WorkerThreadController(inference_worker),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])

    context = build_live_inference_gui_context(
        selection_path=args.selection_path,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        device=args.device,
        frame_interval_ms=args.frame_interval_ms,
        inference_poll_interval_ms=args.inference_poll_interval_ms,
    )
    window = LiveInferenceMainWindow(
        camera_controller=context.camera_controller,
        inference_controller=context.inference_controller,
    )
    window.resize(960, 600)
    window.show()
    return int(app.exec())


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-path", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--frame-interval-ms", type=int, default=100)
    parser.add_argument("--inference-poll-interval-ms", type=int, default=10)
    return parser


def _live_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LiveInferenceGuiContext",
    "build_live_inference_gui_context",
    "main",
]
