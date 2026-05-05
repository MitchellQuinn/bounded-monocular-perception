"""Tests for the live inference GUI application launcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import unittest
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from live_inference.gui import app as gui_app  # noqa: E402


class GuiAppImportTests(unittest.TestCase):
    def test_app_module_imports_without_starting_gui(self) -> None:
        self.assertTrue(callable(gui_app.main))
        self.assertTrue(callable(gui_app.build_live_inference_gui_context))


class GuiAppCliParserTests(unittest.TestCase):
    def test_cli_parser_parses_defaults(self) -> None:
        args = gui_app._argument_parser().parse_args([])

        self.assertEqual(args.synthetic_camera_config, gui_app.default_synthetic_camera_config_path())
        self.assertEqual(args.model_selection, gui_app.default_model_selection_path())
        self.assertFalse(args.auto_start_camera)
        self.assertFalse(args.auto_start_inference)
        self.assertEqual(args.frame_interval_ms, gui_app.DEFAULT_FRAME_INTERVAL_MS)
        self.assertFalse(args.debug)

    def test_cli_parser_parses_explicit_paths_and_smoke_options(self) -> None:
        args = gui_app._argument_parser().parse_args(
            [
                "--synthetic-camera-config",
                "custom_camera.toml",
                "--model-selection",
                "custom_selection.toml",
                "--auto-start-camera",
                "--auto-start-inference",
                "--frame-interval-ms",
                "333",
                "--debug",
            ]
        )

        self.assertEqual(args.synthetic_camera_config, Path("custom_camera.toml"))
        self.assertEqual(args.model_selection, Path("custom_selection.toml"))
        self.assertTrue(args.auto_start_camera)
        self.assertTrue(args.auto_start_inference)
        self.assertEqual(args.frame_interval_ms, 333)
        self.assertTrue(args.debug)


class GuiAppCompositionTests(unittest.TestCase):
    def test_composition_can_be_built_with_lightweight_dependencies(self) -> None:
        records: dict[str, Any] = {}

        context = gui_app.build_live_inference_gui_context(
            model_selection_path=PROJECT_ROOT / "models/selections/current.toml",
            synthetic_camera_config_path=PROJECT_ROOT / "config/synthetic_camera.toml.example",
            device="cpu",
            frame_interval_ms=123,
            inference_poll_interval_ms=17,
            dependency_loader=lambda: _fake_dependencies(records),
        )

        self.assertEqual(context.model_selection_path, PROJECT_ROOT / "models/selections/current.toml")
        self.assertEqual(context.synthetic_camera_base_dir, PROJECT_ROOT)
        self.assertEqual(context.synthetic_camera_config.source_dir, Path("configured_source"))
        self.assertEqual(context.synthetic_camera_config.output_dir, Path("configured_output"))
        self.assertEqual(context.synthetic_camera_config.frame_interval_ms, 123)
        self.assertIsInstance(context.camera_controller.worker, _FakeCameraWorker)
        self.assertIsInstance(context.inference_controller.worker, _FakeInferenceWorker)
        self.assertEqual(records["publisher_base_dir"], PROJECT_ROOT)
        self.assertEqual(records["live_frame_dir"], PROJECT_ROOT / "configured_output")
        self.assertEqual(records["selector_duplicate_skip"], True)
        self.assertEqual(records["roi_device"], "cpu")
        self.assertEqual(records["engine_device"], "cpu")
        self.assertEqual(records["inference_poll_interval_ms"], 17)


@dataclass(frozen=True)
class _FakeSyntheticCameraConfig:
    source_dir: Path
    output_dir: Path = Path("live_frames")
    allowed_extensions: tuple[str, ...] = ("png", "jpg", "jpeg")
    sort_order: str = "modified_time_ascending"
    frame_interval_ms: int = 250
    max_images: int = 2048
    loop: bool = True
    start_index: int = 0
    rescan_on_loop: bool = False
    latest_frame_filename: str = "latest_frame.png"
    temp_frame_filename: str = "latest_frame.tmp.png"


@dataclass(frozen=True)
class _FakeLiveInferenceConfig:
    frame_dir: Path
    latest_frame_filename: str
    temp_frame_filename: str
    inference_poll_interval_ms: int
    duplicate_hash_skip_enabled: bool = True


@dataclass(frozen=True)
class _FakeSelection:
    distance_orientation_root: Path = Path("distance-model")
    roi_fcn_root: Path = Path("roi-model")
    distance_orientation_device: str = "cuda"
    roi_fcn_device: str = "cuda"


class _FakeSyntheticCameraPublisher:
    def __init__(self, config: _FakeSyntheticCameraConfig, *, base_dir: Path) -> None:
        self.config = config
        self.base_dir = base_dir
        self.output_dir = base_dir / config.output_dir


class _FakeReader:
    def __init__(self, live_config: _FakeLiveInferenceConfig) -> None:
        self.live_config = live_config


class _FakeSelector:
    def __init__(self, reader: _FakeReader, *, duplicate_hash_skip_enabled: bool) -> None:
        self.reader = reader
        self.duplicate_hash_skip_enabled = duplicate_hash_skip_enabled


class _FakeRoiLocator:
    def __init__(self, roi_root: Path, *, device: str) -> None:
        self.roi_root = roi_root
        self.device = device


class _FakePreprocessor:
    def __init__(self, *, model_manifest: object, roi_locator: _FakeRoiLocator) -> None:
        self.model_manifest = model_manifest
        self.roi_locator = roi_locator


class _FakeEngine:
    def __init__(self, *, model_root: Path, model_manifest: object, device: str) -> None:
        self.model_root = model_root
        self.model_manifest = model_manifest
        self.device = device


class _FakeCore:
    def __init__(
        self,
        selector: _FakeSelector,
        preprocessor: _FakePreprocessor,
        engine: _FakeEngine,
    ) -> None:
        self.selector = selector
        self.preprocessor = preprocessor
        self.engine = engine


class _FakeCameraWorker:
    def __init__(self, publisher: _FakeSyntheticCameraPublisher) -> None:
        self.publisher = publisher


class _FakeInferenceWorker:
    def __init__(self, core: _FakeCore, *, poll_interval_ms: int) -> None:
        self.core = core
        self.poll_interval_ms = poll_interval_ms


class _FakeWorkerThreadController:
    def __init__(self, worker: object) -> None:
        self.worker = worker
        self.stop_calls = 0
        self.wait_calls = 0

    def request_stop(self) -> None:
        self.stop_calls += 1

    def wait(self, timeout_ms: int | None = None) -> bool:
        self.wait_calls += 1
        return True


def _fake_dependencies(records: dict[str, Any]) -> gui_app._RuntimeDependencies:
    def load_synthetic_camera_config(path: Path) -> _FakeSyntheticCameraConfig:
        records["camera_config_path"] = path
        return _FakeSyntheticCameraConfig(
            source_dir=Path("configured_source"),
            output_dir=Path("configured_output"),
        )

    def make_live_config(**kwargs: Any) -> _FakeLiveInferenceConfig:
        config = _FakeLiveInferenceConfig(**kwargs)
        records["live_frame_dir"] = config.frame_dir
        return config

    def load_model_selection(path: Path) -> _FakeSelection:
        records["model_selection_path"] = path
        return _FakeSelection()

    def load_manifest(model_root: Path, *, roi_locator_root: Path | None = None) -> object:
        records["manifest_model_root"] = model_root
        records["manifest_roi_root"] = roi_locator_root
        return object()

    class RecordingPublisher(_FakeSyntheticCameraPublisher):
        def __init__(self, config: _FakeSyntheticCameraConfig, *, base_dir: Path) -> None:
            super().__init__(config, base_dir=base_dir)
            records["publisher_base_dir"] = base_dir

    class RecordingSelector(_FakeSelector):
        def __init__(self, reader: _FakeReader, *, duplicate_hash_skip_enabled: bool) -> None:
            super().__init__(
                reader,
                duplicate_hash_skip_enabled=duplicate_hash_skip_enabled,
            )
            records["selector_duplicate_skip"] = duplicate_hash_skip_enabled

    class RecordingRoiLocator(_FakeRoiLocator):
        def __init__(self, roi_root: Path, *, device: str) -> None:
            super().__init__(roi_root, device=device)
            records["roi_device"] = device

    class RecordingEngine(_FakeEngine):
        def __init__(self, *, model_root: Path, model_manifest: object, device: str) -> None:
            super().__init__(
                model_root=model_root,
                model_manifest=model_manifest,
                device=device,
            )
            records["engine_device"] = device

    class RecordingInferenceWorker(_FakeInferenceWorker):
        def __init__(self, core: _FakeCore, *, poll_interval_ms: int) -> None:
            super().__init__(core, poll_interval_ms=poll_interval_ms)
            records["inference_poll_interval_ms"] = poll_interval_ms

    return gui_app._RuntimeDependencies(
        synthetic_camera_config_cls=_FakeSyntheticCameraConfig,
        synthetic_camera_publisher_cls=RecordingPublisher,
        load_synthetic_camera_config=load_synthetic_camera_config,
        live_inference_config_cls=make_live_config,
        torch_tri_stream_inference_engine_cls=RecordingEngine,
        latest_frame_handoff_reader_cls=_FakeReader,
        inference_frame_selector_cls=RecordingSelector,
        inference_processing_core_cls=_FakeCore,
        load_live_model_manifest=load_manifest,
        load_model_selection=load_model_selection,
        roi_fcn_locator_cls=RecordingRoiLocator,
        tri_stream_live_preprocessor_cls=_FakePreprocessor,
        camera_worker_cls=_FakeCameraWorker,
        inference_worker_cls=RecordingInferenceWorker,
        worker_thread_controller_cls=_FakeWorkerThreadController,
    )


if __name__ == "__main__":
    unittest.main()
