"""Tests for the synthetic camera frame publisher."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cameras.synthetic_camera.synthetic_camera as synthetic_camera  # noqa: E402
from cameras.synthetic_camera import (  # noqa: E402
    SOURCE_KIND,
    SyntheticCameraConfig,
    SyntheticCameraPublisher,
    SyntheticCameraSortOrder,
    discover_source_images,
    load_synthetic_camera_config,
)
from live_inference.frame_handoff import compute_frame_hash  # noqa: E402


class SyntheticCameraConfigTests(unittest.TestCase):
    def _write_toml(self, tmp_dir: str, body: str) -> Path:
        path = Path(tmp_dir) / "synthetic_camera.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_loads_toml_config(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = self._write_toml(
                tmp_dir,
                """
[synthetic_camera]
source_dir = "synthetic_camera_source"
output_dir = "live_frames"
allowed_extensions = ["png", "jpg"]
sort_order = "name_descending"
frame_interval_ms = 250
max_images = 12
loop = false
start_index = 3
rescan_on_loop = true
latest_frame_filename = "latest_frame.png"
temp_frame_filename = "latest_frame.tmp.png"
metadata_filename = "latest_frame.json"
temp_metadata_filename = "latest_frame.tmp.json"
""",
            )

            config = load_synthetic_camera_config(path)

            self.assertEqual(config.source_dir, Path("synthetic_camera_source"))
            self.assertEqual(config.output_dir, Path("live_frames"))
            self.assertEqual(config.allowed_extensions, ("png", "jpg"))
            self.assertEqual(config.sort_order, SyntheticCameraSortOrder.NAME_DESCENDING)
            self.assertEqual(config.frame_interval_ms, 250)
            self.assertEqual(config.max_images, 12)
            self.assertFalse(config.loop)
            self.assertEqual(config.start_index, 3)
            self.assertTrue(config.rescan_on_loop)

    def test_missing_source_dir_raises_clear_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = self._write_toml(tmp_dir, "[synthetic_camera]\noutput_dir = \"live_frames\"\n")

            with self.assertRaisesRegex(ValueError, "source_dir is required"):
                load_synthetic_camera_config(path)

    def test_defaults_are_applied(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = self._write_toml(
                tmp_dir,
                "[synthetic_camera]\nsource_dir = \"source\"\n",
            )

            config = load_synthetic_camera_config(path)

            self.assertEqual(config.allowed_extensions, ("png", "jpg", "jpeg"))
            self.assertEqual(config.sort_order, SyntheticCameraSortOrder.MODIFIED_TIME_ASCENDING)
            self.assertEqual(config.frame_interval_ms, 100)
            self.assertEqual(config.max_images, 2048)
            self.assertTrue(config.loop)
            self.assertEqual(config.output_dir, Path("live_frames"))

    def test_rejects_absolute_source_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_dir must be a relative path"):
            SyntheticCameraConfig(source_dir=Path("/tmp/source"))

    def test_rejects_absolute_output_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, "output_dir must be a relative path"):
            SyntheticCameraConfig(source_dir=Path("source"), output_dir=Path("/tmp/live_frames"))

    def test_rejects_parent_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain"):
            SyntheticCameraConfig(source_dir=Path("../source"))
        with self.assertRaisesRegex(ValueError, "must not contain"):
            SyntheticCameraConfig(source_dir=Path("source"), output_dir=Path("out/../live"))

    def test_rejects_invalid_numeric_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame_interval_ms must be > 0"):
            SyntheticCameraConfig(source_dir=Path("source"), frame_interval_ms=0)
        with self.assertRaisesRegex(ValueError, "max_images must be > 0"):
            SyntheticCameraConfig(source_dir=Path("source"), max_images=0)
        with self.assertRaisesRegex(ValueError, "start_index must be >= 0"):
            SyntheticCameraConfig(source_dir=Path("source"), start_index=-1)

    def test_normalizes_extensions(self) -> None:
        config = SyntheticCameraConfig(
            source_dir=Path("source"),
            allowed_extensions=(".PNG", "Jpg", "jpeg"),
        )

        self.assertEqual(config.allowed_extensions, ("png", "jpg", "jpeg"))

    def test_rejects_empty_extensions_and_invalid_sort_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "allowed_extensions must not be empty"):
            SyntheticCameraConfig(source_dir=Path("source"), allowed_extensions=())
        with self.assertRaisesRegex(ValueError, "Invalid synthetic camera sort_order"):
            SyntheticCameraConfig(source_dir=Path("source"), sort_order="random")


class SyntheticCameraIndexTests(unittest.TestCase):
    def _make_source_tree(self, tmp_dir: str) -> Path:
        source = Path(tmp_dir) / "source"
        nested = source / "nested"
        nested.mkdir(parents=True)
        (source / "b.jpg").write_bytes(b"b")
        (source / "a.PNG").write_bytes(b"a")
        (source / "ignore.txt").write_bytes(b"text")
        (nested / "c.JPEG").write_bytes(b"c")
        return source

    def _set_mtime(self, path: Path, mtime_ns: int) -> None:
        os.utime(path, ns=(mtime_ns, mtime_ns))

    def test_finds_images_case_insensitively_and_ignores_unsupported(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source_tree(tmp_dir)
            config = SyntheticCameraConfig(
                source_dir=Path("source"),
                sort_order=SyntheticCameraSortOrder.NAME_ASCENDING,
            )

            images = discover_source_images(config, base_dir=Path(tmp_dir))
            names = [path.relative_to(Path(tmp_dir) / "source").as_posix() for path in images]

            self.assertEqual(names, ["a.PNG", "b.jpg", "nested/c.JPEG"])

    def test_name_descending_sort(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source_tree(tmp_dir)
            config = SyntheticCameraConfig(
                source_dir=Path("source"),
                sort_order=SyntheticCameraSortOrder.NAME_DESCENDING,
            )

            images = discover_source_images(config, base_dir=Path(tmp_dir))
            names = [path.relative_to(Path(tmp_dir) / "source").as_posix() for path in images]

            self.assertEqual(names, ["nested/c.JPEG", "b.jpg", "a.PNG"])

    def test_modified_time_sort_orders_and_tie_break(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            source = self._make_source_tree(tmp_dir)
            self._set_mtime(source / "a.PNG", 100)
            self._set_mtime(source / "b.jpg", 100)
            self._set_mtime(source / "nested" / "c.JPEG", 300)

            ascending = discover_source_images(
                SyntheticCameraConfig(
                    source_dir=Path("source"),
                    sort_order=SyntheticCameraSortOrder.MODIFIED_TIME_ASCENDING,
                ),
                base_dir=Path(tmp_dir),
            )
            descending = discover_source_images(
                SyntheticCameraConfig(
                    source_dir=Path("source"),
                    sort_order=SyntheticCameraSortOrder.MODIFIED_TIME_DESCENDING,
                ),
                base_dir=Path(tmp_dir),
            )

            self.assertEqual(
                [path.relative_to(source).as_posix() for path in ascending],
                ["a.PNG", "b.jpg", "nested/c.JPEG"],
            )
            self.assertEqual(
                [path.relative_to(source).as_posix() for path in descending],
                ["nested/c.JPEG", "a.PNG", "b.jpg"],
            )

    def test_max_images_limits_after_sorting(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source_tree(tmp_dir)
            config = SyntheticCameraConfig(
                source_dir=Path("source"),
                sort_order=SyntheticCameraSortOrder.NAME_ASCENDING,
                max_images=2,
            )

            images = discover_source_images(config, base_dir=Path(tmp_dir))

            self.assertEqual(len(images), 2)
            self.assertEqual(
                [path.relative_to(Path(tmp_dir) / "source").as_posix() for path in images],
                ["a.PNG", "b.jpg"],
            )

    def test_empty_source_directory_raises_clear_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            (Path(tmp_dir) / "source").mkdir()
            config = SyntheticCameraConfig(source_dir=Path("source"))

            with self.assertRaisesRegex(ValueError, "No supported source images"):
                discover_source_images(config, base_dir=Path(tmp_dir))


class SyntheticCameraPublisherTests(unittest.TestCase):
    def _make_source(self, tmp_dir: str) -> Path:
        source = Path(tmp_dir) / "source"
        source.mkdir()
        (source / "a.png").write_bytes(b"first")
        (source / "b.jpg").write_bytes(b"second")
        os.utime(source / "a.png", ns=(100, 100))
        os.utime(source / "b.jpg", ns=(200, 200))
        return source

    def _config(self) -> SyntheticCameraConfig:
        return SyntheticCameraConfig(
            source_dir=Path("source"),
            output_dir=Path("live_frames"),
            sort_order=SyntheticCameraSortOrder.NAME_ASCENDING,
            frame_interval_ms=25,
            loop=True,
        )

    def test_publish_next_writes_latest_image_and_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source(tmp_dir)
            publisher = SyntheticCameraPublisher(
                self._config(),
                base_dir=Path(tmp_dir),
                now_utc_fn=lambda: "2026-05-01T10:00:00Z",
            )

            frame = publisher.publish_next()
            latest_path = Path(tmp_dir) / "live_frames" / "latest_frame.png"
            metadata_path = Path(tmp_dir) / "live_frames" / "latest_frame.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(frame.image_path, latest_path)
            self.assertEqual(latest_path.read_bytes(), b"first")
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(metadata["source_kind"], SOURCE_KIND)
            self.assertEqual(metadata["frame_id"], "synthetic-000000-000000")
            self.assertEqual(metadata["sequence_index"], 0)
            self.assertEqual(metadata["loop_index"], 0)
            self.assertEqual(metadata["source_relative_path"], "a.png")
            self.assertEqual(metadata["byte_size"], len(b"first"))
            self.assertEqual(metadata["published_at_utc"], "2026-05-01T10:00:00Z")
            self.assertEqual(metadata["synthetic_captured_at_utc"], "2026-05-01T10:00:00Z")
            self.assertEqual(metadata["frame_hash"], compute_frame_hash(latest_path.read_bytes()).value)
            self.assertFalse((Path(tmp_dir) / "live_frames" / "latest_frame.tmp.json").exists())

    def test_second_publish_replaces_latest_image_and_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source(tmp_dir)
            publisher = SyntheticCameraPublisher(
                self._config(),
                base_dir=Path(tmp_dir),
                now_utc_fn=lambda: "2026-05-01T10:00:00Z",
            )

            publisher.publish_next()
            first_metadata = json.loads(
                (Path(tmp_dir) / "live_frames" / "latest_frame.json").read_text(encoding="utf-8")
            )
            publisher.publish_next()
            latest_path = Path(tmp_dir) / "live_frames" / "latest_frame.png"
            second_metadata = json.loads(
                (Path(tmp_dir) / "live_frames" / "latest_frame.json").read_text(encoding="utf-8")
            )

            self.assertEqual(latest_path.read_bytes(), b"second")
            self.assertEqual(first_metadata["frame_id"], "synthetic-000000-000000")
            self.assertEqual(second_metadata["frame_id"], "synthetic-000000-000001")
            self.assertEqual(second_metadata["source_relative_path"], "b.jpg")

    def test_loop_true_wraps_to_first_source_image(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source(tmp_dir)
            publisher = SyntheticCameraPublisher(
                self._config(),
                base_dir=Path(tmp_dir),
                now_utc_fn=lambda: "2026-05-01T10:00:00Z",
            )

            publisher.publish_next()
            publisher.publish_next()
            publisher.publish_next()
            metadata = json.loads(
                (Path(tmp_dir) / "live_frames" / "latest_frame.json").read_text(encoding="utf-8")
            )

            self.assertEqual((Path(tmp_dir) / "live_frames" / "latest_frame.png").read_bytes(), b"first")
            self.assertEqual(metadata["frame_id"], "synthetic-000001-000000")
            self.assertEqual(metadata["loop_index"], 1)
            self.assertEqual(metadata["sequence_index"], 0)

    def test_loop_false_raises_stop_iteration_after_final_image(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source(tmp_dir)
            config = SyntheticCameraConfig(
                source_dir=Path("source"),
                output_dir=Path("live_frames"),
                sort_order=SyntheticCameraSortOrder.NAME_ASCENDING,
                loop=False,
            )
            publisher = SyntheticCameraPublisher(config, base_dir=Path(tmp_dir))

            publisher.publish_next()
            publisher.publish_next()

            with self.assertRaises(StopIteration):
                publisher.publish_next()

    def test_run_forever_uses_injected_sleep_interval(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source(tmp_dir)
            sleep_calls: list[float] = []
            stop_checks = 0

            def stop_requested() -> bool:
                nonlocal stop_checks
                stop_checks += 1
                return stop_checks > 3

            publisher = SyntheticCameraPublisher(
                self._config(),
                base_dir=Path(tmp_dir),
                sleep_fn=sleep_calls.append,
                now_utc_fn=lambda: "2026-05-01T10:00:00Z",
            )

            publisher.run_forever(stop_requested=stop_requested)

            self.assertEqual(sleep_calls, [0.025])
            self.assertEqual((Path(tmp_dir) / "live_frames" / "latest_frame.png").read_bytes(), b"second")

    def test_output_directory_does_not_accumulate_unique_frame_images(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source(tmp_dir)
            publisher = SyntheticCameraPublisher(
                self._config(),
                base_dir=Path(tmp_dir),
                now_utc_fn=lambda: "2026-05-01T10:00:00Z",
            )

            publisher.publish_next()
            publisher.publish_next()
            output_files = sorted(path.name for path in (Path(tmp_dir) / "live_frames").iterdir())

            self.assertEqual(output_files, ["latest_frame.json", "latest_frame.png"])

    def test_metadata_replace_failure_preserves_previous_metadata_and_surfaces_error(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            self._make_source(tmp_dir)
            publisher = SyntheticCameraPublisher(
                self._config(),
                base_dir=Path(tmp_dir),
                now_utc_fn=lambda: "2026-05-01T10:00:00Z",
            )
            publisher.publish_next()
            metadata_path = Path(tmp_dir) / "live_frames" / "latest_frame.json"
            temp_metadata_path = Path(tmp_dir) / "live_frames" / "latest_frame.tmp.json"
            previous_metadata = metadata_path.read_text(encoding="utf-8")
            original_replace = synthetic_camera.os.replace

            def fail_metadata_replace(src: Path, dst: Path) -> None:
                if Path(dst).name == "latest_frame.json":
                    raise OSError("simulated metadata replace failure")
                original_replace(src, dst)

            with patch.object(
                synthetic_camera.os,
                "replace",
                side_effect=fail_metadata_replace,
            ):
                with self.assertRaisesRegex(OSError, "simulated metadata replace failure"):
                    publisher.publish_next()

            self.assertEqual(metadata_path.read_text(encoding="utf-8"), previous_metadata)
            self.assertTrue(temp_metadata_path.is_file())

    def test_import_hygiene(self) -> None:
        module_path = SRC_ROOT / "cameras" / "synthetic_camera" / "synthetic_camera.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        forbidden_roots = {
            "PySide6",
            "cv2",
            "numpy",
            "torch",
        }
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])

        self.assertTrue(forbidden_roots.isdisjoint(imported_roots))


if __name__ == "__main__":
    unittest.main()
