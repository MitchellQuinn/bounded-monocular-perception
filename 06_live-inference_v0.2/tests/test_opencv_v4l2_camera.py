"""Tests for the OpenCV/V4L2 camera publisher."""

from __future__ import annotations

import ast
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from interfaces import FrameMetadata, FrameReference  # noqa: E402
from live_inference.cameras import (  # noqa: E402
    CameraOpenError,
    FrameCaptureError,
    FrameEncodingError,
    OpenCvV4L2CameraPublisher,
)


NOW = "2026-05-05T10:00:00Z"
WRITTEN_AT = "2026-05-05T10:00:01Z"


class OpenCvV4L2CameraPublisherTests(unittest.TestCase):
    def test_constructs_without_opening_when_auto_open_false(self) -> None:
        cv2 = FakeCv2()

        OpenCvV4L2CameraPublisher(
            handoff_writer=RecordingHandoffWriter(),
            auto_open=False,
            cv2_module=cv2,
        )

        self.assertEqual(cv2.video_capture_calls, [])

    def test_open_uses_v4l2_backend(self) -> None:
        cv2 = FakeCv2()
        publisher = _publisher(cv2=cv2)

        publisher.open()

        self.assertEqual(cv2.video_capture_calls, [("/dev/video-test", cv2.CAP_V4L2)])

    def test_open_sets_requested_width_height_and_fps(self) -> None:
        cv2 = FakeCv2()
        publisher = _publisher(cv2=cv2, width_px=960, height_px=600, fps=80)

        publisher.open()

        self.assertIn((cv2.CAP_PROP_FRAME_WIDTH, 960), cv2.capture.set_calls)
        self.assertIn((cv2.CAP_PROP_FRAME_HEIGHT, 600), cv2.capture.set_calls)
        self.assertIn((cv2.CAP_PROP_FPS, 80), cv2.capture.set_calls)

    def test_open_attempts_requested_yuyv_fourcc(self) -> None:
        cv2 = FakeCv2()
        publisher = _publisher(cv2=cv2, pixel_format="YUYV")

        publisher.open()

        self.assertEqual(cv2.fourcc_calls, [("Y", "U", "Y", "V")])
        self.assertIn((cv2.CAP_PROP_FOURCC, cv2.yuyv_fourcc), cv2.capture.set_calls)

    def test_publish_next_reads_encodes_publishes_and_returns_writer_reference(self) -> None:
        writer = RecordingHandoffWriter()
        cv2 = FakeCv2()
        expected_reference = FrameReference(
            image_path=Path("live_frames/latest_frame.png"),
            completed_at_utc=WRITTEN_AT,
        )
        writer.next_reference = expected_reference
        publisher = _publisher(cv2=cv2, writer=writer)

        frame_reference = publisher.publish_next()

        self.assertIs(frame_reference, expected_reference)
        self.assertEqual(cv2.capture.read_calls, 1)
        self.assertEqual(cv2.imencode_calls, [(".png", cv2.capture.frame)])
        self.assertEqual(len(writer.publish_calls), 1)
        image_bytes, metadata = writer.publish_calls[0]
        self.assertEqual(image_bytes, b"fake-png-bytes")
        self.assertIsInstance(metadata, FrameMetadata)

    def test_publish_metadata_includes_camera_name_and_requested_properties(self) -> None:
        writer = RecordingHandoffWriter()
        cv2 = FakeCv2()
        publisher = _publisher(
            cv2=cv2,
            writer=writer,
            width_px=960,
            height_px=600,
            fps=80,
            pixel_format="YUYV",
            camera_name="Arducam B0495 AR0234",
        )

        publisher.publish_next()

        metadata = writer.publish_calls[0][1]
        self.assertEqual(metadata.source_name, "Arducam B0495 AR0234")
        self.assertEqual(metadata.captured_at_utc, NOW)
        self.assertEqual(metadata.width_px, 960)
        self.assertEqual(metadata.height_px, 600)
        self.assertEqual(metadata.pixel_format, "YUYV")
        self.assertEqual(metadata.encoding, "png")
        self.assertEqual(metadata.byte_size, len(b"fake-png-bytes"))
        self.assertEqual(metadata.extras["requested_width_px"], 960)
        self.assertEqual(metadata.extras["requested_height_px"], 600)
        self.assertEqual(metadata.extras["requested_fps"], 80)
        self.assertEqual(metadata.extras["requested_pixel_format"], "YUYV")
        self.assertEqual(metadata.extras["device"], "/dev/video-test")
        self.assertEqual(metadata.extras["backend"], "opencv_v4l2")
        self.assertEqual(metadata.extras["actual_width_px"], 960)
        self.assertEqual(metadata.extras["actual_height_px"], 600)
        self.assertEqual(metadata.extras["actual_fps"], 80)

    def test_read_failure_raises_clear_capture_error(self) -> None:
        cv2 = FakeCv2(capture=FakeCapture(read_success=False))
        publisher = _publisher(cv2=cv2)

        with self.assertRaisesRegex(FrameCaptureError, "Failed to capture frame"):
            publisher.publish_next()

    def test_encode_failure_raises_clear_encoding_error(self) -> None:
        cv2 = FakeCv2(encode_success=False)
        publisher = _publisher(cv2=cv2)

        with self.assertRaisesRegex(FrameEncodingError, "Failed to encode"):
            publisher.publish_next()

    def test_open_failure_raises_clear_open_error(self) -> None:
        cv2 = FakeCv2(capture=FakeCapture(opened=False))
        publisher = _publisher(cv2=cv2)

        with self.assertRaisesRegex(CameraOpenError, "Could not open OpenCV V4L2 camera"):
            publisher.open()

    def test_close_releases_capture(self) -> None:
        cv2 = FakeCv2()
        publisher = _publisher(cv2=cv2)

        publisher.open()
        publisher.close()

        self.assertTrue(cv2.capture.released)

    def test_camera_module_has_no_pyside6_or_gui_import(self) -> None:
        module_path = SRC_ROOT / "live_inference" / "cameras" / "opencv_v4l2_camera.py"
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        imported_modules: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
                imported_modules.add(node.module)

        self.assertNotIn("PySide6", imported_roots)
        self.assertFalse(any(module.startswith("live_inference.gui") for module in imported_modules))


class RecordingHandoffWriter:
    def __init__(self) -> None:
        self.publish_calls: list[tuple[bytes, FrameMetadata]] = []
        self.next_reference = FrameReference(
            image_path=Path("live_frames/latest_frame.png"),
            completed_at_utc=WRITTEN_AT,
        )

    def publish_frame(self, image_bytes: bytes, metadata: FrameMetadata) -> FrameReference:
        self.publish_calls.append((image_bytes, metadata))
        return self.next_reference


class FakeFrame:
    shape = (600, 960, 3)


class FakeEncodedBytes:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def tobytes(self) -> bytes:
        return self.payload


class FakeCapture:
    def __init__(self, *, opened: bool = True, read_success: bool = True) -> None:
        self.opened = opened
        self.read_success = read_success
        self.frame = FakeFrame()
        self.set_calls: list[tuple[int, int]] = []
        self.read_calls = 0
        self.released = False
        self.properties = {
            FakeCv2.CAP_PROP_FRAME_WIDTH: 960,
            FakeCv2.CAP_PROP_FRAME_HEIGHT: 600,
            FakeCv2.CAP_PROP_FPS: 80,
            FakeCv2.CAP_PROP_FOURCC: FakeCv2.yuyv_fourcc,
        }

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API shape
        return self.opened

    def set(self, prop_id: int, value: int) -> bool:
        self.set_calls.append((prop_id, value))
        self.properties[prop_id] = value
        return True

    def get(self, prop_id: int) -> int:
        return self.properties[prop_id]

    def read(self) -> tuple[bool, FakeFrame | None]:
        self.read_calls += 1
        return self.read_success, self.frame if self.read_success else None

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_V4L2 = 200
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_FOURCC = 6
    yuyv_fourcc = 0x56595559

    def __init__(
        self,
        *,
        capture: FakeCapture | None = None,
        encode_success: bool = True,
    ) -> None:
        self.capture = capture or FakeCapture()
        self.encode_success = encode_success
        self.video_capture_calls: list[tuple[str | int, int]] = []
        self.fourcc_calls: list[tuple[str, str, str, str]] = []
        self.imencode_calls: list[tuple[str, object]] = []

    def VideoCapture(self, device: str | int, backend: int) -> FakeCapture:  # noqa: N802
        self.video_capture_calls.append((device, backend))
        return self.capture

    def VideoWriter_fourcc(self, a: str, b: str, c: str, d: str) -> int:  # noqa: N802
        self.fourcc_calls.append((a, b, c, d))
        return self.yuyv_fourcc

    def imencode(self, extension: str, frame: object) -> tuple[bool, FakeEncodedBytes]:
        self.imencode_calls.append((extension, frame))
        return self.encode_success, FakeEncodedBytes(b"fake-png-bytes")


def _publisher(
    *,
    cv2: FakeCv2,
    writer: RecordingHandoffWriter | None = None,
    width_px: int = 960,
    height_px: int = 600,
    fps: int = 80,
    pixel_format: str = "YUYV",
    camera_name: str = "Camera Test",
) -> OpenCvV4L2CameraPublisher:
    return OpenCvV4L2CameraPublisher(
        handoff_writer=writer or RecordingHandoffWriter(),
        device="/dev/video-test",
        width_px=width_px,
        height_px=height_px,
        fps=fps,
        pixel_format=pixel_format,
        camera_name=camera_name,
        auto_open=False,
        now_utc_fn=lambda: NOW,
        cv2_module=cv2,
    )


if __name__ == "__main__":
    unittest.main()
