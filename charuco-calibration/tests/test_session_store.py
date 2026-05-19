from __future__ import annotations

import json

from rb_camera_calibration.capture import merge_sessions as merge_module
from rb_camera_calibration.capture.merge_sessions import merge_sessions
from rb_camera_calibration.capture.session_store import CalibrationSessionStore
from rb_camera_calibration.contracts import (
    CalibrationSessionConfig,
    CameraCaptureConfig,
    CameraFrame,
    CaptureDecision,
    CaptureDecisionType,
    CapturePolicyConfig,
    CharucoBoardConfig,
    CharucoDetection,
    FrameHash,
    FrameMetadata,
    FrameQualityMetrics,
    PoseSignature,
)


def test_session_store_writes_manifest_and_accepted_metadata(tmp_path) -> None:
    session = _session(tmp_path / "run")
    store = CalibrationSessionStore(session)
    store.initialise()
    frame = _frame()
    detection = _detection()
    decision = _decision(detection)

    accepted = store.store_accepted_frame(frame, detection, decision, overlay_bytes=b"overlay")

    assert accepted.image_path.exists()
    assert accepted.detection_json_path.exists()
    manifest = json.loads((tmp_path / "run" / "session_manifest.json").read_text(encoding="utf-8"))
    assert manifest["accepted_frames"][0]["frame_id"] == "frame-1"


def test_session_store_loads_existing_manifest(tmp_path) -> None:
    session = _session(tmp_path / "run")
    store = CalibrationSessionStore(session)
    store.initialise()
    store.store_accepted_frame(_frame(), _detection(), _decision(_detection()), overlay_bytes=b"overlay")

    resumed = CalibrationSessionStore(session)
    resumed.initialise()

    assert len(resumed.accepted_frames) == 1
    assert resumed.accepted_frames[0].frame_id == "frame-1"


def test_merge_sessions_copies_accepted_frames(tmp_path) -> None:
    source_a = _session(tmp_path / "source_a")
    source_b = _session(tmp_path / "source_b")
    store_a = CalibrationSessionStore(source_a)
    store_b = CalibrationSessionStore(source_b)
    store_a.initialise()
    store_b.initialise()
    store_a.store_accepted_frame(_frame(frame_id="frame-a", hash_value="a"), _detection(frame_id="frame-a"), _decision(_detection(frame_id="frame-a")), overlay_bytes=b"overlay-a")
    store_b.store_accepted_frame(_frame(frame_id="frame-b", hash_value="b"), _detection(frame_id="frame-b"), _decision(_detection(frame_id="frame-b")), overlay_bytes=b"overlay-b")
    board_path = tmp_path / "board.toml"
    board_path.write_text(
        """
[board]
pattern_type = "charuco"
squares_x = 10
squares_y = 15
square_length_m = 0.015
marker_length_m = 0.011
aruco_dictionary = "DICT_4X4_100"
board_name = "merged"
""",
        encoding="utf-8",
    )
    camera_path = tmp_path / "camera.toml"
    camera_path.write_text(
        """
[camera]
camera_source_type = "opencv_v4l2"
camera_device = "/dev/video0"
width_px = 1920
height_px = 1200
fps = 50
pixel_format = "YUYV"
backend = "V4L2"
""",
        encoding="utf-8",
    )

    merged_count, manifest_path = merge_sessions(
        output_session_root=tmp_path / "merged",
        source_session_roots=(tmp_path / "source_a", tmp_path / "source_b"),
        board_config_path=board_path,
        camera_config_path=camera_path,
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert merged_count == 2
    assert len(payload["accepted_frames"]) == 2
    assert (tmp_path / "merged" / "accepted" / "frame_0001.png").exists()
    assert (tmp_path / "merged" / "accepted" / "frame_0002.png").exists()


def test_discover_merge_session_roots_excludes_current(tmp_path, monkeypatch) -> None:
    default_root = tmp_path / "charuco-calibration" / "calibration_runs"
    current = default_root / "current"
    other = default_root / "other"
    legacy = tmp_path / "calibration_runs" / "legacy"
    for root in (current, other, legacy):
        root.mkdir(parents=True)
        (root / "session_manifest.json").write_text(
            '{"accepted_frames": []}',
            encoding="utf-8",
        )
    monkeypatch.setattr(merge_module, "default_calibration_runs_root", lambda: default_root)

    discovered = merge_module.discover_merge_session_roots(current)

    assert other.resolve() in discovered
    assert legacy.resolve() in discovered
    assert current.resolve() not in discovered


def _session(root) -> CalibrationSessionConfig:
    session = CalibrationSessionConfig(
        session_root=root,
        board_config=CharucoBoardConfig(
            squares_x=15,
            squares_y=10,
            square_length_m=0.015,
            marker_length_m=0.011,
            aruco_dictionary="DICT_5X5_100",
        ),
        camera_config=CameraCaptureConfig(),
        capture_policy=CapturePolicyConfig(require_stability=False),
    )
    return session


def _frame(frame_id: str = "frame-1", hash_value: str = "abc") -> CameraFrame:
    metadata = FrameMetadata(
        frame_id=frame_id,
        sequence_index=1,
        captured_at_utc="2026-05-19T00:00:00Z",
        width_px=900,
        height_px=600,
        pixel_format="png",
        source_name="test",
    )
    return CameraFrame(
        frame_id=frame_id,
        metadata=metadata,
        frame_hash=FrameHash(value=hash_value),
        image_bytes=b"png-bytes",
    )


def _detection(frame_id: str = "frame-1") -> CharucoDetection:
    return CharucoDetection(
        frame_id=frame_id,
        detected=True,
        marker_count=12,
        charuco_corner_count=30,
    )


def _decision(detection: CharucoDetection) -> CaptureDecision:
    quality = FrameQualityMetrics(100.0, 120.0, 30.0, 0.0, 0.0, 0.5, 0.5)
    signature = PoseSignature(0.5, 0.5, 0.2, 0.0, 0.1, (1, 1), 1, 0)
    return CaptureDecision(
        decision_type=CaptureDecisionType.ACCEPT,
        accepted=True,
        reason=None,
        message="accepted",
        detection=detection,
        quality=quality,
        pose_signature=signature,
    )
