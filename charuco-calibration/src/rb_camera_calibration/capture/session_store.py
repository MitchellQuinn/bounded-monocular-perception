"""Filesystem storage for calibration sessions."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copy2
from typing import Any

from rb_camera_calibration.contracts import (
    AcceptedCalibrationFrame,
    CalibrationSessionConfig,
    CameraFrame,
    CaptureDecision,
    CharucoDetection,
    FrameHash,
    FrameQualityMetrics,
    PoseSignature,
)
from rb_camera_calibration.detection.charuco_detector import render_detection_overlay
from rb_camera_calibration.io.atomic_write import atomic_write_json
from rb_camera_calibration.io.image_store import write_png_bytes
from rb_camera_calibration.utils.timestamps import utc_now_iso


class CalibrationSessionStore:
    """Create and maintain the on-disk calibration run directory."""

    def __init__(self, session_config: CalibrationSessionConfig) -> None:
        self.session_config = session_config
        self.session_root = Path(session_config.session_root)
        self.accepted_dir = self.session_root / "accepted"
        self.rejected_dir = self.session_root / "rejected_samples"
        self._accepted_frames: list[AcceptedCalibrationFrame] = []

    @property
    def accepted_frames(self) -> tuple[AcceptedCalibrationFrame, ...]:
        return tuple(self._accepted_frames)

    @property
    def manifest_path(self) -> Path:
        return self.session_root / "session_manifest.json"

    def initialise(self, *, load_existing: bool = True) -> None:
        """Create session directories and load or write the session manifest."""
        self.accepted_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        if load_existing and self.manifest_path.exists():
            self._accepted_frames = list(load_accepted_frames_from_manifest(self.manifest_path))
            return
        self.write_session_manifest()

    def write_session_manifest(self) -> Path:
        """Write ``session_manifest.json`` atomically."""
        payload: dict[str, Any] = {
            "generated_at_utc": utc_now_iso(),
            "session_config": self.session_config.to_dict(),
            "accepted_frames": [frame.to_dict() for frame in self._accepted_frames],
        }
        atomic_write_json(self.manifest_path, payload)
        return self.manifest_path

    def store_accepted_frame(
        self,
        frame: CameraFrame,
        detection: CharucoDetection,
        decision: CaptureDecision,
        *,
        overlay_bytes: bytes | None = None,
    ) -> AcceptedCalibrationFrame:
        """Persist an accepted frame, detection JSON, overlay, and manifest entry."""
        if decision.pose_signature is None:
            raise ValueError("Accepted frames require a pose_signature in the capture decision.")
        if overlay_bytes is None and self.session_config.save_debug_overlays:
            overlay_bytes = render_detection_overlay(frame, detection)

        index = len(self._accepted_frames) + 1
        stem = f"frame_{index:04d}"
        image_path = self.accepted_dir / f"{stem}.png"
        detection_path = self.accepted_dir / f"{stem}_detection.json"
        overlay_path = self.accepted_dir / f"{stem}_overlay.png" if overlay_bytes else None
        write_png_bytes(image_path, frame.image_bytes)
        atomic_write_json(detection_path, detection.to_dict())
        if overlay_path is not None and overlay_bytes is not None:
            write_png_bytes(overlay_path, overlay_bytes)
        accepted = AcceptedCalibrationFrame(
            frame_id=frame.frame_id,
            image_path=image_path,
            detection_json_path=detection_path,
            overlay_path=overlay_path,
            frame_hash=frame.frame_hash,
            captured_at_utc=frame.metadata.captured_at_utc,
            charuco_corner_count=detection.charuco_corner_count,
            marker_count=detection.marker_count,
            pose_signature=decision.pose_signature,
            quality=decision.quality,
        )
        self._accepted_frames.append(accepted)
        self.write_session_manifest()
        return accepted

    def store_rejected_sample(
        self,
        frame: CameraFrame,
        detection: CharucoDetection,
        decision: CaptureDecision,
    ) -> Path | None:
        """Persist a rejected sample when configured for later debugging."""
        if not self.session_config.save_rejected_samples:
            return None
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        index = len(tuple(self.rejected_dir.glob("frame_*_decision.json"))) + 1
        stem = f"frame_{index:04d}"
        image_path = self.rejected_dir / f"{stem}.png"
        decision_path = self.rejected_dir / f"{stem}_decision.json"
        write_png_bytes(image_path, frame.image_bytes)
        atomic_write_json(
            decision_path,
            {
                "decision": decision.to_dict(),
                "detection": detection.to_dict(),
            },
        )
        return image_path

    def remove_accepted_frame(self, frame_id: str) -> bool:
        """Remove an accepted frame from the in-memory manifest list.

        Files are left on disk for auditability; recalibration uses the manifest list.
        """
        before = len(self._accepted_frames)
        self._accepted_frames = [frame for frame in self._accepted_frames if frame.frame_id != frame_id]
        changed = len(self._accepted_frames) != before
        if changed:
            self.write_session_manifest()
        return changed

    def reset_manifest(self) -> None:
        """Clear the in-memory accepted-frame list and rewrite the manifest."""
        self._accepted_frames.clear()
        self.write_session_manifest()

    def append_existing_accepted_frame(
        self,
        accepted: AcceptedCalibrationFrame,
        *,
        write_manifest: bool = True,
    ) -> AcceptedCalibrationFrame:
        """Append an already persisted accepted frame record to this session."""
        self._accepted_frames.append(accepted)
        if write_manifest:
            self.write_session_manifest()
        return accepted

    def copy_accepted_frame_from(
        self,
        accepted: AcceptedCalibrationFrame,
        *,
        source_session_root: Path,
    ) -> AcceptedCalibrationFrame:
        """Copy one accepted frame record into this session with a new filename."""
        index = len(self._accepted_frames) + 1
        stem = f"frame_{index:04d}"
        image_path = self.accepted_dir / f"{stem}.png"
        detection_path = self.accepted_dir / f"{stem}_detection.json"
        overlay_path = self.accepted_dir / f"{stem}_overlay.png" if accepted.overlay_path else None

        _copy_required(accepted.image_path, image_path)
        _copy_required(accepted.detection_json_path, detection_path)
        if accepted.overlay_path is not None and overlay_path is not None and accepted.overlay_path.exists():
            copy2(accepted.overlay_path, overlay_path)
        else:
            overlay_path = None

        merged = AcceptedCalibrationFrame(
            frame_id=accepted.frame_id,
            image_path=image_path,
            detection_json_path=detection_path,
            overlay_path=overlay_path,
            frame_hash=accepted.frame_hash,
            captured_at_utc=accepted.captured_at_utc,
            charuco_corner_count=accepted.charuco_corner_count,
            marker_count=accepted.marker_count,
            pose_signature=accepted.pose_signature,
            quality=accepted.quality,
            extras={
                **dict(accepted.extras),
                "merged_from_session": str(source_session_root),
                "merged_from_image_path": str(accepted.image_path),
            },
        )
        return self.append_existing_accepted_frame(merged, write_manifest=False)


def load_accepted_frames_from_manifest(manifest_path: Path) -> tuple[AcceptedCalibrationFrame, ...]:
    """Load accepted-frame records from a session manifest."""
    try:
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return ()
    if not isinstance(payload, dict):
        raise ValueError(f"Session manifest must contain a JSON object: {manifest_path}")
    frames = payload.get("accepted_frames", [])
    if not isinstance(frames, list):
        raise ValueError(f"Session manifest accepted_frames must be a list: {manifest_path}")
    return tuple(_accepted_frame_from_dict(item, Path(manifest_path)) for item in frames)


def _accepted_frame_from_dict(
    payload: MappingLike,
    manifest_path: Path,
) -> AcceptedCalibrationFrame:
    if not isinstance(payload, dict):
        raise ValueError(f"Accepted frame manifest entry must be an object: {manifest_path}")
    frame_hash_payload = payload.get("frame_hash", {})
    pose_payload = payload.get("pose_signature", {})
    quality_payload = payload.get("quality", {})
    return AcceptedCalibrationFrame(
        frame_id=str(payload["frame_id"]),
        image_path=_resolve_manifest_path(payload["image_path"], manifest_path),
        detection_json_path=_resolve_manifest_path(payload["detection_json_path"], manifest_path),
        overlay_path=_optional_manifest_path(payload.get("overlay_path"), manifest_path),
        frame_hash=FrameHash(
            value=str(frame_hash_payload["value"]),
            algorithm=str(frame_hash_payload.get("algorithm", "blake2b-128")),
            digest_size_bytes=int(frame_hash_payload.get("digest_size_bytes", 16)),
        ),
        captured_at_utc=str(payload["captured_at_utc"]),
        charuco_corner_count=int(payload["charuco_corner_count"]),
        marker_count=int(payload["marker_count"]),
        pose_signature=PoseSignature(
            center_x_norm=float(pose_payload["center_x_norm"]),
            center_y_norm=float(pose_payload["center_y_norm"]),
            area_fraction=float(pose_payload["area_fraction"]),
            roll_like_angle_deg=float(pose_payload["roll_like_angle_deg"]),
            perspective_skew_score=float(pose_payload["perspective_skew_score"]),
            grid_cell=tuple(int(v) for v in pose_payload["grid_cell"]),  # type: ignore[arg-type]
            scale_bin=int(pose_payload["scale_bin"]),
            tilt_bin=int(pose_payload["tilt_bin"]),
            extras=dict(pose_payload.get("extras", {})),
        ),
        quality=FrameQualityMetrics(
            laplacian_variance=float(quality_payload["laplacian_variance"]),
            mean_luma=float(quality_payload["mean_luma"]),
            luma_std=float(quality_payload["luma_std"]),
            clipped_black_fraction=float(quality_payload["clipped_black_fraction"]),
            clipped_white_fraction=float(quality_payload["clipped_white_fraction"]),
            contrast_score=float(quality_payload["contrast_score"]),
            blur_score=float(quality_payload["blur_score"]),
            extras=dict(quality_payload.get("extras", {})),
        ),
        extras=dict(payload.get("extras", {})),
    )


def _optional_manifest_path(value: Any, manifest_path: Path) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_manifest_path(value, manifest_path)


def _resolve_manifest_path(value: Any, manifest_path: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    parts = path.parts
    for marker in ("accepted", "rejected_samples"):
        if marker in parts:
            suffix = Path(*parts[parts.index(marker) :])
            candidate = manifest_path.parent / suffix
            if candidate.exists():
                return candidate
            return candidate
    return manifest_path.parent / path


def _copy_required(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Cannot merge accepted frame; source file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy2(source, destination)


MappingLike = dict[str, Any]
