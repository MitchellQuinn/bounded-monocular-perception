"""Export machine-readable calibration artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from rb_camera_calibration.contracts import (
    CAMERA_CALIBRATION_ARTIFACT_VERSION,
    CalibrationArtifactManifest,
    CalibrationResult,
    CalibrationSessionConfig,
)
from rb_camera_calibration.io.atomic_write import atomic_write_json, atomic_write_yaml
from rb_camera_calibration.utils.timestamps import utc_now_iso


class CalibrationArtifactExporter:
    """Write JSON/YAML calibration artifacts and per-view diagnostics."""

    def export(
        self,
        result: CalibrationResult,
        session_config: CalibrationSessionConfig,
    ) -> CalibrationArtifactManifest:
        session_root = Path(session_config.session_root)
        session_root.mkdir(parents=True, exist_ok=True)
        result_json = session_root / "calibration_result.json"
        result_yaml = session_root / "calibration_result.yaml"
        report_csv = session_root / "per_view_reprojection_errors.csv"
        payload = calibration_result_payload(result, session_config)
        atomic_write_json(result_json, payload)
        atomic_write_yaml(result_yaml, payload)
        _write_per_view_csv(report_csv, result)
        manifest = CalibrationArtifactManifest(
            artifact_version=CAMERA_CALIBRATION_ARTIFACT_VERSION,
            generated_at_utc=utc_now_iso(),
            session_root=session_root,
            board_config=session_config.board_config,
            camera_config=session_config.camera_config,
            capture_policy=session_config.capture_policy,
            result_json_path=result_json,
            result_yaml_path=result_yaml,
            accepted_frame_dir=session_root / "accepted",
            rejected_sample_dir=session_root / "rejected_samples",
            report_csv_path=report_csv,
        )
        atomic_write_json(session_root / "artifact_manifest.json", manifest.to_dict())
        return manifest


def calibration_result_payload(
    result: CalibrationResult,
    session_config: CalibrationSessionConfig,
) -> dict[str, Any]:
    """Return a plain artifact payload consumed by runtime code."""
    board = result.board_config
    camera = session_config.camera_config
    return {
        "artifact_version": CAMERA_CALIBRATION_ARTIFACT_VERSION,
        "generated_at_utc": result.generated_at_utc,
        "success": result.success,
        "camera_name": str(camera.extras.get("camera_name", camera.camera_device)),
        "camera_device": str(camera.camera_device),
        "image_size_wh_px": list(result.image_size_wh_px),
        "board": {
            "type": board.pattern_type.value,
            "squares_x": board.squares_x,
            "squares_y": board.squares_y,
            "square_length_m": board.square_length_m,
            "marker_length_m": board.marker_length_m,
            "aruco_dictionary": board.aruco_dictionary,
            "board_name": board.board_name,
        },
        "camera_matrix": [list(row) for row in result.camera_matrix],
        "distortion_coefficients": list(result.distortion_coefficients),
        "rms_reprojection_error_px": result.rms_reprojection_error_px,
        "per_view_errors": [error.to_dict() for error in result.per_view_errors],
        "opencv_version": result.opencv_version,
        "calibration_flags": list(result.calibration_flags),
        "accepted_frame_count": result.accepted_frame_count,
        "used_frame_count": result.used_frame_count,
        "rejected_outlier_count": result.rejected_outlier_count,
        "extras": result.extras,
    }


def _write_per_view_csv(path: Path, result: CalibrationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_id",
                "rms_error_px",
                "mean_error_px",
                "max_error_px",
                "point_count",
                "include_in_final",
            ],
        )
        writer.writeheader()
        for error in result.per_view_errors:
            writer.writerow(
                {
                    "frame_id": error.frame_id,
                    "rms_error_px": error.rms_error_px,
                    "mean_error_px": error.mean_error_px,
                    "max_error_px": error.max_error_px,
                    "point_count": error.point_count,
                    "include_in_final": error.include_in_final,
                }
            )
    tmp_path.replace(path)
