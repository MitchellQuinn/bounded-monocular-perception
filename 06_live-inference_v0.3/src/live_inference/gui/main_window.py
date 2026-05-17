"""Qt Designer backed live inference main window."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QImage
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QCheckBox,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

import interfaces.contracts as contracts
from live_inference.frame_handoff import compute_frame_hash
from live_inference.masking import BackgroundState

from .frame_preview_widget import FramePreviewOverlay, FramePreviewWidget


@dataclass(frozen=True)
class _CapturedSingleFrame:
    image_bytes: bytes
    frame_hash: contracts.FrameHash
    source_path: Path | None
    frame_metadata: contracts.FrameMetadata | None


class LiveInferenceMainWindow(QMainWindow):
    """Small workflow-oriented GUI loaded from ``live_main_window.ui``."""

    def __init__(
        self,
        *,
        camera_controller: object,
        inference_controller: object,
        frame_reader: object | None = None,
        single_frame_runner: object | None = None,
        trace_output_dir: Path | str | None = None,
        background_state: BackgroundState | None = None,
        locator_parameter_state: object | None = None,
        locator_kind: contracts.LocatorKind | str = contracts.LocatorKind.BACKGROUND_EDGE_V1,
        stop_wait_ms: int = 1000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_controller = camera_controller
        self.inference_controller = inference_controller
        self.frame_reader = frame_reader
        self.single_frame_runner = single_frame_runner
        self.trace_output_dir = Path(trace_output_dir) if trace_output_dir is not None else None
        self.background_state = background_state or BackgroundState()
        self.locator_parameter_state = locator_parameter_state
        self.locator_kind = (
            locator_kind
            if isinstance(locator_kind, contracts.LocatorKind)
            else contracts.LocatorKind(str(locator_kind))
        )
        self._stop_wait_ms = int(stop_wait_ms)
        self._captured_single_frame: _CapturedSingleFrame | None = None
        self._base_preview_image: QImage | None = None
        self._last_overlay: FramePreviewOverlay | None = None
        self._debug_paths: dict[str, Path] = {}
        self._last_preview_update_seconds = 0.0
        self._preview_update_interval_seconds = 1.0 / 15.0

        self.setWindowTitle("Live Defender Inference v0.3")
        self._load_ui()
        self._bind_widgets()
        self._connect_ui()
        self._connect_worker_signals()
        self._sync_parameter_widgets()
        self._append_log("INFO", f"Locator: {self.locator_kind.value}")

    def _load_ui(self) -> None:
        loader = QUiLoader()
        loader.registerCustomWidget(FramePreviewWidget)
        ui_path = Path(__file__).with_name("live_main_window.ui")
        ui_file = QFile(str(ui_path))
        if not ui_file.open(QFile.OpenModeFlag.ReadOnly):
            raise OSError(f"Could not open Qt Designer UI file: {ui_path}")
        try:
            root = loader.load(ui_file, self)
        finally:
            ui_file.close()
        if root is None:
            raise RuntimeError(f"Could not load Qt Designer UI file: {ui_path}")
        self.setCentralWidget(root)

    def _bind_widgets(self) -> None:
        self.main_preview_widget = self._require(FramePreviewWidget, "mainPreviewWidget")
        self.start_camera_button = self._require(QPushButton, "startCameraButton")
        self.stop_camera_button = self._require(QPushButton, "stopCameraButton")
        self.capture_background_button = self._require(QPushButton, "captureBackgroundButton")
        self.clear_background_button = self._require(QPushButton, "clearBackgroundButton")
        self.capture_frame_button = self._require(QPushButton, "captureFrameButton")
        self.run_locator_button = self._require(QPushButton, "runLocatorButton")
        self.run_single_inference_button = self._require(QPushButton, "runSingleInferenceButton")
        self.start_continuous_button = self._require(QPushButton, "startContinuousButton")
        self.stop_continuous_button = self._require(QPushButton, "stopContinuousButton")
        self.record_trace_checkbox = self._require(QCheckBox, "recordTraceCheckBox")
        self.show_roi_checkbox = self._require(QCheckBox, "showRoiCheckBox")
        self.show_bbox_checkbox = self._require(QCheckBox, "showBboxCheckBox")
        self.show_foreground_mask_checkbox = self._require(QCheckBox, "showForegroundMaskCheckBox")
        self.show_edges_checkbox = self._require(QCheckBox, "showEdgesCheckBox")
        self.show_candidate_contours_checkbox = self._require(QCheckBox, "showCandidateContoursCheckBox")
        self.show_chosen_contour_checkbox = self._require(QCheckBox, "showChosenContourCheckBox")
        self.background_threshold_spinbox = self._require(QSpinBox, "backgroundThresholdSpinBox")
        self.min_foreground_area_spinbox = self._require(QSpinBox, "minForegroundAreaSpinBox")
        self.canny_low_spinbox = self._require(QSpinBox, "cannyLowSpinBox")
        self.canny_high_spinbox = self._require(QSpinBox, "cannyHighSpinBox")
        self.distance_value = self._require(QLabel, "distanceValue")
        self.yaw_value = self._require(QLabel, "yawValue")
        self.locator_status_value = self._require(QLabel, "locatorStatusValue")
        self.roi_status_value = self._require(QLabel, "roiStatusValue")
        self.frame_hash_value = self._require(QLabel, "frameHashValue")
        self.trace_path_value = self._require(QLabel, "tracePathValue")
        self.warnings_text = self._require(QPlainTextEdit, "warningsText")
        self.artifact_summary_value = self._require(QLabel, "artifactSummaryValue")
        self.log_output = self._require(QPlainTextEdit, "logOutput")

    def _connect_ui(self) -> None:
        self.start_camera_button.clicked.connect(self.start_camera)
        self.stop_camera_button.clicked.connect(self.stop_camera)
        self.capture_background_button.clicked.connect(self.capture_background)
        self.clear_background_button.clicked.connect(self.clear_background)
        self.capture_frame_button.clicked.connect(self.capture_frame)
        self.run_locator_button.clicked.connect(self.run_locator)
        self.run_single_inference_button.clicked.connect(self.run_single_inference)
        self.start_continuous_button.clicked.connect(self.start_continuous_inference)
        self.stop_continuous_button.clicked.connect(self.stop_continuous_inference)
        for checkbox in (
            self.show_roi_checkbox,
            self.show_bbox_checkbox,
            self.show_foreground_mask_checkbox,
            self.show_edges_checkbox,
            self.show_candidate_contours_checkbox,
            self.show_chosen_contour_checkbox,
        ):
            checkbox.toggled.connect(self._refresh_visual_surface)
        for spinbox in (
            self.background_threshold_spinbox,
            self.min_foreground_area_spinbox,
            self.canny_low_spinbox,
            self.canny_high_spinbox,
        ):
            spinbox.valueChanged.connect(self._on_locator_parameters_changed)

    def _connect_worker_signals(self) -> None:
        for controller in (self.camera_controller, self.inference_controller):
            signals = getattr(controller, "signals", None)
            if signals is None:
                continue
            for name, handler in {
                "status_changed": self._on_status_changed,
                "lifecycle_event": self._on_lifecycle_event,
                "warning_occurred": self._on_warning_occurred,
                "error_occurred": self._on_error_occurred,
                "frame_written": self._on_frame_written,
                "frame_skipped": self._on_frame_skipped,
                "result_ready": self._on_inference_result_ready,
                "debug_image_ready": self._on_debug_image_ready,
            }.items():
                signal = getattr(signals, name, None)
                if signal is not None:
                    signal.connect(handler)

    def _require(self, widget_type: type, object_name: str) -> Any:
        widget = self.findChild(widget_type, object_name)
        if widget is None:
            raise RuntimeError(f"UI file is missing {object_name!r}.")
        return widget

    def start_camera(self) -> None:
        start = getattr(self.camera_controller, "start", None)
        if callable(start):
            start()
            self._append_log("INFO", "Camera start requested")

    def stop_camera(self) -> None:
        self._stop_controller(self.camera_controller, "camera")

    def start_continuous_inference(self) -> None:
        start = getattr(self.inference_controller, "start", None)
        if callable(start):
            start()
            self._append_log("INFO", "Continuous inference start requested")

    def stop_continuous_inference(self) -> None:
        self._stop_controller(self.inference_controller, "inference")

    def stop_all(self) -> None:
        self.stop_continuous_inference()
        self.stop_camera()

    def capture_frame(self) -> None:
        latest_frame = self._latest_frame()
        if latest_frame is None:
            self._append_log("WARNING", "Capture Frame requires a completed frame.")
            return
        image_bytes = self.frame_reader.read_frame_bytes(latest_frame)
        frame_hash = compute_frame_hash(image_bytes)
        source_path = _path_or_none(_payload_value(latest_frame, "image_path"))
        metadata = _payload_value(latest_frame, "metadata")
        frame_metadata = metadata if isinstance(metadata, contracts.FrameMetadata) else None
        self._captured_single_frame = _CapturedSingleFrame(
            image_bytes=image_bytes,
            frame_hash=frame_hash,
            source_path=source_path,
            frame_metadata=frame_metadata,
        )
        self._set_base_preview_from_bytes(image_bytes)
        self.frame_hash_value.setText(f"frame hash: {frame_hash.value}")
        self._append_log("INFO", f"Captured frame {frame_hash.value}")

    def capture_background(self) -> None:
        image_bytes = (
            self._captured_single_frame.image_bytes
            if self._captured_single_frame is not None
            else self._latest_frame_bytes()
        )
        if image_bytes is None:
            self._append_log("WARNING", "Capture Background requires a frame.")
            return
        gray = _decode_grayscale(image_bytes)
        revision = self.background_state.capture_background(gray)
        self.background_state.set_enabled(True)
        self._append_log("INFO", f"Background captured; revision={revision}")

    def clear_background(self) -> None:
        revision = self.background_state.clear()
        self._append_log("INFO", f"Background cleared; revision={revision}")

    def run_locator(self) -> None:
        captured = self._require_captured_frame("Run Locator")
        if captured is None:
            return
        runner = self.single_frame_runner
        method = getattr(runner, "run_locator_only", None) or getattr(
            runner,
            "run_roi_locator_only",
            None,
        )
        if not callable(method):
            self._append_log("ERROR", "Locator diagnostics are unavailable.")
            return
        outcome = method(
            captured.image_bytes,
            source_path=captured.source_path,
            frame_metadata=captured.frame_metadata,
            record_trace=self.record_trace_checkbox.isChecked(),
        )
        self._apply_locator_outcome(outcome)

    def run_single_inference(self) -> None:
        captured = self._require_captured_frame("Run Single Inference")
        if captured is None:
            return
        runner = self.single_frame_runner
        if runner is None:
            self._append_log("ERROR", "Single-frame runner unavailable.")
            return
        outcome = runner.run_single_frame(
            captured.image_bytes,
            source_path=captured.source_path,
            frame_metadata=captured.frame_metadata,
            record_trace=self.record_trace_checkbox.isChecked(),
        )
        result = _payload_value(outcome, "result")
        error = _payload_value(outcome, "error")
        trace_path = _path_or_none(_payload_value(outcome, "trace_path"))
        if result is not None:
            self._on_inference_result_ready(result)
            self._append_log("INFO", "Single-frame inference completed")
        if trace_path is not None:
            self._set_trace_path(trace_path)
        if error is not None:
            self._on_error_occurred(error)

    def _apply_locator_outcome(self, outcome: object) -> None:
        result = _payload_value(outcome, "result")
        error = _payload_value(outcome, "error")
        trace_path = _path_or_none(_payload_value(outcome, "trace_path"))
        if result is not None:
            metadata = _mapping_payload(_payload_value(result, "preprocessing_metadata"))
            locator_result = _payload_value(result, "locator_result")
            debug_paths = _mapping_payload(_payload_value(result, "debug_paths"))
            self._debug_paths = {str(k): Path(v) for k, v in debug_paths.items()}
            self._update_status_from_metadata(metadata)
            self._last_overlay = _overlay_from_metadata(metadata, locator_result)
            self._refresh_artifact_summary()
            self._refresh_visual_surface()
            self._append_log("INFO", "Locator run completed")
        if trace_path is not None:
            self._set_trace_path(trace_path)
        if error is not None:
            self._on_error_occurred(error)

    def _on_inference_result_ready(self, result: object) -> None:
        self.distance_value.setText(
            f"distance: {_format_value(_payload_value(result, 'predicted_distance_m'), 'm', 3)}"
        )
        self.yaw_value.setText(
            f"yaw: {_format_value(_payload_value(result, 'predicted_yaw_deg'), 'deg', 2)}"
        )
        roi_metadata = _payload_value(result, "roi_metadata")
        extras = _mapping_payload(_payload_value(roi_metadata, "extras"))
        self._debug_paths = {
            str(key): Path(path)
            for key, path in _mapping_payload(_payload_value(result, "debug_paths")).items()
        }
        self._update_status_from_metadata(extras)
        self._last_overlay = _overlay_from_roi_metadata(roi_metadata)
        self._set_warnings(_sequence_payload(_payload_value(result, "warnings")))
        self._refresh_artifact_summary()
        self._refresh_visual_surface()

    def _on_frame_written(self, frame: object) -> None:
        if self._captured_single_frame is not None:
            return
        now = monotonic()
        if now - self._last_preview_update_seconds < self._preview_update_interval_seconds:
            return
        self._last_preview_update_seconds = now
        path = _path_or_none(_payload_value(frame, "image_path"))
        if path is not None:
            self.main_preview_widget.load_image(path)

    def _on_status_changed(self, status: object) -> None:
        worker = _enum_text(_payload_value(status, "worker_name")) or "worker"
        state = _enum_text(_payload_value(status, "state")) or "state"
        message = _text(_payload_value(status, "message"), default="")
        self._append_log("DEBUG", f"{worker}: {state} {message}".strip())

    def _on_lifecycle_event(self, event: object) -> None:
        worker = _enum_text(_payload_value(event, "worker_name")) or "worker"
        state = _enum_text(_payload_value(event, "state")) or "state"
        self._append_log("INFO", f"{worker}: {state}")

    def _on_warning_occurred(self, warning: object) -> None:
        self._append_log("WARNING", _issue_text(warning))

    def _on_error_occurred(self, error: object) -> None:
        details = _mapping_payload(_payload_value(error, "details"))
        metadata = _mapping_payload(details.get("preprocessing_metadata"))
        if metadata:
            self._update_status_from_metadata(metadata)
            self._debug_paths.update(
                {str(k): Path(v) for k, v in _mapping_payload(metadata.get("debug_paths")).items()}
            )
            self._refresh_artifact_summary()
            self._refresh_visual_surface()
        self._append_log("ERROR", _issue_text(error))

    def _on_frame_skipped(self, skipped: object) -> None:
        reason = _enum_text(_payload_value(skipped, "reason")) or "unknown"
        if reason != "duplicate_hash":
            self._append_log("DEBUG", f"Frame skipped: {reason}")

    def _on_debug_image_ready(self, image: object) -> None:
        kind = _text(_payload_value(image, "image_kind"), default="debug")
        path = _path_or_none(_payload_value(image, "path"))
        if path is not None:
            self._debug_paths[kind] = path
            self._refresh_artifact_summary()

    def _refresh_visual_surface(self) -> None:
        artifact_key = self._selected_artifact_key()
        if artifact_key is not None:
            path = self._debug_paths.get(artifact_key)
            if path is not None and self.main_preview_widget.load_image(path):
                self.main_preview_widget.set_overlay(None)
                return
        if self._base_preview_image is not None:
            self.main_preview_widget.set_image(self._base_preview_image)
        self.main_preview_widget.set_overlay(self._filtered_overlay())

    def _selected_artifact_key(self) -> str | None:
        checks = (
            (self.show_chosen_contour_checkbox, contracts.DISPLAY_ARTIFACT_CHOSEN_CONTOUR),
            (self.show_candidate_contours_checkbox, contracts.DISPLAY_ARTIFACT_CANDIDATE_CONTOURS),
            (self.show_edges_checkbox, contracts.DISPLAY_ARTIFACT_EDGE_MAP),
            (self.show_foreground_mask_checkbox, contracts.DISPLAY_ARTIFACT_FOREGROUND_MASK),
        )
        for checkbox, key in checks:
            if checkbox.isChecked():
                return key
        return None

    def _filtered_overlay(self) -> FramePreviewOverlay | None:
        overlay = self._last_overlay
        if overlay is None:
            return None
        return FramePreviewOverlay(
            source_image_wh_px=overlay.source_image_wh_px,
            bbox_xyxy_px=overlay.bbox_xyxy_px if self.show_bbox_checkbox.isChecked() else None,
            center_xy_px=overlay.center_xy_px,
            roi_bounds_xyxy_px=overlay.roi_bounds_xyxy_px if self.show_roi_checkbox.isChecked() else None,
            label=overlay.label,
        )

    def _update_status_from_metadata(self, metadata: Mapping[str, object]) -> None:
        locator_kind = _text(
            metadata.get(contracts.PREPROCESSING_METADATA_LOCATOR_KIND),
            default=self.locator_kind.value,
        )
        confidence = _first_present(
            metadata.get(contracts.PREPROCESSING_METADATA_LOCATOR_CONFIDENCE),
            metadata.get(contracts.PREPROCESSING_METADATA_ROI_LOCATOR_CONFIDENCE),
            metadata.get(contracts.PREPROCESSING_METADATA_ROI_CONFIDENCE),
        )
        candidate_count = metadata.get(contracts.PREPROCESSING_METADATA_LOCATOR_CANDIDATE_COUNT)
        self.locator_status_value.setText(
            f"locator: {locator_kind}; confidence {_format_optional_float(confidence, 3)}; "
            f"candidates {_text(candidate_count, default='n/a')}"
        )
        accepted = _optional_bool(metadata.get(contracts.PREPROCESSING_METADATA_ROI_ACCEPTED))
        reason = _text(metadata.get(contracts.PREPROCESSING_METADATA_ROI_REJECTION_REASON), default="")
        if accepted is True:
            roi_text = "ROI: accepted"
        elif accepted is False:
            roi_text = f"ROI: rejected {reason}".strip()
        else:
            roi_text = "ROI: n/a"
        self.roi_status_value.setText(roi_text)
        warnings = _sequence_payload(metadata.get(contracts.PREPROCESSING_METADATA_WARNINGS))
        self._set_warnings(warnings)
        paths = _mapping_payload(metadata.get(contracts.PREPROCESSING_METADATA_DEBUG_PATHS))
        self._debug_paths.update({str(k): Path(v) for k, v in paths.items()})

    def _set_warnings(self, warnings: tuple[object, ...]) -> None:
        self.warnings_text.setPlainText("\n".join(str(warning) for warning in warnings))

    def _refresh_artifact_summary(self) -> None:
        if not self._debug_paths:
            self.artifact_summary_value.setText("artifacts: n/a")
            return
        keys = ", ".join(sorted(self._debug_paths))
        self.artifact_summary_value.setText(f"artifacts: {keys}")

    def _on_locator_parameters_changed(self, _value: int) -> None:
        state = self.locator_parameter_state
        update = getattr(state, "update", None)
        if not callable(update):
            return
        config, revision = update(
            background_threshold=self.background_threshold_spinbox.value(),
            min_foreground_area_px=self.min_foreground_area_spinbox.value(),
            canny_low_threshold=self.canny_low_spinbox.value(),
            canny_high_threshold=self.canny_high_spinbox.value(),
        )
        self.background_state.set_threshold(config.background_threshold)
        self._append_log("DEBUG", f"Locator parameters revision={revision}")

    def _sync_parameter_widgets(self) -> None:
        state = self.locator_parameter_state
        snapshot = getattr(state, "snapshot", None)
        if not callable(snapshot):
            return
        config, _revision = snapshot()
        self.background_threshold_spinbox.setValue(int(config.background_threshold))
        self.min_foreground_area_spinbox.setValue(int(config.min_foreground_area_px))
        self.canny_low_spinbox.setValue(int(config.canny_low_threshold))
        self.canny_high_spinbox.setValue(int(config.canny_high_threshold))

    def _latest_frame(self) -> object | None:
        if self.frame_reader is None:
            return None
        latest = getattr(self.frame_reader, "latest_completed_frame", None)
        if not callable(latest):
            return None
        return latest()

    def _latest_frame_bytes(self) -> bytes | None:
        latest_frame = self._latest_frame()
        if latest_frame is None or self.frame_reader is None:
            return None
        return self.frame_reader.read_frame_bytes(latest_frame)

    def _require_captured_frame(self, action: str) -> _CapturedSingleFrame | None:
        if self._captured_single_frame is None:
            self._append_log("WARNING", f"{action} requires Capture Frame first.")
            return None
        return self._captured_single_frame

    def _set_base_preview_from_bytes(self, image_bytes: bytes) -> None:
        image = QImage.fromData(bytes(image_bytes))
        if image.isNull():
            self._append_log("WARNING", "Captured frame could not be displayed.")
            return
        self._base_preview_image = image.copy()
        self.main_preview_widget.set_image(self._base_preview_image)

    def _set_trace_path(self, trace_path: Path) -> None:
        self.trace_path_value.setText(f"trace: {trace_path}")
        self._append_log("INFO", f"Trace written to {trace_path}")

    def _stop_controller(self, controller: object, label: str) -> None:
        request_stop = getattr(controller, "request_stop", None)
        if callable(request_stop):
            request_stop()
        wait = getattr(controller, "wait", None)
        if callable(wait):
            wait(self._stop_wait_ms)
        self._append_log("INFO", f"{label.capitalize()} stop requested")

    def _append_log(self, severity: str, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        self.log_output.appendPlainText(f"[{timestamp}] {severity}: {message}")


def _decode_grayscale(image_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("Could not decode image bytes for background capture.")
    if decoded.ndim == 2:
        return np.ascontiguousarray(decoded.astype(np.uint8, copy=False))
    if decoded.ndim == 3 and int(decoded.shape[2]) == 4:
        return cv2.cvtColor(decoded, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)


def _overlay_from_roi_metadata(roi_metadata: object) -> FramePreviewOverlay | None:
    if roi_metadata is None:
        return None
    extras = _mapping_payload(_payload_value(roi_metadata, "extras"))
    return FramePreviewOverlay(
        source_image_wh_px=_size_tuple(_payload_value(roi_metadata, "source_image_wh_px")),
        bbox_xyxy_px=_xyxy_tuple(_payload_value(roi_metadata, "bbox_xyxy_px")),
        center_xy_px=_xy_tuple(_payload_value(roi_metadata, "center_xy_px")),
        roi_bounds_xyxy_px=_first_xyxy(
            extras,
            contracts.PREPROCESSING_METADATA_ROI_REQUESTED_XYXY_PX,
            contracts.PREPROCESSING_METADATA_ROI_REQUEST_XYXY_PX,
            contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX,
        ),
        label="Pipeline ROI / bbox",
    )


def _overlay_from_metadata(
    metadata: Mapping[str, object],
    locator_result: object,
) -> FramePreviewOverlay | None:
    source_size = _size_tuple(metadata.get(contracts.PREPROCESSING_METADATA_SOURCE_IMAGE_WH_PX))
    if source_size is None:
        source_w = _optional_int(metadata.get(contracts.PREPROCESSING_METADATA_SOURCE_IMAGE_WIDTH_PX))
        source_h = _optional_int(metadata.get(contracts.PREPROCESSING_METADATA_SOURCE_IMAGE_HEIGHT_PX))
        source_size = (source_w, source_h) if source_w is not None and source_h is not None else None
    bbox = _xyxy_tuple(metadata.get(contracts.PREPROCESSING_METADATA_ROI_LOCATOR_BOUNDS_XYXY_PX))
    if bbox is None:
        bbox = _xyxy_tuple(_payload_value(locator_result, "bbox_xyxy_px"))
    center = _xy_tuple(metadata.get(contracts.PREPROCESSING_METADATA_PREDICTED_ROI_CENTER_XY_PX))
    roi_bounds = _first_xyxy(
        metadata,
        contracts.PREPROCESSING_METADATA_ROI_REQUESTED_XYXY_PX,
        contracts.PREPROCESSING_METADATA_ROI_REQUEST_XYXY_PX,
        contracts.PREPROCESSING_METADATA_ROI_SOURCE_XYXY_PX,
    )
    if bbox is None and center is None and roi_bounds is None:
        return None
    return FramePreviewOverlay(
        source_image_wh_px=source_size,
        bbox_xyxy_px=bbox,
        center_xy_px=center,
        roi_bounds_xyxy_px=roi_bounds,
        label="Locator ROI / bbox",
    )


def _payload_value(payload: object, name: str) -> object | None:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def _mapping_payload(payload: object | None) -> Mapping[str, object]:
    if isinstance(payload, Mapping):
        return payload
    to_dict = getattr(payload, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    return {}


def _sequence_payload(payload: object | None) -> tuple[object, ...]:
    if payload is None:
        return ()
    if isinstance(payload, str):
        return (payload,)
    if isinstance(payload, tuple):
        return payload
    if isinstance(payload, list):
        return tuple(payload)
    return ()


def _path_or_none(value: object | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def _enum_text(value: object | None) -> str | None:
    raw = getattr(value, "value", value)
    return str(raw) if raw is not None else None


def _text(value: object | None, *, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    return text if text else default


def _format_value(value: object | None, unit: str, precision: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{precision}f} {unit}"


def _format_optional_float(value: object | None, precision: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return f"{number:.{precision}f}"


def _optional_bool(value: object | None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "accepted"}:
            return True
        if text in {"false", "no", "0", "rejected"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _first_present(*values: object | None) -> object | None:
    for value in values:
        if value is not None:
            return value
    return None


def _first_xyxy(
    payload: Mapping[str, object],
    *keys: str,
) -> tuple[float, float, float, float] | None:
    for key in keys:
        parsed = _xyxy_tuple(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _xy_tuple(value: object | None) -> tuple[float, float] | None:
    parsed = _float_tuple(value, width=2)
    if parsed is None:
        return None
    return parsed[0], parsed[1]


def _xyxy_tuple(value: object | None) -> tuple[float, float, float, float] | None:
    parsed = _float_tuple(value, width=4)
    if parsed is None:
        return None
    return parsed[0], parsed[1], parsed[2], parsed[3]


def _size_tuple(value: object | None) -> tuple[int, int] | None:
    parsed = _float_tuple(value, width=2)
    if parsed is None:
        return None
    return int(parsed[0]), int(parsed[1])


def _float_tuple(value: object | None, *, width: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _issue_text(issue: object) -> str:
    message = _text(_payload_value(issue, "message"), default="")
    issue_type = _text(
        _payload_value(issue, "error_type") or _payload_value(issue, "warning_type"),
        default="issue",
    )
    return f"{issue_type}: {message}" if message else issue_type


__all__ = ["LiveInferenceMainWindow"]
