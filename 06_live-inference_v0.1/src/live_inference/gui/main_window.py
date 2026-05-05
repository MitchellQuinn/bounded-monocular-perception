"""Minimal PySide6 shell for live inference worker control and status."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class LiveInferenceMainWindow(QMainWindow):
    """Small GUI shell around already-composed worker controllers."""

    def __init__(
        self,
        *,
        camera_controller: object,
        inference_controller: object,
        stop_wait_ms: int = 1000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_controller = camera_controller
        self.inference_controller = inference_controller
        self._stop_wait_ms = int(stop_wait_ms)
        self._frames_written_count = 0
        self._frames_processed_count = 0
        self._duplicate_skipped_count = 0
        self._last_skip_log_key: tuple[str, str] | None = None
        self._repeated_skip_count = 0
        self._preview_source_pixmap: QPixmap | None = None

        self.setWindowTitle("Live Inference")
        self._build_ui()
        self._connect_buttons()
        self._connect_worker_signals()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QVBoxLayout(central)

        controls_layout = QHBoxLayout()
        self.start_camera_button = QPushButton("Start Camera")
        self.start_camera_button.setObjectName("start_camera_button")
        self.stop_camera_button = QPushButton("Stop Camera")
        self.stop_camera_button.setObjectName("stop_camera_button")
        self.start_inference_button = QPushButton("Start Inference")
        self.start_inference_button.setObjectName("start_inference_button")
        self.stop_inference_button = QPushButton("Stop Inference")
        self.stop_inference_button.setObjectName("stop_inference_button")
        self.stop_all_button = QPushButton("Stop All")
        self.stop_all_button.setObjectName("stop_all_button")

        for button in (
            self.start_camera_button,
            self.stop_camera_button,
            self.start_inference_button,
            self.stop_inference_button,
            self.stop_all_button,
        ):
            controls_layout.addWidget(button)
        controls_layout.addStretch(1)
        root_layout.addLayout(controls_layout)

        self.frame_preview_label = QLabel("No frame yet")
        self.frame_preview_label.setObjectName("frame_preview_label")
        self.frame_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_preview_label.setMinimumSize(320, 240)
        self.frame_preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        root_layout.addWidget(self.frame_preview_label, stretch=2)

        status_grid = QGridLayout()
        status_grid.setColumnStretch(1, 1)
        self.camera_status_value = self._add_readout(
            status_grid,
            row=0,
            label="Camera status",
            object_name="camera_status_value",
        )
        self.inference_status_value = self._add_readout(
            status_grid,
            row=1,
            label="Inference status",
            object_name="inference_status_value",
        )
        self.frames_written_value = self._add_readout(
            status_grid,
            row=2,
            label="Frames written",
            object_name="frames_written_value",
            initial_text="0",
        )
        self.frames_processed_value = self._add_readout(
            status_grid,
            row=3,
            label="Frames processed",
            object_name="frames_processed_value",
            initial_text="0",
        )
        self.duplicate_skipped_value = self._add_readout(
            status_grid,
            row=4,
            label="Duplicate skipped",
            object_name="duplicate_skipped_value",
            initial_text="0",
        )
        self.distance_value = self._add_readout(
            status_grid,
            row=5,
            label="Predicted distance",
            object_name="distance_value",
        )
        self.yaw_value = self._add_readout(
            status_grid,
            row=6,
            label="Predicted yaw",
            object_name="yaw_value",
        )
        self.inference_time_value = self._add_readout(
            status_grid,
            row=7,
            label="Inference time",
            object_name="inference_time_value",
        )
        self.preprocessing_time_value = self._add_readout(
            status_grid,
            row=8,
            label="Preprocessing time",
            object_name="preprocessing_time_value",
        )
        self.total_time_value = self._add_readout(
            status_grid,
            row=9,
            label="Total time",
            object_name="total_time_value",
        )
        root_layout.addLayout(status_grid)

        self.log_panel = QPlainTextEdit()
        self.log_panel.setObjectName("log_panel")
        self.log_panel.setReadOnly(True)
        self.log_panel.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        root_layout.addWidget(self.log_panel, stretch=1)

        self.setCentralWidget(central)

    def _add_readout(
        self,
        layout: QGridLayout,
        *,
        row: int,
        label: str,
        object_name: str,
        initial_text: str = "n/a",
    ) -> QLabel:
        name_label = QLabel(f"{label}:")
        name_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_label = QLabel(initial_text)
        value_label.setObjectName(object_name)
        value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(name_label, row, 0)
        layout.addWidget(value_label, row, 1)
        return value_label

    def _connect_buttons(self) -> None:
        self.start_camera_button.clicked.connect(self.start_camera)
        self.stop_camera_button.clicked.connect(self.stop_camera)
        self.start_inference_button.clicked.connect(self.start_inference)
        self.stop_inference_button.clicked.connect(self.stop_inference)
        self.stop_all_button.clicked.connect(self.stop_all)

    def _connect_worker_signals(self) -> None:
        self._connect_signals(
            self.camera_controller,
            {
                "status_changed": self._on_camera_status_changed,
                "frame_written": self._on_camera_frame_written,
                "warning_occurred": self._on_warning_occurred,
                "error_occurred": self._on_error_occurred,
                "lifecycle_event": self._on_lifecycle_event,
            },
        )
        self._connect_signals(
            self.inference_controller,
            {
                "status_changed": self._on_inference_status_changed,
                "result_ready": self._on_inference_result_ready,
                "frame_skipped": self._on_inference_frame_skipped,
                "warning_occurred": self._on_warning_occurred,
                "error_occurred": self._on_error_occurred,
                "lifecycle_event": self._on_lifecycle_event,
            },
        )

    def _connect_signals(
        self,
        controller: object,
        handlers: Mapping[str, object],
    ) -> None:
        signals = getattr(controller, "signals", None)
        if signals is None:
            return
        for signal_name, handler in handlers.items():
            signal = getattr(signals, signal_name, None)
            connect = getattr(signal, "connect", None)
            if callable(connect):
                connect(handler)

    def start_camera(self) -> None:
        self._call_controller(self.camera_controller, "start", "Camera start")

    def stop_camera(self) -> None:
        self._call_controller(self.camera_controller, "request_stop", "Camera stop")

    def start_inference(self) -> None:
        self._call_controller(self.inference_controller, "start", "Inference start")

    def stop_inference(self) -> None:
        self._call_controller(
            self.inference_controller,
            "request_stop",
            "Inference stop",
        )

    def stop_all(self) -> None:
        self._call_controller(self.camera_controller, "request_stop", "Camera stop")
        self._call_controller(
            self.inference_controller,
            "request_stop",
            "Inference stop",
        )

    def _call_controller(
        self,
        controller: object,
        method_name: str,
        action_label: str,
    ) -> None:
        method = getattr(controller, method_name, None)
        if not callable(method):
            self._append_log("ERROR", f"{action_label} unavailable.")
            return
        try:
            method()
        except Exception as exc:
            self._append_log("ERROR", f"{action_label} failed: {exc}")

    def _on_camera_status_changed(self, status: object) -> None:
        self.camera_status_value.setText(_status_display_text(status))
        counters = _payload_value(status, "counters")
        frames_written = _counter_int(counters, "frames_written")
        if frames_written is not None:
            self._frames_written_count = frames_written
            self.frames_written_value.setText(str(frames_written))

    def _on_camera_frame_written(self, frame: object) -> None:
        self._frames_written_count += 1
        self.frames_written_value.setText(str(self._frames_written_count))
        path = _payload_value(frame, "image_path")
        self._display_frame_preview(path)
        frame_hash = _hash_value(_payload_value(frame, "frame_hash"))
        timestamp = (
            _payload_value(frame, "completed_at_utc")
            or _payload_value(_payload_value(frame, "metadata"), "written_at_utc")
            or "n/a"
        )
        self._append_log(
            "INFO",
            "Frame written: "
            f"path={_text(path, default='n/a')} "
            f"hash={_text(frame_hash, default='n/a')} "
            f"time={_text(timestamp, default='n/a')}",
        )

    def _display_frame_preview(self, path: object | None) -> None:
        if path is None:
            self._append_log("WARNING", "Frame preview unavailable: missing image path.")
            return

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._append_log(
                "WARNING",
                f"Frame preview unavailable: could not load image path={path}",
            )
            return

        self._preview_source_pixmap = pixmap
        self._set_scaled_frame_preview()

    def _set_scaled_frame_preview(self) -> None:
        if self._preview_source_pixmap is None or self._preview_source_pixmap.isNull():
            return

        target_size = self.frame_preview_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            target_size = self.frame_preview_label.minimumSize()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled_pixmap = self._preview_source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.frame_preview_label.setPixmap(scaled_pixmap)

    def _on_inference_status_changed(self, status: object) -> None:
        self.inference_status_value.setText(_status_display_text(status))
        counters = _payload_value(status, "counters")
        frames_processed = _counter_int(counters, "frames_processed")
        if frames_processed is not None:
            self._frames_processed_count = frames_processed
            self.frames_processed_value.setText(str(frames_processed))

        duplicate_skipped = _counter_int(counters, "frames_skipped_duplicate")
        if duplicate_skipped is not None:
            self._duplicate_skipped_count = duplicate_skipped
            self.duplicate_skipped_value.setText(str(duplicate_skipped))

        self._set_optional_time_from_counter(
            self.inference_time_value,
            counters,
            "last_inference_time_ms",
        )
        self._set_optional_time_from_counter(
            self.preprocessing_time_value,
            counters,
            "last_preprocessing_time_ms",
        )
        self._set_optional_time_from_counter(
            self.total_time_value,
            counters,
            "last_total_time_ms",
        )

    def _on_inference_result_ready(self, result: object) -> None:
        self._frames_processed_count += 1
        self.frames_processed_value.setText(str(self._frames_processed_count))
        self.distance_value.setText(
            _format_value(_payload_value(result, "predicted_distance_m"), "m", precision=3)
        )
        self.yaw_value.setText(
            _format_value(_payload_value(result, "predicted_yaw_deg"), "deg", precision=2)
        )
        self.inference_time_value.setText(
            _format_value(_payload_value(result, "inference_time_ms"), "ms", precision=1)
        )
        self.preprocessing_time_value.setText(
            _format_value(
                _payload_value(result, "preprocessing_time_ms"),
                "ms",
                precision=1,
            )
        )
        self.total_time_value.setText(
            _format_value(_payload_value(result, "total_time_ms"), "ms", precision=1)
        )

    def _on_inference_frame_skipped(self, skipped: object) -> None:
        reason = _enum_text(_payload_value(skipped, "reason")) or "unknown"
        message = _text(_payload_value(skipped, "message"), default="")
        if reason == "duplicate_hash":
            self._duplicate_skipped_count += 1
            self.duplicate_skipped_value.setText(str(self._duplicate_skipped_count))

        key = (reason, message)
        if key == self._last_skip_log_key:
            self._repeated_skip_count += 1
            return

        self._flush_repeated_skip_summary()
        self._last_skip_log_key = key
        self._append_log(
            "DEBUG",
            f"Frame skipped: reason={reason} message={message or 'n/a'}",
        )

    def _on_warning_occurred(self, warning: object) -> None:
        self._flush_repeated_skip_summary()
        self._append_log("WARNING", _issue_display_text(warning))

    def _on_error_occurred(self, error: object) -> None:
        self._flush_repeated_skip_summary()
        self._append_log("ERROR", _issue_display_text(error))

    def _on_lifecycle_event(self, event: object) -> None:
        state = _enum_text(_payload_value(event, "state")) or "unknown"
        event_type = _enum_text(_payload_value(event, "event_type")) or "lifecycle"
        message = _text(_payload_value(event, "message"), default="")
        self._append_log(
            "INFO",
            f"Lifecycle: event={event_type} state={state} message={message or 'n/a'}",
        )

    def _set_optional_time_from_counter(
        self,
        label: QLabel,
        counters: object,
        counter_name: str,
    ) -> None:
        value = _counter_value(counters, counter_name)
        if value is not None:
            label.setText(_format_value(value, "ms", precision=1))

    def _flush_repeated_skip_summary(self) -> None:
        if self._repeated_skip_count <= 0:
            return
        self._append_log(
            "DEBUG",
            f"Previous frame skip repeated {self._repeated_skip_count} time(s).",
        )
        self._repeated_skip_count = 0

    def _append_log(self, severity: str, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00",
            "Z",
        )
        self.log_panel.appendPlainText(f"{timestamp} [{severity}] {message}")

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        self.stop_all()
        self._wait_for_controller(self.camera_controller)
        self._wait_for_controller(self.inference_controller)
        super().closeEvent(event)

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if hasattr(self, "frame_preview_label"):
            self._set_scaled_frame_preview()

    def _wait_for_controller(self, controller: object) -> None:
        wait = getattr(controller, "wait", None)
        if not callable(wait):
            return
        try:
            wait(self._stop_wait_ms)
        except Exception as exc:
            self._append_log("WARNING", f"Worker wait failed: {exc}")


def _status_display_text(status: object) -> str:
    state = _enum_text(_payload_value(status, "state"))
    message = _text(_payload_value(status, "message"), default="")
    if state and message:
        return f"{state} - {message}"
    return state or message or "n/a"


def _issue_display_text(issue: object) -> str:
    issue_type = (
        _payload_value(issue, "warning_type")
        or _payload_value(issue, "error_type")
        or type(issue).__name__
    )
    message = _text(_payload_value(issue, "message"), default="")
    return f"{_text(issue_type, default='issue')}: {message or 'n/a'}"


def _counter_int(counters: object, name: str) -> int | None:
    value = _counter_value(counters, name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _counter_value(counters: object, name: str) -> object | None:
    if counters is None:
        return None
    if isinstance(counters, Mapping):
        return counters.get(name)
    to_dict = getattr(counters, "to_dict", None)
    if callable(to_dict):
        return _counter_value(to_dict(), name)
    return getattr(counters, name, None)


def _payload_value(payload: object, name: str) -> object | None:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


def _hash_value(frame_hash: object) -> object | None:
    if frame_hash is None:
        return None
    if isinstance(frame_hash, Mapping):
        return frame_hash.get("value")
    return getattr(frame_hash, "value", frame_hash)


def _format_value(value: object, unit: str, *, precision: int) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _text(value, default="n/a")
    return f"{number:.{precision}f} {unit}"


def _enum_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _text(value: object | None, *, default: str) -> str:
    if value is None:
        return default
    return str(value)


__all__ = ["LiveInferenceMainWindow"]
