"""Minimal PySide6 shell for live inference worker control and status."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .frame_preview_widget import FramePreviewOverlay, FramePreviewWidget
from live_inference.masking import FrameMaskState


class LiveInferenceMainWindow(QMainWindow):
    """Small GUI shell around already-composed worker controllers."""

    def __init__(
        self,
        *,
        camera_controller: object,
        inference_controller: object,
        mask_state: FrameMaskState | None = None,
        stop_wait_ms: int = 1000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_controller = camera_controller
        self.inference_controller = inference_controller
        self.mask_state = mask_state or FrameMaskState()
        self._stop_wait_ms = int(stop_wait_ms)
        self._frames_written_count = 0
        self._frames_processed_count = 0
        self._duplicate_skipped_count = 0
        self._frame_written_summary_interval = 100
        self._last_skip_log_key: tuple[str, str] | None = None
        self._repeated_skip_count = 0
        self._last_result_warning_logged: str | None = None

        self.setWindowTitle("Live Inference")
        self._build_ui()
        self._connect_buttons()
        self._connect_worker_signals()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)

        preview_layout = QVBoxLayout()
        self.frame_preview_widget = FramePreviewWidget()
        self.frame_preview_widget.setObjectName("frame_preview_widget")
        self.frame_preview_widget.setMinimumSize(320, 240)
        self.frame_preview_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.frame_preview_widget.set_committed_mask_snapshot(self.mask_state.get_snapshot())
        preview_layout.addWidget(self.frame_preview_widget, stretch=1)
        root_layout.addLayout(preview_layout, stretch=3)

        self.right_control_panel = QWidget()
        self.right_control_panel.setObjectName("right_control_panel")
        panel_layout = QVBoxLayout(self.right_control_panel)

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
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
        panel_layout.addWidget(controls_group)

        mask_group = QGroupBox("Mask")
        mask_layout = QGridLayout(mask_group)
        self.draw_mask_button = QPushButton("Draw Mask")
        self.draw_mask_button.setObjectName("draw_mask_button")
        self.apply_mask_button = QPushButton("Stop Mask / Apply Mask")
        self.apply_mask_button.setObjectName("apply_mask_button")
        self.erase_mask_button = QPushButton("Erase Mask")
        self.erase_mask_button.setObjectName("erase_mask_button")
        self.apply_erase_button = QPushButton("Apply Erase")
        self.apply_erase_button.setObjectName("apply_erase_button")
        self.clear_mask_button = QPushButton("Clear Mask")
        self.clear_mask_button.setObjectName("clear_mask_button")
        self.mask_brush_diameter_input = QSpinBox()
        self.mask_brush_diameter_input.setObjectName("mask_brush_diameter_input")
        self.mask_brush_diameter_input.setRange(1, 1000)
        self.mask_brush_diameter_input.setSingleStep(5)
        self.mask_brush_diameter_input.setValue(100)
        self.mask_brush_diameter_input.setSuffix(" px")
        self.mask_fill_white_checkbox = QCheckBox("White")
        self.mask_fill_white_checkbox.setObjectName("mask_fill_white_checkbox")
        self.mask_fill_white_checkbox.setChecked(True)
        self.mask_fill_value_label = QLabel("Mask fill: White (255)")
        self.mask_fill_value_label.setObjectName("mask_fill_value_label")

        mask_layout.addWidget(self.draw_mask_button, 0, 0)
        mask_layout.addWidget(self.apply_mask_button, 0, 1)
        mask_layout.addWidget(self.erase_mask_button, 1, 0)
        mask_layout.addWidget(self.apply_erase_button, 1, 1)
        mask_layout.addWidget(self.clear_mask_button, 2, 0, 1, 2)
        mask_layout.addWidget(QLabel("Brush diameter px:"), 3, 0)
        mask_layout.addWidget(self.mask_brush_diameter_input, 3, 1)
        mask_layout.addWidget(self.mask_fill_white_checkbox, 4, 0)
        mask_layout.addWidget(self.mask_fill_value_label, 4, 1)
        panel_layout.addWidget(mask_group)

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
        self.debug_artifacts_value = self._add_readout(
            status_grid,
            row=10,
            label="Debug artifacts",
            object_name="debug_artifacts_value",
        )
        self.debug_artifacts_value.setWordWrap(True)
        panel_layout.addLayout(status_grid)

        self.log_panel = QPlainTextEdit()
        self.log_panel.setObjectName("log_panel")
        self.log_panel.setReadOnly(True)
        self.log_panel.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        panel_layout.addWidget(self.log_panel, stretch=1)
        root_layout.addWidget(self.right_control_panel, stretch=1)

        self.setCentralWidget(central)
        self._set_mask_button_state(None)

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
        self.draw_mask_button.clicked.connect(self.start_draw_mask)
        self.apply_mask_button.clicked.connect(self.apply_draw_mask)
        self.erase_mask_button.clicked.connect(self.start_erase_mask)
        self.apply_erase_button.clicked.connect(self.apply_erase_mask)
        self.clear_mask_button.clicked.connect(self.clear_mask)
        self.mask_brush_diameter_input.valueChanged.connect(self._on_brush_diameter_changed)
        self.mask_fill_white_checkbox.toggled.connect(self._on_mask_fill_toggled)

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
                "debug_image_ready": self._on_debug_image_ready,
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

    def start_draw_mask(self) -> None:
        self.frame_preview_widget.set_brush_diameter_px(
            self.mask_brush_diameter_input.value()
        )
        self.frame_preview_widget.set_committed_mask_snapshot(self.mask_state.get_snapshot())
        self.frame_preview_widget.begin_mask_edit("draw")
        self._set_mask_button_state("draw")

    def apply_draw_mask(self) -> None:
        self._commit_preview_mask("mask")

    def start_erase_mask(self) -> None:
        self.frame_preview_widget.set_brush_diameter_px(
            self.mask_brush_diameter_input.value()
        )
        self.frame_preview_widget.set_committed_mask_snapshot(self.mask_state.get_snapshot())
        self.frame_preview_widget.begin_mask_edit("erase")
        self._set_mask_button_state("erase")

    def apply_erase_mask(self) -> None:
        self._commit_preview_mask("erase")

    def clear_mask(self) -> None:
        revision = self.mask_state.clear()
        self.frame_preview_widget.clear_masks()
        self._set_mask_button_state(None)
        self._append_log("INFO", f"Mask cleared: revision={revision}.")

    def _commit_preview_mask(self, edit_label: str) -> None:
        result = self.frame_preview_widget.finish_mask_edit(commit=True)
        self._set_mask_button_state(None)
        if result is None:
            self._append_log(
                "WARNING",
                f"Could not apply {edit_label}: no preview frame is loaded.",
            )
            return
        revision = self.mask_state.commit_mask(
            result.mask,
            width_px=result.width_px,
            height_px=result.height_px,
            fill_value=self._current_mask_fill_value(),
        )
        snapshot = self.mask_state.get_snapshot()
        self.frame_preview_widget.set_committed_mask_snapshot(snapshot)
        self._append_log(
            "INFO",
            "Mask committed: "
            f"revision={revision} "
            f"size={result.width_px}x{result.height_px} "
            f"pixels={snapshot.pixel_count} "
            f"fill={snapshot.fill_value}.",
        )

    def _on_brush_diameter_changed(self, value: int) -> None:
        self.frame_preview_widget.set_brush_diameter_px(int(value))

    def _on_mask_fill_toggled(self, checked: bool) -> None:
        fill_value = 255 if checked else 0
        fill_name = "White" if checked else "Black"
        self.mask_fill_value_label.setText(f"Mask fill: {fill_name} ({fill_value})")
        previous = self.mask_state.get_snapshot()
        revision = self.mask_state.set_fill_value(fill_value)
        if previous.enabled:
            self.frame_preview_widget.set_committed_mask_snapshot(
                self.mask_state.get_snapshot()
            )
        self._append_log(
            "INFO",
            f"Mask fill value changed: {fill_name} ({fill_value}), revision={revision}.",
        )

    def _current_mask_fill_value(self) -> int:
        return 255 if self.mask_fill_white_checkbox.isChecked() else 0

    def _set_mask_button_state(self, edit_mode: str | None) -> None:
        self.draw_mask_button.setEnabled(edit_mode != "draw")
        self.apply_mask_button.setEnabled(edit_mode == "draw")
        self.erase_mask_button.setEnabled(edit_mode != "erase")
        self.apply_erase_button.setEnabled(edit_mode == "erase")
        self.clear_mask_button.setEnabled(True)

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
        if (
            self._frame_written_summary_interval > 0
            and self._frames_written_count % self._frame_written_summary_interval == 0
        ):
            self._append_log(
                "INFO",
                f"Frames written: {self._frames_written_count} latest={_text(path, default='n/a')}",
            )

    def _display_frame_preview(self, path: object | None) -> None:
        if path is None:
            self._append_log("WARNING", "Frame preview unavailable: missing image path.")
            return

        if not self.frame_preview_widget.load_image(str(path)):
            self._append_log(
                "WARNING",
                f"Frame preview unavailable: could not load image path={path}",
            )
            return

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
        self._update_preview_overlay(result)
        self._update_debug_artifact_paths(result)
        self._log_result_warnings(result)

    def _on_debug_image_ready(self, image: object) -> None:
        image_kind = _text(_payload_value(image, "image_kind"), default="debug")
        path = _text(_payload_value(image, "path"), default="n/a")
        self._append_log("INFO", f"Debug artifact ready: kind={image_kind} path={path}")

    def _update_preview_overlay(self, result: object) -> None:
        overlay = _overlay_from_result(result)
        if overlay is None:
            self.frame_preview_widget.set_overlay(None)
            return
        if not self.frame_preview_widget.overlay_source_size_matches(
            overlay.source_image_wh_px
        ):
            self.frame_preview_widget.set_overlay(None)
            self._append_log(
                "WARNING",
                "Skipping ROI overlay: result source size "
                f"{overlay.source_image_wh_px} does not match preview frame size "
                f"{self.frame_preview_widget.source_image_size()}.",
            )
            return
        self.frame_preview_widget.set_overlay(overlay)

    def _update_debug_artifact_paths(self, result: object) -> None:
        debug_paths = _mapping_payload(_payload_value(result, "debug_paths"))
        if not debug_paths:
            self.debug_artifacts_value.setText("n/a")
            return

        summary = " | ".join(
            f"{key}: {_text(path, default='n/a')}"
            for key, path in sorted(debug_paths.items())
        )
        self.debug_artifacts_value.setText(summary)
        self._append_log("INFO", f"Debug artifacts: {summary}")

    def _log_result_warnings(self, result: object) -> None:
        warnings = _sequence_payload(_payload_value(result, "warnings"))
        for warning in warnings:
            text = str(warning)
            if "frame mask skipped" not in text:
                continue
            if text == self._last_result_warning_logged:
                return
            self._last_result_warning_logged = text
            self._append_log("WARNING", text)

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
        if hasattr(self, "frame_preview_widget"):
            self.frame_preview_widget.update()

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


def _overlay_from_result(result: object) -> FramePreviewOverlay | None:
    roi_metadata = _payload_value(result, "roi_metadata")
    if roi_metadata is None:
        return None

    source_size = _size_tuple(_payload_value(roi_metadata, "source_image_wh_px"))
    bbox = _xyxy_tuple(_payload_value(roi_metadata, "bbox_xyxy_px"))
    center = _xy_tuple(_payload_value(roi_metadata, "center_xy_px"))
    extras = _mapping_payload(_payload_value(roi_metadata, "extras"))
    roi_bounds = _first_xyxy(
        extras,
        "roi_locator_bounds_xyxy_px",
        "roi_source_xyxy_px",
        "roi_request_xyxy_px",
    )
    if bbox is None and center is None and roi_bounds is None:
        return None
    return FramePreviewOverlay(
        source_image_wh_px=source_size,
        bbox_xyxy_px=bbox,
        center_xy_px=center,
        roi_bounds_xyxy_px=roi_bounds,
        label="Pipeline ROI / bbox",
    )


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


def _first_xyxy(
    payload: Mapping[str, object],
    *keys: str,
) -> tuple[float, float, float, float] | None:
    for key in keys:
        value = _xyxy_tuple(payload.get(key))
        if value is not None:
            return value
    return None


def _xyxy_tuple(value: object | None) -> tuple[float, float, float, float] | None:
    parsed = _float_tuple(value, width=4)
    if parsed is None:
        return None
    return parsed[0], parsed[1], parsed[2], parsed[3]


def _xy_tuple(value: object | None) -> tuple[float, float] | None:
    parsed = _float_tuple(value, width=2)
    if parsed is None:
        return None
    return parsed[0], parsed[1]


def _size_tuple(value: object | None) -> tuple[int, int] | None:
    parsed = _float_tuple(value, width=2)
    if parsed is None:
        return None
    width, height = int(parsed[0]), int(parsed[1])
    if width <= 0 or height <= 0:
        return None
    return width, height


def _float_tuple(value: object | None, *, width: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != int(width):
        return None
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None


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
