"""Minimal PySide6 shell for live inference worker control and status."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import interfaces.contracts as contracts

from live_inference.frame_handoff import compute_frame_hash
from .frame_preview_widget import (
    FramePreviewHeatmapOverlay,
    FramePreviewOverlay,
    FramePreviewWidget,
)
from live_inference.masking import (
    DEFAULT_BACKGROUND_THRESHOLD,
    BackgroundState,
    FrameMaskState,
)
from live_inference.preprocessing import StageTransformPolicyState


@dataclass(frozen=True)
class _CapturedSingleFrame:
    image_bytes: bytes
    frame_hash: contracts.FrameHash
    source_path: Path | None
    frame_metadata: contracts.FrameMetadata | None


class LiveInferenceMainWindow(QMainWindow):
    """Small GUI shell around already-composed worker controllers."""

    def __init__(
        self,
        *,
        camera_controller: object,
        inference_controller: object,
        frame_reader: object | None = None,
        single_frame_runner: object | None = None,
        trace_output_dir: Path | str | None = None,
        mask_state: FrameMaskState | None = None,
        background_state: BackgroundState | None = None,
        stage_policy_state: StageTransformPolicyState | None = None,
        stop_wait_ms: int = 1000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.camera_controller = camera_controller
        self.inference_controller = inference_controller
        self.frame_reader = frame_reader
        self.single_frame_runner = single_frame_runner
        runner_trace_dir = getattr(single_frame_runner, "trace_output_dir", None)
        self.trace_output_dir = (
            Path(trace_output_dir)
            if trace_output_dir is not None
            else Path(runner_trace_dir) if runner_trace_dir is not None else None
        )
        self.mask_state = mask_state or FrameMaskState()
        self.background_state = background_state or BackgroundState()
        self.stage_policy_state = stage_policy_state or StageTransformPolicyState()
        self._stop_wait_ms = int(stop_wait_ms)
        self._frames_written_count = 0
        self._frames_processed_count = 0
        self._duplicate_skipped_count = 0
        self._frame_written_summary_interval = 100
        self._preview_update_interval_seconds = 1.0 / 15.0
        self._background_preview_max_size = (480, 300)
        self._last_preview_update_seconds = 0.0
        self._last_skip_log_key: tuple[str, str] | None = None
        self._repeated_skip_count = 0
        self._last_result_warning_logged: str | None = None
        self._last_inference_state_text = ""
        self._inference_start_requested = False
        self._preview_background_revision: int | None = None
        self._captured_single_frame: _CapturedSingleFrame | None = None

        self.setWindowTitle("Live Inference")
        self._build_ui()
        self._connect_buttons()
        self._connect_worker_signals()

    def _build_ui(self) -> None:
        central = QWidget(self)
        root_layout = QHBoxLayout(central)

        output_layout = QVBoxLayout()

        camera_output_group = QGroupBox("Camera Output Area")
        camera_output_group.setObjectName("camera_output_group")
        camera_output_layout = QHBoxLayout(camera_output_group)

        full_frame_group = QGroupBox("1. Full frame preview")
        full_frame_group.setObjectName("full_frame_preview_group")
        full_frame_layout = QVBoxLayout(full_frame_group)
        self.frame_preview_widget = FramePreviewWidget()
        self.frame_preview_widget.setObjectName("frame_preview_widget")
        self.frame_preview_widget.setMinimumSize(320, 240)
        self.frame_preview_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.frame_preview_widget.set_committed_mask_snapshot(self.mask_state.get_snapshot())
        self._apply_background_snapshot_to_preview(self.background_state.get_snapshot())
        full_frame_layout.addWidget(self.frame_preview_widget, stretch=1)
        camera_output_layout.addWidget(full_frame_group, stretch=3)

        roi_crop_group = QGroupBox("2. ROI crop preview")
        roi_crop_group.setObjectName("roi_crop_preview_group")
        roi_crop_layout = QVBoxLayout(roi_crop_group)
        self.roi_crop_preview_widget = FramePreviewWidget()
        self.roi_crop_preview_widget.setObjectName("roi_crop_preview_widget")
        self.roi_crop_preview_widget.set_placeholder_text("No ROI crop yet")
        self.roi_crop_preview_widget.setMinimumSize(240, 240)
        self.roi_crop_preview_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        roi_crop_layout.addWidget(self.roi_crop_preview_widget, stretch=1)
        camera_output_layout.addWidget(roi_crop_group, stretch=2)
        output_layout.addWidget(camera_output_group, stretch=3)

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

        single_frame_group = QGroupBox("Single Frame")
        single_frame_group.setObjectName("single_frame_group")
        single_frame_layout = QGridLayout(single_frame_group)
        single_frame_layout.setColumnStretch(1, 1)
        self.capture_frame_button = QPushButton("Capture Frame")
        self.capture_frame_button.setObjectName("capture_frame_button")
        self.run_single_inference_button = QPushButton("Run Single Inference")
        self.run_single_inference_button.setObjectName("run_single_inference_button")
        self.record_trace_checkbox = QCheckBox("Record Trace")
        self.record_trace_checkbox.setObjectName("record_trace_checkbox")
        self.record_trace_checkbox.setChecked(False)
        self.trace_output_dir_value = QLabel(_text(self.trace_output_dir, default="n/a"))
        self.trace_output_dir_value.setObjectName("trace_output_dir_value")
        self.trace_output_dir_value.setWordWrap(True)
        self.last_captured_frame_hash_value = QLabel("n/a")
        self.last_captured_frame_hash_value.setObjectName(
            "last_captured_frame_hash_value"
        )
        self.last_captured_frame_hash_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.last_trace_path_value = QLabel("n/a")
        self.last_trace_path_value.setObjectName("last_trace_path_value")
        self.last_trace_path_value.setWordWrap(True)
        self.last_trace_path_value.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        single_frame_layout.addWidget(self.capture_frame_button, 0, 0, 1, 2)
        single_frame_layout.addWidget(self.run_single_inference_button, 1, 0, 1, 2)
        single_frame_layout.addWidget(self.record_trace_checkbox, 2, 0, 1, 2)
        single_frame_layout.addWidget(QLabel("Trace dir:"), 3, 0)
        single_frame_layout.addWidget(self.trace_output_dir_value, 3, 1)
        single_frame_layout.addWidget(QLabel("Captured hash:"), 4, 0)
        single_frame_layout.addWidget(self.last_captured_frame_hash_value, 4, 1)
        single_frame_layout.addWidget(QLabel("Last trace:"), 5, 0)
        single_frame_layout.addWidget(self.last_trace_path_value, 5, 1)
        panel_layout.addWidget(single_frame_group)

        self.control_tabs = QTabWidget()
        self.control_tabs.setObjectName("control_tabs")

        mask_tab = QWidget()
        mask_layout = QGridLayout(mask_tab)
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
        stage_policy = self.stage_policy_state.get_snapshot()
        self.apply_manual_mask_to_roi_locator_checkbox = QCheckBox(
            "Apply manual mask to ROI locator"
        )
        self.apply_manual_mask_to_roi_locator_checkbox.setObjectName(
            "apply_manual_mask_to_roi_locator_checkbox"
        )
        self.apply_manual_mask_to_roi_locator_checkbox.setChecked(
            bool(stage_policy.apply_manual_mask_to_roi_locator)
        )
        self.apply_manual_mask_to_model_preprocessing_checkbox = QCheckBox(
            "Apply manual mask to model preprocessing"
        )
        self.apply_manual_mask_to_model_preprocessing_checkbox.setObjectName(
            "apply_manual_mask_to_model_preprocessing_checkbox"
        )
        self.apply_manual_mask_to_model_preprocessing_checkbox.setChecked(
            bool(stage_policy.apply_manual_mask_to_regressor_preprocessing)
        )

        mask_layout.addWidget(self.draw_mask_button, 0, 0)
        mask_layout.addWidget(self.apply_mask_button, 0, 1)
        mask_layout.addWidget(self.erase_mask_button, 1, 0)
        mask_layout.addWidget(self.apply_erase_button, 1, 1)
        mask_layout.addWidget(self.clear_mask_button, 2, 0, 1, 2)
        mask_layout.addWidget(QLabel("Brush diameter px:"), 3, 0)
        mask_layout.addWidget(self.mask_brush_diameter_input, 3, 1)
        mask_layout.addWidget(self.mask_fill_white_checkbox, 4, 0)
        mask_layout.addWidget(self.mask_fill_value_label, 4, 1)
        mask_layout.addWidget(
            self.apply_manual_mask_to_roi_locator_checkbox,
            5,
            0,
            1,
            2,
        )
        mask_layout.addWidget(
            self.apply_manual_mask_to_model_preprocessing_checkbox,
            6,
            0,
            1,
            2,
        )
        self.control_tabs.addTab(mask_tab, "Mask")

        background_tab = QWidget()
        background_layout = QGridLayout(background_tab)
        self.capture_background_button = QPushButton("Capture Background")
        self.capture_background_button.setObjectName("capture_background_button")
        self.enable_background_removal_checkbox = QCheckBox("Enable Background Removal")
        self.enable_background_removal_checkbox.setObjectName(
            "enable_background_removal_checkbox"
        )
        self.clear_background_button = QPushButton("Clear Background")
        self.clear_background_button.setObjectName("clear_background_button")
        self.background_threshold_input = QSpinBox()
        self.background_threshold_input.setObjectName("background_threshold_input")
        self.background_threshold_input.setRange(0, 255)
        self.background_threshold_input.setSingleStep(1)
        self.background_threshold_input.setValue(
            self.background_state.get_snapshot().threshold
            if self.background_state.get_snapshot().threshold is not None
            else DEFAULT_BACKGROUND_THRESHOLD
        )
        self.background_status_label = QLabel("Background: not captured")
        self.background_status_label.setObjectName("background_status_label")
        self.background_status_label.setWordWrap(True)
        self.apply_background_removal_to_roi_locator_checkbox = QCheckBox(
            "Apply background removal to ROI locator"
        )
        self.apply_background_removal_to_roi_locator_checkbox.setObjectName(
            "apply_background_removal_to_roi_locator_checkbox"
        )
        self.apply_background_removal_to_roi_locator_checkbox.setChecked(
            bool(stage_policy.apply_background_removal_to_roi_locator)
        )
        self.apply_background_removal_to_model_preprocessing_checkbox = QCheckBox(
            "Apply background removal to model preprocessing"
        )
        self.apply_background_removal_to_model_preprocessing_checkbox.setObjectName(
            "apply_background_removal_to_model_preprocessing_checkbox"
        )
        self.apply_background_removal_to_model_preprocessing_checkbox.setChecked(
            bool(stage_policy.apply_background_removal_to_regressor_preprocessing)
        )
        self.background_preview_widget = FramePreviewWidget()
        self.background_preview_widget.setObjectName("background_preview_widget")
        self.background_preview_widget.set_placeholder_text("No background preview yet")
        self.background_preview_widget.setMinimumSize(180, 110)
        self.background_preview_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        background_layout.addWidget(self.capture_background_button, 0, 0, 1, 2)
        background_layout.addWidget(self.enable_background_removal_checkbox, 1, 0, 1, 2)
        background_layout.addWidget(QLabel("Threshold:"), 2, 0)
        background_layout.addWidget(self.background_threshold_input, 2, 1)
        background_layout.addWidget(self.clear_background_button, 3, 0, 1, 2)
        background_layout.addWidget(
            self.apply_background_removal_to_roi_locator_checkbox,
            4,
            0,
            1,
            2,
        )
        background_layout.addWidget(
            self.apply_background_removal_to_model_preprocessing_checkbox,
            5,
            0,
            1,
            2,
        )
        background_layout.addWidget(self.background_status_label, 6, 0, 1, 2)
        background_layout.addWidget(self.background_preview_widget, 7, 0, 1, 2)
        self.control_tabs.addTab(background_tab, "Background")

        roi_fcn_tab = QWidget()
        roi_fcn_layout = QVBoxLayout(roi_fcn_tab)
        self.roi_size_value = QLabel("ROI size: n/a")
        self.roi_size_value.setObjectName("roi_size_value")
        self.roi_locator_canvas_value = QLabel("ROI-FCN canvas: n/a")
        self.roi_locator_canvas_value.setObjectName("roi_locator_canvas_value")
        roi_fcn_layout.addWidget(self.roi_size_value)
        roi_fcn_layout.addWidget(self.roi_locator_canvas_value)
        roi_fcn_layout.addStretch(1)
        self.control_tabs.addTab(roi_fcn_tab, "ROI / FCN")

        debug_tab = QWidget()
        debug_layout = QVBoxLayout(debug_tab)
        self.show_roi_fcn_heatmap_overlay_checkbox = QCheckBox(
            "Show ROI-FCN heatmap overlay"
        )
        self.show_roi_fcn_heatmap_overlay_checkbox.setObjectName(
            "show_roi_fcn_heatmap_overlay_checkbox"
        )
        self.show_roi_fcn_heatmap_overlay_checkbox.setChecked(True)
        self.debug_summary_value = QLabel("Debug artifacts: n/a")
        self.debug_summary_value.setObjectName("debug_summary_value")
        self.debug_summary_value.setWordWrap(True)
        debug_layout.addWidget(self.show_roi_fcn_heatmap_overlay_checkbox)
        debug_layout.addWidget(self.debug_summary_value)
        debug_layout.addStretch(1)
        self.control_tabs.addTab(debug_tab, "Debug")

        panel_layout.addWidget(self.control_tabs, stretch=1)

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
        self.roi_confidence_value = self._add_readout(
            status_grid,
            row=11,
            label="ROI confidence",
            object_name="roi_confidence_value",
        )
        self.roi_clipped_value = self._add_readout(
            status_grid,
            row=12,
            label="ROI clipped",
            object_name="roi_clipped_value",
        )
        self.roi_acceptance_value = self._add_readout(
            status_grid,
            row=13,
            label="ROI status",
            object_name="roi_acceptance_value",
        )
        self.roi_locator_transforms_value = self._add_readout(
            status_grid,
            row=14,
            label="Locator transforms",
            object_name="roi_locator_transforms_value",
        )
        self.roi_locator_transforms_value.setWordWrap(True)

        self.log_panel = QPlainTextEdit()
        self.log_panel.setObjectName("log_panel")
        self.log_panel.setReadOnly(True)
        self.log_panel.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        text_output_group = QGroupBox("Text Outputs")
        text_output_group.setObjectName("text_output_group")
        text_output_layout = QHBoxLayout(text_output_group)
        telemetry_group = QGroupBox("Telemetry")
        telemetry_group.setObjectName("telemetry_group")
        telemetry_group.setLayout(status_grid)
        text_output_layout.addWidget(telemetry_group, stretch=2)
        log_group = QGroupBox("Log")
        log_group.setObjectName("log_group")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_panel, stretch=1)
        text_output_layout.addWidget(log_group, stretch=3)
        output_layout.addWidget(text_output_group, stretch=2)

        root_layout.addLayout(output_layout, stretch=4)
        panel_layout.addStretch(1)
        root_layout.addWidget(self.right_control_panel, stretch=1)

        self.setCentralWidget(central)
        self._set_mask_button_state(None)
        self._sync_background_controls_from_state()

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
        self.capture_frame_button.clicked.connect(self.capture_single_frame)
        self.run_single_inference_button.clicked.connect(self.run_single_inference)
        self.draw_mask_button.clicked.connect(self.start_draw_mask)
        self.apply_mask_button.clicked.connect(self.apply_draw_mask)
        self.erase_mask_button.clicked.connect(self.start_erase_mask)
        self.apply_erase_button.clicked.connect(self.apply_erase_mask)
        self.clear_mask_button.clicked.connect(self.clear_mask)
        self.mask_brush_diameter_input.valueChanged.connect(self._on_brush_diameter_changed)
        self.mask_fill_white_checkbox.toggled.connect(self._on_mask_fill_toggled)
        self.apply_manual_mask_to_roi_locator_checkbox.toggled.connect(
            self._on_apply_manual_mask_to_roi_locator_toggled
        )
        self.apply_manual_mask_to_model_preprocessing_checkbox.toggled.connect(
            self._on_apply_manual_mask_to_model_preprocessing_toggled
        )
        self.capture_background_button.clicked.connect(self.capture_background)
        self.enable_background_removal_checkbox.toggled.connect(
            self._on_background_enabled_toggled
        )
        self.apply_background_removal_to_roi_locator_checkbox.toggled.connect(
            self._on_apply_background_removal_to_roi_locator_toggled
        )
        self.apply_background_removal_to_model_preprocessing_checkbox.toggled.connect(
            self._on_apply_background_removal_to_model_preprocessing_toggled
        )
        self.clear_background_button.clicked.connect(self.clear_background)
        self.background_threshold_input.valueChanged.connect(
            self._on_background_threshold_changed
        )
        self.show_roi_fcn_heatmap_overlay_checkbox.toggled.connect(
            self._on_roi_fcn_heatmap_overlay_toggled
        )

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
        self._inference_start_requested = True

    def stop_inference(self) -> None:
        self._call_controller(
            self.inference_controller,
            "request_stop",
            "Inference stop",
        )
        self._inference_start_requested = False

    def stop_all(self) -> None:
        self._call_controller(self.camera_controller, "request_stop", "Camera stop")
        self._call_controller(
            self.inference_controller,
            "request_stop",
            "Inference stop",
        )
        self._inference_start_requested = False

    def capture_single_frame(self) -> None:
        reader = self.frame_reader
        if reader is None:
            self._append_log("ERROR", "Capture Frame unavailable: no frame reader configured.")
            return

        latest_frame = self._latest_completed_frame(reader)
        if latest_frame is None:
            self._append_log("WARNING", "Capture Frame failed: no completed frame is available.")
            return

        try:
            image_bytes = bytes(reader.read_frame_bytes(latest_frame))
        except Exception as exc:
            self._append_log("ERROR", f"Capture Frame failed: {exc}")
            return

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
        self.last_captured_frame_hash_value.setText(frame_hash.value)
        self._display_captured_frame(image_bytes)
        self._append_log("INFO", f"Captured frame {frame_hash.value}")

    def run_single_inference(self) -> None:
        captured = self._captured_single_frame
        if captured is None:
            self._append_log(
                "WARNING",
                "Run Single Inference requires a captured frame.",
            )
            return
        if self._inference_is_running_or_requested():
            self._append_log(
                "WARNING",
                "Single-frame inference refused because continuous inference is running",
            )
            return
        runner = self.single_frame_runner
        if runner is None:
            self._append_log(
                "ERROR",
                "Run Single Inference unavailable: no single-frame runner configured.",
            )
            return

        self._append_log("INFO", "Single-frame inference started")
        try:
            outcome = runner.run_single_frame(
                captured.image_bytes,
                source_path=captured.source_path,
                frame_metadata=captured.frame_metadata,
                record_trace=self.record_trace_checkbox.isChecked(),
            )
        except Exception as exc:
            self._append_log("ERROR", f"Single-frame inference failed: {exc}")
            return

        result = _payload_value(outcome, "result")
        error = _payload_value(outcome, "error")
        trace_path = _path_or_none(_payload_value(outcome, "trace_path"))

        if result is not None:
            self._on_inference_result_ready(result)
            self._append_log("INFO", "Single-frame inference completed")

        if trace_path is not None:
            self.last_trace_path_value.setText(str(trace_path))
            self._append_log("INFO", f"Trace written to {trace_path}")

        if error is not None:
            self._on_error_occurred(error)

    def _latest_completed_frame(self, reader: object) -> object | None:
        method = getattr(reader, "latest_completed_frame", None)
        if not callable(method):
            self._append_log("ERROR", "Capture Frame unavailable: reader has no latest_completed_frame().")
            return None
        try:
            return method()
        except Exception as exc:
            self._append_log("ERROR", f"Capture Frame failed: {exc}")
            return None

    def _display_captured_frame(self, image_bytes: bytes) -> None:
        image = QImage.fromData(image_bytes)
        if image.isNull():
            self._append_log("WARNING", "Captured frame preview unavailable: bytes could not be decoded.")
            return
        self.frame_preview_widget.set_image(image)
        self._sync_preview_background_if_changed()
        self._refresh_background_preview()

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
        self.frame_preview_widget.set_mask_fill_value(fill_value)

    def _current_mask_fill_value(self) -> int:
        return 255 if self.mask_fill_white_checkbox.isChecked() else 0

    def _on_apply_manual_mask_to_roi_locator_toggled(self, checked: bool) -> None:
        snapshot = self.stage_policy_state.update(
            apply_manual_mask_to_roi_locator=bool(checked)
        )
        self._append_log(
            "INFO",
            "Manual mask ROI locator application "
            f"{_enabled_text(checked)}: revision={snapshot.revision}.",
        )

    def _on_apply_manual_mask_to_model_preprocessing_toggled(
        self,
        checked: bool,
    ) -> None:
        snapshot = self.stage_policy_state.update(
            apply_manual_mask_to_regressor_preprocessing=bool(checked)
        )
        self._append_log(
            "INFO",
            "Manual mask model preprocessing application "
            f"{_enabled_text(checked)}: revision={snapshot.revision}.",
        )

    def capture_background(self) -> None:
        if self._inference_is_running_or_requested():
            message = "Stop inference before capturing background"
            self._append_log("WARNING", message)
            self._sync_background_controls_from_state()
            return

        gray = self.frame_preview_widget.raw_source_gray()
        if gray is None:
            self._append_log(
                "WARNING",
                "Cannot capture background: no preview frame is loaded.",
            )
            self._sync_background_controls_from_state()
            return

        revision = self.background_state.capture_background(gray)
        snapshot = self.background_state.get_snapshot()
        self._apply_background_snapshot_to_preview(snapshot)
        self._sync_background_controls_from_state(snapshot)
        self._append_log(
            "INFO",
            "Background captured: "
            f"revision={revision} size={snapshot.width_px}x{snapshot.height_px}.",
        )

    def clear_background(self) -> None:
        revision = self.background_state.clear()
        snapshot = self.background_state.get_snapshot()
        self._apply_background_snapshot_to_preview(snapshot)
        self._sync_background_controls_from_state(snapshot)
        self._append_log("INFO", f"Background cleared: revision={revision}.")

    def _on_background_enabled_toggled(self, checked: bool) -> None:
        revision = self.background_state.set_enabled(bool(checked))
        snapshot = self.background_state.get_snapshot()
        self._apply_background_snapshot_to_preview(snapshot)
        self._sync_background_controls_from_state(snapshot)
        state = "enabled" if checked else "disabled"
        self._append_log("INFO", f"Background removal {state}: revision={revision}.")

    def _on_apply_background_removal_to_roi_locator_toggled(
        self,
        checked: bool,
    ) -> None:
        snapshot = self.stage_policy_state.update(
            apply_background_removal_to_roi_locator=bool(checked)
        )
        self._append_log(
            "INFO",
            "Background removal ROI locator application "
            f"{_enabled_text(checked)}: revision={snapshot.revision}.",
        )

    def _on_apply_background_removal_to_model_preprocessing_toggled(
        self,
        checked: bool,
    ) -> None:
        snapshot = self.stage_policy_state.update(
            apply_background_removal_to_regressor_preprocessing=bool(checked)
        )
        self._append_log(
            "INFO",
            "Background removal model preprocessing application "
            f"{_enabled_text(checked)}: revision={snapshot.revision}.",
        )

    def _on_background_threshold_changed(self, value: int) -> None:
        self.background_state.set_threshold(int(value))
        snapshot = self.background_state.get_snapshot()
        self._apply_background_snapshot_to_preview(snapshot)
        self._sync_background_controls_from_state(snapshot)

    def _on_roi_fcn_heatmap_overlay_toggled(self, checked: bool) -> None:
        self.frame_preview_widget.set_heatmap_overlay_enabled(bool(checked))

    def _sync_background_controls_from_state(self, snapshot: object | None = None) -> None:
        if snapshot is None:
            snapshot = self.background_state.get_snapshot()
        self.enable_background_removal_checkbox.blockSignals(True)
        self.enable_background_removal_checkbox.setChecked(bool(snapshot.enabled))
        self.enable_background_removal_checkbox.blockSignals(False)
        self.background_threshold_input.blockSignals(True)
        self.background_threshold_input.setValue(int(snapshot.threshold))
        self.background_threshold_input.blockSignals(False)
        self.background_status_label.setText(self._background_status_text(snapshot))

    def _apply_background_snapshot_to_preview(self, snapshot: object) -> None:
        self._preview_background_revision = int(getattr(snapshot, "revision", 0))
        self._refresh_background_preview(snapshot)

    def _sync_preview_background_if_changed(self) -> None:
        revision = self._background_revision()
        if self._preview_background_revision == revision:
            return
        snapshot = self.background_state.get_snapshot()
        self._apply_background_snapshot_to_preview(snapshot)
        self._sync_background_controls_from_state(snapshot)

    def _refresh_background_preview(self, snapshot: object | None = None) -> None:
        widget = getattr(self, "background_preview_widget", None)
        if widget is None:
            return
        source_image = self.frame_preview_widget.raw_image()
        if source_image is None:
            return
        if snapshot is None:
            snapshot = self.background_state.get_snapshot()
        max_width, max_height = self._background_preview_max_size
        widget.set_background_preview_image(
            source_image,
            snapshot,
            max_width_px=max_width,
            max_height_px=max_height,
            fill_value=255,
        )

    def _background_revision(self) -> int:
        revision = getattr(self.background_state, "revision", None)
        if callable(revision):
            try:
                return int(revision())
            except Exception as exc:
                self._append_log(
                    "WARNING",
                    f"Background revision check failed: {exc}",
                )
        return int(getattr(self.background_state.get_snapshot(), "revision", 0))

    def _background_status_text(self, snapshot: object) -> str:
        captured = bool(getattr(snapshot, "captured", False))
        if not captured:
            return "Background: not captured"
        current_size = self.frame_preview_widget.source_image_size()
        captured_size = (
            int(getattr(snapshot, "width_px", 0)),
            int(getattr(snapshot, "height_px", 0)),
        )
        if bool(getattr(snapshot, "enabled", False)):
            if current_size is not None and current_size != captured_size:
                return "Background: size mismatch"
            return "Background: enabled"
        return "Background: captured"

    def _inference_is_running_or_requested(self) -> bool:
        is_running = getattr(self.inference_controller, "is_running", None)
        if callable(is_running):
            try:
                if bool(is_running()):
                    return True
            except Exception as exc:
                self._append_log(
                    "WARNING",
                    f"Inference running-state check failed: {exc}",
                )
        if self._inference_start_requested:
            return True
        state = self._last_inference_state_text.strip().lower()
        return state in {"starting", "running", "stopping"}

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
        if self._should_update_frame_preview():
            self._display_frame_preview(path)
            self._last_preview_update_seconds = monotonic()
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
        self._sync_preview_background_if_changed()
        self._refresh_background_preview()

    def _should_update_frame_preview(self) -> bool:
        now = monotonic()
        interval_seconds = self._current_preview_update_interval_seconds()
        if now - self._last_preview_update_seconds < interval_seconds:
            return False
        return True

    def _current_preview_update_interval_seconds(self) -> float:
        return self._preview_update_interval_seconds

    def _on_inference_status_changed(self, status: object) -> None:
        self._last_inference_state_text = _enum_text(_payload_value(status, "state"))
        if self._last_inference_state_text.lower() in {"stopped", "error"}:
            self._inference_start_requested = False
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
        self._update_roi_fcn_readouts(result)
        self._update_debug_artifact_paths(result)
        self._log_result_warnings(result)

    def _on_debug_image_ready(self, image: object) -> None:
        image_kind = _text(_payload_value(image, "image_kind"), default="debug")
        path = _text(_payload_value(image, "path"), default="n/a")
        self._load_debug_preview(image_kind, path)
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

    def _update_roi_fcn_readouts(self, result: object) -> None:
        roi_metadata = _payload_value(result, "roi_metadata")
        distance_wh = _size_tuple(_payload_value(roi_metadata, "distance_canvas_wh_px"))
        if distance_wh is not None:
            self.roi_size_value.setText(f"ROI size: {distance_wh[0]} x {distance_wh[1]}")
        extras = _mapping_payload(_payload_value(roi_metadata, "extras"))
        locator_metadata = _mapping_payload(
            extras.get(contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA)
        )
        canvas_w = _optional_int(
            locator_metadata.get(
                contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_WIDTH_PX
            )
        )
        canvas_h = _optional_int(
            locator_metadata.get(
                contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_HEIGHT_PX
            )
        )
        if canvas_w is not None and canvas_h is not None:
            self.roi_locator_canvas_value.setText(
                f"ROI-FCN canvas: {canvas_w} x {canvas_h}"
            )
        self._apply_roi_status_from_mapping(extras, locator_metadata=locator_metadata)

    def _apply_roi_status_from_mapping(
        self,
        payload: Mapping[str, object],
        *,
        locator_metadata: Mapping[str, object] | None = None,
    ) -> None:
        locator = locator_metadata or _mapping_payload(
            payload.get(contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA)
        )
        confidence = _first_present(
            payload.get("roi_confidence"),
            locator.get("heatmap_peak_confidence"),
            _mapping_payload(locator.get("decoded_heatmap")).get("confidence"),
        )
        self.roi_confidence_value.setText(_format_optional_float(confidence, precision=3))

        clipped = _optional_bool(payload.get("roi_clipped"))
        self.roi_clipped_value.setText(_yes_no_unknown(clipped))

        accepted = _optional_bool(payload.get("roi_accepted"))
        reason = _text(payload.get("roi_rejection_reason"), default="")
        if accepted is True:
            self.roi_acceptance_value.setText("accepted")
        elif accepted is False:
            suffix = f" - {reason}" if reason else ""
            self.roi_acceptance_value.setText(f"rejected{suffix}")
        else:
            self.roi_acceptance_value.setText("n/a")

        self.roi_locator_transforms_value.setText(_locator_transform_status(payload))

    def _update_debug_artifact_paths(self, result: object) -> None:
        debug_paths = _mapping_payload(_payload_value(result, "debug_paths"))
        if not debug_paths:
            self.debug_artifacts_value.setText("n/a")
            self.debug_summary_value.setText("Debug artifacts: n/a")
            return

        summary = " | ".join(
            f"{key}: {_text(path, default='n/a')}"
            for key, path in sorted(debug_paths.items())
        )
        self.debug_artifacts_value.setText(summary)
        self.debug_summary_value.setText(summary)
        self._load_debug_previews(debug_paths)
        self._append_log("INFO", f"Debug artifacts: {summary}")

    def _load_debug_previews(self, debug_paths: Mapping[str, object]) -> None:
        for image_kind in (contracts.DISPLAY_ARTIFACT_ROI_CROP,):
            path = debug_paths.get(image_kind)
            if path is not None:
                self._load_debug_preview(image_kind, path)

    def _load_debug_preview(self, image_kind: object, path: object) -> None:
        if str(image_kind) != contracts.DISPLAY_ARTIFACT_ROI_CROP:
            return
        if not self.roi_crop_preview_widget.load_image(str(path)):
            self._append_log("WARNING", f"ROI crop preview unavailable: {path}")

    def _log_result_warnings(self, result: object) -> None:
        warnings = _sequence_payload(_payload_value(result, "warnings"))
        for warning in warnings:
            text = str(warning)
            if "frame mask skipped" not in text and "background removal skipped" not in text:
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
        if _is_roi_rejection_error(error):
            details = _mapping_payload(_payload_value(error, "details"))
            self._apply_roi_status_from_mapping(details)
            self.distance_value.setText("n/a")
            self.yaw_value.setText("n/a")
            self.frame_preview_widget.set_overlay(None)
            self.roi_crop_preview_widget.set_image(QImage())
            self.roi_crop_preview_widget.set_placeholder_text("ROI rejected")
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


def _path_or_none(value: object | None) -> Path | None:
    if value is None:
        return None
    try:
        return Path(value)
    except TypeError:
        return None


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
        *contracts.ROI_OVERLAY_BOUNDS_METADATA_KEYS,
    )
    locator_metadata = _mapping_payload(
        extras.get(contracts.PREPROCESSING_METADATA_ROI_LOCATOR_METADATA)
    )
    heatmap = _heatmap_overlay_from_locator_metadata(locator_metadata)
    if bbox is None and center is None and roi_bounds is None and heatmap is None:
        return None
    return FramePreviewOverlay(
        source_image_wh_px=source_size,
        bbox_xyxy_px=bbox,
        center_xy_px=center,
        roi_bounds_xyxy_px=roi_bounds,
        roi_fcn_heatmap=heatmap,
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


def _is_roi_rejection_error(error: object) -> bool:
    if _text(_payload_value(error, "error_type"), default="") == "roi_rejected":
        return True
    details = _mapping_payload(_payload_value(error, "details"))
    return _optional_bool(details.get("roi_accepted")) is False


def _locator_transform_status(payload: Mapping[str, object]) -> str:
    manual_configured = _optional_bool(payload.get("apply_manual_mask_to_roi_locator"))
    manual_applied = _optional_bool(payload.get("manual_mask_applied_to_roi_locator"))
    background_configured = _optional_bool(
        payload.get("apply_background_removal_to_roi_locator")
    )
    background_applied = _optional_bool(
        payload.get("background_removal_applied_to_roi_locator")
    )
    if (
        manual_configured is None
        and manual_applied is None
        and background_configured is None
        and background_applied is None
    ):
        return "n/a"
    return (
        "mask "
        f"{_yes_no_unknown(manual_applied)}"
        f" (configured {_yes_no_unknown(manual_configured)}); "
        "background "
        f"{_yes_no_unknown(background_applied)}"
        f" (configured {_yes_no_unknown(background_configured)})"
    )


def _optional_bool(value: object | None) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "on", "accepted"}:
            return True
        if text in {"false", "no", "0", "off", "rejected"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _yes_no_unknown(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "n/a"


def _format_optional_float(value: object | None, *, precision: int) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _text(value, default="n/a")
    return f"{number:.{precision}f}"


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
        value = _xyxy_tuple(payload.get(key))
        if value is not None:
            return value
    return None


def _heatmap_overlay_from_locator_metadata(
    locator_metadata: Mapping[str, object],
) -> FramePreviewHeatmapOverlay | None:
    heatmap = locator_metadata.get(
        contracts.PREPROCESSING_METADATA_ROI_FCN_HEATMAP_U8
    )
    if heatmap is None:
        return None
    canvas_w = _optional_int(
        locator_metadata.get(contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_WIDTH_PX)
    )
    canvas_h = _optional_int(
        locator_metadata.get(contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_HEIGHT_PX)
    )
    resized_wh = _size_tuple(
        locator_metadata.get(
            contracts.PREPROCESSING_METADATA_ROI_FCN_RESIZED_IMAGE_WH_PX
        )
    )
    padding = _int_tuple(
        locator_metadata.get(contracts.PREPROCESSING_METADATA_ROI_FCN_PADDING_LTRB_PX),
        width=4,
    )
    if canvas_w is None or canvas_h is None or resized_wh is None or padding is None:
        return None
    try:
        return FramePreviewHeatmapOverlay(
            heatmap_u8=heatmap,
            canvas_wh_px=(canvas_w, canvas_h),
            resized_image_wh_px=resized_wh,
            padding_ltrb_px=padding,
        )
    except ValueError:
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


def _int_tuple(value: object | None, *, width: int) -> tuple[int, ...] | None:
    parsed = _float_tuple(value, width=width)
    if parsed is None:
        return None
    return tuple(int(item) for item in parsed)


def _optional_int(value: object | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


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


def _enabled_text(value: object) -> str:
    return "enabled" if bool(value) else "disabled"


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
