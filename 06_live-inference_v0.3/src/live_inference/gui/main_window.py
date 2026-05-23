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
    QComboBox,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QSpinBox,
    QWidget,
)

import interfaces.contracts as contracts
from live_inference.frame_handoff import compute_frame_hash
from live_inference.masking import BackgroundState, FrameMaskState
from live_inference.preprocessing import (
    CAMERA_INTRINSICS_MODE_LABELS,
    SUPPORTED_CAMERA_INTRINSICS_MODES,
    CameraIntrinsicsFrameTransformer,
    CameraIntrinsicsTransformState,
    StageTransformPolicyState,
    normalize_camera_intrinsics_mode,
)

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
        mask_state: FrameMaskState | None = None,
        locator_parameter_state: object | None = None,
        foreground_extraction_policy_state: object | None = None,
        stage_policy_state: StageTransformPolicyState | None = None,
        camera_intrinsics_state: CameraIntrinsicsTransformState | None = None,
        camera_intrinsics_frame_transformer: CameraIntrinsicsFrameTransformer | None = None,
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
        self.mask_state = mask_state or FrameMaskState()
        self.locator_parameter_state = locator_parameter_state
        self.foreground_extraction_policy_state = foreground_extraction_policy_state
        self.stage_policy_state = stage_policy_state or StageTransformPolicyState()
        self.camera_intrinsics_state = camera_intrinsics_state or CameraIntrinsicsTransformState()
        self.camera_intrinsics_frame_transformer = (
            camera_intrinsics_frame_transformer
            if camera_intrinsics_frame_transformer is not None
            else CameraIntrinsicsFrameTransformer(self.camera_intrinsics_state)
        )
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
        self._last_camera_intrinsics_preview_warning: str | None = None
        self._last_preview_update_seconds = 0.0
        self._preview_update_interval_seconds = 1.0 / 15.0

        self.setWindowTitle("Live Defender Inference v0.3")
        self._load_ui()
        self._bind_widgets()
        self._configure_layout()
        self._populate_camera_intrinsics_mode_combo()
        self._connect_ui()
        self._connect_worker_signals()
        self._sync_camera_intrinsics_widgets()
        self._sync_parameter_widgets()
        self._sync_background_widgets()
        self._sync_preprocessing_widgets()
        self._sync_mask_widgets()
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
        self.top_workspace_splitter = self._require(QSplitter, "topWorkspaceSplitter")
        self.main_preview_widget = self._require(FramePreviewWidget, "mainPreviewWidget")
        self.start_camera_button = self._require(QPushButton, "startCameraButton")
        self.stop_camera_button = self._require(QPushButton, "stopCameraButton")
        self.capture_background_button = self._require(QPushButton, "captureBackgroundButton")
        self.clear_background_button = self._require(QPushButton, "clearBackgroundButton")
        self.enable_background_removal_checkbox = self._require(QCheckBox, "enableBackgroundRemovalCheckBox")
        self.apply_background_removal_to_locator_checkbox = self._require(
            QCheckBox,
            "applyBackgroundRemovalToLocatorCheckBox",
        )
        self.apply_background_removal_to_model_preprocessing_checkbox = self._require(
            QCheckBox,
            "applyBackgroundRemovalToModelPreprocessingCheckBox",
        )
        self.capture_frame_button = self._require(QPushButton, "captureFrameButton")
        self.run_locator_button = self._require(QPushButton, "runLocatorButton")
        self.run_single_inference_button = self._require(QPushButton, "runSingleInferenceButton")
        self.start_continuous_button = self._require(QPushButton, "startContinuousButton")
        self.stop_continuous_button = self._require(QPushButton, "stopContinuousButton")
        self.record_trace_checkbox = self._require(QCheckBox, "recordTraceCheckBox")
        self.camera_intrinsics_mode_combo = self._require(
            QComboBox,
            "cameraIntrinsicsModeComboBox",
        )
        self.draw_mask_button = self._require(QPushButton, "drawMaskButton")
        self.erase_mask_button = self._require(QPushButton, "eraseMaskButton")
        self.apply_mask_button = self._require(QPushButton, "applyMaskButton")
        self.cancel_mask_button = self._require(QPushButton, "cancelMaskButton")
        self.clear_mask_button = self._require(QPushButton, "clearMaskButton")
        self.mask_brush_size_spinbox = self._require(QSpinBox, "maskBrushSizeSpinBox")
        self.mask_fill_white_checkbox = self._require(QCheckBox, "maskFillWhiteCheckBox")
        self.show_roi_checkbox = self._require(QCheckBox, "showRoiCheckBox")
        self.show_bbox_checkbox = self._require(QCheckBox, "showBboxCheckBox")
        self.show_foreground_mask_checkbox = self._require(QCheckBox, "showForegroundMaskCheckBox")
        self.show_edges_checkbox = self._require(QCheckBox, "showEdgesCheckBox")
        self.show_candidate_contours_checkbox = self._require(QCheckBox, "showCandidateContoursCheckBox")
        self.show_chosen_contour_checkbox = self._require(QCheckBox, "showChosenContourCheckBox")
        self.use_silhouette_preprocessing_checkbox = self._require(
            QCheckBox,
            "useSilhouettePreprocessingCheckBox",
        )
        self.background_threshold_spinbox = self._require(QSpinBox, "backgroundThresholdSpinBox")
        self.min_foreground_area_spinbox = self._require(QSpinBox, "minForegroundAreaSpinBox")
        self.canny_low_spinbox = self._require(QSpinBox, "cannyLowSpinBox")
        self.canny_high_spinbox = self._require(QSpinBox, "cannyHighSpinBox")
        self.distance_value = self._require(QLabel, "distanceValue")
        self.yaw_value = self._require(QLabel, "yawValue")
        self.locator_status_value = self._require(QLabel, "locatorStatusValue")
        self.roi_status_value = self._require(QLabel, "roiStatusValue")
        self.mask_status_value = self._require(QLabel, "maskStatusValue")
        self.background_removal_status_value = self._require(QLabel, "backgroundRemovalStatusValue")
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
        self.enable_background_removal_checkbox.toggled.connect(
            self._on_background_enabled_toggled
        )
        self.apply_background_removal_to_locator_checkbox.toggled.connect(
            self._on_apply_background_removal_to_locator_toggled
        )
        self.apply_background_removal_to_model_preprocessing_checkbox.toggled.connect(
            self._on_apply_background_removal_to_model_preprocessing_toggled
        )
        self.capture_frame_button.clicked.connect(self.capture_frame)
        self.run_locator_button.clicked.connect(self.run_locator)
        self.run_single_inference_button.clicked.connect(self.run_single_inference)
        self.start_continuous_button.clicked.connect(self.start_continuous_inference)
        self.stop_continuous_button.clicked.connect(self.stop_continuous_inference)
        self.draw_mask_button.clicked.connect(self.start_draw_mask)
        self.erase_mask_button.clicked.connect(self.start_erase_mask)
        self.apply_mask_button.clicked.connect(self.apply_mask)
        self.cancel_mask_button.clicked.connect(self.cancel_mask)
        self.clear_mask_button.clicked.connect(self.clear_mask)
        self.mask_brush_size_spinbox.valueChanged.connect(self._on_mask_brush_size_changed)
        self.mask_fill_white_checkbox.toggled.connect(self._on_mask_fill_toggled)
        self.use_silhouette_preprocessing_checkbox.toggled.connect(
            self._on_use_silhouette_preprocessing_toggled
        )
        self.camera_intrinsics_mode_combo.currentIndexChanged.connect(
            self._on_camera_intrinsics_mode_changed
        )
        for checkbox in (
            self.show_roi_checkbox,
            self.show_bbox_checkbox,
            self.show_foreground_mask_checkbox,
            self.show_edges_checkbox,
            self.show_candidate_contours_checkbox,
            self.show_chosen_contour_checkbox,
        ):
            checkbox.toggled.connect(self._refresh_visual_surface)
        self.background_threshold_spinbox.valueChanged.connect(
            self._on_background_threshold_changed
        )
        for spinbox in (
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

    def _populate_camera_intrinsics_mode_combo(self) -> None:
        self.camera_intrinsics_mode_combo.blockSignals(True)
        self.camera_intrinsics_mode_combo.clear()
        for mode in SUPPORTED_CAMERA_INTRINSICS_MODES:
            self.camera_intrinsics_mode_combo.addItem(
                CAMERA_INTRINSICS_MODE_LABELS.get(mode, mode),
                mode,
            )
        self.camera_intrinsics_mode_combo.blockSignals(False)

    def _require(self, widget_type: type, object_name: str) -> Any:
        widget = self.findChild(widget_type, object_name)
        if widget is None:
            raise RuntimeError(f"UI file is missing {object_name!r}.")
        return widget

    def _configure_layout(self) -> None:
        self.top_workspace_splitter.setChildrenCollapsible(False)
        for index, stretch in enumerate((0, 0, 1, 0)):
            self.top_workspace_splitter.setStretchFactor(index, stretch)
        self.top_workspace_splitter.setSizes([390, 250, 500, 285])

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

    def _inference_is_running(self) -> bool:
        is_running = getattr(self.inference_controller, "is_running", None)
        return bool(is_running()) if callable(is_running) else False

    def capture_frame(self) -> None:
        latest_frame = self._latest_frame()
        if latest_frame is None:
            self._append_log("WARNING", "Capture Frame requires a completed frame.")
            return
        image_bytes = self.frame_reader.read_frame_bytes(latest_frame)
        preview_image = self._preview_qimage_from_bytes(image_bytes)
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
        if preview_image is not None:
            self._set_base_preview_image(preview_image)
        self.frame_hash_value.setText(f"frame hash: {frame_hash.value}")
        self._append_log("INFO", f"Captured frame {frame_hash.value}")

    def capture_background(self) -> None:
        if self._inference_is_running():
            self._append_log("WARNING", "Stop inference before capturing background.")
            self._sync_background_widgets()
            return
        image_bytes = (
            self._captured_single_frame.image_bytes
            if self._captured_single_frame is not None
            else self._latest_frame_bytes()
        )
        if image_bytes is None:
            self._append_log("WARNING", "Capture Background requires a frame.")
            self._sync_background_widgets()
            return
        gray = self._preview_grayscale_from_bytes(image_bytes)
        revision = self.background_state.capture_background(gray)
        snapshot = self.background_state.get_snapshot()
        self._sync_background_widgets(snapshot)
        self._append_log(
            "INFO",
            "Background captured; "
            f"revision={revision}; size={snapshot.width_px}x{snapshot.height_px}; "
            "enable removal before starting inference.",
        )

    def clear_background(self) -> None:
        revision = self.background_state.clear()
        self._sync_background_widgets()
        self._append_log("INFO", f"Background cleared; revision={revision}")

    def start_draw_mask(self) -> None:
        self._prepare_preview_for_mask_edit()
        self.main_preview_widget.set_brush_diameter_px(self.mask_brush_size_spinbox.value())
        self.main_preview_widget.set_committed_mask_snapshot(self.mask_state.get_snapshot())
        self.main_preview_widget.begin_mask_edit("draw")
        self._update_mask_status("drawing")
        self._append_log("INFO", "Draw Mask started")

    def start_erase_mask(self) -> None:
        self._prepare_preview_for_mask_edit()
        self.main_preview_widget.set_brush_diameter_px(self.mask_brush_size_spinbox.value())
        self.main_preview_widget.set_committed_mask_snapshot(self.mask_state.get_snapshot())
        self.main_preview_widget.begin_mask_edit("erase")
        self._update_mask_status("erasing")
        self._append_log("INFO", "Erase Mask started")

    def apply_mask(self) -> None:
        result = self.main_preview_widget.finish_mask_edit(commit=True)
        if result is None:
            self._update_mask_status()
            self._append_log("WARNING", "Apply Mask requires a loaded preview frame.")
            return
        revision = self.mask_state.commit_mask(
            result.mask,
            width_px=result.width_px,
            height_px=result.height_px,
            fill_value=self._current_mask_fill_value(),
        )
        snapshot = self.mask_state.get_snapshot()
        self.main_preview_widget.set_committed_mask_snapshot(snapshot)
        self._update_mask_status()
        self._append_log(
            "INFO",
            "Mask committed: "
            f"revision={revision}; size={result.width_px}x{result.height_px}; "
            f"pixels={snapshot.pixel_count}; fill={snapshot.fill_value}",
        )

    def cancel_mask(self) -> None:
        self.main_preview_widget.cancel_mask_edit()
        self._sync_preview_mask_snapshot()
        self._append_log("INFO", "Mask edit cancelled")

    def clear_mask(self) -> None:
        revision = self.mask_state.clear()
        self.main_preview_widget.clear_masks()
        self._update_mask_status()
        self._append_log("INFO", f"Mask cleared: revision={revision}")

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
        if self.main_preview_widget.is_mask_editing():
            return
        now = monotonic()
        if now - self._last_preview_update_seconds < self._current_preview_update_interval_seconds():
            return
        self._last_preview_update_seconds = now
        path = _path_or_none(_payload_value(frame, "image_path"))
        if path is not None:
            self._load_preview_frame(path)

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
            if path is not None:
                self.main_preview_widget.set_committed_mask_snapshot(None)
            if path is not None and self.main_preview_widget.load_image(path):
                self.main_preview_widget.set_overlay(None)
                return
        if self._base_preview_image is not None:
            self.main_preview_widget.set_image(self._base_preview_image)
            self._sync_preview_mask_snapshot()
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
        self._sync_background_widgets()
        self._append_log("DEBUG", f"Locator parameters revision={revision}")

    def _on_background_threshold_changed(self, value: int) -> None:
        self.background_state.set_threshold(int(value))
        self._on_locator_parameters_changed(value)
        self._sync_background_widgets()

    def _on_background_enabled_toggled(self, checked: bool) -> None:
        revision = self.background_state.set_enabled(bool(checked))
        self._sync_background_widgets()
        state = "enabled" if bool(checked) else "disabled"
        self._append_log("INFO", f"Background removal {state}; revision={revision}")

    def _on_apply_background_removal_to_locator_toggled(self, checked: bool) -> None:
        snapshot = self.stage_policy_state.update(
            apply_background_removal_to_roi_locator=bool(checked)
        )
        self._sync_background_widgets()
        self._append_log(
            "INFO",
            "Background removal locator application "
            f"{'enabled' if bool(checked) else 'disabled'}; revision={snapshot.revision}",
        )

    def _on_apply_background_removal_to_model_preprocessing_toggled(
        self,
        checked: bool,
    ) -> None:
        snapshot = self.stage_policy_state.update(
            apply_background_removal_to_regressor_preprocessing=bool(checked)
        )
        self._sync_background_widgets()
        self._append_log(
            "INFO",
            "Background removal model preprocessing application "
            f"{'enabled' if bool(checked) else 'disabled'}; revision={snapshot.revision}",
        )

    def _sync_parameter_widgets(self) -> None:
        state = self.locator_parameter_state
        snapshot = getattr(state, "snapshot", None)
        if not callable(snapshot):
            return
        config, _revision = snapshot()
        for spinbox in (
            self.background_threshold_spinbox,
            self.min_foreground_area_spinbox,
            self.canny_low_spinbox,
            self.canny_high_spinbox,
        ):
            spinbox.blockSignals(True)
        try:
            self.background_threshold_spinbox.setValue(int(config.background_threshold))
            self.min_foreground_area_spinbox.setValue(int(config.min_foreground_area_px))
            self.canny_low_spinbox.setValue(int(config.canny_low_threshold))
            self.canny_high_spinbox.setValue(int(config.canny_high_threshold))
        finally:
            for spinbox in (
                self.background_threshold_spinbox,
                self.min_foreground_area_spinbox,
                self.canny_low_spinbox,
                self.canny_high_spinbox,
            ):
                spinbox.blockSignals(False)
        self.background_state.set_threshold(int(config.background_threshold))

    def _sync_background_widgets(self, snapshot: object | None = None) -> None:
        background_snapshot = (
            snapshot
            if snapshot is not None
            else self.background_state.get_snapshot()
        )
        stage_snapshot = self.stage_policy_state.get_snapshot()
        self.enable_background_removal_checkbox.blockSignals(True)
        self.apply_background_removal_to_locator_checkbox.blockSignals(True)
        self.apply_background_removal_to_model_preprocessing_checkbox.blockSignals(True)
        self.background_threshold_spinbox.blockSignals(True)
        try:
            self.enable_background_removal_checkbox.setChecked(
                bool(background_snapshot.enabled)
            )
            self.apply_background_removal_to_locator_checkbox.setChecked(
                bool(stage_snapshot.apply_background_removal_to_roi_locator)
            )
            self.apply_background_removal_to_model_preprocessing_checkbox.setChecked(
                bool(stage_snapshot.apply_background_removal_to_regressor_preprocessing)
            )
            self.background_threshold_spinbox.setValue(int(background_snapshot.threshold))
        finally:
            self.enable_background_removal_checkbox.blockSignals(False)
            self.apply_background_removal_to_locator_checkbox.blockSignals(False)
            self.apply_background_removal_to_model_preprocessing_checkbox.blockSignals(False)
            self.background_threshold_spinbox.blockSignals(False)
        self.main_preview_widget.set_background_snapshot(background_snapshot)
        self.background_removal_status_value.setText(
            _background_status_text(background_snapshot, stage_snapshot)
        )

    def _sync_preprocessing_widgets(self) -> None:
        state = self.foreground_extraction_policy_state
        snapshot = _policy_snapshot(state)
        if snapshot is None:
            self.use_silhouette_preprocessing_checkbox.setEnabled(False)
            self.use_silhouette_preprocessing_checkbox.setChecked(False)
            return
        mode = _text(
            _payload_value(
                snapshot,
                contracts.PREPROCESSING_RUNTIME_PARAMETER_FOREGROUND_EXTRACTION_MODE,
            ),
            default=contracts.ForegroundExtractionMode.THRESHOLD_FOREGROUND_V1.value,
        )
        self.use_silhouette_preprocessing_checkbox.blockSignals(True)
        self.use_silhouette_preprocessing_checkbox.setChecked(
            mode == contracts.ForegroundExtractionMode.SILHOUETTE_CONTOUR_V2.value
        )
        self.use_silhouette_preprocessing_checkbox.blockSignals(False)

    def _sync_camera_intrinsics_widgets(self) -> None:
        snapshot = _policy_snapshot(self.camera_intrinsics_state)
        if snapshot is None:
            self.camera_intrinsics_mode_combo.setEnabled(False)
            return
        mode = _text(
            _payload_value(snapshot, "camera_intrinsics_mode")
            or _payload_value(snapshot, "mode"),
            default=SUPPORTED_CAMERA_INTRINSICS_MODES[0],
        )
        try:
            mode = normalize_camera_intrinsics_mode(mode)
        except ValueError:
            mode = SUPPORTED_CAMERA_INTRINSICS_MODES[0]
        self.camera_intrinsics_mode_combo.blockSignals(True)
        index = self.camera_intrinsics_mode_combo.findData(mode)
        self.camera_intrinsics_mode_combo.setCurrentIndex(max(0, index))
        self.camera_intrinsics_mode_combo.blockSignals(False)

    def _on_camera_intrinsics_mode_changed(self, _index: int) -> None:
        mode = self.camera_intrinsics_mode_combo.currentData()
        try:
            normalized = normalize_camera_intrinsics_mode(mode)
        except ValueError as exc:
            self._append_log("ERROR", str(exc))
            self._sync_camera_intrinsics_widgets()
            return
        update = getattr(self.camera_intrinsics_state, "update", None)
        if not callable(update):
            return
        before = _policy_snapshot(self.camera_intrinsics_state)
        previous_mode = _text(
            _payload_value(before, "camera_intrinsics_mode")
            or _payload_value(before, "mode"),
            default=SUPPORTED_CAMERA_INTRINSICS_MODES[0],
        )
        snapshot, revision = update(mode=normalized)
        current_mode = _text(
            _payload_value(snapshot, "camera_intrinsics_mode")
            or _payload_value(snapshot, "mode"),
            default=normalized,
        )
        if current_mode == previous_mode:
            return
        label = CAMERA_INTRINSICS_MODE_LABELS.get(current_mode, current_mode)
        self.background_state.clear()
        self.mask_state.clear()
        self.main_preview_widget.clear_masks()
        self._sync_background_widgets()
        self._debug_paths = {}
        self._last_overlay = None
        self.main_preview_widget.set_overlay(None)
        self._refresh_artifact_summary()
        if self._captured_single_frame is not None:
            preview_image = self._preview_qimage_from_bytes(
                self._captured_single_frame.image_bytes
            )
            if preview_image is not None:
                self._set_base_preview_image(preview_image)
        self._update_mask_status()
        self._append_log(
            "INFO",
            f"Camera intrinsics: {label}; revision={revision}; background and mask cleared",
        )

    def _on_use_silhouette_preprocessing_toggled(self, checked: bool) -> None:
        state = self.foreground_extraction_policy_state
        update = getattr(state, "update", None)
        if not callable(update):
            return
        mode = (
            contracts.ForegroundExtractionMode.SILHOUETTE_CONTOUR_V2.value
            if bool(checked)
            else contracts.ForegroundExtractionMode.THRESHOLD_FOREGROUND_V1.value
        )
        snapshot, revision = update(
            **{
                contracts.PREPROCESSING_RUNTIME_PARAMETER_FOREGROUND_EXTRACTION_MODE: (
                    mode
                )
            }
        )
        current_mode = _text(
            _payload_value(
                snapshot,
                contracts.PREPROCESSING_RUNTIME_PARAMETER_FOREGROUND_EXTRACTION_MODE,
            ),
            default=mode,
        )
        self._append_log(
            "INFO",
            f"Foreground extraction: {current_mode}; revision={revision}",
        )

    def _sync_mask_widgets(self) -> None:
        snapshot = self.mask_state.get_snapshot()
        self.mask_fill_white_checkbox.blockSignals(True)
        self.mask_fill_white_checkbox.setChecked(int(snapshot.fill_value) == 255)
        self.mask_fill_white_checkbox.blockSignals(False)
        self.main_preview_widget.set_mask_fill_value(int(snapshot.fill_value))
        self.main_preview_widget.set_brush_diameter_px(self.mask_brush_size_spinbox.value())
        self._sync_preview_mask_snapshot()

    def _sync_preview_mask_snapshot(self) -> None:
        self.main_preview_widget.set_mask_fill_value(self._current_mask_fill_value())
        self.main_preview_widget.set_committed_mask_snapshot(self.mask_state.get_snapshot())
        self.main_preview_widget.set_background_snapshot(self.background_state.get_snapshot())
        self._update_mask_status()

    def _prepare_preview_for_mask_edit(self) -> None:
        for checkbox in (
            self.show_foreground_mask_checkbox,
            self.show_edges_checkbox,
            self.show_candidate_contours_checkbox,
            self.show_chosen_contour_checkbox,
        ):
            checkbox.setChecked(False)
        self._refresh_visual_surface()

    def _on_mask_brush_size_changed(self, value: int) -> None:
        self.main_preview_widget.set_brush_diameter_px(int(value))

    def _on_mask_fill_toggled(self, checked: bool) -> None:
        fill_value = 255 if bool(checked) else 0
        revision = self.mask_state.set_fill_value(fill_value)
        self.main_preview_widget.set_mask_fill_value(fill_value)
        self.main_preview_widget.set_committed_mask_snapshot(self.mask_state.get_snapshot())
        self._update_mask_status()
        fill_name = "white" if fill_value == 255 else "black"
        self._append_log("INFO", f"Mask fill set to {fill_name}; revision={revision}")

    def _current_mask_fill_value(self) -> int:
        return 255 if self.mask_fill_white_checkbox.isChecked() else 0

    def _update_mask_status(self, mode: str | None = None) -> None:
        snapshot = self.mask_state.get_snapshot()
        if not snapshot.enabled or not snapshot.has_geometry or snapshot.pixel_count <= 0:
            text = f"mask: none; revision {snapshot.revision}"
        else:
            fill_name = "white" if int(snapshot.fill_value) == 255 else "black"
            text = (
                f"mask: revision {snapshot.revision}; "
                f"{snapshot.pixel_count} px; fill {fill_name}; "
                f"{snapshot.width_px}x{snapshot.height_px}"
            )
            source_size = self.main_preview_widget.source_image_size()
            if source_size is not None and not snapshot.dimensions_match(*source_size):
                text += "; preview size mismatch"
        if mode:
            text += f"; {mode}"
        self.mask_status_value.setText(text)

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

    def _preview_qimage_from_bytes(self, image_bytes: bytes) -> QImage | None:
        transformer = self.camera_intrinsics_frame_transformer
        mode = _camera_intrinsics_mode(self.camera_intrinsics_state)
        if transformer is None or mode == SUPPORTED_CAMERA_INTRINSICS_MODES[0]:
            image = QImage.fromData(bytes(image_bytes))
            if image.isNull():
                self._append_log("WARNING", "Preview frame could not be displayed.")
                return None
            return image.copy()
        try:
            result = transformer.transform_image_bytes(image_bytes)
            return _qimage_from_cv_image_array(result.image)
        except Exception as exc:
            warning = f"{type(exc).__name__}: {exc}"
            if warning != self._last_camera_intrinsics_preview_warning:
                self._last_camera_intrinsics_preview_warning = warning
                self._append_log(
                    "WARNING",
                    "Camera intrinsics preview transform failed; showing raw frame: "
                    f"{warning}",
                )
            image = QImage.fromData(bytes(image_bytes))
            return image.copy() if not image.isNull() else None

    def _preview_grayscale_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        transformer = self.camera_intrinsics_frame_transformer
        mode = _camera_intrinsics_mode(self.camera_intrinsics_state)
        if transformer is None or mode == SUPPORTED_CAMERA_INTRINSICS_MODES[0]:
            return _decode_grayscale(image_bytes)
        try:
            result = transformer.transform_image_bytes(image_bytes, grayscale=True)
            return np.ascontiguousarray(np.asarray(result.image, dtype=np.uint8))
        except Exception as exc:
            warning = f"{type(exc).__name__}: {exc}"
            if warning != self._last_camera_intrinsics_preview_warning:
                self._last_camera_intrinsics_preview_warning = warning
                self._append_log(
                    "WARNING",
                    "Camera intrinsics background transform failed; using raw frame: "
                    f"{warning}",
                )
            return _decode_grayscale(image_bytes)

    def _load_preview_frame(self, path: Path) -> None:
        transformer = self.camera_intrinsics_frame_transformer
        mode = _camera_intrinsics_mode(self.camera_intrinsics_state)
        if transformer is None or mode == SUPPORTED_CAMERA_INTRINSICS_MODES[0]:
            if self.main_preview_widget.load_image(path):
                self._sync_preview_mask_snapshot()
            return
        try:
            image = self._preview_qimage_from_bytes(path.read_bytes())
        except Exception as exc:
            self._append_log("WARNING", f"Could not read preview frame: {exc}")
            return
        if image is None or image.isNull():
            return
        self._set_base_preview_image(image)

    def _current_preview_update_interval_seconds(self) -> float:
        mode = _camera_intrinsics_mode(self.camera_intrinsics_state)
        if mode == SUPPORTED_CAMERA_INTRINSICS_MODES[0]:
            return self._preview_update_interval_seconds
        return max(self._preview_update_interval_seconds, 1.0 / 5.0)

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
        self._set_base_preview_image(image)

    def _set_base_preview_image(self, image: QImage) -> None:
        if image.isNull():
            self._append_log("WARNING", "Captured frame could not be displayed.")
            return
        self._base_preview_image = image.copy()
        self.main_preview_widget.set_image(self._base_preview_image)
        self._sync_preview_mask_snapshot()

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


def _background_status_text(snapshot: object, stage_snapshot: object) -> str:
    captured = bool(_payload_value(snapshot, "captured"))
    enabled = bool(_payload_value(snapshot, "enabled"))
    revision = _payload_value(snapshot, "revision")
    threshold = _payload_value(snapshot, "threshold")
    if captured:
        size = f"{_payload_value(snapshot, 'width_px')}x{_payload_value(snapshot, 'height_px')}"
        base = f"background: captured {size}; revision {revision}; threshold {threshold}"
    else:
        base = f"background: not captured; revision {revision}; threshold {threshold}"
    base += "; enabled" if enabled else "; disabled"
    locator = bool(
        _payload_value(stage_snapshot, "apply_background_removal_to_roi_locator")
    )
    model = bool(
        _payload_value(
            stage_snapshot,
            "apply_background_removal_to_regressor_preprocessing",
        )
    )
    base += f"; locator {'on' if locator else 'off'}; model {'on' if model else 'off'}"
    return base


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


def _qimage_from_cv_image_array(image: np.ndarray) -> QImage:
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim == 2:
        gray = np.ascontiguousarray(array)
        qimage = QImage(
            gray.data,
            int(gray.shape[1]),
            int(gray.shape[0]),
            int(gray.strides[0]),
            QImage.Format.Format_Grayscale8,
        )
        return qimage.copy()
    if array.ndim == 3 and int(array.shape[2]) == 3:
        rgb = cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data,
            int(rgb.shape[1]),
            int(rgb.shape[0]),
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        )
        return qimage.copy()
    if array.ndim == 3 and int(array.shape[2]) == 4:
        rgba = cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_BGRA2RGBA)
        qimage = QImage(
            rgba.data,
            int(rgba.shape[1]),
            int(rgba.shape[0]),
            int(rgba.strides[0]),
            QImage.Format.Format_RGBA8888,
        )
        return qimage.copy()
    raise ValueError(f"Unsupported preview image shape: {array.shape!r}.")


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


def _policy_snapshot(state: object | None) -> object | None:
    if state is None:
        return None
    for method_name in ("snapshot", "get_snapshot"):
        method = getattr(state, method_name, None)
        if callable(method):
            return method()
    return None


def _camera_intrinsics_mode(state: object | None) -> str:
    snapshot = _policy_snapshot(state)
    value = _payload_value(snapshot, "camera_intrinsics_mode") or _payload_value(
        snapshot,
        "mode",
    )
    try:
        return normalize_camera_intrinsics_mode(
            value if value is not None else SUPPORTED_CAMERA_INTRINSICS_MODES[0]
        )
    except ValueError:
        return SUPPORTED_CAMERA_INTRINSICS_MODES[0]


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
