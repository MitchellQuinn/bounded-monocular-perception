"""Main PySide6 window for the ChArUco calibration application."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QThread

from rb_camera_calibration.calibration.artifact_export import CalibrationArtifactExporter
from rb_camera_calibration.capture.pose_diversity import SimplePoseDiversityTracker
from rb_camera_calibration.contracts import (
    CalibrationRequest,
    CalibrationResult,
    CalibrationSessionConfig,
    CameraFrame,
    CaptureDecision,
    CaptureRejectReason,
    CharucoDetection,
    DictionaryProbeReport,
    FrameQualityMetrics,
    WorkerState,
)
from rb_camera_calibration.gui.preview_widget import PreviewWidget
from rb_camera_calibration.gui.qt_worker_bridge import (
    CameraPipelineWorker,
    start_calibration_worker,
)
from rb_camera_calibration.capture.merge_sessions import (
    discover_merge_session_roots,
    merge_session_roots_into_store,
)
from rb_camera_calibration.capture.session_store import (
    CalibrationSessionStore,
    load_accepted_frames_from_manifest,
)


class MainWindow(QMainWindow):
    """Operator-facing calibration workflow window."""

    stop_requested = Signal()
    auto_capture_requested = Signal(bool)
    force_accept_requested = Signal()
    probe_dictionary_requested = Signal()
    board_debug_requested = Signal()
    remove_accepted_requested = Signal(str)
    reset_manifest_requested = Signal()

    def __init__(self, session_config: CalibrationSessionConfig) -> None:
        super().__init__()
        self.session_config = session_config
        self.camera_thread: QThread | None = None
        self.camera_worker: CameraPipelineWorker | None = None
        self.calibration_thread: QThread | None = None
        self.calibration_worker = None
        self.accepted_frames = []
        self.latest_frame: CameraFrame | None = None
        self.latest_result: CalibrationResult | None = None
        self._last_reject_reason: CaptureRejectReason | None = None

        self.setWindowTitle("Raccoon Ball ChArUco Calibration")
        self.resize(1320, 820)
        self._build_ui()
        self._install_shortcuts()
        self._load_existing_accepted_frames()
        self._update_session_labels()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        controls = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop")
        self.auto_capture = QCheckBox("Auto-capture")
        self.force_accept_button = QPushButton("Force Accept")
        self.remove_button = QPushButton("Remove Selected")
        self.merge_all_button = QPushButton("Merge All Runs")
        self.calibrate_button = QPushButton("Calibrate")
        self.export_button = QPushButton("Export Artifact")
        self.probe_button = QPushButton("Probe Dictionary")
        self.board_debug_button = QPushButton("Board Debug")

        for widget in (
            self.start_button,
            self.stop_button,
            self.auto_capture,
            self.force_accept_button,
            self.remove_button,
            self.merge_all_button,
            self.calibrate_button,
            self.export_button,
            self.probe_button,
            self.board_debug_button,
        ):
            controls.addWidget(widget)
        controls.addStretch(1)
        root.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.preview = PreviewWidget()
        splitter.addWidget(self.preview)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self._status_group())
        right_layout.addWidget(self._accepted_group())
        right_layout.addWidget(self._log_group())
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.auto_capture.toggled.connect(self.auto_capture_requested.emit)
        self.force_accept_button.clicked.connect(self.force_accept_requested.emit)
        self.remove_button.clicked.connect(self.remove_selected_accepted)
        self.merge_all_button.clicked.connect(self.merge_all_runs)
        self.calibrate_button.clicked.connect(self.calibrate)
        self.export_button.clicked.connect(self.export_artifact)
        self.probe_button.clicked.connect(self.probe_dictionary_requested.emit)
        self.board_debug_button.clicked.connect(self.board_debug_requested.emit)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(False)

    def _status_group(self) -> QGroupBox:
        group = QGroupBox("Status")
        grid = QGridLayout(group)
        self.worker_state_label = QLabel("STOPPED")
        self.session_dir_label = QLabel("")
        self.accepted_count_label = QLabel("0")
        self.decision_label = QLabel("")
        self.reject_label = QLabel("")
        self.marker_count_label = QLabel("0")
        self.charuco_count_label = QLabel("0")
        self.quality_label = QLabel("")
        self.pose_label = QLabel("")
        self.coverage_label = QLabel("")
        rows = [
            ("Worker", self.worker_state_label),
            ("Session", self.session_dir_label),
            ("Accepted", self.accepted_count_label),
            ("Decision", self.decision_label),
            ("Reject", self.reject_label),
            ("Markers", self.marker_count_label),
            ("ChArUco", self.charuco_count_label),
            ("Quality", self.quality_label),
            ("Pose", self.pose_label),
            ("Coverage", self.coverage_label),
        ]
        for row, (name, widget) in enumerate(rows):
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(widget, row, 1)
        return group

    def _accepted_group(self) -> QGroupBox:
        group = QGroupBox("Accepted Frames")
        layout = QVBoxLayout(group)
        self.accepted_table = QTableWidget(0, 5)
        self.accepted_table.setHorizontalHeaderLabels(["Frame", "Corners", "Markers", "Cell", "RMS"])
        self.accepted_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.accepted_table, 1)
        self.error_list = QListWidget()
        layout.addWidget(QLabel("Per-view errors"))
        layout.addWidget(self.error_list, 1)
        return group

    def _log_group(self) -> QGroupBox:
        group = QGroupBox("Log")
        layout = QVBoxLayout(group)
        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        layout.addWidget(self.log_panel)
        return group

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.force_accept_requested.emit)
        QShortcut(QKeySequence("C"), self, activated=self.calibrate)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self.remove_selected_accepted)
        QShortcut(QKeySequence("R"), self, activated=self.reset_session_after_confirmation)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self.stop_camera)

    def _update_session_labels(self) -> None:
        target = self.session_config.capture_policy.target_accepted_frame_count
        self.session_dir_label.setText(str(self.session_config.session_root))
        self.accepted_count_label.setText(f"{len(self.accepted_frames)} / {target}")

    def _load_existing_accepted_frames(self) -> None:
        self.accepted_frames.clear()
        self.accepted_table.setRowCount(0)
        manifest_path = Path(self.session_config.session_root) / "session_manifest.json"
        if not manifest_path.exists():
            return
        try:
            frames = load_accepted_frames_from_manifest(manifest_path)
        except Exception as exc:
            self.append_error(f"Could not load existing session manifest: {exc}")
            return
        for accepted in frames:
            self._append_accepted_frame_to_table(accepted)
        if frames:
            self.append_log(f"Loaded {len(frames)} accepted frames from existing session.")
        self._refresh_coverage_from_accepted_frames()

    @Slot()
    def merge_all_runs(self) -> None:
        if self.camera_thread is not None:
            QMessageBox.warning(
                self,
                "Merge All Runs",
                "Stop the camera before merging runs.",
            )
            return
        if self.calibration_thread is not None:
            QMessageBox.warning(
                self,
                "Merge All Runs",
                "Wait for calibration to finish before merging runs.",
            )
            return
        source_roots = discover_merge_session_roots(Path(self.session_config.session_root))
        if not source_roots:
            self.append_log("Merge All Runs found no other session manifests.")
            return
        answer = QMessageBox.question(
            self,
            "Merge All Runs",
            "Merge accepted frames from "
            f"{len(source_roots)} existing run folder(s) into the current session?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            store = CalibrationSessionStore(self.session_config)
            merged_count, manifest_path = merge_session_roots_into_store(store, source_roots)
        except Exception as exc:
            self.append_error(f"Merge All Runs failed: {exc}")
            QMessageBox.critical(self, "Merge All Runs", str(exc))
            return
        self._load_existing_accepted_frames()
        self._update_session_labels()
        self._refresh_coverage_from_accepted_frames()
        self.latest_result = None
        self.export_button.setEnabled(False)
        self.error_list.clear()
        self.append_log(
            f"Merge All Runs copied {merged_count} accepted frame(s) into {manifest_path}."
        )

    def _refresh_coverage_from_accepted_frames(self) -> None:
        if not self.accepted_frames:
            self.coverage_label.setText("")
            return
        tracker = SimplePoseDiversityTracker(self.session_config.capture_policy)
        coverage = None
        for accepted in self.accepted_frames:
            coverage = tracker.update_accepted(accepted.pose_signature)
        if coverage is None:
            self.coverage_label.setText("")
            return
        self.coverage_label.setText(
            f"{coverage.coverage_score:.2f}; {coverage.suggested_next_pose}"
        )

    @Slot()
    def start_camera(self) -> None:
        if self.camera_thread is not None:
            return
        self.camera_thread = QThread(self)
        self.camera_worker = CameraPipelineWorker(self.session_config)
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.start_pipeline)
        self.stop_requested.connect(self.camera_worker.stop_pipeline)
        self.auto_capture_requested.connect(self.camera_worker.set_auto_capture_enabled)
        self.force_accept_requested.connect(self.camera_worker.force_accept_current)
        self.probe_dictionary_requested.connect(self.camera_worker.probe_dictionary_current)
        self.board_debug_requested.connect(self.camera_worker.debug_board_dimensions_current)
        self.remove_accepted_requested.connect(self.camera_worker.remove_accepted_frame)
        self.reset_manifest_requested.connect(self.camera_worker.reset_session_manifest)
        self.camera_worker.state_changed.connect(self.on_worker_state_changed)
        self.camera_worker.frame_ready.connect(self.on_frame_ready)
        self.camera_worker.detection_ready.connect(self.on_detection_ready)
        self.camera_worker.quality_ready.connect(self.on_quality_ready)
        self.camera_worker.decision_ready.connect(self.on_decision_ready)
        self.camera_worker.accepted_frame_ready.connect(self.on_accepted_frame_ready)
        self.camera_worker.dictionary_probe_ready.connect(self.on_dictionary_probe_ready)
        self.camera_worker.board_dimension_debug_ready.connect(self.on_board_dimension_debug_ready)
        self.camera_worker.log_message.connect(self.append_log)
        self.camera_worker.error_message.connect(self.append_error)
        self.camera_thread.finished.connect(self._camera_thread_finished)
        self.camera_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    @Slot()
    def stop_camera(self) -> None:
        if self.camera_worker is None:
            return
        self.auto_capture.setChecked(False)
        self.stop_requested.emit()
        self.append_log("Camera stop requested.")

    @Slot(object)
    def on_worker_state_changed(self, state: WorkerState) -> None:
        text = state.value if hasattr(state, "value") else str(state)
        self.worker_state_label.setText(text)
        if state == WorkerState.STOPPED and self.camera_thread is not None:
            self.camera_thread.quit()

    @Slot(object)
    def on_frame_ready(self, frame: CameraFrame) -> None:
        self.latest_frame = frame
        self.preview.set_frame(frame)

    @Slot(object)
    def on_detection_ready(self, detection: CharucoDetection) -> None:
        self.preview.set_detection(detection)
        self.marker_count_label.setText(str(detection.marker_count))
        self.charuco_count_label.setText(str(detection.charuco_corner_count))

    @Slot(object)
    def on_quality_ready(self, quality: FrameQualityMetrics) -> None:
        self.quality_label.setText(
            f"lap={quality.laplacian_variance:.1f}, luma={quality.mean_luma:.1f}, "
            f"clip={quality.clipped_black_fraction:.2f}/{quality.clipped_white_fraction:.2f}"
        )

    @Slot(object)
    def on_decision_ready(self, decision: CaptureDecision) -> None:
        self.decision_label.setText(decision.decision_type.value)
        self.reject_label.setText(decision.reason.value if decision.reason else "")
        if decision.pose_signature is not None:
            self.pose_label.setText(
                f"cell={decision.pose_signature.grid_cell}, "
                f"scale={decision.pose_signature.scale_bin}, tilt={decision.pose_signature.tilt_bin}"
            )
        if decision.coverage_state is not None:
            self.coverage_label.setText(
                f"{decision.coverage_state.coverage_score:.2f}; "
                f"{decision.coverage_state.suggested_next_pose}"
            )
        if decision.accepted:
            QApplication.beep()
            self._last_reject_reason = None
            target = self.session_config.capture_policy.target_accepted_frame_count
            if decision.coverage_state and decision.coverage_state.accepted_count >= target:
                QApplication.beep()
                QApplication.beep()
        elif decision.detection.detected and decision.reason != self._last_reject_reason:
            self._last_reject_reason = decision.reason
            QApplication.beep()

    @Slot(object)
    def on_accepted_frame_ready(self, accepted) -> None:
        self._append_accepted_frame_to_table(accepted)
        self._update_session_labels()
        self.append_log(f"Accepted {accepted.frame_id} -> {accepted.image_path}")

    def _append_accepted_frame_to_table(self, accepted) -> None:
        self.accepted_frames.append(accepted)
        row = self.accepted_table.rowCount()
        self.accepted_table.insertRow(row)
        self.accepted_table.setItem(row, 0, QTableWidgetItem(accepted.frame_id))
        self.accepted_table.setItem(row, 1, QTableWidgetItem(str(accepted.charuco_corner_count)))
        self.accepted_table.setItem(row, 2, QTableWidgetItem(str(accepted.marker_count)))
        self.accepted_table.setItem(row, 3, QTableWidgetItem(str(accepted.pose_signature.grid_cell)))
        self.accepted_table.setItem(row, 4, QTableWidgetItem(""))

    @Slot(object)
    def on_dictionary_probe_ready(self, report: DictionaryProbeReport) -> None:
        if report.best_candidate is None:
            self.append_log("Dictionary probe found no likely dictionary.")
        else:
            best = report.best_candidate
            self.append_log(
                f"Dictionary probe best: {best.dictionary_name} "
                f"markers={best.marker_count} ids={list(best.marker_ids)}"
            )
        for candidate in report.candidates[:5]:
            self.append_log(
                f"Probe {candidate.dictionary_name}: markers={candidate.marker_count}, "
                f"ids={list(candidate.marker_ids)}, confidence={candidate.confidence.value}"
            )
        for warning in report.warning_messages:
            self.append_log(f"Probe warning: {warning}")

    @Slot(object)
    def on_board_dimension_debug_ready(self, report) -> None:
        self.append_log(
            "Board debug: "
            f"configured {report.configured_squares_xy} markers={report.configured_marker_count} "
            f"charuco={report.configured_charuco_corner_count}; "
            f"reversed {report.reversed_squares_xy} markers={report.reversed_marker_count} "
            f"charuco={report.reversed_charuco_corner_count}; "
            f"reversed_more_plausible={report.reversing_appears_more_plausible}"
        )
        self.append_log(report.message)

    @Slot()
    def remove_selected_accepted(self) -> None:
        row = self.accepted_table.currentRow()
        if row < 0 or row >= len(self.accepted_frames):
            return
        accepted = self.accepted_frames.pop(row)
        self.accepted_table.removeRow(row)
        self.remove_accepted_requested.emit(accepted.frame_id)
        self._update_session_labels()

    @Slot()
    def calibrate(self) -> None:
        if not self.accepted_frames:
            self.append_error("No accepted frames are available for calibration.")
            return
        if self.calibration_thread is not None:
            self.append_log("Calibration is already running.")
            return
        if self.latest_frame is not None:
            image_size = (self.latest_frame.metadata.width_px, self.latest_frame.metadata.height_px)
        else:
            image_size = (
                self.session_config.camera_config.width_px,
                self.session_config.camera_config.height_px,
            )
        request = CalibrationRequest(
            session_root=Path(self.session_config.session_root),
            board_config=self.session_config.board_config,
            image_size_wh_px=image_size,
            accepted_frames=tuple(self.accepted_frames),
            calibration_flags=(),
        )
        self.append_log("Calibration started.")
        self.calibration_thread, self.calibration_worker = start_calibration_worker(request)
        self.calibration_worker.result_ready.connect(self.on_calibration_result)
        self.calibration_worker.error_message.connect(self.append_error)
        self.calibration_worker.finished.connect(
            lambda: self.append_log("Calibration worker finished; waiting for thread shutdown.")
        )
        self.calibration_thread.finished.connect(self._calibration_thread_finished)

    @Slot(object)
    def on_calibration_result(self, result: CalibrationResult) -> None:
        self.latest_result = result
        self.export_button.setEnabled(result.success)
        self.error_list.clear()
        for error in result.per_view_errors:
            self.error_list.addItem(
                QListWidgetItem(
                    f"{error.frame_id}: rms={error.rms_error_px:.3f}px "
                    f"mean={error.mean_error_px:.3f}px max={error.max_error_px:.3f}px"
                )
            )
        for row in range(self.accepted_table.rowCount()):
            frame_id = self.accepted_table.item(row, 0).text()
            match = next((err for err in result.per_view_errors if err.frame_id == frame_id), None)
            if match is not None:
                self.accepted_table.setItem(row, 4, QTableWidgetItem(f"{match.rms_error_px:.3f}"))
        if result.success:
            self.append_log(
                f"Calibration complete: RMS={result.rms_reprojection_error_px:.4f}px, "
                f"used={result.used_frame_count}/{result.accepted_frame_count}"
            )
        else:
            self.append_error(f"Calibration failed: {result.extras.get('error', 'unknown error')}")

    @Slot()
    def export_artifact(self) -> None:
        if self.latest_result is None:
            self.append_error("Run calibration before exporting an artifact.")
            return
        manifest = CalibrationArtifactExporter().export(self.latest_result, self.session_config)
        self.append_log(f"Exported artifact: {manifest.result_json_path}")

    @Slot()
    def reset_session_after_confirmation(self) -> None:
        answer = QMessageBox.question(
            self,
            "Reset Session",
            "Reset the current in-memory session list?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.accepted_frames.clear()
        self.accepted_table.setRowCount(0)
        self.error_list.clear()
        self.latest_result = None
        self.export_button.setEnabled(False)
        self.reset_manifest_requested.emit()
        self._update_session_labels()
        self.append_log("Session list reset. Existing files remain on disk.")

    @Slot(str)
    def append_log(self, message: str) -> None:
        self.log_panel.append(message)

    @Slot(str)
    def append_error(self, message: str) -> None:
        self.log_panel.append(f"ERROR: {message}")

    @Slot()
    def _camera_thread_finished(self) -> None:
        self.camera_worker = None
        self.camera_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    @Slot()
    def _calibration_thread_finished(self) -> None:
        self.calibration_thread = None
        self.calibration_worker = None
        self.append_log("Calibration thread stopped.")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._wait_for_calibration_thread_for_shutdown()
        self._stop_camera_thread_for_shutdown()
        super().closeEvent(event)

    def _wait_for_calibration_thread_for_shutdown(self) -> None:
        """Avoid destroying a still-running calibration QThread on close."""
        thread = self.calibration_thread
        if thread is None:
            return
        self.append_log("Waiting for calibration to finish before closing.")
        if thread.wait(30_000):
            return
        self.append_error("Calibration did not finish after 30 seconds; forcing shutdown.")
        thread.terminate()
        thread.wait(2_000)

    def _stop_camera_thread_for_shutdown(self) -> None:
        """Stop the camera worker without letting Qt destroy a running thread.

        OpenCV/V4L2 reads can block until the kernel reports ``select()``
        timeout, commonly around ten seconds.  Closing the window must wait long
        enough for that read to unwind and for the worker's queued stop request
        to release the camera.
        """
        thread = self.camera_thread
        if thread is None:
            return
        self.stop_camera()
        if thread.wait(15_000):
            return
        self.append_error("Camera worker did not stop after 15 seconds; forcing shutdown.")
        if self.camera_worker is not None:
            try:
                self.camera_worker.camera.stop()
            except Exception as exc:  # pragma: no cover - shutdown best effort
                self.append_error(f"Forced camera release failed: {exc}")
        thread.quit()
        if thread.wait(2_000):
            return
        thread.terminate()
        thread.wait(2_000)
