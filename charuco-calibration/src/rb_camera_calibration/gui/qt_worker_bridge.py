"""Qt worker bridge for camera, detection, capture, and calibration."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot

from rb_camera_calibration.calibration.solver import OpenCvCalibrationSolver
from rb_camera_calibration.camera.opencv_camera_source import OpenCvCameraSource
from rb_camera_calibration.capture.capture_controller import AutomaticCaptureController
from rb_camera_calibration.capture.capture_quality import OpenCvFrameQualityScorer
from rb_camera_calibration.capture.pose_diversity import SimplePoseDiversityTracker
from rb_camera_calibration.capture.session_store import CalibrationSessionStore
from rb_camera_calibration.contracts import (
    CalibrationRequest,
    CalibrationSessionConfig,
    CameraFrame,
    CharucoDetection,
    FrameQualityMetrics,
    WorkerState,
)
from rb_camera_calibration.detection.charuco_detector import OpenCvCharucoDetector
from rb_camera_calibration.detection.dictionary_probe import probe_image
from rb_camera_calibration.utils import opencv_compat as cvx

LOGGER = logging.getLogger(__name__)


class CameraPipelineWorker(QObject):
    """Camera/detection/capture worker that runs in a Qt thread."""

    state_changed = Signal(object)
    frame_ready = Signal(object)
    detection_ready = Signal(object)
    quality_ready = Signal(object)
    decision_ready = Signal(object)
    accepted_frame_ready = Signal(object)
    dictionary_probe_ready = Signal(object)
    board_dimension_debug_ready = Signal(object)
    log_message = Signal(str)
    error_message = Signal(str)

    def __init__(self, session_config: CalibrationSessionConfig) -> None:
        super().__init__()
        self.session_config = session_config
        self.camera = OpenCvCameraSource(session_config.camera_config)
        self.session_store = CalibrationSessionStore(session_config)
        self.quality_scorer = OpenCvFrameQualityScorer()
        self.pose_tracker = SimplePoseDiversityTracker(session_config.capture_policy)
        self.capture_controller = AutomaticCaptureController(
            session_config.capture_policy,
            self.pose_tracker,
        )
        self.detector: OpenCvCharucoDetector | None = None
        self._timer: QTimer | None = None
        self._auto_capture_enabled = False
        self._latest_frame: CameraFrame | None = None
        self._latest_detection: CharucoDetection | None = None
        self._latest_quality: FrameQualityMetrics | None = None
        self._detector_error: str | None = None
        self._last_rejected_reason: str | None = None
        self._rejected_sample_counter = 0
        self._consecutive_read_failures = 0
        self._configure_detector()

    @Slot()
    def start_pipeline(self) -> None:
        self.state_changed.emit(WorkerState.STARTING)
        try:
            self.session_store.initialise()
            for accepted in self.session_store.accepted_frames:
                self.pose_tracker.update_accepted(accepted.pose_signature)
            if self.session_store.accepted_frames:
                self.log_message.emit(
                    f"Loaded {len(self.session_store.accepted_frames)} existing accepted frames."
                )
            self.camera.start()
            self.log_message.emit(f"Camera opened: {self.camera.actual_properties}")
            if self.detector is None and self._detector_error is not None:
                self.log_message.emit(
                    "Live ChArUco detection is disabled until the board config is corrected: "
                    f"{self._detector_error}"
                )
            interval_ms = max(1, int(1000.0 / max(float(self.session_config.camera_config.fps), 1.0)))
            self._timer = QTimer(self)
            self._timer.setInterval(interval_ms)
            self._timer.timeout.connect(self._process_frame)
            self._timer.start()
            self.state_changed.emit(WorkerState.RUNNING)
        except Exception as exc:
            LOGGER.exception("Camera pipeline failed to start")
            self.error_message.emit(str(exc))
            self.state_changed.emit(WorkerState.ERROR)

    @Slot()
    def stop_pipeline(self) -> None:
        self.state_changed.emit(WorkerState.STOPPING)
        if self._timer is not None:
            self._timer.stop()
            self._timer.deleteLater()
            self._timer = None
        self.camera.stop()
        self.state_changed.emit(WorkerState.STOPPED)
        QThread.currentThread().quit()

    @Slot(bool)
    def set_auto_capture_enabled(self, enabled: bool) -> None:
        self._auto_capture_enabled = enabled
        self.log_message.emit(f"Auto-capture {'enabled' if enabled else 'disabled'}.")

    @Slot()
    def force_accept_current(self) -> None:
        if self._latest_frame is None or self._latest_detection is None or self._latest_quality is None:
            self.error_message.emit("No current detected frame to force accept.")
            return
        if not self._latest_detection.detected:
            self.error_message.emit("Cannot force accept: no ArUco markers are currently detected.")
            return
        if self._latest_detection.charuco_corner_count < 4:
            self.error_message.emit("Cannot force accept: fewer than 4 ChArUco corners are detected.")
            return
        try:
            decision = self.capture_controller.force_accept(self._latest_detection, self._latest_quality)
            accepted = self.session_store.store_accepted_frame(
                self._latest_frame,
                self._latest_detection,
                decision,
            )
            self.decision_ready.emit(decision)
            self.accepted_frame_ready.emit(accepted)
        except Exception as exc:
            LOGGER.exception("Force accept failed")
            self.error_message.emit(str(exc))

    @Slot(str)
    def remove_accepted_frame(self, frame_id: str) -> None:
        if self.session_store.remove_accepted_frame(frame_id):
            self.log_message.emit(f"Removed accepted frame from manifest: {frame_id}")

    @Slot()
    def reset_session_manifest(self) -> None:
        self.session_store.reset_manifest()
        self.log_message.emit("Worker session manifest reset.")

    @Slot()
    def probe_dictionary_current(self) -> None:
        if self._latest_frame is None:
            self.error_message.emit("No current frame available for dictionary probe.")
            return
        try:
            image = cvx.decode_image_bytes(self._latest_frame.image_bytes)
            report = probe_image(image, frame_id=self._latest_frame.frame_id)
            self.dictionary_probe_ready.emit(report)
        except Exception as exc:
            LOGGER.exception("Dictionary probe failed")
            self.error_message.emit(str(exc))

    @Slot()
    def debug_board_dimensions_current(self) -> None:
        if self.detector is None:
            self.error_message.emit("Board dimension debug needs an explicit valid aruco_dictionary.")
            return
        if self._latest_frame is None:
            self.error_message.emit("No current frame available for board dimension debug.")
            return
        try:
            report = self.detector.debug_reversed_dimensions(self._latest_frame)
            self.board_dimension_debug_ready.emit(report)
        except Exception as exc:
            LOGGER.exception("Board dimension debug failed")
            self.error_message.emit(str(exc))

    @Slot()
    def _process_frame(self) -> None:
        try:
            frame = self.camera.read_frame()
            if frame is None:
                self._consecutive_read_failures += 1
                if self._consecutive_read_failures >= 2:
                    self.error_message.emit(
                        "Camera did not return frames after repeated V4L2 reads. "
                        "Stopping camera; check USB bandwidth, cable, exposure, and camera mode."
                    )
                    self.stop_pipeline()
                return
            self._consecutive_read_failures = 0
            self._latest_frame = frame
            self.frame_ready.emit(frame)
            if self.detector is None:
                return
            detection = self.detector.detect(frame)
            quality = self.quality_scorer.score(frame)
            self._latest_detection = detection
            self._latest_quality = quality
            self.detection_ready.emit(detection)
            self.quality_ready.emit(quality)
            if not self._auto_capture_enabled:
                return
            decision = self.capture_controller.evaluate_frame(frame, detection, quality)
            self.decision_ready.emit(decision)
            if decision.accepted:
                accepted = self.session_store.store_accepted_frame(frame, detection, decision)
                self.accepted_frame_ready.emit(accepted)
            elif detection.detected and self._should_store_rejected_sample(decision):
                self.session_store.store_rejected_sample(frame, detection, decision)
        except Exception as exc:
            LOGGER.exception("Camera pipeline frame processing failed")
            self.error_message.emit(str(exc))
            self.stop_pipeline()
            self.state_changed.emit(WorkerState.ERROR)

    def _configure_detector(self) -> None:
        try:
            self.detector = OpenCvCharucoDetector(self.session_config.board_config)
            self._detector_error = None
        except Exception as exc:
            self.detector = None
            self._detector_error = str(exc)

    def _should_store_rejected_sample(self, decision) -> bool:
        if not self.session_config.save_rejected_samples:
            return False
        self._rejected_sample_counter += 1
        reason = decision.reason.value if decision.reason is not None else ""
        changed = reason != self._last_rejected_reason
        self._last_rejected_reason = reason
        return changed or self._rejected_sample_counter % 30 == 0


class CalibrationRunWorker(QObject):
    """Run calibration solving off the GUI thread."""

    result_ready = Signal(object)
    error_message = Signal(str)
    finished = Signal()

    def __init__(self, request: CalibrationRequest) -> None:
        super().__init__()
        self.request = request

    @Slot()
    def run(self) -> None:
        try:
            result = OpenCvCalibrationSolver().solve(self.request)
            self.result_ready.emit(result)
        except Exception as exc:
            LOGGER.exception("Calibration solve failed")
            self.error_message.emit(str(exc))
        finally:
            self.finished.emit()


def start_calibration_worker(request: CalibrationRequest) -> tuple[QThread, CalibrationRunWorker]:
    """Create and start a one-shot calibration worker thread."""
    thread = QThread()
    worker = CalibrationRunWorker(request)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker
