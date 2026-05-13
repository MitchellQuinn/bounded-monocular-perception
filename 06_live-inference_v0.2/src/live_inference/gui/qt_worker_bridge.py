"""PySide6 bridge for running plain Python workers in a QThread."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot


class WorkerQtSignals(QObject):
    """Qt signal collection for generic live worker contract events."""

    status_changed = Signal(object)
    lifecycle_event = Signal(object)
    warning_occurred = Signal(object)
    error_occurred = Signal(object)
    frame_written = Signal(object)
    frame_skipped = Signal(object)
    result_ready = Signal(object)
    debug_image_ready = Signal(object)
    runtime_parameters_available = Signal(object)
    runtime_parameter_update_result = Signal(object)


class QtWorkerEventSink:
    """Plain Python event sink that forwards worker callbacks to Qt signals."""

    def __init__(self, signals: WorkerQtSignals) -> None:
        self.signals = signals

    def status_changed(self, status: object) -> None:
        self.signals.status_changed.emit(status)

    def lifecycle_event(self, event: object) -> None:
        self.signals.lifecycle_event.emit(event)

    def warning_occurred(self, warning: object) -> None:
        self.signals.warning_occurred.emit(warning)

    def error_occurred(self, error: object) -> None:
        self.signals.error_occurred.emit(error)

    def frame_written(self, frame: object) -> None:
        self.signals.frame_written.emit(frame)

    def frame_skipped(self, skipped: object) -> None:
        self.signals.frame_skipped.emit(skipped)

    def result_ready(self, result: object) -> None:
        self.signals.result_ready.emit(result)

    def debug_image_ready(self, image: object) -> None:
        self.signals.debug_image_ready.emit(image)

    def runtime_parameters_available(self, spec: object) -> None:
        self.signals.runtime_parameters_available.emit(spec)

    def runtime_parameter_update_result(self, result: object) -> None:
        self.signals.runtime_parameter_update_result.emit(result)


class WorkerThreadController:
    """Own a QThread container for one plain Python worker."""

    def __init__(
        self,
        worker: object,
        *,
        signals: WorkerQtSignals | None = None,
        event_sink: QtWorkerEventSink | None = None,
        install_event_sink: bool = True,
    ) -> None:
        _validate_worker(worker)

        self.worker = worker
        self.signals = signals or WorkerQtSignals()
        self.event_sink = event_sink or QtWorkerEventSink(self.signals)
        self._last_exception: BaseException | None = None

        if install_event_sink:
            _install_event_sink_if_supported(self.worker, self.event_sink)

        self._thread = QThread()
        self._runner = _WorkerRunner(self.worker)
        self._runner.moveToThread(self._thread)

        self._thread.started.connect(self._runner.run)
        self._runner.finished.connect(self._thread.quit)
        self._runner.failed.connect(self._record_runner_exception)
        self.signals.status_changed.connect(self._quit_thread_if_stopped_payload)
        self.signals.lifecycle_event.connect(self._quit_thread_if_stopped_payload)

    def start(self) -> None:
        """Start the worker thread if it is not already running."""
        if self._thread.isRunning():
            return
        self._last_exception = None
        self._thread.start()

    def request_stop(self) -> None:
        """Ask the plain Python worker to stop without terminating the thread."""
        self.worker.request_stop()

    def is_running(self) -> bool:
        return self._thread.isRunning()

    def wait(self, timeout_ms: int | None = None) -> bool:
        """Wait for the QThread to finish."""
        if timeout_ms is None:
            return self._thread.wait()
        return self._thread.wait(timeout_ms)

    @property
    def last_exception(self) -> BaseException | None:
        return self._last_exception

    def _record_runner_exception(self, exc: object) -> None:
        if isinstance(exc, BaseException):
            self._last_exception = exc

    def _quit_thread_if_stopped_payload(self, payload: object) -> None:
        if _payload_indicates_stopped(payload):
            self._thread.quit()


class _WorkerRunner(QObject):
    finished = Signal()
    failed = Signal(object)

    def __init__(self, worker: object) -> None:
        super().__init__()
        self._worker = worker

    @Slot()
    def run(self) -> None:
        try:
            self._worker.start_work()
        except Exception as exc:
            self.failed.emit(exc)
        finally:
            self.finished.emit()
            QThread.currentThread().quit()


def _validate_worker(worker: object) -> None:
    required_methods = (
        "start_work",
        "request_stop",
        "current_status",
        "current_counters",
    )
    for method_name in required_methods:
        if not callable(getattr(worker, method_name, None)):
            raise TypeError(f"worker must provide callable {method_name}().")


def _install_event_sink_if_supported(
    worker: object,
    event_sink: QtWorkerEventSink,
) -> None:
    set_event_sink = getattr(worker, "set_event_sink", None)
    if callable(set_event_sink):
        set_event_sink(event_sink)
        return

    if hasattr(worker, "event_sink"):
        setattr(worker, "event_sink", event_sink)
        return

    if hasattr(worker, "_event_sink"):
        setattr(worker, "_event_sink", event_sink)


def _payload_indicates_stopped(payload: object) -> bool:
    state = _enum_or_raw_value(getattr(payload, "state", None))
    if _matches_stopped(state):
        return True

    event_type = _enum_or_raw_value(getattr(payload, "event_type", None))
    return _matches_stopped(event_type)


def _enum_or_raw_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _matches_stopped(value: object) -> bool:
    if not isinstance(value, str):
        return False
    return value.lower() == "stopped"


__all__ = [
    "QtWorkerEventSink",
    "WorkerQtSignals",
    "WorkerThreadController",
]
