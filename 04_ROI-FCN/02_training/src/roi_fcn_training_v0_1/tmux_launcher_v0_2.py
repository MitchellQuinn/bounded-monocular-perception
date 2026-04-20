"""tmux/process helpers for the ROI-FCN training control panels."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping, Sequence

from .config import TrainConfig
from .contracts import RESUME_STATE_FILENAME, RUN_CONFIG_FILENAME
from .paths import (
    build_model_run_dir_path,
    build_runs_root_path,
    find_training_root,
    preview_next_run_id,
    resolve_models_root,
    suggest_model_run_id,
)
from .resume_state import load_resume_state
from .utils import read_json

TMUX_CONTROL_PANEL_BUILD_V02 = "2026-04-19-roi-fcn-tmux-v0.2"
TMUX_CONTROL_PANEL_BUILD_V03 = "2026-04-20-roi-fcn-tmux-v0.3"
DEFAULT_TMUX_LOG_FILENAME = "train.log"
TRAINING_MODULE_NAME = "roi_fcn_training_v0_1.train"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_RUN_ID_RE = re.compile(r"^run_([0-9]+)$")


def _require_identifier(value: str, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError(f"{label} must match {_IDENTIFIER_RE.pattern!r}; got {text!r}.")
    return text


def _require_log_filename(log_filename: str) -> str:
    text = str(log_filename).strip()
    if not text:
        raise ValueError("log_filename cannot be empty.")
    if Path(text).name != text:
        raise ValueError(f"log_filename must be a filename only; got {log_filename!r}")
    return text


def _run_sort_index(name: str) -> int:
    match = _RUN_ID_RE.fullmatch(str(name).strip())
    if match is None:
        return -1
    return int(match.group(1))


def _run_tmux(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["tmux", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("tmux executable not found on PATH.") from exc


def _is_no_server_error(stderr_text: str) -> bool:
    text = str(stderr_text).lower()
    return (
        "no server running" in text
        or "failed to connect to server" in text
        or "can't find socket" in text
        or ("error connecting to" in text and "no such file or directory" in text)
    )


def _is_missing_session_error(stderr_text: str) -> bool:
    text = str(stderr_text).lower()
    return (
        "can't find session" in text
        or "session not found" in text
        or "unknown session" in text
    )


def _require_run_dir(path: str | Path) -> Path:
    run_dir = Path(path).expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    run_config_path = run_dir / RUN_CONFIG_FILENAME
    if not run_config_path.exists():
        raise FileNotFoundError(f"Run config not found: {run_config_path}")
    resume_state_path = run_dir / RESUME_STATE_FILENAME
    if not resume_state_path.exists():
        raise FileNotFoundError(f"Resume state not found: {resume_state_path}")
    return run_dir


def _source_model_directory(source_run_dir: Path, source_config: dict[str, Any]) -> str:
    text = str(source_config.get("model_directory", "")).strip()
    if text:
        return _require_identifier(text, label="model_directory")
    return _require_identifier(source_run_dir.parent.parent.name, label="model_directory")


def _resume_train_config_from_source(
    source_run_dir: Path,
    *,
    source_config: dict[str, Any],
    model_directory: str,
    run_id: str,
    additional_epochs: int,
    device_override: str | None = None,
) -> TrainConfig:
    training_dataset = str(source_config.get("training_dataset", "")).strip()
    if not training_dataset:
        raise ValueError(f"Source run config is missing training_dataset: {source_run_dir / RUN_CONFIG_FILENAME}")
    validation_dataset = str(source_config.get("validation_dataset") or training_dataset).strip()
    topology_params = source_config.get("topology_params")
    if not isinstance(topology_params, dict):
        topology_params = {}
    resolved_device = str(device_override or source_config.get("device", "")).strip() or None
    return TrainConfig(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset or None,
        datasets_root=source_config.get("datasets_root", "datasets"),
        models_root=source_config.get("models_root", "models"),
        seed=int(source_config.get("seed", 42)),
        batch_size=int(source_config.get("batch_size", 16)),
        epochs=int(source_config.get("epochs", 8)),
        learning_rate=float(source_config.get("learning_rate", 1e-3)),
        weight_decay=float(source_config.get("weight_decay", 1e-5)),
        gaussian_sigma_px=float(source_config.get("gaussian_sigma_px", 2.5)),
        heatmap_positive_threshold=float(source_config.get("heatmap_positive_threshold", 0.05)),
        early_stopping_patience=int(source_config.get("early_stopping_patience", 4)),
        topology_id=str(source_config.get("topology_id", "roi_fcn_tiny")).strip() or "roi_fcn_tiny",
        topology_variant=str(source_config.get("topology_variant", "tiny_v1")).strip() or "tiny_v1",
        topology_params=dict(topology_params),
        model_name=str(source_config.get("model_name") or source_config.get("model_directory") or model_directory).strip() or "roi-fcn-tiny",
        model_directory=model_directory,
        run_id=run_id,
        device=resolved_device,
        progress_log_interval_steps=int(source_config.get("progress_log_interval_steps", 50)),
        roi_width_px=int(source_config.get("roi_width_px", 300)),
        roi_height_px=int(source_config.get("roi_height_px", 300)),
        evaluation_max_visual_examples=int(source_config.get("evaluation_max_visual_examples", 12)),
        resume_from_run_dir=str(source_run_dir),
        additional_epochs=int(additional_epochs),
    )


@dataclass(frozen=True)
class TmuxTrainingLaunchPlanV02:
    """Structured launch preview for one detached training session."""

    session_name: str
    model_name: str
    model_directory: str
    run_id: str
    run_dir: str
    log_path: str
    working_directory: str
    command: str
    python_executable: str
    training_module: str = TRAINING_MODULE_NAME

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_sessions() -> list[str]:
    """Return active tmux session names."""
    result = _run_tmux(["list-sessions", "-F", "#S"])
    if result.returncode != 0:
        if _is_no_server_error(result.stderr):
            return []
        raise RuntimeError(f"tmux list-sessions failed: {result.stderr.strip()}")
    sessions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return sorted(set(sessions))


def session_exists(session_name: str) -> bool:
    """Return True when the tmux session already exists."""
    return _require_identifier(session_name, label="session_name") in list_sessions()


def default_session_name(model_directory: str, run_id: str) -> str:
    """Build the default tmux session name for a training run."""
    parsed_model_directory = _require_identifier(model_directory, label="model_directory")
    parsed_run_id = _require_identifier(run_id, label="run_id")
    return _require_identifier(f"roi_fcn_{parsed_model_directory}_{parsed_run_id}", label="session_name")


def list_model_directories(models_root_path: str | Path) -> list[str]:
    """Return model directories that contain a runs/ subdirectory."""
    root = Path(models_root_path).expanduser().resolve()
    if not root.exists():
        return []
    names: list[str] = []
    for child in root.iterdir():
        if child.is_dir() and (child / "runs").is_dir():
            names.append(child.name)
    return sorted(names)


def build_tmux_log_path(
    models_root_path: str | Path,
    *,
    model_directory: str,
    run_id: str,
    log_filename: str = DEFAULT_TMUX_LOG_FILENAME,
) -> Path:
    """Return the detached training log path under the predicted run directory."""
    model_dir_name = _require_identifier(model_directory, label="model_directory")
    run_name = _require_identifier(run_id, label="run_id")
    resolved_log_filename = _require_log_filename(log_filename)
    run_dir = build_model_run_dir_path(
        Path(models_root_path).expanduser().resolve(),
        model_directory=model_dir_name,
        run_id=run_name,
    )
    return run_dir / resolved_log_filename


def list_resume_candidates(models_root_path: str | Path, *, model_directory: str) -> list[dict[str, Any]]:
    """List resumable ROI-FCN runs under one model directory."""
    models_root = Path(models_root_path).expanduser().resolve()
    model_dir_name = _require_identifier(model_directory, label="model_directory")
    runs_root = build_runs_root_path(models_root, model_directory=model_dir_name)
    if not runs_root.exists() or not runs_root.is_dir():
        return []

    records: list[dict[str, Any]] = []
    for run_dir in sorted(runs_root.iterdir(), key=lambda path: (_run_sort_index(path.name), path.name)):
        if not run_dir.is_dir():
            continue
        run_config_path = run_dir / RUN_CONFIG_FILENAME
        resume_state_path = run_dir / RESUME_STATE_FILENAME
        latest_checkpoint_path = run_dir / "latest.pt"
        source_config: dict[str, Any] | None = None
        completed_epochs: int | None = None
        state_error: str | None = None
        planned_epochs: int | None = None
        if run_config_path.exists():
            try:
                source_config = read_json(run_config_path)
                planned_epochs = int(source_config.get("epochs")) if source_config.get("epochs") is not None else None
            except Exception as exc:
                state_error = f"run config read failed: {exc}"
        else:
            state_error = f"missing {RUN_CONFIG_FILENAME}"
        if resume_state_path.exists() and state_error is None:
            try:
                state_payload = load_resume_state(resume_state_path, map_location="cpu")
                completed_epochs = int(state_payload.get("epoch"))
            except Exception as exc:
                state_error = f"resume state read failed: {exc}"
        elif state_error is None:
            state_error = f"missing {RESUME_STATE_FILENAME}"

        remaining_epochs = None
        is_complete = None
        if planned_epochs is not None and completed_epochs is not None:
            remaining_epochs = max(int(planned_epochs) - int(completed_epochs), 0)
            is_complete = remaining_epochs == 0

        updated_at = None
        if resume_state_path.exists():
            updated_at = datetime.fromtimestamp(resume_state_path.stat().st_mtime).isoformat(timespec="seconds")
        elif latest_checkpoint_path.exists():
            updated_at = datetime.fromtimestamp(latest_checkpoint_path.stat().st_mtime).isoformat(timespec="seconds")

        model_name = ""
        topology_id = ""
        topology_variant = ""
        training_dataset = ""
        validation_dataset = ""
        if source_config is not None:
            model_name = str(source_config.get("model_name", "")).strip()
            topology_id = str(source_config.get("topology_id", "")).strip()
            topology_variant = str(source_config.get("topology_variant", "")).strip()
            training_dataset = str(source_config.get("training_dataset", "")).strip()
            validation_dataset = str(source_config.get("validation_dataset") or training_dataset).strip()

        records.append(
            {
                "run_id": str(run_dir.name),
                "run_dir": str(run_dir.resolve()),
                "run_config_path": str(run_config_path.resolve()),
                "resume_state_path": str(resume_state_path.resolve()),
                "latest_checkpoint_path": str(latest_checkpoint_path.resolve()),
                "completed_epochs": completed_epochs,
                "planned_epochs": planned_epochs,
                "remaining_epochs": remaining_epochs,
                "is_complete": is_complete,
                "is_resumable": bool(run_config_path.exists() and resume_state_path.exists() and state_error is None),
                "state_error": state_error,
                "updated_at": updated_at,
                "model_name": model_name,
                "topology_id": topology_id,
                "topology_variant": topology_variant,
                "training_dataset": training_dataset,
                "validation_dataset": validation_dataset,
            }
        )
    return records


def latest_resume_candidate(models_root_path: str | Path, *, model_directory: str) -> dict[str, Any] | None:
    """Return the latest resumable run for one model directory."""
    candidates = [row for row in list_resume_candidates(models_root_path, model_directory=model_directory) if row.get("is_resumable")]
    if not candidates:
        return None
    return candidates[-1]


def build_training_launch_command(
    config: TrainConfig | Mapping[str, Any],
    *,
    python_executable: str,
    model_name: str | None = None,
    model_directory: str | None = None,
    run_id: str | None = None,
) -> str:
    """Build a shell-safe detached training command."""
    train_config = config if isinstance(config, TrainConfig) else TrainConfig.from_mapping(dict(config))

    training_dataset = str(train_config.training_dataset or "").strip()
    if not training_dataset:
        raise ValueError("training_dataset cannot be blank.")

    model_text = _require_identifier(model_name or train_config.model_name or "roi-fcn-tiny", label="model_name")
    model_dir_name = _require_identifier(
        model_directory or train_config.model_directory or suggest_model_run_id(model_text, run_name_suffix=train_config.run_name_suffix),
        label="model_directory",
    )
    run_name = _require_identifier(run_id or train_config.run_id or "run_0001", label="run_id")
    python_bin = str(Path(python_executable).expanduser())
    if not python_bin:
        raise ValueError("python_executable cannot be empty.")

    args = [
        python_bin,
        "-u",
        "-m",
        TRAINING_MODULE_NAME,
        "--training-dataset",
        training_dataset,
        "--validation-dataset",
        str(train_config.validation_dataset or "").strip() or training_dataset,
        "--datasets-root",
        str(train_config.datasets_root),
        "--models-root",
        str(train_config.models_root),
        "--seed",
        str(int(train_config.seed)),
        "--batch-size",
        str(int(train_config.batch_size)),
        "--epochs",
        str(int(train_config.epochs)),
        "--learning-rate",
        str(float(train_config.learning_rate)),
        "--weight-decay",
        str(float(train_config.weight_decay)),
        "--gaussian-sigma-px",
        str(float(train_config.gaussian_sigma_px)),
        "--heatmap-positive-threshold",
        str(float(train_config.heatmap_positive_threshold)),
        "--early-stopping-patience",
        str(int(train_config.early_stopping_patience)),
        "--topology-id",
        str(train_config.topology_id),
        "--topology-variant",
        str(train_config.topology_variant),
        "--model-name",
        model_text,
        "--model-directory",
        model_dir_name,
        "--run-id",
        run_name,
        "--progress-log-interval-steps",
        str(int(train_config.progress_log_interval_steps)),
        "--roi-width-px",
        str(int(train_config.roi_width_px)),
        "--roi-height-px",
        str(int(train_config.roi_height_px)),
        "--evaluation-max-visual-examples",
        str(int(train_config.evaluation_max_visual_examples)),
    ]

    device_text = str(train_config.device or "").strip()
    if device_text:
        args.extend(["--device", device_text])

    resume_source = str(train_config.resume_from_run_dir or "").strip()
    if resume_source:
        if train_config.additional_epochs is None:
            raise ValueError("additional_epochs must be set when resume_from_run_dir is provided.")
        args.extend(["--resume-from-run-dir", resume_source])
        args.extend(["--additional-epochs", str(int(train_config.additional_epochs))])
    elif train_config.additional_epochs is not None:
        raise ValueError("additional_epochs cannot be used without resume_from_run_dir.")

    return shlex.join(args)


def plan_tmux_training_launch(
    training_root: str | Path,
    config: TrainConfig | Mapping[str, Any],
    *,
    python_executable: str,
    session_name: str | None = None,
    log_filename: str = DEFAULT_TMUX_LOG_FILENAME,
) -> TmuxTrainingLaunchPlanV02:
    """Validate config and derive a detached tmux launch plan."""
    root = find_training_root(Path(training_root))
    train_config = config if isinstance(config, TrainConfig) else TrainConfig.from_mapping(dict(config))

    model_text = _require_identifier(train_config.model_name or "roi-fcn-tiny", label="model_name")
    if train_config.model_directory is not None and train_config.run_name_suffix:
        raise ValueError("run_name_suffix cannot be used together with model_directory.")

    models_root = resolve_models_root(root, train_config.models_root)
    model_dir_name = (
        _require_identifier(train_config.model_directory, label="model_directory")
        if train_config.model_directory is not None and str(train_config.model_directory).strip()
        else suggest_model_run_id(model_text, run_name_suffix=train_config.run_name_suffix)
    )
    run_name = (
        _require_identifier(train_config.run_id, label="run_id")
        if train_config.run_id is not None and str(train_config.run_id).strip()
        else preview_next_run_id(models_root, model_directory=model_dir_name)
    )
    resolved_session_name = (
        _require_identifier(session_name, label="session_name")
        if session_name is not None and str(session_name).strip()
        else default_session_name(model_dir_name, run_name)
    )

    run_dir = build_model_run_dir_path(models_root, model_directory=model_dir_name, run_id=run_name).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    log_path = build_tmux_log_path(
        models_root,
        model_directory=model_dir_name,
        run_id=run_name,
        log_filename=log_filename,
    )
    if log_path.exists():
        raise FileExistsError(f"run log path already exists: {log_path}")

    working_directory = (root / "src").resolve()
    command = build_training_launch_command(
        train_config,
        python_executable=python_executable,
        model_name=model_text,
        model_directory=model_dir_name,
        run_id=run_name,
    )

    return TmuxTrainingLaunchPlanV02(
        session_name=resolved_session_name,
        model_name=model_text,
        model_directory=model_dir_name,
        run_id=run_name,
        run_dir=str(run_dir),
        log_path=str(log_path),
        working_directory=str(working_directory),
        command=command,
        python_executable=str(Path(python_executable).expanduser()),
    )


def plan_tmux_resume_launch(
    training_root: str | Path,
    *,
    source_run_dir: str | Path,
    additional_epochs: int,
    python_executable: str,
    session_name: str | None = None,
    log_filename: str = DEFAULT_TMUX_LOG_FILENAME,
    device_override: str | None = None,
) -> TmuxTrainingLaunchPlanV02:
    """Build a detached tmux launch plan for a resumed child run."""
    if int(additional_epochs) <= 0:
        raise ValueError(f"additional_epochs must be positive; got {additional_epochs}")
    root = find_training_root(Path(training_root))
    source_dir = _require_run_dir(source_run_dir)
    source_config = read_json(source_dir / RUN_CONFIG_FILENAME)
    model_directory = _source_model_directory(source_dir, source_config)
    models_root = resolve_models_root(root, source_config.get("models_root", "models"))
    run_id = preview_next_run_id(models_root, model_directory=model_directory)
    train_config = _resume_train_config_from_source(
        source_dir,
        source_config=source_config,
        model_directory=model_directory,
        run_id=run_id,
        additional_epochs=int(additional_epochs),
        device_override=device_override,
    )
    return plan_tmux_training_launch(
        root,
        train_config,
        python_executable=python_executable,
        session_name=session_name,
        log_filename=log_filename,
    )


def launch_session(
    session_name: str,
    command: str,
    log_path: str | Path,
    working_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Create a detached tmux session and launch the command with log redirection."""
    run_session_name = _require_identifier(session_name, label="session_name")
    if session_exists(run_session_name):
        raise ValueError(f"Session already exists: {run_session_name}")
    if not str(command).strip():
        raise ValueError("command cannot be empty.")

    resolved_log_path = Path(log_path).expanduser().resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path.touch(exist_ok=False)

    launch_shell_command = f"{command} >> {shlex.quote(str(resolved_log_path))} 2>&1"
    tmux_args: list[str] = ["new-session", "-d", "-s", run_session_name]
    if working_directory is not None:
        tmux_args.extend(["-c", str(Path(working_directory).expanduser().resolve())])
    tmux_args.append(launch_shell_command)

    result = _run_tmux(tmux_args)
    if result.returncode != 0:
        resolved_log_path.unlink(missing_ok=True)
        try:
            resolved_log_path.parent.rmdir()
        except OSError:
            pass
        raise RuntimeError(f"tmux new-session failed: {result.stderr.strip()}")

    return {
        "session_name": run_session_name,
        "log_path": str(resolved_log_path),
        "command": command,
    }


def end_session(session_name: str) -> bool:
    """Kill a tmux session by name. Returns False when the session does not exist."""
    run_session_name = _require_identifier(session_name, label="session_name")
    result = _run_tmux(["kill-session", "-t", run_session_name])
    if result.returncode != 0:
        if _is_no_server_error(result.stderr) or _is_missing_session_error(result.stderr):
            return False
        raise RuntimeError(f"tmux kill-session failed: {result.stderr.strip()}")
    return True


def read_log_tail(log_path: str | Path, max_lines: int = 200) -> str:
    """Read a tail-like slice from a log file (last N lines)."""
    limit = int(max_lines)
    if limit <= 0:
        raise ValueError("max_lines must be positive.")

    path = Path(log_path).expanduser().resolve()
    if not path.exists():
        return f"[log missing] {path}"
    if not path.is_file():
        return f"[not a file] {path}"

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = deque(handle, maxlen=limit)
    if not lines:
        return f"[log empty] {path}"
    return "".join(lines)
