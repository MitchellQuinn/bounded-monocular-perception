"""tmux/process helpers for the ROI-FCN training control panel v0.2."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping, Sequence

from .config import TrainConfig
from .paths import build_model_run_dir_path, find_training_root, resolve_models_root, suggest_model_run_id

TMUX_CONTROL_PANEL_BUILD_V02 = "2026-04-19-roi-fcn-tmux-v0.2"
DEFAULT_TMUX_LOG_FILENAME = "train.log"
TMUX_LOGS_DIRECTORY_NAME = "tmux_logs"
TRAINING_MODULE_NAME = "roi_fcn_training_v0_1.train"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


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


@dataclass(frozen=True)
class TmuxTrainingLaunchPlanV02:
    """Structured launch preview for one detached training session."""

    session_name: str
    model_name: str
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


def default_session_name(model_name: str, run_id: str) -> str:
    """Build the default tmux session name for a training run."""
    _ = _require_identifier(model_name, label="model_name")
    parsed_run_id = _require_identifier(run_id, label="run_id")
    return _require_identifier(f"roi_fcn_{parsed_run_id}", label="session_name")


def build_tmux_log_path(
    models_root_path: str | Path,
    *,
    model_name: str,
    run_id: str,
    log_filename: str = DEFAULT_TMUX_LOG_FILENAME,
) -> Path:
    """Return a notebook-owned tmux log path outside the training run directory."""
    model_text = _require_identifier(model_name, label="model_name")
    run_name = _require_identifier(run_id, label="run_id")
    resolved_log_filename = _require_log_filename(log_filename)
    log_stem = f"{run_name}__{resolved_log_filename}"
    return Path(models_root_path).expanduser().resolve() / model_text / TMUX_LOGS_DIRECTORY_NAME / log_stem


def build_training_launch_command(
    config: TrainConfig | Mapping[str, Any],
    *,
    python_executable: str,
    model_name: str | None = None,
    run_id: str | None = None,
) -> str:
    """Build a shell-safe detached training command."""
    train_config = config if isinstance(config, TrainConfig) else TrainConfig.from_mapping(dict(config))

    training_dataset = str(train_config.training_dataset or "").strip()
    if not training_dataset:
        raise ValueError("training_dataset cannot be blank.")

    model_text = _require_identifier(model_name or train_config.model_name or "roi-fcn-tiny", label="model_name")
    run_name = _require_identifier(run_id or train_config.run_id or suggest_model_run_id(model_text), label="run_id")
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
        "--early-stopping-patience",
        str(int(train_config.early_stopping_patience)),
        "--topology-id",
        str(train_config.topology_id),
        "--topology-variant",
        str(train_config.topology_variant),
        "--model-name",
        model_text,
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

    validation_dataset = str(train_config.validation_dataset or "").strip()
    if validation_dataset:
        args.extend(["--validation-dataset", validation_dataset])

    device_text = str(train_config.device or "").strip()
    if device_text:
        args.extend(["--device", device_text])

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
    if train_config.run_id is not None and train_config.run_name_suffix:
        raise ValueError("run_name_suffix cannot be used together with run_id.")
    run_name = (
        _require_identifier(train_config.run_id, label="run_id")
        if train_config.run_id is not None and str(train_config.run_id).strip()
        else suggest_model_run_id(model_text, run_name_suffix=train_config.run_name_suffix)
    )
    resolved_session_name = (
        _require_identifier(session_name, label="session_name")
        if session_name is not None and str(session_name).strip()
        else default_session_name(model_text, run_name)
    )

    models_root = resolve_models_root(root, train_config.models_root)
    run_dir = build_model_run_dir_path(models_root, model_name=model_text, run_id=run_name).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    log_path = build_tmux_log_path(
        models_root,
        model_name=model_text,
        run_id=run_name,
        log_filename=log_filename,
    )
    if log_path.exists():
        raise FileExistsError(f"tmux log path already exists: {log_path}")

    working_directory = (root / "src").resolve()
    command = build_training_launch_command(
        train_config,
        python_executable=python_executable,
        model_name=model_text,
        run_id=run_name,
    )

    return TmuxTrainingLaunchPlanV02(
        session_name=resolved_session_name,
        model_name=model_text,
        run_id=run_name,
        run_dir=str(run_dir),
        log_path=str(log_path),
        working_directory=str(working_directory),
        command=command,
        python_executable=str(Path(python_executable).expanduser()),
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
        raise RuntimeError(f"tmux new-session failed: {result.stderr.strip()}")

    return {
        "session_name": run_session_name,
        "log_path": str(resolved_log_path),
        "command": command,
    }


def end_session(session_name: str) -> bool:
    """Kill a tmux session by name. Returns False when the session does not exist."""
    run_session_name = _require_identifier(session_name, label="session_name")
    if not session_exists(run_session_name):
        return False
    result = _run_tmux(["kill-session", "-t", run_session_name])
    if result.returncode != 0:
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
