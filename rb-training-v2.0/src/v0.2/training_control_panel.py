"""Minimal tmux/process helpers for the v0.4 training control-panel notebook."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Sequence

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_MODEL_DIRECTORY_RE = re.compile(r"^[0-9]{6}-[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_-]*$")
_RUN_ID_RE = re.compile(r"^run_([0-9]+)$")


def _require_identifier(value: str, *, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError(
            f"{label} must match {_IDENTIFIER_RE.pattern!r}; got {text!r}."
        )
    return text


def _require_model_directory_name(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("model_directory cannot be empty.")
    if not _MODEL_DIRECTORY_RE.fullmatch(text):
        raise ValueError(
            "model_directory must follow yyMMdd-HHmm_suffix format, "
            f"example: 260330-1354_2d-cnn; got {text!r}"
        )
    return text


def _models_root_name(models_root: str | Path) -> str:
    resolved = Path(models_root).expanduser().resolve()
    return resolved.name


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
    text = stderr_text.lower()
    return (
        "no server running" in text
        or "failed to connect to server" in text
        or "can't find socket" in text
        or "error connecting to" in text and "no such file or directory" in text
    )


def _run_register_path(models_root: str | Path, model_directory: str) -> Path:
    model_dir = build_model_dir(models_root=models_root, model_directory=model_directory)
    return model_dir / "run_register.json"


def _load_run_register(models_root: str | Path, model_directory: str) -> dict[str, Any]:
    path = _run_register_path(models_root=models_root, model_directory=model_directory)
    if not path.exists():
        return {"runs": []}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"run_register.json must contain an object: {path}")
    runs = payload.get("runs")
    if runs is None:
        payload["runs"] = []
    elif not isinstance(runs, list):
        raise ValueError(f"run_register.json 'runs' must be a list: {path}")
    return payload


def _save_run_register(
    models_root: str | Path,
    model_directory: str,
    payload: dict[str, Any],
) -> Path:
    path = _run_register_path(models_root=models_root, model_directory=model_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


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
    name = _require_identifier(session_name, label="session_name")
    return name in list_sessions()


def list_model_directories(models_root: str | Path) -> list[str]:
    """List model directories under models root using established naming format."""
    root = Path(models_root).expanduser().resolve()
    if not root.exists():
        return []
    names: list[str] = []
    for child in root.iterdir():
        if child.is_dir() and _MODEL_DIRECTORY_RE.fullmatch(child.name):
            names.append(child.name)
    return sorted(names)


def suggest_model_directory(models_root: str | Path, model_suffix: str = "2d-cnn") -> str:
    """Suggest a new model directory name using the current local timestamp."""
    _ = models_root  # kept for API stability; naming is now timestamp-driven.
    suffix = _require_identifier(model_suffix, label="model_suffix")
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    return f"{timestamp}_{suffix}"


def build_model_dir(models_root: str | Path, model_directory: str) -> Path:
    """Return absolute model directory path."""
    model_dir_name = _require_model_directory_name(model_directory)
    return Path(models_root).expanduser().resolve() / model_dir_name


def preview_next_run_id(models_root: str | Path, model_directory: str) -> str:
    """Return the next run id, e.g. run_0001, without mutating state."""
    payload = _load_run_register(models_root=models_root, model_directory=model_directory)
    max_index = 0
    for row in payload.get("runs", []):
        if not isinstance(row, dict):
            continue
        raw = str(row.get("run_id", "")).strip()
        match = _RUN_ID_RE.fullmatch(raw)
        if not match:
            continue
        max_index = max(max_index, int(match.group(1)))
    return f"run_{max_index + 1:04d}"


def build_run_dir(run_id: str, models_root: str | Path, model_directory: str) -> Path:
    """Return run directory path models/<model_directory>/runs/<run_id>."""
    parsed_run_id = _require_identifier(run_id, label="run_id")
    return build_model_dir(models_root=models_root, model_directory=model_directory) / "runs" / parsed_run_id


def build_log_path(
    run_id: str,
    models_root: str | Path,
    model_directory: str,
    log_filename: str = "train.log",
) -> Path:
    """Return log path derived from model directory and run id."""
    if not str(log_filename).strip():
        raise ValueError("log_filename cannot be empty.")
    return build_run_dir(
        run_id=run_id,
        models_root=models_root,
        model_directory=model_directory,
    ) / log_filename


def resolve_session_log_path(
    session_name: str,
    models_root: str | Path,
) -> Path | None:
    """Resolve a tmux session name to log path by looking up run_register.json files."""
    resolved = resolve_session_run(session_name=session_name, models_root=models_root)
    if resolved is None:
        return None
    return Path(resolved["log_path"]).expanduser().resolve()


def reserve_run(
    *,
    models_root: str | Path,
    model_directory: str,
    model_name: str,
    topology_id: str | None = None,
    topology_variant: str | None = None,
    session_name: str,
    primary_variable_changed: str = "",
    notes: str = "",
    parent_run_id: str = "[reserved for future implementation]]",
) -> dict[str, Any]:
    """Create run directory and append a run_register entry before launch."""
    run_session_name = _require_identifier(session_name, label="session_name")
    model_name_text = str(model_name).strip()
    if not model_name_text:
        raise ValueError("model_name cannot be empty.")
    resolved_topology_id = (
        _require_identifier(topology_id, label="topology_id")
        if topology_id is not None and str(topology_id).strip()
        else None
    )
    resolved_topology_variant = (
        _require_identifier(topology_variant, label="topology_variant")
        if topology_variant is not None and str(topology_variant).strip()
        else model_name_text
    )

    model_dir_name = _require_model_directory_name(model_directory)
    model_dir = build_model_dir(models_root=models_root, model_directory=model_dir_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    run_id = preview_next_run_id(models_root=models_root, model_directory=model_dir_name)
    run_dir = build_run_dir(
        run_id=run_id,
        models_root=models_root,
        model_directory=model_dir_name,
    )
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    log_path = run_dir / "train.log"
    log_path.touch(exist_ok=True)

    models_root_label = _models_root_name(models_root)
    run_dir_rel = f"{models_root_label}/{model_dir_name}/runs/{run_id}"
    log_path_rel = f"{run_dir_rel}/train.log"
    best_ckpt_rel = f"{run_dir_rel}/best.pt"
    latest_ckpt_rel = f"{run_dir_rel}/latest.pt"

    payload = _load_run_register(models_root=models_root, model_directory=model_dir_name)
    runs = payload.setdefault("runs", [])
    if not isinstance(runs, list):
        raise ValueError("run_register.json 'runs' must be a list.")
    runs.append(
        {
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "model_name": model_name_text,
            "topology_id": resolved_topology_id,
            "topology_variant": resolved_topology_variant,
            "session_name": run_session_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": run_dir_rel,
            "log_path": log_path_rel,
            "best_checkpoint_path": best_ckpt_rel,
            "latest_checkpoint_path": latest_ckpt_rel,
            "primary_variable_changed": str(primary_variable_changed),
            "best_epoch": None,
            "best_val_loss": None,
            "best_val_acc_0_10m": None,
            "best_val_acc_0_25m": None,
            "best_val_acc_0_50m": None,
            "best_val_mae": None,
            "best_val_rmse": None,
            "notes": str(notes),
        }
    )
    run_register_path = _save_run_register(
        models_root=models_root,
        model_directory=model_dir_name,
        payload=payload,
    )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "run_register_path": str(run_register_path),
        "model_directory": model_dir_name,
    }


def resolve_session_run(session_name: str, models_root: str | Path) -> dict[str, Any] | None:
    """Find run metadata for a session name by scanning model run_register files."""
    target = _require_identifier(session_name, label="session_name")
    root = Path(models_root).expanduser().resolve()
    if not root.exists():
        return None

    matches: list[dict[str, Any]] = []
    for model_dir in list_model_directories(models_root=root):
        payload = _load_run_register(models_root=root, model_directory=model_dir)
        for row in payload.get("runs", []):
            if not isinstance(row, dict):
                continue
            if str(row.get("session_name", "")).strip() != target:
                continue
            run_id = str(row.get("run_id", "")).strip()
            if not run_id:
                continue
            log_path = build_log_path(
                run_id=run_id,
                models_root=root,
                model_directory=model_dir,
            )
            matches.append(
                {
                    "model_directory": model_dir,
                    "run_id": run_id,
                    "session_name": target,
                    "log_path": str(log_path),
                }
            )
    if not matches:
        return None
    return matches[-1]


def build_launch_command(
    *,
    run_id: str,
    model_directory: str,
    model_architecture_variant: str,
    topology_id: str = "distance_regressor_2d_cnn",
    topology_params_json: str = "{}",
    python_executable: str,
    training_module: str,
    training_data_root: str | Path,
    validation_data_root: str | Path,
    output_root: str | Path,
    seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    enable_lr_scheduler: bool,
    change_note: str = "tmux control-panel launch",
) -> str:
    """Build a shell-safe `python -m src.train ...` command string."""
    parsed_run_id = _require_identifier(run_id, label="run_id")
    parsed_model_directory = _require_model_directory_name(model_directory)
    architecture_variant = _require_identifier(
        model_architecture_variant, label="model_architecture_variant"
    )
    parsed_topology_id = _require_identifier(topology_id, label="topology_id")
    try:
        topology_payload = json.loads(str(topology_params_json))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"topology_params_json must be valid JSON; got {topology_params_json!r}"
        ) from exc
    if not isinstance(topology_payload, dict):
        raise ValueError(
            "topology_params_json must decode to a JSON object/dict."
        )
    normalized_topology_params_json = json.dumps(topology_payload, separators=(",", ":"))

    python_bin = str(Path(python_executable).expanduser())
    if not python_bin:
        raise ValueError("python_executable cannot be empty.")
    module_name = str(training_module).strip()
    if not module_name:
        raise ValueError("training_module cannot be empty.")

    args = [
        python_bin,
        "-u",
        "-m",
        module_name,
        "--training-data-root",
        str(Path(training_data_root).expanduser()),
        "--validation-data-root",
        str(Path(validation_data_root).expanduser()),
        "--output-root",
        str(Path(output_root).expanduser()),
        "--model-name",
        parsed_model_directory,
        "--run-id",
        parsed_run_id,
        "--topology-id",
        parsed_topology_id,
        "--topology-variant",
        architecture_variant,
        "--topology-params-json",
        normalized_topology_params_json,
        "--model-architecture-variant",
        architecture_variant,
        "--seed",
        str(int(seed)),
        "--batch-size",
        str(int(batch_size)),
        "--epochs",
        str(int(epochs)),
        "--learning-rate",
        str(float(learning_rate)),
        "--weight-decay",
        str(float(weight_decay)),
        "--early-stopping-patience",
        str(int(early_stopping_patience)),
        (
            "--enable-lr-scheduler"
            if bool(enable_lr_scheduler)
            else "--no-enable-lr-scheduler"
        ),
        "--change-note",
        str(change_note),
    ]
    return shlex.join(args)


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
    resolved_log_path.touch(exist_ok=True)

    launch_shell_command = f"{command} >> {shlex.quote(str(resolved_log_path))} 2>&1"
    tmux_args: list[str] = ["new-session", "-d", "-s", run_session_name]
    if working_directory is not None:
        tmux_args.extend(["-c", str(Path(working_directory).expanduser().resolve())])
    tmux_args.append(launch_shell_command)

    result = _run_tmux(tmux_args)
    if result.returncode != 0:
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


def read_log_tail(log_path: str | Path, max_lines_or_chars: int = 200) -> str:
    """Read a tail-like slice from a log file (last N lines)."""
    limit = int(max_lines_or_chars)
    if limit <= 0:
        raise ValueError("max_lines_or_chars must be positive.")

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
