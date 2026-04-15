"""Resume-specific helpers for tmux training control panels."""

from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path
import re
import shlex
from types import ModuleType
from typing import Any

from .state import RESUME_STATE_FILENAME, load_resume_state

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_MODEL_DIRECTORY_RE = re.compile(r"^[0-9]{6}-[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_-]*$")
_RUN_ID_RE = re.compile(r"^run_([0-9]+)$")


def _load_base_control_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "v0.2" / "training_control_panel.py"
    spec = importlib.util.spec_from_file_location(
        "rb_training_v0_2_training_control_panel",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training_control_panel module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_control = _load_base_control_module()


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


def _repo_root_from_models_root(models_root: str | Path) -> Path:
    return Path(models_root).expanduser().resolve().parent


def _load_run_register(models_root: str | Path, model_directory: str) -> dict[str, Any]:
    model_dir = base_control.build_model_dir(models_root=models_root, model_directory=model_directory)
    run_register_path = model_dir / "run_register.json"
    if not run_register_path.exists():
        return {"runs": []}
    with run_register_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"run_register.json must contain an object: {run_register_path}")
    runs = payload.get("runs")
    if runs is None:
        payload["runs"] = []
    elif not isinstance(runs, list):
        raise ValueError(f"run_register.json 'runs' must be a list: {run_register_path}")
    return payload


def _run_sort_index(run_id: str) -> int:
    match = _RUN_ID_RE.fullmatch(str(run_id).strip())
    if match is None:
        return -1
    return int(match.group(1))


def _resolve_run_dir(
    *,
    models_root: str | Path,
    model_directory: str,
    row: dict[str, Any],
) -> Path:
    repo_root = _repo_root_from_models_root(models_root)
    raw = str(row.get("run_dir", "")).strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        if candidate.is_dir():
            return candidate

    run_id = str(row.get("run_id", "")).strip()
    if run_id:
        return base_control.build_run_dir(
            run_id=run_id,
            models_root=models_root,
            model_directory=model_directory,
        )
    raise ValueError(f"Could not resolve run directory from register row: {row}")


def list_model_directories(models_root: str | Path) -> list[str]:
    """List model directories that match the established naming convention."""
    return base_control.list_model_directories(models_root=models_root)


def list_resume_candidates(models_root: str | Path, model_directory: str) -> list[dict[str, Any]]:
    """List known runs with resume-state metadata for a model directory."""
    model_dir_name = _require_model_directory_name(model_directory)
    payload = _load_run_register(models_root=models_root, model_directory=model_dir_name)

    records: list[dict[str, Any]] = []
    for row in payload.get("runs", []):
        if not isinstance(row, dict):
            continue

        run_id = str(row.get("run_id", "")).strip()
        if not _RUN_ID_RE.fullmatch(run_id):
            continue

        run_dir = _resolve_run_dir(
            models_root=models_root,
            model_directory=model_dir_name,
            row=row,
        )
        resume_state_path = run_dir / RESUME_STATE_FILENAME
        latest_checkpoint_path = run_dir / "latest.pt"

        last_completed_epoch: int | None = None
        state_error: str | None = None
        if resume_state_path.exists():
            try:
                state_payload = load_resume_state(resume_state_path, map_location="cpu")
                last_completed_epoch = int(state_payload["epoch"])
            except Exception as exc:
                state_error = str(exc)
        else:
            state_error = f"missing {RESUME_STATE_FILENAME}"

        updated_at: str | None = None
        if resume_state_path.exists():
            updated_at = datetime.fromtimestamp(
                resume_state_path.stat().st_mtime
            ).isoformat(timespec="seconds")
        elif latest_checkpoint_path.exists():
            updated_at = datetime.fromtimestamp(
                latest_checkpoint_path.stat().st_mtime
            ).isoformat(timespec="seconds")

        records.append(
            {
                "run_id": run_id,
                "run_dir": str(run_dir.resolve()),
                "session_name": str(row.get("session_name", "")).strip(),
                "resume_state_path": str(resume_state_path.resolve()),
                "latest_checkpoint_path": str(latest_checkpoint_path.resolve()),
                "last_completed_epoch": last_completed_epoch,
                "is_resumable": bool(resume_state_path.exists() and state_error is None),
                "state_error": state_error,
                "updated_at": updated_at,
            }
        )

    records.sort(key=lambda item: _run_sort_index(item["run_id"]))
    return records


def latest_resumable_candidate(models_root: str | Path, model_directory: str) -> dict[str, Any] | None:
    """Return the latest run entry that has a valid resume-state payload."""
    candidates = list_resume_candidates(models_root=models_root, model_directory=model_directory)
    resumable = [row for row in candidates if bool(row.get("is_resumable"))]
    if not resumable:
        return None
    return resumable[-1]


def reserve_resume_run(
    *,
    models_root: str | Path,
    model_directory: str,
    model_name: str,
    session_name: str,
    source_run_id: str,
    primary_variable_changed: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Reserve a run directory and register row for a resumed training run."""
    parent = _require_identifier(source_run_id, label="source_run_id")
    return base_control.reserve_run(
        models_root=models_root,
        model_directory=model_directory,
        model_name=model_name,
        session_name=session_name,
        primary_variable_changed=primary_variable_changed,
        notes=notes,
        parent_run_id=parent,
    )


def _require_run_dir(path: str | Path) -> Path:
    run_dir = Path(path).expanduser().resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    resume_state_path = run_dir / RESUME_STATE_FILENAME
    if not resume_state_path.exists():
        raise FileNotFoundError(
            f"Resume state not found. Expected {RESUME_STATE_FILENAME} in {run_dir}"
        )
    return run_dir


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _csv_tolerances(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw.strip()
        return text
    if isinstance(raw, (list, tuple, set)):
        return ",".join(str(float(value)) for value in raw)
    return str(float(raw))


def _resolved_data_root(config_payload: dict[str, Any], key: str, fallback_key: str) -> str:
    preferred = str(config_payload.get(key, "")).strip()
    if preferred:
        return preferred
    fallback = str(config_payload.get(fallback_key, "")).strip()
    if fallback:
        return fallback
    raise ValueError(f"Source config missing both {key!r} and {fallback_key!r}")


def build_resume_launch_command(
    *,
    run_id: str,
    model_directory: str,
    source_run_dir: str | Path,
    additional_epochs: int,
    python_executable: str,
    training_module: str,
    output_root: str | Path,
    change_note: str,
) -> str:
    """Build a shell-safe launch command for resumed training."""
    parsed_run_id = _require_identifier(run_id, label="run_id")
    parsed_model_directory = _require_model_directory_name(model_directory)

    epochs_to_add = int(additional_epochs)
    if epochs_to_add <= 0:
        raise ValueError(f"additional_epochs must be positive; got {additional_epochs}")

    source_dir = _require_run_dir(source_run_dir)
    source_config = _read_json(source_dir / "config.json")

    topology_id = _require_identifier(
        str(source_config.get("topology_id", "distance_regressor_2d_cnn")).strip(),
        label="topology_id",
    )
    topology_variant = str(source_config.get("topology_variant", "")).strip()
    if not topology_variant:
        topology_variant = str(source_config.get("model_architecture_variant", "")).strip()
    topology_variant = _require_identifier(
        topology_variant,
        label="topology_variant",
    )
    topology_params = source_config.get("topology_params")
    if not isinstance(topology_params, dict):
        topology_params = {}
    topology_params_json = json.dumps(topology_params, separators=(",", ":"))
    source_training_root = _resolved_data_root(
        source_config,
        key="training_data_root_resolved",
        fallback_key="training_data_root",
    )
    source_validation_root = _resolved_data_root(
        source_config,
        key="validation_data_root_resolved",
        fallback_key="validation_data_root",
    )

    python_bin = str(Path(python_executable).expanduser())
    if not python_bin:
        raise ValueError("python_executable cannot be empty.")
    module_name = str(training_module).strip()
    if not module_name:
        raise ValueError("training_module cannot be empty.")

    extra_tolerances = _csv_tolerances(source_config.get("extra_accuracy_tolerances_m", "0.25,0.50"))
    if not extra_tolerances:
        extra_tolerances = "0.25,0.50"

    lr_scheduler_enabled = _as_bool(source_config.get("enable_lr_scheduler"), default=False)
    cache_validation_enabled = _as_bool(source_config.get("cache_validation_in_ram"), default=True)
    internal_test_enabled = _as_bool(source_config.get("enable_internal_test_split"), default=False)

    args: list[str] = [
        python_bin,
        "-u",
        "-m",
        module_name,
        "--training-data-root",
        str(Path(source_training_root).expanduser()),
        "--validation-data-root",
        str(Path(source_validation_root).expanduser()),
        "--output-root",
        str(Path(output_root).expanduser()),
        "--model-name",
        parsed_model_directory,
        "--run-id",
        parsed_run_id,
        "--topology-id",
        topology_id,
        "--topology-variant",
        topology_variant,
        "--topology-params-json",
        topology_params_json,
        "--model-architecture-variant",
        topology_variant,
        "--seed",
        str(int(source_config.get("seed", 42))),
        "--batch-size",
        str(int(source_config.get("batch_size", 4))),
        "--epochs",
        str(int(source_config.get("epochs", 8))),
        "--learning-rate",
        str(float(source_config.get("learning_rate", 1e-3))),
        "--weight-decay",
        str(float(source_config.get("weight_decay", 1e-5))),
        "--huber-delta",
        str(float(source_config.get("huber_delta", 1.0))),
        "--distance-loss-weight",
        str(float(source_config.get("distance_loss_weight", 1.0))),
        "--orientation-loss-weight",
        str(float(source_config.get("orientation_loss_weight", 1.0))),
        "--position-loss-weight",
        str(float(source_config.get("position_loss_weight", 1.0))),
        "--early-stopping-patience",
        str(int(source_config.get("early_stopping_patience", 4))),
        "--padding-mode",
        str(source_config.get("padding_mode", "disabled")),
        "--progress-log-interval-batches",
        str(int(source_config.get("progress_log_interval_batches", 250))),
        "--accuracy-tolerance-m",
        str(float(source_config.get("accuracy_tolerance_m", 0.10))),
        "--extra-accuracy-tolerances-m",
        extra_tolerances,
        "--lr-scheduler-factor",
        str(float(source_config.get("lr_scheduler_factor", 0.5))),
        "--lr-scheduler-patience",
        str(int(source_config.get("lr_scheduler_patience", 1))),
        "--lr-scheduler-min-lr",
        str(float(source_config.get("lr_scheduler_min_lr", 1e-5))),
        "--train-cache-budget-gb",
        str(float(source_config.get("train_cache_budget_gb", 48.0))),
        "--train-shuffle-mode",
        str(source_config.get("train_shuffle_mode", "shard")),
        "--train-active-shard-count",
        str(int(source_config.get("train_active_shard_count", 3))),
        "--validation-cache-budget-gb",
        str(float(source_config.get("validation_cache_budget_gb", 40.0))),
        "--internal-test-fraction",
        str(float(source_config.get("internal_test_fraction", 0.1))),
        "--resume-from-run-dir",
        str(source_dir),
        "--additional-epochs",
        str(epochs_to_add),
        "--change-note",
        str(change_note).strip() or "resume launch",
    ]

    args.append("--enable-lr-scheduler" if lr_scheduler_enabled else "--no-enable-lr-scheduler")
    args.append("--cache-validation-in-ram" if cache_validation_enabled else "--no-cache-validation-in-ram")
    if internal_test_enabled:
        args.append("--enable-internal-test-split")

    return shlex.join(args)


def build_model_dir(models_root: str | Path, model_directory: str) -> Path:
    """Return absolute model directory path."""
    model_dir_name = _require_model_directory_name(model_directory)
    return base_control.build_model_dir(models_root=models_root, model_directory=model_dir_name)


def preview_next_run_id(models_root: str | Path, model_directory: str) -> str:
    """Return the next run id without mutating state."""
    model_dir_name = _require_model_directory_name(model_directory)
    return base_control.preview_next_run_id(models_root=models_root, model_directory=model_dir_name)


def list_sessions() -> list[str]:
    """Return active tmux session names."""
    return base_control.list_sessions()


def session_exists(session_name: str) -> bool:
    """Return True when the tmux session already exists."""
    return base_control.session_exists(session_name)


def launch_session(
    session_name: str,
    command: str,
    log_path: str | Path,
    working_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Create a detached tmux session and launch the command with log redirection."""
    return base_control.launch_session(
        session_name=session_name,
        command=command,
        log_path=log_path,
        working_directory=working_directory,
    )


def end_session(session_name: str) -> bool:
    """Kill a tmux session by name."""
    return base_control.end_session(session_name)


def read_log_tail(log_path: str | Path, max_lines_or_chars: int = 200) -> str:
    """Read a tail-like slice from a log file."""
    return base_control.read_log_tail(log_path, max_lines_or_chars=max_lines_or_chars)


def resolve_session_run(session_name: str, models_root: str | Path) -> dict[str, Any] | None:
    """Resolve a tmux session to model directory + run metadata."""
    return base_control.resolve_session_run(session_name=session_name, models_root=models_root)
