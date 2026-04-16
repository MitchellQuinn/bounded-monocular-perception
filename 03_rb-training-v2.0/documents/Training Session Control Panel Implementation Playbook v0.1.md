# Training Session Control Panel Implementation Playbook v0.1

## Purpose

This document is the implementation-grade template for building and evolving notebook-driven training session control panels in this repository.

It is designed so a new Codex instance (or another coding agent) can reproduce the established design pattern with minimal ambiguity and without reintroducing known bugs.

Use this playbook together with:

- `documents/ML Notebook Generation Template v0.2.md`
- `documents/Raccoon Ball Training Repo Notebook Standard v0.1.1`

---

## Scope

This playbook specifically covers:

- notebook UI/control panel architecture for tmux-managed training runs
- helper-module boundaries and required APIs
- run directory and run register contracts
- launch, monitoring, and session lifecycle behavior
- known failure modes and mandatory safeguards

It does not redefine model-training internals beyond launch contract compatibility.

---

## Canonical Architecture

### Rule 1: Notebook is UI only

The notebook must not own business logic for:

- tmux command construction/execution
- process lifecycle operations
- run-id reservation
- run register persistence
- log file parsing/tail logic

The notebook may only:

- collect user inputs
- call helper functions
- render status and output widgets
- orchestrate refresh flows

### Rule 2: Helper owns system logic

All process/tmux/filesystem behavior must live in helper code under `src/v0.2/`.

Current canonical helper:

- `src/v0.2/training_control_panel.py`

### Rule 3: Training script owns artifacts

Training artifacts are produced by `src/train.py`.

The control panel may pre-create run directory and `train.log` via reservation, but training remains the canonical artifact writer.

---

## Required Files and Responsibilities

### Notebook (UI layer)

- `notebooks/02_train_2d_cnn_v0.4.ipynb`

Responsibilities:

- path/config cell with minimal top-level variables
- controls for launch inputs
- status panel rendering
- session list/selection widgets
- log display, manual refresh, polling toggle
- calling helper APIs only

### Helper module (logic layer)

- `src/v0.2/training_control_panel.py`

Responsibilities:

- identifier validation
- model directory suggestion and validation
- run-id preview/reservation
- run_register load/save
- tmux session operations
- launch command construction
- session-to-log resolution
- log tail reads

### Training paths and package behavior

- `src/paths.py`
- `src/__init__.py`

Responsibilities:

- allow explicit `run_id` launches into pre-created run directories containing only `train.log`
- avoid eager import side effects that trigger runpy warnings

---

## Directory Contracts

### Models root

`models/`

### Model directory naming (established format)

`YYMMDD-HHMM_<suffix>`

Examples:

- `260331-1707_2d-cnn`
- `260330-1354_2d-cnn`

Regex:

- `^[0-9]{6}-[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_-]*$`

### Run directory

`models/<model_directory>/runs/<run_id>/`

### Run ID format

`run_0001`, `run_0002`, ...

Regex:

- `^run_([0-9]+)$`

### Required per-run output targets

At minimum, training flow must support:

- `train.log`
- `best.pt`
- `latest.pt`
- `config.json`
- `dataset_summary.json`
- `model_architecture.json`
- `split_membership.json`
- `split_summary.json`

Legacy compatibility files may still exist, but new flow should follow the contract above.

---

## run_register.json Contract

Location:

- `models/<model_directory>/run_register.json`

Schema shape (current canonical):

```json
{
  "runs": [
    {
      "run_id": "run_0001",
      "parent_run_id": "[reserved for future implementation]]",
      "model_name": "fast_v0_2",
      "session_name": "rb-fast_v0_2-run_0001",
      "created_at": "2026-03-31T16:22:00",
      "run_dir": "models/260331-1707_2d-cnn/runs/run_0001",
      "log_path": "models/260331-1707_2d-cnn/runs/run_0001/train.log",
      "best_checkpoint_path": "models/260331-1707_2d-cnn/runs/run_0001/best.pt",
      "latest_checkpoint_path": "models/260331-1707_2d-cnn/runs/run_0001/latest.pt",
      "primary_variable_changed": "bbox line width 3px -> 1px",
      "best_epoch": null,
      "best_val_loss": null,
      "best_val_acc_0_10m": null,
      "best_val_acc_0_25m": null,
      "best_val_acc_0_50m": null,
      "best_val_mae": null,
      "best_val_rmse": null,
      "notes": ""
    }
  ]
}
```

Rules:

- register is append-only for new reservations
- do not mutate existing entries during launch unless explicit post-run updater is implemented
- always store repo-relative `run_dir`/`log_path`/checkpoint paths

---

## Required Helper APIs

The helper must expose (minimum):

- `list_sessions()`
- `session_exists(session_name)`
- `build_log_path(...)`
- `build_launch_command(...)`
- `launch_session(session_name, command, log_path)`
- `end_session(session_name)`
- `read_log_tail(log_path, max_lines_or_chars)`

For run register flow, also expose:

- `suggest_model_directory(models_root, model_suffix)`
- `preview_next_run_id(models_root, model_directory)`
- `reserve_run(...)`
- `resolve_session_run(session_name, models_root)`
- `build_model_dir(...)`
- `build_run_dir(...)`

All subprocess tmux calls must use `subprocess` in helper layer.

---

## Notebook UI Contract

Required controls and views:

1. current tmux sessions list
2. editable session name
3. launch inputs (minimal set)
4. derived log path display
5. launch action
6. duplicate session rejection
7. end selected session action
8. dedicated status area
9. large log display area
10. manual log refresh
11. configurable polling interval (seconds)
12. auto-refresh toggle
13. selected-session log resolution and display

Additional established controls:

- `Model Name` (training architecture variant)
- `Model Dir` (timestamped model directory)
- preview `Run ID`
- `run_register.json` path display
- `LR Scheduler` toggle defaulting to enabled
- optional run notes/primary variable change fields

---

## Session Naming Rule

Session name must be editable but default automatically.

Current default pattern:

- `rb-<model_name>-<run_id>`

Behavior rules:

- if user has not edited session name, auto-update when run preview changes
- if user has manually changed session name, do not overwrite
- duplicate session names must be rejected pre-launch

---

## Launch Command Contract

Canonical shape:

```bash
python -u -m src.train \
  --training-data-root <path> \
  --validation-data-root <path> \
  --output-root <path> \
  --model-name <model_directory> \
  --run-id <run_id> \
  --model-architecture-variant <model_name> \
  --seed <int> \
  --batch-size <int> \
  --epochs <int> \
  --learning-rate <float> \
  --weight-decay <float> \
  --early-stopping-patience <int> \
  --enable-lr-scheduler|--no-enable-lr-scheduler \
  --change-note <text>
```

Important mapping:

- `--model-name` receives `model_directory` (directory key)
- `--model-architecture-variant` receives UI `Model Name`

This separation avoids coupling model architecture variant to directory naming.

---

## Launch Sequence (Must Follow)

1. derive `model_directory` and `next_run_id`
2. derive default session name from `model_name + run_id`
3. validate `session_name` and reject duplicates
4. reserve run:
- create run directory
- create `train.log`
- append run entry to `run_register.json`
5. build launch command with explicit `run_id` and `model_directory`
6. launch detached tmux session in repo root
7. refresh sessions and log panel
8. show status summary including `lr_scheduler` setting

---

## Log Monitoring Contract

- read from on-disk `train.log`
- do not rely on notebook output streams
- use tail behavior (last N lines)
- keep refresh simple (manual + polling)
- on session selection, resolve its run via `run_register.json` then read corresponding log

---

## Critical Safeguards and Known Bugs

These issues were encountered in real implementation and must be guarded against.

### 1) tmux no-socket startup error

Symptom:

- `tmux list-sessions failed: error connecting to /tmp/tmux-... (No such file or directory)`

Required handling:

- treat as no-server/no-sessions state (return empty list), not fatal

### 2) Pre-created run dir collision

Symptom:

- control panel reserves run dir
- `src.train` attempts to create same path
- `FileExistsError`

Required handling:

- in `src.paths.make_model_run_dir`, allow existing run dir for explicit `run_id` only when contents are subset of `{train.log}`
- otherwise keep failure behavior

### 3) Eager import runpy warning

Symptom:

- warning from `runpy` that `src.train` already in `sys.modules`

Cause:

- `src/__init__.py` imported `train` eagerly

Required handling:

- use lazy wrapper functions in `src/__init__.py` for `train_distance_regressor` and `evaluate_saved_run`

### 4) Wrong model directory source

Symptom:

- model directory reused/derived from prior entities instead of new timestamp

Required handling:

- `suggest_model_directory` must always default to current timestamp format
- never infer model directory from dataset name

### 5) Wrong log path derivation

Symptom:

- log path derived from session alone or wrong directory

Required handling:

- always derive log path from `(models_root, model_directory, run_id)`
- session-to-log mapping should resolve through `run_register.json`

### 6) Missing scheduler visibility

Symptom:

- LR scheduler setting not obvious in UI/state

Required handling:

- expose `LR Scheduler` control in notebook
- default to enabled
- include state in status panel
- pass boolean to launch command (`--enable-lr-scheduler` / `--no-enable-lr-scheduler`)

---

## Validation Checklist (Before Handoff)

### Static checks

- notebook JSON valid (`python3 -m json.tool ...`)
- helper compiles (`python3 -m compileall src/v0.2/training_control_panel.py`)
- changed core modules compile (`src/paths.py`, `src/train.py`, `src/__init__.py`)

### Behavioral checks

- no tmux server: sessions list gracefully empty
- duplicate session name: launch blocked with clear status
- run reservation creates:
- run dir
- `train.log`
- `run_register.json` entry
- launch command includes correct model directory, run_id, scheduler flag
- selecting session shows correct log tail

### Filesystem checks

For a new run, confirm path shape:

- `models/<model_directory>/run_register.json`
- `models/<model_directory>/runs/<run_id>/train.log`
- training artifacts in same run directory

---

## Recommended Status Message Format

Status panel should include:

- scheduler state (enabled/disabled)
- concise action outcome
- clear error text when failures occur

Example:

- `lr_scheduler=enabled`
- `Launched session rb-fast_v0_2-run_0002 for 260331-2016_2d-cnn / run_0002`

---

## Non-Goals (Do Not Add by Default)

Do not add unless explicitly requested:

- streaming infra beyond polling
- cross-process locks for run register
- automatic rollback of reserved run IDs on failed launches
- broad refactors outside session-control boundary
- additional framework abstractions

---

## Extension Notes (Future)

Expected future enhancements that this design supports:

- post-run updater to fill `best_*` metrics in `run_register.json`
- parent-child run lineage via `parent_run_id`
- resume/retry utilities tied to `latest.pt`
- richer status panel with current run phase extraction from logs

Keep these as additive features; do not break current contracts.

---

## Implementation Handoff Requirements

When implementing against this playbook, always report:

1. files changed
2. API changes made (if any)
3. directory contract impact
4. manual validation performed
5. known residual risks/TODOs

If any required rule cannot be followed, explicitly document the deviation and reason.
