"""Operator-facing tmux control panel with resume support for ROI-FCN training."""

from __future__ import annotations

import html
from pathlib import Path
import sys
import threading
import traceback

import ipywidgets as widgets
from IPython.display import display

from .config import EvalConfig, TrainConfig
from .discovery import discover_dataset_references
from .evaluate import evaluate_saved_run
from .paths import find_training_root, preview_next_run_id, resolve_models_root, suggest_model_run_id
from .tmux_launcher_v0_2 import (
    DEFAULT_TMUX_LOG_FILENAME,
    TMUX_CONTROL_PANEL_BUILD_V03,
    default_session_name,
    end_session,
    launch_session,
    latest_resume_candidate,
    list_model_directories,
    list_resume_candidates,
    list_sessions,
    plan_tmux_resume_launch,
    plan_tmux_training_launch,
    read_epoch_summary,
    read_log_tail,
    resolve_session_run_paths,
    session_exists,
)
from .topologies import get_topology_definition, list_topology_ids, list_topology_variants


class RoiFcnTrainingControlPanelV03:
    """Detached tmux launcher and evaluator for fresh and resume ROI-FCN runs."""

    def __init__(self, training_root: Path, *, python_executable: str | None = None) -> None:
        self.training_root = find_training_root(training_root)
        self.models_root = resolve_models_root(self.training_root, "models")
        self.python_executable = str(Path(python_executable).expanduser()) if python_executable else sys.executable

        self.preview_html = widgets.HTML()
        self.status_html = widgets.HTML()
        self.topology_help = widgets.HTML()
        self.resume_note_html = widgets.HTML()

        self.workflow_toggle = widgets.ToggleButtons(
            description="Workflow",
            options=[("Fresh Training", "fresh"), ("Resume Existing", "resume")],
            value="fresh",
        )

        self.train_dataset_dropdown = widgets.Dropdown(description="Train Data")
        self.validation_dataset_dropdown = widgets.Dropdown(description="Val Data")
        self.refresh_datasets_button = widgets.Button(description="Refresh Datasets")

        topology_ids = list_topology_ids()
        self.topology_id_dropdown = widgets.Dropdown(
            description="Topology",
            options=[(value, value) for value in topology_ids],
            value=topology_ids[0],
        )
        default_variants = list_topology_variants(topology_ids[0])
        self.topology_variant_dropdown = widgets.Dropdown(
            description="Variant",
            options=[(value, value) for value in default_variants],
            value=default_variants[0],
        )

        self.model_name_text = widgets.Text(description="Model Name", value="roi-fcn-tiny")
        self.model_directory_text = widgets.Text(description="Model Dir", value="")
        self.model_directory_dropdown = widgets.Dropdown(
            description="Known Models",
            options=[("<no existing models>", "")],
            value="",
        )
        self.refresh_models_button = widgets.Button(description="Refresh Models")

        self.run_id_text = widgets.Text(description="Run ID", value="")
        self.session_name_text = widgets.Text(description="Session", value="")
        self.device_text = widgets.Text(
            description="Device",
            value="",
            placeholder="blank -> auto CUDA; CPU fallback disabled for training",
        )

        self.batch_size = widgets.BoundedIntText(description="Batch Size", value=16, min=1, max=4096)
        self.epochs = widgets.BoundedIntText(description="Epochs", value=8, min=1, max=10000)
        self.learning_rate = widgets.FloatText(description="LR", value=1e-3)
        self.weight_decay = widgets.FloatText(description="Weight Decay", value=1e-5)
        self.gaussian_sigma = widgets.FloatText(description="Sigma Px", value=2.5)
        self.early_stopping_patience = widgets.BoundedIntText(description="Patience", value=4, min=1, max=10000)
        self.progress_log_interval_steps = widgets.BoundedIntText(description="Log Every", value=50, min=1, max=1000000)
        self.roi_width = widgets.BoundedIntText(description="ROI Width", value=300, min=1, max=4096)
        self.roi_height = widgets.BoundedIntText(description="ROI Height", value=300, min=1, max=4096)
        self.evaluation_max_visual_examples = widgets.BoundedIntText(description="Eval Visuals", value=12, min=0, max=512)

        self.resume_extra_epochs = widgets.BoundedIntText(
            description="Extra Epochs",
            value=0,
            min=0,
            max=100000,
        )

        self.source_info_output = widgets.Textarea(
            value="",
            description="Source Info",
            disabled=True,
            layout=widgets.Layout(width="100%", height="180px"),
        )
        self.launch_plan_output = widgets.Textarea(
            value="",
            description="Launch Plan",
            disabled=True,
            layout=widgets.Layout(width="100%", height="220px"),
        )

        self.run_dir_text = widgets.Text(description="Run Dir", value="", placeholder="Predicted run directory or existing run to evaluate")
        self.log_path_text = widgets.Text(description="Log Path", value="", placeholder="Run log path; blank -> <Run Dir>/train.log")

        self.sessions_dropdown = widgets.Dropdown(description="Sessions", options=[("<no active sessions>", "")], value="")
        self.tail_lines_input = widgets.BoundedIntText(description="Tail Lines", value=120, min=1, max=5000)
        self.poll_interval_input = widgets.FloatText(description="Poll Secs", value=5.0, layout=widgets.Layout(width="180px"))
        self.auto_refresh_toggle = widgets.ToggleButton(
            value=False,
            description="Auto Refresh",
            icon="refresh",
            layout=widgets.Layout(width="180px"),
        )

        self.launch_button = widgets.Button(description="Launch In tmux", button_style="primary")
        self.evaluate_button = widgets.Button(description="Evaluate Run")
        self.refresh_sessions_button = widgets.Button(description="Refresh Sessions")
        self.refresh_log_button = widgets.Button(description="Refresh Log")
        self.end_session_button = widgets.Button(description="End Session", button_style="danger")
        self.clear_output_button = widgets.Button(description="Clear Output")

        self.action_output = widgets.Output(layout=widgets.Layout(overflow_y="auto"))
        self.epoch_summary_output = widgets.HTML(value="")
        self.log_tail_output = widgets.Textarea(
            value="",
            description="",
            disabled=True,
            layout=widgets.Layout(width="100%", height="360px"),
        )

        self._model_directory_state = {"last_auto": ""}
        self._run_id_state = {"last_auto": ""}
        self._session_name_state = {"last_auto": ""}
        self._run_dir_state = {"last_auto": ""}
        self._log_path_state = {"last_auto": ""}
        self._session_runtime_state = {"last_launched": ""}
        self._preview_state = {"refreshing": False}
        self._auto_refresh_state = {"thread": None, "stop_event": None}

        self.refresh_datasets_button.on_click(self._on_refresh_datasets)
        self.refresh_models_button.on_click(self._on_refresh_models)
        self.refresh_sessions_button.on_click(self._on_refresh_sessions_clicked)
        self.refresh_log_button.on_click(self._on_refresh_runtime_clicked)
        self.auto_refresh_toggle.observe(self._on_auto_refresh_toggle, names="value")
        self.launch_button.on_click(self._on_launch_clicked)
        self.evaluate_button.on_click(self._on_evaluate_clicked)
        self.end_session_button.on_click(self._on_end_session_clicked)
        self.clear_output_button.on_click(self._on_clear_output_clicked)
        self.workflow_toggle.observe(self._on_workflow_changed, names="value")
        self.topology_id_dropdown.observe(self._on_topology_id_changed, names="value")
        self.model_directory_dropdown.observe(self._on_model_directory_selected, names="value")
        self.sessions_dropdown.observe(self._on_session_selected, names="value")

        observed_widgets = [
            self.train_dataset_dropdown,
            self.validation_dataset_dropdown,
            self.topology_id_dropdown,
            self.topology_variant_dropdown,
            self.model_name_text,
            self.model_directory_text,
            self.run_id_text,
            self.session_name_text,
            self.device_text,
            self.batch_size,
            self.epochs,
            self.learning_rate,
            self.weight_decay,
            self.gaussian_sigma,
            self.early_stopping_patience,
            self.progress_log_interval_steps,
            self.roi_width,
            self.roi_height,
            self.evaluation_max_visual_examples,
            self.resume_extra_epochs,
        ]
        for widget in observed_widgets:
            widget.observe(self._refresh_launch_preview, names="value")

        self._refresh_datasets()
        self._refresh_models()
        self._sync_topology_help()
        self._refresh_sessions()
        self._refresh_launch_preview()
        self._refresh_epoch_summary()

    @property
    def widget(self) -> widgets.Widget:
        return widgets.VBox(
            [
                widgets.HTML(f"<b>ROI-FCN Training Control Panel</b> <code>{TMUX_CONTROL_PANEL_BUILD_V03}</code>"),
                self.workflow_toggle,
                widgets.HBox([self.train_dataset_dropdown, self.validation_dataset_dropdown, self.refresh_datasets_button]),
                widgets.HBox([self.topology_id_dropdown, self.topology_variant_dropdown]),
                self.topology_help,
                widgets.HBox([self.model_name_text, self.model_directory_text]),
                widgets.HBox([self.model_directory_dropdown, self.refresh_models_button]),
                widgets.HBox([self.run_id_text, self.session_name_text, self.device_text]),
                widgets.HBox([self.batch_size, self.epochs, self.learning_rate]),
                widgets.HBox([self.weight_decay, self.gaussian_sigma, self.early_stopping_patience]),
                widgets.HBox([self.progress_log_interval_steps, self.roi_width, self.roi_height, self.evaluation_max_visual_examples]),
                widgets.HBox([self.resume_extra_epochs]),
                self.resume_note_html,
                self.source_info_output,
                self.preview_html,
                self.launch_plan_output,
                self.run_dir_text,
                self.log_path_text,
                widgets.HBox([self.launch_button, self.evaluate_button, self.refresh_sessions_button, self.refresh_log_button, self.end_session_button, self.clear_output_button]),
                widgets.HBox([self.sessions_dropdown, self.tail_lines_input, self.poll_interval_input, self.auto_refresh_toggle]),
                self.status_html,
                self.epoch_summary_output,
                self.log_tail_output,
                self.action_output,
            ]
        )

    def _append_action(self, message: str) -> None:
        with self.action_output:
            print(message)

    def _set_banner(self, target: widgets.HTML, message: str, *, ok: bool, label: str) -> None:
        color = "#0b6f3c" if ok else "#8c1d18"
        target.value = (
            f"<div style='padding:8px 10px;border-left:4px solid {color};'>"
            f"<b>{html.escape(label)}</b><br><code>{html.escape(message)}</code></div>"
        )

    def _set_epoch_summary_text(self, message: str) -> None:
        self.epoch_summary_output.value = (
            "<div style='border:1px solid #d0d7de;padding:8px;border-radius:6px;'>"
            "<strong>Epoch Summary</strong>"
            "<pre style='white-space:pre-wrap;margin:8px 0 0 0;font-family:Menlo,Consolas,monospace;"
            "font-size:12px;line-height:1.35;'>"
            f"{html.escape(message)}"
            "</pre>"
            "</div>"
        )

    def _clear_status(self) -> None:
        self.status_html.value = ""

    def _sync_auto_text(self, widget: widgets.Text, suggested: str, state: dict[str, str]) -> None:
        current = str(widget.value or "").strip()
        if (not current) or (current == state["last_auto"]):
            widget.value = suggested
        state["last_auto"] = suggested

    def _refresh_datasets(self) -> None:
        discovered = discover_dataset_references(self.training_root)
        options = [(dataset.name, dataset.name) for dataset in discovered]
        if not options:
            options = [("<no valid datasets discovered>", "")]
        current_train = self.train_dataset_dropdown.value
        current_validation = self.validation_dataset_dropdown.value
        self.train_dataset_dropdown.options = options
        self.validation_dataset_dropdown.options = options
        valid_values = {value for _, value in options}
        selected_train = current_train if current_train in valid_values else options[0][1]
        selected_validation = current_validation if current_validation in valid_values else selected_train
        self.train_dataset_dropdown.value = selected_train
        self.validation_dataset_dropdown.value = selected_validation

    def _refresh_models(self) -> None:
        model_dirs = list_model_directories(self.models_root)
        current_text = str(self.model_directory_text.value or "").strip()
        current_dropdown = str(self.model_directory_dropdown.value or "").strip()
        if model_dirs:
            options = [("<manual / new>", "")] + [(name, name) for name in model_dirs]
            if current_text in model_dirs:
                preferred = current_text
            elif current_dropdown in model_dirs:
                preferred = current_dropdown
            elif self.workflow_toggle.value == "resume":
                preferred = model_dirs[-1]
            else:
                preferred = ""
        else:
            options = [("<no existing models>", "")]
            preferred = ""
        self.model_directory_dropdown.options = options
        self.model_directory_dropdown.value = preferred

    def _sync_topology_help(self) -> None:
        topology_id = str(self.topology_id_dropdown.value or "").strip()
        if not topology_id:
            self.topology_help.value = "<div>Select a topology.</div>"
            return
        definition = get_topology_definition(topology_id)
        self.topology_help.value = (
            f"<div><b>{html.escape(str(definition.topology_metadata.get('display_name', topology_id)))}</b></div>"
            f"<div>Status: <code>{html.escape(str(definition.topology_metadata.get('status', 'active')))}</code></div>"
            f"<div>{html.escape(str(definition.topology_metadata.get('note', '')))}</div>"
        )

    def _sync_default_model_directory(self) -> None:
        if self.workflow_toggle.value != "fresh":
            return
        model_name = str(self.model_name_text.value or "").strip() or "roi-fcn-tiny"
        try:
            suggested = suggest_model_run_id(model_name)
        except Exception:
            return
        self._sync_auto_text(self.model_directory_text, suggested, self._model_directory_state)

    def _sync_default_run_id(self) -> None:
        model_directory = str(self.model_directory_text.value or "").strip()
        if not model_directory:
            return
        try:
            suggested = preview_next_run_id(self.models_root, model_directory=model_directory)
        except Exception:
            return
        self._sync_auto_text(self.run_id_text, suggested, self._run_id_state)

    def _sync_default_session_name(self) -> None:
        model_directory = str(self.model_directory_text.value or "").strip()
        run_id = str(self.run_id_text.value or "").strip()
        if not model_directory or not run_id:
            return
        try:
            suggested = default_session_name(model_directory, run_id)
        except Exception:
            return
        self._sync_auto_text(self.session_name_text, suggested, self._session_name_state)

    def _build_train_config(self) -> TrainConfig:
        return TrainConfig(
            training_dataset=str(self.train_dataset_dropdown.value or "").strip(),
            validation_dataset=str(self.validation_dataset_dropdown.value or "").strip() or None,
            topology_id=str(self.topology_id_dropdown.value or "").strip(),
            topology_variant=str(self.topology_variant_dropdown.value or "").strip(),
            model_name=str(self.model_name_text.value or "").strip() or "roi-fcn-tiny",
            model_directory=str(self.model_directory_text.value or "").strip() or None,
            run_id=str(self.run_id_text.value or "").strip() or None,
            device=str(self.device_text.value or "").strip() or None,
            batch_size=int(self.batch_size.value),
            epochs=int(self.epochs.value),
            learning_rate=float(self.learning_rate.value),
            weight_decay=float(self.weight_decay.value),
            gaussian_sigma_px=float(self.gaussian_sigma.value),
            early_stopping_patience=int(self.early_stopping_patience.value),
            progress_log_interval_steps=int(self.progress_log_interval_steps.value),
            roi_width_px=int(self.roi_width.value),
            roi_height_px=int(self.roi_height.value),
            evaluation_max_visual_examples=int(self.evaluation_max_visual_examples.value),
        )

    def _session_status_note(self, session_name: str) -> tuple[str, bool]:
        try:
            note = "session already active" if session_exists(session_name) else "session name available"
            return note, note == "session name available"
        except Exception as exc:
            return f"tmux availability not confirmed: {exc}", False

    def _build_model_state_text(self, model_directory: str) -> str:
        model_dir = str(model_directory or "").strip()
        if not model_dir:
            return "model_directory: <blank>"
        try:
            candidates = list_resume_candidates(self.models_root, model_directory=model_dir)
        except Exception as exc:
            return f"model_directory: {model_dir}\nstate_refresh_error: {exc}"
        lines = [f"model_directory: {model_dir}", f"workflow: {self.workflow_toggle.value}", ""]
        if not candidates:
            lines.append("No runs discovered under this model directory.")
            return "\n".join(lines)
        latest = next((row for row in reversed(candidates) if row.get("is_resumable")), None)
        if latest is not None:
            lines.append(
                f"latest_resumable: {latest['run_id']} "
                f"(completed={latest.get('completed_epochs', '?')} / total={latest.get('planned_epochs', '?')})"
            )
        else:
            lines.append("latest_resumable: none")
        lines.append(f"known_runs: {len(candidates)}")
        lines.append("")
        for row in candidates:
            lines.append(
                f"- {row['run_id']}: completed={row.get('completed_epochs', '?')} / total={row.get('planned_epochs', '?')}; "
                f"remaining={row.get('remaining_epochs', '?')}; resumable={'yes' if row.get('is_resumable') else 'no'}"
            )
            if row.get("state_error"):
                lines.append(f"  warning: {row['state_error']}")
        return "\n".join(lines)

    def _resolve_log_path(self) -> str:
        log_path = str(self.log_path_text.value or "").strip()
        if log_path:
            return log_path
        run_dir = str(self.run_dir_text.value or "").strip()
        if not run_dir:
            return ""
        return str(Path(run_dir).expanduser().resolve() / DEFAULT_TMUX_LOG_FILENAME)

    def _refresh_sessions(self, *, selected: str | None = None) -> None:
        current = selected if selected is not None else self.sessions_dropdown.value
        try:
            sessions = list_sessions()
        except Exception:
            self.sessions_dropdown.options = [("<tmux unavailable>", "")]
            self.sessions_dropdown.value = ""
            return
        if not sessions:
            self.sessions_dropdown.options = [("<no active sessions>", "")]
            self.sessions_dropdown.value = ""
            return
        options = [(name, name) for name in sessions]
        self.sessions_dropdown.options = options
        valid_values = {value for _, value in options}
        self.sessions_dropdown.value = current if current in valid_values else options[0][1]

    def _sync_selected_session_runtime_paths(self) -> bool:
        session_name = str(self.sessions_dropdown.value or "").strip()
        if not session_name:
            return False
        info = resolve_session_run_paths(
            self.training_root,
            session_name,
            models_root_path=self.models_root,
        )
        if info is None:
            return False

        previous_refreshing = self._preview_state["refreshing"]
        self._preview_state["refreshing"] = True
        try:
            for key, widget, state in (
                ("model_directory", self.model_directory_text, self._model_directory_state),
                ("run_id", self.run_id_text, self._run_id_state),
                ("session_name", self.session_name_text, self._session_name_state),
                ("run_dir", self.run_dir_text, self._run_dir_state),
                ("log_path", self.log_path_text, self._log_path_state),
            ):
                value = str(info.get(key) or "").strip()
                if value:
                    state["last_auto"] = ""
                    widget.value = value
        finally:
            self._preview_state["refreshing"] = previous_refreshing
        return True

    def _refresh_log(self) -> None:
        self._sync_selected_session_runtime_paths()
        log_path = self._resolve_log_path()
        self._refresh_epoch_summary()
        if not log_path:
            self.log_tail_output.value = "[log path blank]"
            return
        try:
            self.log_tail_output.value = read_log_tail(log_path, max_lines=int(self.tail_lines_input.value))
        except Exception:
            self.log_tail_output.value = traceback.format_exc()

    def _refresh_epoch_summary(self) -> None:
        run_dir = str(self.run_dir_text.value or "").strip()
        self._set_epoch_summary_text(read_epoch_summary(Path(run_dir) if run_dir else None))

    def _refresh_runtime_output(self) -> None:
        self._refresh_sessions()
        self._refresh_log()

    def _stop_auto_refresh(self) -> None:
        stop_event = self._auto_refresh_state.get("stop_event")
        thread = self._auto_refresh_state.get("thread")
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._auto_refresh_state["thread"] = None
        self._auto_refresh_state["stop_event"] = None

    def _auto_refresh_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self._refresh_runtime_output()
            except Exception as exc:
                self._append_action(f"Auto refresh failed: {exc}")
            stop_event.wait(max(0.5, float(self.poll_interval_input.value)))

    def _active_launch_plan(self):
        if self.workflow_toggle.value == "resume":
            model_directory = str(self.model_directory_text.value or "").strip()
            if not model_directory:
                raise ValueError("Model Dir cannot be blank for resume workflow.")
            source_info = latest_resume_candidate(self.models_root, model_directory=model_directory)
            self.source_info_output.value = self._build_model_state_text(model_directory)
            if source_info is None:
                raise ValueError(f"No resumable source run found under {model_directory}.")
            required_remaining = 0
            planned_epochs = source_info.get("planned_epochs")
            completed_epochs = source_info.get("completed_epochs")
            if isinstance(planned_epochs, int) and isinstance(completed_epochs, int):
                required_remaining = max(int(planned_epochs) - int(completed_epochs), 0)
            extra_epochs = int(self.resume_extra_epochs.value)
            additional_epochs = required_remaining + extra_epochs
            if additional_epochs <= 0:
                raise ValueError(
                    "Resume launch requires positive total additional epochs. "
                    f"required_remaining={required_remaining} extra_epochs={extra_epochs}"
                )
            plan = plan_tmux_resume_launch(
                self.training_root,
                source_run_dir=source_info["run_dir"],
                additional_epochs=additional_epochs,
                python_executable=self.python_executable,
                session_name=str(self.session_name_text.value or "").strip() or None,
                device_override=str(self.device_text.value or "").strip() or None,
            )
            preview = {
                "mode": "resume",
                "plan": plan,
                "source_info": source_info,
                "required_remaining": required_remaining,
                "extra_epochs": extra_epochs,
                "additional_epochs": additional_epochs,
            }
            return plan, preview

        self._sync_default_model_directory()
        self._sync_default_run_id()
        self._sync_default_session_name()
        plan = plan_tmux_training_launch(
            self.training_root,
            self._build_train_config(),
            python_executable=self.python_executable,
            session_name=str(self.session_name_text.value or "").strip() or None,
        )
        self.source_info_output.value = self._build_model_state_text(str(self.model_directory_text.value or "").strip())
        preview = {"mode": "fresh", "plan": plan}
        return plan, preview

    def _build_launch_plan_text(self, preview: dict[str, object], session_note: str) -> str:
        plan = preview["plan"]
        assert hasattr(plan, "session_name")
        lines = [
            f"mode={preview['mode']}",
            f"session_name={plan.session_name}",
            f"model_directory={plan.model_directory}",
            f"run_id={plan.run_id}",
            f"run_dir={plan.run_dir}",
            f"log_path={plan.log_path}",
            f"working_directory={plan.working_directory}",
            f"attach_command=tmux attach -t {plan.session_name}",
            f"session_status={session_note}",
        ]
        if preview["mode"] == "resume":
            source_info = preview["source_info"]
            assert isinstance(source_info, dict)
            lines.extend(
                [
                    f"source_run_id={source_info['run_id']}",
                    f"source_run_dir={source_info['run_dir']}",
                    f"required_remaining={preview['required_remaining']}",
                    f"extra_epochs={preview['extra_epochs']}",
                    f"additional_epochs={preview['additional_epochs']}",
                ]
            )
        lines.extend(["", plan.command])
        return "\n".join(lines)

    def _refresh_launch_preview(self, *_args) -> None:
        if self._preview_state["refreshing"]:
            return
        self._preview_state["refreshing"] = True
        try:
            if self.workflow_toggle.value == "resume":
                self.resume_note_html.value = (
                    "<div><b>Resume Mode</b>: launch settings are inherited from the latest resumable run in the selected model directory. "
                    "Only <code>Model Dir</code>, <code>Session</code>, <code>Device</code>, and <code>Extra Epochs</code> affect the launch.</div>"
                )
            else:
                self.resume_note_html.value = (
                    "<div><b>Fresh Mode</b>: the fields above define a new tmux launch. "
                    "Use <code>Resume Existing</code> to continue from the latest resumable child run.</div>"
                )
            plan, preview = self._active_launch_plan()
            self._sync_auto_text(self.model_directory_text, plan.model_directory, self._model_directory_state)
            self._sync_auto_text(self.run_id_text, plan.run_id, self._run_id_state)
            self._sync_auto_text(self.session_name_text, plan.session_name, self._session_name_state)
            self._sync_auto_text(self.run_dir_text, plan.run_dir, self._run_dir_state)
            self._sync_auto_text(self.log_path_text, plan.log_path, self._log_path_state)
            session_note, preview_ok = self._session_status_note(plan.session_name)
            self.launch_plan_output.value = self._build_launch_plan_text(preview, session_note)
            self._set_banner(self.preview_html, f"Launch preview ready for {plan.session_name}. {session_note}.", ok=preview_ok, label="Preview")
        except Exception as exc:
            self.launch_plan_output.value = f"[preview unavailable]\n{exc}"
            self._set_banner(self.preview_html, str(exc), ok=False, label="Preview")
        finally:
            self._preview_state["refreshing"] = False

    def _on_refresh_datasets(self, _button) -> None:
        self._refresh_datasets()
        self._refresh_launch_preview()
        self._append_action("Refreshed dataset references.")

    def _on_refresh_models(self, _button) -> None:
        self._refresh_models()
        self._refresh_launch_preview()
        self._append_action("Refreshed model directories.")

    def _on_refresh_sessions_clicked(self, _button) -> None:
        self._refresh_sessions()
        self._append_action("Refreshed tmux sessions.")

    def _on_refresh_runtime_clicked(self, _button) -> None:
        self._refresh_runtime_output()
        self._append_action("Refreshed runtime output.")

    def _on_topology_id_changed(self, _change) -> None:
        topology_id = str(self.topology_id_dropdown.value or "").strip()
        variants = list_topology_variants(topology_id)
        self.topology_variant_dropdown.options = [(value, value) for value in variants]
        self.topology_variant_dropdown.value = variants[0]
        self._sync_topology_help()
        self._refresh_launch_preview()

    def _on_model_directory_selected(self, change) -> None:
        value = str(change["new"] or "").strip()
        if value:
            self.model_directory_text.value = value

    def _on_session_selected(self, change) -> None:
        value = str(change["new"] or "").strip()
        if not value:
            return
        if self._sync_selected_session_runtime_paths():
            self._refresh_log()

    def _on_workflow_changed(self, _change) -> None:
        self._refresh_models()
        self._refresh_launch_preview()

    def _on_auto_refresh_toggle(self, change) -> None:
        enabled = bool(change["new"])
        if enabled:
            self._stop_auto_refresh()
            self._refresh_runtime_output()
            stop_event = threading.Event()
            thread = threading.Thread(target=self._auto_refresh_loop, args=(stop_event,), daemon=True)
            self._auto_refresh_state["thread"] = thread
            self._auto_refresh_state["stop_event"] = stop_event
            thread.start()
            self._append_action(f"Auto refresh enabled ({max(0.5, float(self.poll_interval_input.value)):.1f}s).")
        else:
            self._stop_auto_refresh()
            self._append_action("Auto refresh disabled.")

    def _on_launch_clicked(self, _button) -> None:
        self._clear_status()
        try:
            plan, _preview = self._active_launch_plan()
            launch_session(
                plan.session_name,
                plan.command,
                plan.log_path,
                working_directory=plan.working_directory,
            )
            self._session_runtime_state["last_launched"] = plan.session_name
            self._preview_state["refreshing"] = True
            try:
                self._model_directory_state["last_auto"] = plan.model_directory
                self._run_id_state["last_auto"] = plan.run_id
                self._session_name_state["last_auto"] = plan.session_name
                self._run_dir_state["last_auto"] = plan.run_dir
                self._log_path_state["last_auto"] = plan.log_path
                self.model_directory_text.value = plan.model_directory
                self.run_id_text.value = plan.run_id
                self.session_name_text.value = plan.session_name
                self.run_dir_text.value = plan.run_dir
                self.log_path_text.value = plan.log_path
            finally:
                self._preview_state["refreshing"] = False
            self._refresh_sessions(selected=plan.session_name)
            self._refresh_log()
            self._append_action(f"Launched tmux session: {plan.session_name}")
            self._append_action(f"Run dir: {plan.run_dir}")
            self._append_action(f"Log path: {plan.log_path}")
            self._append_action(f"Attach with: tmux attach -t {plan.session_name}")
            self._set_banner(self.status_html, f"Training launched in tmux session {plan.session_name}", ok=True, label="Success")
        except Exception:
            self._append_action(traceback.format_exc())
            self._refresh_launch_preview()

    def _on_evaluate_clicked(self, _button) -> None:
        self._clear_status()
        try:
            run_dir_text_value = str(self.run_dir_text.value or "").strip()
            if not run_dir_text_value:
                raise ValueError("Run Dir cannot be blank for evaluation.")
            summary = evaluate_saved_run(
                EvalConfig(
                    model_run_directory=run_dir_text_value,
                    training_dataset=str(self.train_dataset_dropdown.value or "").strip() or None,
                    validation_dataset=str(self.validation_dataset_dropdown.value or "").strip() or None,
                    batch_size=int(self.batch_size.value),
                    roi_width_px=int(self.roi_width.value),
                    roi_height_px=int(self.roi_height.value),
                    device=str(self.device_text.value or "").strip() or None,
                    evaluation_max_visual_examples=int(self.evaluation_max_visual_examples.value),
                )
            )
            self._append_action(str(summary))
            self._set_banner(self.status_html, f"Evaluation completed using checkpoint {summary.get('checkpoint_path')}", ok=True, label="Success")
        except Exception:
            self._append_action(traceback.format_exc())

    def _on_end_session_clicked(self, _button) -> None:
        self._clear_status()
        candidates: list[str] = []
        for raw in [self.sessions_dropdown.value, self._session_runtime_state.get("last_launched"), self.session_name_text.value]:
            text = str(raw or "").strip()
            if text and text not in candidates:
                candidates.append(text)
        try:
            if not candidates:
                raise ValueError("Select an active tmux session or enter a session name.")
            for session_name in candidates:
                if end_session(session_name):
                    if self._session_runtime_state.get("last_launched") == session_name:
                        self._session_runtime_state["last_launched"] = ""
                    self._append_action(f"Ended tmux session: {session_name}")
                    self._set_banner(self.status_html, f"Ended tmux session {session_name}", ok=True, label="Success")
                    self._refresh_sessions()
                    self._refresh_launch_preview()
                    return
            raise ValueError(f"Session not found. Tried: {', '.join(candidates)}")
        except Exception:
            self._append_action(traceback.format_exc())

    def _on_clear_output_clicked(self, _button) -> None:
        self.action_output.clear_output()
        self.log_tail_output.value = ""
        self.epoch_summary_output.value = ""
        self.preview_html.value = ""
        self.status_html.value = ""


def display_training_control_panel_v03(
    start: Path | None = None,
    *,
    python_executable: str | None = None,
) -> RoiFcnTrainingControlPanelV03:
    """Locate the training root, display the v0.3 control panel, and return the panel instance."""
    training_root = find_training_root(start)
    panel = RoiFcnTrainingControlPanelV03(training_root, python_executable=python_executable)
    display(panel.widget)
    return panel


__all__ = ["RoiFcnTrainingControlPanelV03", "display_training_control_panel_v03"]
