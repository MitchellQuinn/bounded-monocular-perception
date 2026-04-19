"""Thin notebook control surface for ROI-FCN training v0.1."""

from __future__ import annotations

import html
from pathlib import Path
import traceback

import ipywidgets as widgets
from IPython.display import display

from .config import EvalConfig, TrainConfig
from .discovery import discover_dataset_references
from .evaluate import evaluate_saved_run
from .paths import find_training_root
from .topologies import get_topology_definition, list_topology_ids, list_topology_variants
from .train import train_roi_fcn

WIDGETS_UI_BUILD_V01 = "2026-04-19-roi-fcn-training-v0.1"


class RoiFcnTrainingControlPanelV01:
    """Thin operator-facing notebook launcher for ROI-FCN training."""

    def __init__(self, training_root: Path) -> None:
        self.training_root = find_training_root(training_root)

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
        self.topology_help = widgets.HTML()

        self.model_name_text = widgets.Text(description="Model Name", value="roi-fcn-tiny")
        self.run_id_text = widgets.Text(description="Run ID", value="")
        self.device_text = widgets.Text(description="Device", value="", placeholder="blank -> auto CUDA; CPU fallback disabled for training")
        self.batch_size = widgets.BoundedIntText(description="Batch Size", value=16, min=1)
        self.epochs = widgets.BoundedIntText(description="Epochs", value=8, min=1)
        self.learning_rate = widgets.FloatText(description="LR", value=1e-3)
        self.weight_decay = widgets.FloatText(description="Weight Decay", value=1e-5)
        self.gaussian_sigma = widgets.FloatText(description="Sigma Px", value=2.5)
        self.early_stopping_patience = widgets.BoundedIntText(description="Patience", value=4, min=1)
        self.roi_width = widgets.BoundedIntText(description="ROI Width", value=300, min=1)
        self.roi_height = widgets.BoundedIntText(description="ROI Height", value=300, min=1)

        self.run_dir_text = widgets.Text(
            description="Run Dir",
            placeholder="Absolute or repo-relative run directory for evaluation",
        )
        self.train_button = widgets.Button(description="Train ROI FCN", button_style="primary")
        self.evaluate_button = widgets.Button(description="Evaluate Run")
        self.clear_log_button = widgets.Button(description="Clear Log")
        self.log_output = widgets.Output(layout=widgets.Layout(height="320px", overflow_y="auto"))
        self.final_verdict_html = widgets.HTML()

        self.refresh_datasets_button.on_click(self._on_refresh_datasets)
        self.topology_id_dropdown.observe(self._on_topology_id_changed, names="value")
        self.train_button.on_click(self._on_train_clicked)
        self.evaluate_button.on_click(self._on_evaluate_clicked)
        self.clear_log_button.on_click(self._on_clear_log_clicked)

        self._refresh_datasets()
        self._sync_topology_help()

    @property
    def widget(self) -> widgets.Widget:
        return widgets.VBox(
            [
                widgets.HTML(f"<b>ROI-FCN Training Control Panel</b> <code>{WIDGETS_UI_BUILD_V01}</code>"),
                widgets.HBox([self.train_dataset_dropdown, self.validation_dataset_dropdown, self.refresh_datasets_button]),
                widgets.HBox([self.topology_id_dropdown, self.topology_variant_dropdown]),
                self.topology_help,
                widgets.HBox([self.model_name_text, self.run_id_text, self.device_text]),
                widgets.HBox([self.batch_size, self.epochs, self.learning_rate]),
                widgets.HBox([self.weight_decay, self.gaussian_sigma, self.early_stopping_patience]),
                widgets.HBox([self.roi_width, self.roi_height]),
                self.run_dir_text,
                widgets.HBox([self.train_button, self.evaluate_button, self.clear_log_button]),
                self.final_verdict_html,
                self.log_output,
            ]
        )

    def _append_log(self, message: str) -> None:
        with self.log_output:
            print(message)

    def _set_verdict(self, message: str, *, ok: bool) -> None:
        color = "#0b6f3c" if ok else "#8c1d18"
        self.final_verdict_html.value = (
            f"<div style='padding:8px 10px;border-left:4px solid {color};'>"
            f"<b>{'Success' if ok else 'Failure'}</b><br><code>{html.escape(message)}</code></div>"
        )

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

    def _on_refresh_datasets(self, _button) -> None:
        self._refresh_datasets()
        self._append_log("Refreshed dataset references.")

    def _on_topology_id_changed(self, _change) -> None:
        topology_id = str(self.topology_id_dropdown.value or "").strip()
        variants = list_topology_variants(topology_id)
        self.topology_variant_dropdown.options = [(value, value) for value in variants]
        self.topology_variant_dropdown.value = variants[0]
        self._sync_topology_help()

    def _on_clear_log_clicked(self, _button) -> None:
        self.log_output.clear_output()
        self.final_verdict_html.value = ""

    def _on_train_clicked(self, _button) -> None:
        self.log_output.clear_output()
        self.final_verdict_html.value = ""
        try:
            config = TrainConfig(
                training_dataset=str(self.train_dataset_dropdown.value or "").strip(),
                validation_dataset=str(self.validation_dataset_dropdown.value or "").strip(),
                topology_id=str(self.topology_id_dropdown.value or "").strip(),
                topology_variant=str(self.topology_variant_dropdown.value or "").strip(),
                model_name=str(self.model_name_text.value or "").strip() or "roi-fcn-tiny",
                run_id=str(self.run_id_text.value or "").strip() or None,
                device=str(self.device_text.value or "").strip() or None,
                batch_size=int(self.batch_size.value),
                epochs=int(self.epochs.value),
                learning_rate=float(self.learning_rate.value),
                weight_decay=float(self.weight_decay.value),
                gaussian_sigma_px=float(self.gaussian_sigma.value),
                early_stopping_patience=int(self.early_stopping_patience.value),
                roi_width_px=int(self.roi_width.value),
                roi_height_px=int(self.roi_height.value),
            )
            summary = train_roi_fcn(config, log_sink=self._append_log)
            self.run_dir_text.value = str(summary.get("run_dir", ""))
            self._set_verdict(
                f"Training completed. Best epoch={summary.get('best_epoch')} run_dir={summary.get('run_dir')}",
                ok=True,
            )
        except Exception as exc:
            self._append_log(traceback.format_exc())
            self._set_verdict(str(exc), ok=False)

    def _on_evaluate_clicked(self, _button) -> None:
        self.log_output.clear_output()
        self.final_verdict_html.value = ""
        try:
            run_dir_text = str(self.run_dir_text.value or "").strip()
            if not run_dir_text:
                raise ValueError("Run Dir cannot be blank for evaluation.")
            config = EvalConfig(
                model_run_directory=run_dir_text,
                training_dataset=str(self.train_dataset_dropdown.value or "").strip() or None,
                validation_dataset=str(self.validation_dataset_dropdown.value or "").strip() or None,
                batch_size=int(self.batch_size.value),
                roi_width_px=int(self.roi_width.value),
                roi_height_px=int(self.roi_height.value),
                device=str(self.device_text.value or "").strip() or None,
            )
            summary = evaluate_saved_run(config)
            self._append_log(str(summary))
            self._set_verdict(
                f"Evaluation completed using checkpoint {summary.get('checkpoint_path')}",
                ok=True,
            )
        except Exception as exc:
            self._append_log(traceback.format_exc())
            self._set_verdict(str(exc), ok=False)


def display_training_control_panel_v01(start: Path | None = None) -> RoiFcnTrainingControlPanelV01:
    """Locate the training root and display the ROI-FCN control panel."""
    training_root = find_training_root(start)
    panel = RoiFcnTrainingControlPanelV01(training_root)
    display(panel.widget)
    return panel
