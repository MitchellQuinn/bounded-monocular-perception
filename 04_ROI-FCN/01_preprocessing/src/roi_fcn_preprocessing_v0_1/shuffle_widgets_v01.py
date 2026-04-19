"""Notebook widget helpers for ROI-FCN input-corpora shuffle."""

from __future__ import annotations

import html
from pathlib import Path
import traceback

import ipywidgets as widgets
from IPython.display import display

from .discovery import discover_dataset_references
from .input_corpora_shuffle import (
    default_shuffled_dataset_reference,
    parse_shuffle_seed,
    shuffle_input_dataset_corpora,
)
from .paths import dataset_input_root, find_preprocessing_root

SHUFFLE_WIDGETS_UI_BUILD_V01 = "2026-04-19-roi-fcn-corpora-shuffle-v0.1"


class RoiFcnCorporaShuffleLauncherV01:
    """Thin notebook control surface for ROI-FCN input-corpora shuffle."""

    def __init__(self, preprocessing_root: Path) -> None:
        self.preprocessing_root = find_preprocessing_root(preprocessing_root)

        self.dataset_dropdown = widgets.Dropdown(description="Dataset")
        self.refresh_button = widgets.Button(description="Refresh Datasets", button_style="")
        self.seed_text = widgets.Text(description="Seed", value="13")
        self.destination_text = widgets.Text(description="Output Name")
        self.source_path_html = widgets.HTML()
        self.destination_path_html = widgets.HTML()
        self.run_button = widgets.Button(description="Shuffle Corpora", button_style="primary")
        self.clear_log_button = widgets.Button(description="Clear Log", button_style="")
        self.log_output = widgets.Output(layout=widgets.Layout(height="320px", overflow_y="auto"))
        self.final_verdict_html = widgets.HTML()

        self.refresh_button.on_click(self._on_refresh_clicked)
        self.run_button.on_click(self._on_run_clicked)
        self.clear_log_button.on_click(self._on_clear_log_clicked)
        self.dataset_dropdown.observe(self._on_dataset_changed, names="value")
        self.destination_text.observe(self._on_destination_changed, names="value")

        self._refresh_dataset_options()

    @property
    def widget(self) -> widgets.Widget:
        controls = widgets.VBox(
            [
                widgets.HTML(
                    "<b>ROI-FCN Input Corpora Shuffle (v0.1)</b> "
                    f"<code>{SHUFFLE_WIDGETS_UI_BUILD_V01}</code>"
                ),
                widgets.HTML(
                    "Shuffle both <code>train</code> and <code>validate</code> corpora into a new "
                    "sibling dataset under <code>input/</code>. "
                    "Rows with <code>capture_success=false</code> are excluded from the shuffled copy."
                ),
                widgets.HBox([self.dataset_dropdown, self.refresh_button]),
                self.source_path_html,
                widgets.HBox([self.seed_text, self.destination_text]),
                self.destination_path_html,
                widgets.HBox([self.run_button, self.clear_log_button]),
                self.final_verdict_html,
            ]
        )
        return widgets.VBox([controls, self.log_output])

    def _append_log(self, message: str) -> None:
        with self.log_output:
            print(message)

    def _set_verdict(self, message: str, *, ok: bool) -> None:
        color = "#0b6f3c" if ok else "#8c1d18"
        self.final_verdict_html.value = (
            f"<div style='padding:8px 10px;border-left:4px solid {color};'>"
            f"<b>{'Success' if ok else 'Failure'}</b><br><code>{html.escape(message)}</code></div>"
        )

    def _refresh_dataset_options(self) -> None:
        discovered = discover_dataset_references(self.preprocessing_root)
        options = [(dataset.name, dataset.name) for dataset in discovered]
        if not options:
            options = [("<no valid datasets discovered>", "")]
        current = self.dataset_dropdown.value
        self.dataset_dropdown.options = options
        valid_values = {value for _, value in options}
        self.dataset_dropdown.value = current if current in valid_values else options[0][1]
        self._sync_destination_name()
        self._update_paths()

    def _sync_destination_name(self) -> None:
        dataset_name = str(self.dataset_dropdown.value or "").strip()
        self.destination_text.value = default_shuffled_dataset_reference(dataset_name) if dataset_name else ""

    def _update_paths(self) -> None:
        dataset_name = str(self.dataset_dropdown.value or "").strip()
        destination_name = str(self.destination_text.value or "").strip()

        if not dataset_name:
            self.source_path_html.value = "<b>Source Path:</b> <code>&lt;select a dataset&gt;</code>"
        else:
            source_path = dataset_input_root(self.preprocessing_root, dataset_name)
            self.source_path_html.value = f"<b>Source Path:</b> <code>{html.escape(str(source_path))}</code>"

        if not destination_name:
            self.destination_path_html.value = "<b>Destination Path:</b> <code>&lt;enter an output name&gt;</code>"
        else:
            destination_path = dataset_input_root(self.preprocessing_root, destination_name)
            self.destination_path_html.value = (
                f"<b>Destination Path:</b> <code>{html.escape(str(destination_path))}</code>"
            )

    def _on_dataset_changed(self, _change) -> None:
        self._sync_destination_name()
        self._update_paths()

    def _on_destination_changed(self, _change) -> None:
        self._update_paths()

    def _on_refresh_clicked(self, _button) -> None:
        self._refresh_dataset_options()
        self._append_log("Refreshed dataset list.")

    def _on_clear_log_clicked(self, _button) -> None:
        self.log_output.clear_output()
        self.final_verdict_html.value = ""

    def _on_run_clicked(self, _button) -> None:
        dataset_name = str(self.dataset_dropdown.value or "").strip()
        destination_name = str(self.destination_text.value or "").strip()
        if not dataset_name:
            self._set_verdict("No valid dataset selected.", ok=False)
            return

        self.log_output.clear_output()
        self.final_verdict_html.value = ""

        try:
            validated_seed = parse_shuffle_seed(self.seed_text.value)
            self._append_log(
                f"Shuffling dataset '{dataset_name}' to '{destination_name or '<default>'}' with seed {validated_seed}."
            )
            summary = shuffle_input_dataset_corpora(
                self.preprocessing_root,
                dataset_name,
                validated_seed,
                destination_dataset_reference=destination_name or None,
            )
            for split_result in summary.split_results:
                self._append_log(
                    f"[{split_result.split_name}] wrote {split_result.output_row_count} rows "
                    f"(excluded capture_success=false: {split_result.excluded_capture_failed_rows})"
                )
            self._set_verdict(
                f"Created shuffled dataset '{summary.destination_dataset_reference}'.",
                ok=True,
            )
            self._refresh_dataset_options()
        except Exception as exc:
            self._append_log(traceback.format_exc())
            self._set_verdict(str(exc), ok=False)


def display_corpora_shuffle_launcher_v01(
    start: Path | None = None,
) -> RoiFcnCorporaShuffleLauncherV01:
    """Locate the preprocessing root and display the corpora-shuffle widget."""

    preprocessing_root = find_preprocessing_root(start)
    launcher = RoiFcnCorporaShuffleLauncherV01(preprocessing_root)
    display(launcher.widget)
    return launcher
