"""Notebook widget helpers for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

import html
from pathlib import Path
import traceback

import ipywidgets as widgets
from IPython.display import display

from .config import BootstrapCenterTargetConfig, PackRoiFcnConfig
from .discovery import discover_dataset_references
from .paths import dataset_output_root, find_preprocessing_root
from .pipeline import run_preprocessing_for_dataset

WIDGETS_UI_BUILD_V01 = "2026-04-19-roi-fcn-preprocessing-v0.1-speed-pass"


class RoiFcnPreprocessingLauncherV01:
    """Thin notebook control surface for ROI-FCN preprocessing."""

    def __init__(self, preprocessing_root: Path) -> None:
        self.preprocessing_root = find_preprocessing_root(preprocessing_root)

        default_bootstrap_config = BootstrapCenterTargetConfig()
        default_pack_config = PackRoiFcnConfig()

        self.dataset_dropdown = widgets.Dropdown(description="Dataset")
        self.refresh_button = widgets.Button(description="Refresh Datasets", button_style="")
        self.output_path_html = widgets.HTML()

        self.detector_backend_dropdown = widgets.Dropdown(
            description="Backend",
            options=[("edge_roi_v1", "edge_roi_v1")],
            value="edge_roi_v1",
        )
        self.edge_blur_k = widgets.BoundedIntText(description="edge_blur_k", value=5, min=1)
        self.edge_low = widgets.BoundedIntText(description="edge_low", value=50, min=0, max=255)
        self.edge_high = widgets.BoundedIntText(description="edge_high", value=150, min=0, max=255)
        self.fg_threshold = widgets.BoundedIntText(description="fg_threshold", value=250, min=0, max=255)
        self.edge_pad = widgets.BoundedIntText(description="edge_pad", value=0, min=0)
        self.min_edge_pixels = widgets.BoundedIntText(description="min_edge_pixels", value=16, min=1)

        self.canvas_width = widgets.IntText(description="canvas_width", value=480)
        self.canvas_height = widgets.IntText(description="canvas_height", value=300)
        self.shard_size = widgets.IntText(description="shard_size", value=8192)
        self.cpu_workers = widgets.BoundedIntText(
            description="cpu_workers",
            value=int(default_bootstrap_config.num_workers),
            min=1,
        )
        self.compress_shards_checkbox = widgets.Checkbox(
            description="compress npz shards",
            value=bool(default_pack_config.compress),
            indent=False,
        )
        self.overwrite_output_checkbox = widgets.Checkbox(
            description="overwrite existing output",
            value=False,
            indent=False,
        )

        self.run_button = widgets.Button(description="Run Preprocessing", button_style="primary")
        self.clear_log_button = widgets.Button(description="Clear Log", button_style="")
        self.log_output = widgets.Output(layout=widgets.Layout(height="320px", overflow_y="auto"))
        self.final_verdict_html = widgets.HTML()

        self.refresh_button.on_click(self._on_refresh_clicked)
        self.run_button.on_click(self._on_run_clicked)
        self.clear_log_button.on_click(self._on_clear_log_clicked)
        self.dataset_dropdown.observe(self._on_dataset_changed, names="value")

        self._refresh_dataset_options()

    @property
    def widget(self) -> widgets.Widget:
        controls = widgets.VBox(
            [
                widgets.HTML(
                    f"<b>ROI-FCN Preprocessing Launcher (v0.1)</b> <code>{WIDGETS_UI_BUILD_V01}</code>"
                ),
                widgets.HBox([self.dataset_dropdown, self.refresh_button]),
                self.output_path_html,
                widgets.HTML("<b>Detection Controls</b>"),
                self.detector_backend_dropdown,
                widgets.HBox([self.edge_blur_k, self.edge_low, self.edge_high]),
                widgets.HBox([self.fg_threshold, self.edge_pad, self.min_edge_pixels]),
                widgets.HTML("<b>Packing Controls</b>"),
                widgets.HBox([self.canvas_width, self.canvas_height, self.shard_size]),
                widgets.HTML("<b>Execution Controls</b>"),
                widgets.HBox([self.cpu_workers, self.compress_shards_checkbox]),
                self.overwrite_output_checkbox,
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
        self._update_output_path()

    def _update_output_path(self) -> None:
        dataset_name = str(self.dataset_dropdown.value or "").strip()
        if not dataset_name:
            self.output_path_html.value = "<b>Output Path:</b> <code>&lt;select a dataset&gt;</code>"
            return
        output_path = dataset_output_root(self.preprocessing_root, dataset_name)
        self.output_path_html.value = f"<b>Output Path:</b> <code>{html.escape(str(output_path))}</code>"

    def _on_dataset_changed(self, _change) -> None:
        self._update_output_path()

    def _on_refresh_clicked(self, _button) -> None:
        self._refresh_dataset_options()
        self._append_log("Refreshed dataset list.")

    def _on_clear_log_clicked(self, _button) -> None:
        self.log_output.clear_output()
        self.final_verdict_html.value = ""

    def _on_run_clicked(self, _button) -> None:
        dataset_name = str(self.dataset_dropdown.value or "").strip()
        if not dataset_name:
            self._set_verdict("No valid dataset selected.", ok=False)
            return

        self.log_output.clear_output()
        self.final_verdict_html.value = ""

        overwrite_enabled = bool(self.overwrite_output_checkbox.value)
        worker_count = int(self.cpu_workers.value)

        bootstrap_config = BootstrapCenterTargetConfig(
            detector_backend=str(self.detector_backend_dropdown.value),
            edge_blur_k=int(self.edge_blur_k.value),
            edge_low=int(self.edge_low.value),
            edge_high=int(self.edge_high.value),
            fg_threshold=int(self.fg_threshold.value),
            edge_pad=int(self.edge_pad.value),
            min_edge_pixels=int(self.min_edge_pixels.value),
            overwrite=overwrite_enabled,
            num_workers=worker_count,
        )
        pack_config = PackRoiFcnConfig(
            canvas_width=int(self.canvas_width.value),
            canvas_height=int(self.canvas_height.value),
            shard_size=int(self.shard_size.value),
            compress=bool(self.compress_shards_checkbox.value),
            overwrite=overwrite_enabled,
            num_workers=worker_count,
        )

        try:
            summary = run_preprocessing_for_dataset(
                self.preprocessing_root,
                dataset_name,
                bootstrap_config=bootstrap_config,
                pack_config=pack_config,
                log_sink=self._append_log,
            )
            self._set_verdict(
                f"Completed dataset '{dataset_name}' with {len(summary.stage_summaries)} stage summaries.",
                ok=True,
            )
        except Exception as exc:
            self._append_log(traceback.format_exc())
            self._set_verdict(str(exc), ok=False)


def display_preprocessing_launcher_v01(start: Path | None = None) -> RoiFcnPreprocessingLauncherV01:
    """Locate the preprocessing root and display the launcher widget."""

    preprocessing_root = find_preprocessing_root(start)
    launcher = RoiFcnPreprocessingLauncherV01(preprocessing_root)
    display(launcher.widget)
    return launcher
