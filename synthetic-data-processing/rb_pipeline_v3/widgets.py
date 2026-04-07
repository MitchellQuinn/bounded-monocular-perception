"""ipywidgets-based notebook UI components for the RB v3 threshold pipeline."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from rb_pipeline.image_io import read_grayscale_uint8
from rb_pipeline.paths import list_input_runs, resolve_manifest_path

from .config import NpyPackStageConfigV3, ShuffleStageConfigV3, ThresholdStageConfigV3
from .manifest import load_samples_csv, samples_csv_path
from .paths import (
    find_project_root,
    input_run_paths,
    list_training_v3_runs,
    threshold_run_paths,
    training_v3_run_paths,
)
from .pipeline import STAGE_ORDER_V3, run_v3_stage_for_run
from .shuffle_stage import run_shuffle_stage_v3
from .threshold_stage import _filter_components, _gimp_threshold_binary, _render_threshold_artifact



def _load_samples(samples_path: Path) -> pd.DataFrame | None:
    if not samples_path.is_file():
        return None
    return pd.read_csv(samples_path)



def _sample_options(samples_df: pd.DataFrame | None) -> list[tuple[str, int]]:
    if samples_df is None or len(samples_df) == 0:
        return []

    options: list[tuple[str, int]] = []
    for row_idx in samples_df.index:
        sample_id = str(samples_df.at[row_idx, "sample_id"]) if "sample_id" in samples_df.columns else f"row_{row_idx}"
        image_filename = str(samples_df.at[row_idx, "image_filename"]) if "image_filename" in samples_df.columns else ""
        label = f"{sample_id} | {image_filename}"
        options.append((label, int(row_idx)))
    return options



def _safe_gray_image(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    try:
        return read_grayscale_uint8(path)
    except Exception:
        return None



def _safe_npy(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        return None


_TRAINING_IMAGE_SOURCE_OPTIONS = [
    ("Threshold output", "threshold_image_filename"),
    ("Threshold debug (raw binary)", "threshold_debug_binary_filename"),
    ("Threshold debug (selected component)", "threshold_debug_selected_component_filename"),
    ("Threshold debug (amalgamated)", "threshold_debug_amalgamated_filename"),
]



def _draw_preview_grid_v3(
    source_img: np.ndarray | None,
    threshold_debug_img: np.ndarray | None,
    threshold_img: np.ndarray | None,
    training_array: np.ndarray | None,
) -> None:
    if training_array is None:
        training_range = (0.0, 1.0)
    elif training_array.dtype.kind == "f":
        training_range = (0.0, 1.0)
    elif training_array.dtype.kind == "b":
        training_range = (0, 1)
    else:
        training_range = (0, 255)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        (axes[0, 0], source_img, "Source Image", (0, 255)),
        (axes[0, 1], threshold_debug_img, "Threshold Debug", (0, 255)),
        (axes[1, 0], threshold_img, "Threshold Output", (0, 255)),
        (axes[1, 1], training_array, "Training Array", training_range),
    ]

    for axis, image, title, value_range in plots:
        axis.set_title(title)
        axis.axis("off")

        if image is None:
            axis.text(0.5, 0.5, "Not available", ha="center", va="center")
            continue

        vmin, vmax = value_range
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.show()



def _draw_three_panel_preview(
    left: np.ndarray | None,
    center: np.ndarray | None,
    right: np.ndarray | None,
    left_title: str,
    center_title: str,
    right_title: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [(axes[0], left, left_title), (axes[1], center, center_title), (axes[2], right, right_title)]

    for axis, image, title in panels:
        axis.set_title(title)
        axis.axis("off")
        if image is None:
            axis.text(0.5, 0.5, "Not available", ha="center", va="center")
            continue

        vmin = 0.0 if image.dtype.kind == "f" else 0
        vmax = 1.0 if image.dtype.kind == "f" else 255
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)

    plt.tight_layout()
    plt.show()


class PreviewPanelV3:
    """Shared 4-panel preview: source, threshold debug, threshold output, training array."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

        self.run_dropdown = widgets.Dropdown(description="Run:", options=[])
        self.sample_dropdown = widgets.Dropdown(description="Sample:", options=[])

        self.refresh_runs_button = widgets.Button(description="Refresh Runs")
        self.refresh_preview_button = widgets.Button(description="Refresh Preview")

        self.output = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px"))

        self.refresh_runs_button.on_click(self._on_refresh_runs)
        self.refresh_preview_button.on_click(self._on_refresh_preview)
        self.run_dropdown.observe(self._on_run_change, names="value")

        self._refresh_runs()

    @property
    def widget(self) -> widgets.Widget:
        return widgets.VBox(
            [
                widgets.HTML("<b>Preview Panel (V3)</b>"),
                self.run_dropdown,
                self.sample_dropdown,
                widgets.HBox([self.refresh_runs_button, self.refresh_preview_button]),
                self.output,
            ]
        )

    def _on_refresh_runs(self, _button: widgets.Button) -> None:
        self._refresh_runs()

    def _on_run_change(self, _change: dict) -> None:
        self._refresh_samples()

    def _on_refresh_preview(self, _button: widgets.Button) -> None:
        self.render_preview()

    def _refresh_runs(self) -> None:
        runs = list_input_runs(self.project_root)
        self.run_dropdown.options = runs
        if runs and self.run_dropdown.value not in runs:
            self.run_dropdown.value = runs[0]
        self._refresh_samples()

    def _refresh_samples(self) -> None:
        run_name = self.run_dropdown.value
        if not run_name:
            self.sample_dropdown.options = []
            return

        input_paths = input_run_paths(self.project_root, run_name)
        samples_df = _load_samples(input_paths.manifests_dir / "samples.csv")
        options = _sample_options(samples_df)

        self.sample_dropdown.options = options
        if options:
            option_values = [value for _, value in options]
            if self.sample_dropdown.value not in option_values:
                self.sample_dropdown.value = option_values[0]

    def render_preview(self) -> None:
        run_name = self.run_dropdown.value
        row_idx = self.sample_dropdown.value

        with self.output:
            self.output.clear_output(wait=True)

            if run_name is None or row_idx is None:
                print("Select a run and sample first.")
                return

            input_paths = input_run_paths(self.project_root, run_name)
            threshold_paths = threshold_run_paths(self.project_root, run_name)
            training_paths = training_v3_run_paths(self.project_root, run_name)

            input_df = _load_samples(input_paths.manifests_dir / "samples.csv")
            threshold_df = _load_samples(threshold_paths.manifests_dir / "samples.csv")
            training_df = _load_samples(training_paths.manifests_dir / "samples.csv")

            if input_df is None or row_idx not in input_df.index:
                print("Sample not available in input manifest.")
                return

            source_path = None
            threshold_debug_path = None
            threshold_path = None
            npy_path = None

            try:
                source_filename = input_df.at[row_idx, "image_filename"]
                source_path = resolve_manifest_path(input_paths.root, "images", source_filename)
            except Exception:
                source_path = None

            if threshold_df is not None and row_idx in threshold_df.index:
                for debug_column in [
                    "threshold_debug_selected_component_filename",
                    "threshold_debug_binary_filename",
                    "threshold_debug_amalgamated_filename",
                ]:
                    if debug_column not in threshold_df.columns:
                        continue
                    try:
                        debug_filename = threshold_df.at[row_idx, debug_column]
                        if str(debug_filename).strip():
                            threshold_debug_path = resolve_manifest_path(
                                threshold_paths.root, "images", debug_filename
                            )
                            break
                    except Exception:
                        continue

                if "threshold_image_filename" in threshold_df.columns:
                    try:
                        threshold_filename = threshold_df.at[row_idx, "threshold_image_filename"]
                        if str(threshold_filename).strip():
                            threshold_path = resolve_manifest_path(
                                threshold_paths.root, "images", threshold_filename
                            )
                    except Exception:
                        threshold_path = None

            if training_df is not None and row_idx in training_df.index and "npy_filename" in training_df.columns:
                try:
                    npy_filename = training_df.at[row_idx, "npy_filename"]
                    if str(npy_filename).strip():
                        npy_path = resolve_manifest_path(training_paths.root, "arrays", npy_filename)
                except Exception:
                    npy_path = None

            source_img = _safe_gray_image(source_path)
            threshold_debug_img = _safe_gray_image(threshold_debug_path)
            threshold_img = _safe_gray_image(threshold_path)
            training_array = _safe_npy(npy_path)

            _draw_preview_grid_v3(source_img, threshold_debug_img, threshold_img, training_array)


class PipelineLauncherV3:
    """Notebook 00 launcher UI for v3 run/stage orchestration."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())
        self._is_running = False

        self.run_select = widgets.SelectMultiple(description="Runs:", options=[])
        self.allow_multi_run_checkbox = widgets.Checkbox(value=False, description="Allow multi-run execution")
        self.stage_dropdown = widgets.Dropdown(
            description="Stage:",
            options=[("all", "all"), ("threshold", "threshold"), ("npy", "npy")],
            value="all",
        )

        self.overwrite_checkbox = widgets.Checkbox(value=False, description="Overwrite existing files")
        self.dry_run_checkbox = widgets.Checkbox(value=False, description="Dry run")
        self.continue_on_error_checkbox = widgets.Checkbox(value=True, description="Continue on error")

        self.mode_dropdown = widgets.Dropdown(
            description="Mode:",
            options=[("outline", "outline"), ("filled", "filled")],
            value="filled",
        )
        self.threshold_low_slider = widgets.IntSlider(description="Threshold low", value=128, min=0, max=255, step=1)
        self.threshold_high_slider = widgets.IntSlider(description="Threshold high", value=255, min=0, max=255, step=1)
        self.invert_selection_checkbox = widgets.Checkbox(value=False, description="Invert threshold selection")
        self.min_component_area_input = widgets.BoundedIntText(description="Min area:", value=1, min=1, max=1_000_000)
        self.outline_thickness_slider = widgets.IntSlider(description="Outline px", value=1, min=1, max=10, step=1)
        self.persist_debug_checkbox = widgets.Checkbox(value=False, description="Persist debug outputs")
        self.amalgamate_debug_checkbox = widgets.Checkbox(
            value=False,
            description="Amalgamate debug outputs",
        )
        self.keep_individual_debug_checkbox = widgets.Checkbox(
            value=True,
            description="Keep individual debug files",
        )
        self.sample_offset_input = widgets.BoundedIntText(description="Sample offset:", value=0, min=0, max=1_000_000)
        self.sample_limit_input = widgets.BoundedIntText(description="Sample limit:", value=0, min=0, max=1_000_000)

        self.training_source_dropdown = widgets.Dropdown(
            description="Train source:",
            options=_TRAINING_IMAGE_SOURCE_OPTIONS,
            value="threshold_image_filename",
        )
        self.npy_dtype_dropdown = widgets.Dropdown(
            description="NPY dtype:",
            options=[("float32", "float32"), ("float16", "float16"), ("uint8", "uint8")],
            value="float32",
        )
        self.pack_dtype_dropdown = widgets.Dropdown(
            description="Pack dtype:",
            options=[("preserve", "preserve"), ("float32", "float32"), ("float16", "float16"), ("uint8", "uint8")],
            value="preserve",
        )
        self.pack_compress_checkbox = widgets.Checkbox(value=True, description="Compress NPZ shards")
        self.pack_shard_size_input = widgets.BoundedIntText(
            description="Shard rows:",
            value=128,
            min=0,
            max=1_000_000,
            step=1,
        )
        self.pack_delete_npy_checkbox = widgets.Checkbox(
            value=True,
            description="Delete source NPY after pack",
        )

        self.refresh_runs_button = widgets.Button(description="Refresh Runs")
        self.execute_button = widgets.Button(description="Run Selected")

        self.progress = widgets.IntProgress(value=0, min=0, max=1, description="Progress")
        self.log_output = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px", height="280px"))

        self.preview_panel = PreviewPanelV3(self.project_root)

        self.refresh_runs_button.on_click(self._on_refresh_runs)
        self.execute_button.on_click(self._on_execute)
        self.run_select.observe(self._on_run_selection_change, names="value")
        self.allow_multi_run_checkbox.observe(self._on_allow_multi_toggle, names="value")
        self.preview_panel.run_dropdown.observe(self._on_preview_run_change, names="value")

        self._refresh_runs()

    def display(self) -> None:
        control_panel = widgets.VBox(
            [
                widgets.HTML("<b>Pipeline Launcher (V3)</b>"),
                widgets.HBox([self.refresh_runs_button, self.execute_button]),
                self.run_select,
                self.allow_multi_run_checkbox,
                self.stage_dropdown,
                self.overwrite_checkbox,
                self.dry_run_checkbox,
                self.continue_on_error_checkbox,
                widgets.HTML("<b>Threshold Parameters</b>"),
                self.mode_dropdown,
                self.threshold_low_slider,
                self.threshold_high_slider,
                self.invert_selection_checkbox,
                self.min_component_area_input,
                self.outline_thickness_slider,
                self.persist_debug_checkbox,
                self.amalgamate_debug_checkbox,
                self.keep_individual_debug_checkbox,
                self.sample_offset_input,
                self.sample_limit_input,
                widgets.HTML("<b>NPY + Pack Parameters</b>"),
                self.training_source_dropdown,
                self.npy_dtype_dropdown,
                self.pack_dtype_dropdown,
                self.pack_compress_checkbox,
                self.pack_shard_size_input,
                self.pack_delete_npy_checkbox,
                self.progress,
                self.log_output,
            ]
        )

        display(widgets.HBox([control_panel, self.preview_panel.widget]))

    def _log(self, message: str) -> None:
        with self.log_output:
            print(message)

    def _refresh_runs(self) -> None:
        runs = list_input_runs(self.project_root)
        self.run_select.options = runs
        if not runs:
            self.run_select.value = ()
            return

        if self.allow_multi_run_checkbox.value:
            current = tuple(run for run in self.run_select.value if run in runs)
            self.run_select.value = current if current else (runs[0],)
            return

        preview_run = self.preview_panel.run_dropdown.value
        if preview_run in runs:
            self.run_select.value = (preview_run,)
            return

        current = tuple(run for run in self.run_select.value if run in runs)
        if len(current) == 1:
            self.run_select.value = current
            return

        self.run_select.value = (runs[0],)

    def _on_refresh_runs(self, _button: widgets.Button) -> None:
        self._refresh_runs()
        self.preview_panel._refresh_runs()
        preview_run = self.preview_panel.run_dropdown.value
        if preview_run in self.run_select.options and not self.allow_multi_run_checkbox.value:
            self.run_select.value = (preview_run,)

    def _on_run_selection_change(self, change: dict) -> None:
        new_selection = tuple(change.get("new", ()))
        if self.allow_multi_run_checkbox.value:
            if len(new_selection) == 1 and self.preview_panel.run_dropdown.value != new_selection[0]:
                self.preview_panel.run_dropdown.value = new_selection[0]
            return

        if not new_selection:
            return

        if len(new_selection) > 1:
            old_selection = tuple(change.get("old", ()))
            newly_selected = [run for run in new_selection if run not in old_selection]
            chosen = newly_selected[-1] if newly_selected else new_selection[-1]
            if self.run_select.value != (chosen,):
                self.run_select.value = (chosen,)
            return

        chosen = new_selection[0]
        if self.preview_panel.run_dropdown.value != chosen:
            self.preview_panel.run_dropdown.value = chosen

    def _on_allow_multi_toggle(self, change: dict) -> None:
        allow_multi = bool(change.get("new", False))
        if allow_multi:
            return

        if not self.run_select.value:
            return

        preferred = self.preview_panel.run_dropdown.value
        if preferred in self.run_select.value:
            self.run_select.value = (preferred,)
        else:
            self.run_select.value = (self.run_select.value[-1],)

    def _on_preview_run_change(self, change: dict) -> None:
        run_name = change.get("new")
        if self.allow_multi_run_checkbox.value:
            return
        if run_name in self.run_select.options and self.run_select.value != (run_name,):
            self.run_select.value = (run_name,)

    def _build_threshold_config(self) -> ThresholdStageConfigV3:
        return ThresholdStageConfigV3(
            representation_mode=self.mode_dropdown.value,
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            threshold_low_value=self.threshold_low_slider.value,
            threshold_high_value=self.threshold_high_slider.value,
            invert_selection=self.invert_selection_checkbox.value,
            min_component_area_px=self.min_component_area_input.value,
            outline_thickness=self.outline_thickness_slider.value,
            persist_debug=self.persist_debug_checkbox.value,
            keep_individual_debug_outputs=self.keep_individual_debug_checkbox.value,
            amalgamate_debug_outputs=self.amalgamate_debug_checkbox.value,
            sample_offset=self.sample_offset_input.value,
            sample_limit=self.sample_limit_input.value,
        )

    def _build_npy_pack_config(self) -> NpyPackStageConfigV3:
        return NpyPackStageConfigV3(
            representation_mode=self.mode_dropdown.value,
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            normalize=True,
            invert=True,
            npy_output_dtype=self.npy_dtype_dropdown.value,
            pack_output_dtype=self.pack_dtype_dropdown.value,
            compress=self.pack_compress_checkbox.value,
            shard_size=self.pack_shard_size_input.value,
            delete_source_npy_after_pack=self.pack_delete_npy_checkbox.value,
            include_optional_filename_arrays=True,
            training_image_source_column=self.training_source_dropdown.value,
        )

    def _on_execute(self, _button: widgets.Button) -> None:
        if self._is_running:
            self._log("Execution is already running. Wait for the current run to finish.")
            return

        self._is_running = True
        self.execute_button.disabled = True
        self.refresh_runs_button.disabled = True
        self.log_output.clear_output(wait=True)

        selected_runs = list(self.run_select.value)
        if not selected_runs:
            self._log("Select at least one run.")
            self._is_running = False
            self.execute_button.disabled = False
            self.refresh_runs_button.disabled = False
            return

        if not self.allow_multi_run_checkbox.value and len(selected_runs) > 1:
            preferred = self.preview_panel.run_dropdown.value
            selected_runs = [preferred] if preferred in selected_runs else [selected_runs[0]]

        selected_stage = self.stage_dropdown.value
        stages = STAGE_ORDER_V3 if selected_stage == "all" else [selected_stage]

        self._log(f"Selected runs: {', '.join(selected_runs)}")
        self._log(f"Selected stage: {selected_stage}")

        threshold_config = self._build_threshold_config()
        npy_pack_config = self._build_npy_pack_config()

        self.progress.max = max(1, len(selected_runs) * len(stages))
        self.progress.value = 0

        try:
            stop_all = False

            for run_name in selected_runs:
                if stop_all:
                    break

                self._log(f"=== Run: {run_name} ===")

                for stage_name in stages:
                    try:
                        summary = run_v3_stage_for_run(
                            self.project_root,
                            run_name,
                            stage_name,
                            threshold_config=threshold_config,
                            npy_pack_config=npy_pack_config,
                            log_sink=self._log,
                        )
                        self._log(
                            f"Completed {stage_name}: success={summary.successful_rows}, "
                            f"failed={summary.failed_rows}, skipped={summary.skipped_rows}"
                        )
                    except Exception as exc:
                        self._log(f"Stage {stage_name} failed: {exc}")
                        if not self.continue_on_error_checkbox.value:
                            stop_all = True
                    finally:
                        self.progress.value = min(self.progress.value + 1, self.progress.max)

            self._log("Pipeline execution finished.")
        finally:
            self._is_running = False
            self.execute_button.disabled = False
            self.refresh_runs_button.disabled = False


class ThresholdStagePanelV3:
    """Notebook 01 UI for threshold tuning and run execution."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())

        self.run_dropdown = widgets.Dropdown(description="Run:", options=[])
        self.sample_dropdown = widgets.Dropdown(description="Sample:", options=[])

        self.mode_dropdown = widgets.Dropdown(
            description="Mode:",
            options=[("outline", "outline"), ("filled", "filled")],
            value="filled",
        )
        self.threshold_low_slider = widgets.IntSlider(description="Threshold low", value=128, min=0, max=255, step=1)
        self.threshold_high_slider = widgets.IntSlider(description="Threshold high", value=255, min=0, max=255, step=1)
        self.invert_selection_checkbox = widgets.Checkbox(value=False, description="Invert threshold selection")
        self.min_component_area_input = widgets.BoundedIntText(description="Min area:", value=1, min=1, max=1_000_000)
        self.outline_thickness_slider = widgets.IntSlider(description="Outline px", value=1, min=1, max=10, step=1)
        self.persist_debug_checkbox = widgets.Checkbox(value=False, description="Persist debug outputs")
        self.amalgamate_debug_checkbox = widgets.Checkbox(
            value=False,
            description="Amalgamate debug outputs",
        )
        self.keep_individual_debug_checkbox = widgets.Checkbox(
            value=True,
            description="Keep individual debug files",
        )
        self.sample_offset_input = widgets.BoundedIntText(description="Sample offset:", value=0, min=0, max=1_000_000)
        self.sample_limit_input = widgets.BoundedIntText(description="Sample limit:", value=0, min=0, max=1_000_000)

        self.overwrite_checkbox = widgets.Checkbox(value=False, description="Overwrite existing files")
        self.dry_run_checkbox = widgets.Checkbox(value=False, description="Dry run")
        self.continue_on_error_checkbox = widgets.Checkbox(value=True, description="Continue on error")

        self.refresh_runs_button = widgets.Button(description="Refresh Runs")
        self.preview_button = widgets.Button(description="Preview Sample")
        self.execute_button = widgets.Button(description="Run Threshold Stage")

        self.preview_output = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px"))
        self.log_output = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px", height="220px"))

        self.refresh_runs_button.on_click(self._on_refresh_runs)
        self.preview_button.on_click(self._on_preview)
        self.execute_button.on_click(self._on_execute)
        self.run_dropdown.observe(self._on_run_change, names="value")

        self._refresh_runs()

    def display(self) -> None:
        display(
            widgets.VBox(
                [
                    widgets.HTML("<b>Threshold Stage (V3)</b>"),
                    widgets.HBox([self.refresh_runs_button, self.preview_button, self.execute_button]),
                    self.run_dropdown,
                    self.sample_dropdown,
                    self.mode_dropdown,
                    self.threshold_low_slider,
                    self.threshold_high_slider,
                    self.invert_selection_checkbox,
                    self.min_component_area_input,
                    self.outline_thickness_slider,
                    self.persist_debug_checkbox,
                    self.amalgamate_debug_checkbox,
                    self.keep_individual_debug_checkbox,
                    self.sample_offset_input,
                    self.sample_limit_input,
                    self.overwrite_checkbox,
                    self.dry_run_checkbox,
                    self.continue_on_error_checkbox,
                    self.preview_output,
                    self.log_output,
                ]
            )
        )

    def _log(self, message: str) -> None:
        with self.log_output:
            print(message)

    def _refresh_runs(self) -> None:
        runs = list_input_runs(self.project_root)
        self.run_dropdown.options = runs
        if runs and self.run_dropdown.value not in runs:
            self.run_dropdown.value = runs[0]
        self._refresh_samples()

    def _refresh_samples(self) -> None:
        run_name = self.run_dropdown.value
        if not run_name:
            self.sample_dropdown.options = []
            return

        run_paths = input_run_paths(self.project_root, run_name)
        samples_df = _load_samples(run_paths.manifests_dir / "samples.csv")
        options = _sample_options(samples_df)

        self.sample_dropdown.options = options
        if options:
            option_values = [value for _, value in options]
            if self.sample_dropdown.value not in option_values:
                self.sample_dropdown.value = option_values[0]

    def _on_run_change(self, _change: dict) -> None:
        self._refresh_samples()

    def _on_refresh_runs(self, _button: widgets.Button) -> None:
        self._refresh_runs()

    def _on_preview(self, _button: widgets.Button) -> None:
        run_name = self.run_dropdown.value
        row_idx = self.sample_dropdown.value

        with self.preview_output:
            self.preview_output.clear_output(wait=True)

            if run_name is None or row_idx is None:
                print("Select a run and sample first.")
                return

            run_paths = input_run_paths(self.project_root, run_name)
            samples_df = _load_samples(run_paths.manifests_dir / "samples.csv")
            if samples_df is None or row_idx not in samples_df.index:
                print("Sample row not found.")
                return

            source_filename = samples_df.at[row_idx, "image_filename"]
            source_path = resolve_manifest_path(run_paths.root, "images", source_filename)
            source_gray = _safe_gray_image(source_path)

            if source_gray is None:
                print(f"Could not load source image: {source_path}")
                return

            low = int(self.threshold_low_slider.value)
            high = int(self.threshold_high_slider.value)
            if low > high:
                low, high = high, low

            gimp_binary = _gimp_threshold_binary(
                source_gray,
                low_value=low,
                high_value=high,
                invert_selection=bool(self.invert_selection_checkbox.value),
            )
            selected_mask = _filter_components(
                gimp_binary,
                min_component_area_px=max(1, int(self.min_component_area_input.value)),
            )
            if not np.any(selected_mask > 0):
                _draw_three_panel_preview(
                    source_gray,
                    gimp_binary,
                    None,
                    "Source",
                    "Threshold (GIMP binary)",
                    "Threshold Output (no component)",
                )
                return

            output_preview = _render_threshold_artifact(
                selected_mask,
                representation_mode=self.mode_dropdown.value,
                outline_thickness=max(1, int(self.outline_thickness_slider.value)),
            )

            _draw_three_panel_preview(
                source_gray,
                gimp_binary,
                output_preview,
                "Source",
                "Threshold (GIMP binary)",
                "Threshold Output",
            )

    def _on_execute(self, _button: widgets.Button) -> None:
        self.log_output.clear_output(wait=True)

        run_name = self.run_dropdown.value
        if not run_name:
            self._log("Select a run first.")
            return

        config = ThresholdStageConfigV3(
            representation_mode=self.mode_dropdown.value,
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            threshold_low_value=self.threshold_low_slider.value,
            threshold_high_value=self.threshold_high_slider.value,
            invert_selection=self.invert_selection_checkbox.value,
            min_component_area_px=self.min_component_area_input.value,
            outline_thickness=self.outline_thickness_slider.value,
            persist_debug=self.persist_debug_checkbox.value,
            keep_individual_debug_outputs=self.keep_individual_debug_checkbox.value,
            amalgamate_debug_outputs=self.amalgamate_debug_checkbox.value,
            sample_offset=self.sample_offset_input.value,
            sample_limit=self.sample_limit_input.value,
        )

        try:
            summary = run_v3_stage_for_run(
                self.project_root,
                run_name,
                "threshold",
                threshold_config=config,
                log_sink=self._log,
            )
            self._log(
                f"Finished threshold stage: success={summary.successful_rows}, failed={summary.failed_rows}, "
                f"skipped={summary.skipped_rows}"
            )
        except Exception as exc:
            self._log(f"Threshold stage failed: {exc}")


class NpyPackPanelV3:
    """Notebook 02 UI for v3 NPY conversion and NPZ packing."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())
        self._is_running = False

        self.run_dropdown = widgets.Dropdown(description="Run:", options=[])

        self.run_npy_checkbox = widgets.Checkbox(value=True, description="Run NPY+Pack stage")

        self.mode_dropdown = widgets.Dropdown(
            description="Mode:",
            options=[("outline", "outline"), ("filled", "filled")],
            value="filled",
        )
        self.training_source_dropdown = widgets.Dropdown(
            description="Train source:",
            options=_TRAINING_IMAGE_SOURCE_OPTIONS,
            value="threshold_image_filename",
        )

        self.overwrite_checkbox = widgets.Checkbox(value=False, description="Overwrite existing files")
        self.dry_run_checkbox = widgets.Checkbox(value=False, description="Dry run")
        self.continue_on_error_checkbox = widgets.Checkbox(value=True, description="Continue on error")
        self.npy_dtype_dropdown = widgets.Dropdown(
            description="NPY dtype:",
            options=[("float32", "float32"), ("float16", "float16"), ("uint8", "uint8")],
            value="float32",
        )
        self.pack_dtype_dropdown = widgets.Dropdown(
            description="Pack dtype:",
            options=[("preserve", "preserve"), ("float32", "float32"), ("float16", "float16"), ("uint8", "uint8")],
            value="preserve",
        )
        self.pack_compress_checkbox = widgets.Checkbox(value=True, description="Compress NPZ shards")
        self.pack_shard_size_input = widgets.BoundedIntText(
            description="Shard rows:",
            value=128,
            min=0,
            max=1_000_000,
            step=1,
        )
        self.pack_delete_npy_checkbox = widgets.Checkbox(
            value=True,
            description="Delete source NPY after pack",
        )

        self.refresh_runs_button = widgets.Button(description="Refresh Runs")
        self.execute_button = widgets.Button(description="Run Selected")
        self.inspect_button = widgets.Button(description="Inspect NPZ")

        self.log_output = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px", height="220px"))
        self.result_output = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px"))

        self.refresh_runs_button.on_click(self._on_refresh_runs)
        self.execute_button.on_click(self._on_execute)
        self.inspect_button.on_click(self._on_inspect)

        self._refresh_runs()

    def display(self) -> None:
        display(
            widgets.VBox(
                [
                    widgets.HTML("<b>NPY + Pack Stage (V3)</b>"),
                    widgets.HBox([self.refresh_runs_button, self.execute_button, self.inspect_button]),
                    self.run_dropdown,
                    self.run_npy_checkbox,
                    self.mode_dropdown,
                    self.training_source_dropdown,
                    self.overwrite_checkbox,
                    self.dry_run_checkbox,
                    self.continue_on_error_checkbox,
                    self.npy_dtype_dropdown,
                    self.pack_dtype_dropdown,
                    self.pack_compress_checkbox,
                    self.pack_shard_size_input,
                    self.pack_delete_npy_checkbox,
                    self.log_output,
                    self.result_output,
                ]
            )
        )

    def _log(self, message: str) -> None:
        with self.log_output:
            print(message)

    def _refresh_runs(self) -> None:
        runs = list_input_runs(self.project_root)
        self.run_dropdown.options = runs
        if runs and self.run_dropdown.value not in runs:
            self.run_dropdown.value = runs[0]

    def _on_refresh_runs(self, _button: widgets.Button) -> None:
        self._refresh_runs()

    def _build_npy_pack_config(self) -> NpyPackStageConfigV3:
        return NpyPackStageConfigV3(
            representation_mode=self.mode_dropdown.value,
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            normalize=True,
            invert=True,
            npy_output_dtype=self.npy_dtype_dropdown.value,
            pack_output_dtype=self.pack_dtype_dropdown.value,
            compress=self.pack_compress_checkbox.value,
            shard_size=self.pack_shard_size_input.value,
            delete_source_npy_after_pack=self.pack_delete_npy_checkbox.value,
            include_optional_filename_arrays=True,
            training_image_source_column=self.training_source_dropdown.value,
        )

    def _on_execute(self, _button: widgets.Button) -> None:
        if self._is_running:
            self._log("Execution is already running. Wait for the current run to finish.")
            return

        self._is_running = True
        self.execute_button.disabled = True
        self.refresh_runs_button.disabled = True
        self.log_output.clear_output(wait=True)
        self.result_output.clear_output(wait=True)

        run_name = self.run_dropdown.value
        if not run_name:
            self._log("Select a run first.")
            self._is_running = False
            self.execute_button.disabled = False
            self.refresh_runs_button.disabled = False
            return

        if not self.run_npy_checkbox.value:
            self._log("Enable 'Run NPY+Pack stage' to execute.")
            self._is_running = False
            self.execute_button.disabled = False
            self.refresh_runs_button.disabled = False
            return

        try:
            summary = run_v3_stage_for_run(
                self.project_root,
                run_name,
                "npy",
                npy_pack_config=self._build_npy_pack_config(),
                log_sink=self._log,
            )
            self._log(
                f"Finished npy+pack stage: success={summary.successful_rows}, failed={summary.failed_rows}, "
                f"skipped={summary.skipped_rows}"
            )

            self._show_npz_summary(run_name)
        except Exception as exc:
            self._log(f"Execution failed: {exc}")
        finally:
            self._is_running = False
            self.execute_button.disabled = False
            self.refresh_runs_button.disabled = False

    def _on_inspect(self, _button: widgets.Button) -> None:
        run_name = self.run_dropdown.value
        if not run_name:
            return
        self.result_output.clear_output(wait=True)
        self._show_npz_summary(run_name)

    def _show_npz_summary(self, run_name: str) -> None:
        training_paths = training_v3_run_paths(self.project_root, run_name)
        single_npz = training_paths.root / f"{run_name}.npz"
        shard_npz_paths = sorted(training_paths.root.glob(f"{run_name}_shard_*.npz"))
        npz_paths: list[Path] = []
        if single_npz.is_file():
            npz_paths.append(single_npz)
        npz_paths.extend(shard_npz_paths)

        with self.result_output:
            self.result_output.clear_output(wait=True)

            if not npz_paths:
                print(f"NPZ not found for run '{run_name}'.")
                return

            if single_npz.is_file() and shard_npz_paths:
                print("Warning: both single-file and sharded NPZ outputs were found.")

            total_samples = 0
            x_dtypes: set[str] = set()
            x_shapes: set[str] = set()
            first_sample_id: np.ndarray | None = None
            first_y: np.ndarray | None = None

            for npz_path in npz_paths:
                with np.load(npz_path, allow_pickle=False) as data:
                    x = data["X"]
                    y = data["y"]
                    sample_id = data["sample_id"]

                    total_samples += int(len(y))
                    x_dtypes.add(str(x.dtype))
                    x_shapes.add(str(x.shape[1:]) if x.ndim >= 3 else str(x.shape))

                    if first_sample_id is None:
                        first_sample_id = sample_id
                        first_y = y

            print(f"NPZ files found: {len(npz_paths)}")
            print(f"Included sample count: {total_samples}")
            print(f"X dtypes: {', '.join(sorted(x_dtypes))}")
            print(f"X sample shapes: {', '.join(sorted(x_shapes))}")

            preview_limit = 8
            if len(npz_paths) <= preview_limit:
                print("Files:")
                for npz_path in npz_paths:
                    print(f"  {npz_path.name}")
            else:
                print("Files (first 8):")
                for npz_path in npz_paths[:preview_limit]:
                    print(f"  {npz_path.name}")

            if first_sample_id is None or first_y is None:
                return

            preview_count = min(5, len(first_y))
            print("Sample preview from first file (sample_id, distance_m):")
            for idx in range(preview_count):
                print(f"  {first_sample_id[idx]} | {float(first_y[idx]):.6f}")


class ShuffleCorpusPanelV3:
    """Notebook 04 UI for creating a shuffled v3 training corpus."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())
        self._is_running = False

        self.source_run_dropdown = widgets.Dropdown(description="Source Run:", options=[])
        self.output_root_text = widgets.Text(description="Output Root:", value="training-data-v3-shuffled")
        self.output_run_text = widgets.Text(description="Output Run:", value="")
        self.seed_input = widgets.IntText(description="Seed:", value=42)
        self.ledger_filename_text = widgets.Text(description="Ledger:", value="shuffle_ledger.csv")

        self.overwrite_checkbox = widgets.Checkbox(value=False, description="Overwrite output run")
        self.dry_run_checkbox = widgets.Checkbox(value=False, description="Dry run")
        self.compress_checkbox = widgets.Checkbox(value=True, description="Compress output NPZ")
        self.strict_unique_checkbox = widgets.Checkbox(value=True, description="Fail if sample_id duplicates exist")

        self.refresh_runs_button = widgets.Button(description="Refresh Runs")
        self.execute_button = widgets.Button(description="Run Shuffle")
        self.inspect_button = widgets.Button(description="Inspect Output")

        self.log_output = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px", height="240px"))
        self.result_output = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px"))

        self.refresh_runs_button.on_click(self._on_refresh_runs)
        self.execute_button.on_click(self._on_execute)
        self.inspect_button.on_click(self._on_inspect)
        self.source_run_dropdown.observe(self._on_source_run_change, names="value")

        self._refresh_runs()

    def display(self) -> None:
        display(
            widgets.VBox(
                [
                    widgets.HTML("<b>Shuffle Training Corpus (V3)</b>"),
                    widgets.HBox([self.refresh_runs_button, self.execute_button, self.inspect_button]),
                    self.source_run_dropdown,
                    self.output_root_text,
                    self.output_run_text,
                    self.seed_input,
                    self.ledger_filename_text,
                    self.overwrite_checkbox,
                    self.dry_run_checkbox,
                    self.compress_checkbox,
                    self.strict_unique_checkbox,
                    self.log_output,
                    self.result_output,
                ]
            )
        )

    def _log(self, message: str) -> None:
        with self.log_output:
            print(message)

    def _default_output_run_name(self, source_run: str | None) -> str:
        if not source_run:
            return ""
        return f"{source_run}_shuffled"

    def _refresh_runs(self) -> None:
        runs = list_training_v3_runs(self.project_root)
        self.source_run_dropdown.options = runs
        if runs and self.source_run_dropdown.value not in runs:
            self.source_run_dropdown.value = runs[0]
        elif not runs:
            self.source_run_dropdown.value = None

        self.output_run_text.value = self._default_output_run_name(self.source_run_dropdown.value)

    def _on_refresh_runs(self, _button: widgets.Button) -> None:
        self._refresh_runs()

    def _on_source_run_change(self, _change: dict) -> None:
        self.output_run_text.value = self._default_output_run_name(self.source_run_dropdown.value)

    def _build_config(self) -> ShuffleStageConfigV3:
        output_root_name = self.output_root_text.value.strip() or "training-data-v3-shuffled"
        ledger_filename = self.ledger_filename_text.value.strip() or "shuffle_ledger.csv"

        return ShuffleStageConfigV3(
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=True,
            output_root_name=output_root_name,
            random_seed=int(self.seed_input.value),
            compress=self.compress_checkbox.value,
            strict_unique_sample_ids=self.strict_unique_checkbox.value,
            ledger_filename=ledger_filename,
        )

    def _on_execute(self, _button: widgets.Button) -> None:
        if self._is_running:
            self._log("Execution is already running. Wait for the current run to finish.")
            return

        self._is_running = True
        self.execute_button.disabled = True
        self.refresh_runs_button.disabled = True
        self.log_output.clear_output(wait=True)
        self.result_output.clear_output(wait=True)

        source_run = self.source_run_dropdown.value
        output_run = self.output_run_text.value.strip()

        if not source_run:
            self._log("Select a source run first.")
            self._is_running = False
            self.execute_button.disabled = False
            self.refresh_runs_button.disabled = False
            return

        if not output_run:
            self._log("Output run cannot be blank.")
            self._is_running = False
            self.execute_button.disabled = False
            self.refresh_runs_button.disabled = False
            return

        config = self._build_config()

        try:
            summary = run_shuffle_stage_v3(
                self.project_root,
                source_run_name=source_run,
                output_run_name=output_run,
                config=config,
                log_sink=self._log,
            )
            self._log(
                f"Finished shuffle: rows={summary.total_rows}, success={summary.successful_rows}, "
                f"failed={summary.failed_rows}, skipped={summary.skipped_rows}"
            )
            self._show_output_summary(config.output_root_name, output_run, config.ledger_filename)
        except Exception as exc:
            self._log(f"Shuffle failed: {exc}")
        finally:
            self._is_running = False
            self.execute_button.disabled = False
            self.refresh_runs_button.disabled = False

    def _on_inspect(self, _button: widgets.Button) -> None:
        output_root_name = self.output_root_text.value.strip() or "training-data-v3-shuffled"
        output_run = self.output_run_text.value.strip()
        ledger_filename = self.ledger_filename_text.value.strip() or "shuffle_ledger.csv"

        if not output_run:
            return

        self.result_output.clear_output(wait=True)
        self._show_output_summary(output_root_name, output_run, ledger_filename)

    def _show_output_summary(self, output_root_name: str, output_run: str, ledger_filename: str) -> None:
        output_run_root = self.project_root / output_root_name / output_run
        npz_paths = sorted(output_run_root.glob("*.npz"))
        manifests_dir = output_run_root / "manifests"
        samples_path = manifests_dir / "samples.csv"
        ledger_path = manifests_dir / ledger_filename

        with self.result_output:
            self.result_output.clear_output(wait=True)

            if not output_run_root.exists():
                print(f"Output run not found: {output_run_root}")
                return

            if not npz_paths:
                print("No NPZ files found in output run.")
                return

            total_samples = 0
            x_dtypes: set[str] = set()
            x_shapes: set[str] = set()

            for npz_path in npz_paths:
                with np.load(npz_path, allow_pickle=False) as data:
                    row_count = int(len(data["sample_id"])) if "sample_id" in data else int(len(data["X"]))
                    total_samples += row_count

                    if "X" in data:
                        x = data["X"]
                        x_dtypes.add(str(x.dtype))
                        x_shapes.add(str(x.shape[1:]) if x.ndim >= 3 else str(x.shape))

            print(f"NPZ files found: {len(npz_paths)}")
            print(f"Included sample count: {total_samples}")
            if x_dtypes:
                print(f"X dtypes: {', '.join(sorted(x_dtypes))}")
            if x_shapes:
                print(f"X sample shapes: {', '.join(sorted(x_shapes))}")

            preview_limit = 8
            if len(npz_paths) <= preview_limit:
                print("Files:")
                for npz_path in npz_paths:
                    print(f"  {npz_path.name}")
            else:
                print("Files (first 8):")
                for npz_path in npz_paths[:preview_limit]:
                    print(f"  {npz_path.name}")

            if samples_path.is_file():
                samples_df = pd.read_csv(samples_path)
                print(f"samples.csv rows: {len(samples_df)}")
                if "sample_id" in samples_df.columns:
                    duplicate_count = int(samples_df["sample_id"].astype(str).duplicated(keep=False).sum())
                    print(f"Duplicate sample_id rows: {duplicate_count}")
            else:
                print(f"samples.csv not found: {samples_path}")

            if ledger_path.is_file():
                ledger_df = pd.read_csv(ledger_path)
                print(f"Ledger rows: {len(ledger_df)} ({ledger_path.name})")
            else:
                print(f"Ledger not found: {ledger_path}")
