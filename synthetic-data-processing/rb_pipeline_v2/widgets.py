"""ipywidgets-based notebook UI components for the RB v2 pipeline."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from rb_pipeline.image_io import read_grayscale_uint8
from rb_pipeline.paths import list_input_runs, resolve_manifest_path

from .algorithms import register_default_components
from .config import NpyPackStageConfigV2, ShuffleStageConfigV2, SilhouetteStageConfigV2
from .manifest import load_samples_csv, samples_csv_path
from .paths import (
    find_project_root,
    input_run_paths,
    list_training_v2_runs,
    silhouette_run_paths,
    training_v2_run_paths,
)
from .pipeline import STAGE_ORDER_V2, run_v2_stage_for_run
from .registry import (
    get_artifact_writer_by_mode,
    get_fallback_strategy,
    get_representation_generator,
    list_registered_component_ids,
)
from .shuffle_stage import run_shuffle_stage_v2



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



def _draw_preview_grid_v2(
    source_img: np.ndarray | None,
    edge_debug_img: np.ndarray | None,
    silhouette_img: np.ndarray | None,
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
        (axes[0, 1], edge_debug_img, "Edge Debug", (0, 255)),
        (axes[1, 0], silhouette_img, "Silhouette", (0, 255)),
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


class PreviewPanelV2:
    """Shared 4-panel preview: source, edge debug, silhouette, training array."""

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
                widgets.HTML("<b>Preview Panel (V2)</b>"),
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
            silhouette_paths = silhouette_run_paths(self.project_root, run_name)
            training_paths = training_v2_run_paths(self.project_root, run_name)

            input_df = _load_samples(input_paths.manifests_dir / "samples.csv")
            silhouette_df = _load_samples(silhouette_paths.manifests_dir / "samples.csv")
            training_df = _load_samples(training_paths.manifests_dir / "samples.csv")

            if input_df is None or row_idx not in input_df.index:
                print("Sample not available in input manifest.")
                return

            source_path = None
            edge_debug_path = None
            silhouette_path = None
            npy_path = None

            try:
                source_filename = input_df.at[row_idx, "image_filename"]
                source_path = resolve_manifest_path(input_paths.root, "images", source_filename)
            except Exception:
                source_path = None

            if silhouette_df is not None and row_idx in silhouette_df.index:
                if "silhouette_edge_debug_filename" in silhouette_df.columns:
                    try:
                        edge_debug_filename = silhouette_df.at[row_idx, "silhouette_edge_debug_filename"]
                        if str(edge_debug_filename).strip():
                            edge_debug_path = resolve_manifest_path(
                                silhouette_paths.root, "images", edge_debug_filename
                            )
                    except Exception:
                        edge_debug_path = None
                if "silhouette_image_filename" in silhouette_df.columns:
                    try:
                        silhouette_filename = silhouette_df.at[row_idx, "silhouette_image_filename"]
                        if str(silhouette_filename).strip():
                            silhouette_path = resolve_manifest_path(
                                silhouette_paths.root, "images", silhouette_filename
                            )
                    except Exception:
                        silhouette_path = None

            if training_df is not None and row_idx in training_df.index and "npy_filename" in training_df.columns:
                try:
                    npy_filename = training_df.at[row_idx, "npy_filename"]
                    if str(npy_filename).strip():
                        npy_path = resolve_manifest_path(training_paths.root, "arrays", npy_filename)
                except Exception:
                    npy_path = None

            source_img = _safe_gray_image(source_path)
            edge_debug_img = _safe_gray_image(edge_debug_path)
            silhouette_img = _safe_gray_image(silhouette_path)
            training_array = _safe_npy(npy_path)

            _draw_preview_grid_v2(source_img, edge_debug_img, silhouette_img, training_array)


class PipelineLauncherV2:
    """Notebook 00 launcher UI for v2 run/stage orchestration."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())
        self._is_running = False

        register_default_components()
        registered = list_registered_component_ids()

        self.run_select = widgets.SelectMultiple(description="Runs:", options=[])
        self.allow_multi_run_checkbox = widgets.Checkbox(value=False, description="Allow multi-run execution")
        self.stage_dropdown = widgets.Dropdown(
            description="Stage:",
            options=[("all", "all"), ("silhouette", "silhouette"), ("npy", "npy")],
            value="all",
        )

        self.overwrite_checkbox = widgets.Checkbox(value=False, description="Overwrite existing files")
        self.dry_run_checkbox = widgets.Checkbox(value=False, description="Dry run")
        self.continue_on_error_checkbox = widgets.Checkbox(value=True, description="Continue on error")

        self.mode_dropdown = widgets.Dropdown(
            description="Mode:",
            options=[("outline", "outline"), ("filled", "filled")],
            value="outline",
        )
        self.generator_dropdown = widgets.Dropdown(
            description="Generator:",
            options=[(value, value) for value in registered["generators"]],
            value=registered["generators"][0] if registered["generators"] else None,
        )
        self.fallback_dropdown = widgets.Dropdown(
            description="Fallback:",
            options=[(value, value) for value in registered["fallbacks"]],
            value=registered["fallbacks"][0] if registered["fallbacks"] else None,
        )
        self.persist_edge_debug_checkbox = widgets.Checkbox(value=False, description="Persist edge debug")
        self.sample_offset_input = widgets.BoundedIntText(description="Sample offset:", value=0, min=0, max=1_000_000)
        self.sample_limit_input = widgets.BoundedIntText(description="Sample limit:", value=0, min=0, max=1_000_000)

        self.blur_kernel_slider = widgets.IntSlider(description="Blur k", value=5, min=1, max=31, step=2)
        self.canny_low_slider = widgets.IntSlider(description="Canny low", value=50, min=0, max=255, step=1)
        self.canny_high_slider = widgets.IntSlider(description="Canny high", value=150, min=0, max=255, step=1)
        self.close_kernel_slider = widgets.IntSlider(description="Close k", value=3, min=1, max=31, step=1)
        self.dilate_kernel_slider = widgets.IntSlider(description="Dilate k", value=3, min=1, max=31, step=1)
        self.min_component_area_input = widgets.BoundedIntText(description="Min area:", value=20, min=1, max=1_000_000)
        self.outline_thickness_slider = widgets.IntSlider(description="Outline px", value=1, min=1, max=10, step=1)

        self.array_exporter_dropdown = widgets.Dropdown(
            description="Array exporter:",
            options=[(value, value) for value in registered["array_exporters"]],
            value=registered["array_exporters"][0] if registered["array_exporters"] else None,
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

        self.preview_panel = PreviewPanelV2(self.project_root)

        self.refresh_runs_button.on_click(self._on_refresh_runs)
        self.execute_button.on_click(self._on_execute)
        self.run_select.observe(self._on_run_selection_change, names="value")
        self.allow_multi_run_checkbox.observe(self._on_allow_multi_toggle, names="value")
        self.preview_panel.run_dropdown.observe(self._on_preview_run_change, names="value")

        self._refresh_runs()

    def display(self) -> None:
        control_panel = widgets.VBox(
            [
                widgets.HTML("<b>Pipeline Launcher (V2)</b>"),
                widgets.HBox([self.refresh_runs_button, self.execute_button]),
                self.run_select,
                self.allow_multi_run_checkbox,
                self.stage_dropdown,
                self.overwrite_checkbox,
                self.dry_run_checkbox,
                self.continue_on_error_checkbox,
                widgets.HTML("<b>Silhouette Parameters</b>"),
                self.mode_dropdown,
                self.generator_dropdown,
                self.fallback_dropdown,
                self.persist_edge_debug_checkbox,
                self.sample_offset_input,
                self.sample_limit_input,
                self.blur_kernel_slider,
                self.canny_low_slider,
                self.canny_high_slider,
                self.close_kernel_slider,
                self.dilate_kernel_slider,
                self.min_component_area_input,
                self.outline_thickness_slider,
                widgets.HTML("<b>NPY + Pack Parameters</b>"),
                self.array_exporter_dropdown,
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

    def _build_silhouette_config(self) -> SilhouetteStageConfigV2:
        return SilhouetteStageConfigV2(
            representation_mode=self.mode_dropdown.value,
            generator_id=self.generator_dropdown.value,
            fallback_id=self.fallback_dropdown.value,
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            blur_kernel_size=self.blur_kernel_slider.value,
            canny_low_threshold=self.canny_low_slider.value,
            canny_high_threshold=self.canny_high_slider.value,
            close_kernel_size=self.close_kernel_slider.value,
            dilate_kernel_size=self.dilate_kernel_slider.value,
            min_component_area_px=self.min_component_area_input.value,
            outline_thickness=self.outline_thickness_slider.value,
            persist_edge_debug=self.persist_edge_debug_checkbox.value,
            sample_offset=self.sample_offset_input.value,
            sample_limit=self.sample_limit_input.value,
        )

    def _build_npy_pack_config(self) -> NpyPackStageConfigV2:
        return NpyPackStageConfigV2(
            representation_mode=self.mode_dropdown.value,
            array_exporter_id=self.array_exporter_dropdown.value,
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
        stages = STAGE_ORDER_V2 if selected_stage == "all" else [selected_stage]

        self._log(f"Selected runs: {', '.join(selected_runs)}")
        self._log(f"Selected stage: {selected_stage}")

        silhouette_config = self._build_silhouette_config()
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
                        summary = run_v2_stage_for_run(
                            self.project_root,
                            run_name,
                            stage_name,
                            silhouette_config=silhouette_config,
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


class SilhouetteStagePanelV2:
    """Notebook 01 UI for silhouette tuning and run execution."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())

        register_default_components()
        registered = list_registered_component_ids()

        self.run_dropdown = widgets.Dropdown(description="Run:", options=[])
        self.sample_dropdown = widgets.Dropdown(description="Sample:", options=[])

        self.mode_dropdown = widgets.Dropdown(
            description="Mode:",
            options=[("outline", "outline"), ("filled", "filled")],
            value="outline",
        )
        self.generator_dropdown = widgets.Dropdown(
            description="Generator:",
            options=[(value, value) for value in registered["generators"]],
            value=registered["generators"][0] if registered["generators"] else None,
        )
        self.fallback_dropdown = widgets.Dropdown(
            description="Fallback:",
            options=[(value, value) for value in registered["fallbacks"]],
            value=registered["fallbacks"][0] if registered["fallbacks"] else None,
        )

        self.blur_kernel_slider = widgets.IntSlider(description="Blur k", value=5, min=1, max=31, step=2)
        self.canny_low_slider = widgets.IntSlider(description="Canny low", value=50, min=0, max=255, step=1)
        self.canny_high_slider = widgets.IntSlider(description="Canny high", value=150, min=0, max=255, step=1)
        self.close_kernel_slider = widgets.IntSlider(description="Close k", value=3, min=1, max=31, step=1)
        self.dilate_kernel_slider = widgets.IntSlider(description="Dilate k", value=3, min=1, max=31, step=1)
        self.min_component_area_input = widgets.BoundedIntText(description="Min area:", value=20, min=1, max=1_000_000)
        self.outline_thickness_slider = widgets.IntSlider(description="Outline px", value=1, min=1, max=10, step=1)
        self.persist_edge_debug_checkbox = widgets.Checkbox(value=False, description="Persist edge debug")
        self.sample_offset_input = widgets.BoundedIntText(description="Sample offset:", value=0, min=0, max=1_000_000)
        self.sample_limit_input = widgets.BoundedIntText(description="Sample limit:", value=0, min=0, max=1_000_000)

        self.overwrite_checkbox = widgets.Checkbox(value=False, description="Overwrite existing files")
        self.dry_run_checkbox = widgets.Checkbox(value=False, description="Dry run")
        self.continue_on_error_checkbox = widgets.Checkbox(value=True, description="Continue on error")

        self.refresh_runs_button = widgets.Button(description="Refresh Runs")
        self.preview_button = widgets.Button(description="Preview Sample")
        self.execute_button = widgets.Button(description="Run Silhouette Stage")

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
                    widgets.HTML("<b>Silhouette Stage (V2)</b>"),
                    widgets.HBox([self.refresh_runs_button, self.preview_button, self.execute_button]),
                    self.run_dropdown,
                    self.sample_dropdown,
                    self.mode_dropdown,
                    self.generator_dropdown,
                    self.fallback_dropdown,
                    self.blur_kernel_slider,
                    self.canny_low_slider,
                    self.canny_high_slider,
                    self.close_kernel_slider,
                    self.dilate_kernel_slider,
                    self.min_component_area_input,
                    self.outline_thickness_slider,
                    self.persist_edge_debug_checkbox,
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

            generator = get_representation_generator(self.generator_dropdown.value)
            fallback = get_fallback_strategy(self.fallback_dropdown.value)
            writer = get_artifact_writer_by_mode(self.mode_dropdown.value)

            generated = generator.generate(
                source_gray,
                blur_kernel_size=max(1, int(self.blur_kernel_slider.value) | 1),
                canny_low_threshold=self.canny_low_slider.value,
                canny_high_threshold=self.canny_high_slider.value,
                close_kernel_size=max(1, int(self.close_kernel_slider.value)),
                dilate_kernel_size=max(1, int(self.dilate_kernel_slider.value)),
                min_component_area_px=max(1, int(self.min_component_area_input.value)),
            )

            contour = generated.contour
            fallback_reason = ""
            if contour is None or contour.ndim != 3 or contour.shape[0] < 3:
                contour, fallback_reason = fallback.recover(generated.fallback_mask)

            if contour is None:
                edge_preview = np.full(generated.edge_binary.shape, 255, dtype=np.uint8)
                edge_preview[generated.edge_binary > 0] = 0
                _draw_three_panel_preview(
                    source_gray,
                    edge_preview,
                    None,
                    "Source",
                    "Edge Debug",
                    f"Silhouette Preview (fallback failed: {fallback_reason})",
                )
                return

            silhouette_preview = writer.render(
                source_gray.shape,
                contour,
                line_thickness=max(1, int(self.outline_thickness_slider.value)),
            )
            edge_preview = np.full(generated.edge_binary.shape, 255, dtype=np.uint8)
            edge_preview[generated.edge_binary > 0] = 0

            suffix = f" (fallback: {fallback_reason})" if fallback_reason else ""
            _draw_three_panel_preview(
                source_gray,
                edge_preview,
                silhouette_preview,
                "Source",
                "Edge Debug",
                f"Silhouette Preview{suffix}",
            )

    def _on_execute(self, _button: widgets.Button) -> None:
        self.log_output.clear_output(wait=True)

        run_name = self.run_dropdown.value
        if not run_name:
            self._log("Select a run first.")
            return

        config = SilhouetteStageConfigV2(
            representation_mode=self.mode_dropdown.value,
            generator_id=self.generator_dropdown.value,
            fallback_id=self.fallback_dropdown.value,
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            blur_kernel_size=self.blur_kernel_slider.value,
            canny_low_threshold=self.canny_low_slider.value,
            canny_high_threshold=self.canny_high_slider.value,
            close_kernel_size=self.close_kernel_slider.value,
            dilate_kernel_size=self.dilate_kernel_slider.value,
            min_component_area_px=self.min_component_area_input.value,
            outline_thickness=self.outline_thickness_slider.value,
            persist_edge_debug=self.persist_edge_debug_checkbox.value,
            sample_offset=self.sample_offset_input.value,
            sample_limit=self.sample_limit_input.value,
        )

        try:
            summary = run_v2_stage_for_run(
                self.project_root,
                run_name,
                "silhouette",
                silhouette_config=config,
                log_sink=self._log,
            )
            self._log(
                f"Finished silhouette stage: success={summary.successful_rows}, failed={summary.failed_rows}, "
                f"skipped={summary.skipped_rows}"
            )
        except Exception as exc:
            self._log(f"Silhouette stage failed: {exc}")


class NpyPackPanelV2:
    """Notebook 02 UI for v2 NPY conversion and NPZ packing."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())
        self._is_running = False

        register_default_components()
        registered = list_registered_component_ids()

        self.run_dropdown = widgets.Dropdown(description="Run:", options=[])

        self.run_npy_checkbox = widgets.Checkbox(value=True, description="Run NPY+Pack stage")

        self.mode_dropdown = widgets.Dropdown(
            description="Mode:",
            options=[("outline", "outline"), ("filled", "filled")],
            value="outline",
        )
        self.array_exporter_dropdown = widgets.Dropdown(
            description="Array exporter:",
            options=[(value, value) for value in registered["array_exporters"]],
            value=registered["array_exporters"][0] if registered["array_exporters"] else None,
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
                    widgets.HTML("<b>NPY + Pack Stage (V2)</b>"),
                    widgets.HBox([self.refresh_runs_button, self.execute_button, self.inspect_button]),
                    self.run_dropdown,
                    self.run_npy_checkbox,
                    self.mode_dropdown,
                    self.array_exporter_dropdown,
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

    def _build_npy_pack_config(self) -> NpyPackStageConfigV2:
        return NpyPackStageConfigV2(
            representation_mode=self.mode_dropdown.value,
            array_exporter_id=self.array_exporter_dropdown.value,
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
            summary = run_v2_stage_for_run(
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
        training_paths = training_v2_run_paths(self.project_root, run_name)
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


class ShuffleCorpusPanelV2:
    """Notebook 04 UI for creating a shuffled v2 training corpus."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())
        self._is_running = False

        self.source_run_dropdown = widgets.Dropdown(description="Source Run:", options=[])
        self.output_root_text = widgets.Text(description="Output Root:", value="training-data-v2-shuffled")
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
                    widgets.HTML("<b>Shuffle Training Corpus (V2)</b>"),
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
        runs = list_training_v2_runs(self.project_root)
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

    def _build_config(self) -> ShuffleStageConfigV2:
        output_root_name = self.output_root_text.value.strip() or "training-data-v2-shuffled"
        ledger_filename = self.ledger_filename_text.value.strip() or "shuffle_ledger.csv"

        return ShuffleStageConfigV2(
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
            summary = run_shuffle_stage_v2(
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
        output_root_name = self.output_root_text.value.strip() or "training-data-v2-shuffled"
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
