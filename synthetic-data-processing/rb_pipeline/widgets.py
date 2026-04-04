"""ipywidgets-based notebook UI components for the RB pipeline."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from .bbox_stage import run_bbox_stage
from .config import BBoxStageConfig, EdgeStageConfig, NpyStageConfig, PackStageConfig, ShuffleStageConfig
from .edge_stage import run_edge_stage
from .image_io import edge_image_black_on_white, read_grayscale_uint8
from .paths import (
    bbox_run_paths,
    edge_run_paths,
    find_project_root,
    input_run_paths,
    list_input_runs,
    list_training_runs,
    resolve_manifest_path,
    training_run_paths,
)
from .pipeline import STAGE_ORDER, run_stage_for_run
from .shuffle_stage import run_shuffle_stage


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


def _draw_preview_grid(
    source_img: np.ndarray | None,
    edge_img: np.ndarray | None,
    bbox_img: np.ndarray | None,
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
        (axes[0, 1], edge_img, "Edge Image", (0, 255)),
        (axes[1, 0], bbox_img, "BBox Image", (0, 255)),
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


def _draw_two_panel_preview(left: np.ndarray | None, right: np.ndarray | None, left_title: str, right_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    panels = [(axes[0], left, left_title), (axes[1], right, right_title)]

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


class PreviewPanel:
    """Shared 4-panel preview: source, edge, bbox, training array."""

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
                widgets.HTML("<b>Preview Panel</b>"),
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
            edge_paths = edge_run_paths(self.project_root, run_name)
            bbox_paths = bbox_run_paths(self.project_root, run_name)
            training_paths = training_run_paths(self.project_root, run_name)

            input_df = _load_samples(input_paths.manifests_dir / "samples.csv")
            edge_df = _load_samples(edge_paths.manifests_dir / "samples.csv")
            bbox_df = _load_samples(bbox_paths.manifests_dir / "samples.csv")
            training_df = _load_samples(training_paths.manifests_dir / "samples.csv")

            if input_df is None or row_idx not in input_df.index:
                print("Sample not available in input manifest.")
                return

            source_path = None
            edge_path = None
            bbox_path = None
            npy_path = None

            try:
                source_filename = input_df.at[row_idx, "image_filename"]
                source_path = resolve_manifest_path(input_paths.root, "images", source_filename)
            except Exception:
                source_path = None

            if edge_df is not None and row_idx in edge_df.index and "edge_image_filename" in edge_df.columns:
                try:
                    edge_filename = edge_df.at[row_idx, "edge_image_filename"]
                    edge_path = resolve_manifest_path(edge_paths.root, "images", edge_filename)
                except Exception:
                    edge_path = None

            if bbox_df is not None and row_idx in bbox_df.index and "bbox_image_filename" in bbox_df.columns:
                try:
                    bbox_filename = bbox_df.at[row_idx, "bbox_image_filename"]
                    bbox_path = resolve_manifest_path(bbox_paths.root, "images", bbox_filename)
                except Exception:
                    bbox_path = None

            if training_df is not None and row_idx in training_df.index and "npy_filename" in training_df.columns:
                try:
                    npy_filename = training_df.at[row_idx, "npy_filename"]
                    npy_path = resolve_manifest_path(training_paths.root, "arrays", npy_filename)
                except Exception:
                    npy_path = None

            source_img = _safe_gray_image(source_path)
            edge_img = _safe_gray_image(edge_path)
            bbox_img = _safe_gray_image(bbox_path)
            training_array = _safe_npy(npy_path)

            _draw_preview_grid(source_img, edge_img, bbox_img, training_array)

class PipelineLauncher:
    """Notebook 00 launcher UI for run/stage orchestration."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())
        self._is_running = False

        self.run_select = widgets.SelectMultiple(description="Runs:", options=[])
        self.allow_multi_run_checkbox = widgets.Checkbox(
            value=False, description="Allow multi-run execution"
        )
        self.stage_dropdown = widgets.Dropdown(
            description="Stage:",
            options=[("all", "all"), ("edge", "edge"), ("bbox", "bbox"), ("npy", "npy")],
            value="all",
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

        self.blur_kernel_slider = widgets.IntSlider(description="Blur k", value=5, min=1, max=31, step=2)
        self.canny_low_slider = widgets.IntSlider(description="Canny low", value=50, min=0, max=255, step=1)
        self.canny_high_slider = widgets.IntSlider(description="Canny high", value=150, min=0, max=255, step=1)

        self.fg_threshold_slider = widgets.IntSlider(description="FG thresh", value=250, min=0, max=255, step=1)
        self.bbox_thickness_slider = widgets.IntSlider(description="Thickness", value=3, min=1, max=20, step=1)
        self.bbox_padding_slider = widgets.IntSlider(description="Padding", value=0, min=0, max=128, step=1)
        self.bbox_post_blur_checkbox = widgets.Checkbox(value=False, description="Post-draw blur")

        self.refresh_runs_button = widgets.Button(description="Refresh Runs")
        self.execute_button = widgets.Button(description="Run Selected")

        self.progress = widgets.IntProgress(value=0, min=0, max=1, description="Progress")
        self.log_output = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px", height="280px"))

        self.preview_panel = PreviewPanel(self.project_root)

        self.refresh_runs_button.on_click(self._on_refresh_runs)
        self.execute_button.on_click(self._on_execute)
        self.run_select.observe(self._on_run_selection_change, names="value")
        self.allow_multi_run_checkbox.observe(self._on_allow_multi_toggle, names="value")
        self.preview_panel.run_dropdown.observe(self._on_preview_run_change, names="value")

        self._refresh_runs()

    def display(self) -> None:
        control_panel = widgets.VBox(
            [
                widgets.HTML("<b>Pipeline Launcher</b>"),
                widgets.HBox([self.refresh_runs_button, self.execute_button]),
                self.run_select,
                self.allow_multi_run_checkbox,
                self.stage_dropdown,
                self.overwrite_checkbox,
                self.dry_run_checkbox,
                self.continue_on_error_checkbox,
                widgets.HTML("<b>Edge Parameters</b>"),
                self.blur_kernel_slider,
                self.canny_low_slider,
                self.canny_high_slider,
                widgets.HTML("<b>BBox Parameters</b>"),
                self.fg_threshold_slider,
                self.bbox_thickness_slider,
                self.bbox_padding_slider,
                self.bbox_post_blur_checkbox,
                widgets.HTML("<b>NPY + Pack Parameters</b>"),
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

    def _build_edge_config(self) -> EdgeStageConfig:
        return EdgeStageConfig(
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            blur_kernel_size=self.blur_kernel_slider.value,
            canny_low_threshold=self.canny_low_slider.value,
            canny_high_threshold=self.canny_high_slider.value,
        )

    def _build_bbox_config(self) -> BBoxStageConfig:
        return BBoxStageConfig(
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            foreground_threshold=self.fg_threshold_slider.value,
            line_thickness=self.bbox_thickness_slider.value,
            padding_px=self.bbox_padding_slider.value,
            post_draw_blur=self.bbox_post_blur_checkbox.value,
        )

    def _build_npy_config(self) -> NpyStageConfig:
        return NpyStageConfig(
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            normalize=True,
            invert=True,
            output_dtype=self.npy_dtype_dropdown.value,
        )

    def _build_pack_config(self) -> PackStageConfig:
        return PackStageConfig(
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            output_dtype=self.pack_dtype_dropdown.value,
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
        stages = STAGE_ORDER if selected_stage == "all" else [selected_stage]

        self._log(f"Selected runs: {', '.join(selected_runs)}")
        self._log(f"Selected stage: {selected_stage}")

        edge_config = self._build_edge_config()
        bbox_config = self._build_bbox_config()
        npy_config = self._build_npy_config()
        pack_config = self._build_pack_config()

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
                        summary = run_stage_for_run(
                            self.project_root,
                            run_name,
                            stage_name,
                            edge_config=edge_config,
                            bbox_config=bbox_config,
                            npy_config=npy_config,
                            pack_config=pack_config,
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

class EdgeStagePanel:
    """Notebook 01 UI for edge parameter tuning and run execution."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())

        self.run_dropdown = widgets.Dropdown(description="Run:", options=[])
        self.sample_dropdown = widgets.Dropdown(description="Sample:", options=[])

        self.blur_kernel_slider = widgets.IntSlider(description="Blur k", value=5, min=1, max=31, step=2)
        self.canny_low_slider = widgets.IntSlider(description="Canny low", value=50, min=0, max=255, step=1)
        self.canny_high_slider = widgets.IntSlider(description="Canny high", value=150, min=0, max=255, step=1)

        self.overwrite_checkbox = widgets.Checkbox(value=False, description="Overwrite existing files")
        self.dry_run_checkbox = widgets.Checkbox(value=False, description="Dry run")
        self.continue_on_error_checkbox = widgets.Checkbox(value=True, description="Continue on error")

        self.refresh_runs_button = widgets.Button(description="Refresh Runs")
        self.preview_button = widgets.Button(description="Preview Sample")
        self.execute_button = widgets.Button(description="Run Edge Stage")

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
                    widgets.HTML("<b>Edge Stage</b>"),
                    widgets.HBox([self.refresh_runs_button, self.preview_button, self.execute_button]),
                    self.run_dropdown,
                    self.sample_dropdown,
                    self.blur_kernel_slider,
                    self.canny_low_slider,
                    self.canny_high_slider,
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

            edge_preview = edge_image_black_on_white(
                source_gray,
                blur_kernel_size=max(1, int(self.blur_kernel_slider.value) | 1),
                canny_low_threshold=self.canny_low_slider.value,
                canny_high_threshold=self.canny_high_slider.value,
            )

            _draw_two_panel_preview(source_gray, edge_preview, "Source", "Edge Preview")

    def _on_execute(self, _button: widgets.Button) -> None:
        self.log_output.clear_output(wait=True)

        run_name = self.run_dropdown.value
        if not run_name:
            self._log("Select a run first.")
            return

        config = EdgeStageConfig(
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            blur_kernel_size=self.blur_kernel_slider.value,
            canny_low_threshold=self.canny_low_slider.value,
            canny_high_threshold=self.canny_high_slider.value,
        )

        try:
            summary = run_edge_stage(self.project_root, run_name, config=config, log_sink=self._log)
            self._log(
                f"Finished edge stage: success={summary.successful_rows}, failed={summary.failed_rows}, "
                f"skipped={summary.skipped_rows}"
            )
        except Exception as exc:
            self._log(f"Edge stage failed: {exc}")

class BBoxStagePanel:
    """Notebook 02 UI for bbox tuning and run execution."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())

        self.run_dropdown = widgets.Dropdown(description="Run:", options=[])
        self.sample_dropdown = widgets.Dropdown(description="Sample:", options=[])

        self.fg_threshold_slider = widgets.IntSlider(description="FG thresh", value=250, min=0, max=255, step=1)
        self.bbox_thickness_slider = widgets.IntSlider(description="Thickness", value=3, min=1, max=20, step=1)
        self.bbox_padding_slider = widgets.IntSlider(description="Padding", value=0, min=0, max=128, step=1)
        self.post_blur_checkbox = widgets.Checkbox(value=False, description="Post-draw blur")

        self.overwrite_checkbox = widgets.Checkbox(value=False, description="Overwrite existing files")
        self.dry_run_checkbox = widgets.Checkbox(value=False, description="Dry run")
        self.continue_on_error_checkbox = widgets.Checkbox(value=True, description="Continue on error")

        self.refresh_runs_button = widgets.Button(description="Refresh Runs")
        self.preview_button = widgets.Button(description="Preview Sample")
        self.execute_button = widgets.Button(description="Run BBox Stage")

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
                    widgets.HTML("<b>BBox Stage</b>"),
                    widgets.HBox([self.refresh_runs_button, self.preview_button, self.execute_button]),
                    self.run_dropdown,
                    self.sample_dropdown,
                    self.fg_threshold_slider,
                    self.bbox_thickness_slider,
                    self.bbox_padding_slider,
                    self.post_blur_checkbox,
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

        edge_paths = edge_run_paths(self.project_root, run_name)
        samples_df = _load_samples(edge_paths.manifests_dir / "samples.csv")
        if samples_df is None:
            input_paths = input_run_paths(self.project_root, run_name)
            samples_df = _load_samples(input_paths.manifests_dir / "samples.csv")

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

            edge_paths = edge_run_paths(self.project_root, run_name)
            samples_df = _load_samples(edge_paths.manifests_dir / "samples.csv")
            if samples_df is None or row_idx not in samples_df.index:
                print("Edge-stage manifest not available. Run edge stage first.")
                return

            if "edge_image_filename" not in samples_df.columns:
                print("edge_image_filename column missing in edge manifest.")
                return

            edge_filename = samples_df.at[row_idx, "edge_image_filename"]
            edge_path = resolve_manifest_path(edge_paths.root, "images", edge_filename)
            edge_img = _safe_gray_image(edge_path)

            if edge_img is None:
                print(f"Could not load edge image: {edge_path}")
                return

            foreground_mask = edge_img < int(self.fg_threshold_slider.value)
            if not np.any(foreground_mask):
                _draw_two_panel_preview(edge_img, None, "Edge", "BBox Preview (no foreground)")
                return

            ys, xs = np.where(foreground_mask)
            padding = max(0, int(self.bbox_padding_slider.value))

            x1 = max(0, int(xs.min()) - padding)
            y1 = max(0, int(ys.min()) - padding)
            x2 = min(edge_img.shape[1] - 1, int(xs.max()) + padding)
            y2 = min(edge_img.shape[0] - 1, int(ys.max()) + padding)

            thickness = max(1, int(self.bbox_thickness_slider.value))
            bbox_canvas = np.full(edge_img.shape, 255, dtype=np.uint8)

            import cv2

            cv2.rectangle(
                bbox_canvas,
                (x1, y1),
                (x2, y2),
                color=0,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            if self.post_blur_checkbox.value:
                bbox_canvas = cv2.GaussianBlur(bbox_canvas, (3, 3), 0)

            _draw_two_panel_preview(edge_img, bbox_canvas, "Edge", "BBox Preview")

    def _on_execute(self, _button: widgets.Button) -> None:
        self.log_output.clear_output(wait=True)

        run_name = self.run_dropdown.value
        if not run_name:
            self._log("Select a run first.")
            return

        config = BBoxStageConfig(
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            foreground_threshold=self.fg_threshold_slider.value,
            line_thickness=self.bbox_thickness_slider.value,
            padding_px=self.bbox_padding_slider.value,
            post_draw_blur=self.post_blur_checkbox.value,
        )

        try:
            summary = run_bbox_stage(self.project_root, run_name, config=config, log_sink=self._log)
            self._log(
                f"Finished bbox stage: success={summary.successful_rows}, failed={summary.failed_rows}, "
                f"skipped={summary.skipped_rows}"
            )
        except Exception as exc:
            self._log(f"BBox stage failed: {exc}")

class NpyPackPanel:
    """Notebook 03 UI for NPY conversion and NPZ packing."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())
        self._is_running = False

        self.run_dropdown = widgets.Dropdown(description="Run:", options=[])

        self.run_npy_checkbox = widgets.Checkbox(value=True, description="Run NPY+Pack stage")

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
                    widgets.HTML("<b>NPY + Pack Stage</b>"),
                    widgets.HBox([self.refresh_runs_button, self.execute_button, self.inspect_button]),
                    self.run_dropdown,
                    self.run_npy_checkbox,
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

    def _build_npy_config(self) -> NpyStageConfig:
        return NpyStageConfig(
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            normalize=True,
            invert=True,
            output_dtype=self.npy_dtype_dropdown.value,
        )

    def _build_pack_config(self) -> PackStageConfig:
        return PackStageConfig(
            overwrite=self.overwrite_checkbox.value,
            dry_run=self.dry_run_checkbox.value,
            continue_on_error=self.continue_on_error_checkbox.value,
            output_dtype=self.pack_dtype_dropdown.value,
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
            summary = run_stage_for_run(
                self.project_root,
                run_name,
                "npy",
                npy_config=self._build_npy_config(),
                pack_config=self._build_pack_config(),
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
        training_paths = training_run_paths(self.project_root, run_name)
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


class ShuffleCorpusPanel:
    """Notebook 04 UI for creating a shuffled training corpus."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = find_project_root(project_root or Path.cwd())
        self._is_running = False

        self.source_run_dropdown = widgets.Dropdown(description="Source Run:", options=[])
        self.output_root_text = widgets.Text(description="Output Root:", value="training-data-shuffled")
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
                    widgets.HTML("<b>Shuffle Training Corpus</b>"),
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
        runs = list_training_runs(self.project_root)
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

    def _build_config(self) -> ShuffleStageConfig:
        output_root_name = self.output_root_text.value.strip() or "training-data-shuffled"
        ledger_filename = self.ledger_filename_text.value.strip() or "shuffle_ledger.csv"

        return ShuffleStageConfig(
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
            summary = run_shuffle_stage(
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
        output_root_name = self.output_root_text.value.strip() or "training-data-shuffled"
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
