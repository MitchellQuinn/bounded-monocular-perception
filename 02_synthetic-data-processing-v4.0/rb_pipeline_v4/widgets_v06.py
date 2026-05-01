"""Tri-stream launcher wrappers for v0.6 notebook flow."""

from __future__ import annotations

from pathlib import Path

import ipywidgets as widgets
import numpy as np
from IPython.display import display

from .config import BrightnessNormalizationConfigV4, PackTriStreamStageConfigV4
from .paths import find_project_root
from .pack_tri_stream_stage import _render_orientation_image_scaled_by_foreground_extent
from .pipeline import TRI_STREAM_STAGE_ORDER, run_tri_stream_stage_sequence_for_run
from .widgets import _preview_float_image_to_uint8, _to_png_bytes
from .widgets_v05 import PipelineLauncherV5

WIDGETS_UI_BUILD_V06 = "2026-04-29-tri-stream-v06-distance-orientation-geometry"


class PipelineLauncherV6(PipelineLauncherV5):
    """v0.6 launcher: complete tri-stream preprocessing with semantic preview outputs."""

    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root)
        self.stage_dropdown.options = [("All", "all"), *[(stage, stage) for stage in TRI_STREAM_STAGE_ORDER]]
        self.stage_dropdown.value = "all"
        self.orientation_context_scale_float = widgets.BoundedFloatText(
            description="Orient scale:",
            value=1.25,
            min=1.0,
            max=10.0,
            step=0.05,
        )
        image_layout = widgets.Layout(width="100%", border="1px solid #ddd")
        self.preview_orientation_image = widgets.Image(format="png", value=b"", layout=image_layout)

    def _build_widget(self) -> widgets.Widget:
        controls = widgets.VBox(
            [
                widgets.HTML(f"<b>Pipeline Launcher (v0.6 tri-stream)</b> <code>{WIDGETS_UI_BUILD_V06}</code>"),
                widgets.HBox([self.run_dropdown, self.stage_dropdown]),
                widgets.HBox([self.refresh_button, self.execute_button]),
                widgets.HTML("<hr><b>Detect</b>"),
                self.detector_backend_dropdown,
                widgets.HBox([self.edge_blur_kernel_slider, self.edge_canny_low_slider, self.edge_canny_high_slider]),
                widgets.HBox([self.edge_foreground_threshold_slider, self.edge_padding_int, self.edge_min_foreground_int]),
                widgets.HTML("<hr><b>Silhouette</b>"),
                self.silhouette_mode,
                self.threshold_low_slider,
                self.threshold_high_slider,
                widgets.HBox([self.min_area_input, self.roi_padding_int]),
                widgets.HBox([self.fill_holes_checkbox, self.close_kernel_slider, self.outline_thickness_slider]),
                widgets.HTML("<hr>"),
                self.brightness_norm_pane,
                widgets.HTML("<hr><b>Pack Tri Stream</b>"),
                widgets.HBox([self.canvas_w_int, self.canvas_h_int]),
                widgets.HBox([self.clip_policy_dropdown, self.shard_size_int, self.orientation_context_scale_float]),
                widgets.HTML("<hr><b>Sampling</b>"),
                widgets.HBox([self.sample_offset_input, self.sample_limit_input]),
                widgets.HTML("<b>Logs</b>"),
                self.log_output,
            ],
            layout=widgets.Layout(width="52%"),
        )

        preview = widgets.VBox(
            [
                widgets.HTML("<b>Preview Panel</b>"),
                self.preview_controls_row,
                self.preview_status_html,
                widgets.HTML("<b>0) Source + Edge Region + ROI Box (context only; display-only)</b>"),
                self.preview_source_overlay,
                widgets.HTML("<b>1) ROI Selection Debug (edge detect path used for ROI geometry)</b>"),
                self.preview_roi_selection_debug,
                self.preview_roi_selection_stats_html,
                widgets.HTML("<b>2) ROI Input Canvas (fixed distance canvas source; no scaling)</b>"),
                self.preview_extracted_roi,
                widgets.HTML("<b>3) Silhouette Debug Strip (raw edge | post morph | selected component)</b>"),
                self.preview_silhouette_debug,
                widgets.HTML("<b>4) Distance Detail Before Final Canvas (brightness-normalized when enabled)</b>"),
                self.preview_silhouette_roi,
                widgets.HTML("<b>5) x_distance_image (fixed unscaled ROI canvas)</b>"),
                self.preview_packed_canvas,
                widgets.HTML("<b>6) x_orientation_image (target-centred, scaled by silhouette extent)</b>"),
                self.preview_orientation_image,
                widgets.HTML("<b>7) Full-Frame Silhouette (auxiliary view; not used by pack payload)</b>"),
                self.preview_silhouette_full,
            ],
            layout=widgets.Layout(width="48%", max_height="980px", overflow_y="auto", padding="0 0 0 8px"),
        )

        return widgets.HBox([controls, preview], layout=widgets.Layout(width="100%"))

    def _build_pack_config(self) -> PackTriStreamStageConfigV4:
        brightness_enabled = bool(self.brightness_norm_enabled_checkbox.value)
        brightness_method = (
            str(self.brightness_norm_method_dropdown.value)
            if brightness_enabled
            else "none"
        )

        return PackTriStreamStageConfigV4(
            canvas_width_px=int(self.canvas_w_int.value),
            canvas_height_px=int(self.canvas_h_int.value),
            clip_policy=str(self.clip_policy_dropdown.value),
            include_v1_compat_arrays=False,
            brightness_normalization=BrightnessNormalizationConfigV4(
                enabled=brightness_enabled,
                method=brightness_method,
                target_median_darkness=float(self.brightness_norm_target_float.value),
                min_gain=float(self.brightness_norm_min_gain_float.value),
                max_gain=float(self.brightness_norm_max_gain_float.value),
                epsilon=float(self.brightness_norm_epsilon_float.value),
                empty_mask_policy=str(self.brightness_norm_empty_policy_dropdown.value),
            ),
            orientation_context_scale=float(self.orientation_context_scale_float.value),
            shard_size=int(self.shard_size_int.value),
            sample_offset=self._sample_offset(),
            sample_limit=self._sample_limit(),
        )

    def _reset_extra_preview_outputs(self) -> None:
        if hasattr(self, "preview_orientation_image"):
            self.preview_orientation_image.value = b""

    def _render_extra_pack_preview(
        self,
        *,
        extracted_roi_gray: np.ndarray,
        background_mask: np.ndarray,
        pack_config: PackTriStreamStageConfigV4,
    ) -> None:
        foreground_mask = (np.asarray(background_mask, dtype=np.float32) < 0.5).astype(np.float32)
        orientation_image, _, _, _ = _render_orientation_image_scaled_by_foreground_extent(
            extracted_roi_gray,
            foreground_mask,
            canvas_height=pack_config.normalized_canvas_height_px(),
            canvas_width=pack_config.normalized_canvas_width_px(),
            context_scale=pack_config.normalized_orientation_context_scale(),
        )
        self.preview_orientation_image.value = _to_png_bytes(_preview_float_image_to_uint8(orientation_image))

    def _saved_path_preview_text(self, pack_config: PackTriStreamStageConfigV4) -> str:
        return (
            "<b>Saved Path:</b> fixed ROI canvas -> x_distance_image; "
            "raw ROI + silhouette foreground extent -> target-centred scaled x_orientation_image; "
            "bbox/ROI context -> x_geometry; "
            f"canvas={pack_config.normalized_canvas_width_px()}x{pack_config.normalized_canvas_height_px()}"
        )

    def _on_execute(self, _button: widgets.Button) -> None:
        if self._execution_in_progress:
            return

        run_name = self.run_dropdown.value
        stage_name = self.stage_dropdown.value
        if not run_name:
            with self.log_output:
                print("No run selected.")
            return

        self._execution_in_progress = True
        self.execute_button.disabled = True

        try:
            self.log_output.clear_output(wait=False)
            with self.log_output:
                print(f"Running tri-stream stage '{stage_name}' for run '{run_name}' ...")

            summaries = run_tri_stream_stage_sequence_for_run(
                self.project_root,
                str(run_name),
                str(stage_name),
                detect_config=self._build_detect_config(),
                silhouette_config=self._build_silhouette_config(),
                pack_tri_stream_config=self._build_pack_config(),
                log_sink=self._log_sink,
            )
            with self.log_output:
                print("Done.")
                for summary in summaries:
                    print(
                        f"- {summary.stage_name}: success={summary.successful_rows}, "
                        f"failed={summary.failed_rows}, skipped={summary.skipped_rows}"
                    )
        except Exception as exc:
            with self.log_output:
                print(f"Error: {exc}")
        finally:
            self._execution_in_progress = False
            self.execute_button.disabled = False
            self.render_preview()

    @property
    def widget(self) -> widgets.Widget:
        if self._widget_root is None:
            self._widget_root = self._build_widget()
        return self._widget_root


def display_pipeline_launcher_v06(start: Path | None = None) -> PipelineLauncherV6:
    """Locate project root and display v0.6 tri-stream pipeline launcher."""

    project_root = find_project_root(start)
    launcher = PipelineLauncherV6(project_root)
    display(launcher.widget)
    return launcher
