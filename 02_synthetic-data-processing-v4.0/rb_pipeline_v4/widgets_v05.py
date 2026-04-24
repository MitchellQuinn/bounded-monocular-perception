"""Edge-only launcher wrappers for v0.5 notebook flow."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

from .config import DetectStageConfigV4
from .paths import find_project_root
from .widgets import PipelineLauncherV4

WIDGETS_UI_BUILD_V05 = "2026-04-24-edge-only-v05-brightness-normalization-before-pack-v3"


class PipelineLauncherV5(PipelineLauncherV4):
    """v0.5 launcher: edge ROI detection only (no YOLO path)."""

    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root)
        self._enforce_edge_only_mode()

    def _enforce_edge_only_mode(self) -> None:
        self.detector_backend_dropdown.options = [("Edge ROI (v1 style)", "edge")]
        self.detector_backend_dropdown.value = "edge"
        self.detector_backend_dropdown.disabled = True

        # v0.5 intentionally removes YOLO controls to keep the flow edge-only.
        self.model_path_text.layout.display = "none"
        self.default_model_ref_text.layout.display = "none"
        self.detect_conf_slider.layout.display = "none"
        self.detect_iou_slider.layout.display = "none"
        self.class_ids_text.layout.display = "none"
        self.class_names_text.layout.display = "none"
        self.preview_ignore_filter_checkbox.layout.display = "none"

    def _build_detect_config(self) -> DetectStageConfigV4:
        base = super()._build_detect_config()
        return replace(
            base,
            detector_backend="edge",
            model_path="",
            default_model_ref="",
            defender_class_ids=(),
            defender_class_names=("defender",),
            conf_threshold=0.0,
            iou_threshold=0.0,
            max_det=1,
            imgsz=1280,
        )

    @property
    def widget(self) -> widgets.Widget:
        root = super().widget
        if isinstance(root, widgets.Box):
            children = list(root.children)
            if children:
                first = children[0]
                if isinstance(first, widgets.Box) and first.children:
                    controls = list(first.children)
                    if controls and isinstance(controls[0], widgets.HTML):
                        controls[0].value = f"<b>Pipeline Launcher (v0.5 edge-only)</b> <code>{WIDGETS_UI_BUILD_V05}</code>"
                        first.children = tuple(controls)
                        children[0] = first
                        root.children = tuple(children)
        return root


def display_pipeline_launcher_v05(start: Path | None = None) -> PipelineLauncherV5:
    """Locate project root and display v0.5 edge-only pipeline launcher."""

    project_root = find_project_root(start)
    launcher = PipelineLauncherV5(project_root)
    display(launcher.widget)
    return launcher
