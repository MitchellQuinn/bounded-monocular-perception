"""Shared contracts and constants for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

INPUT_ROOT_NAME = "input"
OUTPUT_ROOT_NAME = "output"

TRAIN_SPLIT_NAME = "train"
VALIDATE_SPLIT_NAME = "validate"
SPLIT_ORDER = (TRAIN_SPLIT_NAME, VALIDATE_SPLIT_NAME)

RUN_JSON_FILENAME = "run.json"
SAMPLES_FILENAME = "samples.csv"

PREPROCESSING_CONTRACT_KEY = "PreprocessingContract"
PREPROCESSING_CONTRACT_VERSION = "rb-preprocess-roi-fcn-v0_1"
PREPROCESSING_STAGE_ORDER = ("bootstrap_center_target", "pack_roi_fcn")

INPUT_REQUIRED_COLUMNS = [
    "run_id",
    "sample_id",
    "frame_index",
    "image_filename",
    "distance_m",
    "image_width_px",
    "image_height_px",
    "capture_success",
]

BOOTSTRAP_STAGE_COLUMNS = [
    "bootstrap_center_target_stage_status",
    "bootstrap_center_target_stage_error",
    "bootstrap_target_algorithm",
    "bootstrap_confidence",
    "bootstrap_bbox_x1",
    "bootstrap_bbox_y1",
    "bootstrap_bbox_x2",
    "bootstrap_bbox_y2",
    "bootstrap_bbox_w_px",
    "bootstrap_bbox_h_px",
    "bootstrap_center_x_px",
    "bootstrap_center_y_px",
    "bootstrap_debug_image_filename",
]

PACK_STAGE_COLUMNS = [
    "pack_roi_fcn_stage_status",
    "pack_roi_fcn_stage_error",
    "npz_filename",
    "npz_row_index",
    "locator_canvas_width_px",
    "locator_canvas_height_px",
    "locator_resize_scale",
    "locator_resized_width_px",
    "locator_resized_height_px",
    "locator_pad_left_px",
    "locator_pad_right_px",
    "locator_pad_top_px",
    "locator_pad_bottom_px",
    "locator_center_x_px",
    "locator_center_y_px",
]

REQUIRED_ROI_FCN_NPZ_KEYS = {
    "locator_input_image",
    "target_center_xy_original_px",
    "target_center_xy_canvas_px",
    "source_image_wh_px",
    "resized_image_wh_px",
    "padding_ltrb_px",
    "resize_scale",
    "sample_id",
    "image_filename",
    "npz_row_index",
}

TRACEABILITY_ROI_FCN_NPZ_KEYS = {
    "bootstrap_bbox_xyxy_px",
    "bootstrap_confidence",
    "locator_geometry_schema",
}

LOCATOR_GEOMETRY_SCHEMA = (
    "target_center_xy_original_px=(x_px,y_px) in original full-frame image space",
    "target_center_xy_canvas_px=(x_px,y_px) in locator canvas space",
    "source_image_wh_px=(width_px,height_px)",
    "resized_image_wh_px=(width_px,height_px)",
    "padding_ltrb_px=(left_px,top_px,right_px,bottom_px)",
    "resize_scale=scalar applied before padding",
)


@dataclass(frozen=True)
class DatasetReference:
    """A selectable input dataset reference under input/."""

    name: str
    input_root: Path
    output_root: Path


@dataclass
class StageSummaryV01:
    """Standard summary emitted by each ROI-FCN preprocessing stage."""

    dataset_reference: str
    split_name: str
    stage_name: str
    total_rows: int
    successful_rows: int
    failed_rows: int
    skipped_rows: int
    output_path: str
    log_path: str | None
    dry_run: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DatasetRunSummaryV01:
    """Top-level dataset-run summary for train then validate processing."""

    dataset_reference: str
    output_root: str
    stage_summaries: list[StageSummaryV01]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_reference": self.dataset_reference,
            "output_root": self.output_root,
            "stage_summaries": [summary.as_dict() for summary in self.stage_summaries],
        }
