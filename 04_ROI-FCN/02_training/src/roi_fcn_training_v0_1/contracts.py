"""Explicit contracts and shared dataclasses for ROI-FCN training v0.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DATASETS_ROOT_NAME = "datasets"
MODELS_ROOT_NAME = "models"
NOTEBOOKS_ROOT_NAME = "notebooks"

TRAIN_SPLIT_NAME = "train"
VALIDATE_SPLIT_NAME = "validate"
SPLIT_ORDER = (TRAIN_SPLIT_NAME, VALIDATE_SPLIT_NAME)

ARRAYS_DIR_NAME = "arrays"
MANIFESTS_DIR_NAME = "manifests"
RUN_JSON_FILENAME = "run.json"
SAMPLES_FILENAME = "samples.csv"

TRAINING_CONTRACT_VERSION = "rb-roi-fcn-training-v0_1"
TOPOLOGY_CONTRACT_VERSION = "rb-roi-fcn-topology-v0_1"
EVALUATION_CONTRACT_VERSION = "rb-roi-fcn-eval-v0_1"
PREPROCESSING_CONTRACT_KEY = "PreprocessingContract"
EXPECTED_PREPROCESSING_CONTRACT_VERSION = "rb-preprocess-roi-fcn-v0_1"
EXPECTED_REPRESENTATION_KIND = "roi_fcn_locator_npz"
EXPECTED_STORAGE_FORMAT = "npz"
EXPECTED_IMAGE_LAYOUT = "N,C,H,W"
EXPECTED_CHANNEL_COUNT = 1
EXPECTED_TARGET_TYPE = "crop_center_point"
EXPECTED_TARGET_SOURCE = "edge_roi_v1_bootstrap"
EXPECTED_NORMALIZATION_RANGE = (0.0, 1.0)
EXPECTED_GEOMETRY_SCHEMA = (
    "target_center_xy_original_px=(x_px,y_px) in original full-frame image space",
    "target_center_xy_canvas_px=(x_px,y_px) in locator canvas space",
    "source_image_wh_px=(width_px,height_px)",
    "resized_image_wh_px=(width_px,height_px)",
    "padding_ltrb_px=(left_px,top_px,right_px,bottom_px)",
    "resize_scale=scalar applied before padding",
)

REQUIRED_MANIFEST_COLUMNS = (
    "sample_id",
    "image_filename",
    "image_width_px",
    "image_height_px",
    "pack_roi_fcn_stage_status",
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
    "bootstrap_center_x_px",
    "bootstrap_center_y_px",
    "bootstrap_bbox_x1",
    "bootstrap_bbox_y1",
    "bootstrap_bbox_x2",
    "bootstrap_bbox_y2",
)

REQUIRED_NPZ_ARRAY_KEYS = {
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

OPTIONAL_TRACEABILITY_NPZ_KEYS = {
    "bootstrap_bbox_xyxy_px",
    "bootstrap_confidence",
    "locator_geometry_schema",
}

RUN_CONFIG_FILENAME = "run_config.json"
HISTORY_FILENAME = "history.json"
DATASET_CONTRACT_FILENAME = "dataset_contract.json"
SUMMARY_FILENAME = "summary.json"
TRAIN_METRICS_FILENAME = "train_metrics.json"
VALIDATION_METRICS_FILENAME = "validation_metrics.json"
TRAIN_PREDICTIONS_FILENAME = "train_predictions.csv"
VALIDATION_PREDICTIONS_FILENAME = "validation_predictions.csv"
BEST_CHECKPOINT_FILENAME = "best.pt"
LATEST_CHECKPOINT_FILENAME = "latest.pt"
MODEL_ARCHITECTURE_FILENAME = "model_architecture.txt"
LOSS_HISTORY_PLOT_FILENAME = "loss_history.png"
CENTER_ERROR_PLOT_FILENAME = "validation_center_error_histogram.png"
PREDICTION_SCATTER_PLOT_FILENAME = "validation_prediction_scatter.png"
HEATMAP_EXAMPLES_FILENAME = "validation_heatmap_examples.png"
CENTER_EXAMPLES_FILENAME = "validation_center_examples.png"

NUMERIC_MANIFEST_COLUMNS = (
    "image_width_px",
    "image_height_px",
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
    "bootstrap_center_x_px",
    "bootstrap_center_y_px",
)


@dataclass(frozen=True)
class DatasetReference:
    """One dataset reference under 02_training/datasets."""

    name: str
    datasets_root: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "datasets_root": str(self.datasets_root),
        }


@dataclass(frozen=True)
class SplitPaths:
    """Resolved paths for one dataset split."""

    dataset_reference: str
    split_name: str
    split_root: Path
    arrays_dir: Path
    manifests_dir: Path

    @property
    def run_json_path(self) -> Path:
        return self.manifests_dir / RUN_JSON_FILENAME

    @property
    def samples_csv_path(self) -> Path:
        return self.manifests_dir / SAMPLES_FILENAME


@dataclass(frozen=True)
class CorpusGeometryContract:
    """Authoritative model-space input contract for one loaded split."""

    canvas_width_px: int
    canvas_height_px: int
    image_layout: str
    channels: int
    normalization_range: tuple[float, float]
    geometry_schema: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitDatasetContract:
    """Validated split-level contract and artifact pointers."""

    dataset_reference: str
    split_name: str
    split_root: str
    run_json_path: str
    samples_csv_path: str
    row_count: int
    shard_count: int
    geometry: CorpusGeometryContract
    preprocessing_contract_version: str
    representation_kind: str
    representation_storage_format: str
    representation_array_keys: tuple[str, ...]
    bootstrap_bbox_available: bool
    fixed_roi_width_px: int | None
    fixed_roi_height_px: int | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["geometry"] = self.geometry.to_dict()
        return payload


@dataclass(frozen=True)
class RoiFcnBatch:
    """One split batch emitted by the shard iterator."""

    images: Any
    target_center_canvas_px: Any
    target_center_original_px: Any
    source_image_wh_px: Any
    resized_image_wh_px: Any
    padding_ltrb_px: Any
    resize_scale: Any
    sample_id: Any
    image_filename: Any
    npz_filename: tuple[str, ...]
    npz_row_index: Any
    bootstrap_bbox_xyxy_px: Any | None
    bootstrap_confidence: Any | None


@dataclass(frozen=True)
class DecodedHeatmapPoint:
    """Decoded peak location for one sample."""

    output_x: float
    output_y: float
    canvas_x: float
    canvas_y: float
    original_x: float
    original_y: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
