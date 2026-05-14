"""Shared constants for the RB synthetic preprocessing v4 dual-stream pipeline."""

from __future__ import annotations

INPUT_ROOT_NAME = "input-images"
DETECT_ROOT_NAME = "detect-images-v4"
SILHOUETTE_ROOT_NAME = "silhouette-images-v4"
TRAINING_ROOT_NAME = "training-data-v4"
TRAINING_SHUFFLED_ROOT_NAME = "training-data-v4-shuffled"
TRI_STREAM_TRAINING_ROOT_NAME = "training-data-v4-tri-stream"
TRI_STREAM_TRAINING_SHUFFLED_ROOT_NAME = "training-data-v4-tri-stream-shuffled"

RUN_JSON_FILENAME = "run.json"
SAMPLES_FILENAME = "samples.csv"

PREPROCESSING_CONTRACT_KEY = "PreprocessingContract"
PREPROCESSING_CONTRACT_VERSION_V4 = "rb-preprocess-v4-dual-stream-orientation-brightness-v1"
PREPROCESSING_CONTRACT_VERSION_TRI_STREAM_V1 = "rb-preprocess-v4-tri-stream-orientation-v1"
PREPROCESSING_CONTRACT_VERSION_TRI_STREAM_GRAYSCALE_WHITE_V1 = (
    "rb-preprocess-v4-tri-stream-grayscale-white-v1"
)
PREPROCESSING_STAGE_ORDER_V4 = ("detect", "silhouette", "pack_dual_stream")
TRI_STREAM_STAGE_ORDER_V1 = ("detect", "silhouette", "pack_tri_stream")

KNOWN_STAGE_SUBDIRS = {"images", "arrays", "manifests"}

UNITY_REQUIRED_COLUMNS = [
    "run_id",
    "sample_id",
    "frame_index",
    "image_filename",
    "distance_m",
    "image_width_px",
    "image_height_px",
    "capture_success",
]

POSITION_TARGET_COLUMNS = [
    "final_pos_x_m",
    "final_pos_y_m",
    "final_pos_z_m",
]

ORIENTATION_TARGET_COLUMNS = [
    "yaw_deg",
    "yaw_sin",
    "yaw_cos",
]

DETECT_STAGE_COLUMNS = [
    "detect_stage_status",
    "detect_stage_error",
    "detect_model_path",
    "detect_model_sha256",
    "detect_class_id",
    "detect_class_name",
    "detect_confidence",
    "detect_bbox_x1",
    "detect_bbox_y1",
    "detect_bbox_x2",
    "detect_bbox_y2",
    "detect_bbox_w_px",
    "detect_bbox_h_px",
    "detect_center_x_px",
    "detect_center_y_px",
    "detect_candidates_total",
    "detect_debug_image_filename",
]

SILHOUETTE_STAGE_COLUMNS = [
    "silhouette_stage_status",
    "silhouette_stage_error",
    "silhouette_mode",
    "silhouette_image_filename",
    "silhouette_roi_image_filename",
    "silhouette_fallback_used",
    "silhouette_fallback_reason",
    "silhouette_area_px",
    "silhouette_bbox_x1",
    "silhouette_bbox_y1",
    "silhouette_bbox_x2",
    "silhouette_bbox_y2",
    "silhouette_quality_flags",
    "silhouette_debug_roi_filename",
    "silhouette_debug_amalgamated_filename",
]

PACK_STAGE_COLUMNS = [
    "pack_dual_stream_stage_status",
    "pack_dual_stream_stage_error",
    "npz_filename",
    "npz_row_index",
    "canvas_width_px",
    "canvas_height_px",
    "bbox_feat_cx_px",
    "bbox_feat_cy_px",
    "bbox_feat_w_px",
    "bbox_feat_h_px",
    "bbox_feat_cx_norm",
    "bbox_feat_cy_norm",
    "bbox_feat_w_norm",
    "bbox_feat_h_norm",
    "bbox_feat_aspect_ratio",
    "bbox_feat_area_norm",
]

TRI_STREAM_PACK_STAGE_COLUMNS = [
    "pack_tri_stream_stage_status",
    "pack_tri_stream_stage_error",
    "npz_filename",
    "npz_row_index",
    "tri_stream_canvas_width_px",
    "tri_stream_canvas_height_px",
    "bbox_feat_cx_px",
    "bbox_feat_cy_px",
    "bbox_feat_w_px",
    "bbox_feat_h_px",
    "bbox_feat_cx_norm",
    "bbox_feat_cy_norm",
    "bbox_feat_w_norm",
    "bbox_feat_h_norm",
    "bbox_feat_aspect_ratio",
    "bbox_feat_area_norm",
]

BRIGHTNESS_NORMALIZATION_COLUMNS = [
    "brightness_normalization_enabled",
    "brightness_normalization_method",
    "brightness_normalization_status",
    "brightness_normalization_foreground_px",
    "brightness_normalization_current_median_darkness",
    "brightness_normalization_effective_median_darkness",
    "brightness_normalization_gain",
]

FOREGROUND_ENHANCEMENT_COLUMNS = [
    "foreground_enhancement_enabled",
    "foreground_enhancement_method",
    "foreground_enhancement_status",
    "foreground_enhancement_foreground_px",
    "foreground_enhancement_current_median_darkness",
    "foreground_enhancement_effective_median_darkness",
    "foreground_enhancement_gain",
]

BBOX_FEATURE_SCHEMA = (
    "cx_px",
    "cy_px",
    "w_px",
    "h_px",
    "cx_norm",
    "cy_norm",
    "w_norm",
    "h_norm",
    "aspect_ratio",
    "area_norm",
)

TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY = "x_distance_image"
TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY = "x_orientation_image"
TRI_STREAM_GEOMETRY_ARRAY_KEY = "x_geometry"

REQUIRED_DUAL_STREAM_NPZ_KEYS = {
    "silhouette_crop",
    "bbox_features",
    "y_position_3d",
    "y_distance_m",
    "y_yaw_deg",
    "y_yaw_sin",
    "y_yaw_cos",
    "sample_id",
    "image_filename",
    "npz_row_index",
}

REQUIRED_TRI_STREAM_NPZ_KEYS = {
    TRI_STREAM_DISTANCE_IMAGE_ARRAY_KEY,
    TRI_STREAM_ORIENTATION_IMAGE_ARRAY_KEY,
    TRI_STREAM_GEOMETRY_ARRAY_KEY,
    "x_geometry_schema",
    "y_distance_m",
    "y_yaw_deg",
    "y_yaw_sin",
    "y_yaw_cos",
    "sample_id",
    "image_filename",
    "npz_row_index",
}
