"""Concrete preprocessing adapters for live inference."""

from .preprocessing_config import (
    BrightnessNormalizationRuntimeConfig,
    TriStreamPreprocessingConfig,
)
from .roi_fcn_locator import (
    RoiFcnArtifactMetadata,
    RoiFcnLocator,
    decode_roi_fcn_heatmap,
    load_roi_fcn_artifact_metadata,
    resolve_roi_fcn_checkpoint,
)
from .roi_locator import (
    RoiFcnLocatorInput,
    RoiLocation,
    RoiLocator,
    build_roi_fcn_exclusion_mask,
    build_roi_fcn_locator_input,
)
from .stage_policy import (
    ROI_LOCATOR_INPUT_POLARITY_AS_IS,
    ROI_LOCATOR_INPUT_POLARITY_INVERTED,
    SUPPORTED_ROI_LOCATOR_INPUT_POLARITIES,
    StageTransformPolicySnapshot,
    StageTransformPolicyState,
    normalize_roi_locator_input_polarity,
)
from .tri_stream_live_preprocessor import RoiRejectedError, TriStreamLivePreprocessor

__all__ = [
    "BrightnessNormalizationRuntimeConfig",
    "RoiFcnArtifactMetadata",
    "RoiFcnLocatorInput",
    "RoiFcnLocator",
    "RoiLocation",
    "RoiLocator",
    "RoiRejectedError",
    "ROI_LOCATOR_INPUT_POLARITY_AS_IS",
    "ROI_LOCATOR_INPUT_POLARITY_INVERTED",
    "StageTransformPolicySnapshot",
    "StageTransformPolicyState",
    "SUPPORTED_ROI_LOCATOR_INPUT_POLARITIES",
    "TriStreamLivePreprocessor",
    "TriStreamPreprocessingConfig",
    "build_roi_fcn_exclusion_mask",
    "build_roi_fcn_locator_input",
    "decode_roi_fcn_heatmap",
    "load_roi_fcn_artifact_metadata",
    "normalize_roi_locator_input_polarity",
    "resolve_roi_fcn_checkpoint",
]
