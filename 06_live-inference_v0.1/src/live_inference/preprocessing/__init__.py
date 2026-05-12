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
from .stage_policy import StageTransformPolicySnapshot, StageTransformPolicyState
from .tri_stream_live_preprocessor import RoiRejectedError, TriStreamLivePreprocessor

__all__ = [
    "BrightnessNormalizationRuntimeConfig",
    "RoiFcnArtifactMetadata",
    "RoiFcnLocatorInput",
    "RoiFcnLocator",
    "RoiLocation",
    "RoiLocator",
    "RoiRejectedError",
    "StageTransformPolicySnapshot",
    "StageTransformPolicyState",
    "TriStreamLivePreprocessor",
    "TriStreamPreprocessingConfig",
    "build_roi_fcn_exclusion_mask",
    "build_roi_fcn_locator_input",
    "decode_roi_fcn_heatmap",
    "load_roi_fcn_artifact_metadata",
    "resolve_roi_fcn_checkpoint",
]
