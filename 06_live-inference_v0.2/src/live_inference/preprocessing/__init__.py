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
    build_roi_locator_input_representation,
)
from .stage_policy import (
    DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR,
    ROI_LOCATOR_INPUT_MODE_AS_IS,
    ROI_LOCATOR_INPUT_MODE_INVERTED,
    ROI_LOCATOR_INPUT_MODE_SHEET_DARK_FOREGROUND,
    ROI_LOCATOR_INPUT_POLARITY_AS_IS,
    ROI_LOCATOR_INPUT_POLARITY_INVERTED,
    SUPPORTED_DIAGNOSTIC_PROFILES,
    SUPPORTED_ROI_LOCATOR_INPUT_MODES,
    SUPPORTED_ROI_LOCATOR_INPUT_POLARITIES,
    StageTransformPolicySnapshot,
    StageTransformPolicyState,
    diagnostic_profile_updates,
    normalize_diagnostic_profile_name,
    normalize_roi_locator_input_mode,
    normalize_roi_locator_input_polarity,
)
from .tri_stream_live_preprocessor import (
    PreprocessingDebugError,
    RoiLocatorDiagnosticResult,
    RoiRejectedError,
    TriStreamLivePreprocessor,
)

__all__ = [
    "BrightnessNormalizationRuntimeConfig",
    "RoiFcnArtifactMetadata",
    "RoiFcnLocatorInput",
    "RoiFcnLocator",
    "RoiLocatorDiagnosticResult",
    "RoiLocation",
    "RoiLocator",
    "RoiRejectedError",
    "PreprocessingDebugError",
    "DIAGNOSTIC_PROFILE_BASELINE_INVERTED_MASKED_LOCATOR",
    "ROI_LOCATOR_INPUT_MODE_AS_IS",
    "ROI_LOCATOR_INPUT_MODE_INVERTED",
    "ROI_LOCATOR_INPUT_MODE_SHEET_DARK_FOREGROUND",
    "ROI_LOCATOR_INPUT_POLARITY_AS_IS",
    "ROI_LOCATOR_INPUT_POLARITY_INVERTED",
    "StageTransformPolicySnapshot",
    "StageTransformPolicyState",
    "SUPPORTED_DIAGNOSTIC_PROFILES",
    "SUPPORTED_ROI_LOCATOR_INPUT_MODES",
    "SUPPORTED_ROI_LOCATOR_INPUT_POLARITIES",
    "TriStreamLivePreprocessor",
    "TriStreamPreprocessingConfig",
    "build_roi_fcn_exclusion_mask",
    "build_roi_fcn_locator_input",
    "build_roi_locator_input_representation",
    "decode_roi_fcn_heatmap",
    "diagnostic_profile_updates",
    "load_roi_fcn_artifact_metadata",
    "normalize_diagnostic_profile_name",
    "normalize_roi_locator_input_mode",
    "normalize_roi_locator_input_polarity",
    "resolve_roi_fcn_checkpoint",
]
