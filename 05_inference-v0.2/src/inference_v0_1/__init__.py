"""Public exports for the v0.2 inference package."""

from .brightness_analysis import (
    BrightnessSensitivityResult,
    apply_vehicle_darkness_gain,
    run_brightness_sensitivity_analysis,
)
from .discovery import (
    ModelRunArtifact,
    RawCorpus,
    default_raw_corpus_roots,
    discover_model_runs,
    discover_raw_corpora,
    list_corpus_image_names,
    load_corpus_samples,
    normalize_model_family,
    select_sample_row,
)
from .pipeline import (
    InferenceResult,
    ModelContext,
    PreprocessedSample,
    RoiFcnModelContext,
    load_model_context,
    load_roi_fcn_model_context,
    preprocess_single_sample,
    resolve_inference_device,
    run_multi_sample_inference,
    run_single_sample_inference,
    save_inference_result,
)

__all__ = [
    "BrightnessSensitivityResult",
    "InferenceResult",
    "ModelContext",
    "ModelRunArtifact",
    "PreprocessedSample",
    "RawCorpus",
    "RoiFcnModelContext",
    "apply_vehicle_darkness_gain",
    "default_raw_corpus_roots",
    "discover_model_runs",
    "discover_raw_corpora",
    "list_corpus_image_names",
    "load_corpus_samples",
    "load_model_context",
    "load_roi_fcn_model_context",
    "normalize_model_family",
    "preprocess_single_sample",
    "resolve_inference_device",
    "run_brightness_sensitivity_analysis",
    "run_multi_sample_inference",
    "run_single_sample_inference",
    "save_inference_result",
    "select_sample_row",
]
