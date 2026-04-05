"""Default algorithm registrations for rb_pipeline_v2."""

from __future__ import annotations

from ..registry import (
    register_array_exporter,
    register_artifact_writer,
    register_fallback_strategy,
    register_representation_generator,
)
from .array_exporters import GrayscaleArrayExporterV1
from .silhouette_algorithms import (
    ConvexHullFallbackV1,
    ContourSilhouetteGeneratorV1,
    ContourSilhouetteGeneratorV2,
    FilledArtifactWriterV1,
    OutlineArtifactWriterV1,
)


def register_default_components() -> None:
    """Register pass-1 default generators, fallback strategies, and exporters."""

    register_representation_generator(ContourSilhouetteGeneratorV1())
    register_representation_generator(ContourSilhouetteGeneratorV2())
    register_fallback_strategy(ConvexHullFallbackV1())

    register_artifact_writer(OutlineArtifactWriterV1())
    register_artifact_writer(FilledArtifactWriterV1())

    register_array_exporter(GrayscaleArrayExporterV1())
