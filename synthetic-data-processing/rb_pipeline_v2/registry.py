"""Component registries and factory lookups for v2 pipeline plugins."""

from __future__ import annotations

from .contracts import ArrayExporter, ArtifactWriter, FallbackStrategy, RepresentationGenerator

_GENERATORS: dict[str, RepresentationGenerator] = {}
_FALLBACKS: dict[str, FallbackStrategy] = {}
_WRITERS_BY_ID: dict[str, ArtifactWriter] = {}
_WRITERS_BY_MODE: dict[str, ArtifactWriter] = {}
_ARRAY_EXPORTERS: dict[str, ArrayExporter] = {}


def register_representation_generator(generator: RepresentationGenerator) -> None:
    _GENERATORS[str(generator.generator_id).strip()] = generator


def register_fallback_strategy(strategy: FallbackStrategy) -> None:
    _FALLBACKS[str(strategy.fallback_id).strip()] = strategy


def register_artifact_writer(writer: ArtifactWriter) -> None:
    writer_id = str(writer.writer_id).strip()
    mode = str(writer.representation_mode).strip().lower()
    _WRITERS_BY_ID[writer_id] = writer
    _WRITERS_BY_MODE[mode] = writer


def register_array_exporter(exporter: ArrayExporter) -> None:
    _ARRAY_EXPORTERS[str(exporter.exporter_id).strip()] = exporter


def get_representation_generator(generator_id: str) -> RepresentationGenerator:
    key = str(generator_id).strip()
    if key not in _GENERATORS:
        known = ", ".join(sorted(_GENERATORS.keys()))
        raise ValueError(f"Unknown generator_id '{generator_id}'. Registered: {known}")
    return _GENERATORS[key]


def get_fallback_strategy(fallback_id: str) -> FallbackStrategy:
    key = str(fallback_id).strip()
    if key not in _FALLBACKS:
        known = ", ".join(sorted(_FALLBACKS.keys()))
        raise ValueError(f"Unknown fallback_id '{fallback_id}'. Registered: {known}")
    return _FALLBACKS[key]


def get_artifact_writer_by_mode(representation_mode: str) -> ArtifactWriter:
    key = str(representation_mode).strip().lower()
    if key not in _WRITERS_BY_MODE:
        known = ", ".join(sorted(_WRITERS_BY_MODE.keys()))
        raise ValueError(
            f"Unknown representation_mode '{representation_mode}' for artifact writer. Registered: {known}"
        )
    return _WRITERS_BY_MODE[key]


def get_array_exporter(exporter_id: str) -> ArrayExporter:
    key = str(exporter_id).strip()
    if key not in _ARRAY_EXPORTERS:
        known = ", ".join(sorted(_ARRAY_EXPORTERS.keys()))
        raise ValueError(f"Unknown array_exporter_id '{exporter_id}'. Registered: {known}")
    return _ARRAY_EXPORTERS[key]


def list_registered_component_ids() -> dict[str, list[str]]:
    return {
        "generators": sorted(_GENERATORS.keys()),
        "fallbacks": sorted(_FALLBACKS.keys()),
        "artifact_writers": sorted(_WRITERS_BY_ID.keys()),
        "array_exporters": sorted(_ARRAY_EXPORTERS.keys()),
    }
