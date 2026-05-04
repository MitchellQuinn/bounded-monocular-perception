"""Lightweight JSON manifest normalization for live model artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


CHECKPOINT_CANDIDATES = ("best.pt", "best_model.pt", "latest.pt")
ORIENTATION_SOURCE_RAW_GRAYSCALE = "raw_grayscale"
ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE = "inverted_vehicle_on_white"
SUPPORTED_ORIENTATION_SOURCE_MODES = (
    ORIENTATION_SOURCE_RAW_GRAYSCALE,
    ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE,
)

_ORIENTATION_REPRESENTATION_SOURCE_MODES = {
    "target_centered_raw_grayscale_scaled_by_silhouette_extent": (
        ORIENTATION_SOURCE_RAW_GRAYSCALE
    ),
    "target_centered_inverted_vehicle_on_white_scaled_by_silhouette_extent": (
        ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE
    ),
}
_ORIENTATION_CONTENT_SOURCE_MODES = {
    "raw_grayscale_detail_preserving_no_brightness_normalization": (
        ORIENTATION_SOURCE_RAW_GRAYSCALE
    ),
    "inverted_vehicle_detail_on_white_no_brightness_normalization": (
        ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE
    ),
}
_ORIENTATION_POLARITY_SOURCE_MODES = {
    "dark_vehicle_detail_on_white_background": (
        ORIENTATION_SOURCE_INVERTED_VEHICLE_ON_WHITE
    ),
}

_MODEL_METADATA_FILES = {
    "live_model_manifest": "live_model_manifest.json",
    "config": "config.json",
    "run_manifest": "run_manifest.json",
    "dataset_summary": "dataset_summary.json",
    "model_architecture": "model_architecture.json",
}
_ROI_METADATA_FILES = {
    "roi_run_config": "run_config.json",
    "roi_dataset_contract": "dataset_contract.json",
}


@dataclass(frozen=True)
class LiveModelManifest:
    """Normalized metadata view for one live distance/orientation model bundle."""

    model_root: Path
    model_label: str | None
    checkpoint_path: Path | None
    checkpoint_kind: str | None

    topology_id: str | None
    topology_variant: str | None
    topology_contract_version: str | None

    preprocessing_contract_name: str | None
    input_mode: str | None
    representation_kind: str | None

    input_keys: tuple[str, ...]
    distance_image_key: str | None
    orientation_image_key: str | None
    geometry_key: str | None

    geometry_schema: tuple[str, ...]
    geometry_dim: int | None

    distance_canvas_size: tuple[int, int] | None
    orientation_canvas_size: tuple[int, int] | None
    orientation_image_representation: str | None
    orientation_image_content: str | None
    orientation_image_polarity: str | None
    orientation_source_mode: str | None

    model_output_keys: tuple[str, ...]
    distance_output_key: str | None
    yaw_output_key: str | None
    distance_output_width: int | None
    yaw_output_width: int | None

    roi_locator_root: Path | None
    roi_locator_crop_size: tuple[int, int] | None
    roi_locator_canvas_size: tuple[int, int] | None

    source_files: Mapping[str, Path]
    raw_metadata: Mapping[str, Any]


class OrientationSourceModeError(ValueError):
    """Raised when orientation metadata is present but cannot be normalized."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def resolve_orientation_source_mode(metadata: Mapping[str, Any]) -> str | None:
    """Normalize tri-stream orientation image semantics from artifact metadata."""
    preprocessing_contract = _preprocessing_contract_from_metadata(metadata)
    current = _mapping_at(
        preprocessing_contract,
        ("CurrentRepresentation",),
    ) or _mapping_at(preprocessing_contract, ("current_representation",))
    stage = _mapping_at(
        preprocessing_contract,
        ("Stages", "pack_tri_stream"),
    ) or _mapping_at(preprocessing_contract, ("stages", "pack_tri_stream"))
    sources: dict[str, str] = {}

    representation = _first_string(
        stage,
        (
            ("OrientationImageRepresentation",),
            ("orientation_image_representation",),
        ),
    )
    if representation:
        source_mode = _ORIENTATION_REPRESENTATION_SOURCE_MODES.get(representation)
        if source_mode is None:
            supported = ", ".join(sorted(_ORIENTATION_REPRESENTATION_SOURCE_MODES))
            raise OrientationSourceModeError(
                "unsupported_orientation_image_representation",
                "Unsupported tri-stream OrientationImageRepresentation="
                f"{representation!r}. Supported values: {supported}.",
            )
        sources["Stages.pack_tri_stream.OrientationImageRepresentation"] = source_mode

    content = _first_string(
        current,
        (
            ("OrientationImageContent",),
            ("orientation_image_content",),
        ),
    )
    if content:
        source_mode = _ORIENTATION_CONTENT_SOURCE_MODES.get(content)
        if source_mode is None:
            supported = ", ".join(sorted(_ORIENTATION_CONTENT_SOURCE_MODES))
            raise OrientationSourceModeError(
                "unsupported_orientation_image_content",
                f"Unsupported tri-stream OrientationImageContent={content!r}. "
                f"Supported values: {supported}.",
            )
        sources["CurrentRepresentation.OrientationImageContent"] = source_mode

    polarity = _first_string(
        current,
        (
            ("OrientationImagePolarity",),
            ("orientation_image_polarity",),
        ),
    )
    if polarity:
        source_mode = _ORIENTATION_POLARITY_SOURCE_MODES.get(polarity)
        if source_mode is None:
            supported = ", ".join(sorted(_ORIENTATION_POLARITY_SOURCE_MODES))
            raise OrientationSourceModeError(
                "unsupported_orientation_image_polarity",
                f"Unsupported tri-stream OrientationImagePolarity={polarity!r}. "
                f"Supported values: {supported}.",
            )
        sources["CurrentRepresentation.OrientationImagePolarity"] = source_mode

    source_modes = set(sources.values())
    if len(source_modes) > 1:
        details = ", ".join(f"{key}={value}" for key, value in sorted(sources.items()))
        raise OrientationSourceModeError(
            "conflicting_orientation_source_mode",
            "Conflicting tri-stream orientation preprocessing contract fields: "
            f"{details}.",
        )
    if not source_modes:
        return None
    return next(iter(source_modes))


def load_live_model_manifest(
    model_root: Path,
    *,
    roi_locator_root: Path | None = None,
) -> LiveModelManifest:
    """Load normalized manifest metadata without importing or loading model runtimes."""
    root = Path(model_root).expanduser().resolve()
    raw_metadata: dict[str, Any] = {}
    source_files: dict[str, Path] = {}

    for source_name, filename in _MODEL_METADATA_FILES.items():
        _read_optional_json(
            root / filename,
            source_name=source_name,
            raw_metadata=raw_metadata,
            source_files=source_files,
        )

    roi_root = Path(roi_locator_root).expanduser().resolve() if roi_locator_root else None
    if roi_root is not None:
        for source_name, filename in _ROI_METADATA_FILES.items():
            _read_optional_json(
                roi_root / filename,
                source_name=source_name,
                raw_metadata=raw_metadata,
                source_files=source_files,
            )

    checkpoint_path, checkpoint_kind = _discover_checkpoint(root)
    preprocessing_contract = _resolve_preprocessing_contract(raw_metadata)
    current_representation = _mapping_at(
        preprocessing_contract,
        ("CurrentRepresentation",),
    ) or _mapping_at(preprocessing_contract, ("current_representation",))
    pack_tri_stream_stage = _mapping_at(
        preprocessing_contract,
        ("Stages", "pack_tri_stream"),
    ) or _mapping_at(preprocessing_contract, ("stages", "pack_tri_stream"))
    try:
        orientation_source_mode = resolve_orientation_source_mode(preprocessing_contract)
    except OrientationSourceModeError:
        orientation_source_mode = None

    topology_id = _first_string_from_metadata(
        raw_metadata,
        (
            ("topology_id",),
            ("model_topology", "topology_id"),
            ("topology", "topology_id"),
        ),
    )
    topology_variant = _first_string_from_metadata(
        raw_metadata,
        (
            ("topology_variant",),
            ("model_architecture_variant",),
            ("model_topology", "topology_variant"),
            ("topology", "topology_variant"),
        ),
    )
    topology_contract_version = _first_string_from_metadata(
        raw_metadata,
        (
            ("topology_contract_version",),
            ("topology_contract", "contract_version"),
            ("model_topology", "topology_contract", "contract_version"),
        ),
    )

    input_mode = _first_string_from_metadata(
        raw_metadata,
        (
            ("input_mode",),
            ("task_contract", "input_mode"),
            ("model_topology", "task_contract", "input_mode"),
            ("topology_contract", "runtime", "input_mode"),
            ("model_topology", "topology_contract", "runtime", "input_mode"),
        ),
    )

    input_keys = _first_text_tuple_from_metadata(
        raw_metadata,
        (
            ("input_keys",),
            ("model_input_keys",),
            ("task_contract", "input_keys"),
            ("task_contract", "model_input_keys"),
            ("model_topology", "task_contract", "input_keys"),
            ("model_topology", "task_contract", "model_input_keys"),
        ),
    )
    if not input_keys:
        input_keys = _first_text_tuple(
            current_representation,
            (
                ("InputKeys",),
                ("input_keys",),
                ("ArrayKeys",),
                ("array_keys",),
            ),
        )

    distance_canvas_size = _first_size(
        current_representation,
        (
            ("DistanceCanvasSize",),
            ("distance_canvas_size",),
            ("distance_canvas_wh_px",),
        ),
    ) or _size_from_width_height(
        current_representation,
        (
            ("DistanceCanvasWidth", "DistanceCanvasHeight"),
            ("distance_canvas_width_px", "distance_canvas_height_px"),
            ("CanvasWidth", "CanvasHeight"),
            ("canvas_width_px", "canvas_height_px"),
        ),
    ) or _size_from_width_height(
        pack_tri_stream_stage,
        (
            ("DistanceCanvasWidth", "DistanceCanvasHeight"),
            ("CanvasWidth", "CanvasHeight"),
            ("canvas_width_px", "canvas_height_px"),
        ),
    )
    orientation_canvas_size = _first_size(
        current_representation,
        (
            ("OrientationCanvasSize",),
            ("orientation_canvas_size",),
            ("orientation_canvas_wh_px",),
        ),
    ) or _size_from_width_height(
        current_representation,
        (
            ("OrientationCanvasWidth", "OrientationCanvasHeight"),
            ("orientation_canvas_width_px", "orientation_canvas_height_px"),
            ("CanvasWidth", "CanvasHeight"),
            ("canvas_width_px", "canvas_height_px"),
        ),
    ) or _size_from_width_height(
        pack_tri_stream_stage,
        (
            ("OrientationCanvasWidth", "OrientationCanvasHeight"),
            ("CanvasWidth", "CanvasHeight"),
            ("canvas_width_px", "canvas_height_px"),
        ),
    )

    output_specs = _collect_output_specs(raw_metadata)
    resolved_distance_output_key, resolved_distance_output_width = _resolve_output(
        output_specs,
        "distance",
    )
    resolved_yaw_output_key, resolved_yaw_output_width = _resolve_output(output_specs, "yaw")
    model_output_keys = _resolve_model_output_keys(raw_metadata, output_specs)
    distance_output_key = _first_string_from_metadata(
        raw_metadata,
        (("distance_output_key",),),
    ) or resolved_distance_output_key
    yaw_output_key = _first_string_from_metadata(
        raw_metadata,
        (("yaw_output_key",),),
    ) or resolved_yaw_output_key
    explicit_distance_output_width = _first_int_from_metadata(
        raw_metadata,
        (("distance_output_width",),),
    )
    explicit_yaw_output_width = _first_int_from_metadata(
        raw_metadata,
        (("yaw_output_width",),),
    )
    distance_output_width = (
        explicit_distance_output_width
        if explicit_distance_output_width is not None
        else resolved_distance_output_width
    )
    yaw_output_width = (
        explicit_yaw_output_width
        if explicit_yaw_output_width is not None
        else resolved_yaw_output_width
    )
    representation_geometry_dim = _first_int(
        current_representation,
        (
            ("GeometryDim",),
            ("geometry_dim",),
        ),
    )
    geometry_dim = (
        representation_geometry_dim
        if representation_geometry_dim is not None
        else _first_int_from_metadata(
            raw_metadata,
            (
                ("geometry_dim",),
                ("task_contract", "geometry_dim"),
                ("model_topology", "task_contract", "geometry_dim"),
                ("topology_params", "geometry_feature_dim"),
                ("model_topology", "topology_params", "geometry_feature_dim"),
            ),
        )
    )
    orientation_image_representation = _first_string(
        pack_tri_stream_stage,
        (
            ("OrientationImageRepresentation",),
            ("orientation_image_representation",),
        ),
    )
    orientation_image_content = _first_string(
        current_representation,
        (
            ("OrientationImageContent",),
            ("orientation_image_content",),
        ),
    )
    orientation_image_polarity = _first_string(
        current_representation,
        (
            ("OrientationImagePolarity",),
            ("orientation_image_polarity",),
        ),
    )

    return LiveModelManifest(
        model_root=root,
        model_label=_first_string_from_metadata(
            raw_metadata,
            (
                ("model_label",),
                ("label",),
                ("model_name",),
            ),
        ),
        checkpoint_path=checkpoint_path,
        checkpoint_kind=checkpoint_kind,
        topology_id=topology_id,
        topology_variant=topology_variant,
        topology_contract_version=topology_contract_version,
        preprocessing_contract_name=_first_string_from_metadata(
            raw_metadata,
            (
                ("preprocessing_contract_name",),
                ("preprocessing_contract_version",),
            ),
        ) or _first_string(
            preprocessing_contract,
            (
                ("ContractVersion",),
                ("contract_version",),
                ("name",),
            ),
        ),
        input_mode=input_mode,
        representation_kind=_first_string_from_metadata(
            raw_metadata,
            (("representation_kind",),),
        ) or _first_string(
            current_representation,
            (
                ("Kind",),
                ("kind",),
                ("representation_kind",),
            ),
        ),
        input_keys=input_keys,
        distance_image_key=_first_string_from_metadata(
            raw_metadata,
            (("distance_image_key",),),
        ) or _first_string(
            current_representation,
            (
                ("DistanceImageKey",),
                ("distance_image_key",),
            ),
        ),
        orientation_image_key=_first_string_from_metadata(
            raw_metadata,
            (("orientation_image_key",),),
        ) or _first_string(
            current_representation,
            (
                ("OrientationImageKey",),
                ("orientation_image_key",),
            ),
        ),
        geometry_key=_first_string_from_metadata(
            raw_metadata,
            (("geometry_key",),),
        ) or _first_string(
            current_representation,
            (
                ("GeometryKey",),
                ("geometry_key",),
            ),
        ),
        geometry_schema=_first_text_tuple(
            current_representation,
            (
                ("GeometrySchema",),
                ("geometry_schema",),
                ("x_geometry_schema",),
            ),
        ) or _first_text_tuple_from_metadata(
            raw_metadata,
            (
                ("geometry_schema",),
                ("task_contract", "geometry_schema"),
                ("model_topology", "task_contract", "geometry_schema"),
            ),
        ),
        geometry_dim=geometry_dim,
        distance_canvas_size=distance_canvas_size,
        orientation_canvas_size=orientation_canvas_size,
        orientation_image_representation=orientation_image_representation,
        orientation_image_content=orientation_image_content,
        orientation_image_polarity=orientation_image_polarity,
        orientation_source_mode=orientation_source_mode,
        model_output_keys=model_output_keys,
        distance_output_key=distance_output_key,
        yaw_output_key=yaw_output_key,
        distance_output_width=distance_output_width,
        yaw_output_width=yaw_output_width,
        roi_locator_root=roi_root,
        roi_locator_crop_size=_resolve_roi_crop_size(raw_metadata),
        roi_locator_canvas_size=_resolve_roi_canvas_size(raw_metadata),
        source_files=source_files,
        raw_metadata=raw_metadata,
    )


def _read_optional_json(
    path: Path,
    *,
    source_name: str,
    raw_metadata: dict[str, Any],
    source_files: dict[str, Path],
) -> None:
    if not path.is_file():
        return
    try:
        raw_metadata[source_name] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata in {path}: {exc}") from exc
    source_files[source_name] = path.resolve()


def _discover_checkpoint(root: Path) -> tuple[Path | None, str | None]:
    for candidate_name in CHECKPOINT_CANDIDATES:
        candidate = root / candidate_name
        if candidate.is_file():
            return candidate.resolve(), candidate.stem
    return None, None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _metadata_sources(raw_metadata: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        _as_mapping(raw_metadata.get(key))
        for key in (
            "live_model_manifest",
            "config",
            "model_architecture",
            "run_manifest",
            "dataset_summary",
        )
    )


def _mapping_at(source: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
    current: Any = source
    for key in path:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


def _value_at(source: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence):
        items: list[str] = []
        for item in value:
            text = _text(item)
            if text:
                items.append(text)
        return tuple(items)
    return ()


def _first_string(
    source: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> str | None:
    for path in paths:
        text = _text(_value_at(source, path))
        if text is not None:
            return text
    return None


def _first_string_from_metadata(
    raw_metadata: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> str | None:
    for source in _metadata_sources(raw_metadata):
        value = _first_string(source, paths)
        if value is not None:
            return value
    return None


def _first_int(
    source: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> int | None:
    for path in paths:
        value = _int_value(_value_at(source, path))
        if value is not None:
            return value
    return None


def _first_int_from_metadata(
    raw_metadata: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> int | None:
    for source in _metadata_sources(raw_metadata):
        value = _first_int(source, paths)
        if value is not None:
            return value
    return None


def _first_text_tuple(
    source: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    for path in paths:
        values = _text_tuple(_value_at(source, path))
        if values:
            return values
    return ()


def _first_text_tuple_from_metadata(
    raw_metadata: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    for source in _metadata_sources(raw_metadata):
        values = _first_text_tuple(source, paths)
        if values:
            return values
    return ()


def _size_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        width = _first_int(value, (("width",), ("w",), ("canvas_width_px",)))
        height = _first_int(value, (("height",), ("h",), ("canvas_height_px",)))
        return (width, height) if width is not None and height is not None else None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        width = _int_value(value[0])
        height = _int_value(value[1])
        return (width, height) if width is not None and height is not None else None
    return None


def _first_size(
    source: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
) -> tuple[int, int] | None:
    for path in paths:
        value = _size_tuple(_value_at(source, path))
        if value is not None:
            return value
    return None


def _size_from_width_height(
    source: Mapping[str, Any],
    key_pairs: Sequence[tuple[str, str]],
) -> tuple[int, int] | None:
    for width_key, height_key in key_pairs:
        width = _int_value(source.get(width_key))
        height = _int_value(source.get(height_key))
        if width is not None and height is not None:
            return (width, height)
    return None


def _resolve_preprocessing_contract(raw_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = (
        ("live_model_manifest", ("preprocessing_contract",)),
        ("live_model_manifest", ("PreprocessingContract",)),
        ("dataset_summary", ("preprocessing_contract",)),
        ("dataset_summary", ("PreprocessingContract",)),
        ("run_manifest", ("dataset_summary", "preprocessing_contract")),
        ("run_manifest", ("dataset_summary", "PreprocessingContract")),
        ("run_manifest", ("preprocessing_contract",)),
        ("run_manifest", ("PreprocessingContract",)),
        ("config", ("dataset_summary", "preprocessing_contract")),
        ("config", ("dataset_summary", "PreprocessingContract")),
        ("config", ("preprocessing_contract",)),
        ("config", ("PreprocessingContract",)),
    )
    for source_name, path in candidates:
        candidate = _mapping_at(_as_mapping(raw_metadata.get(source_name)), path)
        if candidate:
            return candidate
    return {}


def _preprocessing_contract_from_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    if _mapping_at(metadata, ("CurrentRepresentation",)) or _mapping_at(metadata, ("Stages",)):
        return metadata
    if _mapping_at(metadata, ("current_representation",)) or _mapping_at(metadata, ("stages",)):
        return metadata
    for path in (
        ("preprocessing_contract",),
        ("PreprocessingContract",),
        ("dataset_summary", "preprocessing_contract"),
        ("dataset_summary", "PreprocessingContract"),
    ):
        candidate = _mapping_at(metadata, path)
        if candidate:
            return candidate
    return _resolve_preprocessing_contract(metadata)


def _append_unique(items: list[str], value: Any) -> None:
    text = _text(value)
    if text and text not in items:
        items.append(text)


def _collect_output_specs(raw_metadata: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    specs: dict[str, Mapping[str, Any]] = {}
    for source in _metadata_sources(raw_metadata):
        for path in (
            ("topology_contract", "outputs"),
            ("model_topology", "topology_contract", "outputs"),
            ("model_outputs",),
            ("task_contract", "outputs"),
            ("model_topology", "task_contract", "outputs"),
            ("task_contract", "heads"),
            ("model_topology", "task_contract", "heads"),
            ("outputs",),
        ):
            output_mapping = _mapping_at(source, path)
            for output_name, output_spec in output_mapping.items():
                if isinstance(output_spec, Mapping):
                    specs.setdefault(str(output_name), output_spec)
    return specs


def _resolve_model_output_keys(
    raw_metadata: Mapping[str, Any],
    output_specs: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    keys = list(
        _first_text_tuple_from_metadata(
            raw_metadata,
            (
                ("model_output_keys",),
                ("output_keys",),
                ("task_contract", "model_output_keys"),
                ("task_contract", "output_keys"),
                ("model_topology", "task_contract", "model_output_keys"),
                ("model_topology", "task_contract", "output_keys"),
            ),
        )
    )
    for output_name, output_spec in output_specs.items():
        _append_unique(keys, output_spec.get("output_key"))
        if output_name in {"distance_m", "yaw_sin_cos"}:
            _append_unique(keys, output_name)
    return tuple(keys)


def _resolve_output(
    output_specs: Mapping[str, Mapping[str, Any]],
    role: str,
) -> tuple[str | None, int | None]:
    for output_name, output_spec in output_specs.items():
        output_key = _text(output_spec.get("output_key")) or _text(output_name)
        columns = _text_tuple(output_spec.get("columns"))
        role_texts = {
            str(output_name).strip().lower(),
            str(output_spec.get("metrics_role", "")).strip().lower(),
            str(output_spec.get("loss_role", "")).strip().lower(),
            str(output_spec.get("target", "")).strip().lower(),
            str(output_spec.get("output", "")).strip().lower(),
        }
        if role == "distance":
            matched = (
                "distance" in role_texts
                or output_key == "distance_m"
                or columns == ("distance_m",)
            )
        else:
            matched = (
                bool({"yaw", "orientation"} & role_texts)
                or output_key == "yaw_sin_cos"
                or columns == ("yaw_sin", "yaw_cos")
            )
        if matched:
            return output_key, _resolve_output_width(output_spec)
    return None, None


def _resolve_output_width(output_spec: Mapping[str, Any]) -> int | None:
    width = _first_int(
        output_spec,
        (
            ("output_width",),
            ("width",),
            ("dim",),
            ("output_dim",),
        ),
    )
    if width is not None:
        return width
    columns = _text_tuple(output_spec.get("columns"))
    if columns:
        return len(columns)
    tensor_slice = _value_at(output_spec, ("tensor_slice",))
    if isinstance(tensor_slice, Sequence) and not isinstance(tensor_slice, (str, bytes)) and len(tensor_slice) == 2:
        start = _int_value(tensor_slice[0])
        end = _int_value(tensor_slice[1])
        if start is not None and end is not None:
            return max(0, end - start)
    return None


def _dataset_contract_split(dataset_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("train_split", "validation_split"):
        candidate = _mapping_at(dataset_contract, (key,))
        if candidate:
            return candidate
    return {}


def _resolve_roi_crop_size(raw_metadata: Mapping[str, Any]) -> tuple[int, int] | None:
    run_config = _as_mapping(raw_metadata.get("roi_run_config"))
    dataset_contract = _as_mapping(raw_metadata.get("roi_dataset_contract"))
    split_contract = _dataset_contract_split(dataset_contract)
    return _size_from_width_height(
        run_config,
        (
            ("roi_width_px", "roi_height_px"),
            ("roi_crop_width_px", "roi_crop_height_px"),
            ("crop_width_px", "crop_height_px"),
        ),
    ) or _size_from_width_height(
        split_contract,
        (
            ("fixed_roi_width_px", "fixed_roi_height_px"),
            ("roi_width_px", "roi_height_px"),
        ),
    )


def _resolve_roi_canvas_size(raw_metadata: Mapping[str, Any]) -> tuple[int, int] | None:
    run_config = _as_mapping(raw_metadata.get("roi_run_config"))
    dataset_contract = _as_mapping(raw_metadata.get("roi_dataset_contract"))
    split_contract = _dataset_contract_split(dataset_contract)
    geometry = _mapping_at(split_contract, ("geometry",))
    return _size_from_width_height(
        geometry,
        (
            ("canvas_width_px", "canvas_height_px"),
            ("width", "height"),
        ),
    ) or _size_from_width_height(
        run_config,
        (
            ("locator_canvas_width_px", "locator_canvas_height_px"),
            ("canvas_width_px", "canvas_height_px"),
        ),
    )
