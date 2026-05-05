"""Concrete PyTorch tri-stream distance/orientation inference engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import interfaces.contracts as contracts
from interfaces.contracts import (
    FrameHash,
    InferenceInputMode,
    InferenceResult,
    PreparedInferenceInputs,
    RoiMetadata,
)
from live_inference.model_registry import (
    LiveModelManifest,
    load_live_model_manifest,
    load_model_selection,
    require_live_model_compatibility,
)
from live_inference.runtime.device import (
    normalize_torch_device_policy,
    resolve_torch_device,
)

from .output_decoding import decode_distance_yaw_outputs


DEFAULT_CHECKPOINT_NAME = "best.pt"
DEFAULT_SELECTION_PATH = Path("06_live-inference_v0.1/models/selections/current.toml")
DEFAULT_DISTANCE_ORIENTATION_ROOT = Path(
    "06_live-inference_v0.1/models/distance-orientation/"
    "260504-1100_ts-2d-cnn__run_0001"
)


class TorchTriStreamInferenceEngine:
    """Run a selected tri-stream distance/orientation PyTorch artifact."""

    def __init__(
        self,
        model_root: Path | str | None = None,
        *,
        device: str | None = None,
        checkpoint_name: str | None = None,
        model: Any | None = None,
        model_manifest: LiveModelManifest | None = None,
        load_model: bool | None = None,
        compatibility_checker: Callable[[LiveModelManifest], None] | None = None,
        now_utc_fn: Callable[[], str] | None = None,
    ) -> None:
        selected_root, selected_device = _resolve_selected_model_root_and_device(model_root)
        self._manifest = (
            model_manifest
            if model_manifest is not None
            else load_live_model_manifest(selected_root)
        )
        self._device_policy = normalize_torch_device_policy(
            device or selected_device or "auto"
        )
        self._resolved_device: str | None = None
        self._checkpoint_name = checkpoint_name
        self._model = model
        self._load_model = bool(model is None) if load_model is None else bool(load_model)
        self._compatibility_checker = (
            compatibility_checker
            if compatibility_checker is not None
            else require_live_model_compatibility
        )
        self._compatibility_checked = False
        self._now_utc_fn = now_utc_fn or _utc_now_iso
        self._task_contract: Mapping[str, Any] = _task_contract_from_manifest(self._manifest)
        if self._model is not None:
            _eval_model(self._model)

    @property
    def manifest(self) -> LiveModelManifest:
        """Normalized live model manifest used by this engine."""
        return self._manifest

    @property
    def model_loaded(self) -> bool:
        """Return whether the underlying PyTorch model is already available."""
        return self._model is not None

    def run_inference(self, inputs: PreparedInferenceInputs) -> InferenceResult:
        """Run one live distance/orientation inference request."""
        self._ensure_ready()
        assert self._model is not None

        start = perf_counter()
        torch = _import_torch()
        device_obj = self._torch_device(torch)
        input_mode = _coerce_input_mode(inputs.input_mode)
        if input_mode != InferenceInputMode.TRI_STREAM_V0_4:
            raise ValueError(
                "TorchTriStreamInferenceEngine requires tri-stream prepared inputs; "
                f"got input_mode={input_mode.value!r}."
            )
        model_inputs = _prepare_model_inputs(
            torch,
            inputs,
            device=device_obj,
            manifest=self._manifest,
        )

        with torch.no_grad():
            model_outputs = self._model(model_inputs)

        decoded = decode_distance_yaw_outputs(
            model_outputs,
            distance_key=self._manifest.distance_output_key,
            yaw_key=self._manifest.yaw_output_key,
        )
        inference_time_ms = (perf_counter() - start) * 1000.0

        preprocessing_metadata = dict(inputs.preprocessing_metadata)
        preprocessing_time_ms = _optional_float(
            preprocessing_metadata,
            "preprocessing_time_ms",
            "preprocess_time_ms",
        )
        total_time_ms = _optional_float(
            preprocessing_metadata,
            "total_time_ms",
            "end_to_end_time_ms",
        )
        if total_time_ms is None and preprocessing_time_ms is not None:
            total_time_ms = preprocessing_time_ms + inference_time_ms

        warnings = _metadata_warnings(preprocessing_metadata)
        input_image_path, input_image_hash, traceability_warnings = _source_traceability(
            inputs
        )

        return InferenceResult(
            request_id=inputs.request_id,
            input_image_path=input_image_path,
            input_image_hash=input_image_hash,
            timestamp_utc=self._now_utc_fn(),
            predicted_distance_m=decoded.distance_m,
            predicted_yaw_sin=decoded.yaw_sin,
            predicted_yaw_cos=decoded.yaw_cos,
            predicted_yaw_deg=decoded.yaw_deg,
            inference_time_ms=float(inference_time_ms),
            preprocessing_time_ms=preprocessing_time_ms,
            preprocessing_parameter_revision=_optional_int(
                preprocessing_metadata,
                "runtime_parameter_revision",
                "preprocessing_parameter_revision",
                "parameter_revision",
            ),
            total_time_ms=total_time_ms,
            model_input_mode=input_mode,
            roi_metadata=_roi_metadata_from_preprocessing(preprocessing_metadata),
            debug_paths=_debug_paths_from_preprocessing(preprocessing_metadata),
            warnings=tuple(warnings + traceability_warnings),
            extras={
                "model_root": str(self._manifest.model_root),
                "checkpoint_path": str(self._resolved_checkpoint_path()),
                "device": self._resolved_device or self._device_policy,
                "device_policy": self._device_policy,
                "model_label": self._manifest.model_label,
            },
        )

    def _ensure_ready(self) -> None:
        if not self._compatibility_checked:
            self._compatibility_checker(self._manifest)
            self._compatibility_checked = True
        if self._model is not None:
            return
        if not self._load_model:
            raise RuntimeError(
                "TorchTriStreamInferenceEngine has no loaded model. Instantiate with "
                "load_model=True or provide an injected model."
            )
        self._model = self._load_torch_model()

    def _load_torch_model(self) -> Any:
        torch = _import_torch()
        device_obj = self._torch_device(torch)
        _ensure_training_repo_path()
        from src.topologies.registry import (  # noqa: PLC0415
            build_model_from_spec,
            resolve_topology_spec_from_mapping,
            task_contract_signature,
            topology_contract_signature,
            topology_spec_signature,
        )

        run_config = _read_json_object(self._manifest.model_root / "config.json")
        topology_spec = resolve_topology_spec_from_mapping(run_config)
        _validate_topology_signatures(
            run_config,
            topology_spec,
            topology_spec_signature=topology_spec_signature,
            task_contract_signature=task_contract_signature,
            topology_contract_signature=topology_contract_signature,
        )
        self._task_contract = dict(topology_spec.task_contract)

        model = build_model_from_spec(topology_spec).to(device_obj)
        checkpoint_path = self._resolved_checkpoint_path()
        state = _torch_load_checkpoint(torch, checkpoint_path, map_location=device_obj)
        if isinstance(state, Mapping) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if not isinstance(state, Mapping):
            raise ValueError(f"Checkpoint is not a state-dict mapping: {checkpoint_path}")
        model.load_state_dict(state)
        model.eval()
        return model

    def _resolved_checkpoint_path(self) -> Path:
        if self._checkpoint_name is None:
            checkpoint_path = self._manifest.checkpoint_path
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No checkpoint was found under {self._manifest.model_root}."
                )
            return checkpoint_path
        return resolve_distance_orientation_checkpoint(
            self._manifest.model_root,
            checkpoint_name=self._checkpoint_name,
        )

    def _torch_device(self, torch: Any) -> Any:
        resolved_device = resolve_torch_device(self._device_policy)
        self._resolved_device = resolved_device
        return torch.device(resolved_device)


def resolve_distance_orientation_checkpoint(
    model_root: Path | str,
    *,
    checkpoint_name: str | None = None,
) -> Path:
    """Resolve a distance/orientation checkpoint, defaulting to ``best.pt``."""
    root = Path(model_root).expanduser().resolve()
    name = str(checkpoint_name or DEFAULT_CHECKPOINT_NAME).strip()
    if not name:
        raise ValueError("checkpoint_name cannot be blank.")
    if Path(name).name != name:
        raise ValueError(f"checkpoint_name must be a simple filename; got {name!r}.")
    checkpoint_path = (root / name).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Distance/orientation checkpoint {name!r} was not found under {root}."
        )
    return checkpoint_path


def _resolve_selected_model_root_and_device(
    model_root: Path | str | None,
) -> tuple[Path, str]:
    if model_root is not None:
        return Path(model_root).expanduser().resolve(), "auto"

    repo_root = _repo_root()
    selection_path = (repo_root / DEFAULT_SELECTION_PATH).resolve()
    if selection_path.is_file():
        selection = load_model_selection(selection_path)
        return selection.distance_orientation_root, selection.distance_orientation_device
    return (repo_root / DEFAULT_DISTANCE_ORIENTATION_ROOT).resolve(), "auto"


def _prepare_model_inputs(
    torch: Any,
    inputs: PreparedInferenceInputs,
    *,
    device: Any,
    manifest: LiveModelManifest,
) -> dict[str, Any]:
    raw_inputs = inputs.model_inputs
    required = contracts.TRI_STREAM_INPUT_KEYS
    missing = [key for key in required if key not in raw_inputs]
    if missing:
        raise KeyError(
            "PreparedInferenceInputs.model_inputs is missing required tri-stream "
            f"key(s): {missing}."
        )

    distance_image = _image_tensor(
        torch,
        raw_inputs[contracts.TRI_STREAM_DISTANCE_IMAGE_KEY],
        key=contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
        device=device,
    )
    orientation_image = _image_tensor(
        torch,
        raw_inputs[contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY],
        key=contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
        device=device,
    )
    geometry = _geometry_tensor(
        torch,
        raw_inputs[contracts.TRI_STREAM_GEOMETRY_KEY],
        key=contracts.TRI_STREAM_GEOMETRY_KEY,
        device=device,
    )

    batch_size = int(distance_image.shape[0])
    if int(orientation_image.shape[0]) != batch_size:
        raise ValueError(
            "x_orientation_image batch size must match x_distance_image; "
            f"got {tuple(orientation_image.shape)} vs {tuple(distance_image.shape)}."
        )
    if tuple(orientation_image.shape[2:]) != tuple(distance_image.shape[2:]):
        raise ValueError(
            "x_orientation_image spatial shape must match x_distance_image; "
            f"got {tuple(orientation_image.shape)} vs {tuple(distance_image.shape)}."
        )
    if int(geometry.shape[0]) != batch_size:
        raise ValueError(
            "x_geometry batch size must match x_distance_image; "
            f"got {tuple(geometry.shape)} vs {tuple(distance_image.shape)}."
        )
    if manifest.geometry_dim is not None and int(geometry.shape[1]) != int(
        manifest.geometry_dim
    ):
        raise ValueError(
            "x_geometry width mismatch; "
            f"expected {manifest.geometry_dim}, got {int(geometry.shape[1])}."
        )

    _validate_canvas_size(
        contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
        distance_image,
        manifest.distance_canvas_size,
    )
    _validate_canvas_size(
        contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
        orientation_image,
        manifest.orientation_canvas_size,
    )

    return {
        contracts.TRI_STREAM_DISTANCE_IMAGE_KEY: distance_image,
        contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY: orientation_image,
        contracts.TRI_STREAM_GEOMETRY_KEY: geometry,
    }


def _image_tensor(
    torch: Any,
    value: Any,
    *,
    key: str,
    device: Any,
) -> Any:
    tensor = _as_float_tensor(torch, value, key=key, device=device)
    if int(tensor.ndim) == 3:
        tensor = tensor.unsqueeze(0)
    elif int(tensor.ndim) != 4:
        raise ValueError(
            f"{key} must have shape (C, H, W) or (B, C, H, W); "
            f"got {tuple(tensor.shape)}."
        )
    if int(tensor.shape[1]) <= 0 or int(tensor.shape[2]) <= 0 or int(tensor.shape[3]) <= 0:
        raise ValueError(f"{key} dimensions must be positive; got {tuple(tensor.shape)}.")
    return tensor


def _geometry_tensor(
    torch: Any,
    value: Any,
    *,
    key: str,
    device: Any,
) -> Any:
    tensor = _as_float_tensor(torch, value, key=key, device=device)
    if int(tensor.ndim) == 1:
        tensor = tensor.unsqueeze(0)
    elif int(tensor.ndim) != 2:
        raise ValueError(f"{key} must have shape (F,) or (B, F); got {tuple(tensor.shape)}.")
    if int(tensor.shape[1]) <= 0:
        raise ValueError(f"{key} width must be positive; got {tuple(tensor.shape)}.")
    return tensor


def _as_float_tensor(
    torch: Any,
    value: Any,
    *,
    key: str,
    device: Any,
) -> Any:
    if torch.is_tensor(value):
        tensor = value
    else:
        try:
            tensor = torch.as_tensor(value)
        except Exception as exc:
            raise TypeError(f"{key} could not be converted to a torch tensor.") from exc
    return tensor.to(device=device, dtype=torch.float32)


def _validate_canvas_size(
    key: str,
    tensor: Any,
    expected_size: tuple[int, int] | None,
) -> None:
    if expected_size is None:
        return
    expected_w, expected_h = int(expected_size[0]), int(expected_size[1])
    actual_h, actual_w = int(tensor.shape[2]), int(tensor.shape[3])
    if (actual_w, actual_h) != (expected_w, expected_h):
        raise ValueError(
            f"{key} canvas size mismatch; expected {(expected_w, expected_h)}, "
            f"got {(actual_w, actual_h)}."
        )


def _source_traceability(
    inputs: PreparedInferenceInputs,
) -> tuple[Path, FrameHash, list[str]]:
    warnings: list[str] = []
    source_frame = inputs.source_frame
    if source_frame is None:
        warnings.append(
            "PreparedInferenceInputs.source_frame is missing; input traceability is incomplete."
        )
        return Path(""), FrameHash(""), warnings

    image_path = Path(source_frame.image_path)
    frame_hash = source_frame.frame_hash
    if frame_hash is None:
        warnings.append(
            "PreparedInferenceInputs.source_frame.frame_hash is missing; "
            "input_image_hash is empty."
        )
        frame_hash = FrameHash("")
    return image_path, frame_hash, warnings


def _roi_metadata_from_preprocessing(metadata: Mapping[str, Any]) -> RoiMetadata | None:
    bbox = _optional_float_tuple(
        metadata,
        "silhouette_bbox_xyxy_px",
        "roi_locator_bounds_xyxy_px",
        "roi_source_xyxy_px",
        width=4,
    )
    center = _optional_float_tuple(
        metadata,
        "predicted_roi_center_xy_px",
        "roi_center_xy_px",
        width=2,
    )
    if center is None and bbox is not None:
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    source_wh = _source_wh(metadata)
    distance_wh = _canvas_wh(
        metadata,
        width_key="distance_canvas_width_px",
        height_key="distance_canvas_height_px",
    )
    orientation_wh = _canvas_wh(
        metadata,
        width_key="orientation_canvas_width_px",
        height_key="orientation_canvas_height_px",
    )
    geometry_schema = _string_tuple(metadata.get("geometry_schema"))

    extras = {
        key: metadata[key]
        for key in (
            "roi_request_xyxy_px",
            "roi_source_xyxy_px",
            "roi_canvas_insert_xyxy_px",
            "roi_locator_bounds_xyxy_px",
            "roi_locator_metadata",
            "silhouette_bbox_inclusive_xyxy_px",
            "silhouette_area_px",
            "silhouette_fallback_used",
            "silhouette_primary_break_reason",
            "orientation_source_extent_xyxy_px",
            "orientation_crop_source_xyxy_px",
            "orientation_crop_size_px",
        )
        if key in metadata
    }

    if (
        bbox is None
        and center is None
        and source_wh is None
        and distance_wh is None
        and orientation_wh is None
        and not geometry_schema
        and not extras
    ):
        return None

    return RoiMetadata(
        bbox_xyxy_px=bbox,
        center_xy_px=center,
        source_image_wh_px=source_wh,
        distance_canvas_wh_px=distance_wh,
        orientation_canvas_wh_px=orientation_wh,
        geometry_schema=geometry_schema,
        extras=extras,
    )


def _debug_paths_from_preprocessing(metadata: Mapping[str, Any]) -> Mapping[str, Path]:
    raw = metadata.get("debug_paths")
    if raw is None:
        raw = metadata.get("debug_image_paths")
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): Path(value) for key, value in raw.items()}


def _metadata_warnings(metadata: Mapping[str, Any]) -> list[str]:
    raw = metadata.get("warnings")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return [f"preprocessing_metadata.warnings had unsupported type {type(raw).__name__}."]


def _coerce_input_mode(value: Any) -> InferenceInputMode:
    if isinstance(value, InferenceInputMode):
        return value
    return InferenceInputMode(str(value))


def _optional_float(metadata: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            try:
                return float(metadata[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"preprocessing_metadata[{key!r}] must be numeric.") from exc
    return None


def _optional_int(metadata: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in metadata and metadata[key] is not None:
            try:
                return int(metadata[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"preprocessing_metadata[{key!r}] must be an integer.") from exc
    return None


def _optional_float_tuple(
    metadata: Mapping[str, Any],
    *keys: str,
    width: int,
) -> tuple[float, ...] | None:
    for key in keys:
        if key not in metadata or metadata[key] is None:
            continue
        value = metadata[key]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"preprocessing_metadata[{key!r}] must be a sequence.")
        if len(value) != int(width):
            raise ValueError(
                f"preprocessing_metadata[{key!r}] must have width {width}; "
                f"got {len(value)}."
            )
        return tuple(float(item) for item in value)
    return None


def _source_wh(metadata: Mapping[str, Any]) -> tuple[int, int] | None:
    if "source_image_wh_px" in metadata:
        value = metadata["source_image_wh_px"]
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return int(value[0]), int(value[1])
    return _canvas_wh(
        metadata,
        width_key="source_image_width_px",
        height_key="source_image_height_px",
    )


def _canvas_wh(
    metadata: Mapping[str, Any],
    *,
    width_key: str,
    height_key: str,
) -> tuple[int, int] | None:
    if width_key not in metadata or height_key not in metadata:
        return None
    width = metadata[width_key]
    height = metadata[height_key]
    if width is None or height is None:
        return None
    return int(width), int(height)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _task_contract_from_manifest(manifest: LiveModelManifest) -> Mapping[str, Any]:
    for source_name in ("config", "model_architecture", "run_manifest"):
        source = manifest.raw_metadata.get(source_name)
        if isinstance(source, Mapping) and isinstance(source.get("task_contract"), Mapping):
            return dict(source["task_contract"])
    return {}


def _validate_topology_signatures(
    run_config: Mapping[str, Any],
    topology_spec: Any,
    *,
    topology_spec_signature: Callable[[Any], str],
    task_contract_signature: Callable[[Any], str],
    topology_contract_signature: Callable[[Any], str],
) -> None:
    checks = (
        ("topology_signature", topology_spec_signature),
        ("task_contract_signature", task_contract_signature),
        ("topology_contract_signature", topology_contract_signature),
    )
    for field, signature_fn in checks:
        expected = str(run_config.get(field, "")).strip()
        if not expected:
            continue
        actual = signature_fn(topology_spec)
        if actual != expected:
            raise ValueError(
                f"Run config {field} does not match resolved topology: "
                f"expected={expected} actual={actual}"
            )


def _read_json_object(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required model metadata file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Model metadata file must contain a JSON object: {path}")
    return payload


def _torch_load_checkpoint(torch: Any, path: Path, *, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _eval_model(model: Any) -> None:
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Torch is required to run TorchTriStreamInferenceEngine, but it is not installed."
        ) from exc


def _ensure_training_repo_path() -> None:
    training_root = _repo_root() / "03_rb-training-v2.0"
    resolved = str(training_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "DEFAULT_CHECKPOINT_NAME",
    "TorchTriStreamInferenceEngine",
    "resolve_distance_orientation_checkpoint",
]
