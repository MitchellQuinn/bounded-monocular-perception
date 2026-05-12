"""Concrete ROI-FCN locator adapter for live preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import interfaces.contracts as contracts
import numpy as np

from live_inference.masking import (
    BackgroundSnapshot,
    compute_background_removal_mask_from_arrays,
)
from live_inference.runtime.device import (
    normalize_torch_device_policy,
    resolve_torch_device,
)

from .roi_locator import (
    RoiFcnLocatorInput,
    RoiLocation,
    build_roi_fcn_exclusion_mask,
    build_roi_fcn_locator_input,
)


DEFAULT_CHECKPOINT_NAME = "best.pt"


@dataclass(frozen=True)
class RoiFcnArtifactMetadata:
    """Resolved metadata needed to run one ROI-FCN live locator artifact."""

    roi_model_root: Path
    checkpoint_path: Path
    checkpoint_name: str
    run_config: Mapping[str, Any]
    dataset_contract: Mapping[str, Any]
    canvas_width_px: int
    canvas_height_px: int
    roi_width_px: int
    roi_height_px: int
    topology_id: str
    topology_variant: str
    topology_params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def locator_canvas_size(self) -> tuple[int, int]:
        return (int(self.canvas_width_px), int(self.canvas_height_px))

    @property
    def roi_crop_size(self) -> tuple[int, int]:
        return (int(self.roi_width_px), int(self.roi_height_px))


class RoiFcnLocator:
    """Load a selected ROI-FCN artifact and produce source-image ROI locations."""

    def __init__(
        self,
        roi_model_root: Path,
        *,
        device: str = "auto",
        checkpoint_name: str | None = None,
        load_model: bool = True,
        model: Any | None = None,
    ) -> None:
        self._metadata = load_roi_fcn_artifact_metadata(
            roi_model_root,
            checkpoint_name=checkpoint_name,
        )
        self._device_policy = normalize_torch_device_policy(device)
        self._resolved_device: str | None = None
        self._model = model
        if self._model is not None:
            _eval_model(self._model)
        elif load_model:
            self._model = self._load_model()
        self._background_locator_cache: tuple[tuple[int, int, int, int, int, int], np.ndarray] | None = None

    @property
    def metadata(self) -> RoiFcnArtifactMetadata:
        return self._metadata

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def locate(
        self,
        source_gray_image: Any,
        *,
        excluded_source_mask: np.ndarray | None = None,
        background_snapshot: BackgroundSnapshot | None = None,
        background_fill_value: int = 0,
    ) -> RoiLocation:
        """Run ROI-FCN on one grayscale source image and return source-space ROI."""
        if self._model is None:
            raise RuntimeError(
                "RoiFcnLocator was created without a loaded model; instantiate with "
                "load_model=True or provide an injected model before calling locate()."
            )

        source_gray = _coerce_source_gray(source_gray_image)
        locator_input = build_roi_fcn_locator_input(
            source_gray,
            canvas_width_px=self._metadata.canvas_width_px,
            canvas_height_px=self._metadata.canvas_height_px,
        )
        locator_input, background_metadata = self._apply_background_to_locator_input(
            locator_input,
            background_snapshot=background_snapshot,
            fill_value=background_fill_value,
        )
        heatmap = self._run_model(locator_input)
        return decode_roi_fcn_heatmap(
            heatmap,
            locator_input=locator_input,
            excluded_source_mask=excluded_source_mask,
            canvas_width_px=self._metadata.canvas_width_px,
            canvas_height_px=self._metadata.canvas_height_px,
            roi_width_px=self._metadata.roi_width_px,
            roi_height_px=self._metadata.roi_height_px,
            metadata={
                "roi_model_root": str(self._metadata.roi_model_root),
                "checkpoint_path": str(self._metadata.checkpoint_path),
                "checkpoint_name": self._metadata.checkpoint_name,
                "device": self._resolved_device or self._device_policy,
                "device_policy": self._device_policy,
                "topology_id": self._metadata.topology_id,
                "topology_variant": self._metadata.topology_variant,
                **background_metadata,
            },
        )

    def _apply_background_to_locator_input(
        self,
        locator_input: RoiFcnLocatorInput,
        *,
        background_snapshot: BackgroundSnapshot | None,
        fill_value: int,
    ) -> tuple[RoiFcnLocatorInput, dict[str, Any]]:
        metadata = {
            contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_REMOVAL_APPLIED: False,
            contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_REMOVE_PIXEL_COUNT: 0,
            contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_WARNING: None,
        }
        snapshot = background_snapshot
        if snapshot is None or not snapshot.captured or not snapshot.enabled:
            return locator_input, metadata

        src_w, src_h = (int(value) for value in locator_input.source_image_wh_px.tolist())
        if not snapshot.dimensions_match(src_w, src_h):
            metadata[contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_WARNING] = (
                "background removal skipped for ROI-FCN input: background size "
                f"{(snapshot.width_px, snapshot.height_px)} does not match source image "
                f"size {(src_w, src_h)}."
            )
            return locator_input, metadata

        current_u8 = _locator_image_uint8(locator_input.locator_image)
        background_u8 = self._background_locator_image(snapshot, locator_input)
        mask = compute_background_removal_mask_from_arrays(
            current_u8,
            background_u8,
            threshold=snapshot.threshold,
        )
        masked_locator_image = np.array(locator_input.locator_image, dtype=np.float32, copy=True)
        masked_locator_image[0][mask] = float(_coerce_fill_value(fill_value)) / 255.0
        metadata[
            contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_REMOVAL_APPLIED
        ] = True
        metadata[
            contracts.PREPROCESSING_METADATA_ROI_FCN_BACKGROUND_REMOVE_PIXEL_COUNT
        ] = int(np.count_nonzero(mask))
        return replace(locator_input, locator_image=masked_locator_image), metadata

    def _background_locator_image(
        self,
        snapshot: BackgroundSnapshot,
        locator_input: RoiFcnLocatorInput,
    ) -> np.ndarray:
        canvas_h, canvas_w = (int(value) for value in locator_input.locator_image.shape[-2:])
        key = (
            int(snapshot.revision),
            int(snapshot.width_px),
            int(snapshot.height_px),
            canvas_w,
            canvas_h,
            id(snapshot.grayscale_background),
        )
        if self._background_locator_cache is not None:
            cache_key, cached = self._background_locator_cache
            if cache_key == key:
                return cached
        background_input = build_roi_fcn_locator_input(
            snapshot.grayscale_background,
            canvas_width_px=canvas_w,
            canvas_height_px=canvas_h,
        )
        background_u8 = _locator_image_uint8(background_input.locator_image)
        self._background_locator_cache = (key, background_u8)
        return background_u8

    def _load_model(self) -> Any:
        torch = _import_torch()
        _ensure_roi_training_src_path()
        from roi_fcn_training_v0_1.topologies import (  # noqa: PLC0415
            build_model_from_spec,
            resolve_topology_spec,
        )

        device_obj = self._torch_device(torch)

        spec = resolve_topology_spec(
            topology_id=self._metadata.topology_id,
            topology_variant=self._metadata.topology_variant,
            topology_params=self._metadata.topology_params,
        )
        model = build_model_from_spec(spec).to(device_obj)
        checkpoint = _torch_load_checkpoint(
            torch,
            self._metadata.checkpoint_path,
            map_location=device_obj,
        )
        if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
            raise ValueError(
                "ROI-FCN checkpoint is missing model_state_dict: "
                f"{self._metadata.checkpoint_path}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _run_model(self, locator_input: RoiFcnLocatorInput) -> np.ndarray:
        torch = _import_torch()
        device_obj = self._torch_device(torch)
        input_tensor = torch.from_numpy(locator_input.locator_image[None, ...]).to(
            device=device_obj,
            dtype=torch.float32,
        )
        with torch.no_grad():
            heatmap_tensor = self._model(input_tensor)
        if heatmap_tensor.ndim != 4 or int(heatmap_tensor.shape[0]) != 1 or int(heatmap_tensor.shape[1]) != 1:
            raise ValueError(
                "ROI-FCN output must have shape (1, 1, H, W); "
                f"got {tuple(heatmap_tensor.shape)}"
            )
        return heatmap_tensor[0, 0].detach().cpu().numpy()

    def _torch_device(self, torch: Any) -> Any:
        resolved_device = resolve_torch_device(self._device_policy)
        self._resolved_device = resolved_device
        return torch.device(resolved_device)


def load_roi_fcn_artifact_metadata(
    roi_model_root: Path,
    *,
    checkpoint_name: str | None = None,
) -> RoiFcnArtifactMetadata:
    """Read live-local ROI-FCN artifact metadata without importing Torch."""
    root = Path(roi_model_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ROI-FCN model root does not exist: {root}")

    run_config = _read_json(root / "run_config.json")
    dataset_contract = _read_json(root / "dataset_contract.json")
    split_contract = _dataset_contract_split(dataset_contract)
    geometry = _mapping(split_contract.get("geometry"))
    output_hw = _mapping(run_config.get("output_hw"))

    canvas_width_px = _positive_int(
        geometry.get("canvas_width_px"),
        output_hw.get("width"),
        run_config.get(contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_WIDTH_PX),
        default=480,
        field_name="locator canvas width",
    )
    canvas_height_px = _positive_int(
        geometry.get("canvas_height_px"),
        output_hw.get("height"),
        run_config.get(contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_HEIGHT_PX),
        default=300,
        field_name="locator canvas height",
    )
    roi_width_px = _positive_int(
        run_config.get("roi_width_px"),
        split_contract.get("fixed_roi_width_px"),
        default=300,
        field_name="ROI crop width",
    )
    roi_height_px = _positive_int(
        run_config.get("roi_height_px"),
        split_contract.get("fixed_roi_height_px"),
        default=300,
        field_name="ROI crop height",
    )
    topology_id = _required_text(run_config.get("topology_id"), field_name="topology_id")
    topology_variant = _required_text(
        run_config.get("topology_variant"),
        field_name="topology_variant",
    )
    checkpoint_path = resolve_roi_fcn_checkpoint(root, checkpoint_name=checkpoint_name)

    return RoiFcnArtifactMetadata(
        roi_model_root=root,
        checkpoint_path=checkpoint_path,
        checkpoint_name=checkpoint_path.name,
        run_config=run_config,
        dataset_contract=dataset_contract,
        canvas_width_px=canvas_width_px,
        canvas_height_px=canvas_height_px,
        roi_width_px=roi_width_px,
        roi_height_px=roi_height_px,
        topology_id=topology_id,
        topology_variant=topology_variant,
        topology_params=_mapping(run_config.get("topology_params")),
    )


def resolve_roi_fcn_checkpoint(
    roi_model_root: Path,
    *,
    checkpoint_name: str | None = None,
) -> Path:
    """Resolve the requested ROI-FCN checkpoint without silent fallback."""
    root = Path(roi_model_root).expanduser().resolve()
    name = str(checkpoint_name or DEFAULT_CHECKPOINT_NAME).strip()
    if not name:
        raise ValueError("checkpoint_name cannot be blank.")
    if Path(name).name != name:
        raise ValueError(f"checkpoint_name must be a simple filename; got {name!r}.")
    checkpoint_path = (root / name).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"ROI-FCN checkpoint {name!r} was not found under {root}."
        )
    return checkpoint_path


def decode_roi_fcn_heatmap(
    heatmap: np.ndarray,
    *,
    locator_input: RoiFcnLocatorInput,
    excluded_source_mask: np.ndarray | None = None,
    canvas_width_px: int,
    canvas_height_px: int,
    roi_width_px: int,
    roi_height_px: int,
    metadata: Mapping[str, Any] | None = None,
) -> RoiLocation:
    """Decode an ROI-FCN heatmap peak into a live `RoiLocation`."""
    _ensure_roi_training_src_path()
    from roi_fcn_training_v0_1.geometry import (  # noqa: PLC0415
        decode_heatmap_argmax,
        derive_roi_bounds,
    )

    heatmap_array = np.asarray(heatmap)
    excluded_count = 0
    heatmap_before_exclusion = np.asarray(heatmap_array, dtype=np.float32)
    if excluded_source_mask is not None:
        excluded_heatmap_mask = build_roi_fcn_exclusion_mask(
            np.asarray(excluded_source_mask, dtype=bool),
            locator_input=locator_input,
            output_hw=tuple(int(value) for value in heatmap_array.shape),
        )
        excluded_count = int(np.count_nonzero(excluded_heatmap_mask))
        if excluded_count >= int(excluded_heatmap_mask.size):
            raise ValueError("ROI-FCN exclusion mask covers the entire output heatmap.")
        heatmap_array = np.array(heatmap_array, copy=True)
        heatmap_array[excluded_heatmap_mask] = -np.inf

    decoded = decode_heatmap_argmax(
        heatmap_array,
        canvas_hw=(int(canvas_height_px), int(canvas_width_px)),
        resize_scale=float(locator_input.resize_scale),
        pad_left_px=float(locator_input.padding_ltrb_px[0]),
        pad_top_px=float(locator_input.padding_ltrb_px[1]),
        source_wh_px=locator_input.source_image_wh_px,
    )
    center_xy = (float(decoded.original_x), float(decoded.original_y))
    roi_bounds = derive_roi_bounds(
        np.asarray(center_xy, dtype=np.float32),
        roi_width_px=float(roi_width_px),
        roi_height_px=float(roi_height_px),
    )
    metadata_payload: dict[str, Any] = dict(metadata or {})
    metadata_payload.update(
        {
            contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_WIDTH_PX: int(canvas_width_px),
            contracts.PREPROCESSING_METADATA_LOCATOR_CANVAS_HEIGHT_PX: int(canvas_height_px),
            contracts.PREPROCESSING_METADATA_ROI_WIDTH_PX: int(roi_width_px),
            contracts.PREPROCESSING_METADATA_ROI_HEIGHT_PX: int(roi_height_px),
            contracts.PREPROCESSING_METADATA_ROI_FCN_HEATMAP_U8: (
                _normalized_heatmap_uint8(heatmap_array)
            ),
            "roi_fcn_heatmap_pre_exclusion_u8": _normalized_heatmap_uint8(
                heatmap_before_exclusion
            ),
            "roi_fcn_heatmap_post_exclusion_u8": _normalized_heatmap_uint8(
                heatmap_array
            ),
            "locator_input_shape": tuple(int(value) for value in locator_input.locator_image.shape),
            contracts.PREPROCESSING_METADATA_SOURCE_IMAGE_WH_PX: tuple(
                int(value) for value in locator_input.source_image_wh_px.tolist()
            ),
            contracts.PREPROCESSING_METADATA_ROI_FCN_RESIZED_IMAGE_WH_PX: tuple(
                int(value) for value in locator_input.resized_image_wh_px.tolist()
            ),
            contracts.PREPROCESSING_METADATA_ROI_FCN_PADDING_LTRB_PX: tuple(
                int(value) for value in locator_input.padding_ltrb_px.tolist()
            ),
            "resize_scale": float(locator_input.resize_scale),
            "heatmap_shape": tuple(int(value) for value in heatmap_array.shape),
            "decoded_heatmap": decoded.to_dict(),
            "heatmap_peak_confidence": float(decoded.confidence),
            "heatmap_exclusion_mask_applied": excluded_source_mask is not None,
            "heatmap_exclusion_pixel_count": excluded_count,
        }
    )
    return RoiLocation(
        center_xy_px=center_xy,
        roi_bounds_xyxy_px=_array4_to_tuple(roi_bounds),
        metadata=metadata_payload,
    )


def _normalized_heatmap_uint8(heatmap: np.ndarray) -> np.ndarray:
    numeric = np.asarray(heatmap, dtype=np.float32)
    if numeric.ndim != 2:
        return np.zeros((0, 0), dtype=np.uint8)

    finite_mask = np.isfinite(numeric)
    if not bool(np.any(finite_mask)):
        return np.zeros(numeric.shape, dtype=np.uint8)

    finite = numeric[finite_mask]
    finite_min = float(np.min(finite))
    finite_max = float(np.max(finite))
    normalized = np.zeros(numeric.shape, dtype=np.float32)
    if finite_min >= 0.0 and finite_max <= 1.0:
        normalized[finite_mask] = np.clip(numeric[finite_mask], 0.0, 1.0)
    elif finite_max > finite_min:
        normalized[finite_mask] = (
            (numeric[finite_mask] - finite_min) / (finite_max - finite_min)
        )
    elif finite_max > 0.0:
        normalized[finite_mask] = 1.0
    return np.ascontiguousarray(np.rint(normalized * 255.0).astype(np.uint8))


def _ensure_roi_training_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    src_root = repo_root / "04_ROI-FCN" / "02_training" / "src"
    resolved = str(src_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Torch is required to run RoiFcnLocator model inference, but it is not installed."
        ) from exc


def _torch_load_checkpoint(torch: Any, path: Path, *, map_location: Any) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _eval_model(model: Any) -> None:
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()


def _coerce_source_gray(source_gray_image: Any) -> np.ndarray:
    source_gray = np.asarray(source_gray_image)
    if source_gray.ndim != 2:
        raise ValueError(f"Expected grayscale 2D image, got {source_gray.shape}")
    if source_gray.dtype != np.uint8:
        source_gray = np.clip(source_gray, 0, 255).astype(np.uint8)
    return source_gray


def _locator_image_uint8(locator_image: np.ndarray) -> np.ndarray:
    image = np.asarray(locator_image, dtype=np.float32)
    if image.ndim == 3 and int(image.shape[0]) == 1:
        image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Expected ROI-FCN locator image shape (1, H, W), got {image.shape}.")
    return np.ascontiguousarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8))


def _coerce_fill_value(fill_value: int) -> int:
    value = int(fill_value)
    if value not in {0, 255}:
        raise ValueError(f"background_fill_value must be 0 or 255; got {fill_value!r}.")
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required ROI-FCN metadata file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON metadata in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"ROI-FCN metadata file must contain a JSON object: {path}")
    return payload


def _dataset_contract_split(dataset_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("train_split", "validation_split"):
        candidate = dataset_contract.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(
    *values: Any,
    default: int,
    field_name: str,
) -> int:
    for value in values:
        parsed = _int_value(value)
        if parsed is not None:
            if parsed <= 0:
                raise ValueError(f"{field_name} must be > 0; got {parsed}.")
            return parsed
    if int(default) <= 0:
        raise ValueError(f"{field_name} default must be > 0; got {default}.")
    return int(default)


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


def _required_text(value: Any, *, field_name: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"Missing required ROI-FCN metadata field {field_name}.")
    return text


def _array4_to_tuple(values: np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float32).reshape(4)
    return (
        float(array[0]),
        float(array[1]),
        float(array[2]),
        float(array[3]),
    )


__all__ = [
    "DEFAULT_CHECKPOINT_NAME",
    "RoiFcnArtifactMetadata",
    "RoiFcnLocator",
    "decode_roi_fcn_heatmap",
    "load_roi_fcn_artifact_metadata",
    "resolve_roi_fcn_checkpoint",
]
