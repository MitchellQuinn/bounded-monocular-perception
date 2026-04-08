"""Registry and resolution helpers for training/evaluation model topologies."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Mapping

from torch import nn

from . import topology_2d_cnn, topology_global_pool_cnn


@dataclass(frozen=True)
class TopologyDefinition:
    """Static registration record for one topology family."""

    topology_id: str
    model_class_name: str
    default_variant: str
    supported_variants: tuple[str, ...]
    build_model_fn: Callable[[str, Mapping[str, Any] | None], nn.Module]
    architecture_text_fn: Callable[[nn.Module], str]


@dataclass(frozen=True)
class ResolvedTopologySpec:
    """Normalized topology identity used by train/eval/resume."""

    topology_id: str
    topology_variant: str
    topology_params: dict[str, Any]
    model_class_name: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize topology spec as a plain mapping."""
        return {
            "topology_id": str(self.topology_id),
            "topology_variant": str(self.topology_variant),
            "topology_params": dict(self.topology_params),
            "model_class_name": str(self.model_class_name),
        }


DEFAULT_TOPOLOGY_ID = topology_2d_cnn.TOPOLOGY_ID


_REGISTRY: dict[str, TopologyDefinition] = {
    topology_2d_cnn.TOPOLOGY_ID: TopologyDefinition(
        topology_id=topology_2d_cnn.TOPOLOGY_ID,
        model_class_name=topology_2d_cnn.MODEL_CLASS_NAME,
        default_variant=topology_2d_cnn.DEFAULT_VARIANT,
        supported_variants=topology_2d_cnn.supported_variants(),
        build_model_fn=topology_2d_cnn.build_model,
        architecture_text_fn=topology_2d_cnn.architecture_text,
    ),
    topology_global_pool_cnn.TOPOLOGY_ID: TopologyDefinition(
        topology_id=topology_global_pool_cnn.TOPOLOGY_ID,
        model_class_name=topology_global_pool_cnn.MODEL_CLASS_NAME,
        default_variant=topology_global_pool_cnn.DEFAULT_VARIANT,
        supported_variants=topology_global_pool_cnn.supported_variants(),
        build_model_fn=topology_global_pool_cnn.build_model,
        architecture_text_fn=topology_global_pool_cnn.architecture_text,
    ),
}



def _normalize_topology_id(raw: Any) -> str:
    text = str(raw).strip()
    if not text:
        raise ValueError("topology_id cannot be empty.")
    return text



def canonicalize_topology_params(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Canonicalize topology params into a JSON-stable dictionary."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"topology_params must be a mapping/object; got {type(raw)}"
        )

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError("topology_params keys cannot be empty.")
        normalized[key_text] = value

    try:
        canonical_json = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError as exc:
        raise ValueError(
            "topology_params must be JSON-serializable for reproducible artifact tracking."
        ) from exc

    parsed = json.loads(canonical_json)
    if not isinstance(parsed, dict):
        raise ValueError("topology_params canonicalization failed to produce an object.")
    return parsed



def list_topology_ids() -> tuple[str, ...]:
    """Return all registered topology identifiers."""
    return tuple(sorted(_REGISTRY.keys()))



def list_topology_variants(topology_id: str) -> tuple[str, ...]:
    """Return supported variants for a topology id."""
    return get_topology_definition(topology_id).supported_variants



def get_topology_definition(topology_id: str) -> TopologyDefinition:
    """Lookup one topology definition."""
    key = _normalize_topology_id(topology_id)
    definition = _REGISTRY.get(key)
    if definition is None:
        raise ValueError(
            f"Unsupported topology_id={key!r}; expected one of {sorted(_REGISTRY)}"
        )
    return definition



def resolve_topology_spec(
    *,
    topology_id: str | None,
    topology_variant: str | None,
    topology_params: Mapping[str, Any] | None,
    legacy_model_architecture_variant: str | None = None,
) -> ResolvedTopologySpec:
    """Resolve topology config with legacy-variant compatibility."""
    resolved_topology_id = _normalize_topology_id(topology_id or DEFAULT_TOPOLOGY_ID)
    definition = get_topology_definition(resolved_topology_id)

    variant_text = str(topology_variant).strip() if topology_variant is not None else ""
    legacy_variant = (
        str(legacy_model_architecture_variant).strip()
        if legacy_model_architecture_variant is not None
        else ""
    )
    if variant_text and legacy_variant and variant_text != legacy_variant:
        raise ValueError(
            "topology_variant and model_architecture_variant disagree: "
            f"topology_variant={variant_text!r} "
            f"model_architecture_variant={legacy_variant!r}"
        )
    selected_variant = variant_text or legacy_variant or definition.default_variant
    if selected_variant not in definition.supported_variants:
        raise ValueError(
            f"Unsupported topology_variant={selected_variant!r} for topology_id={resolved_topology_id!r}; "
            f"expected one of {list(definition.supported_variants)}"
        )

    params = canonicalize_topology_params(topology_params)
    return ResolvedTopologySpec(
        topology_id=resolved_topology_id,
        topology_variant=selected_variant,
        topology_params=params,
        model_class_name=definition.model_class_name,
    )



def resolve_topology_spec_from_mapping(payload: Mapping[str, Any]) -> ResolvedTopologySpec:
    """Resolve topology spec from run config/manifest payloads."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping payload, got {type(payload)}")

    topology_block = payload.get("model_topology")
    topology_id = payload.get("topology_id")
    topology_variant = payload.get("topology_variant")
    topology_params = payload.get("topology_params")

    if isinstance(topology_block, Mapping):
        topology_id = topology_id or topology_block.get("topology_id")
        topology_variant = topology_variant or topology_block.get("topology_variant")
        if topology_params is None:
            topology_params = topology_block.get("topology_params")

    legacy_variant = payload.get("model_architecture_variant")
    params_mapping = topology_params if isinstance(topology_params, Mapping) else None

    return resolve_topology_spec(
        topology_id=(str(topology_id).strip() if topology_id is not None else None),
        topology_variant=(
            str(topology_variant).strip() if topology_variant is not None else None
        ),
        topology_params=params_mapping,
        legacy_model_architecture_variant=(
            str(legacy_variant).strip() if legacy_variant is not None else None
        ),
    )



def build_model_from_spec(spec: ResolvedTopologySpec) -> nn.Module:
    """Instantiate a model using a resolved topology spec."""
    definition = get_topology_definition(spec.topology_id)
    return definition.build_model_fn(
        spec.topology_variant,
        spec.topology_params,
    )



def architecture_text_from_spec(model: nn.Module, spec: ResolvedTopologySpec) -> str:
    """Render architecture text with topology identity prefix."""
    definition = get_topology_definition(spec.topology_id)
    body = definition.architecture_text_fn(model)
    return (
        f"topology_id={spec.topology_id}\n"
        f"model_class_name={spec.model_class_name}\n"
        f"{body}"
    )



def topology_spec_signature(spec: ResolvedTopologySpec | Mapping[str, Any]) -> str:
    """Return stable hash signature for topology-compatibility checks."""
    if isinstance(spec, ResolvedTopologySpec):
        payload = spec.to_dict()
    elif isinstance(spec, Mapping):
        raw_params = spec.get("topology_params")
        params_mapping = raw_params if isinstance(raw_params, Mapping) else None
        payload = {
            "topology_id": str(spec.get("topology_id", "")).strip(),
            "topology_variant": str(spec.get("topology_variant", "")).strip(),
            "topology_params": canonicalize_topology_params(params_mapping),
            "model_class_name": str(spec.get("model_class_name", "")).strip(),
        }
    else:
        raise ValueError(f"Unsupported spec type for signature: {type(spec)}")

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()
