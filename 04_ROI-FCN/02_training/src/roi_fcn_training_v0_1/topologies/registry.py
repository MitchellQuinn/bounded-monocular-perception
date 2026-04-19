"""Registry and resolution helpers for ROI-FCN model topologies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from torch import nn

from ..utils import canonical_json_hash
from . import tiny_locator_fcn


@dataclass(frozen=True)
class TopologyDefinition:
    """Static registration record for one topology family."""

    topology_id: str
    model_class_name: str
    default_variant: str
    supported_variants: tuple[str, ...]
    build_model_fn: Callable[[str, Mapping[str, Any] | None], nn.Module]
    architecture_text_fn: Callable[[nn.Module], str]
    topology_metadata: dict[str, Any]
    resolve_topology_contract_fn: Callable[[str, Mapping[str, Any] | None], Mapping[str, Any]]


@dataclass(frozen=True)
class ResolvedTopologySpec:
    """Normalized topology identity used by train/eval/reporting."""

    topology_id: str
    topology_variant: str
    topology_params: dict[str, Any]
    model_class_name: str
    topology_metadata: dict[str, Any]
    topology_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "topology_variant": self.topology_variant,
            "topology_params": dict(self.topology_params),
            "model_class_name": self.model_class_name,
            "topology_metadata": dict(self.topology_metadata),
            "topology_contract": dict(self.topology_contract),
        }


def _build_definition(module: Any) -> TopologyDefinition:
    return TopologyDefinition(
        topology_id=module.TOPOLOGY_ID,
        model_class_name=module.MODEL_CLASS_NAME,
        default_variant=module.DEFAULT_VARIANT,
        supported_variants=module.supported_variants(),
        build_model_fn=module.build_model,
        architecture_text_fn=module.architecture_text,
        topology_metadata=dict(module.TOPOLOGY_METADATA),
        resolve_topology_contract_fn=module.resolve_topology_contract,
    )


_REGISTRY: dict[str, TopologyDefinition] = {
    tiny_locator_fcn.TOPOLOGY_ID: _build_definition(tiny_locator_fcn),
}


def _normalize_topology_id(raw: Any) -> str:
    text = str(raw).strip()
    if not text:
        raise ValueError("topology_id cannot be empty.")
    return text


def _canonicalize_topology_params(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"topology_params must be a mapping; got {type(raw)}")
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError("topology_params keys cannot be empty.")
        normalized[key_text] = value
    return normalized


def list_topology_ids(*, include_deprecated: bool = True) -> tuple[str, ...]:
    """Return registered topology ids."""
    _ = include_deprecated
    return tuple(sorted(_REGISTRY.keys()))


def list_topology_variants(topology_id: str) -> tuple[str, ...]:
    """Return supported variants for one topology id."""
    return get_topology_definition(topology_id).supported_variants


def get_topology_definition(topology_id: str) -> TopologyDefinition:
    """Lookup one topology definition."""
    key = _normalize_topology_id(topology_id)
    definition = _REGISTRY.get(key)
    if definition is None:
        raise ValueError(f"Unsupported topology_id={key!r}; expected one of {sorted(_REGISTRY)}")
    return definition


def resolve_topology_spec(
    *,
    topology_id: str,
    topology_variant: str | None,
    topology_params: Mapping[str, Any] | None,
) -> ResolvedTopologySpec:
    """Resolve topology config into a normalized spec."""
    definition = get_topology_definition(topology_id)
    selected_variant = str(topology_variant).strip() or definition.default_variant
    if selected_variant not in definition.supported_variants:
        raise ValueError(
            f"Unsupported topology_variant={selected_variant!r} for topology_id={topology_id!r}; "
            f"expected one of {list(definition.supported_variants)}"
        )
    params = _canonicalize_topology_params(topology_params)
    topology_contract = dict(definition.resolve_topology_contract_fn(selected_variant, params))
    return ResolvedTopologySpec(
        topology_id=definition.topology_id,
        topology_variant=selected_variant,
        topology_params=params,
        model_class_name=definition.model_class_name,
        topology_metadata=dict(definition.topology_metadata),
        topology_contract=topology_contract,
    )


def build_model_from_spec(spec: ResolvedTopologySpec) -> nn.Module:
    """Instantiate a model from a resolved topology spec."""
    definition = get_topology_definition(spec.topology_id)
    return definition.build_model_fn(spec.topology_variant, spec.topology_params)


def architecture_text_from_spec(spec: ResolvedTopologySpec) -> str:
    """Render a human-readable model architecture string."""
    model = build_model_from_spec(spec)
    definition = get_topology_definition(spec.topology_id)
    return definition.architecture_text_fn(model)


def topology_contract_signature(spec: ResolvedTopologySpec) -> str:
    """Hash the topology contract for run-time traceability."""
    return canonical_json_hash(spec.topology_contract)


def topology_spec_signature(spec: ResolvedTopologySpec) -> str:
    """Hash the full resolved topology identity."""
    return canonical_json_hash(spec.to_dict())
