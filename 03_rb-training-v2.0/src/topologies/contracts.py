"""Helpers for topology-declared runtime/output/reporting contracts."""

from __future__ import annotations

import json
from typing import Any, Mapping

TOPOLOGY_CONTRACT_VERSION = "rb-topology-output-reporting-v1"

_POSITION_3D_COLUMNS = (
    "final_pos_x_m",
    "final_pos_y_m",
    "final_pos_z_m",
)
_COLUMN_TO_NPZ_KEY = {
    "distance_m": "y_distance_m",
    "yaw_deg": "y_yaw_deg",
    "yaw_sin": "y_yaw_sin",
    "yaw_cos": "y_yaw_cos",
}
_RUNTIME_TASK_CONTRACT_KEYS = (
    "task_family",
    "prediction_mode",
    "input_mode",
    "output_kind",
    "target_columns",
    "debug_target_columns",
    "heads",
)


def _canonicalize_mapping(raw: Mapping[str, Any] | None, *, label: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be a mapping/object; got {type(raw)}")
    try:
        canonical_json = json.dumps(
            raw,
            sort_keys=True,
            separators=(",", ":"),
        )
    except TypeError as exc:
        raise ValueError(
            f"{label} must be JSON-serializable for reproducible artifact tracking."
        ) from exc
    parsed = json.loads(canonical_json)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} canonicalization failed to produce an object.")
    return parsed


def _text_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    values = raw if isinstance(raw, (list, tuple)) else [raw]
    items: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            items.append(text)
    return items


def _unique_text_list(raw: Any) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for text in _text_list(raw):
        if text not in seen:
            items.append(text)
            seen.add(text)
    return items


def _infer_target_kind(
    head_spec: Mapping[str, Any],
    columns: tuple[str, ...],
) -> str:
    metrics_role = str(head_spec.get("metrics_role", "")).strip()
    if metrics_role == "orientation":
        return "circular_regression"
    if metrics_role == "position" or tuple(columns) == _POSITION_3D_COLUMNS:
        return "vector_regression"
    return "regression"


def _npz_key_fields_for_target(
    *,
    kind: str,
    columns: tuple[str, ...],
    debug_columns: tuple[str, ...],
) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if kind == "vector_regression" and tuple(columns) == _POSITION_3D_COLUMNS:
        fields["target_npz_key"] = "y_position_3d"
    else:
        target_npz_keys = [_COLUMN_TO_NPZ_KEY.get(column) for column in columns]
        if target_npz_keys and all(key is not None for key in target_npz_keys):
            keys = [str(key) for key in target_npz_keys if key is not None]
            if len(keys) == 1:
                fields["target_npz_key"] = keys[0]
            elif keys:
                fields["target_npz_keys"] = keys

    debug_npz_keys = [_COLUMN_TO_NPZ_KEY.get(column) for column in debug_columns]
    if debug_npz_keys and all(key is not None for key in debug_npz_keys):
        keys = [str(key) for key in debug_npz_keys if key is not None]
        if len(keys) == 1:
            fields["debug_target_npz_key"] = keys[0]
        elif keys:
            fields["debug_target_npz_keys"] = keys
    return fields


def _reporting_family_from_runtime_contract(task_contract: Mapping[str, Any]) -> str:
    raw_heads = task_contract.get("heads")
    heads = raw_heads if isinstance(raw_heads, Mapping) else {}
    metrics_roles = {
        str(spec.get("metrics_role", "")).strip()
        for spec in heads.values()
        if isinstance(spec, Mapping)
    }
    if "orientation" in metrics_roles:
        return "distance_orientation_multitask"
    if "position" in metrics_roles:
        return "position_3d_regression"
    return "distance_regression"


def _legacy_reporting_contract(task_contract: Mapping[str, Any]) -> dict[str, Any]:
    raw_heads = task_contract.get("heads")
    heads = raw_heads if isinstance(raw_heads, Mapping) else {}
    loss_roles = [
        str(spec.get("loss_role", head_name)).strip() or str(head_name)
        for head_name, spec in heads.items()
        if isinstance(spec, Mapping)
    ]
    unique_loss_roles: list[str] = []
    seen: set[str] = set()
    for role in loss_roles:
        if role not in seen:
            unique_loss_roles.append(role)
            seen.add(role)

    train_losses = ["total_loss"]
    if len(unique_loss_roles) > 1:
        train_losses.extend(f"{role}_loss" for role in unique_loss_roles)

    family = _reporting_family_from_runtime_contract(task_contract)
    validation_metrics: list[str] = []
    orientation_thresholds_deg: list[float] = []
    if family == "distance_orientation_multitask":
        validation_metrics = [
            "yaw_mean_error_deg",
            "yaw_median_error_deg",
            "yaw_p95_error_deg",
        ]
    elif family == "position_3d_regression":
        validation_metrics = [
            "mean_position_error_m",
            "median_position_error_m",
            "p95_position_error_m",
        ]

    return {
        "family": family,
        "train_losses": train_losses,
        "validation_metrics": validation_metrics,
        "orientation_accuracy_thresholds_deg": orientation_thresholds_deg,
    }


def canonicalize_topology_contract(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Canonicalize a full topology output/reporting contract."""
    return _canonicalize_mapping(raw, label="topology_contract")


def task_contract_from_topology_contract(
    topology_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the runtime task contract consumed by train/eval from a topology contract."""
    contract = canonicalize_topology_contract(topology_contract)
    runtime = contract.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("topology_contract must define a runtime mapping.")
    raw_targets = contract.get("targets")
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        raise ValueError("topology_contract must define a non-empty targets mapping.")
    raw_outputs = contract.get("outputs")
    if not isinstance(raw_outputs, Mapping) or not raw_outputs:
        raise ValueError("topology_contract must define a non-empty outputs mapping.")
    raw_heads = runtime.get("heads")
    if not isinstance(raw_heads, Mapping) or not raw_heads:
        raise ValueError("topology_contract runtime must define a non-empty heads mapping.")

    task_heads: dict[str, Any] = {}
    target_columns: list[str] = []
    debug_target_columns: list[str] = []

    output_kind = str(runtime.get("output_kind", "tensor")).strip() or "tensor"
    for head_name, head_spec_raw in raw_heads.items():
        if not isinstance(head_spec_raw, Mapping):
            raise ValueError(
                f"topology_contract runtime head {head_name!r} must be a mapping; "
                f"got {type(head_spec_raw)}"
            )
        head_name_text = str(head_name).strip()
        target_name = str(head_spec_raw.get("target", head_name_text)).strip() or head_name_text
        output_name = str(head_spec_raw.get("output", target_name)).strip() or target_name

        target_spec = raw_targets.get(target_name)
        if not isinstance(target_spec, Mapping):
            raise ValueError(
                f"topology_contract runtime head {head_name_text!r} references unknown target "
                f"{target_name!r}."
            )
        output_spec = raw_outputs.get(output_name)
        if not isinstance(output_spec, Mapping):
            raise ValueError(
                f"topology_contract runtime head {head_name_text!r} references unknown output "
                f"{output_name!r}."
            )

        columns = _unique_text_list(target_spec.get("columns"))
        if not columns:
            raise ValueError(
                f"topology_contract target {target_name!r} must declare non-empty columns."
            )
        debug_columns = _unique_text_list(target_spec.get("debug_columns"))

        for column in columns:
            if column not in target_columns:
                target_columns.append(column)
        for column in debug_columns:
            if column not in debug_target_columns:
                debug_target_columns.append(column)

        task_head: dict[str, Any] = {
            "target_columns": list(columns),
            "metrics_role": str(head_spec_raw.get("metrics_role", head_name_text)).strip()
            or head_name_text,
            "loss_role": str(head_spec_raw.get("loss_role", head_name_text)).strip()
            or head_name_text,
            "target_kind": str(target_spec.get("kind", "regression")).strip() or "regression",
        }
        if debug_columns:
            task_head["debug_target_columns"] = list(debug_columns)
        if output_kind == "mapping":
            output_key = str(output_spec.get("output_key", "")).strip()
            if not output_key:
                raise ValueError(
                    f"topology_contract output {output_name!r} must define output_key for "
                    "runtime output_kind='mapping'."
                )
            task_head["output_key"] = output_key
        if "tensor_slice" in output_spec:
            raw_slice = output_spec["tensor_slice"]
            if not isinstance(raw_slice, (list, tuple)) or len(raw_slice) != 2:
                raise ValueError(
                    f"topology_contract output {output_name!r} tensor_slice must be a "
                    f"two-item list/tuple; got {raw_slice!r}"
                )
            task_head["tensor_slice"] = [int(raw_slice[0]), int(raw_slice[1])]
        for field_name in (
            "target_npz_key",
            "target_npz_keys",
            "debug_target_npz_key",
            "debug_target_npz_keys",
        ):
            if field_name in target_spec:
                task_head[field_name] = target_spec[field_name]
        task_heads[head_name_text] = task_head

    task_contract = {
        "task_family": str(contract.get("task_family", runtime.get("task_family", "regression"))).strip()
        or "regression",
        "prediction_mode": str(runtime.get("prediction_mode", "")).strip(),
        "input_mode": str(runtime.get("input_mode", "image_tensor")).strip() or "image_tensor",
        "output_kind": output_kind,
        "target_columns": list(target_columns),
        "debug_target_columns": list(debug_target_columns),
        "heads": task_heads,
        "reporting": dict(contract.get("reporting", {}))
        if isinstance(contract.get("reporting"), Mapping)
        else {},
    }
    return _canonicalize_mapping(task_contract, label="task_contract")


def legacy_task_contract_to_topology_contract(
    task_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Synthesize a topology contract for legacy topologies with runtime-only contracts."""
    runtime_contract = canonicalize_task_contract(task_contract)
    raw_heads = runtime_contract.get("heads")
    if not isinstance(raw_heads, Mapping) or not raw_heads:
        raise ValueError("Legacy task contract must define a non-empty heads mapping.")

    targets: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    runtime_heads: dict[str, Any] = {}

    for head_name, head_spec_raw in raw_heads.items():
        if not isinstance(head_spec_raw, Mapping):
            raise ValueError(
                f"Legacy task contract head {head_name!r} must be a mapping; got {type(head_spec_raw)}"
            )
        head_name_text = str(head_name).strip()
        columns = tuple(_unique_text_list(head_spec_raw.get("target_columns")))
        if not columns:
            raise ValueError(f"Legacy task contract head {head_name_text!r} is missing target_columns.")
        debug_columns = tuple(_unique_text_list(head_spec_raw.get("debug_target_columns")))
        kind = _infer_target_kind(head_spec_raw, columns)

        target_spec: dict[str, Any] = {
            "kind": kind,
            "columns": list(columns),
        }
        if debug_columns:
            target_spec["debug_columns"] = list(debug_columns)
        target_spec.update(
            _npz_key_fields_for_target(
                kind=kind,
                columns=columns,
                debug_columns=debug_columns,
            )
        )
        targets[head_name_text] = target_spec

        output_spec: dict[str, Any] = {
            "kind": kind,
            "columns": list(columns),
        }
        output_key = str(head_spec_raw.get("output_key", "")).strip()
        if output_key:
            output_spec["output_key"] = output_key
        raw_slice = head_spec_raw.get("tensor_slice")
        if isinstance(raw_slice, (list, tuple)) and len(raw_slice) == 2:
            output_spec["tensor_slice"] = [int(raw_slice[0]), int(raw_slice[1])]
        outputs[head_name_text] = output_spec

        runtime_heads[head_name_text] = {
            "target": head_name_text,
            "output": head_name_text,
            "metrics_role": str(head_spec_raw.get("metrics_role", head_name_text)).strip()
            or head_name_text,
            "loss_role": str(head_spec_raw.get("loss_role", head_name_text)).strip()
            or head_name_text,
        }

    topology_contract = {
        "contract_version": TOPOLOGY_CONTRACT_VERSION,
        "task_family": str(runtime_contract.get("task_family", "regression")).strip() or "regression",
        "targets": targets,
        "outputs": outputs,
        "runtime": {
            "prediction_mode": str(runtime_contract.get("prediction_mode", "")).strip(),
            "input_mode": str(runtime_contract.get("input_mode", "image_tensor")).strip()
            or "image_tensor",
            "output_kind": str(runtime_contract.get("output_kind", "tensor")).strip() or "tensor",
            "heads": runtime_heads,
        },
        "reporting": _legacy_reporting_contract(runtime_contract),
    }
    return canonicalize_topology_contract(topology_contract)


def canonicalize_task_contract(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Canonicalize only the runtime fields of a task contract."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"task_contract must be a mapping/object; got {type(raw)}")
    if "targets" in raw and "outputs" in raw and "runtime" in raw:
        return canonicalize_task_contract(task_contract_from_topology_contract(raw))

    payload = {
        key: raw[key]
        for key in _RUNTIME_TASK_CONTRACT_KEYS
        if key in raw
    }
    return _canonicalize_mapping(payload, label="task_contract")


def reporting_family(contract: Mapping[str, Any]) -> str:
    """Return the declared reporting family for a task or topology contract."""
    if not isinstance(contract, Mapping):
        return ""
    if "reporting" in contract and isinstance(contract["reporting"], Mapping):
        return str(contract["reporting"].get("family", "")).strip()
    if "task_contract" in contract and isinstance(contract["task_contract"], Mapping):
        return reporting_family(contract["task_contract"])
    if "targets" in contract and "outputs" in contract and "runtime" in contract:
        return reporting_family(task_contract_from_topology_contract(contract))
    return ""


def reporting_train_losses(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the declared train-loss names for artifact/logging selection."""
    if not isinstance(contract, Mapping):
        return ("total_loss",)
    reporting = contract.get("reporting")
    if isinstance(reporting, Mapping):
        names = tuple(_unique_text_list(reporting.get("train_losses")))
        if names:
            return names
    if "task_contract" in contract and isinstance(contract["task_contract"], Mapping):
        return reporting_train_losses(contract["task_contract"])
    if "targets" in contract and "outputs" in contract and "runtime" in contract:
        return reporting_train_losses(task_contract_from_topology_contract(contract))
    return ("total_loss",)


def reporting_validation_metrics(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the declared validation-metric names for artifact/logging selection."""
    if not isinstance(contract, Mapping):
        return ()
    reporting = contract.get("reporting")
    if isinstance(reporting, Mapping):
        names = tuple(_unique_text_list(reporting.get("validation_metrics")))
        if names:
            return names
    if "task_contract" in contract and isinstance(contract["task_contract"], Mapping):
        return reporting_validation_metrics(contract["task_contract"])
    if "targets" in contract and "outputs" in contract and "runtime" in contract:
        return reporting_validation_metrics(task_contract_from_topology_contract(contract))
    return ()


def reporting_orientation_accuracy_thresholds_deg(
    contract: Mapping[str, Any],
) -> tuple[float, ...]:
    """Return declared angular-accuracy thresholds for orientation reporting."""
    if not isinstance(contract, Mapping):
        return ()
    reporting = contract.get("reporting")
    if isinstance(reporting, Mapping):
        raw_thresholds = reporting.get("orientation_accuracy_thresholds_deg")
        if isinstance(raw_thresholds, (list, tuple)):
            values = sorted(
                {
                    float(value)
                    for value in raw_thresholds
                    if float(value) > 0.0
                }
            )
            return tuple(values)
    if "task_contract" in contract and isinstance(contract["task_contract"], Mapping):
        return reporting_orientation_accuracy_thresholds_deg(contract["task_contract"])
    if "targets" in contract and "outputs" in contract and "runtime" in contract:
        return reporting_orientation_accuracy_thresholds_deg(
            task_contract_from_topology_contract(contract)
        )
    return ()
