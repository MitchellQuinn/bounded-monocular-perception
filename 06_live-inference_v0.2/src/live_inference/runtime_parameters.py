"""Runtime parameter state management for live inference workers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Mapping

from interfaces import (
    RuntimeParameterSetSpec,
    RuntimeParameterSpec,
    RuntimeParameterUpdate,
    RuntimeParameterUpdateResult,
    RuntimeParameterValueType,
)


REQUEST_REJECTION_KEY = "__request__"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RuntimeParameterStateManager:
    """Validate and apply all-or-nothing runtime parameter updates."""

    def __init__(self, spec: RuntimeParameterSetSpec) -> None:
        self._spec = spec
        self._parameters_by_name = _index_parameters(spec.parameters)

    def current_spec(self) -> RuntimeParameterSetSpec:
        return self._spec

    def current_revision(self) -> int:
        return self._spec.revision

    def current_values(self) -> dict[str, Any]:
        return {
            parameter.name: parameter.current_value
            for parameter in self._spec.parameters
        }

    def apply_update(self, update: RuntimeParameterUpdate) -> RuntimeParameterUpdateResult:
        request_rejection = self._validate_request(update)
        if request_rejection is not None:
            return self._rejected_result(
                _request_rejections(update.updates, request_rejection),
                request_rejection,
            )

        validated_updates, rejected_updates = self._validate_parameter_updates(update.updates)
        if rejected_updates:
            return self._rejected_result(rejected_updates, "Rejected runtime parameter update.")

        changed = any(
            validated_updates[name] != self._parameters_by_name[name].current_value
            for name in validated_updates
        )
        if changed:
            self._replace_spec(validated_updates)
            message = (
                "Accepted runtime parameter update; "
                f"revision advanced to {self._spec.revision}."
            )
        else:
            message = "Accepted runtime parameter update; no values changed."

        return RuntimeParameterUpdateResult(
            owner=self._spec.owner,
            namespace=self._spec.namespace,
            accepted=True,
            revision=self._spec.revision,
            timestamp_utc=_utc_now_iso(),
            applied_updates=dict(validated_updates),
            rejected_updates={},
            message=message,
        )

    def _validate_request(self, update: RuntimeParameterUpdate) -> str | None:
        if update.owner != self._spec.owner:
            return (
                "Update owner does not match parameter set owner: "
                f"{update.owner!r} != {self._spec.owner!r}."
            )
        if update.namespace != self._spec.namespace:
            return (
                "Update namespace does not match parameter set namespace: "
                f"{update.namespace!r} != {self._spec.namespace!r}."
            )
        if update.base_revision is not None and update.base_revision != self._spec.revision:
            return (
                "Stale runtime parameter update revision: "
                f"{update.base_revision} != {self._spec.revision}."
            )
        return None

    def _validate_parameter_updates(
        self,
        updates: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        validated_updates: dict[str, Any] = {}
        rejected_updates: dict[str, str] = {}

        for name, value in updates.items():
            parameter = self._parameters_by_name.get(name)
            if parameter is None:
                rejected_updates[name] = "Unknown runtime parameter."
                continue
            if parameter.read_only:
                rejected_updates[name] = "Runtime parameter is read-only."
                continue

            is_valid, normalized_value, reason = _validate_value(parameter, value)
            if is_valid:
                validated_updates[name] = normalized_value
            else:
                rejected_updates[name] = reason

        if rejected_updates:
            validated_updates.clear()
        return validated_updates, rejected_updates

    def _replace_spec(self, updates: Mapping[str, Any]) -> None:
        updated_parameters = tuple(
            replace(parameter, current_value=updates[parameter.name])
            if parameter.name in updates
            else parameter
            for parameter in self._spec.parameters
        )
        self._spec = replace(
            self._spec,
            revision=self._spec.revision + 1,
            parameters=updated_parameters,
            timestamp_utc=_utc_now_iso(),
        )
        self._parameters_by_name = _index_parameters(self._spec.parameters)

    def _rejected_result(
        self,
        rejected_updates: Mapping[str, str],
        message: str,
    ) -> RuntimeParameterUpdateResult:
        return RuntimeParameterUpdateResult(
            owner=self._spec.owner,
            namespace=self._spec.namespace,
            accepted=False,
            revision=self._spec.revision,
            timestamp_utc=_utc_now_iso(),
            applied_updates={},
            rejected_updates=dict(rejected_updates),
            message=message,
        )


def _index_parameters(
    parameters: tuple[RuntimeParameterSpec, ...],
) -> dict[str, RuntimeParameterSpec]:
    indexed: dict[str, RuntimeParameterSpec] = {}
    for parameter in parameters:
        if parameter.name in indexed:
            raise ValueError(f"Duplicate runtime parameter name: {parameter.name!r}")
        indexed[parameter.name] = parameter
    return indexed


def _request_rejections(updates: Mapping[str, Any], reason: str) -> dict[str, str]:
    if not updates:
        return {REQUEST_REJECTION_KEY: reason}
    return {name: reason for name in updates}


def _validate_value(
    parameter: RuntimeParameterSpec,
    value: Any,
) -> tuple[bool, Any, str]:
    if parameter.value_type == RuntimeParameterValueType.BOOL:
        return _validate_bool(value)
    if parameter.value_type == RuntimeParameterValueType.INT:
        return _validate_int(parameter, value)
    if parameter.value_type == RuntimeParameterValueType.FLOAT:
        return _validate_float(parameter, value)
    if parameter.value_type == RuntimeParameterValueType.STRING:
        return _validate_string(parameter, value)
    if parameter.value_type == RuntimeParameterValueType.ENUM:
        return _validate_enum(parameter, value)
    return False, None, f"Unsupported runtime parameter value type: {parameter.value_type!r}."


def _validate_bool(value: Any) -> tuple[bool, Any, str]:
    if isinstance(value, bool):
        return True, value, ""
    return False, None, "Expected a bool value."


def _validate_int(
    parameter: RuntimeParameterSpec,
    value: Any,
) -> tuple[bool, Any, str]:
    if isinstance(value, bool) or not isinstance(value, int):
        return False, None, "Expected an int value."
    return _validate_bounded_choice(parameter, value)


def _validate_float(
    parameter: RuntimeParameterSpec,
    value: Any,
) -> tuple[bool, Any, str]:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False, None, "Expected a float value."
    return _validate_bounded_choice(parameter, float(value))


def _validate_string(
    parameter: RuntimeParameterSpec,
    value: Any,
) -> tuple[bool, Any, str]:
    if not isinstance(value, str):
        return False, None, "Expected a string value."
    return _validate_bounded_choice(parameter, value)


def _validate_enum(
    parameter: RuntimeParameterSpec,
    value: Any,
) -> tuple[bool, Any, str]:
    if value not in parameter.choices:
        return False, None, f"Expected one of {tuple(parameter.choices)!r}."
    return True, value, ""


def _validate_bounded_choice(
    parameter: RuntimeParameterSpec,
    value: Any,
) -> tuple[bool, Any, str]:
    if parameter.minimum is not None and value < parameter.minimum:
        return False, None, f"Value {value!r} is below minimum {parameter.minimum!r}."
    if parameter.maximum is not None and value > parameter.maximum:
        return False, None, f"Value {value!r} is above maximum {parameter.maximum!r}."
    if parameter.choices and value not in parameter.choices:
        return False, None, f"Expected one of {tuple(parameter.choices)!r}."
    return True, value, ""


__all__ = [
    "RuntimeParameterStateManager",
]
