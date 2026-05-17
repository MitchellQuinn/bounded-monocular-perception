"""Output decoding helpers for live distance/orientation inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import interfaces.contracts as contracts


@dataclass(frozen=True)
class DecodedDistanceYaw:
    """Decoded scalar prediction values for one live inference result."""

    distance_m: float
    yaw_sin: float
    yaw_cos: float
    yaw_deg: float


def yaw_degrees_from_sin_cos(yaw_sin: float, yaw_cos: float) -> float:
    """Decode yaw degrees from sin/cos components in the range [0, 360)."""
    yaw_deg = math.degrees(math.atan2(float(yaw_sin), float(yaw_cos))) % 360.0
    if yaw_deg < 0.0:
        yaw_deg += 360.0
    return float(yaw_deg)


def decode_distance_yaw_outputs(
    model_outputs: Any,
    *,
    distance_key: str | None = None,
    yaw_key: str | None = None,
) -> DecodedDistanceYaw:
    """Decode distance and yaw heads from model outputs.

    Mapping outputs are decoded by their declared output keys. Tensor-like or
    sequence outputs are treated as a single row with columns:
    ``distance_m, yaw_sin, yaw_cos``.
    """
    resolved_distance_key = distance_key or contracts.MODEL_OUTPUT_DISTANCE_KEY
    resolved_yaw_key = yaw_key or contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY

    if isinstance(model_outputs, Mapping):
        if resolved_distance_key not in model_outputs:
            raise KeyError(
                "Model output is missing distance head "
                f"{resolved_distance_key!r}."
            )
        if resolved_yaw_key not in model_outputs:
            raise KeyError(f"Model output is missing yaw head {resolved_yaw_key!r}.")
        distance_m = _first_scalar(
            model_outputs[resolved_distance_key],
            label=resolved_distance_key,
        )
        yaw_sin, yaw_cos = _first_vector2(
            model_outputs[resolved_yaw_key],
            label=resolved_yaw_key,
        )
    else:
        row = _first_vector(model_outputs, label="model_outputs", min_width=3)
        distance_m = _finite_float(row[0], label="model_outputs[0]")
        yaw_sin = _finite_float(row[1], label="model_outputs[1]")
        yaw_cos = _finite_float(row[2], label="model_outputs[2]")

    return DecodedDistanceYaw(
        distance_m=distance_m,
        yaw_sin=yaw_sin,
        yaw_cos=yaw_cos,
        yaw_deg=yaw_degrees_from_sin_cos(yaw_sin, yaw_cos),
    )


def _first_scalar(value: Any, *, label: str) -> float:
    plain = _plain_value(value)
    if _is_sequence(plain):
        vector = _first_vector(plain, label=label, min_width=1)
        return _finite_float(vector[0], label=label)
    return _finite_float(plain, label=label)


def _first_vector2(value: Any, *, label: str) -> tuple[float, float]:
    vector = _first_vector(value, label=label, min_width=2)
    return (
        _finite_float(vector[0], label=f"{label}[0]"),
        _finite_float(vector[1], label=f"{label}[1]"),
    )


def _first_vector(value: Any, *, label: str, min_width: int) -> list[Any]:
    plain = _plain_value(value)
    if not _is_sequence(plain):
        raise ValueError(f"{label} must be a sequence; got scalar {plain!r}.")
    row = list(plain)
    if row and _is_sequence(row[0]):
        row = list(row[0])
    if len(row) < int(min_width):
        raise ValueError(
            f"{label} must contain at least {min_width} value(s); got {len(row)}."
        )
    return row


def _plain_value(value: Any) -> Any:
    current = value
    detach = getattr(current, "detach", None)
    if callable(detach):
        current = detach()
    cpu = getattr(current, "cpu", None)
    if callable(cpu):
        current = cpu()
    numpy = getattr(current, "numpy", None)
    if callable(numpy):
        current = numpy()
    tolist = getattr(current, "tolist", None)
    if callable(tolist):
        return tolist()
    return current


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _finite_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric; got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; got {number!r}.")
    return number


__all__ = [
    "DecodedDistanceYaw",
    "decode_distance_yaw_outputs",
    "yaw_degrees_from_sin_cos",
]
