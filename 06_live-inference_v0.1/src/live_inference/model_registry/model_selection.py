"""Selection config parsing for live inference model artifacts.

The selection file chooses artifact roots only. Model compatibility remains the
responsibility of the model_registry compatibility checks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(frozen=True)
class ModelSelection:
    """Resolved model artifact selection for one live inference process."""

    selection_path: Path
    distance_orientation_root: Path
    roi_fcn_root: Path
    distance_orientation_device: str = "cuda"
    roi_fcn_device: str = "cuda"


class ModelSelectionError(ValueError):
    """Raised when a model selection TOML cannot be used safely."""


def load_model_selection(selection_path: Path) -> ModelSelection:
    """Load a live inference model selection TOML.

    Model root paths are resolved relative to the TOML file location. Absolute
    paths are rejected, and resolved roots must stay inside the live models tree.
    """
    path = Path(selection_path).resolve(strict=False)
    if not path.is_file():
        raise ModelSelectionError(f"Selection file does not exist: {selection_path}")

    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ModelSelectionError(f"Invalid selection TOML: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ModelSelectionError("Selection TOML must contain tables.")

    models_root = _models_root_for_selection(path)
    distance_orientation_root = _resolve_model_root(
        payload,
        selection_path=path,
        models_root=models_root,
        section_name="distance_orientation",
    )
    roi_fcn_root = _resolve_model_root(
        payload,
        selection_path=path,
        models_root=models_root,
        section_name="roi_fcn",
    )
    devices = _optional_table(payload, "device")

    return ModelSelection(
        selection_path=path,
        distance_orientation_root=distance_orientation_root,
        roi_fcn_root=roi_fcn_root,
        distance_orientation_device=_optional_string(
            devices,
            "distance_orientation",
            default="cuda",
        ),
        roi_fcn_device=_optional_string(devices, "roi_fcn", default="cuda"),
    )


def _models_root_for_selection(selection_path: Path) -> Path:
    for parent in selection_path.parents:
        if parent.name == "models":
            return parent.resolve()
    raise ModelSelectionError("Selection file must live under a models/ directory.")


def _resolve_model_root(
    payload: Mapping[str, Any],
    *,
    selection_path: Path,
    models_root: Path,
    section_name: str,
) -> Path:
    section = _required_table(payload, section_name)
    raw_root = section.get("root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise ModelSelectionError(f"[{section_name}].root must be a non-empty string.")

    relative_root = Path(raw_root)
    if relative_root.is_absolute():
        raise ModelSelectionError(f"[{section_name}].root must be relative, not absolute.")

    raw_candidate = selection_path.parent / relative_root
    candidate = raw_candidate.resolve(strict=False)
    if not _is_inside(candidate, models_root):
        raise ModelSelectionError(
            f"[{section_name}].root resolves outside the live models directory."
        )
    _reject_symlink_component(raw_candidate, models_root, section_name=section_name)
    if not candidate.is_dir():
        raise ModelSelectionError(f"[{section_name}].root does not exist: {candidate}")
    return candidate


def _required_table(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ModelSelectionError(f"Selection TOML must include a [{name}] table.")
    return value


def _optional_table(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ModelSelectionError(f"[{name}] must be a table when provided.")
    return value


def _optional_string(payload: Mapping[str, Any], key: str, *, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ModelSelectionError(f"[device].{key} must be a non-empty string.")
    return value


def _is_inside(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def _reject_symlink_component(path: Path, root: Path, *, section_name: str) -> None:
    if not path.exists():
        return

    relative_parts = path.relative_to(root).parts
    current = root
    for part in relative_parts:
        current = current / part
        if current.is_symlink():
            raise ModelSelectionError(f"[{section_name}].root must not use symlinks.")
