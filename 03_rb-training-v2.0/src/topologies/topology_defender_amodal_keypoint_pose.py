"""Defender amodal keypoint pose topology family dispatcher."""

from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from torch import nn

from . import topology_defender_amodal_keypoint_pose_v0_1
from .contracts import TOPOLOGY_CONTRACT_VERSION, task_contract_from_topology_contract
from .topology_tri_stream_yaw_common import parse_common_topology_params

TOPOLOGY_ID = "defender_amodal_keypoint_pose"
MODEL_CLASS_NAME = "DefenderAmodalKeypointPoseRegressor"
DEFAULT_VARIANT = topology_defender_amodal_keypoint_pose_v0_1.VARIANT
TOPOLOGY_METADATA = {
    "status": "experimental",
    "display_name": "Defender Amodal Keypoint Pose Regressor",
    "note": "Tri-stream Defender distance/yaw/center/amodal-keypoint/visibility multitask topology.",
    "replacement": "",
}

_SCHEMA_REPO_RELATIVE_PATH = "03_rb-training-v2.0/schemas/defender_keypoint_schema.json"
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "defender_keypoint_schema.json"
_CENTER_COLUMNS = (
    "defender_center_x_m",
    "defender_center_y_m",
    "defender_center_z_m",
)
_KEYPOINT_COLUMNS = tuple(
    f"defender_keypoint_{index:02d}_{axis}_m"
    for index in range(topology_defender_amodal_keypoint_pose_v0_1.NUM_DEFENDER_KEYPOINTS)
    for axis in ("x", "y", "z")
)
_VISIBILITY_COLUMNS = tuple(
    f"defender_keypoint_{index:02d}_visible"
    for index in range(topology_defender_amodal_keypoint_pose_v0_1.NUM_DEFENDER_KEYPOINTS)
)
_EXTRA_PARAM_KEYS = ("ablation_mode", "keypoint_hidden")


def _canonical_schema_hash(payload: Mapping[str, Any]) -> str:
    schema_payload = dict(payload)
    schema_payload.pop("schema_hash", None)
    canonical = json.dumps(schema_payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _load_schema() -> dict[str, Any]:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Defender keypoint schema must be a JSON object: {_SCHEMA_PATH}")
    declared_hash = str(payload.get("schema_hash", "")).strip()
    actual_hash = _canonical_schema_hash(payload)
    if declared_hash != actual_hash:
        raise ValueError(
            "Defender keypoint schema hash mismatch: "
            f"declared={declared_hash!r} actual={actual_hash!r} path={_SCHEMA_PATH}"
        )
    keypoints = payload.get("keypoints")
    if not isinstance(keypoints, list) or len(keypoints) != int(payload.get("num_keypoints", -1)):
        raise ValueError("Defender keypoint schema keypoints length must match num_keypoints.")
    return payload


_DEFENDER_KEYPOINT_SCHEMA = _load_schema()
_DEFENDER_SCHEMA_METADATA = {
    "defender_keypoint_schema_version": str(_DEFENDER_KEYPOINT_SCHEMA["schema_version"]),
    "defender_keypoint_schema_hash": str(_DEFENDER_KEYPOINT_SCHEMA["schema_hash"]),
    "defender_keypoint_schema_path": _SCHEMA_REPO_RELATIVE_PATH,
    "coordinate_space": str(_DEFENDER_KEYPOINT_SCHEMA["coordinate_space"]),
    "num_keypoints": int(_DEFENDER_KEYPOINT_SCHEMA["num_keypoints"]),
    "coordinate_width": int(_DEFENDER_KEYPOINT_SCHEMA["coordinate_width"]),
    "flattening_order": str(_DEFENDER_KEYPOINT_SCHEMA["flattening_order"]),
}
_DEFENDER_SCHEMA_REQUIREMENTS = {
    "defender_keypoint_schema": {
        "required": True,
        "required_npz_metadata": dict(_DEFENDER_SCHEMA_METADATA),
        "schema_id": str(_DEFENDER_KEYPOINT_SCHEMA["schema_id"]),
        "schema_version": str(_DEFENDER_KEYPOINT_SCHEMA["schema_version"]),
        "schema_hash": str(_DEFENDER_KEYPOINT_SCHEMA["schema_hash"]),
        "schema_path": _SCHEMA_REPO_RELATIVE_PATH,
        "schema_status": str(_DEFENDER_KEYPOINT_SCHEMA.get("schema_status", "")),
        "training_allowed": bool(_DEFENDER_KEYPOINT_SCHEMA.get("training_allowed", True)),
        "unresolved_keypoint_indices": [
            int(item["index"])
            for item in _DEFENDER_KEYPOINT_SCHEMA["keypoints"]
            if str(item.get("status", "")).strip() == "blocked_unresolved"
        ],
        "blocker": "Defender keypoint schema is marked unavailable for training.",
    }
}

_VARIANT_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    topology_defender_amodal_keypoint_pose_v0_1.VARIANT: topology_defender_amodal_keypoint_pose_v0_1.build_model,
}
_SUPPORTED_VARIANTS = frozenset(_VARIANT_BUILDERS)

TOPOLOGY_CONTRACT = {
    "contract_version": TOPOLOGY_CONTRACT_VERSION,
    "task_family": "defender_amodal_keypoint_pose",
    "schema_requirements": copy.deepcopy(_DEFENDER_SCHEMA_REQUIREMENTS),
    "targets": {
        "distance": {
            "kind": "regression",
            "columns": ["distance_m"],
            "target_npz_key": "y_distance_m",
        },
        "yaw": {
            "kind": "circular_regression",
            "columns": ["yaw_sin", "yaw_cos"],
            "debug_columns": ["yaw_deg"],
            "target_npz_keys": ["y_yaw_sin", "y_yaw_cos"],
            "debug_target_npz_key": "y_yaw_deg",
        },
        "defender_center_3d": {
            "kind": "vector_regression",
            "columns": list(_CENTER_COLUMNS),
            "target_npz_key": "y_defender_center_3d",
        },
        "defender_keypoints_3d": {
            "kind": "keypoint_regression",
            "columns": list(_KEYPOINT_COLUMNS),
            "target_npz_key": "y_defender_keypoints_3d_flat",
        },
        "defender_keypoints_visible": {
            "kind": "binary_classification",
            "columns": list(_VISIBILITY_COLUMNS),
            "target_npz_key": "y_defender_keypoints_visible",
        },
    },
    "outputs": {
        "distance": {
            "kind": "regression",
            "columns": ["distance_m"],
            "output_key": "distance_m",
        },
        "yaw": {
            "kind": "circular_regression",
            "columns": ["yaw_sin", "yaw_cos"],
            "output_key": "yaw_sin_cos",
        },
        "defender_center_3d": {
            "kind": "vector_regression",
            "columns": list(_CENTER_COLUMNS),
            "output_key": "defender_center_3d",
        },
        "defender_keypoints_3d": {
            "kind": "keypoint_regression",
            "columns": list(_KEYPOINT_COLUMNS),
            "output_key": "defender_keypoints_3d_flat",
        },
        "defender_keypoints_visible": {
            "kind": "binary_logits",
            "columns": list(_VISIBILITY_COLUMNS),
            "output_key": "defender_keypoints_visible_logits",
        },
    },
    "runtime": {
        "prediction_mode": "defender_amodal_keypoint_pose",
        "input_mode": "tri_stream_distance_orientation_geometry",
        "output_kind": "mapping",
        "schema_requirements": copy.deepcopy(_DEFENDER_SCHEMA_REQUIREMENTS),
        "heads": {
            "distance": {
                "output": "distance",
                "target": "distance",
                "metrics_role": "distance",
                "loss_role": "distance",
                "loss_kind": "huber",
            },
            "orientation": {
                "output": "yaw",
                "target": "yaw",
                "metrics_role": "orientation",
                "loss_role": "orientation",
                "loss_kind": "huber",
            },
            "defender_center_3d": {
                "output": "defender_center_3d",
                "target": "defender_center_3d",
                "metrics_role": "defender_center_3d",
                "loss_role": "center_3d",
                "loss_kind": "huber",
            },
            "defender_keypoints_3d": {
                "output": "defender_keypoints_3d",
                "target": "defender_keypoints_3d",
                "metrics_role": "defender_keypoints_3d",
                "loss_role": "keypoint_3d",
                "loss_kind": "huber",
            },
            "defender_keypoints_visible": {
                "output": "defender_keypoints_visible",
                "target": "defender_keypoints_visible",
                "metrics_role": "defender_keypoints_visible",
                "loss_role": "keypoint_visibility",
                "loss_kind": "bce_with_logits",
            },
        },
    },
    "reporting": {
        "family": "defender_amodal_keypoint_pose",
        "train_losses": [
            "total_loss",
            "distance_loss",
            "orientation_loss",
            "center_3d_loss",
            "keypoint_3d_loss",
            "keypoint_visibility_loss",
            "raw_distance_loss",
            "raw_orientation_loss",
            "raw_center_3d_loss",
            "raw_keypoint_3d_loss",
            "raw_keypoint_visibility_loss",
            "weighted_distance_loss",
            "weighted_orientation_loss",
            "weighted_center_3d_loss",
            "weighted_keypoint_3d_loss",
            "weighted_keypoint_visibility_loss",
        ],
        "validation_metrics": [
            "yaw_mean_error_deg",
            "yaw_median_error_deg",
            "yaw_p95_error_deg",
            "yaw_acc@5deg",
            "yaw_acc@10deg",
            "yaw_acc@15deg",
            "center_mean_error_m",
            "center_median_error_m",
            "center_p95_error_m",
            "keypoint_mean_point_error_m",
            "keypoint_median_point_error_m",
            "keypoint_p95_point_error_m",
            "keypoint_mean_coordinate_error_m",
            "visible_keypoint_mean_error_m",
            "hidden_keypoint_mean_error_m",
            "keypoint_visibility_accuracy",
            "keypoint_visibility_precision",
            "keypoint_visibility_recall",
            "keypoint_visibility_f1",
        ],
        "orientation_accuracy_thresholds_deg": [5.0, 10.0, 15.0],
    },
}


def supported_variants() -> tuple[str, ...]:
    """Return all allowed variants."""
    return tuple(sorted(_SUPPORTED_VARIANTS))


def _normalize_ablation_mode(raw: Any) -> str:
    if raw is None:
        return topology_defender_amodal_keypoint_pose_v0_1.ABLATION_TRI_STREAM
    mode = str(raw).strip().lower() or topology_defender_amodal_keypoint_pose_v0_1.ABLATION_TRI_STREAM
    if mode not in topology_defender_amodal_keypoint_pose_v0_1.SUPPORTED_ABLATION_MODES:
        raise ValueError(
            f"Unsupported ablation_mode={raw!r}; "
            f"expected one of {list(topology_defender_amodal_keypoint_pose_v0_1.SUPPORTED_ABLATION_MODES)}."
        )
    return mode


def _parse_topology_params(topology_params: Mapping[str, Any] | None) -> dict[str, Any]:
    params = dict(topology_params or {})
    ablation_mode = _normalize_ablation_mode(params.pop("ablation_mode", None))
    raw_keypoint_hidden = params.pop("keypoint_hidden", None)
    parsed = parse_common_topology_params(params, topology_id=TOPOLOGY_ID)
    parsed["ablation_mode"] = ablation_mode
    if raw_keypoint_hidden is not None:
        parsed["keypoint_hidden"] = int(raw_keypoint_hidden)
    return parsed


def resolve_topology_contract(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the declared output/reporting contract for this topology."""
    _ = topology_variant
    params = dict(topology_params or {})
    ablation_mode = _normalize_ablation_mode(params.get("ablation_mode"))
    contract = copy.deepcopy(TOPOLOGY_CONTRACT)
    contract["runtime"]["input_mode"] = (
        "geometry_only"
        if ablation_mode == topology_defender_amodal_keypoint_pose_v0_1.ABLATION_GEOMETRY_ONLY
        else "tri_stream_distance_orientation_geometry"
    )
    contract["runtime"]["ablation_mode"] = ablation_mode
    return contract


def resolve_task_contract(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the training/evaluation contract for this topology family."""
    return task_contract_from_topology_contract(
        resolve_topology_contract(topology_variant, topology_params)
    )


def build_model(
    topology_variant: str,
    topology_params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build one Defender amodal keypoint pose model instance."""
    variant = str(topology_variant).strip()
    builder = _VARIANT_BUILDERS.get(variant)
    if builder is None:
        raise ValueError(
            f"Unsupported topology_variant={topology_variant}; "
            f"expected one of {supported_variants()}"
        )
    return builder(**_parse_topology_params(topology_params))


def architecture_text(model: nn.Module) -> str:
    """Render architecture text persisted in run artifacts."""
    variant = getattr(model, "architecture_variant", "unknown")
    ablation_mode = getattr(model, "ablation_mode", "unknown")
    return f"architecture_variant={variant}\nablation_mode={ablation_mode}\n{model}"
