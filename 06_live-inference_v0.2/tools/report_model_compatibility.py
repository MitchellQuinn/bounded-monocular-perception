"""Generate the live inference model compatibility report.

This is a metadata-only report generator. It uses the live model manifest
loader and compatibility checker for distance/orientation artifacts, and reads
ROI-FCN JSON sidecars without importing or loading model runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import interfaces.contracts as contracts  # noqa: E402
from live_inference.model_registry import (  # noqa: E402
    check_live_model_compatibility,
    load_live_model_manifest,
)
from live_inference.model_registry.compatibility import ERROR, WARNING  # noqa: E402
from live_inference.model_registry.model_manifest import CHECKPOINT_CANDIDATES  # noqa: E402


DISTANCE_BASES = (
    ("old-tree", REPO_ROOT / "05_inference-v0.4-ts/models/distance-orientation"),
    ("live-tree", PROJECT_ROOT / "models/distance-orientation"),
)
ROI_BASES = (
    ("old-tree", REPO_ROOT / "05_inference-v0.4-ts/models/roi-fcn"),
    ("live-tree", PROJECT_ROOT / "models/roi-fcn"),
)
DOCUMENTS_ROOT = PROJECT_ROOT / "documents"
MARKDOWN_REPORT = DOCUMENTS_ROOT / "model_compatibility_matrix.md"
JSON_REPORT = DOCUMENTS_ROOT / "model_compatibility_matrix.json"

MODEL_METADATA_FILES = {
    "live_model_manifest.json",
    "config.json",
    "run_manifest.json",
    "dataset_summary.json",
    "model_architecture.json",
}
ROI_METADATA_FILES = {"run_config.json", "dataset_contract.json"}


@dataclass(frozen=True)
class ArtifactRoot:
    root: Path
    location: str
    kind: str


def main() -> None:
    DOCUMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    distance_registry_roots = _discover_distance_registry_roots()
    distance_artifacts = _discover_distance_artifacts()
    roi_artifacts = _discover_roi_artifacts()

    distance_rows = [
        _distance_registry_row(root)
        for root in distance_registry_roots
    ] + [_distance_artifact_row(artifact) for artifact in distance_artifacts]
    roi_rows = [_roi_artifact_row(artifact) for artifact in roi_artifacts]

    compatible_distance_rows = [
        row
        for row in distance_rows
        if row["artifact_kind"] == "run-artifact" and row["compatible"] == "yes"
    ]
    pairing_rows = _build_pairing_rows(compatible_distance_rows, roi_rows)
    recommendation = _build_recommendation(compatible_distance_rows, roi_rows)
    blockers = _build_blockers(
        distance_artifacts=distance_artifacts,
        roi_artifacts=roi_artifacts,
        distance_registry_roots=distance_registry_roots,
    )
    summary = {
        "distance_orientation_artifacts_scanned": len(distance_artifacts),
        "distance_orientation_registry_roots_inspected": len(distance_registry_roots),
        "roi_fcn_artifacts_scanned": len(roi_artifacts),
        "compatible_live_tri_stream_distance_orientation_models": len(
            compatible_distance_rows
        ),
        "likely_current_deployment_candidate": recommendation[
            "distance_orientation"
        ],
        "roi_fcn_candidate": recommendation["roi_fcn"],
        "blockers": blockers,
    }

    payload = {
        "generated": date.today().isoformat(),
        "scope": {
            "distance_orientation_bases": [
                {"location": location, "root": _relpath(root)}
                for location, root in DISTANCE_BASES
            ],
            "roi_fcn_bases": [
                {"location": location, "root": _relpath(root)}
                for location, root in ROI_BASES
            ],
            "metadata_only": True,
            "checkpoint_loading": False,
            "symlinks_created": False,
        },
        "summary": summary,
        "distance_orientation": distance_rows,
        "roi_fcn": roi_rows,
        "pairing_notes": pairing_rows,
        "recommendation": recommendation,
    }

    JSON_REPORT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    MARKDOWN_REPORT.write_text(_render_markdown(payload), encoding="utf-8")


def _discover_distance_registry_roots() -> list[ArtifactRoot]:
    roots: list[ArtifactRoot] = []
    for location, base in DISTANCE_BASES:
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / "run_register.json").is_file():
                roots.append(ArtifactRoot(child, location, "registry-root"))
    return roots


def _discover_distance_artifacts() -> list[ArtifactRoot]:
    artifacts: list[ArtifactRoot] = []
    seen: set[Path] = set()
    for location, base in DISTANCE_BASES:
        if not base.is_dir():
            continue
        for candidate in sorted(path for path in base.rglob("*") if path.is_dir()):
            if _has_any_file(candidate, MODEL_METADATA_FILES) or _has_checkpoint(candidate):
                resolved = candidate.resolve()
                if resolved not in seen:
                    artifacts.append(ArtifactRoot(candidate, location, "run-artifact"))
                    seen.add(resolved)
    return artifacts


def _discover_roi_artifacts() -> list[ArtifactRoot]:
    artifacts: list[ArtifactRoot] = []
    seen: set[Path] = set()
    for location, base in ROI_BASES:
        if not base.is_dir():
            continue
        for candidate in sorted(path for path in base.rglob("*") if path.is_dir()):
            if _has_any_file(candidate, ROI_METADATA_FILES) or _has_checkpoint(candidate):
                resolved = candidate.resolve()
                if resolved not in seen:
                    artifacts.append(ArtifactRoot(candidate, location, "run-artifact"))
                    seen.add(resolved)
    return artifacts


def _distance_registry_row(artifact: ArtifactRoot) -> dict[str, Any]:
    manifest = load_live_model_manifest(artifact.root)
    result = check_live_model_compatibility(manifest)
    run_dirs = _registered_run_dirs(artifact.root)
    family = _likely_family(artifact.root, manifest)
    errors = [
        "metadata missing: no loader-recognized model metadata files or checkpoint "
        "at the requested registry root",
        (
            "loader discovery gap: root contains run_register.json; actual "
            f"artifacts are under {_join_paths(run_dirs) if run_dirs else 'runs/'}"
        ),
    ]
    return {
        "artifact_root": _relpath(artifact.root),
        "artifact_kind": artifact.kind,
        "location": artifact.location,
        "compatible": "no",
        "likely_family": family,
        "checkpoint_selected": _checkpoint_label(manifest.checkpoint_path),
        "topology_id": _missing(manifest.topology_id),
        "topology_variant": _missing(manifest.topology_variant),
        "topology_contract_version": _missing(manifest.topology_contract_version),
        "preprocessing_contract_name": _missing(manifest.preprocessing_contract_name),
        "input_mode": _missing(manifest.input_mode),
        "representation_kind": _missing(manifest.representation_kind),
        "orientation_source_mode": _missing(manifest.orientation_source_mode),
        "input_keys_discovered": _join_values(manifest.input_keys),
        "geometry_schema_status": "metadata missing",
        "output_keys_discovered": _join_values(manifest.model_output_keys),
        "distance_output_width": _missing(manifest.distance_output_width),
        "yaw_output_width": _missing(manifest.yaw_output_width),
        "compatibility_errors": "; ".join(errors),
        "compatibility_warnings": _format_issues(result.issues, WARNING),
        "notes": "Requested registry root; the current loader does not descend into runs/.",
        "issue_codes": _issue_codes(result.issues, ERROR),
    }


def _distance_artifact_row(artifact: ArtifactRoot) -> dict[str, Any]:
    manifest = load_live_model_manifest(artifact.root)
    result = check_live_model_compatibility(manifest)
    family = _likely_family(artifact.root, manifest)
    return {
        "artifact_root": _relpath(artifact.root),
        "artifact_kind": artifact.kind,
        "location": artifact.location,
        "compatible": "yes" if result.ok else "no",
        "likely_family": family,
        "checkpoint_selected": _checkpoint_label(manifest.checkpoint_path),
        "topology_id": _missing(manifest.topology_id),
        "topology_variant": _missing(manifest.topology_variant),
        "topology_contract_version": _missing(manifest.topology_contract_version),
        "preprocessing_contract_name": _missing(manifest.preprocessing_contract_name),
        "input_mode": _missing(manifest.input_mode),
        "representation_kind": _missing(manifest.representation_kind),
        "orientation_source_mode": _missing(manifest.orientation_source_mode),
        "input_keys_discovered": _join_values(manifest.input_keys),
        "geometry_schema_status": _geometry_status(manifest, family),
        "output_keys_discovered": _join_values(manifest.model_output_keys),
        "distance_output_width": _missing(manifest.distance_output_width),
        "yaw_output_width": _missing(manifest.yaw_output_width),
        "compatibility_errors": _distance_error_summary(result.issues, family),
        "compatibility_warnings": _format_issues(result.issues, WARNING),
        "notes": _distance_notes(artifact.root, family, result.ok),
        "issue_codes": _issue_codes(result.issues, ERROR),
    }


def _roi_artifact_row(artifact: ArtifactRoot) -> dict[str, Any]:
    run_config_path = artifact.root / "run_config.json"
    dataset_contract_path = artifact.root / "dataset_contract.json"
    summary_path = artifact.root / "summary.json"
    run_config = _read_json(run_config_path)
    dataset_contract = _read_json(dataset_contract_path)
    summary = _read_json(summary_path)
    errors = []
    if run_config_path.exists() and not isinstance(run_config, Mapping):
        errors.append("run_config.json is not a JSON object")
    if dataset_contract_path.exists() and not isinstance(dataset_contract, Mapping):
        errors.append("dataset_contract.json is not a JSON object")

    crop_size = _roi_crop_size(run_config, dataset_contract)
    canvas_size = _roi_canvas_size(run_config, dataset_contract)
    contract_version = _roi_contract_version(run_config, dataset_contract)
    notes = _roi_notes(run_config, summary, crop_size, canvas_size)
    if not run_config_path.is_file():
        errors.append("run_config.json missing")
    if not dataset_contract_path.is_file():
        errors.append("dataset_contract.json missing")
    if crop_size is None:
        errors.append("ROI crop size metadata missing")
    if canvas_size is None:
        errors.append("locator canvas size metadata missing")

    return {
        "artifact_root": _relpath(artifact.root),
        "artifact_kind": artifact.kind,
        "location": artifact.location,
        "checkpoint_selected": _checkpoint_name(artifact.root),
        "run_config_found": "yes" if run_config_path.is_file() else "no",
        "dataset_contract_found": "yes" if dataset_contract_path.is_file() else "no",
        "locator_canvas_size": _size_label(canvas_size),
        "roi_crop_size": _size_label(crop_size),
        "artifact_contract_version": contract_version,
        "notes_errors": "; ".join(errors) if errors else notes,
        "crop_size": crop_size,
        "canvas_size": canvas_size,
        "started_at_utc": _string_at(run_config, ("started_at_utc",)),
        "validation_mean_center_error_px": _value_at(
            summary,
            ("validation_metrics", "mean_center_error_px"),
        ),
    }


def _build_pairing_rows(
    compatible_distance_rows: Sequence[Mapping[str, Any]],
    roi_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for distance_row in compatible_distance_rows:
        distance_root = REPO_ROOT / str(distance_row["artifact_root"])
        pairable: list[str] = []
        warnings: list[str] = []
        for roi_row in roi_rows:
            roi_root = REPO_ROOT / str(roi_row["artifact_root"])
            manifest = load_live_model_manifest(distance_root, roi_locator_root=roi_root)
            result = check_live_model_compatibility(manifest)
            roi_label = str(roi_row["artifact_root"])
            if result.ok and manifest.roi_locator_crop_size and manifest.roi_locator_canvas_size:
                pairable.append(roi_label)
            else:
                issue_text = _format_issues(result.issues, ERROR)
                warnings.append(
                    f"{roi_label}: {issue_text or 'metadata insufficient to prove pairability'}"
                )
        rows.append(
            {
                "distance_orientation_artifact": distance_row["artifact_root"],
                "pairable_roi_fcn_artifacts": pairable,
                "metadata_warnings": warnings,
                "notes": _pairing_note(pairable, warnings),
            }
        )
    return rows


def _build_recommendation(
    compatible_distance_rows: Sequence[Mapping[str, Any]],
    roi_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    distance_choice = _most_recent_named(
        compatible_distance_rows,
        preferred_name="260504-1100_ts-2d-cnn",
    )
    roi_choice = _most_recent_roi(roi_rows)
    live_tree_note = (
        "Live-tree artifact copies were included when available."
        if _is_live_tree_choice(distance_choice) or _is_live_tree_choice(roi_choice)
        else "No live-tree artifact copy was found in this scan."
    )
    return {
        "distance_orientation": distance_choice
        or "none: no compatible live tri-stream distance/orientation artifact found",
        "roi_fcn": roi_choice or "none: no ROI-FCN artifact with sufficient metadata found",
        "notes": f"Recommended roots are loader-readable run artifacts. {live_tree_note}",
    }


def _build_blockers(
    *,
    distance_artifacts: Sequence[ArtifactRoot],
    roi_artifacts: Sequence[ArtifactRoot],
    distance_registry_roots: Sequence[ArtifactRoot],
) -> list[str]:
    blockers: list[str] = []
    if not any(artifact.location == "live-tree" for artifact in distance_artifacts):
        blockers.append("No live-tree distance/orientation artifacts were found")
    if not any(artifact.location == "live-tree" for artifact in roi_artifacts):
        blockers.append("No live-tree ROI-FCN artifacts were found")
    if distance_registry_roots:
        blockers.append(
            "Parent registry roots contain run_register.json but are not directly "
            "loader-readable model bundles; use/copy a run artifact root or add "
            "manifest metadata at the selected root."
        )
    return blockers


def _render_markdown(payload: Mapping[str, Any]) -> str:
    summary = payload["summary"]
    recommendation = payload["recommendation"]
    lines = [
        "# Live Model Compatibility Matrix",
        "",
        f"Generated: {payload['generated']}",
        "",
        "## Summary",
        "",
        "- Metadata-only scan; no checkpoints were loaded and no model runtimes were imported.",
        "- Called `load_live_model_manifest()` and `check_live_model_compatibility()` from `06_live-inference_v0.1/src/live_inference/model_registry/` for distance/orientation compatibility.",
        "- Distance/orientation parent directories with `run_register.json` are included as registry-root loader checks; the actual loader-readable artifacts are their `runs/run_*` directories.",
        "- ROI-FCN is treated as an independently selected locator/preprocessing dependency, not as an input head for the distance/orientation regressor.",
        f"- Distance/orientation artifacts scanned: {summary['distance_orientation_artifacts_scanned']} registered run artifact(s); {summary['distance_orientation_registry_roots_inspected']} parent registry root(s) inspected.",
        f"- ROI-FCN artifacts scanned: {summary['roi_fcn_artifacts_scanned']}.",
        f"- Compatible live tri-stream distance/orientation models: {summary['compatible_live_tri_stream_distance_orientation_models']}.",
        f"- Likely current deployment candidate: `{summary['likely_current_deployment_candidate']}`.",
        f"- ROI-FCN candidate: `{summary['roi_fcn_candidate']}`.",
        "- Blockers: "
        + ("; ".join(summary["blockers"]) if summary["blockers"] else "none"),
        "",
        "Expected live distance/orientation contract:",
        "",
        f"- topology contract version: `{contracts.MODEL_TOPOLOGY_CONTRACT_VERSION}`",
        f"- preprocessing contract name: `{contracts.PREPROCESSING_CONTRACT_NAME}`",
        f"- input mode: `{contracts.TRI_STREAM_INPUT_MODE}`",
        f"- representation kind: `{contracts.TRI_STREAM_REPRESENTATION_KIND}`",
        f"- required input keys: `{_join_values(contracts.TRI_STREAM_INPUT_KEYS)}`",
        f"- required output keys: `{contracts.MODEL_OUTPUT_DISTANCE_KEY}`, `{contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY}`",
        "",
        "## Distance/Orientation Model Matrix",
        "",
        _markdown_table(
            payload["distance_orientation"],
            (
                ("artifact_root", "artifact root"),
                ("location", "location"),
                ("compatible", "compatible"),
                ("likely_family", "likely family"),
                ("checkpoint_selected", "checkpoint selected"),
                ("topology_id", "topology id"),
                ("topology_variant", "topology variant"),
                ("topology_contract_version", "topology contract version"),
                ("preprocessing_contract_name", "preprocessing contract name"),
                ("input_mode", "input mode"),
                ("representation_kind", "representation kind"),
                ("orientation_source_mode", "orientation source mode"),
                ("input_keys_discovered", "input keys discovered"),
                ("geometry_schema_status", "geometry schema status"),
                ("output_keys_discovered", "output keys discovered"),
                ("distance_output_width", "distance output width"),
                ("yaw_output_width", "yaw output width"),
                ("compatibility_errors", "compatibility errors"),
                ("compatibility_warnings", "compatibility warnings"),
                ("notes", "notes"),
            ),
        ),
        "",
        "## ROI-FCN Artifact Inventory",
        "",
        _markdown_table(
            payload["roi_fcn"],
            (
                ("artifact_root", "artifact root"),
                ("location", "location"),
                ("checkpoint_selected", "checkpoint selected"),
                ("run_config_found", "run config found"),
                ("dataset_contract_found", "dataset contract found"),
                ("locator_canvas_size", "locator canvas size if discoverable"),
                ("roi_crop_size", "ROI crop size if discoverable"),
                ("artifact_contract_version", "artifact contract/version if discoverable"),
                ("notes_errors", "notes/errors"),
            ),
        ),
        "",
        "## Pairing Notes",
        "",
        "The distance/orientation model consumes `x_distance_image`, `x_orientation_image`, and `x_geometry`. It does not consume ROI-FCN heatmaps or logits directly. Pairing below only checks metadata-discoverable locator crop/canvas compatibility.",
        "",
        _markdown_table(
            payload["pairing_notes"],
            (
                ("distance_orientation_artifact", "compatible distance/orientation model"),
                ("pairable_roi_fcn_artifacts", "ROI-FCN artifacts that appear pairable"),
                ("metadata_warnings", "metadata warnings"),
                ("notes", "notes"),
            ),
        ),
        "",
        "## Recommended Current Selection",
        "",
        f"- `distance_orientation`: `{recommendation['distance_orientation']}`",
        f"- `roi_fcn`: `{recommendation['roi_fcn']}`",
        f"- Notes: {recommendation['notes']}",
        "",
        "Do not treat this as runtime validation. It proves only that the available metadata matches the live tri-stream and locator pairing checks exposed by the current lightweight loader/checker.",
        "",
        "## Machine-Readable Output",
        "",
        f"- JSON: `{_relpath(JSON_REPORT)}`",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
) -> str:
    if not rows:
        headers = [header for _, header in columns]
        return (
            "| " + " | ".join(headers) + " |\n"
            + "| " + " | ".join("---" for _ in headers) + " |\n"
            + "| " + " | ".join("none found" if index == 0 else "" for index, _ in enumerate(headers)) + " |"
        )
    headers = [header for _, header in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row.get(key, "")) for key, _ in columns)
            + " |"
        )
    return "\n".join(lines)


def _markdown_cell(value: Any) -> str:
    if value is None:
        text = "none"
    elif isinstance(value, (list, tuple)):
        text = _join_values(value)
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    return text or "none"


def _has_any_file(root: Path, filenames: set[str]) -> bool:
    return any((root / filename).is_file() for filename in filenames)


def _has_checkpoint(root: Path) -> bool:
    return any((root / filename).is_file() for filename in CHECKPOINT_CANDIDATES)


def _registered_run_dirs(root: Path) -> list[str]:
    runs_root = root / "runs"
    if not runs_root.is_dir():
        return []
    return [_relpath(path) for path in sorted(runs_root.iterdir()) if path.is_dir()]


def _likely_family(root: Path, manifest: Any) -> str:
    text = " ".join(
        str(value or "").lower()
        for value in (
            root.name,
            root.parent.name,
            manifest.topology_id,
            manifest.topology_variant,
            manifest.input_mode,
            manifest.preprocessing_contract_name,
            manifest.representation_kind,
        )
    )
    if "dual_stream" in text or "_ds-" in text:
        return "ds"
    if "tri_stream" in text or "_ts-" in text:
        return "ts"
    return "unknown"


def _geometry_status(manifest: Any, family: str) -> str:
    if (
        manifest.geometry_schema == contracts.TRI_STREAM_GEOMETRY_SCHEMA
        and manifest.geometry_dim == len(contracts.TRI_STREAM_GEOMETRY_SCHEMA)
    ):
        return f"ok: {manifest.geometry_dim}-field tri-stream geometry schema"
    if not manifest.geometry_schema:
        if family == "ds":
            return "legacy dual-stream / no x_geometry schema"
        return "metadata missing"
    return (
        f"mismatch: discovered {len(manifest.geometry_schema)} field(s), "
        f"expected {len(contracts.TRI_STREAM_GEOMETRY_SCHEMA)}"
    )


def _distance_error_summary(issues: Sequence[Any], family: str) -> str:
    errors = [issue for issue in issues if issue.severity == ERROR]
    if not errors:
        return "none"
    if family == "ds":
        missing_inputs = [
            str(issue.expected)
            for issue in errors
            if issue.code == "missing_input_key" and issue.expected
        ]
        details = [
            "input mode mismatch",
            "preprocessing contract mismatch",
            "representation kind mismatch",
            "missing required tri-stream input keys "
            + (_join_values(missing_inputs) if missing_inputs else "unknown"),
            "missing tri-stream geometry schema/dimension metadata",
        ]
        return (
            "legacy dual-stream / wrong input contract: "
            + "; ".join(details)
        )
    return _format_issues(errors, ERROR)


def _distance_notes(root: Path, family: str, ok: bool) -> str:
    if ok:
        note = "passes current live tri-stream compatibility checker"
    elif family == "ds":
        note = (
            "legacy dual-stream artifact; consumes silhouette/bbox-style inputs, "
            "not x_distance_image + x_orientation_image + x_geometry"
        )
    else:
        note = "does not pass current live tri-stream compatibility checker"
    if "260504-1100_ts-2d-cnn" in str(root):
        note += "; most recent compatible distance/orientation artifact found"
    return note


def _format_issues(issues: Sequence[Any], severity: str) -> str:
    selected = [issue for issue in issues if issue.severity == severity]
    if not selected:
        return "none"
    return "; ".join(f"{issue.code}: {issue.message}" for issue in selected)


def _issue_codes(issues: Sequence[Any], severity: str) -> list[str]:
    return [issue.code for issue in issues if issue.severity == severity]


def _checkpoint_label(path: Path | None) -> str:
    if path is None:
        return "not found"
    return path.name


def _checkpoint_name(root: Path) -> str:
    for candidate in CHECKPOINT_CANDIDATES:
        if (root / candidate).is_file():
            return candidate
    return "not found"


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _roi_crop_size(
    run_config: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
) -> tuple[int, int] | None:
    return _size_from_width_height(
        run_config,
        (
            ("roi_width_px", "roi_height_px"),
            ("roi_crop_width_px", "roi_crop_height_px"),
            ("crop_width_px", "crop_height_px"),
        ),
    ) or _size_from_width_height(
        _dataset_split(dataset_contract),
        (
            ("fixed_roi_width_px", "fixed_roi_height_px"),
            ("roi_width_px", "roi_height_px"),
        ),
    )


def _roi_canvas_size(
    run_config: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
) -> tuple[int, int] | None:
    output_hw = _mapping_at(run_config, ("output_hw",))
    split_geometry = _mapping_at(_dataset_split(dataset_contract), ("geometry",))
    return _size_from_width_height(
        split_geometry,
        (("canvas_width_px", "canvas_height_px"), ("width", "height")),
    ) or _size_from_width_height(
        run_config,
        (
            ("locator_canvas_width_px", "locator_canvas_height_px"),
            ("canvas_width_px", "canvas_height_px"),
        ),
    ) or _size_from_width_height(output_hw, (("width", "height"),))


def _roi_contract_version(
    run_config: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
) -> str:
    values = [
        _string_at(run_config, ("topology_contract", "contract_version")),
        _string_at(run_config, ("training_contract_version",)),
        _string_at(dataset_contract, ("training_contract_version",)),
        _string_at(
            _dataset_split(dataset_contract),
            ("preprocessing_contract_version",),
        ),
    ]
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return _join_values(deduped)


def _roi_notes(
    run_config: Mapping[str, Any],
    summary: Mapping[str, Any],
    crop_size: tuple[int, int] | None,
    canvas_size: tuple[int, int] | None,
) -> str:
    notes: list[str] = []
    if crop_size and canvas_size:
        notes.append(
            f"metadata declares ROI crop {_size_label(crop_size)} on locator canvas {_size_label(canvas_size)}"
        )
    if _string_at(run_config, ("resume", "source_run_id")):
        notes.append(f"resumed from {_string_at(run_config, ('resume', 'source_run_id'))}")
    validation_error = _value_at(summary, ("validation_metrics", "mean_center_error_px"))
    if validation_error is not None:
        notes.append(f"validation mean center error {validation_error:.3f}px")
    elif not summary:
        notes.append("summary.json not found")
    return "; ".join(notes) if notes else "metadata present"


def _dataset_split(dataset_contract: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("validation_split", "train_split"):
        split = _mapping_at(dataset_contract, (key,))
        if split:
            return split
    return {}


def _pairing_note(pairable: Sequence[str], warnings: Sequence[str]) -> str:
    if pairable and not warnings:
        return (
            "metadata pairable: ROI crop size matches the distance canvas and "
            "fits inside the locator canvas; runtime behavior not checked"
        )
    if pairable:
        return "some ROI-FCN artifacts are metadata-pairable; see warnings for the rest"
    return "metadata insufficient or incompatible for all scanned ROI-FCN artifacts"


def _most_recent_named(
    rows: Sequence[Mapping[str, Any]],
    *,
    preferred_name: str,
) -> str | None:
    preferred_rows = [
        row for row in rows if preferred_name in str(row["artifact_root"])
    ]
    for row in preferred_rows:
        if row.get("location") == "live-tree":
            return str(row["artifact_root"])
    if preferred_rows:
        return str(preferred_rows[-1]["artifact_root"])
    for row in rows:
        if row.get("location") == "live-tree":
            return str(row["artifact_root"])
    return str(rows[-1]["artifact_root"]) if rows else None


def _most_recent_roi(rows: Sequence[Mapping[str, Any]]) -> str | None:
    sufficient = [
        row
        for row in rows
        if row.get("checkpoint_selected") != "not found"
        and row.get("run_config_found") == "yes"
        and row.get("dataset_contract_found") == "yes"
        and row.get("crop_size") is not None
        and row.get("canvas_size") is not None
    ]
    if not sufficient:
        return None
    sufficient.sort(
        key=lambda row: (
            str(row.get("started_at_utc") or ""),
            str(row.get("artifact_root") or ""),
        )
    )
    return str(sufficient[-1]["artifact_root"])


def _is_live_tree_choice(choice: str | None) -> bool:
    return bool(choice and choice.startswith("06_live-inference_v0.1/"))


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


def _string_at(source: Mapping[str, Any], path: Sequence[str]) -> str | None:
    value = _value_at(source, path)
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


def _size_label(size: tuple[int, int] | None) -> str:
    if size is None:
        return "metadata missing"
    return f"{size[0]}x{size[1]}"


def _missing(value: Any) -> str:
    if value is None or value == ():
        return "metadata missing"
    return str(value)


def _join_values(values: Any) -> str:
    if values is None:
        return "none"
    if isinstance(values, str):
        return values or "none"
    items = [str(value) for value in values if str(value)]
    return ", ".join(items) if items else "none"


def _join_paths(paths: Sequence[str]) -> str:
    return ", ".join(paths) if paths else "none"


def _relpath(path: Path) -> str:
    path = path.resolve(strict=False)
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
