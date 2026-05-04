"""Compatibility checks for normalized live model manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import interfaces.contracts as contracts

from .model_manifest import LiveModelManifest


ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class CompatibilityIssue:
    """One model compatibility diagnostic."""

    severity: str
    code: str
    message: str
    field: str | None = None
    expected: Any = None
    actual: Any = None


@dataclass(frozen=True)
class CompatibilityResult:
    """Structured compatibility outcome for a live model manifest."""

    ok: bool
    issues: tuple[CompatibilityIssue, ...]


class ModelCompatibilityError(ValueError):
    """Raised when a live model manifest cannot be used safely."""


def check_live_model_compatibility(
    manifest: LiveModelManifest,
) -> CompatibilityResult:
    """Return all compatibility issues found in a normalized manifest."""
    issues: list[CompatibilityIssue] = []

    _require_equal(
        issues,
        field="topology_contract_version",
        expected=contracts.MODEL_TOPOLOGY_CONTRACT_VERSION,
        actual=manifest.topology_contract_version,
    )
    _require_equal(
        issues,
        field="input_mode",
        expected=contracts.TRI_STREAM_INPUT_MODE,
        actual=manifest.input_mode,
    )
    _require_equal(
        issues,
        field="preprocessing_contract_name",
        expected=contracts.PREPROCESSING_CONTRACT_NAME,
        actual=manifest.preprocessing_contract_name,
    )
    _require_equal(
        issues,
        field="representation_kind",
        expected=contracts.TRI_STREAM_REPRESENTATION_KIND,
        actual=manifest.representation_kind,
    )

    _require_contains_all(
        issues,
        field="input_keys",
        values=manifest.input_keys,
        expected=contracts.TRI_STREAM_INPUT_KEYS,
        missing_code="missing_input_key",
    )
    _check_optional_equal(
        issues,
        field="distance_image_key",
        expected=contracts.TRI_STREAM_DISTANCE_IMAGE_KEY,
        actual=manifest.distance_image_key,
    )
    _check_optional_equal(
        issues,
        field="orientation_image_key",
        expected=contracts.TRI_STREAM_ORIENTATION_IMAGE_KEY,
        actual=manifest.orientation_image_key,
    )
    _check_optional_equal(
        issues,
        field="geometry_key",
        expected=contracts.TRI_STREAM_GEOMETRY_KEY,
        actual=manifest.geometry_key,
    )

    _require_equal(
        issues,
        field="geometry_schema",
        expected=contracts.TRI_STREAM_GEOMETRY_SCHEMA,
        actual=manifest.geometry_schema or None,
    )
    _require_equal(
        issues,
        field="geometry_dim",
        expected=len(contracts.TRI_STREAM_GEOMETRY_SCHEMA),
        actual=manifest.geometry_dim,
    )

    _require_contains_all(
        issues,
        field="model_output_keys",
        values=manifest.model_output_keys,
        expected=(
            contracts.MODEL_OUTPUT_DISTANCE_KEY,
            contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY,
        ),
        missing_code="missing_model_output_key",
    )
    _check_optional_equal(
        issues,
        field="distance_output_key",
        expected=contracts.MODEL_OUTPUT_DISTANCE_KEY,
        actual=manifest.distance_output_key,
    )
    _check_optional_equal(
        issues,
        field="yaw_output_key",
        expected=contracts.MODEL_OUTPUT_YAW_SIN_COS_KEY,
        actual=manifest.yaw_output_key,
    )
    _check_optional_equal(
        issues,
        field="distance_output_width",
        expected=1,
        actual=manifest.distance_output_width,
    )
    _check_optional_equal(
        issues,
        field="yaw_output_width",
        expected=2,
        actual=manifest.yaw_output_width,
    )

    _check_checkpoint(issues, manifest)
    _check_positive_size(issues, field="distance_canvas_size", size=manifest.distance_canvas_size)
    _check_positive_size(
        issues,
        field="orientation_canvas_size",
        size=manifest.orientation_canvas_size,
    )
    _check_roi_compatibility(issues, manifest)

    issue_tuple = tuple(issues)
    return CompatibilityResult(
        ok=not any(issue.severity == ERROR for issue in issue_tuple),
        issues=issue_tuple,
    )


def require_live_model_compatibility(manifest: LiveModelManifest) -> None:
    """Raise if the manifest has compatibility errors."""
    result = check_live_model_compatibility(manifest)
    errors = tuple(issue for issue in result.issues if issue.severity == ERROR)
    if not errors:
        return
    lines = ["Live model compatibility check failed:"]
    lines.extend(f"- {issue.code}: {issue.message}" for issue in errors)
    raise ModelCompatibilityError("\n".join(lines))


def _add_issue(
    issues: list[CompatibilityIssue],
    *,
    code: str,
    message: str,
    field: str | None = None,
    expected: Any = None,
    actual: Any = None,
    severity: str = ERROR,
) -> None:
    issues.append(
        CompatibilityIssue(
            severity=severity,
            code=code,
            message=message,
            field=field,
            expected=expected,
            actual=actual,
        )
    )


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def _require_equal(
    issues: list[CompatibilityIssue],
    *,
    field: str,
    expected: Any,
    actual: Any,
) -> None:
    if _is_missing(actual):
        _add_issue(
            issues,
            code=f"missing_{field}",
            field=field,
            expected=expected,
            actual=actual,
            message=f"Missing required manifest field {field}.",
        )
        return
    if actual != expected:
        _add_issue(
            issues,
            code=f"{field}_mismatch",
            field=field,
            expected=expected,
            actual=actual,
            message=f"{field} must be {expected!r}; got {actual!r}.",
        )


def _check_optional_equal(
    issues: list[CompatibilityIssue],
    *,
    field: str,
    expected: Any,
    actual: Any,
) -> None:
    if _is_missing(actual):
        return
    if actual != expected:
        _add_issue(
            issues,
            code=f"{field}_mismatch",
            field=field,
            expected=expected,
            actual=actual,
            message=f"{field} must be {expected!r}; got {actual!r}.",
        )


def _require_contains_all(
    issues: list[CompatibilityIssue],
    *,
    field: str,
    values: tuple[str, ...],
    expected: tuple[str, ...],
    missing_code: str,
) -> None:
    if not values:
        _add_issue(
            issues,
            code=f"missing_{field}",
            field=field,
            expected=expected,
            actual=values,
            message=f"Missing required manifest field {field}.",
        )
        return
    present = set(values)
    for expected_value in expected:
        if expected_value not in present:
            _add_issue(
                issues,
                code=missing_code,
                field=field,
                expected=expected_value,
                actual=values,
                message=f"{field} is missing required key {expected_value!r}.",
            )


def _check_checkpoint(
    issues: list[CompatibilityIssue],
    manifest: LiveModelManifest,
) -> None:
    if manifest.checkpoint_path is None:
        _add_issue(
            issues,
            code="missing_checkpoint",
            field="checkpoint_path",
            message="No checkpoint candidate was found.",
        )
        return
    if not manifest.checkpoint_path.is_file():
        _add_issue(
            issues,
            code="checkpoint_not_found",
            field="checkpoint_path",
            actual=manifest.checkpoint_path,
            message=f"Checkpoint path does not exist: {manifest.checkpoint_path}.",
        )


def _check_positive_size(
    issues: list[CompatibilityIssue],
    *,
    field: str,
    size: tuple[int, int] | None,
) -> None:
    if size is None:
        return
    width, height = size
    if width <= 0 or height <= 0:
        _add_issue(
            issues,
            code=f"invalid_{field}",
            field=field,
            expected="positive width and height",
            actual=size,
            message=f"{field} must have positive width and height; got {size!r}.",
        )


def _check_roi_compatibility(
    issues: list[CompatibilityIssue],
    manifest: LiveModelManifest,
) -> None:
    _check_positive_size(
        issues,
        field="roi_locator_crop_size",
        size=manifest.roi_locator_crop_size,
    )
    _check_positive_size(
        issues,
        field="roi_locator_canvas_size",
        size=manifest.roi_locator_canvas_size,
    )

    if manifest.roi_locator_crop_size and manifest.distance_canvas_size:
        if manifest.roi_locator_crop_size != manifest.distance_canvas_size:
            _add_issue(
                issues,
                code="roi_locator_crop_size_mismatch",
                field="roi_locator_crop_size",
                expected=manifest.distance_canvas_size,
                actual=manifest.roi_locator_crop_size,
                message=(
                    "ROI locator crop size must match the distance preprocessing "
                    f"canvas size {manifest.distance_canvas_size!r}; "
                    f"got {manifest.roi_locator_crop_size!r}."
                ),
            )

    if manifest.roi_locator_crop_size and manifest.roi_locator_canvas_size:
        crop_w, crop_h = manifest.roi_locator_crop_size
        canvas_w, canvas_h = manifest.roi_locator_canvas_size
        if crop_w > canvas_w or crop_h > canvas_h:
            _add_issue(
                issues,
                code="roi_locator_crop_exceeds_canvas",
                field="roi_locator_crop_size",
                expected=f"<= {manifest.roi_locator_canvas_size!r}",
                actual=manifest.roi_locator_crop_size,
                message=(
                    "ROI locator crop size must fit within the ROI locator canvas; "
                    f"crop={manifest.roi_locator_crop_size!r}, "
                    f"canvas={manifest.roi_locator_canvas_size!r}."
                ),
            )
