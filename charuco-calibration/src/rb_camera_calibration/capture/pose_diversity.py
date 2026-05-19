"""Pose diversity heuristics for automatic ChArUco capture."""

from __future__ import annotations

from rb_camera_calibration.contracts import (
    CapturePolicyConfig,
    CharucoDetection,
    FrameQualityMetrics,
    PoseCoverageState,
    PoseSignature,
)


class SimplePoseDiversityTracker:
    """Track coarse coverage across location, scale, and tilt bins."""

    def __init__(self, policy: CapturePolicyConfig) -> None:
        self.policy = policy
        self._accepted_signatures: list[PoseSignature] = []

    @property
    def accepted_count(self) -> int:
        return len(self._accepted_signatures)

    def evaluate(
        self,
        detection: CharucoDetection,
        quality: FrameQualityMetrics,
    ) -> PoseSignature:
        """Convert a detection into a binned pose signature."""
        image_size = detection.extras.get("image_size_wh_px", (1, 1))
        width = max(float(image_size[0]), 1.0)
        height = max(float(image_size[1]), 1.0)
        if detection.board_center_xy_px is None:
            center_x, center_y = 0.5, 0.5
        else:
            center_x = _clamp(detection.board_center_xy_px[0] / width)
            center_y = _clamp(detection.board_center_xy_px[1] / height)
        grid_col = _bin_index(center_x, self.policy.pose_grid_cols, upper_inclusive=True)
        grid_row = _bin_index(center_y, self.policy.pose_grid_rows, upper_inclusive=True)
        area_fraction = _clamp(float(detection.board_area_fraction))
        scale_bin = _area_bin(area_fraction, self.policy.scale_bin_count)
        skew = _clamp(float(detection.extras.get("perspective_skew_score", 0.0)))
        tilt_bin = _bin_index(skew, self.policy.tilt_bin_count, upper_inclusive=True)
        roll = float(detection.extras.get("roll_like_angle_deg", 0.0))
        return PoseSignature(
            center_x_norm=center_x,
            center_y_norm=center_y,
            area_fraction=area_fraction,
            roll_like_angle_deg=roll,
            perspective_skew_score=skew,
            grid_cell=(grid_col, grid_row),
            scale_bin=scale_bin,
            tilt_bin=tilt_bin,
            extras={"quality_blur_score": quality.blur_score},
        )

    def update_accepted(self, signature: PoseSignature) -> PoseCoverageState:
        """Record an accepted pose and return updated coverage."""
        self._accepted_signatures.append(signature)
        return self.coverage_state()

    def coverage_state(self) -> PoseCoverageState:
        """Return the current coverage state."""
        cells = tuple(sorted({sig.grid_cell for sig in self._accepted_signatures}))
        scales = tuple(sorted({sig.scale_bin for sig in self._accepted_signatures}))
        tilts = tuple(sorted({sig.tilt_bin for sig in self._accepted_signatures}))
        max_cells = self.policy.pose_grid_cols * self.policy.pose_grid_rows
        cell_score = len(cells) / max(max_cells, 1)
        scale_score = len(scales) / max(self.policy.scale_bin_count, 1)
        tilt_score = len(tilts) / max(self.policy.tilt_bin_count, 1)
        coverage_score = round((cell_score + scale_score + tilt_score) / 3.0, 4)
        return PoseCoverageState(
            accepted_count=len(self._accepted_signatures),
            occupied_center_cells=cells,
            occupied_scale_bins=scales,
            occupied_tilt_bins=tilts,
            coverage_score=coverage_score,
            suggested_next_pose=self.suggest_next_pose(cells, scales, tilts),
        )

    def has_seen(self, signature: PoseSignature) -> bool:
        """Return whether the exact coarse pose bin has already been accepted."""
        return any(_pose_key(existing) == _pose_key(signature) for existing in self._accepted_signatures)

    def suggest_next_pose(
        self,
        cells: tuple[tuple[int, int], ...] | None = None,
        scales: tuple[int, ...] | None = None,
        tilts: tuple[int, ...] | None = None,
    ) -> str:
        """Suggest a simple next movement to improve coverage."""
        cells = cells if cells is not None else tuple(sorted({s.grid_cell for s in self._accepted_signatures}))
        scales = scales if scales is not None else tuple(sorted({s.scale_bin for s in self._accepted_signatures}))
        tilts = tilts if tilts is not None else tuple(sorted({s.tilt_bin for s in self._accepted_signatures}))

        for row in range(self.policy.pose_grid_rows):
            for col in range(self.policy.pose_grid_cols):
                if (col, row) not in cells:
                    return _cell_suggestion(col, row, self.policy.pose_grid_cols, self.policy.pose_grid_rows)
        for scale_bin in range(self.policy.scale_bin_count):
            if scale_bin not in scales:
                if scale_bin == 0:
                    return "try farther/smaller"
                if scale_bin == self.policy.scale_bin_count - 1:
                    return "try closer/larger"
                if scales and max(scales) < scale_bin:
                    return "try closer/larger"
                if scales and min(scales) > scale_bin:
                    return "try farther/smaller"
                return "add a mid-size board view"
        for tilt_bin in range(self.policy.tilt_bin_count):
            if tilt_bin not in tilts:
                if tilt_bin == 0:
                    return "add flatter front-on view"
                return "add stronger tilt"
        return "coverage target looks broad; add a clean sharp view"


def _pose_key(signature: PoseSignature) -> tuple[tuple[int, int], int, int]:
    return (signature.grid_cell, signature.scale_bin, signature.tilt_bin)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _bin_index(value: float, count: int, *, upper_inclusive: bool = False) -> int:
    if upper_inclusive and value >= 1.0:
        return count - 1
    return max(0, min(count - 1, int(value * count)))


def _area_bin(area_fraction: float, count: int) -> int:
    """Bin board scale with thresholds biased toward practical fixed-focus use.

    The AR0234 setup can only stay sharp once the board is far enough from the
    fixed-focus lens.  In practice a useful ChArUco board often occupies less
    than 10% of the full image area, so the bins deliberately split the
    low-area range instead of demanding an impractically close view.
    """
    if count <= 1:
        return 0
    if count == 2:
        return 0 if area_fraction < 0.07 else 1
    if area_fraction < 0.05:
        return 0
    if area_fraction < 0.085:
        return min(1, count - 1)
    return count - 1


def _cell_suggestion(col: int, row: int, cols: int, rows: int) -> str:
    horizontal = "left" if col == 0 else "right" if col == cols - 1 else "center"
    vertical = "upper" if row == 0 else "lower" if row == rows - 1 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "hold board near center"
    if horizontal == "center":
        return f"move board toward {vertical}"
    if vertical == "middle":
        return f"move board toward {horizontal}"
    return f"move board toward {vertical}-{horizontal}"
