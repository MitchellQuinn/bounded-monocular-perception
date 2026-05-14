"""Configuration dataclasses for ROI-FCN preprocessing v0.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os


_VALID_DETECTOR_BACKENDS = {"edge_roi_v1"}


def _default_worker_count() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


@dataclass(frozen=True)
class BootstrapCenterTargetConfig:
    """Config for stage 1 edge-bootstrapped crop-center generation."""

    detector_backend: str = "edge_roi_v1"
    edge_blur_k: int = 5
    edge_low: int = 50
    edge_high: int = 150
    fg_threshold: int = 250
    edge_pad: int = 0
    edge_ignore_border_px: int = 8
    min_edge_pixels: int = 16
    edge_close_kernel_size: int = 1

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True
    persist_debug_images: bool = False
    num_workers: int = field(default_factory=_default_worker_count)

    def normalized_detector_backend(self) -> str:
        backend = str(self.detector_backend).strip().lower()
        if backend not in _VALID_DETECTOR_BACKENDS:
            allowed = ", ".join(sorted(_VALID_DETECTOR_BACKENDS))
            raise ValueError(
                f"Unsupported detector_backend '{self.detector_backend}'. Allowed: {allowed}."
            )
        return backend

    def normalized_edge_blur_k(self) -> int:
        kernel = max(1, int(self.edge_blur_k))
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def normalized_edge_low(self) -> int:
        return max(0, min(255, int(self.edge_low)))

    def normalized_edge_high(self) -> int:
        return max(0, min(255, int(self.edge_high)))

    def normalized_fg_threshold(self) -> int:
        return max(0, min(255, int(self.fg_threshold)))

    def normalized_edge_pad(self) -> int:
        return max(0, int(self.edge_pad))

    def normalized_edge_ignore_border_px(self) -> int:
        return max(0, int(self.edge_ignore_border_px))

    def normalized_min_edge_pixels(self) -> int:
        return max(1, int(self.min_edge_pixels))

    def normalized_edge_close_kernel_size(self) -> int:
        return max(1, int(self.edge_close_kernel_size))

    def normalized_num_workers(self) -> int:
        return max(1, int(self.num_workers))

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["detector_backend"] = self.normalized_detector_backend()
        payload["edge_blur_k"] = self.normalized_edge_blur_k()
        payload["edge_low"] = self.normalized_edge_low()
        payload["edge_high"] = self.normalized_edge_high()
        payload["fg_threshold"] = self.normalized_fg_threshold()
        payload["edge_pad"] = self.normalized_edge_pad()
        payload["edge_ignore_border_px"] = self.normalized_edge_ignore_border_px()
        payload["min_edge_pixels"] = self.normalized_min_edge_pixels()
        payload["edge_close_kernel_size"] = self.normalized_edge_close_kernel_size()
        payload["num_workers"] = self.normalized_num_workers()
        return payload


@dataclass(frozen=True)
class PackRoiFcnConfig:
    """Config for stage 2 full-frame locator-canvas packing."""

    canvas_width: int = 480
    canvas_height: int = 300
    shard_size: int = 8192
    compress: bool = False

    overwrite: bool = False
    dry_run: bool = False
    continue_on_error: bool = True
    num_workers: int = field(default_factory=_default_worker_count)

    def normalized_canvas_width(self) -> int:
        return max(1, int(self.canvas_width))

    def normalized_canvas_height(self) -> int:
        return max(1, int(self.canvas_height))

    def normalized_shard_size(self) -> int:
        size = int(self.shard_size)
        if size < 0:
            raise ValueError("shard_size must be >= 0")
        return size

    def normalized_num_workers(self) -> int:
        return max(1, int(self.num_workers))

    def to_log_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["canvas_width"] = self.normalized_canvas_width()
        payload["canvas_height"] = self.normalized_canvas_height()
        payload["shard_size"] = self.normalized_shard_size()
        payload["num_workers"] = self.normalized_num_workers()
        return payload
