"""Configuration dataclasses for ROI-FCN training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """Configuration for one ROI-FCN training run."""

    training_dataset: str = ""
    validation_dataset: str | None = None
    datasets_root: str | Path = "datasets"
    models_root: str | Path = "models"
    seed: int = 42
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gaussian_sigma_px: float = 2.5
    early_stopping_patience: int = 4
    topology_id: str = "roi_fcn_tiny"
    topology_variant: str = "tiny_v1"
    topology_params: dict[str, Any] = field(default_factory=dict)
    model_name: str = "roi-fcn-tiny"
    run_id: str | None = None
    run_name_suffix: str | None = None
    device: str | None = None
    progress_log_interval_steps: int = 50
    roi_width_px: int = 300
    roi_height_px: int = 300
    evaluation_max_visual_examples: int = 12
    entrypoint_type: str = "python"
    entrypoint_path: str = "src/roi_fcn_training_v0_1/train.py"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainConfig":
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalConfig:
    """Configuration for evaluating a saved ROI-FCN run."""

    model_run_directory: str | Path
    datasets_root: str | Path = "datasets"
    training_dataset: str | None = None
    validation_dataset: str | None = None
    batch_size: int = 16
    roi_width_px: int = 300
    roi_height_px: int = 300
    device: str | None = None
    evaluation_max_visual_examples: int = 12
    entrypoint_type: str = "python"
    entrypoint_path: str = "src/roi_fcn_training_v0_1/evaluate.py"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "EvalConfig":
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
