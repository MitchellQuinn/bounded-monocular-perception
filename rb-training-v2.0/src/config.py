"""Configuration dataclasses for training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    """Configuration for distance-regression training across topology families."""

    training_data_root: str | Path = "training-data"
    validation_data_root: str | Path = "validation-data"
    output_root: str | Path = "models"
    seed: int = 42
    batch_size: int = 4
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    huber_delta: float = 1.0
    early_stopping_patience: int = 4
    model_name: str = "2d-cnn"
    run_id: str | None = None
    run_name_suffix: str | None = None
    enable_internal_test_split: bool = False
    internal_test_fraction: float = 0.1
    padding_mode: str = "disabled"
    topology_id: str = "distance_regressor_2d_cnn"
    topology_variant: str | None = None
    topology_params: dict[str, Any] = field(default_factory=dict)
    model_architecture_variant: str | None = None
    progress_log_interval_batches: int = 250
    accuracy_tolerance_m: float = 0.10
    extra_accuracy_tolerances_m: tuple[float, ...] = (0.25, 0.50)
    enable_lr_scheduler: bool = False
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 1
    lr_scheduler_min_lr: float = 1e-5
    train_cache_budget_gb: float = 48.0
    train_shuffle_mode: str = "shard"
    train_active_shard_count: int = 3
    cache_validation_in_ram: bool = True
    validation_cache_budget_gb: float = 40.0
    resume_from_run_dir: str | Path | None = None
    additional_epochs: int | None = None
    change_note: str = "Distance-regression falsification run."
    entrypoint_type: str = "cli"
    entrypoint_path: str = "src/train.py"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainConfig":
        """Create config from a dict while retaining defaults."""
        return cls(**{**asdict(cls()), **values})  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration as a plain dictionary."""
        return asdict(self)


@dataclass
class EvalConfig:
    """Configuration for evaluating a saved run."""

    model_run_directory: str | Path
    validation_data_root: str | Path = "validation-data"
    training_data_root: str | Path | None = None
    batch_size: int = 4
    padding_mode_override: str | None = None
    evaluate_internal_test_if_present: bool = False
    entrypoint_type: str = "cli"
    entrypoint_path: str = "src/evaluate.py"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "EvalConfig":
        """Create config from dict values."""
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration as a plain dictionary."""
        return asdict(self)
