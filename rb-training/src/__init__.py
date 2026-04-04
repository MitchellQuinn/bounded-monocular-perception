"""Project Raccoon Ball training package."""

from .config import EvalConfig, TrainConfig


def evaluate_saved_run(*args, **kwargs):
    """Lazy proxy to avoid importing evaluate.py on package import."""
    from .evaluate import evaluate_saved_run as _impl

    return _impl(*args, **kwargs)


def train_distance_regressor(*args, **kwargs):
    """Lazy proxy to avoid importing train.py on package import."""
    from .train import train_distance_regressor as _impl

    return _impl(*args, **kwargs)

__all__ = [
    "EvalConfig",
    "TrainConfig",
    "evaluate_saved_run",
    "train_distance_regressor",
]
