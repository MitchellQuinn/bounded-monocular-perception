"""Raccoon Ball synthetic dataset post-processing pipeline."""

from .config import (
    BBoxStageConfig,
    CommonStageOptions,
    EdgeStageConfig,
    NpyStageConfig,
    PackStageConfig,
    ShuffleStageConfig,
    StageSummary,
)
from .paths import find_project_root, list_input_runs, list_training_runs

__all__ = [
    "CommonStageOptions",
    "EdgeStageConfig",
    "BBoxStageConfig",
    "NpyStageConfig",
    "PackStageConfig",
    "ShuffleStageConfig",
    "StageSummary",
    "find_project_root",
    "list_input_runs",
    "list_training_runs",
]

try:
    from .edge_stage import run_edge_stage

    __all__.append("run_edge_stage")
except (ModuleNotFoundError, ImportError):
    pass

try:
    from .bbox_stage import run_bbox_stage

    __all__.append("run_bbox_stage")
except (ModuleNotFoundError, ImportError):
    pass

try:
    from .npy_pack_stage import run_npy_pack_stage

    __all__.append("run_npy_pack_stage")
except (ModuleNotFoundError, ImportError):
    pass

try:
    from .npy_stage import run_npy_stage

    __all__.append("run_npy_stage")
except (ModuleNotFoundError, ImportError):
    pass

try:
    from .pack_stage import run_pack_stage

    __all__.append("run_pack_stage")
except (ModuleNotFoundError, ImportError):
    pass

try:
    from .shuffle_stage import run_shuffle_stage

    __all__.append("run_shuffle_stage")
except (ModuleNotFoundError, ImportError):
    pass

try:
    from .pipeline import STAGE_ORDER, run_stage_for_run, run_stage_sequence_for_run

    __all__.extend(["STAGE_ORDER", "run_stage_for_run", "run_stage_sequence_for_run"])
except (ModuleNotFoundError, ImportError):
    pass

try:
    from .widgets import (
        BBoxStagePanel,
        EdgeStagePanel,
        NpyPackPanel,
        PipelineLauncher,
        PreviewPanel,
        ShuffleCorpusPanel,
    )

    __all__.extend(
        [
            "PreviewPanel",
            "PipelineLauncher",
            "EdgeStagePanel",
            "BBoxStagePanel",
            "NpyPackPanel",
            "ShuffleCorpusPanel",
        ]
    )
except (ModuleNotFoundError, ImportError):
    pass
