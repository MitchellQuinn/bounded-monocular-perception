# Synthetic Data Processing v4.0

Minimal dual-stream preprocessing pipeline for Raccoon Ball.

## Stage Flow

1. `detect` - YOLO defender detection metadata.
2. `silhouette` - ROI silhouette generation with full-frame compatibility image.
3. `pack_dual_stream` - writes NPZ shards with:
   - `silhouette_crop` `(N, C, H, W)` as inverted ROI grayscale detail (dark vehicle detail on white background)
   - `bbox_features` `(N, 10)`
   - `y_position_3d` `(N, 3)`
   - `y_distance_m` `(N,)`
   - `y_yaw_deg` `(N,)`
   - `y_yaw_sin` `(N,)`
   - `y_yaw_cos` `(N,)`

`pack_dual_stream` can optionally apply deterministic foreground-only brightness normalization before writing
`silhouette_crop`. The first supported method is `masked_median_darkness_gain`: it computes foreground darkness
from the silhouette mask, scales only vehicle pixels toward `TargetMedianDarkness`, clamps the gain, and leaves
the white background and all geometry unchanged. The default is disabled and `Method="none"`.

Optional:

- `shuffle` - packed corpus shuffling for train/val split workflows.

## Python API

```python
from pathlib import Path
from rb_pipeline_v4 import (
    BrightnessNormalizationConfigV4,
    DetectStageConfigV4,
    SilhouetteStageConfigV4,
    PackDualStreamStageConfigV4,
    run_v4_stage_sequence_for_run,
)

project_root = Path.cwd()
run_v4_stage_sequence_for_run(
    project_root,
    run_name="your_run",
    stage_name="all",
    detect_config=DetectStageConfigV4(model_path="/abs/path/to/yolo.pt"),
    silhouette_config=SilhouetteStageConfigV4(),
    pack_dual_stream_config=PackDualStreamStageConfigV4(
        brightness_normalization=BrightnessNormalizationConfigV4(
            enabled=True,
            method="masked_median_darkness_gain",
            target_median_darkness=0.55,
            min_gain=0.5,
            max_gain=2.0,
            epsilon=1e-6,
            empty_mask_policy="skip",
        ),
    ),
)
```

## Notebook UI

Use the notebooks in `rb_ui_v4/` to keep the stage-oriented workflow used in previous versions.

- `-1_pipeline_corpus-shuffle.ipynb` provides fast input-corpus shuffle (preprocessing shuffle).
- `00_pipeline_launcher_v05.ipynb` runs `detect -> silhouette -> pack_dual_stream`.
- In launcher detect settings, if `YOLO weights` is blank, `Fallback model` (default `yolov8n.pt`) is used.
