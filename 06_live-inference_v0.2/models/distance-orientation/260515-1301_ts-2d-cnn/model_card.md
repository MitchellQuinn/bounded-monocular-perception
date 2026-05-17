# Model Card - run_0001

## Purpose
Bounded falsification test over the topology's declared regression targets 
without geometry-altering transforms.

## Data
- Training source root: `training`
- Validation source root: `validation`
- Training samples: 300000
- Validation samples: 60000
- Internal test enabled: False

## Model
- Topology id: distance_regressor_tri_stream_yaw
- Topology variant: tri_stream_yaw_v0_4
- Model class: DistanceRegressorTriStreamYaw
- Topology params: {"dropout_p": 0.0}
- Prediction mode: `distance_yaw_sincos`
- Targets: `distance_m`, `yaw_sin`, `yaw_cos`
- Input: distance image tensor plus orientation image tensor plus geometry vector

## Training
- Loss: Weighted multitask Huber (delta=1.0, distance=1.0, orientation=1.0)
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 1e-05
- LR scheduler: ReduceLROnPlateau(factor=0.5, patience=1, min_lr=1e-05)
- Early stopping patience: 3
- Accuracy metrics: fraction within +/-0.10 m, +/-0.25 m, +/-0.50 m
- Train shuffle mode: shard
- Active shard count (reservoir mode): 3
- Training shard RAM cache budget: 48.0 GiB
- Validation RAM cache enabled: True (budget 40.0 GiB)

## Results (This Run)
- Validation accuracy (+/-0.10m): 0.999067
- Validation accuracy (+/-0.25m): 0.999783
- Validation accuracy (+/-0.50m): 0.999917
- Best validation MAE: 0.014845
- Best validation RMSE: 0.024120
- Best validation loss: 0.000638

- Validation mean angular error: 1.032462 deg
- Validation median angular error: 0.780045 deg
- Validation p95 angular error: 2.436653 deg
- Validation yaw accuracy (<= 5 deg): 0.997933
- Validation yaw accuracy (<= 10 deg): 0.999117
- Validation yaw accuracy (<= 15 deg): 0.999167
## Artifact Tracking
- `best.pt` and `latest.pt` are in this run directory.
- `history.csv`, `metrics.json`, plots, and split membership files are co-located.
