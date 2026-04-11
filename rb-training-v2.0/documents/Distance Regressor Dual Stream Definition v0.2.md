# Topology: `distance_regressor_dual_stream_v0_2`

## Purpose

`dual_stream_v0_2` is the reset of the retired `v0.1` dual-stream experiment.
It keeps the same input/output contract so comparisons stay clean, but it
removes the pieces that were most likely driving the unstable validation
behavior observed in `v0.1`.

## Changes From v0.1

- `BatchNorm2d` was removed from the shape CNN.
- `GroupNorm` is used in every convolution stage instead.
- Fusion-head dropout now defaults to `0.0`.
- The topology stays on the same `distance_regressor_dual_stream` family id.

## Registry Contract

| Symbol | Value |
| --- | --- |
| `TOPOLOGY_ID` | `"distance_regressor_dual_stream"` |
| `MODEL_CLASS_NAME` | `"DistanceRegressorDualStream"` |
| `DEFAULT_VARIANT` | `"dual_stream_v0_2"` |
| `supported_variants()` | `("dual_stream_v0_2",)` |

The historical `dual_stream_v0_1` module is kept only as source reference and
is no longer registered for new runs.

## Supported `topology_params`

`dual_stream_v0_2` accepts the same keys as `v0.1`:

- `input_channels`
- `bbox_feature_dim`
- `canvas_size`
- `output_dim`
- `output_mode`
- `dropout_p`
- `geom_hidden`
- `geom_feature_dim`
- `shape_feature_dim`
- `fusion_hidden`

Unknown keys still raise `ValueError`.

## Default Training Posture

For the cleanest first rerun, use:

```json
{"output_mode":"scalar_distance","output_dim":1,"dropout_p":0.0}
```

That keeps the comparison against the retired variant focused on the
normalization change rather than on multiple moving parts at once.
