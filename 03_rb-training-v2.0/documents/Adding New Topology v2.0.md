# Adding New Topology v2.0

This guide defines the canonical process for adding a new training topology in `rb-training-v2.0`.

## 1. Create a topology module

Copy:

- `src/topologies/topology_template.py`

To:

- `src/topologies/topology_<your_name>.py`

Fill in the required exports:

- `TOPOLOGY_ID` (stable string, e.g. `distance_regressor_resnet`) 
- `MODEL_CLASS_NAME`
- `DEFAULT_VARIANT`
- `supported_variants()`
- `build_model(topology_variant, topology_params)`
- `architecture_text(model)`

## 2. Keep `topology_params` strict

Inside `build_model(...)`:

- parse only supported keys from `topology_params`
- raise `ValueError` if unknown keys remain

This prevents silent typos in launch JSON and makes experiments reproducible.

## 3. Register topology in the registry

Edit `src/topologies/registry.py`:

1. import your new module near the top
2. add a new entry in `_REGISTRY` with:

- `topology_id`
- `model_class_name`
- `default_variant`
- `supported_variants`
- `build_model_fn`
- `architecture_text_fn`

After this, `train.py`/`evaluate.py` can resolve it automatically.

## 4. Run with the topology

Training CLI now supports:

- `--topology-id`
- `--topology-variant`
- `--topology-params-json`

Example:

```bash
python -m src.train \
  --topology-id distance_regressor_resnet \
  --topology-variant resnet18_v0_1 \
  --topology-params-json '{"dropout_p":0.1,"input_channels":1}'
```

## 5. Add/extend tests

At minimum update `tests/test_topology_registry.py` to verify:

- topology appears in `list_topology_ids()`
- at least one supported variant resolves
- model builds successfully from `build_model_from_spec(...)`

## 6. Artifact expectations (automatic)

When registered correctly, training artifacts include:

- `topology_id`
- `topology_variant`
- `topology_params`
- `topology_signature`
- `model_topology`

These are written to run config/manifest and used for resume compatibility checks.

## Naming recommendations

- topology ids: snake-style, stable, semantic
  - example: `distance_regressor_global_pool_cnn`
- variants: include family + version
  - example: `tiny_v0_1`, `wide_v0_1`

## Backward compatibility

`--model-architecture-variant` is still accepted as a deprecated alias for `--topology-variant`.
Use topology flags for all new work.
