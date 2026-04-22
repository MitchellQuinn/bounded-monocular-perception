# Raccoon Ball

Raccoon Ball is a bounded monocular perception research-engineering repository.

At its current snapshot, the repository substantiates an end-to-end stack for a controlled synthetic vision problem around one known vehicle family, a fixed calibrated camera, explicit preprocessing contracts, trained distance and orientation models, an ROI-FCN crop-centre localiser, and raw-image inference pipelines that compose those pieces into runnable paths.

This repository should be read as evidence of bounded applied ML, computer vision, and research-engineering discipline. It is not evidence of unconstrained scene understanding, general object detection, real-world transfer, or deployment readiness.

## What This Repository Contains

- synthetic full-frame image generation in Unity
- contract-driven preprocessing and representation packing
- training and evaluation code for distance and joint distance-plus-yaw regression
- ROI-FCN preprocessing and training for crop-centre localisation
- raw-image inference pipelines that compose trained components into runtime paths
- technical write-ups, benchmark notes, and failure-analysis material

## What Is Deliberately Not Distributed Here

- trained `.pt` model weights
- the full synthetic training corpus
- the original Defender `.fbx` source asset

The intent is to publish a bounded example-image corpus only, rather than the full training assets or model checkpoints.

## Repository Layout

- [`01_rb_synthetic-data_3`](01_rb_synthetic-data_3) Unity-based synthetic data generation project
- [`02_synthetic-data-processing-v3.0`](02_synthetic-data-processing-v3.0) preprocessing and contract-driven corpus packing
- [`03_rb-training-v2.0`](03_rb-training-v2.0) distance and distance-plus-yaw training and evaluation
- [`04_ROI-FCN`](04_ROI-FCN) ROI-FCN preprocessing, training, and assessment
- [`05_inference-v0.1`](05_inference-v0.1) raw-image inference via reused preprocessing stages
- [`05_inference-v0.2`](05_inference-v0.2) raw-image inference via ROI-FCN plus distance-and-yaw composition
- [`documents`](documents) technical write-ups and supporting notes
- [`examples/defender-images`](examples/defender-images) bounded example-image corpus scaffold and notices
- [`scripts/run-tests.sh`](scripts/run-tests.sh) repo-level test runner for the checked-in subprojects

For a more detailed technical walkthrough, see [`documents/raccoon-ball-technical-writeup_v0.5.md`](documents/raccoon-ball-technical-writeup_v0.5.md).

## Sample Images

A scaffold for the bounded example-image corpus lives under [`examples/defender-images`](examples/defender-images).

When populated, that directory is intended to contain:

- a curated subset of Defender example images
- a directory-level notice for those example images
- a plain-text notice suitable for inclusion in a `.zip` distribution of the same corpus

The example-image corpus is separate from the repository's own code and documents, and it has its own third-party provenance and license notice. See [`THIRD_PARTY.md`](THIRD_PARTY.md) and [`examples/defender-images/NOTICE.md`](examples/defender-images/NOTICE.md).

## Rights and Third-Party Material

Unless otherwise stated, repo-authored code and documentation are provided under ordinary copyright terms only. No open-source license is attached to repo-authored material at this time.

See:

- [`COPYRIGHT.md`](COPYRIGHT.md) for repo-authored material
- [`THIRD_PARTY.md`](THIRD_PARTY.md) for third-party asset provenance

## Validation

Use the repository virtual environment Python for checks:

```bash
./scripts/run-tests.sh
```

That runner executes the main checked-in test suites from the correct project roots. A plain repo-root `pytest` run is not the intended entry point for this multi-project layout.
