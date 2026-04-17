# Raccoon Ball Repository Generation Standards v0.1

## 1. Purpose

This document defines the generation contract for code generation tasks in this repository.

Its purpose is to keep generated changes aligned with the repository's existing discipline:

- explicit contracts
- clear separation of concerns
- narrow, versioned scope
- deterministic naming and traceability
- fail-loud validation over silent drift

This is not a generic style guide.

It is a repo-specific implementation standard for producing code and documents that fit the architecture already present in:

- `01_rb_synthetic-data_3`
- `02_synthetic-data-processing-v3.0`
- `03_rb-training-v2.0`
- `04_ROI-FCN`
- `05_inference-v0.1`
- `documents`

---

## 2. Core Repo Model

The repository is an end-to-end bounded ML system with distinct responsibility layers.

The canonical flow is:

1. Unity synthetic generation writes raw images plus authoritative manifests.
2. Preprocessing consumes those manifests and writes representation artifacts plus preprocessing contracts.
3. Training and evaluation consume prepared corpora and write model/run artifacts.
4. Inference consumes trained model artifacts and reuses training plus preprocessing contracts.
5. New research components must declare narrow responsibility boundaries before implementation.

Generated code must preserve that layered model.

It must not collapse multiple stages into one ambiguous implementation layer.

---

## 3. Non-Negotiable Principles

### 3.1 Explicit contracts over implied behavior

If a component has an input shape, output shape, schema, stage status, manifest field, naming rule, or reusable result shape, the generated implementation should make that contract explicit in code and, when needed, in documentation.

Prefer:

- typed dataclasses
- protocol interfaces
- canonical config objects
- explicit schema validation
- signature or version checks

Do not rely on informal assumptions spread across multiple files.

### 3.2 One authority per concern

Generated code must respect existing sources of truth.

In this repo, examples include:

- `run.json` and `samples.csv` are authoritative corpus manifests
- preprocessing semantics live in `PreprocessingContract`
- training task semantics live in topology/task contracts
- notebooks are control surfaces, not the primary home of reusable logic
- training artifacts are written by training code, not improvised by notebooks

Do not introduce a second authority layer unless the task explicitly requires it and the new authority is documented.

### 3.3 Fail loudly on invalid state

This repo consistently prefers strict validation over silent fallback.

Generated code should raise clear errors for:

- missing required files
- unsupported variants or stage names
- malformed manifests
- unknown config keys where strictness is expected
- signature mismatches
- partially populated contract fields

Do not hide invalid state behind best-effort behavior unless the existing subsystem already defines that fallback.

### 3.4 Extend by composition, not duplication

If functionality already exists in a neighboring subsystem, generated code should reuse it rather than reimplement it with slightly different semantics.

The inference layer already demonstrates this pattern by reusing:

- preprocessing stage config derivation
- training-side model loading
- task-contract-driven runtime logic

Do not fork existing logic just to make a local version easier to write.

### 3.5 Keep v1 scope narrow

The repo repeatedly favors bounded first-pass designs.

Generated v0.x or v1 features should:

- solve the declared task cleanly
- avoid speculative extra heads, modes, or UI
- avoid introducing framework-heavy abstractions without a concrete need
- keep responsibility boundaries simple and obvious

---

## 4. Canonical Separation of Concerns

### 4.1 Unity synthetic generation: `01_rb_synthetic-data_3`

Unity generation owns:

- scene-driven sample generation
- image capture
- sample planning
- run metadata and manifest writing
- deterministic file naming

Unity generation does not own:

- OpenCV preprocessing
- YOLO or edge-detection preprocessing stages
- model training
- downstream inference policy

Within Unity, generated code should preserve the existing internal split:

- `Core/` for data objects and configuration models
- `Interfaces/` for contracts between components
- `Runtime/` for execution services
- `UnityAdapters/` for MonoBehaviour and scene integration

Do not put Unity scene behavior into `Core/`.
Do not put domain logic only needed by runtime services into adapters if it can live in `Runtime/`.

### 4.2 Preprocessing pipeline: `02_synthetic-data-processing-v3.0`

Preprocessing owns:

- staged transformation from raw captures to training-ready representations
- row-wise stage status updates
- run-wise preprocessing contract updates
- representation-specific artifact packing
- validation of preprocessing outputs

Preprocessing does not own:

- training loop semantics
- notebook UI layout
- model topology policy
- ad hoc downstream interpretation of artifacts

Generated preprocessing code must preserve stage boundaries.

Examples already present:

- config dataclasses in `config.py`
- protocol/data contracts in `contracts.py`
- manifest mutation helpers in `manifest.py`
- one module per stage or focused concern
- orchestration in `pipeline.py`

If stage semantics change, update the contract rather than burying the change inside stage code only.

### 4.3 Training and evaluation: `03_rb-training-v2.0`

Training owns:

- topology registration and resolution
- task/topology contract enforcement
- dataset loading and schema validation
- run artifact creation
- evaluation outputs and metrics
- resume compatibility

Training does not own:

- raw image generation
- preprocessing-stage implementation
- notebook-heavy business logic

Generated training code must preserve these boundaries:

- `src/topologies/` defines model families and their contracts
- `src/data.py` validates corpora and representation compatibility
- `src/train.py` and `src/evaluate.py` own run/eval artifact production
- notebooks orchestrate and present, but do not become the main logic layer

### 4.4 Notebook and helper split

Where notebooks are used, the boundary is strict.

Notebooks may own:

- widget creation
- widget arrangement
- display ordering
- event wiring
- light orchestration

Helpers or scripts must own:

- validation
- business logic
- filesystem/process interaction
- command construction
- structured result payloads

Do not generate notebook code that embeds process management, manifest mutation logic, or opaque business rules inline.

### 4.5 Inference: `05_inference-v0.1`

Inference owns:

- selection of trained runs and raw-image corpora
- contract-driven reconstruction of model inputs
- prediction and result packaging

Inference must reuse:

- training topology/task contracts
- preprocessing contracts from training artifacts or corpus manifests
- existing preprocessing helpers where semantics must match training

Inference must not define a subtly different representation contract just because it is operating in a different directory.

### 4.6 ROI FCN: `04_ROI-FCN`

New research modules must have clean single-purpose boundaries.

For ROI FCN specifically, the current contract is:

- input: full-frame image
- output: crop centre in original image coordinates
- non-goals: bbox prediction, orientation, downstream regression, system policy

Generated code in this area must keep the component focused on ROI centre localization only unless the document contract is deliberately revised.

---

## 5. Contract Discipline

### 5.1 Contract changes must be visible

If generated code changes any of the following, the change must be explicit and reviewable:

- field names
- array keys
- tensor shapes
- target semantics
- stage names
- status values
- result object shape
- artifact naming rules
- directory layout assumptions

The change should be reflected in one or more of:

- versioned constants
- contract helper code
- validation code
- tests
- documents

### 5.2 Version when semantics change

Introduce or update versioned identifiers when the meaning of a representation or contract changes, not only when code changes.

Examples already used in the repo:

- `rb-preprocess-v4-dual-stream`
- `rb-topology-output-reporting-v1`
- versioned topology variants such as `dual_stream_v0_2`
- versioned documents such as `v0.1`, `v0.2`, `v0.4`

Do not silently reuse an old contract/version label for a meaningfully different output.

### 5.3 Keep contracts canonical and serializable

Generated contract-like structures should be:

- plain mappings or dataclasses with clear serialization
- JSON-serializable when persisted
- deterministic in ordering when signatures or hashes depend on them

Do not store non-serializable runtime-only objects inside persisted contracts.

### 5.4 Preserve compatibility checks

If a subsystem already verifies compatibility with signatures or strict schema checks, generated code must preserve or extend those checks.

Do not bypass:

- topology signature checks
- task contract signature checks
- preprocessing contract consistency checks
- strict manifest authority checks

---

## 6. Ownership Rules for Generated Changes

### 6.1 Put logic in the layer that already owns it

Generated code should follow these defaults:

- new reusable Python logic goes in `src/` or the relevant package module
- new topology definitions go in `src/topologies/`
- new preprocessing logic goes in a focused module under `rb_pipeline_v4/`
- new Unity domain models go in `Core/`
- new Unity runtime services go in `Runtime/`
- new Unity scene entry points go in `UnityAdapters/`
- new repo-level standards or architecture docs go in `documents/`

Do not place logic in a notebook, Markdown document, or adapter layer if the repo already has a better home for it.

### 6.2 Reuse existing extension points

Before creating new patterns, generated code should look for and prefer:

- topology templates
- registry integration points
- config dataclasses
- protocol interfaces
- manifest helper functions
- existing test scaffolds

New abstractions should usually be added because the existing extension points are insufficient, not because they were overlooked.

### 6.3 Keep command construction and UI separate

For notebook control panels and process-launch tooling:

- UI layout belongs in notebooks
- command derivation belongs in helper logic
- previews and executable commands should come from the same underlying rules

Do not build command previews in one place and the real command in another with different logic.

### 6.4 Keep artifact writing centralized

Generated code must not create side channels for canonical outputs.

If a script or service already owns artifact writing, use or extend that path rather than writing overlapping canonical files elsewhere.

---

## 7. Naming, Paths, and Traceability

### 7.1 Use deterministic, repo-native naming

Generated code should preserve established naming patterns such as:

- versioned documents: `... v0.1.md`
- versioned modules/variants: `..._v0_1`, `..._v0_2`
- stage names: lower-case, explicit, stable
- run directories: timestamped and structured
- run ids: `run_0001`, `run_0002`, ...

Do not invent a competing naming scheme inside one subsystem.

### 7.2 Prefer repo-relative paths in saved metadata

Where metadata is persisted for later review, generated code should prefer repo-relative paths when that is already the house style.

This is especially important for:

- run manifests
- dataset summaries
- inference result payloads
- run register entries

### 7.3 Preserve traceability

A generated feature should make it easy to answer:

- what input data was used
- which contract/version was used
- which code path or topology was used
- what artifacts were produced
- how outputs map back to upstream manifests

Do not sacrifice traceability for convenience.

---

## 8. Testing and Verification Standard

Generated code must add or update tests when behavior, contracts, or artifact semantics change.

Preferred test posture in this repo:

- unit tests for focused validation or registry logic
- contract tests for schemas/signatures/result semantics
- integration tests for stage sequences and artifact generation

At minimum, add tests when generated code changes:

- topology registration or resolution
- task or topology contracts
- preprocessing stage outputs
- manifest mutation behavior
- inference reconstruction logic
- result object shapes

If a change cannot be meaningfully tested in the current repo shape, the generated output should say so explicitly and explain what remains unverified.

---

## 9. Documentation Standard

Generated code must update documentation when it changes behavior that other layers depend on.

Update or add documents when a change affects:

- subsystem purpose or non-goals
- input/output contracts
- stage ordering
- artifact expectations
- naming rules
- helper/notebook boundaries
- extension procedures for future contributors

Prefer concise, versioned documents that state:

- purpose
- scope
- ownership
- contract
- non-goals
- acceptance criteria or validation expectations

---

## 10. Anti-Patterns

Generated changes must avoid the following unless explicitly requested:

- notebook-only canonical business logic
- helper modules that assemble UI layout
- duplicated command-building logic
- duplicated preprocessing semantics in inference
- silent contract drift
- relaxed validation that hides bad state
- unknown config keys being ignored in strict subsystems
- new source-of-truth files created beside existing authoritative manifests
- broad refactors that collapse clear stage or layer boundaries
- speculative future-phase abstractions without an active use case

---

## 11. Required Generation Workflow

For non-trivial code generation tasks in this repo, the expected workflow is:

1. Identify the owning subsystem.
2. Read the relevant local documents and adjacent code before changing anything.
3. Identify the existing contract or authority for the affected behavior.
4. Implement the narrowest change that satisfies the request inside the correct layer.
5. Update validation, tests, and documents if the contract or semantics changed.
6. Verify that the change did not create a second source of truth or boundary violation.

If a requested change conflicts with existing repo discipline, the generated output should either:

- adapt the implementation to preserve the discipline, or
- state the conflict explicitly and document the intentional exception

---

## 12. Acceptance Criteria for Generated Work

A generated change is considered repo-aligned only if all of the following are true:

1. The owning layer of the change is clear.
2. No existing source of truth has been silently bypassed or duplicated.
3. Contracts remain explicit, versioned where needed, and validated.
4. Responsibilities remain separated in the same style the repo already uses.
5. Naming, paths, and artifacts remain traceable and deterministic.
6. Tests and documentation are updated when behavior or semantics changed.
7. The implementation solves the current task without speculative scope growth.

---

## 13. Bottom Line

The standard for generated code in this repository is not just "works."

It is:

- fits the correct layer
- preserves the contract model
- keeps authority clear
- fails loudly when invariants are broken
- remains traceable across generation, preprocessing, training, evaluation, and inference

If a generated change would make the repo harder to reason about, easier to drift, or less explicit about ownership, it does not meet the repository standard even if it passes locally.
