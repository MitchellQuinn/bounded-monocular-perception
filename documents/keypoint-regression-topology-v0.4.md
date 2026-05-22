# Engineering Justification for Amodal Keypoint Regression in Known-Object Pose and Distance Estimation

**Project:** Raccoon Ball  
**Object:** 1/12 scale Defender  
**Document version:** v0.4  
**Status:** Standalone engineering justification, topology proposal, and implementation specification  
**Supersedes:** Defender Keypoint Regression Topology v0.3

---

## 1. Executive Summary

This document proposes a new Raccoon Ball model family for known-object monocular pose and distance estimation. The current live path predicts distance and yaw directly from tri-stream image-derived inputs. That direct regression approach is compact and useful, but it hides the geometric state the model has inferred. When a prediction fails, the system can report that the final scalar values were wrong, but it has limited ability to explain whether the model misunderstood scale, translation, orientation, object extent, occlusion, preprocessing, or the relationship between those stages.

The proposed topology uses a precise representation: **amodal semantic keypoint regression**.

The model should infer:

```text
tri-stream image-derived inputs
  -> Defender centre in camera-space coordinates
  -> all fixed semantic external Defender keypoints, including occluded keypoints
  -> keypoint visibility / in-frame state
  -> direct distance and yaw heads for compatibility
  -> optional rigid fit to known Defender geometry
  -> derived distance/yaw and diagnostic residuals
```

The key design decision is that the model should predict **all ten fixed external keypoints**, not only the currently visible ones. The hidden keypoints are not arbitrary hallucinations. For a known rigid object in a bounded monocular scene, their camera-space locations are constrained by visible appearance, object geometry, ROI geometry, camera setup, and the pose distribution represented in the training data.

That constraint is not absolute. Direct front, direct rear, direct side, near-symmetric, or low-information views may leave multiple plausible poses compatible with the same image evidence. The topology should therefore treat hidden keypoints as **amodal inferred targets**, not as visually observed points or guaranteed-certainty outputs.

The design separates three ideas that are often accidentally conflated:

```text
amodal keypoint target: where the fixed object keypoint is in 3D, whether visible or hidden
visibility: whether the keypoint is directly visible in the image
confidence / uncertainty: how reliable the inferred keypoint is likely to be
```

This gives the system a richer supervised target, a more inspectable intermediate representation, and a stronger diagnostic surface than scalar distance/yaw regression alone.

The first implementation should remain deliberately modest: reuse the existing tri-stream inputs, add the amodal keypoint and visibility heads, preserve the existing distance/yaw outputs, normalise multi-head losses carefully, and evaluate whether the image streams contribute information beyond the ROI geometry vector.

This document is the detailed engineering specification. It is not intended to be the first artifact shown to an employer or reviewer. For first-contact portfolio use, it should be paired with a short technical summary that states the problem, the bounded claim, the key design decisions, the current implementation status, and the evidence required before external claims are made.

## 2. Version Scope

This v0.4 document is intended to stand alone. A reader should not need the v0.1 or v0.2 proposals to understand the problem, the engineering reasoning, the proposed topology, the implementation plan, or the risks.

v0.4 preserves the core v0.3 design:

1. Use **ordered semantic keypoints** or **amodal keypoints**, not generic point-cloud framing.
2. Predict all ten fixed keypoints, including occluded keypoints.
3. Add visibility/in-frame outputs as first-class auxiliary targets.
4. Treat confidence/uncertainty as separate from visibility.
5. Keep direct distance/yaw heads for compatibility and baseline comparison.
6. Use rigid fitting as a diagnostic and pose-normalisation stage, not as a replacement for raw model outputs.
7. Position the work in relation to amodal perception and keypoint-based pose estimation without overstating it as general 3D reconstruction.

v0.4 adds the following specification refinements:

1. The executive summary now describes the proposal rather than the version history.
2. The full document is explicitly positioned as a detailed specification, not the first employer-facing artifact; a companion two-page technical summary is recommended for applications and first conversations.
3. The two non-box reference keypoints are explicitly provisional and must be selected and frozen in `defender_keypoint_schema.json` before the first training run.
4. The architecture section specifies that the first keypoint head should be a single linear projection or shallow one-hidden-layer MLP rather than an over-designed head.
5. A distinct risk is added for L2/Huber averaging under symmetric or near-symmetric views, where the model may emit a physically invalid mean-pose keypoint configuration.
6. Schema metadata handling is tightened: missing schema metadata must warn clearly at inference time and block external claims rather than failing silently.
7. The geometry-only ablation criterion is tightened: without the ablation, the implementation may remain an internal experiment, but external claims about image-stream contribution are not allowed.
8. Appendix A adds a reference on pose ambiguity from monocular visual data.

This proposal is not a replacement for the current live demo-stabilisation path. The live path should remain focused on making the current inference system stable, inspectable, and usable. The amodal keypoint topology should be developed as a separate model family that can be trained, evaluated, and compared without destabilising the existing live application.

## 3. Current System Context

Raccoon Ball is already structured around a deliberately bounded monocular perception problem:

- one known vehicle family
- one fixed or controlled camera setup
- constrained scene geometry
- synthetic labelled data
- contract-driven preprocessing
- trained distance/orientation models
- live inference with trace/debug artifacts

The current system uses a tri-stream model pattern. In practical terms, the model receives:

```text
1. a scene/scale-aware image stream
   - currently represented by x_distance_image
   - preserves distance/scale-related cues
   - may include unscaled ROI or scene context depending on preprocessing contract

2. a scale-normalised appearance/orientation image stream
   - currently represented by x_orientation_image
   - helps learn object-facing direction and appearance features

3. an ROI geometry vector
   - currently represented by x_geometry
   - includes bounding-box position, size, aspect ratio, and normalised area
```

The live v0.3 runtime uses tri-stream model inputs, deterministic locator options, debug views, masks, background handling, trace bundles, and a model output contract centred on distance and yaw.

The proposed topology should preserve the existing discipline:

- explicit topology ID
- explicit preprocessing contract
- explicit model output keys
- explicit target keys
- visible metrics
- traceable runtime artifacts
- no silent changes to existing distance/yaw paths

The key architectural change is not to throw away the existing input structure. Instead, the new branch asks the model to emit a richer, inspectable geometric hypothesis from the same bounded input evidence.

---

## 4. Problem Statement

The existing direct distance/yaw regressors learn a direct mapping:

```text
image streams + ROI geometry -> distance + yaw
```

This can work, but it has two important limitations.

First, the model's internal geometric interpretation is hidden. If the model predicts the wrong distance or yaw, the runtime can inspect preprocessing artifacts, but it cannot directly inspect the model's inferred object geometry.

Second, direct regression provides sparse supervision. Each training sample may contain enough information to compute the object centre, object pose, projected keypoints, visibility, and rigid-body consistency, but the current scalar output contract mostly asks the model for final operational quantities.

The proposed topology asks a richer question:

```text
Where is the Defender centre?
Where are the fixed semantic external keypoints?
Which keypoints are visible or in frame?
What distance and yaw follow from the inferred geometry?
How physically coherent is the inferred geometry?
```

This turns the model output into an inspectable geometric state rather than only an answer.

---

## 5. Engineering Thesis

The central thesis is:

> For a known rigid object in a bounded monocular scene, it is valid and useful to supervise all fixed semantic external keypoints, including occluded keypoints, because their positions are functions of the object's pose, which is partially recoverable from visible evidence, and because the training distribution and object geometry resolve much of the residual ambiguity.

This is not a claim that a neural network can recover arbitrary hidden structure from arbitrary images. The claim is narrower and more engineering-grounded:

```text
known object geometry
+ bounded camera setup
+ synthetic labels from known transforms
+ ROI geometry
+ visible object evidence
+ constrained pose distribution
= learnable amodal keypoint targets
```

A hidden keypoint should not be treated as visually detected. It should be treated as **inferred**.

The important boundary is that pose is not always uniquely determined by the image. Symmetric or near-symmetric views can produce genuinely ambiguous evidence. A direct side view may not fully constrain front/rear orientation. A direct front or rear view may produce weak evidence for depth-extreme hidden corners. Under an L2-like loss, such cases can collapse toward averages that are numerically plausible but physically wrong.

The model should therefore output all keypoints, but the system should evaluate and use those predictions carefully. Visible and hidden keypoints should be reported separately. Ambiguous yaw and symmetry cases should be binned separately. A rigid-fit residual should be used as a diagnostic. If ambiguity becomes a measured failure mode, uncertainty heads, mixture outputs, or symmetry-aware evaluation should be considered rather than hiding the issue behind a single mean keypoint error.

## 6. Terminology and Representation

### 6.1 Avoid "point cloud" as the primary term

The original proposal described the output as a point-cloud pose topology. That phrase was directionally useful, but it is technically imprecise for this use case.

A point cloud usually implies an unordered or weakly ordered set of points. For this project, point identity matters. The model is not predicting an arbitrary unordered sample of object surface points. It is predicting a fixed set of labelled object landmarks, each with a stable semantic meaning and known local-coordinate definition.

Preferred terms:

```text
amodal semantic keypoint regression
amodal keypoint pose topology
ordered 3D keypoint regression
semantic landmark regression
```

Terms to avoid in outward-facing explanation:

```text
point cloud reconstruction
full 3D reconstruction
general object detection
general autonomous perception
```

The system is narrower and more credible than those terms suggest.

### 6.2 Amodal keypoints

A **modal** representation describes only what is visible in the image.

An **amodal** representation describes the whole object or whole object extent, including parts that are occluded or truncated.

For this project, an amodal keypoint is:

```text
a fixed semantic point on the known Defender geometry,
expressed in camera-space coordinates for the current frame,
regardless of whether that point is directly visible in the image.
```

The model should predict all amodal keypoints every time.

### 6.3 Semantic keypoints rather than arbitrary points

Each keypoint must have a stable definition:

```text
keypoint_id
semantic_name
local_x_m
local_y_m
local_z_m
notes
```

Example semantic keypoints might include:

- front upper left body corner
- front upper right body corner
- front lower left body corner
- front lower right body corner
- rear upper left body corner
- rear upper right body corner
- rear lower left body corner
- rear lower right body corner
- selected front body-shell reference point
- selected rear body-shell or deliberately named accessory reference point

The keypoint schema should represent the Defender as a **box-like set of body-shell datums**, not as a detailed mesh and not as a claim that the real Defender is literally box-shaped. The goal is not high-fidelity shape reconstruction. The goal is a compact physically meaningful representation that supports pose, distance, diagnostics, and comparison against the existing direct regressors.

### 6.4 Visibility is not confidence

Visibility and confidence should not be conflated.

A visible keypoint may be uncertain because of blur, low resolution, bad masking, glare, or partial truncation. A hidden keypoint may be fairly well constrained if the visible object pose is clear and the object is rigid.

Therefore:

```text
visibility head:
  predicts whether a keypoint is directly visible / in-frame

confidence or uncertainty:
  estimates how reliable a predicted location is
  can be learned later or derived from residuals and disagreement
```

The first implementation should include visibility/in-frame labels and metrics. A learned uncertainty head is useful, but it can be added after the baseline amodal keypoint topology trains cleanly.

---

## 7. Related Work and Technical Positioning

This project is not attempting to copy a general-purpose pose-estimation paper directly. It is a bounded engineering system with its own constraints. The proposed design is aligned with three established computer-vision ideas: amodal perception, keypoint-based pose estimation, and occlusion-aware keypoint localisation.

The useful transfer is conceptual, not architectural. Amodal perception supports the idea that a model can be trained against full object extent rather than only visible pixels. Keypoint-to-pose systems support the idea that structured landmarks can act as an intermediate pose representation. Occlusion-aware pose methods support the idea that occluded or truncated keypoints do not have to be discarded if uncertainty and geometry are handled carefully.

The non-claims are equally important:

```text
Raccoon Ball is not general 3D reconstruction.
Raccoon Ball is not general amodal object detection.
Raccoon Ball is not autonomous-driving perception.
Raccoon Ball is not claiming arbitrary hidden geometry can be recovered from arbitrary monocular images.
```

The narrower claim is that, for one known rigid vehicle in a bounded camera setup with synthetic transform-derived labels, amodal semantic keypoints are a useful supervised representation and diagnostic surface.

Full references are kept in Appendix A so this engineering document remains decision-focused rather than paper-shaped.

## 8. Why Predict All Ten Keypoints?

### 8.1 The visible-only target is unnecessarily lossy

Predicting only visible points forces the model to discard information that is still geometrically meaningful. For a rigid known object, the hidden rear lower corner is not independent of the visible front face, roofline, bounding box, and yaw evidence. It is a consequence of the object's pose.

Visible-only prediction also creates a downstream pose problem: the set of available keypoints changes from frame to frame. Heavy occlusion or self-occlusion can remove exactly the points that would be useful for stable pose solving.

### 8.2 Amodal keypoints preserve the rigid-body hypothesis

Predicting all ten keypoints asks the model to emit a complete object hypothesis:

```text
not:  "which keypoints can I see?"
but:  "where is the known Defender in 3D, and where do its fixed landmarks fall?"
```

This creates a representation that can be checked against known object geometry. If the model predicts an impossible shape, the rigid-fit residual can expose that failure. Direct distance/yaw regression alone cannot expose the same failure mode.

### 8.3 Hidden keypoints are supervised labels, not guessed annotations

In synthetic data, hidden keypoints are known because the object transform and local point schema are known. In a real validation rig, hidden keypoints can be computed from measured rig transforms.

The hidden keypoint label is therefore not a human guess. It is a derived ground-truth coordinate from the known object pose.

### 8.4 The boundary: ambiguity still exists

Amodal keypoint prediction is not magic. If two object poses create indistinguishable image evidence, the model cannot know the true pose from pixels alone. It may regress toward an average, especially under an L2-like loss, and that average may be physically wrong.

Mitigations:

- use yaw sine/cosine rather than naive angle regression
- evaluate symmetry-aware cases separately
- report visible and hidden keypoint errors separately
- add visibility/in-frame diagnostics
- add rigid-fit residuals
- consider uncertainty heads or mixture outputs if ambiguity becomes a measured failure mode

---

## 9. Proposed Topology v0.4

### 9.1 Preferred naming

Preferred outward-facing name:

```text
Defender Amodal Keypoint Pose Regressor
```

Preferred topology identifiers:

```text
TOPOLOGY_ID = "defender_amodal_keypoint_pose"
MODEL_CLASS_NAME = "DefenderAmodalKeypointPoseRegressor"
DEFAULT_VARIANT = "defender_amodal_keypoint_pose_v0_4"
```

Suggested files:

```text
03_rb-training-v2.0/src/topologies/topology_defender_amodal_keypoint_pose.py
03_rb-training-v2.0/src/topologies/topology_defender_amodal_keypoint_pose_v0_4.py
```

If implementation work has already started using `defender_amodal_keypoint_pose_v0_2` or `defender_amodal_keypoint_pose_v0_3`, those names can be retained as compatibility aliases. Do not churn working artifact names purely for documentation neatness. The important change is the v0.4 contract content: schema versioning, frozen keypoint definitions, loss normalisation, ambiguity handling, and ablation requirements.

If earlier implementation work used `defender_pointcloud_pose`, that name can also be kept as a compatibility alias. Documentation and outward-facing explanation should use amodal keypoint terminology.

Suggested aliases:

```text
defender_pointcloud_pose -> defender_amodal_keypoint_pose
defender_points_3d_flat -> defender_keypoints_3d_flat
y_defender_points_3d_flat -> y_defender_keypoints_3d_flat
```

### 9.2 Inputs

The first version should reuse the existing tri-stream input structure:

```text
x_distance_image
x_orientation_image
x_geometry
```

Recommended semantic interpretation:

```text
x_distance_image:
  scene-aware or scale-preserving ROI image stream
  retains apparent scale and distance cues

x_orientation_image:
  scale-normalised object appearance stream
  supports yaw/orientation and shape cues

x_geometry:
  explicit ROI geometry vector
```

The existing geometry vector can remain:

```text
cx_px
cy_px
w_px
h_px
cx_norm
cy_norm
w_norm
h_norm
aspect_ratio
area_norm
```

This avoids unnecessary churn in preprocessing and live inference. The model starts from the same evidence as the current tri-stream system but learns additional structured outputs.

However, `x_geometry` is a strong cue in a fixed-camera setup. It can carry enough position and scale information for a model to achieve plausible synthetic metrics while ignoring image texture/shape evidence. v0.4 therefore requires a geometry-only ablation baseline and, ideally, a geometry-perturbation test.

### 9.3 Outputs

Minimum v0.4 outputs:

```python
{
    "distance_m": distance,                         # shape: (B,)
    "yaw_sin_cos": yaw_sin_cos,                    # shape: (B, 2)
    "defender_center_3d": center_3d,               # shape: (B, 3)
    "defender_keypoints_3d_flat": keypoints_3d,    # shape: (B, 30)
    "defender_keypoints_visible_logits": visible,  # shape: (B, 10)
}
```

Recommended optional outputs:

```python
{
    "defender_keypoints_in_frame_logits": in_frame,       # shape: (B, 10)
    "defender_keypoints_uncertainty_logvar": logvar,      # shape: (B, 10) or (B, 30)
    "derived_distance_m": derived_distance,               # from rigidified keypoints
    "derived_yaw_sin_cos": derived_yaw_sin_cos,           # from rigidified keypoints
    "rigid_fit_residual_m": rigid_fit_residual,           # diagnostic
}
```

The direct distance and yaw heads should remain present. They preserve comparability with existing models and allow the live runtime to use the new model before full keypoint-aware runtime support exists.

### 9.4 High-level architecture

The first architecture should be deliberately boring:

```text
x_distance_image      -> distance/scene image encoder
x_orientation_image   -> orientation/appearance image encoder
x_geometry            -> geometry MLP

encodings             -> fusion trunk

fusion trunk          -> distance_head
fusion trunk          -> yaw_head
fusion trunk          -> center_3d_head
fusion trunk          -> keypoints_3d_head
fusion trunk          -> visibility_head
optional fusion trunk -> uncertainty_head
```

The first implementation should keep the heads simple. The keypoint head should be either:

```text
fusion trunk -> Linear(30)
```

or, if the shared trunk representation is too thin:

```text
fusion trunk -> Linear(hidden) -> activation -> Linear(30)
```

Do not add a specialised graph module, transformer decoder, iterative solver, or differentiable rigid fitting inside the initial keypoint head before the representation has been proven useful. Architectural complexity should be earned by measured failure modes, not added pre-emptively.

The important engineering step is not architectural novelty. It is the supervised representation, the schema discipline, the ablations, and the diagnostics that follow from the representation.

## 10. Canonical Keypoint Schema

The canonical keypoint schema is not illustrative metadata. It is the geometric data contract for this topology.

The exact local coordinates for each keypoint must be specified in a separate `defender_keypoint_schema.json` document, and the schema version must be tracked in all training data, model artifacts, evaluation outputs, inference traces, and live-runtime extras.

### 10.1 Local coordinate frame

The Defender keypoints must be defined in a stable Defender-local coordinate frame. The exact convention must be documented in the keypoint schema.

Recommended v0.4 convention:

```text
+X = Defender right
+Y = Defender up
+Z = Defender forward
origin = midpoint of the canonical measured body-shell bounding box
units = metres
```

The proposed origin is deliberately **not** centre of mass and should not be described as such. It is the midpoint between the measured minimum and maximum body-shell extents along X, Y, and Z, according to the schema's inclusion/exclusion rules.

Default inclusion rule:

```text
body shell / stable external body datum: included
wheels / tyres: excluded unless explicitly added as wheel keypoints
mirrors / flexible accessories: excluded
roof rack / rails: excluded unless present on both synthetic and real targets and explicitly included
rear spare wheel: excluded unless deliberately represented by a named spare-wheel keypoint
```

If any of these choices change, that is a new schema version. Silent coordinate-frame drift is not allowed.

### 10.2 Keypoint definition requirements

Each keypoint should have:

```text
keypoint_id
semantic_name
body_part_or_datum
surface_or_reference_rule
local_x_m
local_y_m
local_z_m
measurement_method
measurement_uncertainty_m
visibility_test_method
wireframe_edges
notes
```

The labels should be stable and physically meaningful. Avoid labels that are convenient in Unity but difficult to identify or measure on the real model.

For v0.4, keypoints should be defined as **body-shell datum points** unless there is a deliberate reason to use chassis points, wheel points, or accessory points. This avoids the ambiguity of phrases such as "upper left body corner" on a curved or chamfered real model. Where the physical model has rounded edges, the keypoint may be a derived intersection of canonical body-shell bounding planes rather than a literal infinitesimal physical corner.

Real-vehicle measurement should be possible with simple tools first: calipers, ruler/tape, flat reference surface, marked datum board, and photographs for sanity checking. The first schema does not need sub-millimetre truth, but it does need repeatable definitions.

### 10.3 Candidate ten-keypoint set

A candidate first schema:

```text
0 front_upper_left_body_shell_corner
1 front_upper_right_body_shell_corner
2 front_lower_left_body_shell_corner
3 front_lower_right_body_shell_corner
4 rear_upper_left_body_shell_corner
5 rear_upper_right_body_shell_corner
6 rear_lower_left_body_shell_corner
7 rear_lower_right_body_shell_corner
8 provisional_front_reference_to_be_frozen
9 provisional_rear_reference_to_be_frozen
```

The first eight points should be treated as canonical body-shell bounding datums unless the schema says otherwise.

The final two points remain provisional in this proposal document only. They must be selected and frozen in `defender_keypoint_schema.json` before the first training run. Changing them after training begins constitutes a schema version change. Training data, model artifacts, evaluation reports, and live inference traces must not use ambiguous `or` names such as `front_roof_or_bonnet_centre_reference` or `rear_roof_or_spare_wheel_reference`.

The front reference must resolve to one specific datum, for example:

```text
front_roof_centre_body_shell_datum
```

or:

```text
bonnet_front_centre_body_shell_datum
```

Those are different points with different visibility and transfer properties. They are not interchangeable once training starts.

If the rear spare wheel is used, name it explicitly, for example:

```text
rear_spare_wheel_center
```

Do not hide that choice behind a generic phrase such as "rear reference". A spare-wheel keypoint has different visibility, geometry, and transfer implications from a body-shell roof datum.

### 10.4 Ordered loss, not unordered set loss

Because keypoint identity matters, the primary loss should preserve point correspondence:

```text
predicted keypoint 0 is compared with target keypoint 0
predicted keypoint 1 is compared with target keypoint 1
...
predicted keypoint 9 is compared with target keypoint 9
```

Do not use unordered Chamfer distance as the primary loss for this topology. Chamfer distance can hide point identity swaps, which are damaging for pose estimation and diagnostics.

### 10.5 Schema-version tracking

Every generated sample, packed NPZ shard, training run, model card, checkpoint metadata, evaluation report, and trace bundle should record:

```text
defender_keypoint_schema_version
defender_keypoint_schema_path_or_hash
coordinate_space
local_coordinate_frame
num_keypoints
coordinate_width
flattening_order
```

If a model is trained under one schema and evaluated under another, the runtime should fail clearly rather than silently comparing incompatible coordinates.

If a model is loaded for inference without keypoint schema metadata, the runtime must not fail silently. It should emit a clear compatibility warning and preserve the warning in result extras or trace metadata. Such a model may still be usable for internal debugging if the operator knowingly accepts the risk, but it should not be used for external claims about keypoint accuracy, rigid-fit diagnostics, or derived pose quality.

## 11. Dataset and Label Contract

### 11.1 Required labels

Each synthetic sample should provide:

```text
y_distance_m
  shape: scalar or (1,)

 y_yaw_sin
  shape: scalar or (1,)

 y_yaw_cos
  shape: scalar or (1,)

 y_defender_center_3d
  shape: (3,)
  coordinate space: camera-space preferred

 y_defender_keypoints_3d
  shape: (10, 3)
  coordinate space: same as centre

 y_defender_keypoints_3d_flat
  shape: (30,)
  flattening: keypoint-major [p0_x, p0_y, p0_z, p1_x, ...]

 y_defender_keypoints_visible
  shape: (10,)
  values: 0/1
```

Each dataset or shard manifest should also provide:

```text
defender_keypoint_schema_version
schema coordinate convention
coordinate space for y_defender_center_3d and y_defender_keypoints_3d_flat
```

The schema version is required metadata, not optional annotation.

### 11.2 Recommended optional labels

```text
y_defender_keypoints_2d
  shape: (10, 2)
  projected image pixel coordinates

 y_defender_keypoints_in_frame
  shape: (10,)
  values: 0/1

 y_defender_pose_rotation
  shape: yaw, quaternion, rotation matrix, or rotation vector depending on later pose representation

 y_defender_occlusion_fraction
  shape: scalar or per-keypoint/per-sample diagnostic
```

### 11.3 Visibility and in-frame semantics

Recommended definitions:

```text
visible:
  the keypoint is directly visible to the camera and not occluded by the object itself or scene geometry

in_frame:
  the projected keypoint coordinate lies inside the image bounds
```

A keypoint may be in-frame but hidden by self-occlusion. A keypoint may be outside the frame because the object is truncated. These should remain separate labels.

### 11.4 Camera-space targets

For v0.3, camera-space 3D targets are recommended:

```text
x/y/z coordinates are expressed in the live camera coordinate frame
```

This makes the keypoints directly relevant to live inference. It also keeps the target close to the operational quantities of distance and yaw.

The proposal depends on a bounded camera setup and known object scale. It should not be described as solving general monocular scale ambiguity.

## 12. Topology Contract Proposal

The topology contract should extend the existing mapping-output pattern. The exact helper names should be adapted to the repository conventions rather than forced literally.

Recommended contract shape:

```python
TOPOLOGY_CONTRACT = {
    "contract_version": TOPOLOGY_CONTRACT_VERSION,
    "task_family": "multitask_regression",
    "topology_id": "defender_amodal_keypoint_pose",
    "default_variant": "defender_amodal_keypoint_pose_v0_3",
    "input_mode": "tri_stream_distance_orientation_geometry",
    "output_kind": "mapping",
    "schema": {
        "defender_keypoint_schema_version": "defender_keypoint_schema_v1",
        "schema_key": "defender_keypoint_schema_version",
        "schema_path": "schemas/defender_keypoint_schema.json",
        "num_keypoints": 10,
        "coordinate_width": 3,
        "flattening_order": "keypoint_major_xyz",
        "coordinate_space": "camera_space_m",
    },
    "targets": {
        "distance": {
            "kind": "regression",
            "columns": ["distance_m"],
            "target_npz_key": "y_distance_m",
        },
        "yaw": {
            "kind": "circular_regression",
            "columns": ["yaw_sin", "yaw_cos"],
            "debug_columns": ["yaw_deg"],
            "target_npz_keys": ["y_yaw_sin", "y_yaw_cos"],
            "debug_target_npz_key": "y_yaw_deg",
        },
        "defender_center_3d": {
            "kind": "vector_regression",
            "columns": [
                "defender_center_x_m",
                "defender_center_y_m",
                "defender_center_z_m",
            ],
            "target_npz_key": "y_defender_center_3d",
            "width": 3,
        },
        "defender_keypoints_3d": {
            "kind": "ordered_keypoint_regression_3d",
            "target_npz_key": "y_defender_keypoints_3d_flat",
            "width": 30,
            "num_keypoints": 10,
            "coordinate_width": 3,
            "schema_key": "defender_keypoint_schema_version",
            "loss_reduction": "mean_over_keypoints_and_coordinates",
        },
        "defender_keypoints_visible": {
            "kind": "multi_label_binary_classification",
            "target_npz_key": "y_defender_keypoints_visible",
            "width": 10,
        },
        "defender_keypoints_in_frame": {
            "kind": "multi_label_binary_classification",
            "target_npz_key": "y_defender_keypoints_in_frame",
            "width": 10,
            "optional": True,
        },
    },
    "outputs": {
        "distance": {
            "kind": "regression",
            "columns": ["distance_m"],
            "output_key": "distance_m",
        },
        "yaw": {
            "kind": "circular_regression",
            "columns": ["yaw_sin", "yaw_cos"],
            "output_key": "yaw_sin_cos",
        },
        "defender_center_3d": {
            "kind": "vector_regression",
            "output_key": "defender_center_3d",
            "width": 3,
        },
        "defender_keypoints_3d": {
            "kind": "ordered_keypoint_regression_3d",
            "output_key": "defender_keypoints_3d_flat",
            "width": 30,
            "num_keypoints": 10,
            "coordinate_width": 3,
        },
        "defender_keypoints_visible": {
            "kind": "multi_label_binary_classification",
            "output_key": "defender_keypoints_visible_logits",
            "width": 10,
        },
    },
    "runtime": {
        "prediction_mode": "defender_amodal_keypoint_pose",
        "required_live_outputs": ["distance_m", "yaw_sin_cos"],
        "optional_live_outputs": [
            "defender_center_3d",
            "defender_keypoints_3d_flat",
            "defender_keypoints_visible_logits",
            "rigid_fit_residual_m",
        ],
    },
}
```

If the existing contract system does not yet support `ordered_keypoint_regression_3d`, the first implementation can represent the keypoint head as generic vector regression with metadata:

```text
kind: vector_regression
width: 30
semantic_role: ordered_keypoints_3d
num_keypoints: 10
coordinate_width: 3
schema_key: defender_keypoint_schema_version
loss_reduction: mean_over_keypoints_and_coordinates
```

Do not block initial implementation on creating a perfect new contract kind if generic vector regression is enough to train and measure v0.3. Do not skip schema metadata just because the first implementation uses a generic vector-regression helper.

## 13. Loss Design

### 13.1 Minimum v0.4 loss

Start simple and measurable:

```text
total_loss =
    distance_weight      * distance_loss
  + yaw_weight           * orientation_loss
  + center_weight        * center_3d_loss
  + keypoint_weight      * keypoint_3d_loss
  + visibility_weight    * keypoint_visibility_loss
```

Recommended initial losses:

```text
distance_loss:
  Huber / SmoothL1 over distance_m
  reduction: mean over batch

orientation_loss:
  existing sin/cos yaw loss
  reduction: mean over batch and two circular coordinates

center_3d_loss:
  Huber / SmoothL1 over x/y/z
  reduction: mean over batch and 3 coordinates

keypoint_3d_loss:
  Huber / SmoothL1 over all 30 ordered keypoint coordinates
  reduction: mean over batch, keypoint count, and coordinate count
  do not sum all 30 coordinates before weighting

keypoint_visibility_loss:
  binary cross entropy with logits over 10 visibility labels
  reduction: mean over batch and keypoint count
```

Recommended starting weights:

```text
distance_weight = 1.0
orientation_weight = 1.0
center_weight = 1.0
keypoint_weight = 1.0
visibility_weight = 0.1 to 0.5
```

The exact values should be measured rather than assumed. The keypoint loss must be normalised so that it does not dominate merely because it has 30 coordinates. With mean reductions, a `keypoint_weight` of `1.0` is a sensible starting hypothesis. With summed reductions, it is not.

Training logs should report at least:

```text
raw_distance_loss
raw_orientation_loss
raw_center_3d_loss
raw_keypoint_3d_loss
raw_keypoint_visibility_loss
weighted_distance_loss
weighted_orientation_loss
weighted_center_3d_loss
weighted_keypoint_3d_loss
weighted_keypoint_visibility_loss
```

If possible, also log gradient norms or head-specific contribution summaries during early runs. The objective is to detect loss-scale dominance early, not after several long training runs.

### 13.2 Do not mask hidden keypoints out of the amodal 3D loss

The amodal keypoint head should be supervised on all ten keypoints. Hidden keypoints are part of the target representation.

Masking hidden keypoints out of the 3D keypoint loss would turn the task back into modal visible-keypoint detection, which is not the proposed topology.

Visibility may be used for:

- auxiliary visibility classification
- metric breakdowns
- 2D projection loss masks
- debugging overlays
- confidence interpretation

It should not be used as a default reason to remove hidden 3D keypoints from the amodal target.

### 13.3 Projection loss

If projected 2D labels and camera intrinsics are available, add:

```text
projection_loss = SmoothL1(project(predicted_3d_keypoints), target_2d_keypoints)
```

This should normally be masked or weighted by in-frame/visibility state, because a projected hidden keypoint can still have a valid coordinate but may not correspond to an observable visual feature.

Projection loss is useful because it connects the 3D hypothesis back to image-space evidence.

### 13.4 Pairwise geometry loss

The canonical Defender keypoints have known internal distances. For each predicted keypoint set, compute pairwise distances and compare them with the canonical distances:

```text
pairwise_distance_loss = error(||p_i - p_j||, ||P_i - P_j||)
```

Where:

```text
p_i, p_j = predicted camera-space keypoints
P_i, P_j = canonical local-space keypoints
```

Because rigid transforms preserve distances, this loss penalises stretched or compressed Defender shapes.

### 13.5 Rigid-fit residual loss

Fit the known canonical keypoints to the predicted keypoints using a fixed-scale rigid transform. Penalise the residual between raw predicted keypoints and the nearest physically valid rigid shape.

This encourages the network to emit geometrically coherent keypoints while still preserving raw predictions for diagnostics.

### 13.6 Derived pose consistency loss

Derive distance and yaw from the rigidified keypoints and compare them to the existing distance/yaw targets:

```text
derived_pose_loss =
    derived_distance_loss
  + derived_yaw_loss
```

This encourages the amodal keypoint representation to support the operational quantities the live system needs.

### 13.7 Uncertainty-aware loss

If an uncertainty head is later added, one option is to predict a per-keypoint or per-coordinate log variance and train with a Gaussian negative log likelihood style loss.

This is useful if ambiguous hidden keypoints otherwise collapse toward physically invalid averages. It is not required for the first v0.4 implementation.

## 14. Rigidification and Pose Derivation

The raw amodal keypoints should be treated as the model's geometric hypothesis, not final physical truth.

A rigidification stage should fit the known Defender geometry to the raw keypoints:

```text
canonical Defender keypoints
  -> fixed-scale rigid transform fit
  -> raw predicted camera-space keypoints
  -> best centre + rotation
  -> rigidified physically valid keypoints
```

The rigidification stage should output:

```text
rigidified_center_3d
rigidified_keypoints_3d
rigid_rotation
rigid_yaw_deg
rigid_distance_m
rigid_fit_residual_m
```

Raw and rigidified predictions should both be preserved.

A high rigid-fit residual is not merely an error. It may indicate:

- uncertain model prediction
- out-of-distribution input
- locator/preprocessing failure
- synthetic-to-real mismatch
- wrong keypoint schema
- object not represented coherently

This is one of the main engineering advantages of the topology. It turns failure into a diagnosable signal.

---

## 15. Evaluation Metrics

### 15.1 Existing distance/yaw metrics

Retain the existing metrics:

```text
distance_mae_m
distance_rmse_m
distance_acc@0.10m
distance_acc@0.25m
distance_acc@0.50m
yaw_mean_error_deg
yaw_median_error_deg
yaw_p95_error_deg
yaw_acc@5deg
yaw_acc@10deg
yaw_acc@15deg
```

### 15.2 Centre and keypoint metrics

Add:

```text
center_mean_error_m
center_median_error_m
center_p95_error_m
keypoint_mean_point_error_m
keypoint_median_point_error_m
keypoint_p95_point_error_m
keypoint_mean_coordinate_error_m
```

Keypoint error should reshape:

```text
(B, 30) -> (B, 10, 3)
```

Then compute Euclidean point error per keypoint.

### 15.3 Visibility/in-frame metrics

If visibility labels exist, report:

```text
visible_keypoint_mean_error_m
hidden_keypoint_mean_error_m
in_frame_keypoint_mean_error_m
out_of_frame_keypoint_mean_error_m
keypoint_visibility_accuracy
keypoint_visibility_precision
keypoint_visibility_recall
keypoint_visibility_f1
```

Do not use only a single averaged keypoint error. The visible-vs-hidden split is central to evaluating the engineering claim.

### 15.4 Rigidification metrics

If rigidification is implemented, report:

```text
rigid_fit_mean_residual_m
rigid_fit_median_residual_m
rigid_fit_p95_residual_m
rigidified_center_error_m
rigidified_keypoint_error_m
derived_distance_mae_m
derived_yaw_mean_error_deg
raw_vs_rigidified_distance_delta_m
raw_vs_rigidified_yaw_delta_deg
```

### 15.5 Reprojection metrics

If projected 2D labels are implemented, report:

```text
reprojection_mean_error_px
reprojection_median_error_px
reprojection_p95_error_px
visible_reprojection_mean_error_px
hidden_reprojection_mean_error_px
```

### 15.6 Occlusion-bin evaluation

Evaluate by occlusion and truncation level:

```text
0-20% occluded
20-50% occluded
50%+ occluded
self-occluded but in-frame
partially truncated
small ROI
large ROI
ambiguous yaw / symmetry cases
```

Amodal keypoint prediction should not be judged only on easy fully visible cases.

### 15.7 Model comparisons

Compare at least:

```text
A. current direct distance/yaw model
B. direct distance/yaw + centre head
C. visible-only keypoint model
D. all-keypoint amodal model
E. all-keypoint amodal model + visibility head
F. all-keypoint amodal model + rigidification diagnostics
```

The visible-only model is a useful baseline, not the intended endpoint.

---

## 16. Live Inference Integration

### 16.1 Phase 1: Keep live runtime compatible

The first amodal keypoint model should still emit:

```text
distance_m
yaw_sin_cos
```

That allows the existing live engine to keep decoding distance/yaw with minimal changes.

The extra keypoint outputs can initially be ignored unless the live engine rejects unknown mapping keys. If it rejects them, adjust validation so extended models are accepted when they provide at least:

```text
distance_m
yaw_sin_cos
```

and preserve extra keys in result extras or debug metadata.

### 16.2 Phase 2: Preserve keypoint outputs in extras

Extend the inference result path to preserve optional fields:

```text
predicted_defender_center_3d
predicted_defender_keypoints_3d_flat
predicted_defender_keypoints_visible_logits
predicted_defender_keypoints_visible_probs
rigidified_defender_center_3d
rigidified_defender_keypoints_3d_flat
rigid_fit_residual_m
defender_keypoint_schema_version
defender_keypoint_schema_hash
```

These should not replace the existing distance/yaw fields.

### 16.3 Phase 3: Add trace artifacts

Add trace JSON artifacts:

```text
predicted_defender_keypoints.json
rigidified_defender_pose.json
keypoint_metrics.json
```

Each should include:

```text
request_id
frame_hash
model_root
checkpoint_path
preprocessing_contract
keypoint_schema_version
keypoint_schema_hash
coordinate_space
raw_keypoints
visibility_logits
visibility_probs
rigidified_keypoints
center
rigid_fit_residual
warnings
```

### 16.4 Deferred: visual debug overlays

Visual overlays are valuable, but they are not part of the first implementation milestone. Do not let GUI rendering decisions block the topology, labels, losses, metrics, or trace preservation.

Move overlay specifics into a separate diagnostic visualisation note after the model can produce keypoint outputs. That note can define:

```text
2D projected raw keypoints
2D projected rigidified keypoints
wireframe connecting canonical keypoint IDs
keypoint labels
centre marker
visibility state per keypoint
raw-vs-rigidified disagreement
```

These overlays should be debug artifacts first. GUI rendering can come later.

## 17. Real-World Validation

Real-world validation should be split into two tiers: a minimal first-pass check and a later full validation rig.

### 17.1 Minimal first-pass validation

The first real-world validation attempt should answer a narrow question:

```text
Does the synthetic-trained keypoint/distance/yaw model land in the right ballpark on real camera frames?
```

This does not require a full calibrated training rig. It can start with:

```text
fixed camera position
known camera intrinsics if already available
flat base surface or marked floor/grid
several measured distance marks
a small number of measured yaw angles
manual capture of 10-30 representative frames
manual notes on visible keypoints and obvious failure modes
trace bundles for every validation frame
```

For this tier, treat measurements as diagnostic references, not as high-precision ground truth. The objective is to catch large synthetic-to-real failures early: wrong scale, wrong orientation family, broken preprocessing, schema mismatch, or keypoint predictions that are visibly incoherent.

Manual visible-keypoint annotation can be useful here, but do not attempt to hand-label hidden 3D keypoints in photographs. Hidden keypoints should come from transforms or from the schema/pose setup, not from guesswork.

### 17.2 Full validation rig

A full real-world validation rig is needed if this branch is used for real-labelled evaluation, fine-tuning, or strong external claims.

The goal is not to manually measure every hidden keypoint in every photograph. The better strategy is:

```text
measure the Defender's local keypoint geometry once
measure the rig/camera/Defender pose per capture
compute all keypoint positions from known transforms
```

A practical setup:

```text
fixed calibrated camera
flat baseboard
printed coordinate grid or fiducial board
sliding distance rail or marked distance positions
rotary plate with angle markings/detents
Defender cradle mounted to the rotary plate
```

For each capture, record:

```text
camera intrinsics
camera-to-baseboard transform
baseboard-to-rotary-plate transform
rotary angle
Defender cradle offset
Defender local keypoint schema version
Defender local keypoint definitions
```

From this, compute camera-space positions for all labelled keypoints, including hidden keypoints.

The fiducial/grid should be useful for validation but should not become a cue the model relies on. For training or model input, it may need to be masked or kept outside the ROI.

## 18. Risks and Mitigations

### Risk: The model predicts distorted geometry

Mitigations:

- preserve raw keypoint predictions
- add pairwise distance loss
- add rigid-fit residual diagnostics
- rigidify before deriving final pose
- use high residual as uncertainty/failure signal

### Risk: L2/Huber averaging produces physically invalid mean-pose keypoints

Symmetric or near-symmetric views can produce genuinely multimodal evidence. Under a single-output regression loss, the model may average between valid pose hypotheses and emit a keypoint configuration that is not a valid Defender pose. This is not merely local deformation. It is mode collapse under ambiguity: a numerically smooth prediction that may correspond to no physically real object state.

Mitigations:

- bin evaluation by yaw symmetry / near-symmetry cases
- compare raw keypoint error, rigid-fit residual, and derived-pose error for ambiguous views
- inspect whether failures concentrate around direct front, direct rear, or direct side views
- consider uncertainty heads, mixture outputs, or symmetry-aware evaluation if this becomes a measured failure mode
- avoid external claims that hidden keypoints are uniquely recovered in views where the evidence is genuinely multimodal


### Risk: Hidden keypoints are too ambiguous

Mitigations:

- report visible and hidden keypoint metrics separately
- avoid treating hidden-keypoint error as the only success measure
- evaluate symmetry and near-symmetry cases separately
- evaluate whether hidden-keypoint error correlates with yaw ambiguity, especially near direct front/rear/side views
- evaluate whether rigidified distance/yaw remains stable even if hidden keypoints are less accurate
- add uncertainty, mixture outputs, or symmetry-aware evaluation if ambiguity becomes a measured failure mode

### Risk: The model learns dataset priors rather than image evidence

Mitigations:

- evaluate on pose distributions outside the training centre
- include ablations with geometry stream removed or perturbed
- inspect whether hidden keypoint predictions collapse toward mean pose
- use occlusion-bin and yaw-bin reporting

### Risk: The model ignores the image streams and relies mainly on `x_geometry`

In a fixed-camera setup, the ROI geometry vector contains strong cues: bounding-box position, size, aspect ratio, and normalised area. A model may achieve good synthetic metrics while relying mostly on that vector and barely using the distance/orientation image streams. This would be especially dangerous if live or real-image ROI extraction is imperfect.

Mitigations:

- train and evaluate a geometry-only baseline using `x_geometry` without image streams
- compare the full model against the geometry-only baseline on the same validation split
- run a geometry-perturbation test to see whether small ROI/bbox errors produce large output shifts
- if the full model is not substantially better than the geometry-only baseline, do not claim that the image encoders are contributing meaningfully
- consider dropout/noise on geometry features during training if measured dependence is excessive

### Risk: Synthetic keypoint labels do not transfer to the real model

Mitigations:

- choose physically meaningful keypoint definitions
- measure the real Defender's approximate local geometry
- version the keypoint schema
- start with minimal real validation before building a full rig
- compare synthetic-only, real-validation, and mixed calibration results separately

### Risk: Visibility and confidence get conflated

Mitigations:

- make visibility a separate supervised head
- use rigid-fit residual and direct-vs-derived pose disagreement as reliability signals
- add a learned uncertainty head only after the baseline is measurable

### Risk: This distracts from the current demo path

Mitigations:

- implement as a separate topology family
- do not mutate existing model artifacts
- keep existing distance/yaw outputs
- treat live keypoint support as optional until training evidence exists
- defer GUI overlays until the model and traces exist

### Risk: The first implementation gets bogged down in geometric perfection

Mitigations:

- v0.3 first pass should add schema, labels, heads, losses, metrics, visibility reporting, and ablation checks
- rigidification can begin as offline evaluation/post-processing
- do not block initial training on differentiable pose mathematics
- do not block initial training on full real-world rig construction

## 19. Minimum Implementation Plan

### Objective

Implement a new Raccoon Ball topology family that predicts distance, yaw, Defender centre, all ten amodal semantic keypoints, and keypoint visibility from the existing tri-stream inputs, while preserving the current distance/yaw live path.

### Non-goals for the first pass

Do not:

- replace the current live v0.3 demo path
- remove or mutate existing `distance_regressor_tri_stream_yaw` models
- require the live GUI to render keypoint overlays immediately
- implement full real-world validation rig support immediately
- require differentiable rigid fitting before baseline training works
- claim general 3D reconstruction

### Files to inspect first

```text
03_rb-training-v2.0/src/topologies/registry.py
03_rb-training-v2.0/src/topologies/contracts.py
03_rb-training-v2.0/src/topologies/topology_tri_stream_yaw.py
03_rb-training-v2.0/src/topologies/topology_tri_stream_yaw_v0_5.py
03_rb-training-v2.0/src/topologies/topology_tri_stream_yaw_common.py
03_rb-training-v2.0/src/task_runtime.py
06_live-inference_v0.3/src/live_inference/interfaces/contracts.py
06_live-inference_v0.3/src/live_inference/engines/torch_tri_stream_engine.py
```

### Step 1: Define canonical names and schema artifact

Preferred names:

```text
defender_center_3d
defender_keypoints_3d_flat
defender_keypoints_visible_logits
y_defender_center_3d
y_defender_keypoints_3d_flat
y_defender_keypoints_visible
defender_keypoint_schema_version
defender_keypoint_schema_hash
```

Create:

```text
schemas/defender_keypoint_schema.json
```

The schema should include:

```text
schema_version
coordinate_frame
origin_definition
units
keypoints[10]
wireframe_edges
inclusion_exclusion_rules
measurement_notes
```

Optional compatibility aliases:

```text
defender_points_3d_flat -> defender_keypoints_3d_flat
y_defender_points_3d_flat -> y_defender_keypoints_3d_flat
```

### Step 2: Add topology family

Create:

```text
03_rb-training-v2.0/src/topologies/topology_defender_amodal_keypoint_pose.py
03_rb-training-v2.0/src/topologies/topology_defender_amodal_keypoint_pose_v0_4.py
```

Register it in:

```text
03_rb-training-v2.0/src/topologies/registry.py
```

Use:

```text
TOPOLOGY_ID = "defender_amodal_keypoint_pose"
MODEL_CLASS_NAME = "DefenderAmodalKeypointPoseRegressor"
DEFAULT_VARIANT = "defender_amodal_keypoint_pose_v0_4"
```

If `v0_2` topology files already exist, keep them as aliases or transitional variants rather than breaking working code.

### Step 3: Reuse tri-stream encoders

Build v0.3 by copying or factoring from the existing tri-stream yaw topology.

Input keys:

```text
x_distance_image
x_orientation_image
x_geometry
```

Output mapping:

```python
{
    "distance_m": distance,
    "yaw_sin_cos": yaw_sin_cos,
    "defender_center_3d": center_3d,
    "defender_keypoints_3d_flat": keypoints_3d_flat,
    "defender_keypoints_visible_logits": keypoints_visible_logits,
}
```

### Step 4: Add contract support

Add topology contract entries for:

```text
defender_center_3d
defender_keypoints_3d_flat
defender_keypoints_visible_logits
defender_keypoint_schema_version
```

If arbitrary vector-regression heads are not already supported, extend them minimally and add tests.

The first implementation may treat the keypoint output as generic vector regression with metadata rather than introducing a specialised keypoint kind immediately.

### Step 5: Add loss support

Add loss handling for:

```text
center_3d_loss
keypoint_3d_loss
keypoint_visibility_loss
```

Recommended first-pass behaviour:

```text
SmoothL1 / Huber over centre vector
SmoothL1 / Huber over flattened 30D keypoint vector
BCE-with-logits over 10 visibility labels
mean reduction, normalised by vector width/keypoint count
raw and weighted component losses logged separately
```

### Step 6: Add metrics

Add evaluation metrics:

```text
center_mean_error_m
center_median_error_m
center_p95_error_m
keypoint_mean_point_error_m
keypoint_median_point_error_m
keypoint_p95_point_error_m
visible_keypoint_mean_error_m
hidden_keypoint_mean_error_m
keypoint_visibility_f1
```

Keypoint error should reshape `(B, 30)` into `(B, 10, 3)` and compute Euclidean point error.

### Step 7: Add dataset/NPZ validation

Ensure dataset loading validates presence and shape of:

```text
y_defender_center_3d: shape (3,)
y_defender_keypoints_3d_flat: shape (30,)
y_defender_keypoints_visible: shape (10,)
defender_keypoint_schema_version: present in shard/dataset metadata
```

The new topology should fail clearly if required labels or schema metadata are missing.

### Step 8: Add tests

Add focused tests for:

```text
registry lists defender_amodal_keypoint_pose
default variant builds
forward pass returns expected output keys
output shapes are correct
topology contract canonicalises
schema version is required and preserved
legacy point/pointcloud aliases resolve if supported
loss computation accepts vector targets and visibility targets
keypoint loss is mean-normalised rather than summed over 30 coordinates
metrics computation accepts vector outputs
missing target keys fail clearly
missing/mismatched schema version fails clearly
existing tri-stream yaw topology tests still pass
```

### Step 9: Live inference compatibility check

Verify whether `torch_tri_stream_engine.py` accepts mapping outputs with extra keys.

If it rejects extra keys, change it so extended models are accepted when they provide at least:

```text
distance_m
yaw_sin_cos
```

and preserve additional keys in result extras/debug metadata.

### Step 10: Documentation

Add a technical note under a repo documentation path such as:

```text
documents/defender_amodal_keypoint_pose_topology.md
```

It should explain:

```text
what the topology predicts
why all ten keypoints are predicted
why hidden keypoints are valid amodal targets
where the pose-ambiguity boundary is
how it differs from direct distance/yaw regression
what labels it needs
what metrics it reports
what is intentionally not implemented yet
```

### Step 11: Geometry-only ablation

Train or evaluate at least one baseline that uses `x_geometry` only, with the image streams removed, zeroed, or replaced by constants.

Report:

```text
geometry_only_distance_metrics
geometry_only_yaw_metrics
geometry_only_center_metrics_if_applicable
geometry_only_keypoint_metrics_if_applicable
full_model_minus_geometry_only_delta
```

If the full model is not materially better than the geometry-only baseline, the image streams are not yet proven to contribute meaningful signal.

## 20. Acceptance Criteria

The first implementation is acceptable when:

1. The new topology family is registered and selectable.
2. A canonical `defender_keypoint_schema.json` exists.
3. The schema version is recorded in training data metadata and model/evaluation artifacts.
4. A synthetic batch can be passed through the model.
5. The model outputs distance, yaw, centre, flattened keypoints, and visibility logits.
6. Training loss includes distance, yaw, centre, keypoint, and visibility components.
7. Keypoint loss is mean-normalised across keypoints and coordinates rather than summed across all 30 coordinates.
8. Raw and weighted loss components are visible in training logs.
9. Evaluation reports centre and keypoint error metrics.
10. Evaluation reports visible-vs-hidden keypoint metrics.
11. Missing keypoint labels fail with a clear error.
12. Missing or mismatched keypoint schema metadata fails with a clear error.
13. Existing tri-stream yaw topology tests still pass.
14. Existing live v0.3 distance/yaw models remain compatible.
15. The live engine can either ignore or preserve extra keypoint outputs without breaking distance/yaw inference.
16. A geometry-only ablation result exists before any external claim that the image streams contribute meaningful signal beyond ROI geometry. If the ablation is not complete, the topology may remain an internal experiment, but it should not be used as external evidence for image-based geometric understanding.
17. GUI overlays are not required for the first implementation milestone.
18. The documentation is understandable without reading v0.1 or v0.2.

## 21. Future Work

After the first v0.4 implementation works:

1. Refine the canonical keypoint schema after measuring the physical Defender.
2. Add or refine keypoint wireframe edge schema.
3. Add pairwise distance loss.
4. Add fixed-scale rigid-fit post-processing.
5. Add projection/reprojection diagnostics.
6. Add keypoint trace artifacts if not completed in first pass.
7. Write a separate diagnostic visualisation note for wireframe overlays.
8. Add wireframe overlay debug images.
9. Add direct-vs-derived pose disagreement reporting.
10. Add learned uncertainty head if ambiguity is a measured issue.
11. Build a full real-world validation rig if minimal validation suggests the branch is worth extending.
12. Compare direct regression vs visible-only keypoints vs amodal keypoints vs rigidified amodal keypoints.
13. Compare full model vs geometry-only and image-only ablations.
14. Evaluate synthetic-to-real transfer separately from synthetic validation.

## 22. Portfolio / External Positioning

This branch should be described as a structured representation experiment inside a bounded perception system.

Good phrasing:

```text
This topology investigates whether monocular image evidence can be compressed into a physically meaningful 3D hypothesis for a known rigid vehicle, rather than predicting distance and orientation only as final scalar outputs.
```

Also good:

```text
The model is trained to emit an inspectable amodal keypoint state. That state can be checked against known object constraints, projected back into the image, and used to diagnose failures in a way that direct scalar regression cannot easily support.
```

Avoid describing it as:

```text
general 3D reconstruction
general object detection
autonomous driving perception
photogrammetry replacement
```

The strongest engineering story is not that the model is more exotic. It is that the representation is more testable, inspectable, constrained, and diagnostically useful.

For employer-facing use, this full specification should not be the first artifact handed over. Use a two-page technical summary first, then offer this document as the deeper implementation specification if the reader asks for details.

---

## Appendix A. References and Related Work

These references are included to position the design. They are not claims that Raccoon Ball implements the same methods.

1. Abhishek Kar, Shubham Tulsiani, Joao Carreira, Jitendra Malik. "Amodal Completion and Size Constancy in Natural Scenes." arXiv:1509.08147.  
   https://arxiv.org/abs/1509.08147

2. Zhile Ren Deng, Sinisa Todorovic, Longin Jan Latecki. "Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes from 2D Ones in RGB-Depth Images." CVPR 2017.  
   https://openaccess.thecvf.com/content_cvpr_2017/html/Deng_Amodal_Detection_of_CVPR_2017_paper.html

3. Bugra Tekin, Sudipta N. Sinha, Pascal Fua. "Real-Time Seamless Single Shot 6D Object Pose Prediction." arXiv:1711.08848.  
   https://arxiv.org/abs/1711.08848

4. Sida Peng, Yuan Liu, Qixing Huang, Hujun Bao, Xiaowei Zhou. "PVNet: Pixel-wise Voting Network for 6DoF Pose Estimation." arXiv:1812.11788.  
   https://arxiv.org/abs/1812.11788

5. Fabian Manhardt, Diego Martin Arroyo, Christian Rupprecht, Benjamin Busam, Tolga Birdal, Nassir Navab, Federico Tombari. "Explaining the Ambiguity of Object Detection and 6D Pose From Visual Data." ICCV 2019 / arXiv:1812.00287.  
   https://arxiv.org/abs/1812.00287
