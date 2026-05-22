# Defender Amodal Keypoint Pose Regressor - Technical Summary v0.4

**Project:** Raccoon Ball  
**Object:** 1/12 scale Defender  
**Artifact type:** employer-facing technical summary  
**Companion document:** `Defender_Keypoint_Regression_Topology_v0.4.md`

---

## 1. Problem

Raccoon Ball is a bounded monocular perception project. The system observes a known 1/12 scale Defender model from a fixed or controlled camera setup and estimates vehicle distance and yaw/orientation.

The current live path predicts distance and yaw directly from tri-stream image-derived inputs:

```text
x_distance_image + x_orientation_image + x_geometry -> distance + yaw
```

That direct scalar regression path is compact and operationally useful, but it hides the geometric state the model has inferred. When a prediction fails, the system can inspect camera frames, crops, masks, preprocessing artifacts, and model outputs, but it cannot directly inspect whether the model misunderstood object scale, pose, extent, occlusion, or the relationship between those elements.

The proposed v0.4 topology adds a more inspectable intermediate representation: **amodal semantic keypoint regression**.

---

## 2. Proposal

Instead of predicting only final distance and yaw, the model should emit a structured object-state hypothesis:

```text
tri-stream image-derived inputs
  -> Defender centre in camera-space coordinates
  -> all fixed semantic external Defender keypoints, including occluded keypoints
  -> keypoint visibility / in-frame state
  -> direct distance and yaw heads for compatibility
  -> optional rigid fit to known Defender geometry
  -> derived distance/yaw and diagnostic residuals
```

The key design choice is to predict **all ten fixed external keypoints**, not only the currently visible ones. Hidden keypoints are not treated as visually detected. They are treated as amodal inferred targets derived from known object geometry, camera setup, visible evidence, ROI geometry, and the synthetic training distribution.

The representation deliberately separates three concepts:

```text
amodal target: where the fixed keypoint is in 3D
visibility: whether that keypoint is directly visible in the image
confidence / uncertainty: how reliable the prediction is likely to be
```

That split matters because a visible point may be uncertain due to blur or bad preprocessing, while a hidden point may be well constrained if the object pose is clear.

---

## 3. Why this is useful

Direct distance/yaw regression gives the system an answer, but not much explanation. Amodal keypoints give the system an inspectable object hypothesis.

The predicted keypoints can be checked against the known Defender geometry. If the model predicts an impossible shape, a rigid-fit residual can expose that failure. If hidden keypoints are much worse than visible keypoints, the evaluation can show that directly. If the keypoint-derived pose disagrees with the direct distance/yaw heads, that disagreement becomes a diagnostic signal rather than an invisible internal failure.

The engineering claim is narrow:

```text
For a known rigid object in a bounded monocular scene, fixed semantic keypoints are a useful supervised representation for pose, distance, and failure diagnostics.
```

The non-claims are just as important:

```text
This is not general 3D reconstruction.
This is not general object detection.
This is not autonomous-driving perception.
This is not a claim that arbitrary hidden geometry can be recovered from arbitrary monocular images.
```

---

## 4. Key design decisions

### Amodal keypoints, not a point cloud

The model is not predicting an unordered surface sample. It predicts a fixed ordered set of semantic landmarks. Keypoint identity matters: keypoint 0 must always mean the same physical datum.

### Canonical schema before training

The ten keypoints must be defined in `defender_keypoint_schema.json` before the first training run. The schema must specify local coordinates, coordinate frame, measurement method, visibility rules, and schema version.

The final two reference points are provisional in the design document only. They must be selected and frozen before training. Changing them after training begins is a schema version change and makes previous model artifacts incompatible with the new schema.

### Preserve direct distance/yaw heads

The new model should still output `distance_m` and `yaw_sin_cos`. That preserves compatibility with the existing live runtime and gives a direct comparison against current scalar-regression models.

### Keep the architecture simple first

The first implementation should reuse the existing tri-stream encoders and fusion trunk. The keypoint head should initially be a single linear projection or a shallow one-hidden-layer MLP. More complex geometry modules should only be added after measured failures justify them.

### Normalise the keypoint loss

A 30-coordinate keypoint output can dominate scalar distance and 2D yaw losses if losses are simply summed. The keypoint loss should be mean-normalised across keypoints and coordinates, and raw/weighted component magnitudes should be logged.

### Require geometry-only ablation before external claims

The ROI geometry vector is a strong cue in a fixed-camera setup. The model may appear successful while relying mainly on bounding-box position and size rather than image evidence. A geometry-only baseline is required before claiming that the image streams contribute meaningful geometric understanding.

---

## 5. Main risks

### Ambiguous views

Direct front, rear, side, or near-symmetric views may support multiple plausible poses. Under a single-output regression loss, the model may average between valid hypotheses and produce a physically invalid keypoint configuration. This must be measured separately rather than hidden inside a mean error value.

### Hidden-keypoint uncertainty

Hidden keypoints should be predicted and evaluated, but not treated as equally certain in every view. Visible and hidden keypoint metrics must be reported separately, and hidden-keypoint error should be checked against yaw ambiguity.

### Synthetic-to-real transfer

The current evidence base is synthetic-heavy. Real-world validation should start minimally: fixed camera, known distances and angles, a few representative frames, and manual sanity checks. A full calibrated rig is useful later, but should not block first-pass validation.

### Image-stream bypass

If the full model does not substantially outperform a geometry-only baseline, then the image encoders have not yet been shown to contribute meaningful signal. In that case the topology remains an internal experiment, not an external evidence claim.

---

## 6. First implementation milestone

The first implementation is acceptable when:

1. The topology family is registered and selectable.
2. `defender_keypoint_schema.json` exists and is versioned.
3. A synthetic batch runs through the model.
4. The model outputs distance, yaw, centre, flattened 3D keypoints, and visibility logits.
5. Training loss includes distance, yaw, centre, keypoint, and visibility components.
6. Evaluation reports centre/keypoint metrics and visible-vs-hidden keypoint metrics.
7. Missing labels or schema metadata fail clearly.
8. Existing live v0.3 distance/yaw models remain compatible.
9. Extra keypoint outputs do not break live distance/yaw inference.
10. External claims about image-stream contribution are blocked until the geometry-only ablation exists.

GUI wireframe overlays, differentiable rigid fitting, a full real-world validation rig, and learned uncertainty heads are useful later work, not first-milestone blockers.

---

## 7. Portfolio framing

The strongest portfolio framing is:

```text
This topology investigates whether monocular image evidence can be compressed into a physically meaningful 3D hypothesis for a known rigid vehicle, rather than predicting distance and orientation only as final scalar outputs.
```

The value is not architectural novelty for its own sake. The value is that the representation is constrained, testable, inspectable, and useful for diagnosing composed perception-system failures.
