Yes — you’re right. The previous version still had too much “internal scratchpad” texture. Here’s a cleaner employer-facing version: professional, evidence-led, and suitable for a repository report or portfolio appendix.

---

# Incident Report: Camera-Model Alignment and Pose-Dependent Distance Bias in Live Distance Regression

## Executive Summary

This investigation examined whether live-camera distance prediction errors in Project Raccoon Ball were being caused by a mismatch between the real AR0234 camera model and the Unity synthetic camera model used during training.

The system was evaluated using three physical distance sweeps:

1. **Baseline sweep A** before camera-model correction.
2. **Baseline sweep B** before camera-model correction, repeated after several hours and a mask redraw.
3. **Corrected sweep C** after applying a camera-model delta transform before inference.

The camera-model correction used OpenCV/ChArUco calibration data for the real AR0234 camera and an equivalent extracted calibration model from the Unity camera. The correction was applied upstream of inference, transforming the input image before localisation, preprocessing, geometry extraction, and model prediction.

The correction produced a modest aggregate improvement in distance error:

| Metric                | Baseline A | Baseline B |  Corrected C |
| --------------------- | ---------: | ---------: | -----------: |
| Mean absolute error   |   0.1275 m |   0.1267 m | **0.1058 m** |
| RMSE                  |   0.1552 m |   0.1567 m | **0.1394 m** |
| Median absolute error |   0.1000 m |   0.1200 m | **0.0750 m** |

However, the main structural failure remained. At the same measured floor distance, the predicted distance continued to vary substantially depending on whether the vehicle faced front, side, or rear. In all three sweeps, front-facing views generally predicted farther away than rear-facing views.

The current conclusion is that camera-model mismatch is likely a contributing factor, but not the dominant remaining failure mode. The stronger remaining issue appears to be pose-dependent representation error: foreground geometry, bounding-box features, silhouette/appearance differences, or learned correlations between pose and distance.

---

## 1. System Context

Project Raccoon Ball is a bounded monocular computer-vision system for estimating the distance and yaw of a known vehicle from a fixed camera view.

The live inference pipeline is composed of several stages:

```text
real camera frame
  -> optional camera-model correction
  -> target localisation
  -> ROI extraction
  -> foreground / geometry preprocessing
  -> tri-stream model inputs
  -> PyTorch distance/yaw regression
```

The model was trained primarily on Unity-generated synthetic imagery. The live runtime uses a real AR0234 camera. This creates a plausible source of error: the real camera may not project the vehicle into image space in exactly the same way as the synthetic Unity camera.

Because the model uses apparent scale, foreground geometry, and image-space features as evidence for distance, camera-model mismatch can affect prediction quality even when the rest of the pipeline is functioning correctly.

---

## 2. Investigation Objective

The investigation tested whether applying a real-camera-to-Unity-camera image correction would improve live distance predictions.

The hypothesis was:

> The trained model expects Unity camera projection geometry. If the AR0234 real camera produces different image-space geometry, applying a camera-model delta transform before inference should make live frames more consistent with the model’s training distribution and improve distance regression.

This was explicitly **not** a post-processing correction to the predicted distance value. The intervention was applied to the image before the normal inference pipeline.

---

## 3. Measurement Method

The Defender model was placed manually on marked floor positions on a white hardboard surface. The floor markings were measured using a tape measure.

At each usable measured distance, three poses were tested:

```text
front-facing
side-facing
rear-facing
```

The measured positions were:

```text
1.59 m
1.77 m
1.97 m
2.18 m
```

A fifth mark at `2.39 m` was excluded because the Defender clipped at the top of the frame.

The distances should be understood as **measured reference distances**, not laboratory-grade ground truth. Manual placement, model footprint, and centre-of-volume ambiguity introduce some unavoidable tolerance. The model’s synthetic distance target corresponds to the Unity vehicle object position, not a directly visible point on the real model.

This limitation affects absolute accuracy, but it does not explain the main signal under investigation: distance should remain broadly pose-invariant at the same floor marking.

---

## 4. Expected Behaviour

At a fixed measured distance, the model’s predicted distance should be similar regardless of vehicle yaw.

For example, if the Defender is placed at the `1.77 m` mark, the predicted distance should remain close to `1.77 m` whether the vehicle is facing front, side, or rear.

Expected pattern:

```text
front prediction ≈ side prediction ≈ rear prediction
```

Observed pattern:

```text
front prediction > side prediction > rear prediction
```

This repeated pose-linked ordering is the core failure mode.

---

## 5. Data Captures

### 5.1 Session A — Baseline Sweep Before Camera-Model Correction

|   Sample | Measured mark | Pose  | Predicted distance |   Error |
| -------: | ------------: | ----- | -----------------: | ------: |
| LD-A-001 |        1.59 m | Front |             1.77 m | +0.18 m |
| LD-A-002 |        1.59 m | Side  |             1.69 m | +0.10 m |
| LD-A-003 |        1.59 m | Rear  |             1.64 m | +0.05 m |
| LD-A-004 |        1.77 m | Front |             2.00 m | +0.23 m |
| LD-A-005 |        1.77 m | Side  |             1.86 m | +0.09 m |
| LD-A-006 |        1.77 m | Rear  |             1.75 m | -0.02 m |
| LD-A-007 |        1.97 m | Front |             2.25 m | +0.28 m |
| LD-A-008 |        1.97 m | Side  |             2.00 m | +0.03 m |
| LD-A-009 |        1.97 m | Rear  |             1.90 m | -0.07 m |
| LD-A-010 |        2.18 m | Front |             2.08 m | -0.10 m |
| LD-A-011 |        2.18 m | Side  |             2.08 m | -0.10 m |
| LD-A-012 |        2.18 m | Rear  |             1.90 m | -0.28 m |

Session A showed substantial pose-linked spread. The largest spread occurred at the `1.97 m` mark:

```text
front: 2.25 m
side:  2.00 m
rear:  1.90 m
spread: 0.35 m
```

---

### 5.2 Session B — Baseline Repeat Before Camera-Model Correction

Session B repeated the baseline sweep several hours later. The mask was redrawn before the sweep.

|   Sample | Measured mark | Pose  | Predicted distance |   Error | Notes                                                               |
| -------: | ------------: | ----- | -----------------: | ------: | ------------------------------------------------------------------- |
| LD-B-001 |        1.59 m | Front |             1.80 m | +0.21 m |                                                                     |
| LD-B-002 |        1.59 m | Side  |             1.64 m | +0.05 m |                                                                     |
| LD-B-003 |        1.59 m | Rear  |             1.60 m | +0.01 m |                                                                     |
| LD-B-004 |        1.77 m | Front |             1.92 m | +0.15 m |                                                                     |
| LD-B-005 |        1.77 m | Side  |             1.86 m | +0.09 m | Prediction fluctuated between 1.83 m and 1.90 m; recorded as 1.86 m |
| LD-B-006 |        1.77 m | Rear  |             1.72 m | -0.05 m |                                                                     |
| LD-B-007 |        1.97 m | Front |             1.96 m | -0.01 m |                                                                     |
| LD-B-008 |        1.97 m | Side  |             1.90 m | -0.07 m |                                                                     |
| LD-B-009 |        1.97 m | Rear  |             1.78 m | -0.19 m |                                                                     |
| LD-B-010 |        2.18 m | Front |             2.00 m | -0.18 m |                                                                     |
| LD-B-011 |        2.18 m | Side  |             2.00 m | -0.18 m |                                                                     |
| LD-B-012 |        2.18 m | Rear  |             1.85 m | -0.33 m |                                                                     |

The exact values differed from Session A, which is expected given manual placement and mask redraw. However, the same structural pattern remained: front-facing views generally predicted farther than rear-facing views.

---

### 5.3 Session C — Corrected Sweep After Camera-Model Delta

Session C was collected after applying the camera-model delta transform before inference.

|   Sample | Measured mark | Pose  | Predicted distance |   Error |
| -------: | ------------: | ----- | -----------------: | ------: |
| LD-C-001 |        1.59 m | Front |             1.84 m | +0.25 m |
| LD-C-002 |        1.59 m | Side  |             1.63 m | +0.04 m |
| LD-C-003 |        1.59 m | Rear  |             1.60 m | +0.01 m |
| LD-C-004 |        1.77 m | Front |             1.99 m | +0.22 m |
| LD-C-005 |        1.77 m | Side  |             1.84 m | +0.07 m |
| LD-C-006 |        1.77 m | Rear  |             1.75 m | -0.02 m |
| LD-C-007 |        1.97 m | Front |             2.09 m | +0.12 m |
| LD-C-008 |        1.97 m | Side  |             2.00 m | +0.03 m |
| LD-C-009 |        1.97 m | Rear  |             1.85 m | -0.12 m |
| LD-C-010 |        2.18 m | Front |             2.15 m | -0.03 m |
| LD-C-011 |        2.18 m | Side  |             2.10 m | -0.08 m |
| LD-C-012 |        2.18 m | Rear  |             1.90 m | -0.28 m |

The corrected sweep improved aggregate distance error, but it did not remove pose sensitivity.

---

## 6. Aggregate Results

| Metric                 | Session A baseline | Session B repeat baseline | Session C corrected |
| ---------------------- | -----------------: | ------------------------: | ------------------: |
| Mean absolute error    |           0.1275 m |                  0.1267 m |        **0.1058 m** |
| Mean signed error      |          +0.0325 m |                 -0.0417 m |           +0.0175 m |
| RMSE                   |           0.1552 m |                  0.1567 m |        **0.1394 m** |
| Median absolute error  |           0.1000 m |                  0.1200 m |        **0.0750 m** |
| Maximum absolute error |           0.2800 m |                  0.3300 m |            0.2800 m |
| Samples within 10 cm   |             8 / 12 |                    6 / 12 |              7 / 12 |
| Samples within 5 cm    |             3 / 12 |                    4 / 12 |              5 / 12 |

The corrected session shows a modest improvement in average error and median error. This suggests that the camera-model delta may have moved the input distribution closer to the model’s expectations.

However, operationally, the corrected session remains outside the desired reliability envelope. Only 7 of 12 samples were within 10 cm, and the pose-dependent divergence remained substantial.

---

## 7. Pose-Dependent Error Analysis

### 7.1 Mean Error by Pose

| Session | Front mean error | Side mean error | Rear mean error |
| ------- | ---------------: | --------------: | --------------: |
| A       |        +0.1475 m |       +0.0300 m |       -0.0800 m |
| B       |        +0.0425 m |       -0.0275 m |       -0.1400 m |
| C       |        +0.1400 m |       +0.0150 m |       -0.1025 m |

Across all three sessions, front-facing views tended to over-predict distance, while rear-facing views tended to under-predict distance.

This is the strongest repeated finding in the investigation.

---

### 7.2 Pose Spread by Measured Mark

Pose spread is the difference between the highest and lowest prediction at the same measured distance.

| Session | 1.59 m spread | 1.77 m spread | 1.97 m spread | 2.18 m spread | Average spread |
| ------- | ------------: | ------------: | ------------: | ------------: | -------------: |
| A       |        0.13 m |        0.25 m |        0.35 m |        0.18 m |       0.2275 m |
| B       |        0.20 m |        0.20 m |        0.18 m |        0.15 m |       0.1825 m |
| C       |        0.24 m |        0.24 m |        0.24 m |        0.25 m |       0.2425 m |

The corrected session produced the highest average pose spread. This is significant because the camera-model correction was expected to reduce image-space distortion effects. Instead, the main pose-dependent structure remained and became more uniform.

The corrected sweep is especially structured:

```text
1.59 m spread: 0.24 m
1.77 m spread: 0.24 m
1.97 m spread: 0.24 m
2.18 m spread: 0.25 m
```

This suggests the remaining error is not random. The system appears to be applying a stable pose-dependent distance bias.

---

## 8. Interpretation

The camera-model correction appears to have helped the aggregate distance regression slightly, but it did not resolve the central failure mode.

The evidence supports three conclusions:

### 8.1 Camera-model mismatch is probably a contributing factor

The corrected sweep improved mean absolute error, RMSE, and median absolute error. This suggests that the camera correction changed the input image in a useful direction.

### 8.2 Camera-model mismatch is not the sole cause

The strongest repeated failure remained after correction. Distance predictions continued to vary by vehicle pose at the same measured floor marking.

If lens distortion or intrinsics mismatch were the primary cause, the corrected sweep would be expected to reduce the pose spread more clearly. It did not.

### 8.3 The dominant remaining issue is likely representation-level pose sensitivity

The model appears to be using pose-dependent visual or geometric cues as distance evidence.

Likely contributors include:

```text
- foreground bounding-box width and height changing by pose
- area_norm changing by pose
- aspect_ratio changing by pose
- front/rear appearance differences in the real Defender
- synthetic-to-real material or reflectance mismatch
- insufficient disentanglement between yaw and distance in training
- foreground extraction producing pose-dependent geometry
```

This does not imply that the model is behaving irrationally. It may be responding consistently to the features it receives. The next task is to determine whether the runtime representation is giving it pose-dependent geometry that correlates incorrectly with distance.

---

## 9. Technical Significance

This incident narrows the problem.

Before this investigation, it was plausible that a camera-model mismatch could be the main cause of live distance error. After applying the correction, the evidence suggests a more specific diagnosis:

> Camera alignment helps slightly, but the live pipeline still produces pose-dependent distance evidence.

That is a useful engineering result. It prevents time being spent on camera calibration alone when the stronger remaining problem is probably in representation, preprocessing, or training distribution.

The investigation also demonstrates several important engineering practices:

```text
- isolating a plausible failure source
- collecting repeat baseline measurements
- applying a targeted upstream correction
- comparing before/after behaviour using structured metrics
- distinguishing aggregate metric improvement from structural failure-mode resolution
- preserving measurement caveats without hiding useful signal
```

The most important learning is that aggregate error metrics alone are insufficient. The corrected session improved MAE, but the pose-invariance failure remained. For this system, pose spread at fixed measured distance is a more diagnostic metric than average distance error alone.

---

## 10. Limitations

This investigation should be read with the following constraints:

```text
- The measured distances are practical reference marks, not calibrated laboratory ground truth.
- The Defender was manually positioned by hand.
- The physical mark may not correspond exactly to the Unity object-position / centre-of-volume target.
- Each session contains only 12 usable samples.
- The mask was redrawn between some sessions.
- The 2.39 m mark was excluded because the vehicle clipped at the top of frame.
```

These limitations do not invalidate the result. They do mean the data should be interpreted as diagnostic evidence rather than final validation.

The key finding is not a precise centimetre-level claim. The key finding is the repeated pose-linked prediction structure.

---

## 11. Current Conclusion

The live distance regression system shows a repeatable pose-dependent bias at fixed measured floor positions. Front-facing views generally predict farther away; rear-facing views generally predict closer; side-facing views tend to sit between them.

Applying the AR0234-to-Unity camera-model delta before inference modestly improved aggregate distance error, reducing mean absolute error from approximately 12.7 cm in the two baseline sweeps to approximately 10.6 cm in the corrected sweep.

However, the correction did not reduce the primary structural failure. The corrected sweep retained a consistent front/side/rear prediction spread of approximately 24–25 cm across all four usable measured distances.

The current best interpretation is:

> Camera-model mismatch contributes to the live distance error, but the dominant remaining failure is pose-dependent representation error.

The next investigation should focus on comparing trace artifacts and geometry features across front, side, and rear poses at the same measured distance.

---

## 12. Recommended Next Step

The next analysis should inspect the actual model inputs, not just the predicted distances.

For one or two measured marks, preferably `1.77 m` and `1.97 m`, capture front, side, and rear traces with the camera-model correction enabled.

For each trace, compare:

```text
- accepted raw frame
- corrected frame
- locator bounding box
- ROI crop
- foreground mask
- foreground bounding box
- foreground area
- x_geometry vector
- x_distance_image
- x_orientation_image
- predicted distance
- predicted yaw
```

The most important fields to compare are the geometry features:

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

The next diagnostic question is:

> At the same measured distance, do front, side, and rear poses produce geometry features that differ in a way that explains the distance prediction bias?

If the answer is yes, the next remediation path should focus on representation design and training data, not further camera calibration.
