# Raccoon Ball Redux v0.3

## Monocular Temporal Geometry Inference for a Known Vehicle Instance

## 1. Project summary

This project investigates whether a **fixed camera** observing a **known vehicle instance** over time can infer useful vehicle state from a **temporally stacked geometric representation** derived from 2D image observations.

The system is intentionally constrained:

- one fixed camera
    
- one known vehicle instance
    
- bounded scene
    
- constrained motion space
    
- staged outputs
    
- synthetic-first training with limited real-world validation
    

The goal is **not** to solve general monocular 3D perception.

The goal is to test whether, under constrained conditions, **temporal stacking of geometric observations provides useful information beyond trivial geometric heuristics and beyond single-frame learned baselines**.

---

## 2. Core research question

Can a fixed-camera system observing a known vehicle instance in a constrained scene infer **distance band** and/or **direction of travel** from temporally stacked 2D geometric slices, and does this approach outperform simple heuristic baselines and single-frame learned baselines?

### Secondary question

Can a model trained primarily on synthetic data transfer to real webcam footage when scene geometry, object instance, and motion regime are aligned?

---

## 3. Why this project exists

This project exists to turn the original Raccoon Ball idea into a **bounded, falsifiable, legible applied-research artifact**.

It is intended to produce:

- a real technical system
    
- a benchmarked experiment
    
- a visible demo
    
- a portfolio-quality proof of capability
    
- a test of whether the temporal-stacking idea has practical value in a constrained domain
    

It also exists to answer a harder question than “can a network learn something in simulation?”:

> Does temporal stacking provide meaningful value over simpler alternatives?

---

## 4. Non-claims

This project does **not** claim to solve:

- open-world monocular 3D vision
    
- arbitrary-object inference
    
- arbitrary-camera inference
    
- arbitrary-environment generalization
    
- robust vehicle tracking in unconstrained scenes
    
- broad vehicle-category generalization in the first experiment
    

This is a **constrained vehicle-state inference experiment**, not a universal vision system.

---

## 5. Stage structure

## Stage 1 — Direction of travel or distance bands

The first stage should prioritize the target most likely to produce a clean early signal **while still being strong enough to test whether temporal stacking adds value over simpler baselines**.

### Preferred first target

- **direction of travel classification**
    
    - toward
        
    - away
        
    - left-to-right
        
    - right-to-left
        

### Alternate first target

- **distance band classification**
    
    - near
        
    - mid
        
    - far
        
    - or similar calibrated bins
        

### Why classification first

Classification is more stable, easier to evaluate, easier to explain, and less vulnerable to noisy labels than early regression.

### Important caution

The first target should not be so easy that trivial heuristics or a single-frame network already solve it well enough to make temporal stacking irrelevant.

---

## 6. Later targets

Only after Stage 1 produces non-embarrassing results:

### Stage 2

- direction of travel
    
- distance bands
    

### Stage 3

- approximate speed band
    
- optional heading/orientation class
    

### Later only if justified

- continuous distance regression
    
- continuous speed regression
    
- camera-shift robustness
    
- sparse implementation
    

---

## 7. Domain and object choice

### Domain

Fixed-camera observation of a known vehicle moving through a constrained path.

### Preferred object

A boxy, visually asymmetric vehicle.

Reason:

- stronger edge cues
    
- stronger front/back distinction
    
- easier geometric interpretation
    
- better logistics/transport relevance
    

### Desired alignment

Choose a vehicle for which both can be obtained:

- a usable 3D model for synthetic generation
    
- a physical RC/scale version for real-world footage
    

This is important because synthetic-to-real transfer only matters if the two domains are meaningfully aligned.

### Scope for v0.2

The first experiment should use **one known vehicle instance**, not an entire family of vehicles. Broader generalization can be tested later if the first setup works.

---

## 8. Data strategy

## 8.1 Synthetic data

Synthetic data is expected to be the main training source.

### Why

- cheap
    
- fully labelled
    
- controllable
    
- good for ablations
    
- supports falsification under clean conditions
    

### Synthetic labels

At minimum:

- distance band or true distance
    
- direction of travel
    
- optional speed band
    
- clip/scenario metadata
    

### Variables to control

- camera height
    
- camera pitch/yaw
    
- field of view
    
- object scale
    
- path geometry
    
- lighting
    
- background simplicity/clutter
    
- motion speed
    
- temporal window length
    

## 8.2 Real data

Real data is used for:

- sanity checking
    
- demo value
    
- limited validation
    
- sim-to-real assessment
    

### Real setup

- one fixed webcam
    
- constrained path
    
- stable lighting initially
    
- known floor/background geometry
    
- repeated trajectories
    

---

## 9. Task formulation

## 9.1 Input

A temporal window of observations of the target vehicle.

Each sample consists of:

- a temporally stacked representation
    
- its label(s)
    
- metadata
    

## 9.2 Output

Depending on stage:

- direction class
    
- distance band
    
- later: speed band
    

---

## 10. Representation

## 10.1 Core concept

Each frame is converted into a simplified 2D representation of the vehicle.

Those 2D slices are stacked across time into a 3D tensor.

### Intended axes

- X = horizontal structure
    
- Y = vertical structure
    
- T = time
    

## 10.2 Candidate preprocessing forms

Candidate representations include:

- binary silhouette
    
- edge map
    
- contour/geometry mask
    
- sparse occupancy slice
    
- cropped but scale-preserving local view
    

### Critical requirement

Preprocessing must preserve the cues needed for the task.

For example:

- distance cues must not be normalized away
    
- motion cues must not be destroyed by over-aggressive cropping
    
- orientation cues must remain recoverable
    

---

## 11. Baselines

This section is mandatory.

The project is only interesting if it is compared against simpler alternatives.

## 11.1 Chance baseline

The trivial lower bound for classification.

## 11.2 Heuristic baselines

At least one or more of:

- bounding-box area or height as a distance heuristic
    
- centroid displacement across frames as a direction heuristic
    
- simple calibrated perspective rules
    
- optical-flow magnitude summary
    
- hand-engineered feature + shallow classifier baseline
    

## 11.3 Single-frame learned baseline

A 2D CNN using a single frame or single-slice representation.

### Why this matters

If the project claims that **temporal stacking** is useful, it must be compared against a model that only sees a single frame.

Otherwise, it is impossible to tell whether any gain comes from:

- temporal structure
    
- or just “a neural network on the object crop”
    

## 11.4 Dense learned baseline

A dense 3D CNN over the stacked representation.

### Key principle

The project does **not** begin by asking whether sparse methods are elegant.

It begins by asking whether the representation itself is useful.

---

## 12. Model strategy

## 12.1 First learned model

A dense 3D CNN baseline.

Reason:

- simpler tooling
    
- faster iteration
    
- fewer dependency traps
    
- answers the main question first
    

## 12.2 Later model

Only if the dense baseline shows meaningful value:

- compact/sparse representation experiments
    
- efficient variants
    
- lower-precision or sparse-processing branches
    

Sparse processing is a **second-stage optimization question**, not a first-stage belief system.

---

## 13. Evaluation protocol

## 13.1 Metrics

### For classification

- accuracy
    
- macro F1
    
- per-class precision/recall
    
- confusion matrix
    

### For regression, if later used

- MAE
    
- RMSE
    
- error by range band
    

## 13.2 Splits

The evaluation must not be sloppy.

Potential split schemes:

- train/val/test by clip
    
- held-out trajectories
    
- held-out lighting conditions
    
- held-out background variants
    
- held-out real footage
    

At least one split must test whether the model is learning more than a narrow memorization of a single path.

## 13.3 Required comparisons

Every learned approach must be compared against:

- chance
    
- heuristic baseline(s)
    
- single-frame learned baseline
    
- dense learned baseline
    

---

## 14. Ablation questions

This is where the project starts to look like real research.

At minimum, test some of the following:

- raw crop vs silhouette/edge representation
    
- short temporal window vs longer temporal window
    
- direction-only vs distance-only task
    
- dense tensor vs compact geometric representation
    
- synthetic-only vs synthetic plus limited real calibration
    
- simple background vs cluttered background
    
- scale-preserving crop vs more aggressive normalization
    

### Explicit temporal-stacking question

A core ablation question must be:

> Does the temporal stack improve performance beyond single-frame representation, and if so, on which targets?

The point is not to run infinite experiments.

The point is to answer:

> what is actually carrying the signal?

---

## 15. Sim-to-real question

This should be explicit.

### Question

Does a model trained mainly on synthetic temporally stacked vehicle observations remain useful on real webcam footage when camera placement, vehicle instance, and motion regime are aligned?

### Minimum acceptable outcome

The real-world result does not need to be production-ready.

It only needs to be:

- visibly non-random
    
- better than chance
    
- not obviously fake or brittle under the initial constrained setup
    

---

## 16. Success criteria

## Minimum success

- full synthetic generation pipeline exists
    
- preprocessing pipeline exists
    
- dense baseline trains successfully
    
- model beats chance on at least one target
    
- model is benchmarked against heuristic baseline(s)
    
- model is benchmarked against a single-frame learned baseline
    
- real-world demo produces plausible outputs under constrained conditions
    

## Strong success

- system meaningfully beats heuristic baseline(s)
    
- system beats the single-frame learned baseline on at least one target
    
- direction classification is reliable
    
- distance-band classification is credible
    
- synthetic-to-real transfer is non-embarrassing
    
- artifact is clean enough to show another person without apology
    

## Stretch success

- multiple ablations completed
    
- representation choice is justified empirically
    
- sparse/efficient branch becomes worth testing
    
- the demo clearly suggests relevance to transport, yard, depot, or fixed-camera monitoring tasks
    

---

## 17. Failure criteria

This branch should be considered falsified or heavily weakened if:

- the learned system fails to beat trivial heuristics
    
- the learned system fails to beat the single-frame learned baseline
    
- the temporal stack carries no useful signal beyond single-frame cues
    
- the model only works in an over-clean synthetic regime and collapses immediately on aligned real footage
    
- the complexity of the representation is not justified by the gain
    
- the project becomes dependent on fragile story-telling rather than measurable advantage
    

Clean falsification is a valid outcome.

---

## 18. Artifact goals

This project must produce visible artifacts.

Required artifacts:

- synthetic generation pipeline
    
- preprocessing code
    
- training/evaluation code
    
- baseline comparison outputs
    
- result plots
    
- confusion matrices or error summaries
    
- short write-up
    
- real-world demo video
    

### Important principle

The video is not enough.

The project cuts through when the video is backed by:

- metrics
    
- baselines
    
- ablations
    
- failure analysis
    

### Demo value vs research value

These should be treated separately.

- **Demo value** = looks real, compelling, legible
    
- **Research value** = measurably beats simpler alternatives
    

A strong project should aim for both.

---

## 19. Portfolio value

The project becomes a strong portfolio artifact if it clearly demonstrates:

- end-to-end ownership of an ML pipeline
    
- good experimental hygiene
    
- ability to define and beat baselines
    
- practical handling of sim-to-real issues
    
- ability to scope a hard problem into a falsifiable bounded system
    
- discipline in making explicit non-claims
    

---

## 20. Immediate next steps

### Technical

1. choose the first target
    
    - preferably direction classification or distance-band classification
        
2. choose the vehicle
    
3. lock the camera geometry
    
4. define the constrained path set
    
5. implement synthetic generation
    
6. define at least one heuristic baseline
    
7. define the first stacked representation
    
8. define the single-frame learned baseline
    
9. train the first dense 3D CNN baseline
    
10. compare against heuristic and single-frame learned baselines
    
11. record first real demo clips
    

### Strategic

1. keep industrial sound anomaly work alive as a separate applied-ML track
    
2. treat this vehicle branch as Raccoon Ball proper
    
3. do not let sparse tooling derail baseline establishment
    
4. do not broaden the claim
    
5. kill the branch early if it cannot beat trivial heuristics or single-frame learned baselines
    

---

## 21. Strategic position

This project is worth pursuing if it remains:

- bounded
    
- benchmarked
    
- visual
    
- falsifiable
    
- legible to another person in under two minutes
    

Its job is not to solve everything.

Its job is to become:

- a serious experiment
    
- a compelling demo
    
- a defensible portfolio artifact
    
- proof that the underlying idea can survive contact with baselines
    

---