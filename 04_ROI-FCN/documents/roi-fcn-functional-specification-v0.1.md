# ROI FCN Functional Specification v0.1

## 1. Purpose

Create a neural-network-based ROI locator that takes a full-frame image and predicts the centre point around which a fixed `300 x 300` crop should be extracted.

The purpose of this component is to replace or supplement the current edge-based crop-placement step when operating on live full-frame imagery.

It is a **crop-centre localiser**, not a general object detector.

---

## 2. What it must do

Given a full-frame image containing the Defender, the component must:

1. process the full-frame image
2. infer the most likely crop centre for the Defender
3. return that centre in the coordinate system of the original full-frame image

The returned centre must be suitable for extracting a fixed `300 x 300` ROI that can then be passed into the existing downstream preprocessing and regression pipeline.

---

## 3. What it is not responsible for

This component is not responsible for:

* choosing ROI size
* predicting a bounding box width or height
* estimating distance
* estimating orientation
* extracting silhouettes
* performing downstream regression
* deciding overall system behaviour beyond ROI placement

Its responsibility ends at producing the crop centre.

---

## 4. Input definition

### External input

The component receives a single full-frame image.

The source image may be of arbitrary size.

For first-pass operation, the important semantic assumption is:

* the image contains one Defender of interest that should be localised for ROI placement

### Internal network input

The image must be converted into a fixed neural-network input representation so the model can be trained and run consistently.

That fixed representation should be:

* grayscale
* aspect-ratio preserved
* resized into a standard locator canvas
* padded as necessary to fit the standard canvas exactly
* normalized for network input

---

## 5. Necessary preprocessing

The preprocessing stage must do the minimum needed to make the locator network stable and consistent.

### Required preprocessing steps

1. convert the source image to grayscale
2. resize the image while preserving aspect ratio
3. place the resized image into a fixed locator canvas
4. pad any unused canvas area
5. normalize pixel values into a standard numeric range

### Important constraint

The preprocessing must retain the metadata needed to map a predicted centre point from network space back into original full-frame image coordinates.

That means the system must know, for each image:

* the resize scale used
* the padding offsets introduced

---

## 6. Neural-network input contract

The neural network must consume a fixed-size single-channel image tensor.

Semantically, that tensor represents:

> the entire full-frame source image, converted into a standard grayscale locator canvas

The network input must therefore be:

* single-channel
* fixed-size
* normalized
* spatially aligned with the stored resize/padding metadata

---

## 7. Core model behaviour

The model must behave as a **spatial localiser**.

Rather than outputting a bounding box, it must output a spatial estimate of where the crop centre should be.

The recommended first-pass formulation is:

* the model outputs a **centre-likelihood heatmap**
* brighter values indicate stronger belief that the crop centre should be placed there

This keeps the task spatial and avoids unnecessary box regression.

---

## 8. First-pass topology requirement

The model should be a **tiny FCN**.

FCN = **Fully Convolutional Network**

In this context, that means:

* the model operates convolutionally over the image
* the output remains spatial
* the network produces a map rather than a single class label

### Topology characteristics required for v0.1

The first-pass topology should be:

* small
* cheap
* single-stream
* single-channel input
* single-output-head
* heatmap-producing

It should use a small stack of convolution layers with downsampling, followed by a final head that produces a single output heatmap.

It should not, in v0.1, include:

* anchor machinery
* bounding box heads
* class heads
* multi-scale detector machinery
* orientation heads
* segmentation branches
* additional task coupling

The topology must stay focused on ROI centre localisation only.

---

## 9. Training target definition

The model is trained to reproduce the **desired crop centre**.

The correct training target for each image is:

> the centre point around which the fixed `300 x 300` ROI should be extracted

This target must be defined in original full-frame coordinates first.

For v0.1, the training corpus can be bootstrapped from the existing edge-based ROI placement system by using the crop centres that the current pipeline produces.

That means the FCN is initially trained to learn the current crop-placement behaviour.

---

## 10. Target representation for learning

Because the model outputs a spatial map, the point target must be transformed into a spatial supervision signal.

The recommended first-pass target representation is:

* a **Gaussian heatmap** centred on the correct crop centre location in the model’s output space

Semantically, this means:

* the highest value is at the correct centre
* nearby positions are somewhat correct
* farther positions are progressively less correct

This should be used instead of a single hard positive pixel, because it makes optimisation smoother and less brittle.

---

## 11. Output definition

There are two levels of output.

### Raw model output

The raw network output is a single heatmap representing crop-centre likelihood across the image.

### Final usable output

The system must decode that heatmap into:

* `center_x`
* `center_y`

in **original full-frame image coordinates**

This decoded point is the actual output contract of the component.

It is the point around which the fixed `300 x 300` ROI will be extracted.

---

## 12. Required post-processing

Post-processing must:

1. find the peak location in the output heatmap
2. map that peak from heatmap space back into the fixed input canvas space
3. undo the resize and padding transforms
4. return the resulting point in original full-frame coordinates

This mapping must be deterministic and consistent with the preprocessing metadata.

---

## 13. Behavioural requirement of the final output

The predicted centre does not need to be a philosophically perfect “true centre of the vehicle.”

It needs to be:

> the centre that places the fixed `300 x 300` ROI well enough for the downstream model to receive a useful crop

That is the real success criterion.

So this component should be judged primarily by ROI placement usefulness, not by abstract geometric purity.

---

## 14. Relationship to downstream robustness

This FCN should learn the cleanest available crop-centre target.

Deliberate jitter should **not** be part of the FCN target definition in v0.1.

If crop-position robustness training is needed, that belongs in the downstream distance/orientation regressor training, where the cropped ROI can be deliberately offset to simulate locator error.

That keeps the FCN’s job clean:

> predict the intended crop centre as accurately as practical

---

## 15. Summary in one sentence

The ROI FCN is a tiny fully convolutional centre localiser that takes a full-frame image, converts it into a fixed grayscale locator input, produces a centre-likelihood heatmap, and returns the predicted crop centre in original image coordinates for extraction of a fixed `300 x 300` ROI.

---

