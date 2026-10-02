CV Preprocessing Experiments
1. Objective

The objective of this experiment is to systematically compare different image preprocessing pipelines for the handwritten prescription dataset.

The main principle of this work is:

Preprocessing techniques should not be assumed to improve recognition. Their effects must be measured experimentally.

The experiments therefore evaluate the visual and numerical effects of different preprocessing operations before any final preprocessing pipeline is selected for recognition.

2. Dataset

The experiments use the Doctor's Handwritten Prescription BD dataset.

Property	Value
Total images	4,680
Medicine classes	78
Generic medicine classes	15
Training images	3,120
Validation images	780
Testing images	780

The existing dataset split was retained throughout the experiments.

3. Image Size

All preprocessing pipelines produce a standardized spatial size of:

Height = 64
Width  = 256

The resizing operation preserves the original aspect ratio and uses padding where necessary.

4. Preprocessing Pipelines

Six preprocessing pipelines were evaluated.

P0 — RGB Baseline
RGB
  ↓
Resize + Padding
  ↓
Tensor

Output:

Channels = 3
Shape = [3, 64, 256]

P0 serves as the raw RGB baseline against which the other pipelines can be compared.

P1 — Grayscale
RGB
  ↓
Grayscale
  ↓
Resize + Padding
  ↓
Tensor

Output:

Channels = 1
Shape = [1, 64, 256]

The purpose is to determine whether color information is necessary for this handwritten-text recognition task.

P2 — Grayscale + Median Denoising
RGB
  ↓
Grayscale
  ↓
Median Filter
  ↓
Resize + Padding
  ↓
Tensor

Median filtering uses a 3 × 3 neighborhood.

Output:

Channels = 1
Shape = [1, 64, 256]

The purpose is to reduce high-frequency noise while attempting to preserve handwriting structure.

P3 — Grayscale + Denoising + CLAHE
RGB
  ↓
Grayscale
  ↓
Median Filter
  ↓
CLAHE
  ↓
Resize + Padding
  ↓
Tensor

CLAHE parameters:

clipLimit = 2.0
tileGridSize = (8, 8)

Output:

Channels = 1
Shape = [1, 64, 256]

The purpose is to investigate whether local contrast enhancement improves the visibility of faint handwriting.

P4 — Grayscale + Denoising + Otsu Thresholding
RGB
  ↓
Grayscale
  ↓
Median Filter
  ↓
Otsu Threshold
  ↓
Resize + Padding
  ↓
Tensor

Output:

Channels = 1
Shape = [1, 64, 256]

The purpose is to investigate whether converting handwriting into a predominantly binary representation improves separation between foreground strokes and the background.

P5 — Grayscale + Denoising + Deskew
RGB
  ↓
Grayscale
  ↓
Median Filter
  ↓
Deskew
  ↓
Resize + Padding
  ↓
Tensor

Output:

Channels = 1
Shape = [1, 64, 256]

The purpose is to investigate whether correcting the orientation of handwriting improves geometric consistency.

5. Numerical Experiment

All six pipelines were applied to the complete dataset.

Total preprocessing operations:

4,680 images × 6 pipelines = 28,080 operations

The following aggregate statistics were obtained.

Pipeline	Channels	Height	Width	Mean	Std	Foreground Ratio	Time / Image (ms)
P0 RGB Baseline	3	64	256	0.903940	0.228167	0.151162	0.214
P1 Grayscale	1	64	256	0.903921	0.228140	0.151288	0.098
P2 Grayscale + Denoise	1	64	256	0.911515	0.211289	0.146845	0.349
P3 Grayscale + Denoise + CLAHE	1	64	256	0.914468	0.205302	0.146697	0.412
P4 Grayscale + Denoise + Otsu	1	64	256	0.905147	0.256367	0.118608	0.370
P5 Grayscale + Denoise + Deskew	1	64	256	0.929533	0.190345	0.116893	0.622
Interpretation

The numerical measurements describe how the preprocessing operations change the image representation. They do not by themselves establish which pipeline provides better recognition.

P0 and P1 have almost identical aggregate mean and standard deviation values, showing that grayscale conversion does not drastically change these aggregate intensity statistics for this dataset.

P2 reduces the standard deviation and foreground ratio, which is consistent with smoothing and removal of some high-frequency variation.

P3 produces a further change in the intensity statistics while keeping the foreground ratio close to P2. This is consistent with local contrast modification.

P4 produces a noticeably lower foreground ratio and higher standard deviation. This is consistent with the stronger black/white separation produced by thresholding.

P5 produces the largest change in the aggregate statistics. This is expected because deskewing changes image geometry, introduces rotation, and requires interpolation and padding.

These measurements demonstrate that the pipelines produce different image representations, but they do not demonstrate a recognition advantage.

6. Visual Experiment

Controlled visual comparisons were generated for:

Training
Validation
Testing

Each comparison uses the same source image across P0–P5 so that the effect of each preprocessing operation can be visually inspected.

Generated files:

CV/preprocessing_experiments/train_comparison.png
CV/preprocessing_experiments/validation_comparison.png
CV/preprocessing_experiments/test_comparison.png
7. Visual Findings
P0 → P1: RGB to Grayscale

The inspected samples show that grayscale conversion preserves the major handwriting structure and stroke geometry.

Observed:

handwriting remains clearly visible
stroke connectivity is generally preserved
major curves and loops remain intact
no obvious structural degradation was observed in the inspected samples
the representation is reduced from three channels to one

However, grayscale conversion is still a mathematical transformation of the original RGB image. Therefore, the observation should be stated as:

No visually observable structural degradation was found in the inspected samples.

It should not be described as absolute pixel-level information preservation.

P1 → P2: Median Denoising

Median filtering produces mild smoothing.

Observed:

high-frequency image granularity is reduced
handwriting structure is generally preserved
stroke geometry remains stable in inspected samples
very fine image texture can become slightly smoother

The main benefit observed is noise reduction.

The main potential drawback is that excessive smoothing could remove fine handwriting details.

P2 → P3: CLAHE

CLAHE modifies local contrast.

Observed:

faint handwriting becomes more visually prominent in several samples
stroke edges can appear stronger
local contrast is increased
some faint background regions or paper patches also become more visible
localized contrast artifacts can appear

Therefore, CLAHE can improve visibility of weak strokes but may also amplify unwanted background variation.

P2 → P4: Otsu Thresholding

Otsu produces a strongly binarized representation.

Observed:

background regions become predominantly white
low-contrast background variation is greatly reduced
handwriting becomes strongly separated from the background
edges become more pixelated
thin or faint stroke segments can become narrower
delicate connections may show small gaps
heavy intersections and loops can become visually merged or filled
grayscale and anti-aliased intensity information is removed

Therefore, thresholding produces strong foreground/background separation but introduces a greater risk of losing subtle stroke information.

P2 → P5: Deskew

Deskewing changes the orientation of the handwriting.

Observed:

tilted baselines are corrected in inspected samples
handwriting becomes more geometrically aligned
stroke structure is generally preserved
rotation introduces interpolation effects
rotated images can contain white triangular/corner padding regions
slight resampling blur may occur

Deskewing therefore addresses geometric variation but introduces additional image transformation artifacts.

8. Processing-Time Observation

The approximate preprocessing times measured per image were:

P1  ≈ 0.098 ms
P0  ≈ 0.214 ms
P2  ≈ 0.349 ms
P4  ≈ 0.370 ms
P3  ≈ 0.412 ms
P5  ≈ 0.622 ms

These measurements represent preprocessing time only.

They should not be interpreted as complete model inference or training time.

The measurements indicate that more complex transformations generally require additional preprocessing computation, with deskewing being the most computationally expensive of the tested pipelines.

9. Summary of Observed Effects
Pipeline	Main observed benefit	Main observed concern
P0	Preserves original RGB representation	Uses 3 channels
P1	Reduces representation to one channel while preserving visual structure in inspected samples	Removes color information
P2	Reduces high-frequency noise	Mild smoothing
P3	Improves visibility of faint strokes	Can amplify background variation
P4	Strong foreground/background separation	Can remove subtle grayscale information and alter fine strokes
P5	Corrects handwriting orientation	Rotation/interpolation and corner padding artifacts
10. Current Conclusion

The experiments demonstrate that each preprocessing technique has measurable effects on the image representation.

However, preprocessing statistics and visual quality alone are insufficient to determine which pipeline produces the best handwriting recognition.

Therefore:

No final preprocessing pipeline is selected at this stage.

The final selection should be based on recognition experiments using the ML model while keeping the comparison controlled.

The preprocessing experiments completed by Member 2 therefore provide a set of documented candidate pipelines for downstream recognition evaluation.

11. Next Stage

The next stage is ML-based evaluation.

The ML teammate should evaluate the candidate preprocessing pipelines using the same recognition task and controlled experimental conditions.

The recognition results should determine whether the observed visual improvements translate into improved recognition performance.

Until those experiments are completed, all P0–P5 pipelines remain experimental candidates.