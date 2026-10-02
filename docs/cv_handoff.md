Member 2 → Member 3 Preprocessing Handoff
1. Purpose

This document summarizes the computer-vision preprocessing work completed by Member 2 and defines the preprocessing inputs available for ML-based recognition experiments.

The goal is to provide reproducible preprocessing candidates without assuming that any particular preprocessing method improves recognition.

2. Dataset

The preprocessing experiments were performed on the complete dataset:

Total images:       4,680
Medicine classes:      78

Training:           3,120
Validation:           780
Testing:              780

The existing train/validation/test split should remain unchanged during ML evaluation.

3. Available Preprocessing Pipelines
P0 — RGB Baseline
RGB → Resize + Padding → Tensor

Tensor:

[3, 64, 256]
P1 — Grayscale
RGB → Grayscale → Resize + Padding → Tensor

Tensor:

[1, 64, 256]
P2 — Grayscale + Denoising
RGB → Grayscale → Median Filter → Resize + Padding → Tensor

Tensor:

[1, 64, 256]
P3 — Grayscale + Denoising + CLAHE
RGB → Grayscale → Median Filter → CLAHE → Resize + Padding → Tensor

Tensor:

[1, 64, 256]

CLAHE:

clipLimit = 2.0
tileGridSize = (8, 8)
P4 — Grayscale + Denoising + Otsu
RGB → Grayscale → Median Filter → Otsu → Resize + Padding → Tensor

Tensor:

[1, 64, 256]
P5 — Grayscale + Denoising + Deskew
RGB → Grayscale → Median Filter → Deskew → Resize + Padding → Tensor

Tensor:

[1, 64, 256]
4. Important CNN Input Difference

P0 produces three channels:

[3, 64, 256]

P1–P5 produce one channel:

[1, 64, 256]

Therefore, when the CNN is implemented, its first convolutional layer must accept the appropriate number of input channels for the pipeline being evaluated.

This is an ML-side configuration and does not require modification to the preprocessing implementation.

5. What Member 2 Has Already Evaluated

The following have already been completed:

Dataset audit
RGB baseline
Grayscale conversion
Median denoising
CLAHE enhancement
Otsu thresholding
Deskew
Resize and padding
Image-quality analysis
Numerical preprocessing comparison
Controlled visual comparison

All six pipelines have been processed across the dataset.

6. What the Preprocessing Experiments Show

The experiments indicate different trade-offs.

P0

Maintains the original RGB representation and provides the baseline reference.

P1

Reduces the image to one channel while preserving the major handwriting structure in inspected samples.

P2

Reduces high-frequency noise with relatively small visual changes to handwriting structure.

P3

Can make faint handwriting more visible, but can also increase the visibility of background variation.

P4

Provides strong foreground/background separation but can remove subtle grayscale information and modify delicate handwriting structures.

P5

Corrects handwriting orientation but introduces rotation, interpolation, and padding effects.

These observations are descriptive only. They do not establish recognition performance.

7. Required ML Evaluation

The ML experiments should determine whether the preprocessing differences actually affect recognition.

For a fair comparison, the following should remain controlled as much as practical:

same dataset split
same CNN architecture
same training procedure
same validation procedure
same test set
same optimization strategy
same number of epochs
same augmentation policy
same evaluation metric
fixed random seed where practical

The main experimental variable should be the preprocessing pipeline.

Conceptually:

                 ┌── P0 ──→ CNN ──→ Evaluation
                 │
Dataset ─────────┼── P1 ──→ CNN ──→ Evaluation
                 │
                 ├── P2 ──→ CNN ──→ Evaluation
                 │
                 ├── P3 ──→ CNN ──→ Evaluation
                 │
                 ├── P4 ──→ CNN ──→ Evaluation
                 │
                 └── P5 ──→ CNN ──→ Evaluation

The objective is not to assume that the visually cleanest image will produce the best recognition.

8. Results to Return to Member 2

For each preprocessing pipeline, the ML experiment should record at least:

Metric	Purpose
Training loss	Monitor training behavior
Validation loss	Monitor generalization during training
Validation accuracy	Compare recognition performance
Test accuracy	Final held-out evaluation
Per-class performance	Identify classes affected by preprocessing
Training/inference time	Measure computational implications

Additional metrics may be added depending on the final ML task.

9. Final Pipeline Selection

The final preprocessing pipeline should not be selected using only:

mean intensity
standard deviation
foreground ratio
preprocessing speed
visual appearance

These measurements describe the preprocessing operation but do not directly measure medicine-name recognition.

Final selection should be based on the combined evidence from:

Preprocessing analysis
        +
Visual analysis
        +
Recognition performance
        +
Computational considerations

Until the ML experiments are completed, P0–P5 should be treated as candidate pipelines.

10. Member 2 Deliverables Completed

Member 2 has produced:

CV/preprocessing.py
CV/image_quality.py
CV/preprocessing_experiments.py

CV/preprocessing_experiments/
├── preprocessing_experiment_results.csv
├── preprocessing_experiment_summary.csv
├── train_comparison.png
├── validation_comparison.png
└── test_comparison.png

These files contain the implementation, numerical experiment results, and controlled visual comparisons required for the preprocessing stage.

11. Current Status
Dataset Audit                    COMPLETE
Raw RGB Baseline                 COMPLETE
Grayscale                        COMPLETE
Median Denoising                 COMPLETE
CLAHE                            COMPLETE
Otsu Thresholding                COMPLETE
Deskew                           COMPLETE
Resize / Normalization           COMPLETE
Image Quality Analysis           COMPLETE
Preprocessing Experiments        COMPLETE

ML Recognition Evaluation        PENDING
Final Pipeline Selection         PENDING
Member 3 Integration             PENDING

The preprocessing stage is therefore ready for downstream ML evaluation.