# Experiment Log

## EXP-001 — Dataset Validation

Date: 2026-09-28

Dataset:
Doctor's Handwritten Prescription BD

Total images:
4680

Training:
3120

Validation:
780

Testing:
780

Medicine classes:
78

Generic classes:
15

Missing values:
0

Duplicate rows:
0

Missing images:
0

Status:
PASS

---

## EXP-002 — Image Inspection

Date: 2026-09-28

Purpose:
Inspect image format, dimensions, and content.

Observations:

- Images contain individual handwritten medicine words.
- Image dimensions are variable.
- Images can be RGB or RGBA.
- Images are converted to RGB during preprocessing.

Status:
PASS

---

## EXP-003 — Metadata Pipeline

Date: 2026-09-28

Purpose:
Create a unified metadata file.

Output:

data/metadata/dataset.csv

Columns:

- image_path
- medicine_name
- generic_name
- split

Total records:
4680

Status:
PASS

---

## EXP-004 — Preprocessing

Date: 2026-09-28

Purpose:
Convert variable-sized images into model-compatible tensors.

Current target tensor:

[3, 64, 256]

Status:
IN PROGRESS

---
EXP-001 Dataset Validation       PASS
EXP-002 Image Inspection         PASS
EXP-003 Metadata Pipeline        PASS
EXP-004 Image Preprocessing      PASS
EXP-005 DataLoader Verification  PASS



## EXP-006: Preprocessing Ablation Study

### Objective
Evaluate the effect of six image preprocessing pipelines
on handwritten medicine recognition.

### Dataset
- Total images: 4680
- Training: 3120
- Validation: 780
- Testing: 780
- Medicine classes: 78

### Controlled Parameters
- Model: MedicineClassifier
- Image size: 64 × 256
- Batch size: 32
- Epochs: 20
- Learning rate: 0.001
- Weight decay: 0.0001
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Random seed: 42
- GPU: NVIDIA GeForce RTX 3050 6GB

### Results

| Pipeline | Best Validation Accuracy | Best Epoch |
|---|---:|---:|
| P0 RGB Baseline | 29.10% | 15 |
| P1 Grayscale | 36.41% | 19 |
| P2 Grayscale + Denoise | 23.46% | 9 |
| P3 Grayscale + Denoise + CLAHE | 18.97% | 9 |
| P4 Grayscale + Denoise + Otsu | 17.69% | 5 |
| P5 Grayscale + Denoise + Deskew | 13.72% | 15 |

### Observation

Validation performance varied substantially across preprocessing
pipelines. P1 achieved the highest validation accuracy among the
six pipelines at 36.41%. Final preprocessing selection will be
performed after evaluating the corresponding best checkpoints on
the held-out test set.



## EXP-009: ML-003 ResNet-18 Transfer Learning

### Objective

Evaluate transfer learning using a pretrained ResNet-18 model
for handwritten medicine-name recognition.

### Configuration

- Model: ResNet-18
- Transfer learning: ImageNet pretrained weights
- Classes: 78 medicine names
- Preprocessing: P1 Grayscale
- Image size: 64 × 256
- Training samples: 3120
- Validation samples: 780
- Testing samples: 780
- Batch size: 32
- Epochs: 20
- Learning rate: 0.001
- Weight decay: 0.0001
- Optimizer: Adam
- Random seed: 42
- GPU: NVIDIA GeForce RTX 3050 6GB

### Training Result

Best validation accuracy:

95.38%

Best epoch:

9

### Test Result

| Metric | Result |
|---|---:|
| Accuracy | 89.10% |
| Weighted Precision | 90.63% |
| Weighted Recall | 89.10% |
| Weighted F1 | 88.65% |
| Macro F1 | 89% |

### Observation

The ResNet-18 transfer-learning model substantially improved
recognition performance compared with the previous baseline
and custom CNN experiments.

The model achieved 89.10% accuracy and 88.65% weighted F1
on the held-out test set of 780 images.

### Decision

Retain ML-003 as the reference recognition model for the
subsequent model optimization and deployment experiments.

The saved checkpoint is:

ml/checkpoints/best_model_ML003.pth