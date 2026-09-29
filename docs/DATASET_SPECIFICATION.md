## Dataset 1:
# Doctor's Handwritten Prescription BD

Purpose:
OCR model development

# Dataset Specification

# Doctor's Handwritten Prescription BD Dataset

## 1. Dataset Overview

This project uses the Doctor's Handwritten Prescription BD
dataset for the handwritten medicine recognition stage.

The dataset contains individual handwritten medicine-word images
with corresponding medicine and generic medicine labels.

---

## 2. Dataset Statistics

| Property | Value |
|---|---:|
| Total images | 4,680 |
| Training images | 3,120 |
| Validation images | 780 |
| Testing images | 780 |
| Medicine classes | 78 |
| Generic medicine classes | 15 |

---

## 3. Dataset Split

### Training

3,120 images

78 medicine classes

40 images per medicine class

### Validation

780 images

78 medicine classes

10 images per medicine class

### Testing

780 images

78 medicine classes

10 images per medicine class

---

## 4. Data Quality

Missing IMAGE values: 0

Missing MEDICINE_NAME values: 0

Missing GENERIC_NAME values: 0

Duplicate rows: 0

Missing image files: 0

---

## 5. Dataset Columns

### IMAGE

Filename of the handwritten image.

Example:

0.png

### MEDICINE_NAME

The medicine/brand name represented by the handwritten image.

Example:

Aceta

### GENERIC_NAME

The corresponding generic medicine name.

Example:

Paracetamol

---

## 6. Primary Machine Learning Task

The primary recognition task is:

Image → MEDICINE_NAME

Example:

Handwritten image → Aceta

---

## 7. Secondary Verification Task

The recognized medicine name can later be mapped to its generic medicine:

MEDICINE_NAME → GENERIC_NAME

Example:

Aceta → Paracetamol

---

## 8. Image Characteristics

The images contain individual handwritten medicine words.

Image dimensions are variable.

Examples:

238 × 92
207 × 84
324 × 89
283 × 96

Images may be RGB or RGBA.

The preprocessing pipeline converts images to RGB.

---

## 9. Standardized Metadata

The project creates:

data/metadata/dataset.csv

with the following fields:

image_path
medicine_name
generic_name
split

---

## 10. Data Leakage Prevention

The training, validation, and testing splits supplied by the
dataset are maintained.

The test set must not be used for model training or hyperparameter
tuning.

---

## 11. Current Pipeline

Raw Dataset

↓

Metadata Validation

↓

dataset.csv

↓

Image Preprocessing

↓

DataLoader

↓

Machine Learning Model

---

## 12. Future Integration

The handwritten medicine recognition stage will later be
integrated with:

- OCR
- Medicine verification
- FastAPI
- Frontend
- Docker
- Cloud deployment
- Model optimization

---
Dataset
├── Total images: 4680
├── Training: 3120
├── Validation: 780
├── Testing: 780
├── Medicine classes: 78
├── Generic classes: 15
├── Missing values: 0
├── Duplicate rows: 0
└── Missing images: 0