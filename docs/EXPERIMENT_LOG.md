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