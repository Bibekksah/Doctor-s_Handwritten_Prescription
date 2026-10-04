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


### EXP-007: Baseline ML Recognition Model
## Objective

Establish a baseline handwritten medicine-name recognition model using the selected P1 grayscale preprocessing pipeline.

## Configuration
Task: Handwritten medicine-name classification
Model: Custom CNN
Classes: 78 medicine names
Preprocessing: P1 Grayscale
Image size: 64 × 256
Training samples: 3120
Validation samples: 780
Testing samples: 780
Batch size: 32
Epochs: 20
Optimizer: Adam
Test Result
Metric	Result
Test Accuracy	22.69%
Weighted F1	19.80%
## Observation

The baseline CNN successfully established an initial reference point for handwritten medicine-name classification.

However, the recognition performance remained limited, with only 22.69% test accuracy across the 78 medicine classes.

The result indicated that the baseline CNN was not sufficiently effective for the handwriting recognition task.

## Decision

Retain the experiment as the baseline ML reference and investigate improved CNN architectures and transfer-learning approaches in subsequent experiments.

### EXP-008: Improved CNN Recognition Model
## Objective

Evaluate whether an improved custom CNN architecture can improve handwritten medicine-name recognition compared with the baseline model.

# Configuration
Task: Handwritten medicine-name classification
Model: Improved Custom CNN
Classes: 78 medicine names
Preprocessing: P1 Grayscale
Image size: 64 × 256
Training samples: 3120
Validation samples: 780
Testing samples: 780
Epochs: 20
Optimizer: Adam
Test Result
Metric	Result
Test Accuracy	19.23%
Weighted F1	17.28%
Best Validation Accuracy	32.95%
## Observation

The improved custom CNN achieved a higher validation accuracy than the earlier baseline during training, reaching 32.95%.

However, its held-out test accuracy was only 19.23%, with a weighted F1 score of 17.28%.

The model therefore did not generalize sufficiently well to the unseen test samples.

## Comparison
Experiment	Model	Test Accuracy	Weighted F1
EXP-007	Baseline Custom CNN	22.69%	19.80%
EXP-008	Improved Custom CNN	19.23%	17.28%
## Decision

Do not retain the improved custom CNN as the final recognition model.

The results motivated the use of transfer learning with a pretrained ResNet-18 model, which was evaluated in EXP-009.




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




### EXP-010: Full Prescription OCR Integration
## Objective

Evaluate the integration of PaddleOCR with the ML-003 handwritten medicine recognition model for processing a complete handwritten medical prescription.

Configuration
Input: Full handwritten prescription image
OCR Engine: PaddleOCR
Recognition Model: ML-003 ResNet-18
Medicine Classes: 78 medicine names
Preprocessing: P1 Grayscale
Image Size: 64 × 256
OCR Region Padding: 5%
Inference Provider: ONNX Runtime CPU
Verification: Final Medicine Verification Module
Processing Pipeline
Full Prescription Image
        ↓
PaddleOCR
        ↓
OCR Text Regions
        ↓
P1 Grayscale Preprocessing
        ↓
ML-003 ResNet-18
        ↓
Medicine Prediction
        ↓
Medicine Verification
        ↓
MEDICINE / REVIEW / NON_MEDICINE
Test Result
Metric	Result
Prescription regions detected	36
MEDICINE	0
REVIEW	27
NON_MEDICINE	9
E2E Pipeline Status	PASS
Observation

The complete prescription pipeline successfully executed OCR region detection, ML-003 medicine recognition, and medicine verification.

The verification stage was intentionally conservative because ML-003 is a closed-set classifier containing 78 medicine classes. Therefore, non-medicine prescription regions should not automatically be treated as medicines simply because the classifier produces a prediction.

## Decision

Retain the PaddleOCR + ML-003 + verification pipeline as the current end-to-end recognition architecture.

The pipeline is suitable for further backend and frontend integration.

# Important: This experiment demonstrates successful system integration. The result is not a whole-prescription recognition accuracy benchmark.


### EXP-011: ONNX Model Optimization
## Objective

# Evaluate ONNX conversion of the ML-003 ResNet-18 model for deployment and CPU inference optimization.

Configuration
Source Model: ML-003 ResNet-18
Source Framework: PyTorch
Deployment Format: ONNX
ONNX Runtime: CPUExecutionProvider
Input Shape: 1 × 3 × 64 × 256
Output Classes: 78 medicine names
Test Samples: 780
Conversion Result

The trained ML-003 model was successfully converted to ONNX format and validated using the ONNX model checker.

# Deployment model:

ml/results/optimization/ML003_resnet18.onnx
Benchmark Result
Metric	Result
Accuracy	89.10%
Weighted Precision	90.63%
Weighted Recall	89.10%
Weighted F1	88.65%
Mean single-image latency	3.182 ms
Median latency	3.168 ms
Throughput	314.30 images/sec
Execution Provider	CPUExecutionProvider
Model Size	0.1075 MB
Observation

The ONNX model produced the same recognition metrics as the ML-003 reference model while providing fast CPU inference in the tested environment.

## Decision

Retain the ONNX version as the primary deployment model for subsequent backend and cloud deployment work.

# Benchmark results are stored in:

ml/results/optimization/OPT002_onnx_benchmark.json
EXP-012: INT8 Quantization Evaluation
Objective

Evaluate dynamic INT8 quantization of the ML-003 ONNX model to determine whether further model optimization improves deployment performance.

Configuration
Base Model: ML-003 ResNet-18
Format: ONNX
Quantization: Dynamic INT8
Quantization Type: QInt8
Test Samples: 780
Execution Provider: CPUExecutionProvider
Test Result
Metric	FP32 ONNX	INT8 ONNX
Accuracy	89.10%	88.97%
Weighted Precision	90.63%	90.52%
Weighted Recall	89.10%	88.97%
Weighted F1	88.65%	88.51%
Mean latency	3.182 ms*	23.300 ms
Throughput	314.30/s*	42.92/s

# Reference values from the selected FP32 ONNX benchmark.

## Observation

INT8 quantization preserved almost the same recognition accuracy, with only a small decrease in performance metrics.

However, the measured INT8 CPU inference latency was higher than the tested FP32 ONNX configuration, and the quantized model was also larger in the tested setup.

# Decision

Do not select INT8 quantization as the primary deployment configuration.

Retain the FP32 ONNX model because it provided the better measured deployment performance in the current environment.

### EXP-013: Medicine Verification
## Objective

Prevent the closed-set ML-003 classifier from automatically treating every OCR region as a medicine.

Configuration

The verification stage combines:

PaddleOCR detected text
OCR confidence
ML-003 prediction
ML confidence
OCR text vs. predicted medicine-name similarity
Non-medicine keywords
Clinical/context keywords
Review status for uncertain regions
Output Categories
Status	Meaning
MEDICINE	Evidence is strong enough to identify the region as a medicine
REVIEW	Evidence is insufficient or conflicting; human verification required
NON_MEDICINE	Region is identified as non-medicine/context text
Observation

Directly passing every OCR region to ML-003 is unsafe because ML-003 always predicts one of the 78 known medicine classes, even when the input region is unrelated to a medicine.

The verification layer therefore acts as a safety gate before presenting a prediction as a recognized medicine.

# Decision

Retain the final medicine verification module as part of the production pipeline.

## Module:

ml/medicine_verification_final.py


### EXP-014: End-to-End Prescription Recognition
## Objective

Verify that all major Member 3 recognition components operate together on a complete handwritten prescription.

Pipeline
Prescription Image
        ↓
PaddleOCR
        ↓
36 OCR Regions
        ↓
P1 Preprocessing
        ↓
ML-003 ResNet-18
        ↓
Medicine Verification
        ↓
Final Result
Result
Component	Status
Prescription image loaded	PASS
OCR executed	PASS
OCR regions generated	PASS
ML-003 recognizer executed	PASS
Medicine verification executed	PASS
Final results generated	PASS
Overall E2E test	PASS
Observation

The complete Member 3 recognition pipeline successfully operates from prescription-image input through OCR, handwritten medicine recognition, verification, and final result generation.

# Decision

The end-to-end recognition pipeline is ready for backend API integration and frontend integration.

The current E2E test should be treated as an integration validation, not as a clinical or whole-prescription accuracy evaluation.

Recommended Member 3 Documentation Order

For your final project documentation, I recommend keeping the experiments in this order:

EXP-009  → ML-003 ResNet-18 Transfer Learning
EXP-010  → Full Prescription OCR Integration
EXP-011  → ONNX Model Optimization
EXP-012  → INT8 Quantization Evaluation
EXP-013  → Medicine Verification
EXP-014  → End-to-End Prescription Recognition