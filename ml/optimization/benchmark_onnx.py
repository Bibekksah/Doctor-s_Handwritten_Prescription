from pathlib import Path
import json
import time

import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from ml.dataset import PrescriptionDataset
from ml.dataloader import create_class_mapping
from CV.preprocessing import get_grayscale_transform


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ONNX_PATH = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
    / "ML003_resnet18.onnx"
)

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 78

WARMUP_RUNS = 20
BENCHMARK_RUNS = 100

SEED = 42

np.random.seed(SEED)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("OPT-002: ML-003 ONNX INFERENCE BENCHMARK")
print("=" * 80)

print(f"ONNX model: {ONNX_PATH}")
print(f"Dataset:    {CSV_PATH}")


# ============================================================
# CHECK FILES
# ============================================================

if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found:\n{ONNX_PATH}"
    )

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset CSV not found:\n{CSV_PATH}"
    )


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print("\nLoading ONNX model...")

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

input_name = input_info.name
output_name = output_info.name

print(f"Input name:  {input_name}")
print(f"Output name: {output_name}")
print(f"Providers:   {session.get_providers()}")


# ============================================================
# MODEL SIZE
# ============================================================

model_size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)

print(f"ONNX model size: {model_size_mb:.4f} MB")


# ============================================================
# CLASS MAPPING
# ============================================================

class_to_idx, idx_to_class = create_class_mapping()

print(f"\nNumber of classes: {len(class_to_idx)}")

if len(class_to_idx) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_to_idx)}."
    )


# ============================================================
# TEST DATASET
# ============================================================

print("\nLoading test dataset...")

transform = get_grayscale_transform()

test_dataset = PrescriptionDataset(
    csv_path=CSV_PATH,
    split="test",
    transform=transform,
    class_to_idx=class_to_idx,
)

print(f"Test samples: {len(test_dataset)}")


# ============================================================
# FIRST SAMPLE
# ============================================================

print("\nChecking first sample...")

sample_image, sample_medicine, sample_generic, sample_path, sample_label = (
    test_dataset[0]
)

sample_input = sample_image.unsqueeze(0).numpy().astype(np.float32)

print(f"Image path:  {sample_path}")
print(f"Medicine:    {sample_medicine}")
print(f"Generic:     {sample_generic}")
print(f"Label index: {sample_label}")
print(f"Input shape: {sample_input.shape}")


# ============================================================
# FIRST ONNX INFERENCE
# ============================================================

sample_output = session.run(
    [output_name],
    {input_name: sample_input},
)[0]

sample_prediction = int(
    np.argmax(sample_output, axis=1)[0]
)

print(f"Output shape:       {sample_output.shape}")
print(f"Predicted index:    {sample_prediction}")
print(f"Predicted medicine: {idx_to_class[sample_prediction]}")
print(f"Actual medicine:    {idx_to_class[sample_label]}")


# ============================================================
# WARM-UP
# ============================================================

print(f"\nWarm-up runs: {WARMUP_RUNS}")

for _ in range(WARMUP_RUNS):

    session.run(
        [output_name],
        {input_name: sample_input},
    )


# ============================================================
# LATENCY BENCHMARK
# ============================================================

print(f"Benchmark runs: {BENCHMARK_RUNS}")

latencies = []

for _ in range(BENCHMARK_RUNS):

    start = time.perf_counter()

    session.run(
        [output_name],
        {input_name: sample_input},
    )

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    latencies.append(latency_ms)


latencies = np.asarray(latencies)

mean_latency_ms = float(np.mean(latencies))
median_latency_ms = float(np.median(latencies))
min_latency_ms = float(np.min(latencies))
max_latency_ms = float(np.max(latencies))

throughput = 1000.0 / mean_latency_ms


# ============================================================
# TEST SET EVALUATION
# ============================================================

print("\nEvaluating ONNX model on test set...")

y_true = []
y_pred = []

test_inference_times = []

for index in range(len(test_dataset)):

    (
        image,
        medicine_name,
        generic_name,
        image_path,
        label,
    ) = test_dataset[index]

    model_input = image.unsqueeze(0).numpy().astype(np.float32)

    start = time.perf_counter()

    output = session.run(
        [output_name],
        {input_name: model_input},
    )[0]

    end = time.perf_counter()

    test_inference_times.append(
        (end - start) * 1000
    )

    prediction = int(
        np.argmax(output, axis=1)[0]
    )

    y_true.append(int(label))
    y_pred.append(prediction)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


# ============================================================
# TEST INFERENCE PERFORMANCE
# ============================================================

test_latency_ms = float(
    np.mean(test_inference_times)
)

test_throughput = (
    1000.0 / test_latency_ms
)


# ============================================================
# PYTORCH REFERENCE
# ============================================================

PYTORCH_TEST_ACCURACY = 89.10
PYTORCH_WEIGHTED_F1 = 88.65

accuracy_difference = (
    accuracy * 100
    - PYTORCH_TEST_ACCURACY
)

f1_difference = (
    f1 * 100
    - PYTORCH_WEIGHTED_F1
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("OPT-002 RESULTS")
print("=" * 80)

print(f"Test samples:       {len(test_dataset)}")

print("\nAccuracy metrics:")
print(f"Accuracy:            {accuracy * 100:.2f}%")
print(f"Weighted Precision:  {precision * 100:.2f}%")
print(f"Weighted Recall:     {recall * 100:.2f}%")
print(f"Weighted F1:         {f1 * 100:.2f}%")

print("\nSingle-image latency benchmark:")
print(f"Mean latency:        {mean_latency_ms:.3f} ms")
print(f"Median latency:      {median_latency_ms:.3f} ms")
print(f"Minimum latency:     {min_latency_ms:.3f} ms")
print(f"Maximum latency:     {max_latency_ms:.3f} ms")
print(f"Throughput:          {throughput:.2f} images/sec")

print("\nFull test-set inference:")
print(f"Mean latency:        {test_latency_ms:.3f} ms")
print(f"Throughput:          {test_throughput:.2f} images/sec")

print("\nComparison with ML-003 PyTorch reference:")

print(
    f"PyTorch test accuracy: {PYTORCH_TEST_ACCURACY:.2f}%"
)

print(
    f"ONNX test accuracy:    {accuracy * 100:.2f}%"
)

print(
    f"Accuracy difference:   {accuracy_difference:+.2f} percentage points"
)

print(
    f"PyTorch weighted F1:   {PYTORCH_WEIGHTED_F1:.2f}%"
)

print(
    f"ONNX weighted F1:      {f1 * 100:.2f}%"
)

print(
    f"F1 difference:         {f1_difference:+.2f} percentage points"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "experiment": "OPT-002",
    "model": "ML-003 ResNet-18",
    "runtime": "ONNX Runtime",
    "onnx_runtime_version": ort.__version__,
    "providers": session.get_providers(),

    "dataset": {
        "split": "test",
        "samples": len(test_dataset),
        "classes": len(class_to_idx),
    },

    "model": {
        "path": str(ONNX_PATH),
        "size_mb": model_size_mb,
        "input_shape": list(sample_input.shape),
    },

    "metrics": {
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    },

    "latency": {
        "mean_ms": mean_latency_ms,
        "median_ms": median_latency_ms,
        "min_ms": min_latency_ms,
        "max_ms": max_latency_ms,
        "throughput_images_per_second": throughput,
    },

    "test_inference": {
        "mean_latency_ms": test_latency_ms,
        "throughput_images_per_second": test_throughput,
    },

    "pytorch_reference": {
        "test_accuracy_percent": PYTORCH_TEST_ACCURACY,
        "weighted_f1_percent": PYTORCH_WEIGHTED_F1,
    },

    "difference": {
        "accuracy_percentage_points": accuracy_difference,
        "weighted_f1_percentage_points": f1_difference,
    },
}


output_file = (
    OUTPUT_DIR
    / "OPT002_onnx_benchmark.json"
)

with open(output_file, "w") as file:
    json.dump(
        results,
        file,
        indent=4,
    )


print("\nResults saved to:")
print(output_file)

print("\n" + "=" * 80)
print("OPT-002 COMPLETE")
print("=" * 80)