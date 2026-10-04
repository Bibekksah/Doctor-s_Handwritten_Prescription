from pathlib import Path
import json
import time

import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from ml.dataset import PrescriptionDataset
from ml.dataloader import create_class_mapping
from CV.preprocessing import get_grayscale_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
    / "ML003_resnet18_INT8_test.onnx"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
    / "OPT004_int8_benchmark.json"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 1
WARMUP_RUNS = 20
BENCHMARK_RUNS = 100


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class_to_idx, idx_to_class = create_class_mapping()

test_dataset = PrescriptionDataset(
    split="test",
    transform=get_grayscale_transform(),
    class_to_idx=class_to_idx,
)


# ---------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print("=" * 80)
print("OPT-004: ML-003 ONNX INT8 BENCHMARK")
print("=" * 80)

print(f"Model: {MODEL_PATH}")
print(f"Test samples: {len(test_dataset)}")
print(f"Input name: {input_name}")
print(f"Output name: {output_name}")
print()


# ---------------------------------------------------------
# Prediction on complete test set
# ---------------------------------------------------------

y_true = []
y_pred = []

print("Running test-set inference...")

for i in range(len(test_dataset)):

    image, _, _, _, label = test_dataset[i]

    # Dataset grayscale transform returns 1 channel.
    # ML-003 expects 3 channels.
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    image_np = image.unsqueeze(0).numpy().astype(np.float32)

    outputs = session.run(
        [output_name],
        {input_name: image_np},
    )

    logits = outputs[0]
    prediction = int(np.argmax(logits, axis=1)[0])

    y_true.append(int(label))
    y_pred.append(prediction)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


print()
print("Test-set results")
print("-" * 50)
print(f"Accuracy:            {accuracy * 100:.2f}%")
print(f"Weighted Precision:  {precision * 100:.2f}%")
print(f"Weighted Recall:     {recall * 100:.2f}%")
print(f"Weighted F1:         {f1 * 100:.2f}%")
print()


# ---------------------------------------------------------
# Single-image latency
# ---------------------------------------------------------

sample_image, _, _, _, _ = test_dataset[0]

if sample_image.ndim == 3 and sample_image.shape[0] == 1:
    sample_image = sample_image.repeat(3, 1, 1)

sample_np = (
    sample_image
    .unsqueeze(0)
    .numpy()
    .astype(np.float32)
)


# Warm-up
for _ in range(WARMUP_RUNS):
    session.run(
        [output_name],
        {input_name: sample_np},
    )


latencies_ms = []

for _ in range(BENCHMARK_RUNS):

    start = time.perf_counter()

    session.run(
        [output_name],
        {input_name: sample_np},
    )

    end = time.perf_counter()

    latencies_ms.append(
        (end - start) * 1000
    )


latencies_ms = np.array(latencies_ms)

mean_latency = float(np.mean(latencies_ms))
median_latency = float(np.median(latencies_ms))
min_latency = float(np.min(latencies_ms))
max_latency = float(np.max(latencies_ms))

throughput = 1000.0 / mean_latency


print("Single-image CPU latency")
print("-" * 50)
print(f"Mean latency:        {mean_latency:.3f} ms")
print(f"Median latency:      {median_latency:.3f} ms")
print(f"Minimum latency:     {min_latency:.3f} ms")
print(f"Maximum latency:     {max_latency:.3f} ms")
print(f"Throughput:          {throughput:.2f} images/sec")
print()


# ---------------------------------------------------------
# Model size
# ---------------------------------------------------------

model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = {
    "experiment": "OPT-004",
    "model": "ML-003 ResNet-18 ONNX INT8",
    "model_path": str(MODEL_PATH),
    "execution_provider": "CPUExecutionProvider",

    "test_samples": len(test_dataset),
    "num_classes": 78,

    "metrics": {
        "accuracy": accuracy,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1,
    },

    "latency": {
        "mean_ms": mean_latency,
        "median_ms": median_latency,
        "min_ms": min_latency,
        "max_ms": max_latency,
        "throughput_images_per_second": throughput,
    },

    "model_size_mb": model_size_mb,

    "reference": {
        "pytorch_accuracy": 0.8910,
        "pytorch_weighted_f1": 0.8865,
        "onnx_fp32_accuracy": 0.8910,
        "onnx_fp32_weighted_f1": 0.8865,
        "onnx_fp32_mean_latency_ms": 6.066,
        "onnx_fp32_throughput_images_per_second": 164.86,
        "onnx_fp32_size_mb": 0.1075,
    },
}


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=4)


print(f"Results saved to:")
print(OUTPUT_PATH)

print()
print("=" * 80)
print("OPT-004 COMPLETE")
print("=" * 80) 