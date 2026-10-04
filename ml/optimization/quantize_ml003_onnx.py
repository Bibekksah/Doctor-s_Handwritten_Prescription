from pathlib import Path
import json

import onnx
from onnxruntime.quantization import (
    quantize_dynamic,
    QuantType,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_MODEL = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
    / "ML003_resnet18_preprocessed.onnx"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL = (
    OUTPUT_DIR
    / "ML003_resnet18_INT8.onnx"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "OPT003_int8_metadata.json"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("OPT-003: ML-003 ONNX INT8 QUANTIZATION")
print("=" * 80)

print(f"Input model:  {INPUT_MODEL}")
print(f"Output model: {OUTPUT_MODEL}")


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_MODEL.exists():
    raise FileNotFoundError(
        f"Input ONNX model not found:\n{INPUT_MODEL}"
    )


# ============================================================
# ORIGINAL SIZE
# ============================================================

original_size_bytes = INPUT_MODEL.stat().st_size
original_size_mb = original_size_bytes / (1024 * 1024)

print(f"\nOriginal ONNX size: {original_size_mb:.4f} MB")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nChecking input ONNX model...")

model = onnx.load(str(INPUT_MODEL))

onnx.checker.check_model(model)

print("Input ONNX validation: PASS")


# ============================================================
# DYNAMIC INT8 QUANTIZATION
# ============================================================

print("\nApplying dynamic INT8 quantization...")

quantize_dynamic(
    model_input=str(INPUT_MODEL),
    model_output=str(OUTPUT_MODEL),
    weight_type=QuantType.QInt8,
)

print("INT8 quantization complete.")


# ============================================================
# VALIDATE QUANTIZED MODEL
# ============================================================

print("\nChecking quantized ONNX model...")

quantized_model = onnx.load(str(OUTPUT_MODEL))

onnx.checker.check_model(quantized_model)

print("INT8 ONNX validation: PASS")


# ============================================================
# SIZE COMPARISON
# ============================================================

quantized_size_bytes = OUTPUT_MODEL.stat().st_size
quantized_size_mb = quantized_size_bytes / (1024 * 1024)

size_reduction_mb = (
    original_size_mb - quantized_size_mb
)

size_reduction_percent = (
    size_reduction_mb
    / original_size_mb
    * 100
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("OPT-003 RESULTS")
print("=" * 80)

print(
    f"Original ONNX size:  {original_size_mb:.4f} MB"
)

print(
    f"INT8 ONNX size:      {quantized_size_mb:.4f} MB"
)

print(
    f"Size reduction:      {size_reduction_mb:.4f} MB"
)

print(
    f"Size reduction:      {size_reduction_percent:.2f}%"
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "experiment": "OPT-003",
    "optimization": "Dynamic INT8 quantization",

    "input_model": str(INPUT_MODEL),
    "output_model": str(OUTPUT_MODEL),

    "input_model_size_mb": original_size_mb,
    "quantized_model_size_mb": quantized_size_mb,

    "size_reduction_mb": size_reduction_mb,
    "size_reduction_percent": size_reduction_percent,

    "quantization": {
        "method": "dynamic",
        "weight_type": "QInt8",
    },

    "validation": {
        "input_onnx": "PASS",
        "quantized_onnx": "PASS",
    },
}


with open(METADATA_FILE, "w") as file:
    json.dump(
        metadata,
        file,
        indent=4,
    )


print("\nMetadata saved to:")
print(METADATA_FILE)

print("\n" + "=" * 80)
print("OPT-003 COMPLETE")
print("=" * 80)