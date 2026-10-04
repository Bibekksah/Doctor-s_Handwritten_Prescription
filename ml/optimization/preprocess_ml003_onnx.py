from pathlib import Path

import onnx
from onnxruntime.quantization.shape_inference import quant_pre_process


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_MODEL = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
    / "ML003_resnet18.onnx"
)

OUTPUT_MODEL = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
    / "ML003_resnet18_preprocessed.onnx"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("OPT-003A: ONNX PREPROCESSING FOR INT8 QUANTIZATION")
print("=" * 80)

print(f"Input:  {INPUT_MODEL}")
print(f"Output: {OUTPUT_MODEL}")


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_MODEL.exists():
    raise FileNotFoundError(
        f"Input ONNX model not found:\n{INPUT_MODEL}"
    )


# ============================================================
# VALIDATE ORIGINAL
# ============================================================

print("\nChecking original ONNX model...")

model = onnx.load(str(INPUT_MODEL))

onnx.checker.check_model(model)

print("Original ONNX validation: PASS")


# ============================================================
# PREPROCESS
# ============================================================

print("\nRunning ONNX Runtime quantization preprocessing...")

quant_pre_process(
    input_model=str(INPUT_MODEL),
    output_model_path=str(OUTPUT_MODEL),
    skip_optimization=False,
    skip_onnx_shape=False,
    skip_symbolic_shape=True,
)

print("Preprocessing complete.")


# ============================================================
# VALIDATE OUTPUT
# ============================================================

print("\nChecking preprocessed ONNX model...")

preprocessed_model = onnx.load(
    str(OUTPUT_MODEL)
)

onnx.checker.check_model(
    preprocessed_model
)

print("Preprocessed ONNX validation: PASS")


# ============================================================
# SIZE
# ============================================================

size_mb = (
    OUTPUT_MODEL.stat().st_size
    / (1024 * 1024)
)

print(f"\nPreprocessed model size: {size_mb:.4f} MB")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("OPT-003A COMPLETE")
print("=" * 80)