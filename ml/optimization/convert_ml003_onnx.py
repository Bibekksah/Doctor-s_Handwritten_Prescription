from pathlib import Path
import json

import torch
import onnx

from ml.model_ml003 import TransferMedicineClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "checkpoints"
    / "best_model_ML003.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
)

ONNX_PATH = (
    RESULTS_DIR
    / "ML003_resnet18.onnx"
)


def main():

    print("=" * 80)
    print("OPT-001: ML-003 PYTORCH → ONNX")
    print("=" * 80)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device("cpu")

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output:     {ONNX_PATH}")

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]

    num_classes = len(class_to_idx)

    print(f"Classes: {num_classes}")
    print(
        f"Reference validation accuracy: "
        f"{checkpoint['best_validation_accuracy'] * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    model = TransferMedicineClassifier(
        num_classes=num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Dummy input
    # ---------------------------------------------------------

    dummy_input = torch.randn(
        1,
        3,
        64,
        256,
        device=device
    )

    print(f"Input shape: {tuple(dummy_input.shape)}")

    # ---------------------------------------------------------
    # Export ONNX
    # ---------------------------------------------------------

    print("\nExporting model...")

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {
                0: "batch_size"
            },
            "logits": {
                0: "batch_size"
            },
        },
        opset_version=18,
    )

    print("ONNX export complete.")

    # ---------------------------------------------------------
    # Validate ONNX model
    # ---------------------------------------------------------

    print("\nChecking ONNX model...")

    onnx_model = onnx.load(
        ONNX_PATH
    )

    onnx.checker.check_model(
        onnx_model
    )

    print("ONNX model validation: PASS")

    # ---------------------------------------------------------
    # Model size
    # ---------------------------------------------------------

    size_mb = (
        ONNX_PATH.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"ONNX model size: "
        f"{size_mb:.2f} MB"
    )

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata = {
        "experiment": "OPT-001",
        "description": "ML-003 ResNet-18 exported to ONNX",
        "source_model": "ML-003",
        "num_classes": num_classes,
        "input_shape": [1, 3, 64, 256],
        "opset_version": 18,
        "reference_validation_accuracy":
            checkpoint["best_validation_accuracy"],
        "onnx_model_size_mb": size_mb,
        "onnx_path": str(ONNX_PATH),
    }

    metadata_path = (
        RESULTS_DIR
        / "OPT001_onnx_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"Metadata saved: "
        f"{metadata_path}"
    )

    print("\n" + "=" * 80)
    print("OPT-001 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()