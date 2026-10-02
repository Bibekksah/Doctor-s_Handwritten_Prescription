from pathlib import Path
import json

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from ml.dataloader import create_dataloaders
from ml.model import MedicineClassifier
from ml.device import get_device, get_device_name


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "checkpoints"
    / "best_model.pth"
)

RESULTS_DIR = PROJECT_ROOT / "ml" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = RESULTS_DIR / "test_classification_report.txt"
METRICS_PATH = RESULTS_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"


# ============================================================
# EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("MEDICINE CLASSIFIER TEST EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print("Device:", device)
    print("Device name:", get_device_name(device))

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("Checkpoint:", CHECKPOINT_PATH)

    # --------------------------------------------------------
    # Load dataloaders
    # --------------------------------------------------------

    _, _, test_loader, class_to_idx, idx_to_class = create_dataloaders(
        batch_size=32
    )

    print("Testing samples:", len(test_loader.dataset))
    print("Medicine classes:", len(class_to_idx))

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    num_classes = checkpoint["num_classes"]

    model = MedicineClassifier(
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Best validation accuracy:",
        f"{checkpoint['validation_accuracy'] * 100:.2f}%"
    )

    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []

    print()
    print("Running test evaluation...")

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    target_names = [
        idx_to_class[i]
        for i in range(num_classes)
    ]

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        target_names=target_names,
        zero_division=0
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(report)

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "MEDICINE CLASSIFIER TEST EVALUATION\n"
        )

        file.write("=" * 70 + "\n\n")

        file.write(
            f"Device: {device}\n"
        )

        file.write(
            f"Device name: {get_device_name(device)}\n"
        )

        file.write(
            f"Test samples: {len(test_loader.dataset)}\n"
        )

        file.write(
            f"Number of classes: {num_classes}\n"
        )

        file.write(
            f"Best validation accuracy: "
            f"{checkpoint['validation_accuracy'] * 100:.2f}%\n"
        )

        file.write(
            f"Best epoch: {checkpoint['epoch']}\n\n"
        )

        file.write(
            f"Accuracy : {accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Precision: {precision * 100:.2f}%\n"
        )

        file.write(
            f"Recall   : {recall * 100:.2f}%\n"
        )

        file.write(
            f"F1 Score : {f1 * 100:.2f}%\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(report)

    # --------------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------------

    metrics = {
        "device": str(device),
        "device_name": get_device_name(device),
        "test_samples": len(test_loader.dataset),
        "num_classes": num_classes,
        "best_epoch": checkpoint["epoch"],
        "best_validation_accuracy":
            checkpoint["validation_accuracy"],
        "test_accuracy": accuracy,
        "test_precision_weighted": precision,
        "test_recall_weighted": recall,
        "test_f1_weighted": f1
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(num_classes))
    )

    fig, ax = plt.subplots(
        figsize=(24, 24)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=False
    )

    ax.set_title(
        "Medicine Classifier - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print("Classification report:")
    print(REPORT_PATH)

    print("Metrics:")
    print(METRICS_PATH)

    print("Confusion matrix:")
    print(CONFUSION_MATRIX_PATH)


if __name__ == "__main__":
    main()