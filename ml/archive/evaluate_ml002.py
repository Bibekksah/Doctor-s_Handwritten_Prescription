from pathlib import Path
import json

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

from ml.archive.model_ml002 import ImprovedMedicineClassifier
from ml.dataloader import create_dataloaders
from ml.device import get_device, get_device_name
from CV.preprocessing import get_grayscale_transform


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "checkpoints"
    / "best_model_ML002.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ML002"
)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("ML-002 TEST EVALUATION")
    print("P1 GRAYSCALE + IMPROVED CNN")
    print("=" * 80)

    print(f"Device: {device}")
    print(f"Device name: {get_device_name(device)}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # P1 grayscale only.
    # No random augmentation during testing.
    transform = get_grayscale_transform()

    (
        train_loader,
        validation_loader,
        test_loader,
        class_to_idx,
        idx_to_class,
    ) = create_dataloaders(
        batch_size=32,
        train_transform=transform,
        validation_transform=transform,
        test_transform=transform,
    )

    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Testing samples: {len(test_loader.dataset)}")

    model = ImprovedMedicineClassifier(
        num_classes=len(class_to_idx)
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nRunning ML-002 test evaluation...\n")

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch[0].to(device)
            labels = batch[4].to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    print("=" * 80)
    print("ML-002 TEST RESULTS")
    print("=" * 80)

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    # Classification report

    target_names = [
        idx_to_class[index]
        for index in range(len(idx_to_class))
    ]

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(idx_to_class))),
        target_names=target_names,
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("ML-002 CLASSIFICATION REPORT")
    print("=" * 80)

    print(report)

    report_path = OUTPUT_DIR / "test_classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("ML-002 Test Classification Report\n")
        file.write("=" * 80 + "\n\n")

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

        file.write(report)

    # Metrics JSON

    metrics = {
        "experiment": "ML002",
        "preprocessing": "P1 Grayscale",
        "model": "ImprovedMedicineClassifier",
        "test_samples": len(test_loader.dataset),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "best_validation_accuracy": checkpoint.get(
            "best_validation_accuracy"
        ),
        "best_epoch": checkpoint.get("best_epoch"),
        "device": str(device),
        "device_name": get_device_name(device),
    }

    metrics_path = OUTPUT_DIR / "test_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Confusion matrix

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(idx_to_class))),
    )

    plt.figure(figsize=(20, 18))

    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=True,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("ML-002 Confusion Matrix")

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()

    confusion_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        confusion_path,
        dpi=200,
    )

    plt.close()

    print("\nFiles saved:")
    print(report_path)
    print(metrics_path)
    print(confusion_path)

    print("\n" + "=" * 80)
    print("ML-002 EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
    