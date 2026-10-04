from pathlib import Path
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from ml.model_ml003 import TransferMedicineClassifier
from ml.dataloader import create_dataloaders
from ml.device import get_device, get_device_name
from CV.preprocessing import get_grayscale_transform


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "checkpoints"
    / "best_model_ML003.pth"
)

RESULTS_DIR = PROJECT_ROOT / "ml" / "results" / "ML003"


def main():

    print("=" * 80)
    print("ML-003 TEST EVALUATION")
    print("RESNET-18 TRANSFER LEARNING + P1 GRAYSCALE")
    print("=" * 80)

    device = get_device()

    print(f"Device: {device}")
    print(f"Device name: {get_device_name(device)}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = checkpoint["idx_to_class"]

    print(f"Best validation accuracy: "
          f"{checkpoint['best_validation_accuracy'] * 100:.2f}%")

    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Number of classes: {len(class_to_idx)}")

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------

    transform = get_grayscale_transform()

    _, _, test_loader, _, _ = create_dataloaders(
        batch_size=32,
        train_transform=transform,
        validation_transform=transform,
        test_transform=transform,
    )

    print(f"Testing samples: {len(test_loader.dataset)}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = TransferMedicineClassifier(
        num_classes=len(class_to_idx)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch[0].to(device)
            labels = batch[4].to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    print("\n" + "=" * 80)
    print("ML-003 TEST RESULTS")
    print("=" * 80)

    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")

    # ---------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------

    target_names = [
    idx_to_class[i]
    for i in range(len(idx_to_class))
]

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(report)

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics = {
        "experiment": "ML003",
        "model": "ResNet-18 Transfer Learning",
        "preprocessing": "P1 Grayscale",
        "best_validation_accuracy": checkpoint[
            "best_validation_accuracy"
        ],
        "best_epoch": checkpoint["best_epoch"],
        "test_samples": len(all_labels),
        "num_classes": len(class_to_idx),
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "device": str(device),
        "device_name": get_device_name(device),
    }

    metrics_path = RESULTS_DIR / "test_metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # ---------------------------------------------------------
    # Save classification report
    # ---------------------------------------------------------

    report_path = (
        RESULTS_DIR
        / "test_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(target_names))),
    )

    np.save(
        RESULTS_DIR / "confusion_matrix.npy",
        cm
    )

    print("\nResults saved:")
    print(f"Metrics: {metrics_path}")
    print(f"Report : {report_path}")
    print(
        f"Confusion matrix: "
        f"{RESULTS_DIR / 'confusion_matrix.npy'}"
    )

    print("\n" + "=" * 80)
    print("ML-003 TEST EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()