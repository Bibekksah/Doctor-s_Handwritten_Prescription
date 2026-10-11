from pathlib import Path
import json

import torch
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
from ml.archive.model import MedicineClassifier
from ml.device import get_device, get_device_name

from ml.archive.augmentation import get_validation_transform


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "checkpoints"
    / "best_model_ML001.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ML001"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("ML-001 TEST EVALUATION")
    print("P1 GRAYSCALE + DATA AUGMENTATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()
    print("Device:", device)
    print(
        "Device name:",
        get_device_name(device)
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"ML-001 checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    print()
    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    num_classes = checkpoint["num_classes"]

    best_validation_accuracy = (
        checkpoint["validation_accuracy"]
    )

    best_epoch = checkpoint["epoch"]

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Number of classes:",
        num_classes
    )

    # ========================================================
    # TEST TRANSFORM
    # ========================================================

    # IMPORTANT:
    # No random augmentation is used during testing.
    #
    # ML-001 training:
    # P1 + RandomAffine
    #
    # ML-001 testing:
    # P1 Grayscale only

    test_transform = (
        get_validation_transform()
    )

    # ========================================================
    # DATALOADER
    # ========================================================

    (
        _,
        _,
        test_loader,
        class_to_idx,
        idx_to_class
    ) = create_dataloaders(
        batch_size=32,
        train_transform=test_transform,
        validation_transform=test_transform,
        test_transform=test_transform
    )

    print()
    print(
        "Testing samples:",
        len(test_loader.dataset)
    )

    print(
        "Medicine classes:",
        len(class_to_idx)
    )

    # --------------------------------------------------------
    # Validate class count
    # --------------------------------------------------------

    if num_classes != len(class_to_idx):

        raise ValueError(
            "Class count mismatch: "
            f"checkpoint={num_classes}, "
            f"dataset={len(class_to_idx)}"
        )

    # ========================================================
    # MODEL
    # ========================================================

    model = MedicineClassifier(
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ========================================================
    # INFERENCE
    # ========================================================

    all_labels = []

    all_predictions = []

    print()
    print(
        "Running ML-001 test evaluation..."
    )

    with torch.no_grad():

        for batch in test_loader:

            # PrescriptionDataset returns:
            #
            # batch[0] = image
            # batch[1] = medicine_name
            # batch[2] = generic_name
            # batch[3] = image_path
            # batch[4] = label

            images = batch[0].to(device)

            labels = batch[4].to(device)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    # ========================================================
    # METRICS
    # ========================================================

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

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("ML-001 TEST RESULTS")
    print("=" * 80)

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score : {f1 * 100:.2f}%"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

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
    print("=" * 80)
    print("ML-001 CLASSIFICATION REPORT")
    print("=" * 80)

    print(report)

    # ========================================================
    # SAVE CLASSIFICATION REPORT
    # ========================================================

    report_path = (
        RESULTS_DIR
        / "test_classification_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "MEDICINE CLASSIFIER TEST EVALUATION\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            "Experiment: ML001\n"
        )

        file.write(
            "Preprocessing: P1 Grayscale\n"
        )

        file.write(
            "Training augmentation: RandomAffine\n"
        )

        file.write(
            f"Checkpoint: "
            f"{CHECKPOINT_PATH.name}\n"
        )

        file.write(
            f"Device: {device}\n"
        )

        file.write(
            f"Device name: "
            f"{get_device_name(device)}\n"
        )

        file.write(
            f"Test samples: "
            f"{len(test_loader.dataset)}\n"
        )

        file.write(
            f"Number of classes: "
            f"{num_classes}\n"
        )

        file.write(
            f"Best validation accuracy: "
            f"{best_validation_accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Best epoch: {best_epoch}\n\n"
        )

        file.write(
            "TEST METRICS\n"
        )

        file.write(
            "-" * 80 + "\n"
        )

        file.write(
            f"Accuracy : "
            f"{accuracy * 100:.2f}%\n"
        )

        file.write(
            f"Precision: "
            f"{precision * 100:.2f}%\n"
        )

        file.write(
            f"Recall   : "
            f"{recall * 100:.2f}%\n"
        )

        file.write(
            f"F1 Score : "
            f"{f1 * 100:.2f}%\n\n"
        )

        file.write(
            "CLASSIFICATION REPORT\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(report)

    # ========================================================
    # SAVE METRICS JSON
    # ========================================================

    metrics = {

        "experiment":
            "ML001",

        "preprocessing_version":
            "P1",

        "preprocessing_name":
            "Grayscale",

        "training_augmentation":
            "RandomAffine",

        "checkpoint":
            CHECKPOINT_PATH.name,

        "device":
            str(device),

        "device_name":
            get_device_name(device),

        "test_samples":
            len(test_loader.dataset),

        "num_classes":
            num_classes,

        "best_epoch":
            best_epoch,

        "best_validation_accuracy":
            best_validation_accuracy,

        "test_accuracy":
            accuracy,

        "test_precision_weighted":
            precision,

        "test_recall_weighted":
            recall,

        "test_f1_weighted":
            f1
    }

    metrics_path = (
        RESULTS_DIR
        / "test_metrics.json"
    )

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

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

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
        "ML-001 - P1 Grayscale + RandomAffine\n"
        "Medicine Classifier Confusion Matrix"
    )

    plt.tight_layout()

    confusion_matrix_path = (
        RESULTS_DIR
        / "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=200
    )

    plt.close()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("ML-001 EVALUATION COMPLETE")
    print("=" * 80)

    print(
        "Accuracy:",
        f"{accuracy * 100:.2f}%"
    )

    print(
        "Precision:",
        f"{precision * 100:.2f}%"
    )

    print(
        "Recall:",
        f"{recall * 100:.2f}%"
    )

    print(
        "F1:",
        f"{f1 * 100:.2f}%"
    )

    print()
    print(
        "Classification report:",
        report_path
    )

    print(
        "Metrics:",
        metrics_path
    )

    print(
        "Confusion matrix:",
        confusion_matrix_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()