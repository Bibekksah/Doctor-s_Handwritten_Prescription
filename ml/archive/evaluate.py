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
from ml.model import MedicineClassifier
from ml.device import get_device, get_device_name

from CV.preprocessing import (
    get_baseline_transform,
    get_grayscale_transform,
    get_grayscale_denoise_transform,
    get_grayscale_denoise_clahe_transform,
    get_grayscale_denoise_threshold_transform,
    get_grayscale_denoise_deskew_transform,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "checkpoints"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PREPROCESSING PIPELINES
# ============================================================

PIPELINES = {
    "P0": {
        "name": "RGB Baseline",
        "transform": get_baseline_transform,
        "checkpoint": "best_model_P0.pth",
    },

    "P1": {
        "name": "Grayscale",
        "transform": get_grayscale_transform,
        "checkpoint": "best_model_P1.pth",
    },

    "P2": {
        "name": "Grayscale + Denoise",
        "transform": get_grayscale_denoise_transform,
        "checkpoint": "best_model_P2.pth",
    },

    "P3": {
        "name": "Grayscale + Denoise + CLAHE",
        "transform": get_grayscale_denoise_clahe_transform,
        "checkpoint": "best_model_P3.pth",
    },

    "P4": {
        "name": "Grayscale + Denoise + Otsu",
        "transform": get_grayscale_denoise_threshold_transform,
        "checkpoint": "best_model_P4.pth",
    },

    "P5": {
        "name": "Grayscale + Denoise + Deskew",
        "transform": get_grayscale_denoise_deskew_transform,
        "checkpoint": "best_model_P5.pth",
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def evaluate_pipeline(
    pipeline_id,
    pipeline_info,
    device
):
    """
    Evaluate one preprocessing pipeline using its
    corresponding best checkpoint.
    """

    pipeline_name = pipeline_info["name"]
    checkpoint_name = pipeline_info["checkpoint"]
    transform_function = pipeline_info["transform"]

    print()
    print("=" * 80)
    print(f"EVALUATING {pipeline_id}")
    print("=" * 80)

    print("Pipeline:", pipeline_name)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / checkpoint_name
    )

    pipeline_results_dir = (
        RESULTS_DIR
        / pipeline_id
    )

    pipeline_results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        pipeline_results_dir
        / "test_classification_report.txt"
    )

    metrics_path = (
        pipeline_results_dir
        / "test_metrics.json"
    )

    confusion_matrix_path = (
        pipeline_results_dir
        / "confusion_matrix.png"
    )

    # --------------------------------------------------------
    # Check checkpoint
    # --------------------------------------------------------

    if not checkpoint_path.exists():

        print(
            f"WARNING: Checkpoint not found:"
            f"\n{checkpoint_path}"
        )

        return None

    print("Checkpoint:", checkpoint_path)

    # --------------------------------------------------------
    # Create matching preprocessing transform
    # --------------------------------------------------------

    transform = transform_function()

    print(
        "Preprocessing transform:",
        pipeline_name
    )

    # --------------------------------------------------------
    # Create dataloaders
    # --------------------------------------------------------

    _, _, test_loader, class_to_idx, idx_to_class = (
        create_dataloaders(
            batch_size=32,
            transform=transform
        )
    )

    print(
        "Testing samples:",
        len(test_loader.dataset)
    )

    print(
        "Medicine classes:",
        len(class_to_idx)
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    num_classes = checkpoint["num_classes"]

    # --------------------------------------------------------
    # Validate number of classes
    # --------------------------------------------------------

    if num_classes != len(class_to_idx):

        raise ValueError(
            f"Class count mismatch for {pipeline_id}: "
            f"checkpoint has {num_classes}, "
            f"dataset has {len(class_to_idx)}."
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MedicineClassifier(
        num_classes=num_classes
    ).to(device)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Checkpoint information
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []

    print()
    print("Running test evaluation...")

    with torch.no_grad():

        for batch in test_loader:

            # ------------------------------------------------
            # PrescriptionDataset returns:
            #
            # image
            # medicine_name
            # generic_name
            # image_path
            # label
            # ------------------------------------------------

            images = batch[0].to(device)
            labels = batch[4].to(device)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            outputs = model(images)

            # ------------------------------------------------
            # Predicted class
            # ------------------------------------------------

            predictions = outputs.argmax(
                dim=1
            )

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

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
    print("-" * 80)
    print(f"{pipeline_id} TEST RESULTS")
    print("-" * 80)

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

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    target_names = [
        idx_to_class[i]
        for i in range(num_classes)
    ]

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        target_names=target_names,
        zero_division=0
    )

    print()
    print("-" * 80)
    print(f"{pipeline_id} CLASSIFICATION REPORT")
    print("-" * 80)

    print(report)

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

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
            f"Pipeline ID: {pipeline_id}\n"
        )

        file.write(
            f"Pipeline: {pipeline_name}\n"
        )

        file.write(
            f"Checkpoint: {checkpoint_name}\n"
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
            f"Best epoch: "
            f"{best_epoch}\n\n"
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

    # --------------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------------

    metrics = {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline_name,
        "checkpoint": checkpoint_name,
        "device": str(device),
        "device_name": get_device_name(device),
        "test_samples": len(test_loader.dataset),
        "num_classes": num_classes,
        "best_epoch": best_epoch,
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
        f"{pipeline_id} - "
        f"{pipeline_name}\n"
        f"Medicine Classifier Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        confusion_matrix_path,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    result = {
        "pipeline_id": pipeline_id,
        "pipeline_name": pipeline_name,
        "checkpoint": checkpoint_name,
        "best_epoch": best_epoch,
        "best_validation_accuracy":
            best_validation_accuracy,
        "test_accuracy":
            accuracy,
        "test_precision_weighted":
            precision,
        "test_recall_weighted":
            recall,
        "test_f1_weighted":
            f1,
        "test_samples":
            len(test_loader.dataset),
        "num_classes":
            num_classes
    }

    print()
    print(
        "Saved classification report:"
    )
    print(report_path)

    print(
        "Saved metrics:"
    )
    print(metrics_path)

    print(
        "Saved confusion matrix:"
    )
    print(confusion_matrix_path)

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("MEDICINE CLASSIFIER")
    print("P0-P5 PREPROCESSING TEST EVALUATION")
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
    # Evaluate all pipelines
    # --------------------------------------------------------

    results = []

    for pipeline_id, pipeline_info in PIPELINES.items():

        result = evaluate_pipeline(
            pipeline_id,
            pipeline_info,
            device
        )

        if result is not None:
            results.append(result)

    # --------------------------------------------------------
    # Check whether evaluation produced results
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "No pipeline was successfully evaluated."
        )

    # --------------------------------------------------------
    # Save aggregate comparison
    # --------------------------------------------------------

    comparison_path = (
        RESULTS_DIR
        / "preprocessing_comparison.json"
    )

    comparison = {
        "experiment": (
            "P0-P5 Preprocessing "
            "Ablation Test Evaluation"
        ),
        "device": str(device),
        "device_name": get_device_name(device),
        "results": results
    }

    with open(
        comparison_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            comparison,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Print final comparison
    # --------------------------------------------------------

    print()
    print()
    print("=" * 100)
    print("FINAL P0-P5 TEST COMPARISON")
    print("=" * 100)

    print(
        f"{'Pipeline':<10}"
        f"{'Best Val':>12}"
        f"{'Test Acc':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print("-" * 100)

    for result in results:

        print(
            f"{result['pipeline_id']:<10}"
            f"{result['best_validation_accuracy'] * 100:>11.2f}%"
            f"{result['test_accuracy'] * 100:>11.2f}%"
            f"{result['test_precision_weighted'] * 100:>11.2f}%"
            f"{result['test_recall_weighted'] * 100:>11.2f}%"
            f"{result['test_f1_weighted'] * 100:>11.2f}%"
        )

    print("-" * 100)

    # --------------------------------------------------------
    # Results location
    # --------------------------------------------------------

    print()
    print(
        "Aggregate comparison:"
    )

    print(comparison_path)

    print()
    print("=" * 80)
    print("ALL PIPELINE EVALUATIONS COMPLETE")
    print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()