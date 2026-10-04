from pathlib import Path
from collections import Counter

import torch
import pandas as pd

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

PREDICTIONS_PATH = (
    RESULTS_DIR / "prediction_analysis.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32

# Number of incorrect predictions to display
TOP_INCORRECT = 30


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDICINE CLASSIFIER PREDICTION ANALYSIS")
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
    # Load test dataloader
    # --------------------------------------------------------

    _, _, test_loader, class_to_idx, idx_to_class = (
        create_dataloaders(
            batch_size=BATCH_SIZE
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
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    num_classes = checkpoint["num_classes"]

    print(
        "Best validation accuracy:",
        f"{checkpoint['validation_accuracy'] * 100:.2f}%"
    )

    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = MedicineClassifier(
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction_records = []

    correct = 0
    total = 0

    print()
    print("Running prediction analysis...")

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidences, predictions = (
                probabilities.max(dim=1)
            )

            for i in range(len(labels)):

                actual_index = labels[i].item()
                predicted_index = predictions[i].item()
                confidence = confidences[i].item()

                actual_name = idx_to_class[
                    actual_index
                ]

                predicted_name = idx_to_class[
                    predicted_index
                ]

                is_correct = (
                    actual_index == predicted_index
                )

                if is_correct:
                    correct += 1

                total += 1

                prediction_records.append(
                    {
                        "actual_index": actual_index,
                        "actual_medicine": actual_name,
                        "predicted_index": predicted_index,
                        "predicted_medicine": predicted_name,
                        "confidence": confidence,
                        "correct": is_correct
                    }
                )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    results = pd.DataFrame(
        prediction_records
    )

    # --------------------------------------------------------
    # Save all predictions
    # --------------------------------------------------------

    results.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------------

    accuracy = correct / total

    print()
    print("=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)

    print("Total samples:", total)
    print("Correct predictions:", correct)
    print("Incorrect predictions:", total - correct)

    print(
        "Accuracy:",
        f"{accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Incorrect predictions
    # --------------------------------------------------------

    incorrect_results = results[
        results["correct"] == False
    ].copy()

    incorrect_results = incorrect_results.sort_values(
        by="confidence",
        ascending=False
    )

    print()
    print("=" * 70)
    print(
        f"TOP {TOP_INCORRECT} INCORRECT PREDICTIONS"
    )
    print("=" * 70)

    display_columns = [
        "actual_medicine",
        "predicted_medicine",
        "confidence"
    ]

    print(
        incorrect_results[
            display_columns
        ].head(TOP_INCORRECT).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Most predicted classes among incorrect results
    # --------------------------------------------------------

    predicted_counter = Counter(
        incorrect_results[
            "predicted_medicine"
        ]
    )

    print()
    print("=" * 70)
    print(
        "MOST COMMON WRONG PREDICTIONS"
    )
    print("=" * 70)

    print(
        f"{'Predicted Medicine':<25}"
        f"{'Wrong Predictions':>18}"
    )

    print("-" * 45)

    for medicine, count in (
        predicted_counter.most_common(15)
    ):

        print(
            f"{medicine:<25}"
            f"{count:>18}"
        )

    # --------------------------------------------------------
    # Per-class correct/incorrect count
    # --------------------------------------------------------

    class_summary = []

    for index in range(num_classes):

        medicine = idx_to_class[index]

        class_results = results[
            results["actual_index"] == index
        ]

        class_total = len(class_results)

        class_correct = int(
            class_results["correct"].sum()
        )

        class_accuracy = (
            class_correct / class_total
            if class_total > 0
            else 0
        )

        class_summary.append(
            {
                "medicine": medicine,
                "total": class_total,
                "correct": class_correct,
                "incorrect": class_total - class_correct,
                "accuracy": class_accuracy
            }
        )

    class_summary_df = pd.DataFrame(
        class_summary
    )

    class_summary_df = class_summary_df.sort_values(
        by="accuracy"
    )

    print()
    print("=" * 70)
    print("LOWEST PER-CLASS ACCURACY")
    print("=" * 70)

    print(
        class_summary_df.head(15).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Highest confidence wrong predictions
    # --------------------------------------------------------

    high_confidence_wrong = (
        incorrect_results
        .sort_values(
            by="confidence",
            ascending=False
        )
        .head(10)
    )

    print()
    print("=" * 70)
    print(
        "HIGH-CONFIDENCE WRONG PREDICTIONS"
    )
    print("=" * 70)

    print(
        high_confidence_wrong[
            [
                "actual_medicine",
                "predicted_medicine",
                "confidence"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "All predictions saved to:"
    )

    print(
        PREDICTIONS_PATH
    )


if __name__ == "__main__":
    main()