from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ml.dataloader import create_dataloaders
from ml.archive.model import MedicineClassifier
from ml.device import get_device, get_device_name


# ============================================================
# PROJECT CONFIGURATION
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


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Choose preprocessing pipeline here.
#
# P0 = RGB Baseline
# P1 = Grayscale
# P2 = Grayscale + Denoising
# P3 = Grayscale + Denoising + CLAHE
# P4 = Grayscale + Denoising + Otsu
# P5 = Grayscale + Denoising + Deskew
# ------------------------------------------------------------

PREPROCESSING_VERSION = "P5"


# ------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------

NUM_EPOCHS = 20

BATCH_SIZE = 32

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

SEED = 42


# ============================================================
# CREATE DIRECTORIES
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

def get_selected_transform():
    """
    Return the preprocessing transform selected by
    PREPROCESSING_VERSION.
    """

    from CV.preprocessing import (
        get_baseline_transform,
        get_grayscale_transform,
        get_grayscale_denoise_transform,
        get_grayscale_denoise_clahe_transform,
        get_grayscale_denoise_threshold_transform,
        get_grayscale_denoise_deskew_transform,
    )

    pipelines = {

        "P0":
            get_baseline_transform,

        "P1":
            get_grayscale_transform,

        "P2":
            get_grayscale_denoise_transform,

        "P3":
            get_grayscale_denoise_clahe_transform,

        "P4":
            get_grayscale_denoise_threshold_transform,

        "P5":
            get_grayscale_denoise_deskew_transform,
    }

    if PREPROCESSING_VERSION not in pipelines:

        raise ValueError(
            f"Unknown preprocessing pipeline: "
            f"{PREPROCESSING_VERSION}\n"
            f"Available pipelines: "
            f"{list(pipelines.keys())}"
        )

    transform_function = pipelines[
        PREPROCESSING_VERSION
    ]

    transform = transform_function()

    return transform


# ============================================================
# PREPROCESSING NAME
# ============================================================

def get_preprocessing_name():

    names = {

        "P0":
            "RGB Baseline",

        "P1":
            "Grayscale",

        "P2":
            "Grayscale + Denoise",

        "P3":
            "Grayscale + Denoise + CLAHE",

        "P4":
            "Grayscale + Denoise + Otsu",

        "P5":
            "Grayscale + Denoise + Deskew",
    }

    return names[PREPROCESSING_VERSION]


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for batch in progress:

        images = batch[
            0
        ].to(device)

        labels = batch[
            4
        ].to(device)

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            images
        )

        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update model
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for batch in loader:

            images = batch[
                0
            ].to(device)

            labels = batch[
                4
            ].to(device)

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = (
                outputs.argmax(
                    dim=1
                )
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # Set random seed
    # --------------------------------------------------------

    set_seed()

    # --------------------------------------------------------
    # Get device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Get preprocessing transform
    # --------------------------------------------------------

    transform = get_selected_transform()

    preprocessing_name = (
        get_preprocessing_name()
    )

    # --------------------------------------------------------
    # Create pipeline-specific result directory
    # --------------------------------------------------------

    pipeline_results_dir = (
        RESULTS_DIR
        / PREPROCESSING_VERSION
    )

    pipeline_results_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Pipeline-specific checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"best_model_{PREPROCESSING_VERSION}.pth"
    )

    # ========================================================
    # PRINT EXPERIMENT INFORMATION
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "MEDICINE CLASSIFIER TRAINING"
    )

    print(
        "=" * 70
    )

    print(
        "Preprocessing:",
        PREPROCESSING_VERSION
    )

    print(
        "Preprocessing name:",
        preprocessing_name
    )

    print(
        "Device:",
        device
    )

    print(
        "Device name:",
        get_device_name(device)
    )

    print(
        "Epochs:",
        NUM_EPOCHS
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Learning rate:",
        LEARNING_RATE
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY
    )

    print(
        "Seed:",
        SEED
    )

    # ========================================================
    # CREATE DATALOADERS
    # ========================================================

    (
        train_loader,
        validation_loader,
        test_loader,
        class_to_idx,
        idx_to_class
    ) = create_dataloaders(
        batch_size=BATCH_SIZE,
        transform=transform
    )

    num_classes = len(
        class_to_idx
    )

    print(
        "Medicine classes:",
        num_classes
    )

    print(
        "Training samples:",
        len(
            train_loader.dataset
        )
    )

    print(
        "Validation samples:",
        len(
            validation_loader.dataset
        )
    )

    print(
        "Testing samples:",
        len(
            test_loader.dataset
        )
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = MedicineClassifier(
        num_classes=num_classes
    ).to(device)

    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = (
        nn.CrossEntropyLoss()
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # BEST MODEL TRACKING
    # ========================================================

    best_validation_accuracy = 0.0

    best_epoch = 0

    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        NUM_EPOCHS
    ):

        print()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            validation_loss,
            validation_accuracy
        ) = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        epoch_result = {

            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "validation_loss":
                validation_loss,

            "validation_accuracy":
                validation_accuracy
        }

        history.append(
            epoch_result
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if (
            validation_accuracy
            > best_validation_accuracy
        ):

            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = (
                epoch + 1
            )

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "class_to_idx":
                    class_to_idx,

                "idx_to_class":
                    idx_to_class,

                "num_classes":
                    num_classes,

                "image_height":
                    64,

                "image_width":
                    256,

                "validation_accuracy":
                    validation_accuracy,

                "epoch":
                    epoch + 1,

                "preprocessing_version":
                    PREPROCESSING_VERSION,

                "preprocessing_name":
                    preprocessing_name,

                "batch_size":
                    BATCH_SIZE,

                "learning_rate":
                    LEARNING_RATE,

                "weight_decay":
                    WEIGHT_DECAY,

                "seed":
                    SEED
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print(
                "✓ Best model saved:",
                checkpoint_path
            )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    history_path = (
        pipeline_results_dir
        / "training_history.json"
    )

    with open(
        history_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    # ========================================================
    # SAVE EXPERIMENT SUMMARY
    # ========================================================

    experiment_summary = {

        "preprocessing_version":
            PREPROCESSING_VERSION,

        "preprocessing_name":
            preprocessing_name,

        "device":
            str(device),

        "device_name":
            get_device_name(device),

        "epochs":
            NUM_EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "seed":
            SEED,

        "num_classes":
            num_classes,

        "training_samples":
            len(train_loader.dataset),

        "validation_samples":
            len(validation_loader.dataset),

        "testing_samples":
            len(test_loader.dataset),

        "best_validation_accuracy":
            best_validation_accuracy,

        "best_epoch":
            best_epoch,

        "checkpoint":
            str(checkpoint_path),

        "training_history":
            str(history_path)
    }

    summary_path = (
        pipeline_results_dir
        / "experiment_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            experiment_summary,
            file,
            indent=4
        )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Preprocessing:",
        PREPROCESSING_VERSION,
        "-",
        preprocessing_name
    )

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Model:",
        checkpoint_path
    )

    print(
        "History:",
        history_path
    )

    print(
        "Summary:",
        summary_path
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()