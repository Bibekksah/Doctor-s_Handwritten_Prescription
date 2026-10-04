from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ml.dataloader import create_dataloaders
from ml.model import MedicineClassifier
from ml.device import get_device, get_device_name
from ml.augmentation import (
    get_training_transform,
    get_validation_transform,
)


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

EXPERIMENT_NAME = "ML001"

PREPROCESSING_VERSION = "P1"

PREPROCESSING_NAME = "Grayscale"

AUGMENTATION_NAME = (
    "RandomAffine"
)

NUM_EPOCHS = 20

BATCH_SIZE = 32

LEARNING_RATE = 0.001

WEIGHT_DECAY = 0.0001

SEED = 42


# ============================================================
# OUTPUT PATHS
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXPERIMENT_RESULTS_DIR = (
    RESULTS_DIR
    / EXPERIMENT_NAME
)

EXPERIMENT_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / f"best_model_{EXPERIMENT_NAME}.pth"
)


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

        images = batch[0].to(device)

        labels = batch[4].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
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

            images = batch[0].to(device)

            labels = batch[4].to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
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
# MAIN
# ============================================================

def main():

    set_seed()

    device = get_device()

    print("=" * 70)
    print("ML-001: P1 GRAYSCALE + DATA AUGMENTATION")
    print("=" * 70)

    print(
        "Experiment:",
        EXPERIMENT_NAME
    )

    print(
        "Preprocessing:",
        PREPROCESSING_NAME
    )

    print(
        "Augmentation:",
        AUGMENTATION_NAME
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
    # TRANSFORMS
    # ========================================================

    train_transform = (
        get_training_transform()
    )

    validation_transform = (
        get_validation_transform()
    )

    test_transform = (
        get_validation_transform()
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    (
        train_loader,
        validation_loader,
        test_loader,
        class_to_idx,
        idx_to_class
    ) = create_dataloaders(
        batch_size=BATCH_SIZE,

        train_transform=train_transform,

        validation_transform=validation_transform,

        test_transform=test_transform
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
        len(train_loader.dataset)
    )

    print(
        "Validation samples:",
        len(validation_loader.dataset)
    )

    print(
        "Testing samples:",
        len(test_loader.dataset)
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = MedicineClassifier(
        num_classes=num_classes
    ).to(device)

    # ========================================================
    # LOSS
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
    # TRACKING
    # ========================================================

    best_validation_accuracy = 0.0

    best_epoch = 0

    history = []

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        print()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        validation_loss, validation_accuracy = (
            validate(
                model,
                validation_loader,
                criterion,
                device
            )
        )

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

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        })

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

            best_epoch = epoch + 1

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

                "experiment":
                    EXPERIMENT_NAME,

                "preprocessing_version":
                    PREPROCESSING_VERSION,

                "preprocessing_name":
                    PREPROCESSING_NAME,

                "augmentation":
                    AUGMENTATION_NAME,

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
                CHECKPOINT_PATH
            )

            print(
                "✓ Best model saved:",
                CHECKPOINT_PATH
            )

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history_path = (
        EXPERIMENT_RESULTS_DIR
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
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "experiment":
            EXPERIMENT_NAME,

        "preprocessing_version":
            PREPROCESSING_VERSION,

        "preprocessing_name":
            PREPROCESSING_NAME,

        "augmentation":
            AUGMENTATION_NAME,

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
            str(CHECKPOINT_PATH),

        "training_history":
            str(history_path)
    }

    summary_path = (
        EXPERIMENT_RESULTS_DIR
        / "experiment_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print()
    print("=" * 70)
    print("ML-001 TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "History:",
        history_path
    )

    print(
        "Summary:",
        summary_path
    )


if __name__ == "__main__":
    main()