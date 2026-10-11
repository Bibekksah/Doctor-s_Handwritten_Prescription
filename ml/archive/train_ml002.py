from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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

RESULTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ML002"
)


SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:

        images = batch[0].to(device)
        labels = batch[4].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            images = batch[0].to(device)
            labels = batch[4].to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():

    set_seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 80)
    print("ML-002: IMPROVED CNN ARCHITECTURE")
    print("P1 GRAYSCALE")
    print("=" * 80)

    print(f"Device: {device}")
    print(f"Device name: {get_device_name(device)}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Seed: {SEED}")

    # P1 grayscale for ALL splits.
    # No random augmentation.
    transform = get_grayscale_transform()

    (
        train_loader,
        validation_loader,
        test_loader,
        class_to_idx,
        idx_to_class,
    ) = create_dataloaders(
        batch_size=BATCH_SIZE,
        train_transform=transform,
        validation_transform=transform,
        test_transform=transform,
    )

    print(f"Medicine classes: {len(class_to_idx)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}")

    model = ImprovedMedicineClassifier(
        num_classes=len(class_to_idx)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_validation_accuracy = 0.0
    best_epoch = 0

    history = []

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}: "
            f"train loss {train_loss:.4f}, "
            f"train acc {train_accuracy * 100:.2f}%, "
            f"val loss {validation_loss:.4f}, "
            f"val acc {validation_accuracy * 100:.2f}%"
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "idx_to_class": idx_to_class,
                    "best_validation_accuracy": best_validation_accuracy,
                    "best_epoch": best_epoch,
                    "experiment": "ML002",
                },
                CHECKPOINT_PATH,
            )

            print(
                f"  -> Best model saved "
                f"({best_validation_accuracy * 100:.2f}%)"
            )

    history_path = RESULTS_DIR / "training_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    summary = {
        "experiment": "ML002",
        "description": "P1 Grayscale + Improved CNN Architecture",
        "num_classes": len(class_to_idx),
        "training_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "testing_samples": len(test_loader.dataset),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "best_validation_accuracy": best_validation_accuracy,
        "best_epoch": best_epoch,
        "device": str(device),
        "device_name": get_device_name(device),
    }

    summary_path = RESULTS_DIR / "experiment_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 80)
    print("ML-002 TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(f"Best epoch: {best_epoch}")

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()