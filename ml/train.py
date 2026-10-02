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


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "ml" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "ml" / "results"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
SEED = 42


def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


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

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def main():

    set_seed()

    device = get_device()

    print("=" * 70)
    print("MEDICINE CLASSIFIER TRAINING")
    print("=" * 70)

    print("Device:", device)
    print("Device name:", get_device_name(device))
    print("Epochs:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)

    (
        train_loader,
        validation_loader,
        test_loader,
        class_to_idx,
        idx_to_class
    ) = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    num_classes = len(class_to_idx)

    print("Medicine classes:", num_classes)
    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(validation_loader.dataset))
    print("Testing samples:", len(test_loader.dataset))

    model = MedicineClassifier(
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_validation_accuracy = 0.0

    history = []

    for epoch in range(NUM_EPOCHS):

        print()
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f}"
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

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        }

        history.append(epoch_result)

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
                "num_classes": num_classes,
                "image_height": 64,
                "image_width": 256,
                "validation_accuracy": validation_accuracy,
                "epoch": epoch + 1
            }

            checkpoint_path = (
                CHECKPOINT_DIR / "best_model.pth"
            )

            torch.save(
                checkpoint,
                checkpoint_path
            )

            print(
                "✓ Best model saved:",
                checkpoint_path
            )

    history_path = (
        RESULTS_DIR / "training_history.json"
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

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        "Model:",
        CHECKPOINT_DIR / "best_model.pth"
    )

    print(
        "History:",
        history_path
    )


if __name__ == "__main__":
    main()