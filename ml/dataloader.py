from pathlib import Path
import platform

import pandas as pd
from torch.utils.data import DataLoader

from ml.dataset import PrescriptionDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset.csv"
)


def get_num_workers():

    system = platform.system()

    if system in ["Windows", "Darwin"]:
        return 0

    return 2


def create_class_mapping():

    data = pd.read_csv(CSV_PATH)

    classes = sorted(
        data["medicine_name"]
        .dropna()
        .unique()
        .tolist()
    )

    class_to_idx = {
        name: index
        for index, name in enumerate(classes)
    }

    idx_to_class = {
        index: name
        for name, index in class_to_idx.items()
    }

    return class_to_idx, idx_to_class


def create_dataloaders(
    batch_size=32,
    train_transform=None,
    validation_transform=None,
    test_transform=None,
):
    if train_transform is None:
        from CV.preprocessing import get_grayscale_transform
        train_transform = get_grayscale_transform()

    if validation_transform is None:
        from CV.preprocessing import get_grayscale_transform
        validation_transform = get_grayscale_transform()

    if test_transform is None:
        from CV.preprocessing import get_grayscale_transform
        test_transform = get_grayscale_transform()

    class_to_idx, idx_to_class = create_class_mapping()

    train_dataset = PrescriptionDataset(
        CSV_PATH,
        split="train",
        transform=train_transform,
        class_to_idx=class_to_idx
    )

    validation_dataset = PrescriptionDataset(
        CSV_PATH,
        split="validation",
        transform=validation_transform,
        class_to_idx=class_to_idx
    )

    test_dataset = PrescriptionDataset(
        CSV_PATH,
        split="test",
        transform=test_transform,
        class_to_idx=class_to_idx
    )

    num_workers = get_num_workers()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        class_to_idx,
        idx_to_class
    )