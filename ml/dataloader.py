from torch.utils.data import DataLoader

from ml.dataset import PrescriptionDataset
from CV.preprocessing import get_baseline_transform


CSV_PATH = "data/metadata/dataset.csv"


def create_dataloaders(batch_size=32):

    transform = get_baseline_transform()

    train_dataset = PrescriptionDataset(
        CSV_PATH,
        split="train",
        transform=transform
    )

    validation_dataset = PrescriptionDataset(
        CSV_PATH,
        split="validation",
        transform=transform
    )

    test_dataset = PrescriptionDataset(
        CSV_PATH,
        split="test",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, validation_loader, test_loader