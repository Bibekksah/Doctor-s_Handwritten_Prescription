import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from ml.dataset import PrescriptionDataset
from CV.preprocessing import (
    get_grayscale_denoise_threshold_transform
)


CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset.csv"
)


def main():

    print("=" * 60)
    print("GRAYSCALE + DENOISING + OTSU THRESHOLDING TEST")
    print("=" * 60)

    transform = (
        get_grayscale_denoise_threshold_transform()
    )

    dataset = PrescriptionDataset(
        CSV_PATH,
        split="train",
        transform=transform
    )

    sample = dataset[0]

    image_tensor = sample["image"]

    print(
        "Dataset size:",
        len(dataset)
    )

    print("\nTensor shape:")
    print(image_tensor.shape)

    print("\nTensor statistics:")
    print(
        "Minimum:",
        image_tensor.min().item()
    )

    print(
        "Maximum:",
        image_tensor.max().item()
    )

    print(
        "Mean:",
        image_tensor.mean().item()
    )

    print(
        "Unique pixel values:",
        image_tensor.unique().tolist()
    )

    print("\nMedicine:")
    print(sample["medicine_name"])

    print("\nGeneric:")
    print(sample["generic_name"])


if __name__ == "__main__":
    main()