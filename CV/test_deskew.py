from pathlib import Path

from CV.preprocessing import (
    get_grayscale_denoise_deskew_transform,
    estimate_skew_angle,
)

from ml.dataset import PrescriptionDataset
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset.csv"
)


def main():

    transform = get_grayscale_denoise_deskew_transform()

    dataset = PrescriptionDataset(
        csv_file=CSV_PATH,
        split="train",
        transform=transform,
    )

    print("Dataset size:", len(dataset))

    sample = dataset[0]

    print(
        "Tensor shape:",
        sample["image"].shape
    )

    print(
        "Minimum:",
        sample["image"].min().item()
    )

    print(
        "Maximum:",
        sample["image"].max().item()
    )

    print(
        "Mean:",
        sample["image"].mean().item()
    )

    print(
        "Medicine:",
        sample["medicine_name"]
    )

    print(
        "Generic:",
        sample["generic_name"]
    )

    # Inspect estimated angle separately
    image = Image.open(
        sample["image_path"]
    ).convert("L")

    angle = estimate_skew_angle(image)

    print(
        "Estimated skew angle:",
        angle
    )


if __name__ == "__main__":
    main()