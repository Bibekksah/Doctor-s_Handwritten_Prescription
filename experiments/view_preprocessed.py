import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml.dataset import PrescriptionDataset

from CV.preprocessing import (
    get_baseline_transform,
    get_grayscale_transform,
    get_grayscale_denoise_transform,
    get_grayscale_denoise_clahe_transform,
)

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"


def main():
    print("=" * 60)
    print("PREPROCESSING COMPARISON")
    print("=" * 60)

    baseline_transform = get_baseline_transform()
    grayscale_transform = get_grayscale_transform()
    denoise_transform = get_grayscale_denoise_transform()

    dataset_baseline = PrescriptionDataset(
        CSV_PATH,
        split="train",
        transform=baseline_transform
    )

    dataset_grayscale = PrescriptionDataset(
        CSV_PATH,
        split="train",
        transform=grayscale_transform
    )

    dataset_denoise = PrescriptionDataset(
        CSV_PATH,
        split="train",
        transform=denoise_transform
    )

    dataset_clahe = PrescriptionDataset(
        CSV_PATH,
        split="train",
        transform=get_grayscale_denoise_clahe_transform()
    )

    fig, axes = plt.subplots(
        8,
        4,
        figsize=(16, 14)
    )

    for i in range(8):

        baseline_sample = dataset_baseline[i]
        grayscale_sample = dataset_grayscale[i]
        denoise_sample = dataset_denoise[i]
        clahe_sample = dataset_clahe[i]

        baseline_img = (
            baseline_sample["image"]
            .permute(1, 2, 0)
            .numpy()
        )

        grayscale_img = (
            grayscale_sample["image"]
            .squeeze(0)
            .numpy()
        )

        denoise_img = (
            denoise_sample["image"]
            .squeeze(0)
            .numpy()
        )

        clahe_img = (
            clahe_sample["image"]
            .squeeze(0)
            .numpy()
        )

        # Baseline RGB
        axes[i, 0].imshow(baseline_img)

        # Grayscale
        axes[i, 1].imshow(
            grayscale_img,
            cmap="gray"
        )

        # Grayscale + Median 3x3
        axes[i, 2].imshow(
            denoise_img,
            cmap="gray"
        )

        axes[i, 3].imshow(
            clahe_img,
            cmap="gray"
        )

        medicine_name = baseline_sample["medicine_name"]

        axes[i, 0].set_ylabel(
            medicine_name,
            fontsize=8
        )

        for column in range(4):
            axes[i, column].axis("off")

    axes[0, 0].set_title(
        "B0: Baseline RGB",
        fontsize=10
    )

    axes[0, 1].set_title(
        "G1: Grayscale",
        fontsize=10
    )

    axes[0, 2].set_title(
        "N1: Grayscale + Median 3×3",
        fontsize=10
    )

    axes[0, 3].set_title(
        "C1: Grayscale + Median 3×3 + CLAHE",
        fontsize=10
    )

    plt.tight_layout()

    output_path = (
        PROJECT_ROOT
        / "experiments"
        / "preprocessing_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150
    )

    print(f"Saved visualization to:")
    print(output_path)


if __name__ == "__main__":
    main()