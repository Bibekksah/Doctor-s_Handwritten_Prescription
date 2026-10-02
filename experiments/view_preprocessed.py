import sys
from pathlib import Path
<<<<<<< HEAD

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
            
=======
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml.dataset import PrescriptionDataset
<<<<<<< HEAD

from CV.preprocessing import (
    get_baseline_transform,
    get_grayscale_transform,
    get_grayscale_denoise_transform,
    get_grayscale_denoise_clahe_transform,
)

=======
from CV.preprocessing import get_baseline_transform
>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"


def main():
<<<<<<< HEAD

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

    fig, axes = plt.subplots(
        8,
        3,
        figsize=(12, 14)
    )

    for i in range(8):

        baseline_sample = dataset_baseline[i]
        grayscale_sample = dataset_grayscale[i]
        denoise_sample = dataset_denoise[i]
        dataset_clahe = PrescriptionDataset(
        CSV_PATH,
        split="train",
        transform=get_grayscale_denoise_clahe_transform()
        )

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

        medicine_name = baseline_sample["medicine_name"]

        axes[i, 0].set_ylabel(
            medicine_name,
            fontsize=8
        )

        for column in range(3):
            axes[i, column].axis("off")

        clahe_sample = dataset_clahe[i]

        clahe_img = (
            clahe_sample["image"]
            .squeeze(0)
            .numpy()
        )

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
=======
    print("=" * 60)
    print("VIEW PREPROCESSED SAMPLES")
    print("=" * 60)

    transform = get_baseline_transform()
    dataset = PrescriptionDataset(CSV_PATH, split="train", transform=transform)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, i in zip(axes.flat, range(8)):
        sample = dataset[i]
        img = sample["image"].permute(1, 2, 0).numpy()

        ax.imshow(img)
        ax.set_title(sample["medicine_name"], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    output_path = PROJECT_ROOT / "experiments" / "preprocessed_samples.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")
>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687


if __name__ == "__main__":
    main()