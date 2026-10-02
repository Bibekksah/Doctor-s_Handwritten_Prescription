import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from CV.preprocessing import (
    get_grayscale_denoise_transform,
    get_grayscale_denoise_threshold_transform,
)


DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "Doctor’s Handwritten Prescription BD dataset"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "CV"
    / "thresholding"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


SAMPLES_PER_SPLIT = 5


denoised_transform = (
    get_grayscale_denoise_transform()
)

threshold_transform = (
    get_grayscale_denoise_threshold_transform()
)


def process_split(
    split_name,
    csv_name,
    image_folder
):

    csv_path = DATASET_ROOT / split_name / csv_name
    image_dir = DATASET_ROOT / split_name / image_folder

    df = pd.read_csv(csv_path)

    print(f"\n{split_name}")
    print(f"Records: {len(df)}")

    samples = df.head(SAMPLES_PER_SPLIT)

    fig, axes = plt.subplots(
        SAMPLES_PER_SPLIT,
        2,
        figsize=(12, 3 * SAMPLES_PER_SPLIT)
    )

    if SAMPLES_PER_SPLIT == 1:
        axes = [axes]

    for row_index, (_, row) in enumerate(
        samples.iterrows()
    ):

        image_name = str(row["IMAGE"])
        medicine_name = str(
            row["MEDICINE_NAME"]
        )

        image_path = image_dir / image_name

        if not image_path.exists():

            print(
                f"WARNING: Image not found: "
                f"{image_path}"
            )

            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Pipeline A
        # Grayscale + Median 3×3
        # ----------------------------------------------------

        denoised = denoised_transform(image)

        denoised_display = (
            denoised.squeeze(0).numpy()
        )

        # ----------------------------------------------------
        # Pipeline B
        # Grayscale + Median 3×3 + Otsu
        # ----------------------------------------------------

        thresholded = threshold_transform(image)

        threshold_display = (
            thresholded.squeeze(0).numpy()
        )

        # ----------------------------------------------------
        # Display Pipeline A
        # ----------------------------------------------------

        axes[row_index][0].imshow(
            denoised_display,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row_index][0].set_title(
            f"Grayscale + Median 3×3\n"
            f"{medicine_name}\n"
            f"Tensor: {tuple(denoised.shape)}"
        )

        axes[row_index][0].axis("off")

        # ----------------------------------------------------
        # Display Pipeline B
        # ----------------------------------------------------

        axes[row_index][1].imshow(
            threshold_display,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row_index][1].set_title(
            f"Grayscale + Median 3×3 + Otsu\n"
            f"{medicine_name}\n"
            f"Tensor: {tuple(thresholded.shape)}"
        )

        axes[row_index][1].axis("off")

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        print(
            f"\nImage: {image_name}"
        )

        print(
            f"Medicine: {medicine_name}"
        )

        print(
            "Denoised -> "
            f"min={denoised.min().item():.4f}, "
            f"max={denoised.max().item():.4f}, "
            f"mean={denoised.mean().item():.4f}"
        )

        print(
            "Otsu     -> "
            f"min={thresholded.min().item():.4f}, "
            f"max={thresholded.max().item():.4f}, "
            f"mean={thresholded.mean().item():.4f}"
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{split_name.lower()}_threshold_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved: {output_path}"
    )


def main():

    print("=" * 60)
    print("OTSU THRESHOLDING VISUAL COMPARISON")
    print("=" * 60)

    print(
        "\nPipeline A:"
        "\nGrayscale → Median 3×3 → Resize + Padding"
    )

    print(
        "\nPipeline B:"
        "\nGrayscale → Median 3×3 → Otsu → "
        "Resize + Padding"
    )

    print(
        "\nTarget CNN tensor:"
        "\n[1, 64, 256]"
    )

    process_split(
        "Training",
        "training_labels.csv",
        "training_words"
    )

    process_split(
        "Validation",
        "validation_labels.csv",
        "validation_words"
    )

    process_split(
        "Testing",
        "testing_labels.csv",
        "testing_words"
    )


if __name__ == "__main__":
    main()