import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORT PREPROCESSING PIPELINES
# ============================================================

from CV.preprocessing import (
    get_grayscale_denoise_transform,
    get_grayscale_denoise_clahe_transform,
    get_grayscale_denoise_threshold_transform
)


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "Doctor’s Handwritten Prescription BD dataset"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "CV"
    / "clahe"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLES_PER_SPLIT = 5


# ============================================================
# PREPROCESSING PIPELINES
# ============================================================

grayscale_denoise_transform = (
    get_grayscale_denoise_transform()
)

grayscale_denoise_clahe_transform = (
    get_grayscale_denoise_clahe_transform()
)


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

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

    # Select the same images for both pipelines
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
        medicine_name = str(row["MEDICINE_NAME"])

        image_path = image_dir / image_name

        if not image_path.exists():

            print(
                f"WARNING: Image not found: "
                f"{image_path}"
            )

            continue

        # ----------------------------------------------------
        # Load original image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Pipeline 1:
        # Grayscale + Median 3×3
        # ----------------------------------------------------

        denoised = grayscale_denoise_transform(
            image
        )

        # Tensor shape:
        # [1, 64, 256]

        denoised_display = (
            denoised.squeeze(0).numpy()
        )

        # ----------------------------------------------------
        # Pipeline 2:
        # Grayscale + Median 3×3 + CLAHE
        # ----------------------------------------------------

        clahe = grayscale_denoise_clahe_transform(
            image
        )

        # Tensor shape:
        # [1, 64, 256]

        clahe_display = (
            clahe.squeeze(0).numpy()
        )

        # ----------------------------------------------------
        # Display Pipeline 1
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
        # Display Pipeline 2
        # ----------------------------------------------------

        axes[row_index][1].imshow(
            clahe_display,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row_index][1].set_title(
            f"Grayscale + Median 3×3 + CLAHE\n"
            f"{medicine_name}\n"
            f"Tensor: {tuple(clahe.shape)}"
        )

        axes[row_index][1].axis("off")

        # ----------------------------------------------------
        # Print statistics
        # ----------------------------------------------------

        print(
            f"\nImage: {image_name}"
        )

        print(
            f"Medicine: {medicine_name}"
        )

        print(
            "Denoised  -> "
            f"min={denoised.min().item():.4f}, "
            f"max={denoised.max().item():.4f}, "
            f"mean={denoised.mean().item():.4f}"
        )

        print(
            "CLAHE     -> "
            f"min={clahe.min().item():.4f}, "
            f"max={clahe.max().item():.4f}, "
            f"mean={clahe.mean().item():.4f}"
        )

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{split_name.lower()}_clahe_comparison.png"
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("CLAHE VISUAL COMPARISON")
    print("=" * 60)

    print(
        "\nPipeline A:"
        "\nGrayscale → Median 3×3 → Resize + Padding"
    )

    print(
        "\nPipeline B:"
        "\nGrayscale → Median 3×3 → CLAHE → "
        "Resize + Padding"
    )

    print(
        "\nTarget CNN tensor:"
        "\n[1, 64, 256]"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    process_split(
        "Training",
        "training_labels.csv",
        "training_words"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    process_split(
        "Validation",
        "validation_labels.csv",
        "validation_words"
    )

    # --------------------------------------------------------
    # Testing
    # --------------------------------------------------------

    process_split(
        "Testing",
        "testing_labels.csv",
        "testing_words"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()