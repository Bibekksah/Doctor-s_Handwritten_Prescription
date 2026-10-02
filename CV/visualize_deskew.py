from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

from CV.preprocessing import (
    deskew_image,
    ResizeWithPadding,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "Doctor’s Handwritten Prescription BD dataset"
)


SPLITS = {
    "training": "Training/training_words",
    "validation": "Validation/validation_words",
    "testing": "Testing/testing_words",
}


# ============================================================
# PREPROCESSING WITHOUT DESKEW
# ============================================================

def preprocess_without_deskew(image):

    image = image.convert("L")

    image = image.filter(
        ImageFilter.MedianFilter(size=3)
    )

    image = ResizeWithPadding(
        height=64,
        width=256,
        mode="L"
    )(image)

    return image


# ============================================================
# PREPROCESSING WITH DESKEW
# ============================================================

def preprocess_with_deskew(image):

    image = image.convert("L")

    image = image.filter(
        ImageFilter.MedianFilter(size=3)
    )

    image = deskew_image(image)

    image = ResizeWithPadding(
        height=64,
        width=256,
        mode="L"
    )(image)

    return image


# ============================================================
# GET SAMPLE IMAGES
# ============================================================

def get_images(folder, count=5):

    image_files = sorted(
        [
            path
            for path in folder.iterdir()
            if path.suffix.lower()
            in {".png", ".jpg", ".jpeg"}
        ]
    )

    return image_files[:count]


# ============================================================
# CREATE DESKEW COMPARISON
# ============================================================

def create_comparison(split_name, relative_folder):

    folder = DATASET_ROOT / relative_folder

    image_files = get_images(folder)

    if not image_files:

        print(
            f"No images found for {split_name}"
        )

        return

    fig, axes = plt.subplots(
        len(image_files),
        2,
        figsize=(12, 3 * len(image_files))
    )

    if len(image_files) == 1:
        axes = [axes]

    for row, image_path in enumerate(image_files):

        original = Image.open(
            image_path
        ).convert("RGB")

        without_deskew = preprocess_without_deskew(
            original
        )

        with_deskew = preprocess_with_deskew(
            original
        )

        # ----------------------------------------------------
        # Without deskew
        # ----------------------------------------------------

        axes[row][0].imshow(
            without_deskew,
            cmap="gray"
        )

        axes[row][0].set_title(
            f"{image_path.name}\n"
            "Denoised"
        )

        # ----------------------------------------------------
        # With deskew
        # ----------------------------------------------------

        axes[row][1].imshow(
            with_deskew,
            cmap="gray"
        )

        axes[row][1].set_title(
            "Denoised + Deskew"
        )

        axes[row][0].axis("off")
        axes[row][1].axis("off")

    fig.suptitle(
        f"{split_name.capitalize()} - Deskew Comparison",
        fontsize=14
    )

    fig.tight_layout()

    # ========================================================
    # SAVE INSIDE CV/deskew
    # ========================================================

    output_dir = (
        PROJECT_ROOT
        / "CV"
        / "deskew"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"{split_name}_deskew_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    for split_name, folder in SPLITS.items():

        create_comparison(
            split_name,
            folder
        )


if __name__ == "__main__":
    main()