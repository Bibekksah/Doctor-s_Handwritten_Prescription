from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from CV.preprocessing import (
    get_baseline_transform,
    get_grayscale_transform,
    get_grayscale_denoise_transform,
    get_grayscale_denoise_clahe_transform,
    get_grayscale_denoise_threshold_transform,
    get_grayscale_denoise_deskew_transform,
)

from ml.dataset import resolve_image_path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "CV"
    / "preprocessing_experiments"
)


# ============================================================
# PIPELINES
# ============================================================

PIPELINES = [
    (
        "P0 - RGB Baseline",
        get_baseline_transform,
    ),
    (
        "P1 - Grayscale",
        get_grayscale_transform,
    ),
    (
        "P2 - Grayscale + Denoise",
        get_grayscale_denoise_transform,
    ),
    (
        "P3 - Grayscale + Denoise + CLAHE",
        get_grayscale_denoise_clahe_transform,
    ),
    (
        "P4 - Grayscale + Denoise + Otsu",
        get_grayscale_denoise_threshold_transform,
    ),
    (
        "P5 - Grayscale + Denoise + Deskew",
        get_grayscale_denoise_deskew_transform,
    ),
]


# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================

SAMPLES_PER_SPLIT = 5


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    df = pd.read_csv(
        CSV_PATH
    )

    return df


# ============================================================
# CONVERT TENSOR FOR DISPLAY
# ============================================================

def tensor_to_display_image(tensor):

    tensor = tensor.detach().cpu()

    # RGB tensor: [3, H, W]
    if tensor.shape[0] == 3:

        image = tensor.permute(
            1,
            2,
            0
        ).numpy()

    # Grayscale tensor: [1, H, W]
    else:

        image = tensor.squeeze(0).numpy()

    return image


# ============================================================
# SELECT REPRESENTATIVE SAMPLES
# ============================================================

def select_samples(df, split):

    split_df = (
        df[df["split"] == split]
        .reset_index(drop=True)
    )

    # Select evenly distributed samples rather than
    # simply taking the first five images.
    if len(split_df) <= SAMPLES_PER_SPLIT:
        return split_df

    indices = [
        int(i)
        for i in (
            __import__("numpy")
            .linspace(
                0,
                len(split_df) - 1,
                SAMPLES_PER_SPLIT
            )
        )
    ]

    return split_df.iloc[
        indices
    ].reset_index(drop=True)


# ============================================================
# CREATE COMPARISON FIGURE
# ============================================================

def create_comparison(
    df,
    split
):

    samples = select_samples(
        df,
        split
    )

    figure, axes = plt.subplots(
        len(samples),
        len(PIPELINES),
        figsize=(20, 12)
    )

    # Handle the case where only one sample exists
    if len(samples) == 1:

        axes = axes.reshape(
            1,
            len(PIPELINES)
        )

    for row_index, (_, row) in enumerate(
        samples.iterrows()
    ):

        image_path = resolve_image_path(
            row["image_path"]
        )

        original_image = Image.open(
            image_path
        ).convert("RGB")

        for column_index, (
            pipeline_name,
            transform_function,
        ) in enumerate(PIPELINES):

            transform = transform_function()

            tensor = transform(
                original_image
            )

            display_image = (
                tensor_to_display_image(
                    tensor
                )
            )

            ax = axes[
                row_index,
                column_index
            ]

            if tensor.shape[0] == 3:

                ax.imshow(
                    display_image
                )

            else:

                ax.imshow(
                    display_image,
                    cmap="gray"
                )

            ax.axis("off")

            # Pipeline title only on first row
            if row_index == 0:

                ax.set_title(
                    pipeline_name,
                    fontsize=10
                )

        # Put medicine name on the left side
        axes[
            row_index,
            0
        ].set_ylabel(
            row.get(
                "medicine_name",
                ""
            ),
            fontsize=9,
            rotation=0,
            labelpad=55,
            va="center"
        )

    figure.suptitle(
        f"{split.capitalize()} — Controlled Preprocessing Comparison",
        fontsize=16
    )

    figure.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{split}_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CONTROLLED PREPROCESSING VISUAL COMPARISON")
    print("=" * 70)

    df = load_dataset()

    print(
        f"Total images: {len(df)}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print()
        print(
            f"Creating {split} comparison..."
        )

        create_comparison(
            df,
            split
        )

    print()
    print("=" * 70)
    print("VISUAL COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()