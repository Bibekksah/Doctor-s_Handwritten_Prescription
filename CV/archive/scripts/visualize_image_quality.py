from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================
# PROJECT PATHS
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
    / "image_quality"
)


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image_path(raw_path):
    """
    Resolve dataset CSV image paths relative to the project root.
    """

    raw_str = str(raw_path).strip()

    if "Doctor-s_Handwritten_Prescription" in raw_str:

        clean_relative = (
            raw_str
            .split("Doctor-s_Handwritten_Prescription")[-1]
            .lstrip("/\\")
        )

    elif "data/metadata" in raw_str:

        clean_relative = (
            raw_str
            .split("data/metadata")[-1]
            .lstrip("/\\")
        )

        return (
            PROJECT_ROOT
            / "data"
            / "metadata"
            / clean_relative
        )

    else:

        clean_relative = raw_str.lstrip("/\\")

    return PROJECT_ROOT / Path(clean_relative)


# ============================================================
# IMAGE QUALITY CALCULATION
# ============================================================

def calculate_image_quality(image_path):

    image = Image.open(
        image_path
    ).convert("L")

    image_array = np.array(image)

    height, width = image_array.shape

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = float(
        np.mean(image_array)
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = float(
        np.std(image_array)
    )

    # --------------------------------------------------------
    # Blur score
    # --------------------------------------------------------

    blur_score = float(
        cv2.Laplacian(
            image_array,
            cv2.CV_64F
        ).var()
    )

    # --------------------------------------------------------
    # Temporary foreground mask
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        image_array,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    foreground_pixels = np.count_nonzero(
        binary
    )

    foreground_ratio = (
        foreground_pixels
        / image_array.size
    )

    # --------------------------------------------------------
    # Border contact
    # --------------------------------------------------------

    border = max(
        1,
        int(min(height, width) * 0.02)
    )

    border_region = np.zeros_like(
        binary,
        dtype=np.uint8
    )

    border_region[:border, :] = 1
    border_region[-border:, :] = 1
    border_region[:, :border] = 1
    border_region[:, -border:] = 1

    touches_border = bool(
        np.any(
            (binary > 0)
            & (border_region > 0)
        )
    )

    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    aspect_ratio = (
        width / height
    )

    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "foreground_ratio": foreground_ratio,
        "touches_border": touches_border,
    }


# ============================================================
# BUILD QUALITY DATAFRAME
# ============================================================

def build_quality_dataframe():

    df = pd.read_csv(
        CSV_PATH
    )

    results = []

    for _, row in df.iterrows():

        image_path = resolve_image_path(
            row["image_path"]
        )

        try:

            quality = calculate_image_quality(
                image_path
            )

            quality["image_path"] = str(
                image_path
            )

            quality["medicine_name"] = row.get(
                "medicine_name",
                ""
            )

            quality["split"] = row.get(
                "split",
                ""
            )

            results.append(
                quality
            )

        except Exception as error:

            print(
                f"Skipping {image_path}: {error}"
            )

    return pd.DataFrame(results)


# ============================================================
# SELECT REPRESENTATIVE IMAGES
# ============================================================

def select_samples(df):

    samples = []

    # --------------------------------------------------------
    # Lowest / highest blur
    # --------------------------------------------------------

    samples.append(
        ("Lowest Blur", df.nsmallest(1, "blur_score").iloc[0])
    )

    samples.append(
        ("Highest Blur Score", df.nlargest(1, "blur_score").iloc[0])
    )

    # --------------------------------------------------------
    # Lowest / highest contrast
    # --------------------------------------------------------

    samples.append(
        ("Lowest Contrast", df.nsmallest(1, "contrast").iloc[0])
    )

    samples.append(
        ("Highest Contrast", df.nlargest(1, "contrast").iloc[0])
    )

    # --------------------------------------------------------
    # Lowest / highest brightness
    # --------------------------------------------------------

    samples.append(
        ("Lowest Brightness", df.nsmallest(1, "brightness").iloc[0])
    )

    samples.append(
        ("Highest Brightness", df.nlargest(1, "brightness").iloc[0])
    )

    # --------------------------------------------------------
    # Lowest / highest foreground ratio
    # --------------------------------------------------------

    samples.append(
        (
            "Lowest Foreground Ratio",
            df.nsmallest(1, "foreground_ratio").iloc[0]
        )
    )

    samples.append(
        (
            "Highest Foreground Ratio",
            df.nlargest(1, "foreground_ratio").iloc[0]
        )
    )

    # --------------------------------------------------------
    # Lowest / highest aspect ratio
    # --------------------------------------------------------

    samples.append(
        (
            "Lowest Aspect Ratio",
            df.nsmallest(1, "aspect_ratio").iloc[0]
        )
    )

    samples.append(
        (
            "Highest Aspect Ratio",
            df.nlargest(1, "aspect_ratio").iloc[0]
        )
    )

    # --------------------------------------------------------
    # Border contact example
    # --------------------------------------------------------

    border_samples = df[
        df["touches_border"] == True
    ]

    if len(border_samples) > 0:

        samples.append(
            (
                "Border Contact",
                border_samples.iloc[0]
            )
        )

    return samples


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(samples):

    rows = len(samples)

    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(14, 3.5 * rows)
    )

    if rows == 1:
        axes = [axes]

    for ax, (label, row) in zip(
        axes,
        samples
    ):

        image_path = Path(
            row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("L")

        ax.imshow(
            image,
            cmap="gray"
        )

        title = (
            f"{label} | "
            f"{image_path.name}\n"
            f"Size: {int(row['width'])}×{int(row['height'])} | "
            f"Aspect: {row['aspect_ratio']:.2f} | "
            f"Brightness: {row['brightness']:.2f} | "
            f"Contrast: {row['contrast']:.2f} | "
            f"Blur: {row['blur_score']:.2f} | "
            f"Foreground: {row['foreground_ratio']:.3f} | "
            f"Border: {row['touches_border']}"
        )

        ax.set_title(
            title,
            fontsize=9
        )

        ax.axis("off")

    fig.suptitle(
        "Image Quality - Representative Extremes",
        fontsize=16
    )

    fig.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "image_quality_extremes.png"
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

    print("=" * 60)
    print("IMAGE QUALITY VISUAL ANALYSIS")
    print("=" * 60)

    df = build_quality_dataframe()

    print(
        f"Images analysed: {len(df)}"
    )

    samples = select_samples(
        df
    )

    print(
        f"Representative images selected: {len(samples)}"
    )

    create_contact_sheet(
        samples
    )


if __name__ == "__main__":
    main()