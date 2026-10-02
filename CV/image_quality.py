from pathlib import Path

import cv2
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


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image_path(raw_path):
    """
    Resolve dataset CSV image paths relative to the project root.

    Supports paths that contain:
        - Doctor-s_Handwritten_Prescription
        - data/metadata
        - project-relative paths
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
# PORTABLE DATASET PATH
# ============================================================

def get_portable_image_path(image_path):
    """
    Convert an absolute/local image path into a project-relative
    path that can be used on any team member's device.

    Example:

        /Users/name/project/data/metadata/image.png

    becomes:

        data/metadata/image.png
    """

    try:

        relative_path = image_path.relative_to(
            PROJECT_ROOT
        )

        # Use forward slashes so the CSV remains portable
        # across macOS, Windows, and Linux.
        return relative_path.as_posix()

    except ValueError:

        # Fallback in case the image is unexpectedly outside
        # the project directory.
        return image_path.as_posix()


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

            # ------------------------------------------------
            # IMPORTANT:
            # Save a project-relative path instead of the
            # absolute path on the current computer.
            # ------------------------------------------------

            quality["image_path"] = (
                get_portable_image_path(
                    image_path
                )
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

    return pd.DataFrame(
        results
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("IMAGE QUALITY AUDIT")
    print("=" * 60)

    df = build_quality_dataframe()

    print(
        f"Images analysed: {len(df)}"
    )

    # --------------------------------------------------------
    # Display statistics
    # --------------------------------------------------------

    print()
    print("-" * 60)
    print("IMAGE DIMENSIONS")
    print("-" * 60)

    print(
        df[
            ["width", "height"]
        ].describe()
    )

    print()
    print("-" * 60)
    print("IMAGE QUALITY STATISTICS")
    print("-" * 60)

    print(
        df[
            [
                "aspect_ratio",
                "brightness",
                "contrast",
                "blur_score",
                "foreground_ratio",
            ]
        ].describe()
    )

    print()
    print("-" * 60)
    print("BORDER CONTACT")
    print("-" * 60)

    print(
        df[
            "touches_border"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Save portable results
    # --------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "CV"
        / "image_quality"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "image_quality_results.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    print()
    print(
        "Image paths saved as project-relative paths."
    )


if __name__ == "__main__":
    main()