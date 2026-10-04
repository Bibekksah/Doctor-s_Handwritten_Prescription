from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset.csv"
)


def resolve_image_path(image_path):
    """Resolve a project-relative image path."""

    path = Path(str(image_path))

    if path.is_absolute() and path.exists():
        return path

    candidate = PROJECT_ROOT / path

    if candidate.exists():
        return candidate

    # Compatibility fallback for duplicated project paths
    parts = path.parts

    for i, part in enumerate(parts):
        if part == "data":
            candidate = PROJECT_ROOT / Path(*parts[i:])

            if candidate.exists():
                return candidate

    return None


def main():

    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    # ---------------------------------------------------------
    # Basic information
    # ---------------------------------------------------------

    print(f"\nMetadata file:")
    print(CSV_PATH)

    print(f"\nTotal samples: {len(df)}")

    print("\nColumns:")
    print(df.columns.tolist())

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    print("\nMissing values:")
    print(df.isnull().sum())

    # ---------------------------------------------------------
    # Split distribution
    # ---------------------------------------------------------

    if "split" in df.columns:

        print("\nSplit Distribution:")
        print(df["split"].value_counts())

    # ---------------------------------------------------------
    # Medicine classes
    # ---------------------------------------------------------

    if "medicine_name" in df.columns:

        print(
            f"\nUnique medicine classes: "
            f"{df['medicine_name'].nunique()}"
        )

        print("\nMedicine class distribution:")

        print(
            df["medicine_name"]
            .value_counts()
            .head(20)
        )

    # ---------------------------------------------------------
    # Generic medicines
    # ---------------------------------------------------------

    if "generic_name" in df.columns:

        print(
            f"\nUnique generic medicines: "
            f"{df['generic_name'].nunique()}"
        )

    # ---------------------------------------------------------
    # Duplicate rows
    # ---------------------------------------------------------

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    # ---------------------------------------------------------
    # Image path validation
    # ---------------------------------------------------------

    print("\nChecking image paths...")

    missing_images = []

    for image_path in df["image_path"]:

        resolved = resolve_image_path(image_path)

        if resolved is None:
            missing_images.append(image_path)

    print(f"Missing images: {len(missing_images)}")

    if missing_images:

        print("\nFirst missing images:")

        for path in missing_images[:10]:
            print(path)

    # ---------------------------------------------------------
    # Per-split analysis
    # ---------------------------------------------------------

    if "split" in df.columns:

        print("\n" + "=" * 60)
        print("PER-SPLIT ANALYSIS")
        print("=" * 60)

        for split_name in ["train", "validation", "test"]:

            split_df = df[df["split"] == split_name]

            if split_df.empty:
                continue

            print(f"\n{split_name.upper()}")

            print(f"Samples: {len(split_df)}")

            print(
                "Medicine classes:",
                split_df["medicine_name"].nunique()
            )

            print(
                "Generic classes:",
                split_df["generic_name"].nunique()
            )

            print(
                "Missing values:",
                split_df.isnull().sum().sum()
            )

            print(
                "Duplicate rows:",
                split_df.duplicated().sum()
            )

    # ---------------------------------------------------------
    # Final status
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 60)

    if len(missing_images) == 0:
        print("Image path check: PASS")
    else:
        print("Image path check: FAIL")

    if df.isnull().sum().sum() == 0:
        print("Missing value check: PASS")
    else:
        print("Missing value check: FAIL")

    if df.duplicated().sum() == 0:
        print("Duplicate check: PASS")
    else:
        print("Duplicate check: FAIL")


if __name__ == "__main__":
    main()