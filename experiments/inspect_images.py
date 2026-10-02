from pathlib import Path
import pandas as pd
from PIL import Image

# 1. Dynamically find the project root directory
# __file__ is experiments/inspect_images.py -> .parents[1] goes up to Doctor-s_Handwritten_Prescription/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. Construct cross-platform path to dataset.csv
csv_path = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"

# Load metadata
df = pd.read_csv(csv_path)

print("=" * 60)
print("IMAGE INSPECTION")
print("=" * 60)


def get_absolute_image_path(raw_path: str) -> Path:
    """Resolves relative CSV paths to absolute system paths, preventing duplicate directory errors across OS platforms."""
    raw_str = str(raw_path).strip()

    # Strip duplicate folder prefix if present in CSV
    if "Doctor-s_Handwritten_Prescription" in raw_str:
        clean_relative = raw_str.split("Doctor-s_Handwritten_Prescription")[
            -1
        ].lstrip("/\\")
    else:
        clean_relative = raw_str.lstrip("/\\")

    # Path object converts slashes automatically for macOS/Linux (/) or Windows (\)
    return PROJECT_ROOT / Path(clean_relative)


# Inspect first 10 images
for _, row in df.head(10).iterrows():

    # Clean and resolve path for current OS
    image_path = get_absolute_image_path(row["image_path"])

    if not image_path.exists():
        print(f"\n[ERROR] File not found at: {image_path}")
        continue

    # Load image safely
    with Image.open(image_path) as image:
        print("\nImage:", image_path.name)
        print("Full Path:", image_path)
        print("Medicine:", row["medicine_name"])
        print("Generic:", row["generic_name"])
        print("Size:", image.size)
        print("Mode:", image.mode)


print("\n" + "=" * 60)
print("IMAGE INSPECTION COMPLETE")
print("=" * 60)