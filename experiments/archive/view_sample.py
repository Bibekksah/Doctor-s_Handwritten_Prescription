import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# 1. Resolve Project Root dynamically (Doctor-s_Handwritten_Prescription)
# experiments/ is 1 directory level down from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 2. Build cross-platform path to metadata CSV
CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Metadata CSV not found at: {CSV_PATH}")

# Load standardized metadata
df = pd.read_csv(CSV_PATH)

# Take 9 training samples
samples = df[df["split"] == "train"].head(9)


def resolve_image_path(raw_path: str) -> Path:
    """Resolves relative CSV entries to absolute system paths,

    stripping duplicate root prefixes across OS platforms.
    """
    raw_str = str(raw_path).strip()

    # Strip duplicate folder prefix if stored inside CSV
    if "Doctor-s_Handwritten_Prescription" in raw_str:
        clean_relative = raw_str.split("Doctor-s_Handwritten_Prescription")[
            -1
        ].lstrip("/\\")
    elif "data/metadata" in raw_str:
        clean_relative = raw_str.split("data/metadata")[-1].lstrip("/\\")
        return PROJECT_ROOT / "data" / "metadata" / clean_relative
    else:
        clean_relative = raw_str.lstrip("/\\")

    return PROJECT_ROOT / Path(clean_relative)


fig, axes = plt.subplots(3, 3, figsize=(10, 8))

for ax, (_, row) in zip(axes.flat, samples.iterrows()):
    image_path = resolve_image_path(row["image_path"])

    if not image_path.exists():
        print(f"❌ File Not Found: {image_path}")
        ax.axis("off")
        continue

    # Open image safely using context manager
    with Image.open(image_path) as image:
        ax.imshow(image)

    med_name = row.get("medicine_name", "Unknown")
    gen_name = row.get("generic_name", "Unknown")
    ax.set_title(f"{med_name} → {gen_name}", fontsize=9)
    ax.axis("off")

plt.tight_layout()

# Save output figure inside experiments/
output_path = PROJECT_ROOT / "experiments" / "sample_images.png"
plt.savefig(output_path, dpi=150)
print(f"✅ Sample grid visualization saved to: {output_path}")

plt.show()