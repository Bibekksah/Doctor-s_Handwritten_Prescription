import pandas as pd
from pathlib import Path

CSV_PATH = Path("data/metadata/dataset.csv")

df = pd.read_csv(CSV_PATH)

print("=" * 50)
print("METADATA VALIDATION")
print("=" * 50)

print(f"Total records: {len(df)}")

missing = []

for path in df["image_path"]:
    if not Path(path).exists():
        missing.append(path)

print(f"Missing images: {len(missing)}")

if missing:
    print("\nFirst missing files:")
    for path in missing[:10]:
        print(path)
else:
    print("All image paths are valid.")

print("\nSplit distribution:")
print(df["split"].value_counts())

print("\nMedicine classes:")
print(df["medicine_name"].nunique())

print("\nGeneric classes:")
print(df["generic_name"].nunique())