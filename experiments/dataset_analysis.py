import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"


def main():
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"Total samples: {len(df)}")
    if "split" in df.columns:
        print("\nSplit Distribution:")
        print(df["split"].value_counts())

    if "medicine_name" in df.columns:
        print(f"\nUnique Medicines: {df['medicine_name'].nunique()}")
    if "generic_name" in df.columns:
        print(f"Unique Generics: {df['generic_name'].nunique()}")

    print("=" * 60)


if __name__ == "__main__":
    main()