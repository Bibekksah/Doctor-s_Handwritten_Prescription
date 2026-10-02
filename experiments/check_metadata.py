import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"


def main():
    print("=" * 60)
    print("METADATA INTEGRITY CHECK")
    print("=" * 60)

    df = pd.read_csv(CSV_PATH)

    print(f"Columns: {list(df.columns)}")
    print(f"Missing Values:\n{df.isnull().sum()}")

    # Check for empty string paths or missing critical fields
    missing_paths = df["image_path"].isnull().sum()
    print(f"\nMissing Image Paths: {missing_paths}")

    print("=" * 60)


if __name__ == "__main__":
    main()