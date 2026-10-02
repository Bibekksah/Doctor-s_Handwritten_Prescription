import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml.dataset import PrescriptionDataset
from CV.preprocessing import get_baseline_transform

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"


def main():
    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)

    transform = get_baseline_transform()
    dataset = PrescriptionDataset(CSV_PATH, split="train", transform=transform)

    print(f"Dataset Size: {len(dataset)}")
    sample = dataset[0]

    print(f"Sample Image Shape: {sample['image'].shape}")
    print(f"Medicine: {sample['medicine_name']}")
    print(f"Generic: {sample['generic_name']}")
    print("=" * 60)


if __name__ == "__main__":
    main()