import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ml.dataset import PrescriptionDataset
from CV.preprocessing import get_baseline_transform

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"


def main():
    print("=" * 60)
    print("VIEW PREPROCESSED SAMPLES")
    print("=" * 60)

    transform = get_baseline_transform()
    dataset = PrescriptionDataset(CSV_PATH, split="train", transform=transform)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax, i in zip(axes.flat, range(8)):
        sample = dataset[i]
        img = sample["image"].permute(1, 2, 0).numpy()

        ax.imshow(img)
        ax.set_title(sample["medicine_name"], fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    output_path = PROJECT_ROOT / "experiments" / "preprocessed_samples.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    main()