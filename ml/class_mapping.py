import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"

MAPPING_PATH = PROJECT_ROOT / "ml" / "config" / "class_mapping.json"


def create_class_mapping(csv_path=CSV_PATH, output_path=MAPPING_PATH):

    data = pd.read_csv(csv_path)

    classes = sorted(
        data["medicine_name"]
        .dropna()
        .unique()
        .tolist()
    )

    class_to_idx = {
        name: index
        for index, name in enumerate(classes)
    }

    idx_to_class = {
        str(index): name
        for name, index in class_to_idx.items()
    }

    mapping = {
        "num_classes": len(classes),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            mapping,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("=" * 60)
    print("CLASS MAPPING")
    print("=" * 60)
    print("Number of medicine classes:", len(classes))
    print("Saved:", output_path)

    for index, name in enumerate(classes):
        print(f"{index:3d} -> {name}")


if __name__ == "__main__":
    create_class_mapping()