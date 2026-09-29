import pandas as pd
from pathlib import Path


BASE = Path(
    "data/metadata/Doctor’s Handwritten Prescription BD dataset"
)

datasets = {
    "train": {
        "csv": BASE / "Training" / "training_labels.csv",
        "images": BASE / "Training" / "training_words",
    },
    "validation": {
        "csv": BASE / "Validation" / "validation_labels.csv",
        "images": BASE / "Validation" / "validation_words",
    },
    "test": {
        "csv": BASE / "Testing" / "testing_labels.csv",
        "images": BASE / "Testing" / "testing_words",
    },
}


all_data = []


for split, info in datasets.items():

    df = pd.read_csv(info["csv"])

    for _, row in df.iterrows():

        image_path = info["images"] / str(row["IMAGE"])

        all_data.append({
            "image_path": str(image_path),
            "medicine_name": row["MEDICINE_NAME"],
            "generic_name": row["GENERIC_NAME"],
            "split": split
        })


dataset = pd.DataFrame(all_data)

output_path = Path(r"data/metadata/dataset.csv")

dataset.to_csv(output_path, index=False)

print("=" * 50)
print("DATASET METADATA CREATED")
print("=" * 50)

print(f"Total samples: {len(dataset)}")
print(f"Training: {(dataset['split'] == 'train').sum()}")
print(f"Validation: {(dataset['split'] == 'validation').sum()}")
print(f"Testing: {(dataset['split'] == 'test').sum()}")

print(f"\nMedicine classes: {dataset['medicine_name'].nunique()}")
print(f"Generic classes: {dataset['generic_name'].nunique()}")

print(f"\nSaved to: {output_path}")