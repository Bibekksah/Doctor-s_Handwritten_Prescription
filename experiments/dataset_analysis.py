<<<<<<< HEAD
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "Doctor’s Handwritten Prescription BD dataset"
)

train_csv = DATASET_ROOT / "Training" / "training_labels.csv"
validation_csv = DATASET_ROOT / "Validation" / "validation_labels.csv"
test_csv = DATASET_ROOT / "Testing" / "testing_labels.csv"

train = pd.read_csv(train_csv)
validation = pd.read_csv(validation_csv)
test = pd.read_csv(test_csv)
print("=" * 50)
print("DATASET REPORT")
print("=" * 50)

print("\nTraining samples:", len(train))
print("Validation samples:", len(validation))
print("Testing samples:", len(test))

print("\nTotal samples:", len(train) + len(validation) + len(test))

print("\nNumber of medicine classes:")
print("Training:", train["MEDICINE_NAME"].nunique())
print("Validation:", validation["MEDICINE_NAME"].nunique())
print("Testing:", test["MEDICINE_NAME"].nunique())

print("\nNumber of generic medicines:")
print("Training:", train["GENERIC_NAME"].nunique())
print("Validation:", validation["GENERIC_NAME"].nunique())
print("Testing:", test["GENERIC_NAME"].nunique())

print("\nTop medicine classes:")
print(train["MEDICINE_NAME"].value_counts().head(20))

print("\nMissing values:")
print(train.isnull().sum())

print("\nDuplicate rows:")
print(train.duplicated().sum())

print("\nDataset columns:")
print(train.columns.tolist())

print("===============================================================")
# Check for missing images in our  dataset
print("Checking for missing images in the dataset...")
train_images = DATASET_ROOT / "Training" / "training_words"
val_images = DATASET_ROOT / "Validation" / "validation_words"
test_images = DATASET_ROOT / "Testing" / "testing_words"
=======
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset.csv"
>>>>>>> origin/computervision


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


<<<<<<< HEAD
train_missing = check_images(train, train_images)
validation_missing = check_images(validation, val_images)
test_missing = check_images(test, test_images)

print("\nMissing training images:", len(train_missing))
print("Missing validation images:", len(validation_missing))
print("Missing testing images:", len(test_missing))

print("\n===============================================================")
# checking class distribution in the dataset
print("Checking class distribution in the dataset...")
print("\nTraining class distribution:")
print(train["MEDICINE_NAME"].value_counts())
=======
if __name__ == "__main__":
    main()
>>>>>>> origin/computervision
