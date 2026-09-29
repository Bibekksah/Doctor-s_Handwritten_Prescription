import pandas as pd
from pathlib import Path

# this is our base path to the dataset
BASE = Path(
    "/core/final_year_project2/data/metadata/Doctor’s Handwritten Prescription BD dataset"
)

train_csv = BASE / "Training" / "training_labels.csv"
val_csv = BASE / "Validation" / "validation_labels.csv"
test_csv = BASE / "Testing" / "testing_labels.csv"

train = pd.read_csv(train_csv)
val = pd.read_csv(val_csv)
test = pd.read_csv(test_csv)

print("=" * 50)
print("DATASET REPORT")
print("=" * 50)

print("\nTraining samples:", len(train))
print("Validation samples:", len(val))
print("Testing samples:", len(test))

print("\nTotal samples:", len(train) + len(val) + len(test))

print("\nNumber of medicine classes:")
print("Training:", train["MEDICINE_NAME"].nunique())
print("Validation:", val["MEDICINE_NAME"].nunique())
print("Testing:", test["MEDICINE_NAME"].nunique())

print("\nNumber of generic medicines:")
print("Training:", train["GENERIC_NAME"].nunique())
print("Validation:", val["GENERIC_NAME"].nunique())
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
train_images = BASE / "Training" / "training_words"
val_images = BASE / "Validation" / "validation_words"
test_images = BASE / "Testing" / "testing_words"


def check_images(df, image_dir):
    missing = []

    for image_name in df["IMAGE"]:
        image_path = image_dir / str(image_name)

        if not image_path.exists():
            missing.append(image_name)

    return missing


train_missing = check_images(train, train_images)
val_missing = check_images(val, val_images)
test_missing = check_images(test, test_images)

print("\nMissing training images:", len(train_missing))
print("Missing validation images:", len(val_missing))
print("Missing testing images:", len(test_missing))

print("\n===============================================================")
# checking class distribution in the dataset
print("Checking class distribution in the dataset...")
print("\nTraining class distribution:")
print(train["MEDICINE_NAME"].value_counts())