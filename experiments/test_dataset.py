from ml.dataset import PrescriptionDataset


CSV_PATH = "data/metadata/dataset.csv"


train_dataset = PrescriptionDataset(
    CSV_PATH,
    split="train"
)

validation_dataset = PrescriptionDataset(
    CSV_PATH,
    split="validation"
)

test_dataset = PrescriptionDataset(
    CSV_PATH,
    split="test"
)


print("=" * 50)
print("DATASET TEST")
print("=" * 50)

print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Testing:", len(test_dataset))


sample = train_dataset[0]

print("\nFirst sample:")
print("Image:", sample["image"])
print("Medicine:", sample["medicine_name"])
print("Generic:", sample["generic_name"])
print("Path:", sample["image_path"])