from ml.dataset import PrescriptionDataset
from CV.preprocessing import get_baseline_transform


CSV_PATH = "data/metadata/dataset.csv"


transform = get_baseline_transform()


dataset = PrescriptionDataset(
    CSV_PATH,
    split="train",
    transform=transform
)


sample = dataset[0]

image_tensor = sample["image"]

print("\nTensor statistics:")
print("Minimum:", image_tensor.min().item())
print("Maximum:", image_tensor.max().item())
print("Mean:", image_tensor.mean().item())

print("=" * 50)
print("PREPROCESSING TEST")
print("=" * 50)

print("Dataset size:", len(dataset))

print("Image tensor:")
print(sample["image"])

print("\nTensor shape:")
print(sample["image"].shape)

print("\nMedicine:")
print(sample["medicine_name"])

print("\nGeneric:")
print(sample["generic_name"])

