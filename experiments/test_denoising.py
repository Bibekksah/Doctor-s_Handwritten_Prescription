from ml.dataset import PrescriptionDataset
from CV.preprocessing import get_grayscale_denoise_transform


CSV_PATH = "data/metadata/dataset.csv"


transform = get_grayscale_denoise_transform()


dataset = PrescriptionDataset(
    CSV_PATH,
    split="train",
    transform=transform
)


sample = dataset[0]

image_tensor = sample["image"]


print("=" * 50)
print("GRAYSCALE + DENOISING TEST")
print("=" * 50)

print("Dataset size:", len(dataset))

print("\nTensor shape:")
print(image_tensor.shape)

print("\nTensor statistics:")
print("Minimum:", image_tensor.min().item())
print("Maximum:", image_tensor.max().item())
print("Mean:", image_tensor.mean().item())

print("\nMedicine:")
print(sample["medicine_name"])

print("\nGeneric:")
print(sample["generic_name"])