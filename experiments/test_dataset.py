from ml.dataset import PrescriptionDataset
from CV.preprocessing import get_baseline_transform


def main():
    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)

    transform = get_baseline_transform()

    dataset = PrescriptionDataset(
        split="train",
        transform=transform
    )

    print(f"Dataset Size: {len(dataset)}")

    sample = dataset[0]

    image = sample[0]
    medicine_name = sample[1]
    generic_name = sample[2]
    image_path = sample[3]
    label = sample[4]

    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Image Type: {type(image).__name__}")
    print(f"Medicine Name: {medicine_name}")
    print(f"Generic Name: {generic_name}")
    print(f"Image Path: {image_path}")
    print(f"Label: {label}")

    # Expected P0 output
    assert image.shape == (3, 64, 256), (
        f"Unexpected image shape: {image.shape}"
    )

    print("\nDataset test: PASS")


if __name__ == "__main__":
    main()