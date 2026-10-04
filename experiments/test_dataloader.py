import torch

from ml.dataloader import create_dataloaders


def main():
    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    train_loader, validation_loader, test_loader, class_to_idx, idx_to_class = (
        create_dataloaders(batch_size=32)
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Testing batches: {len(test_loader)}")
    print(f"Medicine classes: {len(class_to_idx)}")

    images, medicine_names, generic_names, image_paths, labels = next(
        iter(train_loader)
    )

    print(f"\nBatch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")

    print(f"First medicine: {medicine_names[0]}")
    print(f"First generic: {generic_names[0]}")
    print(f"First image: {image_paths[0]}")
    print(f"First label: {labels[0].item()}")

    print(f"\nImage dtype: {images.dtype}")
    print(f"Image min: {images.min().item():.4f}")
    print(f"Image max: {images.max().item():.4f}")

    # Verify CNN input format
    assert images.shape[1:] == (3, 64, 256), (
        f"Unexpected image shape: {images.shape}"
    )

    assert labels.shape[0] == images.shape[0]

    assert len(class_to_idx) == 78

    print("\nDataLoader test: PASS")


if __name__ == "__main__":
    main()