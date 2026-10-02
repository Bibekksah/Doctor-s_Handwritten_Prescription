from ml.dataloader import create_dataloaders


def main():

    (
        train_loader,
        val_loader,
        test_loader,
        class_to_idx,
        idx_to_class
    ) = create_dataloaders(
        batch_size=32
    )

    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))

    print("Medicine classes:", len(class_to_idx))

    batch = next(iter(train_loader))

    print("\nBatch information:")

    print(
        "Images:",
        batch["image"].shape
    )

    print(
        "Labels:",
        batch["label"].shape
    )

    print(
        "First medicine:",
        batch["medicine_name"][0]
    )

    print(
        "First label:",
        batch["label"][0].item()
    )

    print(
        "First generic:",
        batch["generic_name"][0]
    )

    assert len(class_to_idx) == 78

    assert batch["image"].shape[1:] == (
        3,
        64,
        256
    )

    assert batch["label"].shape[0] == 32

    print("\nDATALOADER TEST: PASS")


if __name__ == "__main__":
    main()