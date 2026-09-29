from ml.dataloader import create_dataloaders


train_loader, val_loader, test_loader = create_dataloaders(
    batch_size=32
)

print("=" * 50)
print("DATALOADER TEST")
print("=" * 50)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))

batch = next(iter(train_loader))

print("\nBatch information:")
print("Images:", batch["image"].shape)
print("Medicine labels:", len(batch["medicine_name"]))
print("Generic labels:", len(batch["generic_name"]))

print("\nFirst sample:")
print("Medicine:", batch["medicine_name"][0])
print("Generic:", batch["generic_name"][0])