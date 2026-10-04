import torch

from ml.model import MedicineClassifier
from ml.device import get_device


def main():

    device = get_device()

    model = MedicineClassifier(
        num_classes=78
    ).to(device)

    dummy_input = torch.randn(
        2,
        3,
        64,
        256
    ).to(device)

    output = model(dummy_input)

    print("=" * 60)
    print("MODEL TEST")
    print("=" * 60)

    print("Device:", device)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    assert output.shape == (2, 78)

    print("Model test: PASS")


if __name__ == "__main__":
    main()