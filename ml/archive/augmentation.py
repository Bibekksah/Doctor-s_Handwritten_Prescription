import torchvision.transforms as transforms

from CV.preprocessing import ResizeWithPadding


def get_training_transform():
    """
    P1 Grayscale + training-only data augmentation.
    """

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),

        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
        ),

        ResizeWithPadding(
            height=64,
            width=256
        ),

        transforms.ToTensor(),
    ])


def get_validation_transform():
    """
    P1 Grayscale without random augmentation.
    """

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),

        ResizeWithPadding(
            height=64,
            width=256
        ),

        transforms.ToTensor(),
    ])