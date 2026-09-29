from PIL import Image
import torchvision.transforms as transforms


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256


class ResizeWithPadding:

    def __init__(
        self,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH
    ):
        self.height = height
        self.width = width

    def __call__(self, image):

        # Always work with RGB
        image = image.convert("RGB")

        original_width, original_height = image.size

        # Calculate scale while preserving aspect ratio
        scale = min(
            self.width / original_width,
            self.height / original_height
        )

        new_width = max(
            1,
            int(original_width * scale)
        )

        new_height = max(
            1,
            int(original_height * scale)
        )

        # Resize without distorting handwriting
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

        # Create white background
        canvas = Image.new(
            "RGB",
            (self.width, self.height),
            "white"
        )

        # Center the resized image
        x = (self.width - new_width) // 2
        y = (self.height - new_height) // 2

        canvas.paste(image, (x, y))

        return canvas


def get_baseline_transform():

    transform = transforms.Compose([
        ResizeWithPadding(
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH
        ),

        transforms.ToTensor(),
    ])

    return transform