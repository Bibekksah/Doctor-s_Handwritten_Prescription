from io import BytesIO

from PIL import Image


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

MIN_IMAGE_WIDTH = 32
MIN_IMAGE_HEIGHT = 32

MAX_IMAGE_WIDTH = 10000
MAX_IMAGE_HEIGHT = 10000


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""


def validate_image(
    file_bytes: bytes,
    content_type: str,
) -> tuple[int, int]:
    """
    Validate an uploaded image.

    Returns:
        (width, height)

    Raises:
        ImageValidationError:
            If the image is invalid or violates configured limits.
    """

    if not file_bytes:
        raise ImageValidationError("Uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ImageValidationError(
            f"Image exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)} MB."
        )

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            "Unsupported image type. Only JPEG and PNG are allowed."
        )

    try:
        with Image.open(BytesIO(file_bytes)) as image:
            image.verify()

        with Image.open(BytesIO(file_bytes)) as image:
            width, height = image.size

    except Exception as exc:
        raise ImageValidationError(
            "Uploaded file is not a valid image."
        ) from exc

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        raise ImageValidationError(
            f"Image dimensions are too small. "
            f"Minimum size is {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}."
        )

    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        raise ImageValidationError(
            f"Image dimensions are too large. "
            f"Maximum size is {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}."
        )

    return width, height