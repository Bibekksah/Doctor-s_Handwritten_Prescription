from io import BytesIO

from PIL import Image

from backend.utils import (
    ImageValidationError,
    validate_image,
)


def create_test_image() -> bytes:
    image = Image.new(
        "RGB",
        (256, 256),
        color="white",
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def main() -> None:
    image_bytes = create_test_image()

    width, height = validate_image(
        file_bytes=image_bytes,
        content_type="image/png",
    )

    print("Image validation: OK")
    print(f"Image size: {width}x{height}")

    try:
        validate_image(
            file_bytes=image_bytes,
            content_type="text/plain",
        )
    except ImageValidationError as exc:
        print("Invalid content type: correctly rejected")
        print(f"Reason: {exc}")


if __name__ == "__main__":
    main()