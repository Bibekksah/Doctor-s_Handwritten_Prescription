from pathlib import Path
import cv2

from ml.ocr import PrescriptionOCR


PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "Doctor’s Handwritten Prescription BD dataset"
    / "Testing"
    / "testing_words"
    / "0.png"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "results" / "ocr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("OCR MODULE TEST")
print("=" * 70)

print(f"Input image:")
print(IMAGE_PATH)

ocr = PrescriptionOCR()

image = ocr.load_image(IMAGE_PATH)

print()
print(f"Image size: {image.shape[1]} x {image.shape[0]}")

regions = ocr.detect_regions(image)

print()
print(f"Detected regions: {len(regions)}")

for i, region in enumerate(regions):
    print(
        f"Region {i}: "
        f"x={region['x']}, "
        f"y={region['y']}, "
        f"w={region['width']}, "
        f"h={region['height']}"
    )


# Draw detected regions
visualization = ocr.draw_regions(
    image,
    regions,
)

output_path = OUTPUT_DIR / "ocr_regions.png"

cv2.imwrite(
    str(output_path),
    visualization,
)

print()
print(f"Visualization saved to:")
print(output_path)


# Extract crops
crops = ocr.crop_regions(
    image,
    regions,
)

print()
print(f"Extracted crops: {len(crops)}")


for crop in crops:

    region_id = crop["region_id"]

    crop_path = (
        OUTPUT_DIR
        / f"region_{region_id}.png"
    )

    cv2.imwrite(
        str(crop_path),
        crop["image"],
    )

    print(
        f"Saved region {region_id}: "
        f"{crop_path}"
    )


print()
print("=" * 70)
print("OCR MODULE TEST COMPLETE")
print("=" * 70)