from pathlib import Path
import json
import time

from PIL import Image
from paddleocr import PaddleOCR


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Full prescription image
INPUT_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "ocr_samples"
    / "images.jpeg"
)

# OCR output directory
OUTPUT_DIR = PROJECT_ROOT / "ml" / "results" / "ocr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OCR_JSON = OUTPUT_DIR / "existing_ocr_result.json"
OCR_VISUALIZATION = OUTPUT_DIR / "ocr_regions.png"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OCR MODULE TEST")
    print("=" * 70)

    print(f"Input image:")
    print(INPUT_IMAGE)

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not INPUT_IMAGE.exists():
        print("\n❌ Input image not found:")
        print(INPUT_IMAGE)
        return

    image = Image.open(INPUT_IMAGE).convert("RGB")

    print(f"\nImage size: {image.width} x {image.height}")

    # --------------------------------------------------------
    # Initialize PaddleOCR
    # --------------------------------------------------------

    print("\nInitializing PaddleOCR...")

    ocr = PaddleOCR(
        lang="en"
    )

    # --------------------------------------------------------
    # Run OCR
    # --------------------------------------------------------

    print("\nRunning OCR...")

    start_time = time.perf_counter()

    result = ocr.predict(str(INPUT_IMAGE))

    processing_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Parse OCR result
    # --------------------------------------------------------

    regions = []

    for page in result:

        # PaddleOCR result object
        data = page.json

        if callable(data):
            data = data()

        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, dict):

            # PaddleOCR may store OCR information inside
            # the "res" field.
            ocr_data = data.get("res", data)

            texts = ocr_data.get("rec_texts", [])
            scores = ocr_data.get("rec_scores", [])
            polygons = ocr_data.get("rec_polys", [])

            for i, text in enumerate(texts):

                text = str(text).strip()

                if not text:
                    continue

                confidence = (
                    float(scores[i])
                    if i < len(scores)
                    else None
                )

                polygon = None

                if i < len(polygons):
                    polygon = polygons[i]

                    # Convert numpy arrays to normal lists
                    if hasattr(polygon, "tolist"):
                        polygon = polygon.tolist()

                # Calculate bounding box
                bbox = None

                if polygon:
                    xs = [float(point[0]) for point in polygon]
                    ys = [float(point[1]) for point in polygon]

                    x_min = int(min(xs))
                    y_min = int(min(ys))
                    x_max = int(max(xs))
                    y_max = int(max(ys))

                    bbox = {
                        "x": x_min,
                        "y": y_min,
                        "width": x_max - x_min,
                        "height": y_max - y_min,
                    }

                regions.append(
                    {
                        "region_id": len(regions) + 1,
                        "text": text,
                        "recognition_confidence": confidence,
                        "detection_confidence": None,
                        "polygon": polygon,
                        "bbox": bbox,
                    }
                )

    # --------------------------------------------------------
    # Save OCR JSON
    # --------------------------------------------------------

    output_data = {
        "engine": "PaddleOCR",
        "image": str(INPUT_IMAGE),
        "image_width": image.width,
        "image_height": image.height,
        "processing_time_seconds": processing_time,
        "detected_region_count": len(regions),
        "regions": regions,
    }

    with open(OCR_JSON, "w", encoding="utf-8") as f:
        json.dump(
            output_data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    try:

        # PaddleOCR's result object may provide a visualization
        # method depending on installed version.

        for page in result:

            if hasattr(page, "save_to_img"):

                page.save_to_img(
                    str(OCR_VISUALIZATION)
                )

                break

    except Exception as e:

        print(
            f"\nVisualization could not be generated: {e}"
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OCR RESULTS")
    print("=" * 70)

    print(f"Detected regions: {len(regions)}")

    for region in regions:

        confidence = region["recognition_confidence"]

        if confidence is not None:
            confidence_text = f"{confidence:.4f}"
        else:
            confidence_text = "N/A"

        print(
            f"{region['region_id']:>2}. "
            f"{region['text']:<40} "
            f"confidence={confidence_text}"
        )

    print("\nOCR JSON saved to:")
    print(OCR_JSON)

    if OCR_VISUALIZATION.exists():
        print("\nVisualization saved to:")
        print(OCR_VISUALIZATION)

    print(f"\nProcessing time: {processing_time:.3f} seconds")

    print("\n" + "=" * 70)
    print("OCR MODULE TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()