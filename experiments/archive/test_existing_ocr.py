from pathlib import Path
import json
import time

import cv2
import numpy as np
from paddleocr import PaddleOCR


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Actual full prescription image supplied for OCR evaluation
IMAGE_PATH = Path(
    __import__("os").environ.get(
        "OCR_IMAGE",
        str(
            PROJECT_ROOT
            / "data"
            / "ocr_samples"
            / "images.jpeg"
        ),
    )
)
OUTPUT_DIR = PROJECT_ROOT / "ml" / "results" / "ocr"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_JSON = OUTPUT_DIR / "existing_ocr_result.json"
RESULT_IMAGE = OUTPUT_DIR / "existing_ocr_visualization.jpg"


# ============================================================
# CHECK INPUT
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Prescription image not found:\n{IMAGE_PATH}"
    )


print("=" * 75)
print("EXISTING OCR ENGINE EVALUATION")
print("=" * 75)

print(f"Input image:")
print(IMAGE_PATH)


# ============================================================
# IMAGE INFORMATION
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise ValueError(
        f"OpenCV could not read image:\n{IMAGE_PATH}"
    )

height, width = image.shape[:2]

print()
print(f"Image width : {width}")
print(f"Image height: {height}")
print(f"Channels    : {image.shape[2]}")


# ============================================================
# INITIALIZE PADDLEOCR
# ============================================================

print()
print("Initializing PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print("PaddleOCR initialized.")


# ============================================================
# RUN OCR
# ============================================================

print()
print("Running OCR...")
print("Please wait if models are being downloaded.")

start_time = time.perf_counter()

results = ocr.predict(str(IMAGE_PATH))

elapsed_time = time.perf_counter() - start_time

print()
print(f"OCR processing time: {elapsed_time:.3f} seconds")


# ============================================================
# PROCESS RESULTS
# ============================================================

all_results = []

visualization = image.copy()

region_count = 0


for result_index, result in enumerate(results):

    # PaddleOCR result objects expose a dictionary-like
    # representation through .json.
    try:
        result_json = result.json

        if callable(result_json):
            result_json = result_json()

        if isinstance(result_json, str):
            result_data = json.loads(result_json)
        else:
            result_data = result_json

    except Exception as exc:
        print()
        print("Could not directly parse result.json.")
        print(f"Reason: {exc}")

        # Keep a readable fallback representation.
        result.print()

        continue


    # --------------------------------------------------------
    # Extract OCR result dictionary
    # --------------------------------------------------------

    if isinstance(result_data, dict) and "res" in result_data:
        data = result_data["res"]
    else:
        data = result_data


    # --------------------------------------------------------
    # Detection polygons
    # --------------------------------------------------------

    polygons = data.get("dt_polys", [])

    # --------------------------------------------------------
    # Recognition text
    # --------------------------------------------------------

    texts = data.get("rec_texts", [])

    # --------------------------------------------------------
    # Recognition confidence
    # --------------------------------------------------------

    text_scores = data.get("rec_scores", [])

    # --------------------------------------------------------
    # Detection confidence
    # --------------------------------------------------------

    detection_scores = data.get("dt_scores", [])


    # --------------------------------------------------------
    # Process detected regions
    # --------------------------------------------------------

    for i, polygon in enumerate(polygons):

        polygon_array = np.asarray(
            polygon,
            dtype=np.int32
        )

        if polygon_array.ndim != 2:
            continue

        if polygon_array.shape[0] < 4:
            continue


        # Text
        if i < len(texts):
            text = str(texts[i])
        else:
            text = ""


        # Recognition confidence
        if i < len(text_scores):
            recognition_confidence = float(
                text_scores[i]
            )
        else:
            recognition_confidence = None


        # Detection confidence
        if i < len(detection_scores):
            detection_confidence = float(
                detection_scores[i]
            )
        else:
            detection_confidence = None


        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(
            polygon_array
        )


        region_count += 1


        region = {
            "region_id": region_count,
            "text": text,
            "recognition_confidence": recognition_confidence,
            "detection_confidence": detection_confidence,
            "polygon": polygon_array.tolist(),
            "bbox": {
                "x": int(x),
                "y": int(y),
                "width": int(w),
                "height": int(h),
            },
        }

        all_results.append(region)


        # ----------------------------------------------------
        # Draw polygon
        # ----------------------------------------------------

        cv2.polylines(
            visualization,
            [polygon_array],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2,
        )


        # ----------------------------------------------------
        # Draw region number
        # ----------------------------------------------------

        cv2.putText(
            visualization,
            f"R{region_count}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 75)
print("OCR RESULTS")
print("=" * 75)

print()
print(f"Detected text regions: {len(all_results)}")

if not all_results:

    print()
    print("No text regions were detected.")

else:

    for region in all_results:

        print()
        print(
            f"Region {region['region_id']}"
        )

        print(
            f"  Text       : "
            f"{region['text']}"
        )

        print(
            f"  Recognition: "
            f"{region['recognition_confidence']}"
        )

        print(
            f"  Detection  : "
            f"{region['detection_confidence']}"
        )

        print(
            f"  Bounding box: "
            f"{region['bbox']}"
        )


# ============================================================
# SAVE VISUALIZATION
# ============================================================

cv2.imwrite(
    str(RESULT_IMAGE),
    visualization
)

print()
print("Visualization saved:")
print(RESULT_IMAGE)


# ============================================================
# SAVE JSON RESULTS
# ============================================================

output_data = {
    "engine": "PaddleOCR",
    "image": str(IMAGE_PATH),
    "image_width": width,
    "image_height": height,
    "processing_time_seconds": round(
        elapsed_time,
        4
    ),
    "detected_region_count": len(all_results),
    "regions": all_results,
}


with open(
    RESULT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output_data,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("JSON results saved:")
print(RESULT_JSON)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("OCR EVALUATION COMPLETE")
print("=" * 75)

print()
print(
    f"Detected regions : {len(all_results)}"
)

print(
    f"Processing time  : "
    f"{elapsed_time:.3f} seconds"
)

print()
print(
    "IMPORTANT:"
)

print(
    "This is an OCR-engine evaluation only."
)

print(
    "ResNet-18 ML-003 has NOT been connected yet."
)

print(
    "The next decision will be based on the OCR results."
)

print("=" * 75)