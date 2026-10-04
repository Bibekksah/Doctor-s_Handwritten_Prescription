
from pathlib import Path
import json
import csv
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OCR_JSON = PROJECT_ROOT / "ml/results/ocr/existing_ocr_result.json"
IMAGE_PATH = PROJECT_ROOT / "data/ocr_samples/images.jpeg"
OUTPUT_DIR = PROJECT_ROOT / "ml/results/ocr_ml003/medicine_word_segments"
MANIFEST = OUTPUT_DIR / "medicine_word_segments.csv"

PAD_RATIO = 0.05
HORIZONTAL_KERNEL_RATIO = 0.025
MIN_SEGMENT_WIDTH = 8
MIN_FOREGROUND_RATIO = 0.015
GAP_MIN_WIDTH = 3


def crop_region(image, bbox):
    x = int(bbox["x"])
    y = int(bbox["y"])
    w = int(bbox["width"])
    h = int(bbox["height"])

    px = max(2, int(w * PAD_RATIO))
    py = max(2, int(h * PAD_RATIO))

    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(image.shape[1], x + w + px)
    y2 = min(image.shape[0], y + h + py)

    return image[y1:y2, x1:x2], (x1, y1)


def make_word_segments(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    h, w = binary.shape

    # Join nearby handwritten strokes/letters horizontally.
    kernel_width = max(5, int(w * HORIZONTAL_KERNEL_RATIO))
    if kernel_width % 2 == 0:
        kernel_width += 1

    joined = cv2.dilate(
        binary,
        np.ones((2, kernel_width), np.uint8),
        iterations=1,
    )

    projection = (joined > 0).sum(axis=0)
    threshold = max(1, int(h * 0.025))
    active = projection > threshold

    runs = []
    start = None

    for i, value in enumerate(active):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, w - 1))

    # Merge very small gaps.
    merged = []
    for run in runs:
        if not merged:
            merged.append(list(run))
            continue

        gap = run[0] - merged[-1][1] - 1
        if gap < GAP_MIN_WIDTH:
            merged[-1][1] = run[1]
        else:
            merged.append(list(run))

    segments = []

    for x1, x2 in merged:
        if x2 - x1 + 1 < MIN_SEGMENT_WIDTH:
            continue

        original = binary[:, x1:x2 + 1]
        foreground_ratio = float(np.count_nonzero(original)) / original.size

        if foreground_ratio < MIN_FOREGROUND_RATIO:
            continue

        margin_x = max(2, int((x2 - x1 + 1) * 0.08))
        sx1 = max(0, x1 - margin_x)
        sx2 = min(w, x2 + margin_x + 1)

        segments.append((sx1, sx2))

    return segments


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not OCR_JSON.exists():
        raise FileNotFoundError(OCR_JSON)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    with open(OCR_JSON, encoding="utf-8") as f:
        data = json.load(f)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    rows = []
    total_segments = 0

    print("=" * 70)
    print("MEDICINE WORD SEGMENTATION - P0")
    print("=" * 70)
    print("Input regions :", len(data["regions"]))
    print("Source image  :", IMAGE_PATH)
    print("Output        :", OUTPUT_DIR)
    print()

    for region in data["regions"]:
        region_id = int(region["region_id"])
        bbox = region["bbox"]
        text = region.get("text", "")

        crop, (offset_x, offset_y) = crop_region(image, bbox)
        segments = make_word_segments(crop)

        region_dir = OUTPUT_DIR / f"region_{region_id:02d}"
        region_dir.mkdir(parents=True, exist_ok=True)

        print(f"Region {region_id:02d}: {text}")
        print(f"  segments: {len(segments)}")

        for segment_id, (sx1, sx2) in enumerate(segments, start=1):
            segment = crop[:, sx1:sx2]
            filename = f"segment_{segment_id:02d}.png"
            output_path = region_dir / filename

            cv2.imwrite(str(output_path), segment)

            rows.append({
                "region_id": region_id,
                "ocr_text": text,
                "segment_id": segment_id,
                "segment_path": str(output_path.relative_to(PROJECT_ROOT)),
                "source_x": offset_x + sx1,
                "source_y": offset_y,
                "source_width": sx2 - sx1,
                "source_height": crop.shape[0],
            })

            total_segments += 1

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "region_id",
                "ocr_text",
                "segment_id",
                "segment_path",
                "source_x",
                "source_y",
                "source_width",
                "source_height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("SEGMENTATION SUMMARY")
    print("=" * 70)
    print("Regions processed :", len(data["regions"]))
    print("Segments created  :", total_segments)
    print("Manifest          :", MANIFEST)
    print("=" * 70)


if __name__ == "__main__":
    main()
