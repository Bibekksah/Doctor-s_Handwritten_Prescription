from pathlib import Path
import json
import csv
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_JSON = (
    PROJECT_ROOT
    / "ml/results/ocr/existing_ocr_result.json"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "data/ocr_samples/images.jpeg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_word_segments_p1"
)

MANIFEST = OUTPUT_DIR / "medicine_word_segments_p1.csv"

PAD_RATIO = 0.05
MIN_SEGMENT_WIDTH = 6
MIN_FOREGROUND_RATIO = 0.008


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


def create_binary(crop):

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove tiny isolated noise.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8)
    )

    return binary


def find_segments(binary):

    h, w = binary.shape

    # Vertical projection.
    projection = (binary > 0).sum(axis=0)

    # Columns containing foreground.
    active = projection > max(0, int(h * 0.01))

    # Find continuous foreground runs.
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

    if not runs:
        return []

    # Calculate gaps between foreground runs.
    gaps = []

    for i in range(len(runs) - 1):

        gap_start = runs[i][1] + 1
        gap_end = runs[i + 1][0] - 1

        gap_width = gap_end - gap_start + 1

        gaps.append(
            (
                gap_width,
                gap_start,
                gap_end
            )
        )

    # Determine natural gap threshold from the region.
    gap_values = [g[0] for g in gaps]

    if not gap_values:
        return [
            (
                max(0, runs[0][0] - 2),
                min(w, runs[-1][1] + 3)
            )
        ]

    median_gap = float(np.median(gap_values))

    # Conservative threshold.
    gap_threshold = max(
        3,
        int(median_gap * 1.8)
    )

    # Split only at relatively large natural gaps.
    split_gaps = [
        g
        for g in gaps
        if g[0] >= gap_threshold
    ]

    segments = []

    start_x = runs[0][0]

    for _, _, gap_end in split_gaps:

        next_run = None

        for run in runs:

            if run[0] > gap_end:
                next_run = run
                break

        if next_run is None:
            break

        end_x = next_run[0] - 1

        if end_x - start_x + 1 >= MIN_SEGMENT_WIDTH:

            segment_binary = binary[:, start_x:end_x + 1]

            foreground_ratio = (
                np.count_nonzero(segment_binary)
                / segment_binary.size
            )

            if foreground_ratio >= MIN_FOREGROUND_RATIO:

                segments.append(
                    (
                        max(0, start_x - 3),
                        min(w, end_x + 4)
                    )
                )

        start_x = next_run[0]

    # Final segment.
    end_x = runs[-1][1]

    if end_x - start_x + 1 >= MIN_SEGMENT_WIDTH:

        segment_binary = binary[:, start_x:end_x + 1]

        foreground_ratio = (
            np.count_nonzero(segment_binary)
            / segment_binary.size
        )

        if foreground_ratio >= MIN_FOREGROUND_RATIO:

            segments.append(
                (
                    max(0, start_x - 3),
                    min(w, end_x + 4)
                )
            )

    return segments


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OCR_JSON, encoding="utf-8") as f:
        data = json.load(f)

    image = cv2.imread(
        str(IMAGE_PATH)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {IMAGE_PATH}"
        )

    rows = []

    print("=" * 70)
    print("MEDICINE WORD SEGMENTATION - P1")
    print("=" * 70)

    print(
        "Input regions :",
        len(data["regions"])
    )

    print(
        "Output        :",
        OUTPUT_DIR
    )

    print()

    total = 0

    for region in data["regions"]:

        region_id = int(
            region["region_id"]
        )

        text = region.get(
            "text",
            ""
        )

        crop, offset = crop_region(
            image,
            region["bbox"]
        )

        binary = create_binary(
            crop
        )

        segments = find_segments(
            binary
        )

        region_dir = (
            OUTPUT_DIR
            / f"region_{region_id:02d}"
        )

        region_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"Region {region_id:02d}: {text}"
        )

        print(
            f"  segments: {len(segments)}"
        )

        for segment_id, (x1, x2) in enumerate(
            segments,
            start=1
        ):

            segment = crop[:, x1:x2]

            filename = (
                f"segment_{segment_id:02d}.png"
            )

            output_path = (
                region_dir / filename
            )

            cv2.imwrite(
                str(output_path),
                segment
            )

            rows.append(
                {
                    "region_id": region_id,
                    "ocr_text": text,
                    "segment_id": segment_id,
                    "segment_path": str(
                        output_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "source_x": (
                        offset[0] + x1
                    ),
                    "source_y": offset[1],
                    "source_width": (
                        x2 - x1
                    ),
                    "source_height": (
                        crop.shape[0]
                    ),
                }
            )

            total += 1

    with open(
        MANIFEST,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

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
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()

    print("=" * 70)
    print("SEGMENTATION SUMMARY")
    print("=" * 70)

    print(
        "Regions processed :",
        len(data["regions"])
    )

    print(
        "Segments created  :",
        total
    )

    print(
        "Manifest          :",
        MANIFEST
    )

    print("=" * 70)


if __name__ == "__main__":
    main()