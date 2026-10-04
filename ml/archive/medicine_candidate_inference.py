import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "data/ocr_samples/images.jpeg"

GATE_RESULTS = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_candidate_gate_results.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "ml/results/optimization/ML003_resnet18.onnx"
)

CLASS_MAPPING_PATH = (
    PROJECT_ROOT
    / "ml/config/class_mapping.json"
)

OUTPUT_DIR = PROJECT_ROOT / "ml/results/ocr_ml003"
OUTPUT_CSV = OUTPUT_DIR / "medicine_candidate_inference_results.csv"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

def load_class_mapping():
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "idx_to_class" in data:
        return {
            int(k): v
            for k, v in data["idx_to_class"].items()
        }

    if "class_to_idx" in data:
        return {
            int(v): k
            for k, v in data["class_to_idx"].items()
        }

    raise ValueError(
        "Unsupported class_mapping.json format. "
        f"Found keys: {list(data.keys())}"
    )


# ============================================================
# RESIZE + PADDING
# Same basic geometry used by ML-003 P1 preprocessing.
# ============================================================

def resize_with_padding(image):
    image = image.convert("RGB")

    original_width, original_height = image.size

    scale = min(
        IMAGE_WIDTH / original_width,
        IMAGE_HEIGHT / original_height
    )

    new_width = max(1, int(original_width * scale))
    new_height = max(1, int(original_height * scale))

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        "white"
    )

    x = (IMAGE_WIDTH - new_width) // 2
    y = (IMAGE_HEIGHT - new_height) // 2

    canvas.paste(image, (x, y))

    return canvas


# ============================================================
# P1 PREPROCESSING
# Grayscale -> Resize/Pad -> Tensor-like array
# ============================================================

def preprocess_crop(crop):
    image = Image.fromarray(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    )

    image = image.convert("L")

    image = resize_with_padding(
        image.convert("RGB")
    ).convert("L")

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # Shape:
    # H,W -> 1,H,W
    array = np.expand_dims(array, axis=0)

    # ML-003 ResNet18 expects 3 channels.
    # Repeat grayscale channel 3 times.
    array = np.repeat(array, 3, axis=0)

    # 1,C,H,W
    array = np.expand_dims(array, axis=0)

    return array.astype(np.float32)


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):
    logits = logits - np.max(logits)

    exp_values = np.exp(logits)

    return exp_values / np.sum(exp_values)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=providers
    )

    input_name = session.get_inputs()[0].name

    return session, input_name


# ============================================================
# CROP OCR REGION
# ============================================================

def crop_region(image, bbox, padding_ratio=0.05):
    height, width = image.shape[:2]

    x = int(bbox["x"])
    y = int(bbox["y"])
    w = int(bbox["width"])
    h = int(bbox["height"])

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)

    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    crop = image[y1:y2, x1:x2]

    return crop


# ============================================================
# ML-003 PREDICTION
# ============================================================

def predict(session, input_name, crop, idx_to_class):
    input_tensor = preprocess_crop(crop)

    logits = session.run(
        None,
        {input_name: input_tensor}
    )[0][0]

    probabilities = softmax(logits)

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    predictions = []

    for rank, class_id in enumerate(top_indices, start=1):
        class_id = int(class_id)

        predictions.append({
            "rank": rank,
            "medicine": idx_to_class.get(
                class_id,
                f"class_{class_id}"
            ),
            "class_id": class_id,
            "confidence": round(
                float(probabilities[class_id]),
                6
            )
        })

    return predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ROUTED ML-003 MEDICINE CANDIDATE INFERENCE")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    required_files = [
        IMAGE_PATH,
        GATE_RESULTS,
        MODEL_PATH,
        CLASS_MAPPING_PATH
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise RuntimeError(
            f"Could not load image: {IMAGE_PATH}"
        )

    print(f"Prescription image : {IMAGE_PATH}")
    print(
        f"Image size         : "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    # --------------------------------------------------------
    # Load gate results
    # --------------------------------------------------------

    with open(
        GATE_RESULTS,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:
        reader = csv.DictReader(f)
        gate_rows = list(reader)

    print(f"Gate regions       : {len(gate_rows)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading ML-003 ONNX model...")

    session, input_name = load_model()

    print(
        "ONNX providers     :",
        session.get_providers()
    )

    # --------------------------------------------------------
    # Load class mapping
    # --------------------------------------------------------

    idx_to_class = load_class_mapping()

    # --------------------------------------------------------
    # Process only REVIEW + MEDICINE_CANDIDATE
    # --------------------------------------------------------

    routed_rows = []

    skipped = 0
    processed = 0

    for row in gate_rows:

        decision = row["filter_decision"]

        if decision not in {
            "REVIEW",
            "MEDICINE_CANDIDATE"
        }:
            skipped += 1
            continue

        region_id = row["region_id"]

        # ----------------------------------------------------
        # Read bbox dimensions
        # ----------------------------------------------------

        bbox_width = float(
            row["bbox_width"]
        )

        bbox_height = float(
            row["bbox_height"]
        )

        # The gate CSV contains dimensions but not x/y.
        # Recover coordinates from OCR JSON separately.
        #
        # We will load OCR JSON below.
        # ----------------------------------------------------

        routed_rows.append({
            "region_id": region_id,
            "ocr_text": row["ocr_text"],
            "ocr_confidence": row["ocr_confidence"],
            "gate_score": row["candidate_score"],
            "gate_decision": decision,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height
        })

    # --------------------------------------------------------
    # Load OCR JSON for exact bounding boxes
    # --------------------------------------------------------

    OCR_JSON = (
        PROJECT_ROOT
        / "ml/results/ocr/existing_ocr_result.json"
    )

    with open(
        OCR_JSON,
        "r",
        encoding="utf-8"
    ) as f:
        ocr_data = json.load(f)

    bbox_by_region = {
        str(region["region_id"]): region["bbox"]
        for region in ocr_data["regions"]
    }

    # --------------------------------------------------------
    # Run ML-003
    # --------------------------------------------------------

    output_rows = []

    for row in routed_rows:

        region_id = str(row["region_id"])

        bbox = bbox_by_region.get(region_id)

        if bbox is None:
            print(
                f"WARNING: bbox not found for region "
                f"{region_id}"
            )
            continue

        crop = crop_region(
            image,
            bbox,
            padding_ratio=0.05
        )

        if crop.size == 0:
            print(
                f"WARNING: empty crop for region "
                f"{region_id}"
            )
            continue

        predictions = predict(
            session,
            input_name,
            crop,
            idx_to_class
        )

        top1 = predictions[0]
        top2 = predictions[1]
        top3 = predictions[2]

        output_rows.append({
            "region_id": region_id,
            "ocr_text": row["ocr_text"],
            "ocr_confidence": row["ocr_confidence"],
            "gate_score": row["gate_score"],
            "gate_decision": row["gate_decision"],

            "ml_top1_medicine": top1["medicine"],
            "ml_top1_class_id": top1["class_id"],
            "ml_top1_confidence": top1["confidence"],

            "ml_top2_medicine": top2["medicine"],
            "ml_top2_class_id": top2["class_id"],
            "ml_top2_confidence": top2["confidence"],

            "ml_top3_medicine": top3["medicine"],
            "ml_top3_class_id": top3["class_id"],
            "ml_top3_confidence": top3["confidence"]
        })

        print()
        print(f"Region {region_id}: {row['ocr_text']}")
        print(
            f"  Gate decision : {row['gate_decision']}"
        )
        print(
            f"  Top-1         : "
            f"{top1['medicine']} "
            f"({top1['confidence']:.4f})"
        )
        print(
            f"  Top-2         : "
            f"{top2['medicine']} "
            f"({top2['confidence']:.4f})"
        )
        print(
            f"  Top-3         : "
            f"{top3['medicine']} "
            f"({top3['confidence']:.4f})"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "region_id",
        "ocr_text",
        "ocr_confidence",
        "gate_score",
        "gate_decision",

        "ml_top1_medicine",
        "ml_top1_class_id",
        "ml_top1_confidence",

        "ml_top2_medicine",
        "ml_top2_class_id",
        "ml_top2_confidence",

        "ml_top3_medicine",
        "ml_top3_class_id",
        "ml_top3_confidence"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(output_rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)

    print(
        f"Total gate regions : {len(gate_rows)}"
    )

    print(
        f"Sent to ML-003    : {len(output_rows)}"
    )

    print(
        f"Skipped           : {skipped}"
    )

    print()
    print(
        f"Results saved to:\n{OUTPUT_CSV}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()