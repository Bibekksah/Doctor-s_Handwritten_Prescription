from pathlib import Path
import csv
import json
import re

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MATCHING_PATH = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_candidate_matching_results.csv"
)

OCR_JSON_PATH = (
    PROJECT_ROOT
    / "ml/results/ocr/existing_ocr_result.json"
)

ONNX_MODEL_PATH = (
    PROJECT_ROOT
    / "ml/results/optimization/ML003_resnet18.onnx"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_verification_results.csv"
)

CLASS_MAPPING_PATH = (
    PROJECT_ROOT
    / "ml/config/class_mapping.json"
)


# ============================================================
# THRESHOLDS
# ============================================================

ML_HIGH_CONFIDENCE = 0.80
ML_MEDIUM_CONFIDENCE = 0.60

SIMILARITY_STRONG = 0.60
SIMILARITY_MEDIUM = 0.40

ACCEPT_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.40


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

def load_class_mapping():

    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "idx_to_class" not in data:
        raise ValueError(
            "idx_to_class not found in class_mapping.json"
        )

    return {
        int(k): v
        for k, v in data["idx_to_class"].items()
    }


# ============================================================
# LOAD ONNX MODEL
# ============================================================

def load_onnx_model():

    print("\nLoading ML-003 ONNX model...")

    session = ort.InferenceSession(
        str(ONNX_MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    print(f"Model input : {input_name}")
    print(
        f"Input shape : "
        f"{session.get_inputs()[0].shape}"
    )

    return session, input_name


# ============================================================
# OCR JSON
# ============================================================

def load_ocr_regions():

    with open(OCR_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        int(item["region_id"]): item
        for item in data["regions"]
    }


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def resize_with_padding(image, height=64, width=256):

    image = image.convert("RGB")

    original_width, original_height = image.size

    scale = min(
        width / original_width,
        height / original_height
    )

    new_width = max(
        1,
        int(original_width * scale)
    )

    new_height = max(
        1,
        int(original_height * scale)
    )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    x = (width - new_width) // 2
    y = (height - new_height) // 2

    canvas.paste(image, (x, y))

    return canvas


def preprocess_crop(crop):

    # RGB
    image = Image.fromarray(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    )

    # P1 preprocessing:
    # RGB → grayscale → resize/padding
    image = image.convert("L")

    image = resize_grayscale_with_padding(
        image,
        height=64,
        width=256
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # P1 = 1 x 64 x 256
    image = np.expand_dims(
        image,
        axis=0
    )

    # ML-003 ResNet18 expects 3 channels
    image = np.repeat(
        image,
        3,
        axis=0
    )

    # Batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image.astype(np.float32)


def resize_grayscale_with_padding(
    image,
    height=64,
    width=256
):

    original_width, original_height = image.size

    scale = min(
        width / original_width,
        height / original_height
    )

    new_width = max(
        1,
        int(original_width * scale)
    )

    new_height = max(
        1,
        int(original_height * scale)
    )

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "L",
        (width, height),
        255
    )

    x = (width - new_width) // 2
    y = (height - new_height) // 2

    canvas.paste(
        image,
        (x, y)
    )

    return canvas


# ============================================================
# SOFTMAX
# ============================================================

def softmax(logits):

    logits = logits - np.max(logits)

    exp_values = np.exp(logits)

    return exp_values / np.sum(exp_values)


# ============================================================
# ML-003 PREDICTION
# ============================================================

def predict_crop(
    session,
    input_name,
    crop,
    idx_to_class
):

    tensor = preprocess_crop(crop)

    outputs = session.run(
        None,
        {
            input_name: tensor
        }
    )

    logits = outputs[0][0]

    probabilities = softmax(logits)

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    predictions = []

    for index in top_indices:

        class_id = int(index)

        medicine = idx_to_class[
            class_id
        ]

        confidence = float(
            probabilities[index]
        )

        predictions.append(
            {
                "medicine": medicine,
                "class_id": class_id,
                "confidence": confidence
            }
        )

    return predictions


# ============================================================
# CROP OCR REGION
# ============================================================

def crop_region(
    image,
    region,
    padding_ratio=0.05
):

    bbox = region["bbox"]

    x = int(bbox["x"])
    y = int(bbox["y"])

    width = int(bbox["width"])
    height = int(bbox["height"])

    pad_x = int(
        width * padding_ratio
    )

    pad_y = int(
        height * padding_ratio
    )

    image_height, image_width = image.shape[:2]

    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        image_width,
        x + width + pad_x
    )

    y2 = min(
        image_height,
        y + height + pad_y
    )

    return image[
        y1:y2,
        x1:x2
    ]


# ============================================================
# VERIFY REGION
# ============================================================

def verify_region(
    row,
    ml_predictions
):

    ocr_text = row["ocr_text"]

    ocr_confidence = float(
        row["ocr_confidence"]
    )

    filter_decision = row[
        "filter_decision"
    ]

    best_medicine = row[
        "best_medicine"
    ]

    best_similarity = float(
        row["best_similarity"]
    )

    ml_top1 = ml_predictions[0]

    ml_medicine = ml_top1[
        "medicine"
    ]

    ml_confidence = ml_top1[
        "confidence"
    ]

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    score = 0.0

    reasons = []

    # ML-003 evidence
    if ml_confidence >= ML_HIGH_CONFIDENCE:

        score += 0.55

        reasons.append(
            "high ML-003 confidence"
        )

    elif ml_confidence >= ML_MEDIUM_CONFIDENCE:

        score += 0.40

        reasons.append(
            "medium ML-003 confidence"
        )

    else:

        score += 0.20

        reasons.append(
            "low ML-003 confidence"
        )

    # --------------------------------------------------------
    # OCR candidate ↔ ML agreement
    # --------------------------------------------------------

    if (
        best_medicine.strip().lower()
        ==
        ml_medicine.strip().lower()
    ):

        score += 0.25

        reasons.append(
            "OCR candidate agrees with ML-003"
        )

        agreement = "YES"

    else:

        reasons.append(
            "OCR candidate disagrees with ML-003"
        )

        agreement = "NO"

    # --------------------------------------------------------
    # OCR similarity
    # --------------------------------------------------------

    if best_similarity >= SIMILARITY_STRONG:

        score += 0.10

        reasons.append(
            "strong OCR candidate similarity"
        )

    elif best_similarity >= SIMILARITY_MEDIUM:

        score += 0.07

        reasons.append(
            "medium OCR candidate similarity"
        )

    elif best_similarity >= 0.30:

        score += 0.03

        reasons.append(
            "weak OCR candidate similarity"
        )

    # --------------------------------------------------------
    # OCR confidence
    # --------------------------------------------------------

    if ocr_confidence >= 0.90:

        score += 0.05

        reasons.append(
            "high OCR confidence"
        )

    elif ocr_confidence >= 0.70:

        score += 0.03

        reasons.append(
            "medium OCR confidence"
        )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    if filter_decision == "PASS_TO_ML":

        score += 0.05

        reasons.append(
            "candidate filter passed"
        )

    elif filter_decision == "REJECT":

        score -= 0.20

        reasons.append(
            "candidate filter rejected"
        )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    if score >= ACCEPT_THRESHOLD:

        decision = "ACCEPT"

    elif score >= REVIEW_THRESHOLD:

        decision = "REVIEW"

    else:

        decision = "REJECT"

    # If ML and OCR disagree, never automatically accept.
    if agreement == "NO" and decision == "ACCEPT":

        decision = "REVIEW"

    return {
        "ml_prediction": ml_medicine,
        "ml_class_id": ml_top1["class_id"],
        "ml_confidence": round(
            ml_confidence,
            4
        ),

        "ml_top2": ml_predictions[1]["medicine"],
        "ml_top2_confidence": round(
            ml_predictions[1]["confidence"],
            4
        ),

        "ml_top3": ml_predictions[2]["medicine"],
        "ml_top3_confidence": round(
            ml_predictions[2]["confidence"],
            4
        ),

        "ocr_ml_agreement": agreement,

        "verification_score": round(
            score,
            4
        ),

        "final_medicine": ml_medicine,

        "decision": decision,

        "verification_reasons":
            "; ".join(reasons)
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDICINE VERIFIER - OCR + ML-003")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        MATCHING_PATH,
        OCR_JSON_PATH,
        ONNX_MODEL_PATH,
        CLASS_MAPPING_PATH
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    idx_to_class = load_class_mapping()

    ocr_regions = load_ocr_regions()

    session, input_name = load_onnx_model()

    with open(
        MATCHING_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    print(
        f"\nInput regions : {len(rows)}"
    )

    # --------------------------------------------------------
    # Load prescription image
    # --------------------------------------------------------

    image_path = (
        PROJECT_ROOT
        / "data/ocr_samples/images.jpeg"
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Prescription image not found:\n"
            f"{image_path}"
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            "Could not read prescription image."
        )

    # --------------------------------------------------------
    # Process regions
    # --------------------------------------------------------

    results = []

    for row in rows:

        region_id = int(
            row["region_id"]
        )

        print(
            f"\nRegion {region_id}: "
            f"{row['ocr_text']}"
        )

        if region_id not in ocr_regions:

            print(
                "  WARNING: OCR region not found"
            )

            continue

        # Crop OCR region
        crop = crop_region(
            image,
            ocr_regions[region_id]
        )

        if crop.size == 0:

            print(
                "  WARNING: Empty crop"
            )

            continue

        # ML-003 prediction
        ml_predictions = predict_crop(
            session,
            input_name,
            crop,
            idx_to_class
        )

        # Verify
        verification = verify_region(
            row,
            ml_predictions
        )

        combined = dict(row)

        combined.update(
            verification
        )

        results.append(
            combined
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"  OCR candidate : "
            f"{row['best_medicine']} "
            f"({row['best_similarity']})"
        )

        print(
            f"  ML-003        : "
            f"{verification['ml_prediction']}"
        )

        print(
            f"  ML confidence : "
            f"{verification['ml_confidence']:.4f}"
        )

        print(
            f"  Agreement     : "
            f"{verification['ocr_ml_agreement']}"
        )

        print(
            f"  Score         : "
            f"{verification['verification_score']:.3f}"
        )

        print(
            f"  Decision      : "
            f"{verification['decision']}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if results:

        fieldnames = list(
            results[0].keys()
        )

        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                results
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    counts = {
        "ACCEPT": 0,
        "REVIEW": 0,
        "REJECT": 0
    }

    for result in results:

        counts[
            result["decision"]
        ] += 1

    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(
        f"ACCEPT : {counts['ACCEPT']}"
    )

    print(
        f"REVIEW : {counts['REVIEW']}"
    )

    print(
        f"REJECT : {counts['REJECT']}"
    )

    print(
        f"\nSaved:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()