"""
Final medicine verification / integration pipeline.

Pipeline:
    PaddleOCR regions
        -> candidate filtering
        -> ML-003 ONNX top-3 inference
        -> OCR text vs medicine-name evidence
        -> final MEDICINE / REVIEW / NON_MEDICINE decision

Important:
- ML-003 is a closed-set 78-class medicine classifier.
- High ML confidence alone is NOT sufficient to accept a medicine.
- This module is an engineering/prototype verification layer, not clinical validation.
"""

from pathlib import Path
import csv
import json
import re
import sys

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OCR_JSON = PROJECT_ROOT / "ml/results/ocr/existing_ocr_result.json"
MODEL_PATH = PROJECT_ROOT / "ml/results/optimization/ML003_resnet18.onnx"
CLASS_MAPPING_PATH = PROJECT_ROOT / "ml/config/class_mapping.json"

OUTPUT_CSV = PROJECT_ROOT / "ml/results/ocr_ml003/final_medicine_verification.csv"


IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256


# Terms that strongly indicate that an OCR region is not a medicine name.
NON_MEDICINE_TERMS = {
    "dr", "doctor", "name", "address", "phone", "mobile", "age",
    "gender", "gonder", "date", "clinic", "hospital", "hyderabad",
    "ecg", "bp", "bpm", "spo2", "pulse", "temp", "temperature",
    "weight", "height", "registration", "reg", "patient"
}

# Clinical/context words which normally indicate instructions, symptoms,
# duration, measurements, or other non-medicine content.
CLINICAL_TERMS = {
    "days", "day", "week", "weeks", "mg", "ml", "kg", "cm",
    "fever", "pain", "cough", "cold", "hypertension", "syrup",
    "tablet", "tab", "morning", "night", "daily", "twice",
    "before", "after", "food"
}


class ResizeWithPadding:
    def __init__(self, height=IMAGE_HEIGHT, width=IMAGE_WIDTH):
        self.height = height
        self.width = width

    def __call__(self, image):
        image = image.convert("RGB")

        original_width, original_height = image.size
        scale = min(
            self.width / original_width,
            self.height / original_height,
        )

        new_width = max(1, int(original_width * scale))
        new_height = max(1, int(original_height * scale))

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (self.width, self.height),
            "white",
        )

        x = (self.width - new_width) // 2
        y = (self.height - new_height) // 2

        canvas.paste(image, (x, y))
        return canvas


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def load_class_names():
    with open(CLASS_MAPPING_PATH, encoding="utf-8") as f:
        mapping = json.load(f)

    if "idx_to_class" in mapping:
        raw = mapping["idx_to_class"]
    elif "class_to_idx" in mapping:
        raw = {
            str(v): k
            for k, v in mapping["class_to_idx"].items()
        }
    else:
        raise ValueError(
            "class_mapping.json must contain idx_to_class or class_to_idx"
        )

    return {
        int(k): str(v)
        for k, v in raw.items()
    }


def crop_region(image, region, padding_ratio=0.05):
    bbox = region["bbox"]

    x = int(bbox["x"])
    y = int(bbox["y"])
    w = int(bbox["width"])
    h = int(bbox["height"])

    px = max(1, int(w * padding_ratio))
    py = max(1, int(h * padding_ratio))

    left = max(0, x - px)
    top = max(0, y - py)
    right = min(image.width, x + w + px)
    bottom = min(image.height, y + h + py)

    return image.crop((left, top, right, bottom))


def preprocess(image):
    # ResizeWithPadding converts to RGB internally,
    # so convert back to grayscale AFTER resizing.
    image = ResizeWithPadding()(image)
    image = ImageOps.grayscale(image)

    # [H, W]
    arr = np.asarray(image, dtype=np.float32) / 255.0

    # [H, W] -> [1, H, W]
    arr = arr[np.newaxis, ...]

    # [1, H, W] -> [3, H, W]
    arr = np.repeat(arr, 3, axis=0)

    # [3, H, W] -> [1, 3, H, W]
    arr = arr[np.newaxis, ...]

    return arr.astype(np.float32)

def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def ml_predict(session, image, class_names):
    tensor = preprocess(image)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor})

    probabilities = softmax(outputs[0])[0]
    top_indices = np.argsort(probabilities)[::-1][:3]

    predictions = []

    for idx in top_indices:
        idx = int(idx)
        predictions.append(
            {
                "medicine": class_names.get(idx, f"class_{idx}"),
                "class_id": idx,
                "confidence": float(probabilities[idx]),
            }
        )

    return predictions


def fuzzy_similarity(a, b):
    """
    Lightweight character similarity without external dependencies.
    Returns [0, 1].
    """
    from difflib import SequenceMatcher

    a = compact_text(a)
    b = compact_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def candidate_evidence(ocr_text, ocr_confidence, ml_top1, class_names):
    normalized = normalize_text(ocr_text)
    tokens = set(normalized.split())

    non_medical_hits = sorted(tokens & NON_MEDICINE_TERMS)
    clinical_hits = sorted(tokens & CLINICAL_TERMS)

    similarity = fuzzy_similarity(ocr_text, ml_top1)

    score = 0
    reasons = []

    # Strong explicit non-medicine evidence.
    if non_medical_hits:
        score -= 6
        reasons.append(
            "non-medicine keyword: " + ",".join(non_medical_hits)
        )

    if clinical_hits:
        score -= 3
        reasons.append(
            "clinical/context keyword: " + ",".join(clinical_hits)
        )

    # OCR confidence is supporting evidence only.
    if ocr_confidence >= 0.90:
        score += 1
    elif ocr_confidence < 0.60:
        score -= 1

    # OCR text and ML medicine-name agreement.
    if similarity >= 0.70:
        score += 5
        reasons.append("strong OCR/medicine text agreement")
    elif similarity >= 0.55:
        score += 3
        reasons.append("moderate OCR/medicine text agreement")
    elif similarity >= 0.40:
        score += 1
        reasons.append("weak OCR/medicine text agreement")
    else:
        reasons.append("OCR/medicine text disagreement")

    # A very short/noisy OCR string is not enough to accept.
    if len(compact_text(ocr_text)) < 4:
        score -= 2
        reasons.append("very short/noisy OCR text")

    # Important safety rule:
    # exact/near text agreement is required for automatic acceptance.
    #
    # We deliberately do NOT accept a region merely because ML confidence
    # is high.
    if (
        not non_medical_hits
        and similarity >= 0.70
        and ocr_confidence >= 0.70
    ):
        decision = "MEDICINE"
    elif non_medical_hits:
        decision = "NON_MEDICINE"
    else:
        decision = "REVIEW"

    if not reasons:
        reasons.append("insufficient evidence")

    return decision, score, similarity, "; ".join(reasons)


def main():
    print("=" * 70)
    print("FINAL MEDICINE VERIFICATION / INTEGRATION")
    print("=" * 70)

    if not OCR_JSON.exists():
        raise FileNotFoundError(OCR_JSON)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(CLASS_MAPPING_PATH)

    with open(OCR_JSON, encoding="utf-8") as f:
        ocr_data = json.load(f)

    regions = ocr_data["regions"]
    image_path = PROJECT_ROOT / ocr_data["image"]

    if not image_path.exists():
        # Fallback for an absolute path stored by the OCR experiment.
        image_path = Path(ocr_data["image"])

    if not image_path.exists():
        raise FileNotFoundError(
            f"Prescription image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    class_names = load_class_names()

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    rows = []

    for region in regions:
        region_id = region["region_id"]
        ocr_text = region.get("text", "")
        ocr_confidence = float(
            region.get("recognition_confidence") or 0.0
        )

        crop = crop_region(image, region)
        predictions = ml_predict(
            session,
            crop,
            class_names,
        )

        top1 = predictions[0]
        top2 = predictions[1]
        top3 = predictions[2]

        decision, evidence_score, similarity, reasons = candidate_evidence(
            ocr_text,
            ocr_confidence,
            top1["medicine"],
            class_names,
        )

        rows.append(
            {
                "region_id": region_id,
                "ocr_text": ocr_text,
                "ocr_confidence": f"{ocr_confidence:.6f}",

                "ml_top1_medicine": top1["medicine"],
                "ml_top1_confidence": f"{top1['confidence']:.6f}",

                "ml_top2_medicine": top2["medicine"],
                "ml_top2_confidence": f"{top2['confidence']:.6f}",

                "ml_top3_medicine": top3["medicine"],
                "ml_top3_confidence": f"{top3['confidence']:.6f}",

                "ocr_ml_similarity": f"{similarity:.6f}",
                "evidence_score": evidence_score,
                "decision": decision,
                "reason": reasons,
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else []

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1

    print(f"Prescription image : {image_path}")
    print(f"OCR regions        : {len(regions)}")
    print(f"MEDICINE           : {counts.get('MEDICINE', 0)}")
    print(f"REVIEW             : {counts.get('REVIEW', 0)}")
    print(f"NON_MEDICINE       : {counts.get('NON_MEDICINE', 0)}")
    print(f"Results saved      : {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
