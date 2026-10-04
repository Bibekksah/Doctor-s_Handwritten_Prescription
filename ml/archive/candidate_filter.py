from pathlib import Path
import json
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OCR_JSON = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ocr"
    / "existing_ocr_result.json"
)

LABELS_CSV = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ocr_ml003"
    / "manual_region_labels.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ocr_ml003"
    / "candidate_filter_v2_results.csv"
)


NON_MEDICINE_PATTERNS = [
    r"\bdr\b",
    r"\bdoctor\b",
    r"\bname\b",
    r"\baddress\b",
    r"\bphone\b",
    r"\bmobile\b",
    r"\bage\b",
    r"\bgender\b",
    r"\bdate\b",
    r"\bclinic\b",
    r"\bhospital\b",
    r"\bhyderabad\b",
    r"\bhypertension\b",
    r"\bfever\b",
    r"\bdengue\b",
    r"\becg\b",
    r"\bbp\b",
    r"\bspo2\b",
    r"\bpulse\b",
    r"\bbpm\b",
    r"\bregd\b",
    r"\bregistration\b",
]


CLINICAL_PATTERNS = [
    r"\bdays?\b",
    r"\bday\b",
    r"\bbp\b",
    r"\bbpm\b",
    r"\bmg\b",
    r"\bml\b",
    r"\bkg\b",
    r"\bcm\b",
    r"\btemp\b",
]


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_pattern(text, patterns):
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


def candidate_score(text, confidence, bbox):
    """
    Heuristic score.

    Higher score = more likely to be a medicine candidate.

    This is NOT a trained ML score.
    It is a rule-based development experiment.
    """

    text = normalize_text(text)

    score = 0
    reasons = []

    # --------------------------------------------------
    # 1. Recognition confidence
    # --------------------------------------------------

    if confidence >= 0.90:
        score += 2
        reasons.append("high_ocr_confidence")

    elif confidence >= 0.70:
        score += 1
        reasons.append("moderate_ocr_confidence")

    else:
        score -= 1
        reasons.append("low_ocr_confidence")

    # --------------------------------------------------
    # 2. Obvious non-medicine text
    # --------------------------------------------------

    if contains_pattern(text, NON_MEDICINE_PATTERNS):
        score -= 5
        reasons.append("non_medicine_keyword")

    # --------------------------------------------------
    # 3. Clinical/instructional pattern
    # --------------------------------------------------

    if contains_pattern(text, CLINICAL_PATTERNS):
        score -= 2
        reasons.append("clinical_pattern")

    # --------------------------------------------------
    # 4. Text length
    # --------------------------------------------------

    if 3 <= len(text) <= 25:
        score += 1
        reasons.append("reasonable_text_length")

    elif len(text) > 35:
        score -= 1
        reasons.append("long_text")

    # --------------------------------------------------
    # 5. Bounding box
    # --------------------------------------------------

    width = bbox.get("width", 0)
    height = bbox.get("height", 0)

    if width > 20 and height > 8:
        score += 1
        reasons.append("reasonable_region_size")

    # --------------------------------------------------
    # Final decision
    # --------------------------------------------------

    if score <= -2:
        decision = "REJECT"

    elif score >= 2:
        decision = "PASS_TO_ML"

    else:
        decision = "REVIEW"

    return score, decision, reasons


def main():

    if not OCR_JSON.exists():
        raise FileNotFoundError(
            f"OCR result not found:\n{OCR_JSON}"
        )

    if not LABELS_CSV.exists():
        raise FileNotFoundError(
            f"Manual labels not found:\n{LABELS_CSV}"
        )

    with open(OCR_JSON, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    labels = pd.read_csv(LABELS_CSV)

    label_map = {
        int(row["region_id"]): row["label"]
        for _, row in labels.iterrows()
    }

    rows = []

    for item in ocr_data["regions"]:

        region_id = int(item["region_id"])

        text = item.get("text", "")

        confidence = float(
            item.get("recognition_confidence") or 0.0
        )

        bbox = item.get("bbox", {})

        score, decision, reasons = candidate_score(
            text=text,
            confidence=confidence,
            bbox=bbox,
        )

        rows.append(
            {
                "region_id": region_id,
                "ocr_text": text,
                "ocr_confidence": round(confidence, 4),
                "bbox_width": bbox.get("width", 0),
                "bbox_height": bbox.get("height", 0),
                "candidate_score": score,
                "filter_decision": decision,
                "manual_label": label_map.get(
                    region_id,
                    "UNKNOWN"
                ),
                "reasons": ";".join(reasons),
            }
        )

    result = pd.DataFrame(rows)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 80)
    print("CANDIDATE FILTER V2")
    print("=" * 80)

    print(f"Total regions : {len(result)}")

    print(
        f"Rejected      : "
        f"{(result['filter_decision'] == 'REJECT').sum()}"
    )

    print(
        f"Review        : "
        f"{(result['filter_decision'] == 'REVIEW').sum()}"
    )

    print(
        f"Passed to ML  : "
        f"{(result['filter_decision'] == 'PASS_TO_ML').sum()}"
    )

    print()
    print(result.to_string(index=False))

    print()
    print(f"Saved:")
    print(OUTPUT_CSV)

    print("=" * 80)


if __name__ == "__main__":
    main()