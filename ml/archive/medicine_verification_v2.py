from pathlib import Path
import csv


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GATE_RESULTS = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_candidate_gate_results.csv"
)

MATCHER_RESULTS = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_candidate_matching_results.csv"
)

INFERENCE_RESULTS = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_candidate_inference_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_verification_v2_results.csv"
)


# ============================================================
# THRESHOLDS
# Preliminary engineering thresholds
# ============================================================

ML_HIGH = 0.80
ML_MEDIUM = 0.60

SIMILARITY_STRONG = 0.60
SIMILARITY_MEDIUM = 0.40

OCR_HIGH = 0.90
OCR_MEDIUM = 0.70

ACCEPT_SCORE = 7
REVIEW_SCORE = 3


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def index_by_region(rows):
    return {
        str(row["region_id"]): row
        for row in rows
    }


def get_ml_prediction(inference_row):
    return inference_row.get("ml_top1_medicine", "").strip()


def get_ml_confidence(inference_row):
    return safe_float(inference_row.get("ml_top1_confidence", 0))


def get_fuzzy_medicine(matcher_row):
    return matcher_row.get("best_medicine", "").strip()


def get_similarity(matcher_row):
    return safe_float(
        matcher_row.get("best_similarity", 0)
    )


# ============================================================
# EVIDENCE FUSION
# ============================================================

def verify_region(gate, matcher, inference):

    gate_decision = gate.get("filter_decision", "").strip()

    ocr_text = gate.get("ocr_text", "").strip()

    ocr_confidence = safe_float(
        gate.get("ocr_confidence", 0)
    )

    ml_medicine = get_ml_prediction(inference)
    ml_confidence = get_ml_confidence(inference)

    fuzzy_medicine = get_fuzzy_medicine(matcher)
    similarity = get_similarity(matcher)

    score = 0
    reasons = []

    # --------------------------------------------------------
    # 1. Gate decision
    # --------------------------------------------------------

    if gate_decision == "NON_MEDICINE":
        return {
            "final_decision": "REJECT",
            "verification_score": -10,
            "reason": "Candidate gate rejected region",
            "ml_medicine": ml_medicine,
            "ml_confidence": ml_confidence,
            "fuzzy_medicine": fuzzy_medicine,
            "similarity": similarity,
        }

    if gate_decision == "MEDICINE_CANDIDATE":
        score += 2
        reasons.append("gate_candidate")

    elif gate_decision == "REVIEW":
        score += 0
        reasons.append("gate_review")

    # --------------------------------------------------------
    # 2. OCR confidence
    # --------------------------------------------------------

    if ocr_confidence >= OCR_HIGH:
        score += 2
        reasons.append("high_ocr_confidence")

    elif ocr_confidence >= OCR_MEDIUM:
        score += 1
        reasons.append("medium_ocr_confidence")

    else:
        score -= 1
        reasons.append("low_ocr_confidence")

    # --------------------------------------------------------
    # 3. Fuzzy medicine-name similarity
    # --------------------------------------------------------

    if similarity >= SIMILARITY_STRONG:
        score += 3
        reasons.append("strong_fuzzy_match")

    elif similarity >= SIMILARITY_MEDIUM:
        score += 2
        reasons.append("medium_fuzzy_match")

    elif similarity >= 0.30:
        score += 1
        reasons.append("weak_fuzzy_match")

    else:
        score -= 1
        reasons.append("very_weak_fuzzy_match")

    # --------------------------------------------------------
    # 4. ML confidence
    # --------------------------------------------------------

    if ml_confidence >= ML_HIGH:
        score += 3
        reasons.append("high_ml_confidence")

    elif ml_confidence >= ML_MEDIUM:
        score += 2
        reasons.append("medium_ml_confidence")

    else:
        score += 0
        reasons.append("low_ml_confidence")

    # --------------------------------------------------------
    # 5. Agreement between fuzzy matcher and ML
    # --------------------------------------------------------

    agreement = False

    if (
        fuzzy_medicine
        and ml_medicine
        and fuzzy_medicine.lower() == ml_medicine.lower()
    ):
        agreement = True
        score += 3
        reasons.append("ocr_ml_agreement")
    else:
        score -= 2
        reasons.append("ocr_ml_disagreement")

    # --------------------------------------------------------
    # 6. Very weak OCR text
    # --------------------------------------------------------

    if len(ocr_text.strip()) <= 2:
        score -= 2
        reasons.append("very_short_ocr_text")

    # --------------------------------------------------------
    # 7. Final decision
    # --------------------------------------------------------

    if score >= ACCEPT_SCORE and agreement:
        decision = "ACCEPT"

    elif score >= REVIEW_SCORE:
        decision = "REVIEW"

    else:
        decision = "REJECT"

    # --------------------------------------------------------
    # Explain disagreement
    # --------------------------------------------------------

    if not agreement:
        reason = "ML/fuzzy candidate disagreement"
    else:
        reason = "Evidence supports medicine candidate"

    return {
        "final_decision": decision,
        "verification_score": score,
        "reason": reason,
        "ml_medicine": ml_medicine,
        "ml_confidence": ml_confidence,
        "fuzzy_medicine": fuzzy_medicine,
        "similarity": similarity,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDICINE VERIFICATION V2 - EVIDENCE FUSION")
    print("=" * 70)

    print(f"Gate results      : {GATE_RESULTS}")
    print(f"Matcher results   : {MATCHER_RESULTS}")
    print(f"Inference results : {INFERENCE_RESULTS}")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        GATE_RESULTS,
        MATCHER_RESULTS,
        INFERENCE_RESULTS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    gate_rows = load_csv(GATE_RESULTS)
    matcher_rows = load_csv(MATCHER_RESULTS)
    inference_rows = load_csv(INFERENCE_RESULTS)

    gate_by_region = index_by_region(gate_rows)
    matcher_by_region = index_by_region(matcher_rows)
    inference_by_region = index_by_region(inference_rows)

    print()
    print(f"Gate regions      : {len(gate_rows)}")
    print(f"Matcher regions   : {len(matcher_rows)}")
    print(f"Inference regions : {len(inference_rows)}")

    # --------------------------------------------------------
    # Verify each region
    # --------------------------------------------------------

    results = []

    for region_id, gate in gate_by_region.items():

        matcher = matcher_by_region.get(region_id, {})

        inference = inference_by_region.get(region_id, {})

        # Non-medicine regions were intentionally skipped
        # by routed inference.
        if gate.get("filter_decision") == "NON_MEDICINE":

            result = {
                "region_id": region_id,
                "ocr_text": gate.get("ocr_text", ""),
                "gate_decision": gate.get(
                    "filter_decision", ""
                ),
                "ml_medicine": "",
                "ml_confidence": "",
                "fuzzy_medicine": matcher.get(
                    "best_medicine", ""
                ),
                "similarity": matcher.get(
                    "best_similarity", ""
                ),
                "verification_score": -10,
                "final_decision": "REJECT",
                "reason": "Candidate gate rejected region",
            }

        else:

            verification = verify_region(
                gate,
                matcher,
                inference,
            )

            result = {
                "region_id": region_id,
                "ocr_text": gate.get("ocr_text", ""),
                "gate_decision": gate.get(
                    "filter_decision", ""
                ),
                "ml_medicine": verification[
                    "ml_medicine"
                ],
                "ml_confidence": verification[
                    "ml_confidence"
                ],
                "fuzzy_medicine": verification[
                    "fuzzy_medicine"
                ],
                "similarity": verification[
                    "similarity"
                ],
                "verification_score": verification[
                    "verification_score"
                ],
                "final_decision": verification[
                    "final_decision"
                ],
                "reason": verification[
                    "reason"
                ],
            }

        results.append(result)

        print()
        print(
            f"Region {region_id}: "
            f"{gate.get('ocr_text', '')}"
        )

        print(
            f"  Gate       : "
            f"{result['gate_decision']}"
        )

        print(
            f"  ML         : "
            f"{result['ml_medicine'] or '-'} "
            f"({result['ml_confidence'] or '-'})"
        )

        print(
            f"  Fuzzy      : "
            f"{result['fuzzy_medicine'] or '-'} "
            f"({result['similarity'] or '-'})"
        )

        print(
            f"  Score      : "
            f"{result['verification_score']}"
        )

        print(
            f"  Decision   : "
            f"{result['final_decision']}"
        )

        print(
            f"  Reason     : "
            f"{result['reason']}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "region_id",
        "ocr_text",
        "gate_decision",
        "ml_medicine",
        "ml_confidence",
        "fuzzy_medicine",
        "similarity",
        "verification_score",
        "final_decision",
        "reason",
    ]

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
        writer.writerows(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    counts = {
        "ACCEPT": 0,
        "REVIEW": 0,
        "REJECT": 0,
    }

    for result in results:
        decision = result["final_decision"]

        if decision in counts:
            counts[decision] += 1

    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"Total regions : {len(results)}")
    print(f"ACCEPT        : {counts['ACCEPT']}")
    print(f"REVIEW        : {counts['REVIEW']}")
    print(f"REJECT        : {counts['REJECT']}")

    print()
    print(f"Results saved to:")
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()