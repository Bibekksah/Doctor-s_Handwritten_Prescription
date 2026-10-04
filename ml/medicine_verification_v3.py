# Medicine Verification V3
# Evidence fusion: Gate + OCR confidence + ML-003 + fuzzy matching
# Fuzzy disagreement is supporting negative evidence only, not a hard rejection.

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = PROJECT_ROOT / "ml/results/ocr_ml003"

GATE_CSV = OUT / "medicine_candidate_gate_results.csv"
MATCHER_CSV = OUT / "medicine_candidate_matching_results.csv"
INFERENCE_CSV = OUT / "medicine_candidate_inference_results.csv"
OUTPUT_CSV = OUT / "medicine_verification_v3_results.csv"


def f(value, default=0.0):
    try:
        if value is None or str(value).strip() in ("", "-"):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def load(path):
    with open(path, encoding="utf-8", newline="") as file:
        return {
            str(row["region_id"]).strip(): row
            for row in csv.DictReader(file)
        }


def verify(gate, matcher, inference):
    gate_decision = gate.get("filter_decision", "")
    ocr_text = gate.get("ocr_text", "").strip()
    ocr_conf = f(gate.get("ocr_confidence"))

    fuzzy = matcher.get("best_medicine", "").strip()
    similarity = f(matcher.get("best_similarity"))

    ml = inference.get("ml_top1_medicine", "").strip()
    ml_conf = f(inference.get("ml_top1_confidence"))
    ml2_conf = f(inference.get("ml_top2_confidence"))

    # Gate rejection remains final.
    if gate_decision == "NON_MEDICINE":
        return -10, "REJECT", "Candidate gate rejected region", [
         "Candidate gate rejected region"
        ]

    score = 0
    reasons = []

    # Candidate gate
    if gate_decision == "MEDICINE_CANDIDATE":
        score += 2
        reasons.append("medicine candidate gate")
    elif gate_decision == "REVIEW":
        score += 1
        reasons.append("gate review")

    # OCR confidence = supporting evidence
    if ocr_conf >= 0.90:
        score += 2
        reasons.append("high OCR confidence")
    elif ocr_conf >= 0.70:
        score += 1
        reasons.append("moderate OCR confidence")
    else:
        reasons.append("low OCR confidence")

    # Very short OCR text is weak evidence
    if len(ocr_text) <= 1:
        score -= 2
        reasons.append("very short OCR text")
    elif len(ocr_text) <= 2:
        score -= 1
        reasons.append("short OCR text")

    # ML-003 is the main medicine recognizer
    if ml:
        if ml_conf >= 0.90:
            score += 4
            reasons.append("very high ML confidence")
        elif ml_conf >= 0.80:
            score += 3
            reasons.append("high ML confidence")
        elif ml_conf >= 0.60:
            score += 2
            reasons.append("moderate ML confidence")
        elif ml_conf >= 0.45:
            score += 1
            reasons.append("weak ML confidence")
        else:
            score -= 1
            reasons.append("low ML confidence")
    else:
        reasons.append("no ML prediction")

    # Fuzzy matching is only supporting evidence.
    if similarity >= 0.60:
        score += 2
        reasons.append("strong fuzzy similarity")
    elif similarity >= 0.40:
        score += 1
        reasons.append("moderate fuzzy similarity")
    elif similarity > 0:
        reasons.append("weak fuzzy similarity")

    # Agreement gets a bonus.
    agreement = (
        bool(fuzzy)
        and bool(ml)
        and fuzzy.casefold() == ml.casefold()
    )

    if agreement:
        score += 3
        reasons.append("fuzzy candidate agrees with ML")

    # Small ML margin means less certainty.
    if ml_conf > 0 and (ml_conf - ml2_conf) < 0.15:
        score -= 1
        reasons.append("small ML top-1 margin")

    # Decision policy.
    #
    # ACCEPT requires independent evidence:
    # high ML + fuzzy agreement + reasonable fuzzy similarity.
    #
    # High ML alone never automatically becomes ACCEPT.
    if (
        ml
        and ml_conf >= 0.90
        and similarity >= 0.40
        and agreement
        and score >= 8
    ):
        decision = "ACCEPT"
        reason = "Strong multi-source agreement"

    elif ml and ml_conf >= 0.80 and score >= 6:
        decision = "REVIEW"
        reason = "Strong ML evidence; human verification required"

    elif ml and ml_conf >= 0.60 and score >= 4:
        decision = "REVIEW"
        reason = "Moderate ML evidence; insufficient independent OCR support"

    elif ml and score >= 2:
        decision = "REVIEW"
        reason = "Insufficient evidence for automatic acceptance"

    else:
        decision = "REJECT"
        reason = "Insufficient evidence"

    return score, decision, reason, reasons


def main():

    print("=" * 70)
    print("MEDICINE VERIFICATION V3 - EVIDENCE FUSION")
    print("=" * 70)

    gates = load(GATE_CSV)
    matchers = load(MATCHER_CSV)
    inferences = load(INFERENCE_CSV)

    print("Gate regions      :", len(gates))
    print("Matcher regions   :", len(matchers))
    print("Inference regions :", len(inferences))
    print()

    results = []
    counts = {
        "ACCEPT": 0,
        "REVIEW": 0,
        "REJECT": 0,
    }

    for region_id in sorted(
        gates,
        key=lambda x: int(x) if x.isdigit() else x
    ):

        gate = gates[region_id]
        matcher = matchers.get(region_id, {})
        inference = inferences.get(region_id, {})

        result = verify(gate, matcher, inference)

        score = result[0]
        decision = result[1]
        reason = result[2]
        details = result[3] if len(result) > 3 else []

        ml = inference.get("ml_top1_medicine", "")
        ml_conf = f(inference.get("ml_top1_confidence"))

        fuzzy = matcher.get("best_medicine", "")
        similarity = f(matcher.get("best_similarity"))

        counts[decision] += 1

        print(f"Region {region_id}: {gate.get('ocr_text', '')}")
        print(f"  Gate       : {gate.get('filter_decision', '')}")

        if ml:
            print(f"  ML         : {ml} ({ml_conf:.6f})")
        else:
            print("  ML         : - (-)")

        if fuzzy:
            print(f"  Fuzzy      : {fuzzy} ({similarity:.4f})")
        else:
            print("  Fuzzy      : - (-)")

        print(f"  Score      : {score}")
        print(f"  Decision   : {decision}")
        print(f"  Reason     : {reason}")
        print()

        results.append({
            "region_id": region_id,
            "ocr_text": gate.get("ocr_text", ""),
            "ocr_confidence": gate.get("ocr_confidence", ""),
            "gate_score": gate.get("candidate_score", ""),
            "gate_decision": gate.get("filter_decision", ""),
            "fuzzy_medicine": fuzzy,
            "fuzzy_similarity": f"{similarity:.4f}",
            "ml_top1_medicine": ml,
            "ml_top1_confidence": f"{ml_conf:.6f}",
            "ml_top2_medicine": inference.get("ml_top2_medicine", ""),
            "ml_top2_confidence": inference.get("ml_top2_confidence", ""),
            "ml_top3_medicine": inference.get("ml_top3_medicine", ""),
            "ml_top3_confidence": inference.get("ml_top3_confidence", ""),
            "evidence_score": score,
            "verification_decision": decision,
            "verification_reason": reason,
            "evidence_details": "; ".join(details),
        })

    fields = list(results[0].keys())

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print("Total regions :", len(results))
    print("ACCEPT        :", counts["ACCEPT"])
    print("REVIEW        :", counts["REVIEW"])
    print("REJECT        :", counts["REJECT"])
    print()
    print("Results saved to:")
    print(OUTPUT_CSV)
    print("=" * 70)


if __name__ == "__main__":
    main()
