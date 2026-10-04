from pathlib import Path
import csv
import re


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_candidate_matching_results.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml/results/ocr_ml003/medicine_candidate_gate_results.csv"
)


# ============================================================
# NON-MEDICINE PATTERNS
# ============================================================

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
    r"\bregd\b",
    r"\bregistration\b",

    # Common medical measurement/diagnostic terms
    r"\becg\b",
    r"\bbp\b",
    r"\bbpm\b",
    r"\bspo2\b",
    r"\bpulse\b",
]


# ============================================================
# CLINICAL / INSTRUCTION PATTERNS
# ============================================================

CLINICAL_PATTERNS = [
    r"\bdays?\b",
    r"\bmg\b",
    r"\bml\b",
    r"\bkg\b",
    r"\bcm\b",
    r"\bbpm\b",
    r"\btemp\b",
    r"\btablet\b",
    r"\btab\b",
    r"\bcap\b",
    r"\bcapsule\b",
    r"\bsyrup\b",
    r"\bdose\b",
    r"\bafter\b",
    r"\bbefore\b",
]


# ============================================================
# DOSAGE / MEASUREMENT PATTERNS
# ============================================================

DOSAGE_PATTERNS = [
    r"\d+\s*mg\b",
    r"\d+\s*ml\b",
    r"\d+\s*kg\b",
    r"\d+\s*cm\b",
    r"\d+\s*bpm\b",
    r"\d+\s*/\s*\d+",
    r"\d+\s*-\s*\d+",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    """
    Normalize OCR text for pattern matching.
    """
    if text is None:
        return ""

    text = str(text).strip()

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def find_matches(text, patterns):
    """
    Return all regex patterns that match the text.
    """
    matches = []

    text_lower = text.lower()

    for pattern in patterns:
        if re.search(pattern, text_lower):
            matches.append(pattern)

    return matches


def safe_float(value, default=0.0):
    """
    Safely convert CSV values to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_gate_score(row):
    """
    Calculate the heuristic candidate score.

    This score is an engineering heuristic.
    It is NOT a probability and is NOT a clinical confidence score.
    """

    text = normalize_text(row.get("ocr_text", ""))

    ocr_confidence = safe_float(
        row.get("ocr_confidence", 0.0)
    )

    similarity = safe_float(
        row.get("best_similarity", 0.0)
    )

    filter_decision = str(
        row.get("filter_decision", "")
    ).strip()

    score = 0

    reasons = []

    # --------------------------------------------------------
    # Text length
    # --------------------------------------------------------

    text_length = len(text)

    if text_length <= 1:
        score -= 4
        reasons.append("very short text")

    elif text_length == 2:
        score -= 2
        reasons.append("short text")

    elif 3 <= text_length <= 25:
        score += 1
        reasons.append("reasonable text length")

    elif text_length > 40:
        score -= 2
        reasons.append("very long text")

    # --------------------------------------------------------
    # Short alphabetic OCR token
    #
    # Helps prevent tokens such as:
    # CVE
    # OEP
    # ECG
    # from becoming strong medicine candidates.
    # --------------------------------------------------------

    compact_text = re.sub(
        r"[^a-z0-9]",
        "",
        text.lower()
    )

    if (
        len(compact_text) <= 4
        and compact_text.isalpha()
    ):
        score -= 2
        reasons.append("short alphabetic OCR token")

    # --------------------------------------------------------
    # Non-medicine patterns
    # --------------------------------------------------------

    non_medicine_matches = find_matches(
        text,
        NON_MEDICINE_PATTERNS
    )

    if non_medicine_matches:
        score -= 6
        reasons.append(
            "non-medicine keyword"
        )

    # --------------------------------------------------------
    # Clinical/instruction patterns
    # --------------------------------------------------------

    clinical_matches = find_matches(
        text,
        CLINICAL_PATTERNS
    )

    if clinical_matches:
        score -= 2
        reasons.append(
            "clinical/instruction pattern"
        )

    # --------------------------------------------------------
    # Dosage patterns
    # --------------------------------------------------------

    dosage_matches = find_matches(
        text,
        DOSAGE_PATTERNS
    )

    if dosage_matches:
        score -= 4
        reasons.append(
            "dosage/measurement pattern"
        )

    # --------------------------------------------------------
    # OCR confidence
    # --------------------------------------------------------

    if ocr_confidence >= 0.90:
        score += 2
        reasons.append(
            "high OCR confidence"
        )

    elif ocr_confidence >= 0.70:
        score += 1
        reasons.append(
            "moderate OCR confidence"
        )

    else:
        score -= 1
        reasons.append(
            "low OCR confidence"
        )

    # --------------------------------------------------------
    # Medicine-name similarity
    # --------------------------------------------------------

    if similarity >= 0.60:
        score += 2
        reasons.append(
            "strong medicine-name similarity"
        )

    elif similarity >= 0.40:
        score += 1
        reasons.append(
            "moderate medicine-name similarity"
        )

    elif similarity >= 0.30:
        score += 0
        reasons.append(
            "weak medicine-name similarity"
        )

    else:
        score -= 1
        reasons.append(
            "very weak medicine-name similarity"
        )

    # --------------------------------------------------------
    # Previous matcher decision
    # --------------------------------------------------------

    if filter_decision == "PASS_TO_ML":
        score += 1
        reasons.append(
            "previous matcher passed"
        )

    elif filter_decision == "REJECT":
        score -= 3
        reasons.append(
            "previous matcher rejected"
        )

    return (
        score,
        reasons,
        non_medicine_matches,
        clinical_matches,
        dosage_matches,
    )


# ============================================================
# FINAL GATE DECISION
# ============================================================

def make_gate_decision(
    text,
    score,
    non_medicine_matches,
    dosage_matches,
):
    """
    Convert heuristic score into one of:

        MEDICINE_CANDIDATE
        REVIEW
        NON_MEDICINE

    Conservative design:
    strong non-medicine evidence always wins.
    """

    reasons = []

    # --------------------------------------------------------
    # Strong non-medicine evidence has priority
    # --------------------------------------------------------

    if non_medicine_matches:
        reasons.append(
            "strong non-medicine evidence"
        )

        return "NON_MEDICINE", reasons

    # --------------------------------------------------------
    # Dosage/measurement text is not a medicine-name region
    # --------------------------------------------------------

    if dosage_matches:
        reasons.append(
            "dosage/instruction pattern"
        )

        return "NON_MEDICINE", reasons

    # --------------------------------------------------------
    # Very short text should not be automatically classified
    # as a medicine.
    # Send it for review instead.
    # --------------------------------------------------------

    if len(text.strip()) <= 2:
        reasons.append(
            "very short text requires review"
        )

        return "REVIEW", reasons

    # --------------------------------------------------------
    # Strong candidate
    # --------------------------------------------------------

    if score >= 5:
        reasons.append(
            "strong candidate score"
        )

        return "MEDICINE_CANDIDATE", reasons

    # --------------------------------------------------------
    # Uncertain candidate
    # --------------------------------------------------------

    if score >= 0:
        reasons.append(
            "uncertain candidate requires review"
        )

        return "REVIEW", reasons

    # --------------------------------------------------------
    # Weak candidate
    # --------------------------------------------------------

    reasons.append(
        "insufficient medicine evidence"
    )

    return "NON_MEDICINE", reasons


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MEDICINE CANDIDATE GATE")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    # --------------------------------------------------------
    # Read matcher results
    # --------------------------------------------------------

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            "Input CSV contains no rows."
        )

    print(f"Input regions : {len(rows)}")
    print()

    # --------------------------------------------------------
    # Process each OCR region
    # --------------------------------------------------------

    results = []

    medicine_count = 0
    review_count = 0
    non_medicine_count = 0

    for row in rows:

        region_id = row.get(
            "region_id",
            ""
        )

        text = normalize_text(
            row.get("ocr_text", "")
        )

        # ----------------------------------------------------
        # Calculate score
        # ----------------------------------------------------

        (
            score,
            score_reasons,
            non_medicine_matches,
            clinical_matches,
            dosage_matches,
        ) = calculate_gate_score(row)

        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        decision, decision_reasons = make_gate_decision(
            text=text,
            score=score,
            non_medicine_matches=non_medicine_matches,
            dosage_matches=dosage_matches,
        )

        all_reasons = (
            score_reasons
            + decision_reasons
        )

        reasons_text = "; ".join(
            all_reasons
        )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        if decision == "MEDICINE_CANDIDATE":
            medicine_count += 1

        elif decision == "REVIEW":
            review_count += 1

        elif decision == "NON_MEDICINE":
            non_medicine_count += 1

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            f"Region {region_id}: {text}"
        )

        print(
            f"  Best medicine : "
            f"{row.get('best_medicine', '')}"
        )

        print(
            f"  Similarity    : "
            f"{row.get('best_similarity', '')}"
        )

        print(
            f"  Gate score    : {score}"
        )

        print(
            f"  Decision      : {decision}"
        )

        print()

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        result = dict(row)

        result["candidate_score"] = score
        result["filter_decision"] = decision
        result["reasons"] = reasons_text

        results.append(result)

    # ========================================================
    # SAVE GATE RESULTS
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Preserve the original CSV columns and make sure our
    # important gate columns are present.
    fieldnames = list(rows[0].keys())

    if "candidate_score" not in fieldnames:
        fieldnames.append("candidate_score")

    if "filter_decision" not in fieldnames:
        fieldnames.append("filter_decision")

    if "reasons" not in fieldnames:
        fieldnames.append("reasons")

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("=" * 70)
    print("GATE SUMMARY")
    print("=" * 70)

    print(
        f"MEDICINE_CANDIDATE : {medicine_count}"
    )

    print(
        f"REVIEW             : {review_count}"
    )

    print(
        f"NON_MEDICINE       : {non_medicine_count}"
    )

    print()

    print(
        f"Gate results saved to:\n{OUTPUT_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()