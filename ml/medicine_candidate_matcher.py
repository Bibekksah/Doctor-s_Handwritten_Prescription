# ml/medicine_candidate_matcher.py

"""
Medicine Candidate Matcher
===========================

Purpose
-------
Use OCR text from the prescription to find the closest candidates
among the 78 medicine classes used by ML-003.

Pipeline:

    PaddleOCR
        ↓
    Candidate Filter V2
        ↓
    Medicine Candidate Matcher
        ↓
    ML-003
        ↓
    Medicine Verifier

IMPORTANT
---------
This module produces medicine CANDIDATES.

It does NOT make the final medicine recognition decision.

A high similarity score does not prove that the OCR region is a
medicine name.
"""

from pathlib import Path
import csv
import json
import re
from difflib import SequenceMatcher


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASS_MAPPING_PATH = (
    PROJECT_ROOT
    / "ml"
    / "config"
    / "class_mapping.json"
)

INPUT_CSV = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ocr_ml003"
    / "candidate_filter_v2_results.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "ocr_ml003"
    / "medicine_candidate_matching_results.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_NUM_CLASSES = 78

TOP_K = 5

# Initial thresholds.
#
# These are engineering thresholds only.
# They have NOT been validated as final production thresholds.

STRONG_MATCH_THRESHOLD = 0.60
POSSIBLE_MATCH_THRESHOLD = 0.30


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize OCR text.

    Example:

        "rab.Arwatsmp -20dy"

    becomes approximately:

        "rab arwatsmp 20dy"

    This function intentionally does NOT aggressively remove
    dosage/instruction information yet. That should be handled
    after we evaluate the raw matcher results.
    """

    if not text:
        return ""

    text = str(text).lower().strip()

    # Replace punctuation with spaces.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def compact_text(text: str) -> str:
    """
    Remove spaces and non-alphanumeric characters.

    Example:

        "Rab.Arwatsmp -20dy"

    becomes:

        "rabarwatsmp20dy"
    """

    if not text:
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(text).lower()
    )


# ============================================================
# SIMILARITY
# ============================================================

def similarity_score(
    ocr_text: str,
    medicine_name: str
) -> float:
    """
    Calculate similarity between OCR text and a medicine name.

    Two character-level comparisons are used:

    1. Normalized text
    2. Compact text

    Final score:

        40% normalized similarity
        60% compact similarity

    Returns:
        float between 0.0 and 1.0
    """

    ocr_normalized = normalize_text(
        ocr_text
    )

    medicine_normalized = normalize_text(
        medicine_name
    )

    if not ocr_normalized:
        return 0.0

    if not medicine_normalized:
        return 0.0

    # --------------------------------------------------------
    # Normalized similarity
    # --------------------------------------------------------

    normalized_score = SequenceMatcher(
        None,
        ocr_normalized,
        medicine_normalized
    ).ratio()

    # --------------------------------------------------------
    # Compact similarity
    # --------------------------------------------------------

    ocr_compact = compact_text(
        ocr_text
    )

    medicine_compact = compact_text(
        medicine_name
    )

    if not ocr_compact:
        return 0.0

    compact_score = SequenceMatcher(
        None,
        ocr_compact,
        medicine_compact
    ).ratio()

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    score = (
        0.40 * normalized_score
        +
        0.60 * compact_score
    )

    return float(score)


# ============================================================
# LOAD CLASS MAPPING
# ============================================================

def load_medicine_classes():
    """
    Load the medicine class mapping used by ML-003.

    Actual project format:

    {
        "num_classes": 78,

        "class_to_idx": {
            "Ace": 0,
            "Aceta": 1,
            ...
        },

        "idx_to_class": {
            "0": "Ace",
            "1": "Aceta",
            ...
        }
    }

    Returns:

        {
            0: "Ace",
            1: "Aceta",
            ...
        }
    """

    print(
        f"\nLoading class mapping:"
    )

    print(
        CLASS_MAPPING_PATH
    )

    if not CLASS_MAPPING_PATH.exists():
        raise FileNotFoundError(
            "\nClass mapping file not found:\n"
            f"{CLASS_MAPPING_PATH}"
        )

    with open(
        CLASS_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    # --------------------------------------------------------
    # Validate number of classes
    # --------------------------------------------------------

    num_classes = data.get(
        "num_classes"
    )

    if num_classes is not None:

        print(
            f"Declared number of classes: "
            f"{num_classes}"
        )

    # --------------------------------------------------------
    # Preferred mapping
    # --------------------------------------------------------

    if "idx_to_class" in data:

        mapping = data["idx_to_class"]

        medicine_classes = {
            int(class_id): str(
                medicine_name
            )
            for class_id, medicine_name
            in mapping.items()
        }

    # --------------------------------------------------------
    # Fallback mapping
    # --------------------------------------------------------

    elif "class_to_idx" in data:

        mapping = data["class_to_idx"]

        medicine_classes = {
            int(class_id): str(
                medicine_name
            )
            for medicine_name, class_id
            in mapping.items()
        }

    else:

        raise ValueError(
            "\nUnsupported class_mapping.json format.\n"
            f"Found keys: {list(data.keys())}"
        )

    # --------------------------------------------------------
    # Validate number of classes
    # --------------------------------------------------------

    actual_classes = len(
        medicine_classes
    )

    print(
        f"Medicine classes loaded: "
        f"{actual_classes}"
    )

    if actual_classes != EXPECTED_NUM_CLASSES:

        print(
            "\nWARNING:"
            f" Expected {EXPECTED_NUM_CLASSES} "
            f"classes but found {actual_classes}."
        )

    # --------------------------------------------------------
    # Display first few classes
    # --------------------------------------------------------

    print(
        "\nFirst medicine classes:"
    )

    for class_id in sorted(
        medicine_classes
    )[:10]:

        print(
            f"  {class_id}: "
            f"{medicine_classes[class_id]}"
        )

    return medicine_classes


# ============================================================
# FIND TOP MEDICINE CANDIDATES
# ============================================================

def find_candidates(
    ocr_text: str,
    medicine_classes: dict,
    top_k: int = TOP_K
):
    """
    Compare OCR text against all medicine classes.

    Returns the top-k candidates.
    """

    candidates = []

    # Empty OCR text cannot produce a meaningful candidate.
    if not ocr_text.strip():

        return [
            {
                "class_id": class_id,
                "medicine_name": medicine_name,
                "similarity": 0.0
            }

            for class_id, medicine_name
            in list(
                sorted(
                    medicine_classes.items()
                )
            )[:top_k]
        ]

    # --------------------------------------------------------
    # Compare against all 78 medicines
    # --------------------------------------------------------

    for class_id, medicine_name in (
        medicine_classes.items()
    ):

        score = similarity_score(
            ocr_text,
            medicine_name
        )

        candidates.append(
            {
                "class_id": class_id,
                "medicine_name": medicine_name,
                "similarity": score
            }
        )

    # --------------------------------------------------------
    # Sort highest similarity first
    # --------------------------------------------------------

    candidates.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    return candidates[:top_k]


# ============================================================
# CLASSIFY MATCH STRENGTH
# ============================================================

def candidate_status(
    best_score: float
) -> str:
    """
    Classify the strength of the best fuzzy match.

    IMPORTANT:
    These thresholds are preliminary and must be evaluated.
    """

    if best_score >= STRONG_MATCH_THRESHOLD:

        return "STRONG_CANDIDATE"

    if best_score >= POSSIBLE_MATCH_THRESHOLD:

        return "POSSIBLE_CANDIDATE"

    return "NO_STRONG_MATCH"


# ============================================================
# READ INPUT CSV
# ============================================================

def validate_input_columns(
    fieldnames
):
    """
    Make sure the candidate-filter CSV has the expected columns.
    """

    if not fieldnames:

        raise ValueError(
            "Input CSV has no columns."
        )

    required_columns = [
        "region_id",
        "ocr_text",
        "ocr_confidence"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in fieldnames
    ]

    if missing_columns:

        raise ValueError(
            "\nMissing required columns:\n"
            f"{missing_columns}\n\n"
            "Available columns:\n"
            f"{fieldnames}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 75
    )

    print(
        "MEDICINE CANDIDATE MATCHER"
    )

    print(
        "=" * 75
    )

    # --------------------------------------------------------
    # Project information
    # --------------------------------------------------------

    print(
        "\nProject root:"
    )

    print(
        PROJECT_ROOT
    )

    # --------------------------------------------------------
    # Load medicine classes
    # --------------------------------------------------------

    medicine_classes = (
        load_medicine_classes()
    )

    # --------------------------------------------------------
    # Check input CSV
    # --------------------------------------------------------

    print(
        "\nInput CSV:"
    )

    print(
        INPUT_CSV
    )

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            "\nCandidate filter CSV not found:\n"
            f"{INPUT_CSV}"
        )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f
        )

        # ----------------------------------------------------
        # Display input columns
        # ----------------------------------------------------

        print(
            "\nInput CSV columns:"
        )

        print(
            reader.fieldnames
        )

        # ----------------------------------------------------
        # Validate CSV
        # ----------------------------------------------------

        validate_input_columns(
            reader.fieldnames
        )

        # ----------------------------------------------------
        # Process each OCR region
        # ----------------------------------------------------

        for row in reader:

            region_id = (
                row.get(
                    "region_id",
                    ""
                )
            )

            ocr_text = (
                row.get(
                    "ocr_text",
                    ""
                )
                or ""
            ).strip()

            ocr_confidence = (
                row.get(
                    "ocr_confidence",
                    ""
                )
                or ""
            )

            bbox_width = (
                row.get(
                    "bbox_width",
                    ""
                )
                or ""
            )

            bbox_height = (
                row.get(
                    "bbox_height",
                    ""
                )
                or ""
            )

            candidate_score = (
                row.get(
                    "candidate_score",
                    ""
                )
                or ""
            )

            filter_decision = (
                row.get(
                    "filter_decision",
                    ""
                )
                or ""
            )

            manual_label = (
                row.get(
                    "manual_label",
                    ""
                )
                or ""
            )

            reasons = (
                row.get(
                    "reasons",
                    ""
                )
                or ""
            )

            # ------------------------------------------------
            # Debug output
            # ------------------------------------------------

            print(
                f"\nRegion {region_id}: "
                f"OCR='{ocr_text}'"
            )

            # ------------------------------------------------
            # Empty OCR text
            # ------------------------------------------------

            if not ocr_text:

                print(
                    "  WARNING: Empty OCR text"
                )

            # ------------------------------------------------
            # Find candidates
            # ------------------------------------------------

            candidates = find_candidates(
                ocr_text,
                medicine_classes,
                TOP_K
            )

            # ------------------------------------------------
            # Best candidate
            # ------------------------------------------------

            best = candidates[0]

            best_similarity = (
                best["similarity"]
            )

            status = candidate_status(
                best_similarity
            )

            # ------------------------------------------------
            # Display best result
            # ------------------------------------------------

            print(
                f"  → {best['medicine_name']} "
                f"({best_similarity:.3f}) "
                f"[{status}]"
            )

            # ------------------------------------------------
            # Prepare output
            # ------------------------------------------------

            result = {
                "region_id": region_id,

                "ocr_text": ocr_text,

                "ocr_confidence": (
                    ocr_confidence
                ),

                "bbox_width": (
                    bbox_width
                ),

                "bbox_height": (
                    bbox_height
                ),

                "candidate_score": (
                    candidate_score
                ),

                "filter_decision": (
                    filter_decision
                ),

                "manual_label": (
                    manual_label
                ),

                "reasons": (
                    reasons
                ),

                "best_medicine": (
                    best["medicine_name"]
                ),

                "best_class_id": (
                    best["class_id"]
                ),

                "best_similarity": round(
                    best_similarity,
                    4
                ),

                "status": status,

                # --------------------------------------------
                # Top 5 candidates
                # --------------------------------------------

                "candidate_1": (
                    candidates[0]["medicine_name"]
                ),

                "candidate_1_score": round(
                    candidates[0]["similarity"],
                    4
                ),

                "candidate_2": (
                    candidates[1]["medicine_name"]
                ),

                "candidate_2_score": round(
                    candidates[1]["similarity"],
                    4
                ),

                "candidate_3": (
                    candidates[2]["medicine_name"]
                ),

                "candidate_3_score": round(
                    candidates[2]["similarity"],
                    4
                ),

                "candidate_4": (
                    candidates[3]["medicine_name"]
                ),

                "candidate_4_score": round(
                    candidates[3]["similarity"],
                    4
                ),

                "candidate_5": (
                    candidates[4]["medicine_name"]
                ),

                "candidate_5_score": round(
                    candidates[4]["similarity"],
                    4
                ),
            }

            results.append(
                result
            )

    # ========================================================
    # OUTPUT CSV
    # ========================================================

    fieldnames = [

        # Original OCR information
        "region_id",
        "ocr_text",
        "ocr_confidence",

        # Region information
        "bbox_width",
        "bbox_height",

        # Candidate filter information
        "candidate_score",
        "filter_decision",
        "manual_label",
        "reasons",

        # Best candidate
        "best_medicine",
        "best_class_id",
        "best_similarity",
        "status",

        # Top 5 candidates
        "candidate_1",
        "candidate_1_score",

        "candidate_2",
        "candidate_2_score",

        "candidate_3",
        "candidate_3_score",

        "candidate_4",
        "candidate_4_score",

        "candidate_5",
        "candidate_5_score",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    strong_count = sum(
        result["status"]
        == "STRONG_CANDIDATE"
        for result in results
    )

    possible_count = sum(
        result["status"]
        == "POSSIBLE_CANDIDATE"
        for result in results
    )

    no_match_count = sum(
        result["status"]
        == "NO_STRONG_MATCH"
        for result in results
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "MATCHING SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"Total OCR regions : "
        f"{len(results)}"
    )

    print(
        f"Strong candidates : "
        f"{strong_count}"
    )

    print(
        f"Possible matches  : "
        f"{possible_count}"
    )

    print(
        f"No strong match   : "
        f"{no_match_count}"
    )

    # --------------------------------------------------------
    # Output file
    # --------------------------------------------------------

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_CSV
    )

    # ========================================================
    # TOP MATCHES
    # ========================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TOP MATCHES"
    )

    print(
        "=" * 75
    )

    for result in results:

        print(
            f"\nRegion {result['region_id']}: "
            f"{result['ocr_text']}"
        )

        print(
            f"  → "
            f"{result['best_medicine']} "
            f"("
            f"{result['best_similarity']:.3f}"
            f") "
            f"["
            f"{result['status']}"
            f"]"
        )

        print(
            f"     1. "
            f"{result['candidate_1']} "
            f"{result['candidate_1_score']:.3f}"
        )

        print(
            f"     2. "
            f"{result['candidate_2']} "
            f"{result['candidate_2_score']:.3f}"
        )

        print(
            f"     3. "
            f"{result['candidate_3']} "
            f"{result['candidate_3_score']:.3f}"
        )

        print(
            f"     4. "
            f"{result['candidate_4']} "
            f"{result['candidate_4_score']:.3f}"
        )

        print(
            f"     5. "
            f"{result['candidate_5']} "
            f"{result['candidate_5_score']:.3f}"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()