from pathlib import Path
import csv
import json
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "data/ocr_samples/images.jpeg"

OCR_JSON = PROJECT_ROOT / "ml/results/ocr/existing_ocr_result.json"
FINAL_CSV = PROJECT_ROOT / "ml/results/ocr_ml003/final_medicine_verification.csv"


def run_command(command, name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    start = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )

    elapsed = time.perf_counter() - start

    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print(f"Time: {elapsed:.2f} seconds")

    if result.returncode != 0:
        print(f"\n❌ FAILED: {name}")
        sys.exit(result.returncode)

    print(f"✅ PASSED: {name}")


def check_ocr_result():
    print("\n" + "=" * 70)
    print("CHECKING OCR OUTPUT")
    print("=" * 70)

    if not OCR_JSON.exists():
        print("❌ OCR JSON not found:")
        print(OCR_JSON)
        sys.exit(1)

    with open(OCR_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions = data.get("regions", [])

    print(f"OCR engine          : {data.get('engine')}")
    print(f"Detected regions    : {len(regions)}")
    print(f"Image               : {data.get('image')}")

    if len(regions) == 0:
        print("❌ No OCR regions detected.")
        sys.exit(1)

    print("✅ OCR output valid")


def check_final_result():
    print("\n" + "=" * 70)
    print("CHECKING FINAL MEDICINE VERIFICATION")
    print("=" * 70)

    if not FINAL_CSV.exists():
        print("❌ Final verification CSV not found:")
        print(FINAL_CSV)
        sys.exit(1)

    with open(FINAL_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("❌ Final CSV contains no results.")
        sys.exit(1)

    required_columns = {
        "region_id",
        "ocr_text",
        "ml_top1_medicine",
        "ml_top1_confidence",
        "decision",
    }

    actual_columns = set(rows[0].keys())

    missing = required_columns - actual_columns

    if missing:
        print("❌ Missing columns:")
        print(missing)
        sys.exit(1)

    decisions = {}

    for row in rows:
        decision = row["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1

    print(f"Total regions       : {len(rows)}")

    for decision, count in decisions.items():
        print(f"{decision:<20}: {count}")

    print("\nSample results:")

    for row in rows[:10]:
        print(
            f"Region {row['region_id']:>2} | "
            f"{row['ocr_text'][:30]:<30} | "
            f"{row['ml_top1_medicine']:<15} | "
            f"{row['ml_top1_confidence']}"
        )

    print("\n✅ Final verification output valid")


def main():

    print("=" * 70)
    print("AI PRESCRIPTION RECOGNITION - END-TO-END TEST")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input image  : {IMAGE_PATH}")

    if not IMAGE_PATH.exists():
        print("\n❌ Input prescription image not found.")
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 1: OCR
    # ---------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "experiments.test_ocr",
        ],
        "STEP 1 - PaddleOCR"
    )

    check_ocr_result()

    # ---------------------------------------------------------
    # STEP 2: Final Medicine Verification
    # ---------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "ml.medicine_verification_final",
        ],
        "STEP 2 - Final Medicine Verification"
    )

    check_final_result()

    # ---------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("END-TO-END TEST COMPLETE")
    print("=" * 70)

    print("✅ Prescription image loaded")
    print("✅ OCR executed")
    print("✅ OCR regions generated")
    print("✅ ML-003 medicine recognizer executed")
    print("✅ Medicine verification executed")
    print("✅ Final results generated")

    print("\nFinal output:")
    print(FINAL_CSV)


if __name__ == "__main__":
    main()