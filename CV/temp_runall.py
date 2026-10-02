import subprocess
import sys
from pathlib import Path

# Add project root to sys.path so subprocesses can resolve module paths cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "Doctor’s Handwritten Prescription BD dataset"
)

cv = [
   "CV.test_denoising",
   "CV.test_grayscale",
   "CV.test_thresholding",
   "CV.test_clahe",
   "CV.visualize_clahe",
   "CV.visualize_threshold",
   "CV.test_deskew",
   "CV.visualize_deskew",
   "CV.image_quality",
   "CV.visualize_image_quality"

]


def run_all_cv():
    for experiment in cv:
        print("\n" + "=" * 70)
        print(f"RUNNING: {experiment}")
        print("=" * 70)

        # Execute using current python interpreter and cwd set to PROJECT_ROOT
        result = subprocess.run(
            [sys.executable, "-m", experiment],
            cwd=PROJECT_ROOT
        )

        if result.returncode != 0:
            print(f"\n❌ FAILED: {experiment}")
            print("Stopping execution.")
            sys.exit(result.returncode)

        print(f"✅ COMPLETED: {experiment}")

    print("\n" + "=" * 70)
    print("ALL cv COMPLETED SUCCESSFULLY")
    print("=" * 70)


# Safe main guard for macOS multiprocessing (Note the DOUBLE underscores)
if __name__ == "__main__":
    run_all_cv()