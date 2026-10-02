import subprocess
import sys
from pathlib import Path

# Add project root to sys.path so subprocesses can resolve module paths cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

experiments = [
    "experiments.dataset_analysis",
    "experiments.check_metadata",
    "experiments.inspect_images",
    "experiments.view_sample",
    "experiments.test_dataset",
    "experiments.view_preprocessed",
    "experiments.test_preprocessing",
    "experiments.test_dataloader",
    "experiments.test_grayscale"


]


def run_all_experiments():
    for experiment in experiments:
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
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 70)


# Safe main guard for macOS multiprocessing (Note the DOUBLE underscores)
if __name__ == "__main__":
    run_all_experiments()