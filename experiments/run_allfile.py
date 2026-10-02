import subprocess
import sys


experiments = [
    "experiments.dataset_analysis",
    "experiments.check_metadata",
    "experiments.inspect_images",
    "experiments.view_sample",
    "experiments.test_dataset",
    "experiments.view_preprocessed",
    "experiments.test_preprocessing",
    "experiments.test_dataloader",
]


for experiment in experiments:
    print("\n" + "=" * 70)
    print(f"RUNNING: {experiment}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, "-m", experiment]
    )

    if result.returncode != 0:
        print(f"\n❌ FAILED: {experiment}")
        print("Stopping execution.")
        sys.exit(result.returncode)

    print(f"✅ COMPLETED: {experiment}")


print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
print("=" * 70)