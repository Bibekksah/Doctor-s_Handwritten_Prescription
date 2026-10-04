from pathlib import Path

from ml.inference import MedicineRecognizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "Doctor’s Handwritten Prescription BD dataset"
    / "Testing"
    / "testing_words"
    / "0.png"
)


def main():

    print("=" * 70)
    print("MEMBER 3 — ML INFERENCE TEST")
    print("=" * 70)

    print(f"Image: {TEST_IMAGE}")

    recognizer = MedicineRecognizer()

    predictions = recognizer.predict(
        TEST_IMAGE,
        top_k=3,
    )

    print()
    print("Top-3 Predictions")
    print("-" * 50)

    for rank, prediction in enumerate(
        predictions,
        start=1,
    ):

        print(
            f"{rank}. "
            f"{prediction['medicine_name']} "
            f"— "
            f"{prediction['confidence_percent']:.2f}%"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()