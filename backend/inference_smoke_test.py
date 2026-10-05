from pathlib import Path

from backend.inference import MockInferenceEngine


def main() -> None:
    engine = MockInferenceEngine()

    result = engine.predict(
        Path("dummy_image.png")
    )

    print("Inference engine: OK")
    print(f"Predicted label: {result.label}")
    print(f"Confidence: {result.confidence}")


if __name__ == "__main__":
    main()