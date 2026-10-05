from pathlib import Path
from io import BytesIO

from PIL import Image

from backend.database.database import SessionLocal
from backend.database.models import Medicine
from backend.inference import MockInferenceEngine
from backend.services.prediction_service import PredictionService


def create_test_image() -> bytes:
    image = Image.new(
        "RGB",
        (256, 256),
        color="white",
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def main() -> None:
    db = SessionLocal()

    try:
        # Ensure the medicine expected by MockInferenceEngine exists.
        medicine = (
            db.query(Medicine)
            .filter(Medicine.name == "Aceta")
            .first()
        )

        if medicine is None:
            medicine = Medicine(
                name="Aceta",
                generic_name="Paracetamol",
            )

            db.add(medicine)
            db.commit()
            db.refresh(medicine)

        service = PredictionService(
            db=db,
            inference_engine=MockInferenceEngine(),
        )

        image_bytes = create_test_image()

        result, resolved_medicine, prediction = service.predict(
            image_bytes=image_bytes,
            content_type="image/png",
            image_path=Path("dummy.png"),
            filename="smoke_test.png",
        )

        print("Prediction service: OK")
        print(f"Predicted label: {result.label}")
        print(f"Confidence: {result.confidence}")
        print(
            f"Resolved medicine: "
            f"{resolved_medicine.name if resolved_medicine else None}"
        )
        print(
            f"Generic name: "
            f"{resolved_medicine.generic_name if resolved_medicine else None}"
        )
        print(f"Prediction ID: {prediction.id}")

    finally:
        db.close()


if __name__ == "__main__":
    main()