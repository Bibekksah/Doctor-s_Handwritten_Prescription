from ml.inference import MedicineRecognizer
from ml.medicine_verifier import MedicineVerifier
from ml.confidence import classify_confidence


class PrescriptionRecognitionService:

    def __init__(self):

        self.recognizer = MedicineRecognizer()
        self.verifier = MedicineVerifier()

    def process(self, image, top_k=3):

        predictions = self.recognizer.predict(
            image,
            top_k=top_k,
        )

        top_prediction = predictions[0]

        medicine_name = (
            top_prediction["medicine_name"]
        )

        confidence = (
            top_prediction["confidence"]
        )

        verification = self.verifier.verify(
            medicine_name
        )

        confidence_info = classify_confidence(
            confidence
        )

        return {
            "medicine_name": medicine_name,

            "generic_name": (
                verification["generic_name"]
            ),

            "confidence": confidence,

            "confidence_percent": (
                top_prediction[
                    "confidence_percent"
                ]
            ),

            "confidence_level": (
                confidence_info["level"]
            ),

            "requires_human_verification": (
                confidence_info[
                    "requires_verification"
                ]
            ),

            "verified": (
                verification["verified"]
            ),

            "top_predictions": predictions,
        }