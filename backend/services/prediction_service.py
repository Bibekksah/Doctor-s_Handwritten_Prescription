import logging
from pathlib import Path

from sqlalchemy.orm import Session

from backend.database.models import (
    ImageMetadata,
    Medicine,
    ModelVersion,
    Prediction,
)
from backend.inference import InferenceEngine, InferenceResult
from backend.utils import validate_image


logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(
        self,
        db: Session,
        inference_engine: InferenceEngine,
    ) -> None:
        self.db = db
        self.inference_engine = inference_engine

    def predict(
        self,
        image_bytes: bytes,
        content_type: str,
        image_path: Path,
        filename: str,
    ) -> tuple[InferenceResult, Medicine | None, Prediction]:
        """
        Validate image, persist image metadata, run inference,
        resolve the predicted medicine, and persist the prediction.
        """

        width, height = validate_image(
            file_bytes=image_bytes,
            content_type=content_type,
        )

        logger.info(
            "Image validation successful: filename=%s width=%s height=%s size=%s",
            filename,
            width,
            height,
            len(image_bytes),
        )

        image_metadata = ImageMetadata(
            filename=filename,
            content_type=content_type,
            file_size=len(image_bytes),
            width=width,
            height=height,
            storage_path=None,
        )

        self.db.add(image_metadata)
        self.db.flush()

        logger.info(
            "Image metadata persisted: image_id=%s",
            image_metadata.id,
        )

        result = self.inference_engine.predict(image_path)

        logger.info(
            "Inference completed: label=%s confidence=%.4f",
            result.label,
            result.confidence,
        )

        model_info = self.inference_engine.model_info

        model_version = (
            self.db.query(ModelVersion)
            .filter(ModelVersion.version == model_info.version)
            .first()
        )

        if model_version is None:
            model_version = ModelVersion(
                version=model_info.version,
                framework=model_info.framework,
                model_path=model_info.model_path,
            )

            self.db.add(model_version)
            self.db.flush()

            logger.info(
                "Model version registered: model_version_id=%s version=%s",
                model_version.id,
                model_version.version,
            )
        else:
            logger.info(
                "Model version resolved: model_version_id=%s version=%s",
                model_version.id,
                model_version.version,
            )


        medicine = (
            self.db.query(Medicine)
            .filter(Medicine.name == result.label)
            .first()
        )

        if medicine is not None:
            logger.info(
                "Medicine resolved: name=%s medicine_id=%s",
                medicine.name,
                medicine.id,
            )
        else:
            logger.warning(
                "Predicted label has no matching medicine: label=%s",
                result.label,
            )

        prediction = Prediction(
            predicted_label=result.label,
            confidence=result.confidence,
            medicine_id=medicine.id if medicine else None,
            image_id=image_metadata.id,
            model_version_id=model_version.id,
        )

        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)

        logger.info(
            "Prediction persisted: prediction_id=%s image_id=%s",
            prediction.id,
            image_metadata.id,
        )

        return result, medicine, prediction