from pathlib import Path

import pytest

from backend.database.models import ImageMetadata
from backend.services.prediction_service import PredictionService


class FailingInferenceEngine:
    @property
    def model_info(self):
        raise RuntimeError("model failure")

    def predict(self, image_path: Path):
        raise RuntimeError("inference failure")


def test_prediction_rolls_back_when_inference_fails(test_session):
    service = PredictionService(
        db=test_session,
        inference_engine=FailingInferenceEngine(),
    )

    image_bytes = b"test-image-bytes"

    # The image validator is deliberately bypassed here.
    # This test focuses only on database transaction rollback.
    def fake_validate_image(file_bytes, content_type):
        return 100, 100

    import backend.services.prediction_service as prediction_service_module

    original_validate_image = prediction_service_module.validate_image
    prediction_service_module.validate_image = fake_validate_image

    try:
        with pytest.raises(RuntimeError, match="inference failure"):
            service.predict(
                image_bytes=image_bytes,
                content_type="image/png",
                image_path=Path("test.png"),
                filename="test.png",
            )
    finally:
        prediction_service_module.validate_image = original_validate_image

    image_count = (
        test_session.query(ImageMetadata)
        .count()
    )

    assert image_count == 0