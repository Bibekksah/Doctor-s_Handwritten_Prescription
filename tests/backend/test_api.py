import io

import pytest
from PIL import Image

from backend.database.models import (
    ImageMetadata,
    ModelVersion,
    Prediction,
)


def make_test_image() -> bytes:
    image = Image.new(
        "RGB",
        (256, 256),
        color="white",
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


@pytest.mark.anyio
async def test_health(client):
    response = await client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["database"] == "ok"
    assert data["model"] == "ok"


@pytest.mark.anyio
async def test_verify_existing_medicine(client):
    response = await client.post(
        "/verify",
        json={"medicine_name": "Aceta"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["medicine"] == "Aceta"
    assert data["exists"] is True
    assert data["generic_name"] == "Paracetamol"


@pytest.mark.anyio
async def test_verify_unknown_medicine(client):
    response = await client.post(
        "/verify",
        json={"medicine_name": "UnknownMedicine"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["medicine"] == "UnknownMedicine"
    assert data["exists"] is False
    assert data["generic_name"] is None


@pytest.mark.anyio
async def test_verify_strips_whitespace(client):
    response = await client.post(
        "/verify",
        json={"medicine_name": "  Aceta  "},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["medicine"] == "Aceta"
    assert data["exists"] is True
    assert data["generic_name"] == "Paracetamol"


@pytest.mark.anyio
async def test_predict_valid_image(client):
    image_bytes = make_test_image()

    response = await client.post(
        "/predict",
        files={
            "file": (
                "test.png",
                image_bytes,
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert set(data.keys()) == {
        "medicine",
        "generic_name",
        "confidence",
    }

    assert isinstance(data["medicine"], str)
    assert data["medicine"] != ""

    assert data["generic_name"] is None or isinstance(
        data["generic_name"],
        str,
    )

    assert isinstance(data["confidence"], float)
    assert 0.0 <= data["confidence"] <= 1.0


# Tests for error handling in the /predict endpoint.


@pytest.mark.anyio
async def test_predict_invalid_content_type(client):
    response = await client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


@pytest.mark.anyio
async def test_predict_invalid_image_bytes(client):
    response = await client.post(
        "/predict",
        files={
            "file": (
                "broken.png",
                b"not a real png image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


@pytest.mark.anyio
async def test_predict_persists_prediction(
    client,
    test_session,
):
    image_bytes = make_test_image()

    response = await client.post(
        "/predict",
        files={
            "file": (
                "test.png",
                image_bytes,
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    prediction = (
        test_session.query(Prediction)
        .one()
    )

    image = (
        test_session.query(ImageMetadata)
        .one()
    )

    model_version = (
        test_session.query(ModelVersion)
        .one()
    )

    assert prediction.predicted_label
    assert 0.0 <= prediction.confidence <= 1.0

    # The synthetic test image is not guaranteed to correspond
    # to a medicine in the isolated test database.
    assert prediction.medicine_id is None

    assert prediction.image_id == image.id
    assert prediction.model_version_id == model_version.id

    assert image.filename == "test.png"
    assert image.content_type == "image/png"
    assert image.file_size == len(image_bytes)
    assert image.width == 256
    assert image.height == 256

    assert model_version.version == "ML003-resnet18-onnx-1.0.0"
    assert model_version.framework == "onnxruntime"


@pytest.mark.anyio
async def test_predict_empty_file(client):
    response = await client.post(
        "/predict",
        files={
            "file": (
                "empty.png",
                b"",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


@pytest.mark.anyio
async def test_predict_unsupported_image_type(client):
    response = await client.post(
        "/predict",
        files={
            "file": (
                "test.gif",
                b"fake gif content",
                "image/gif",
            )
        },
    )

    assert response.status_code == 400


@pytest.mark.anyio
async def test_predict_invalid_dimensions(client):
    image = Image.new(
        "RGB",
        (16, 16),
        color="white",
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    response = await client.post(
        "/predict",
        files={
            "file": (
                "small.png",
                buffer.getvalue(),
                "image/png",
            )
        },
    )

    assert response.status_code == 400

