import pytest
from io import BytesIO

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

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


#  7 tests for the /health endpoint

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

    assert data == {
        "medicine": "Aceta",
        "exists": True,
        "generic_name": "Paracetamol",
    }


@pytest.mark.anyio
async def test_verify_unknown_medicine(client):
    response = await client.post(
        "/verify",
        json={"medicine_name": "UnknownMedicine"},
    )

    assert response.status_code == 200

    data = response.json()

    assert data == {
        "medicine": "UnknownMedicine",
        "exists": False,
        "generic_name": None,
    }


@pytest.mark.anyio
async def test_verify_missing_medicine_name(client):
    response = await client.post(
        "/verify",
        json={},
    )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_verify_empty_medicine_name(client):
    response = await client.post(
        "/verify",
        json={"medicine_name": ""},
    )

    assert response.status_code == 422


@pytest.mark.anyio
async def test_verify_whitespace_medicine_name(client):
    response = await client.post(
        "/verify",
        json={"medicine_name": "   "},
    )

    assert response.status_code == 422


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

    assert data == {
        "medicine": "Aceta",
        "generic_name": "Paracetamol",
        "confidence": 0.94,
    }

# tests for error handling in the /predict endpoint

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

    assert prediction.predicted_label == "Aceta"
    assert prediction.confidence == 0.94

    assert prediction.medicine_id == 1
    assert prediction.image_id == image.id
    assert prediction.model_version_id == model_version.id

    assert image.filename == "test.png"
    assert image.content_type == "image/png"
    assert image.file_size == len(image_bytes)
    assert image.width == 256
    assert image.height == 256

    assert model_version.version == "mock-0.1.0"
    assert model_version.framework == "mock"