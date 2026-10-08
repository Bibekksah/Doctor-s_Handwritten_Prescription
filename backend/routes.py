from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import Medicine
from backend.schemas import (
    PredictResponse,
    VerifyRequest,
    VerifyResponse,
)
from backend.services.prediction_service import PredictionService
from backend.utils import ImageValidationError

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict medicine from a handwritten prescription",
    description=(
        "Upload a doctor's handwritten prescription image and "
        "predict the medicine name using the ML003 model.\n\n"
        "Accepted image formats: JPEG and PNG.\n"
        "Maximum upload size: 10 MB.\n\n"
        "The returned confidence is the model's prediction confidence "
        "and should not be interpreted as medical certainty."
    ),
    responses={
        400: {
            "description": "Invalid or unsupported image.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Uploaded file is not a valid image."
                    }
                }
            },
        },
        422: {
            "description": "Request validation error.",
        },
        500: {
            "description": "Unexpected server error.",
            "content": {
                "application/json": {
                    "example": {
                        "error": "internal_server_error",
                        "message": "An unexpected error occurred.",
                    }
                }
            },
        },
    },
)
async def predict(
    request: Request,
    file: UploadFile = File(
        ...,
        description=(
            "Prescription image in JPEG or PNG format. "
            "Maximum size: 10 MB."
        ),
    ),
    db: Session = Depends(get_db),
) -> PredictResponse:
    image_bytes = await file.read()

    temp_path = None

    try:
        with NamedTemporaryFile(
            suffix=Path(file.filename or "").suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)

        service = PredictionService(
            db=db,
            inference_engine=request.app.state.inference_engine,
        )

        result, medicine, _ = service.predict(
            image_bytes=image_bytes,
            content_type=file.content_type or "",
            image_path=temp_path,
            filename=file.filename or "unknown",
        )

        return PredictResponse(
            medicine=result.label,
            generic_name=(
                medicine.generic_name
                if medicine is not None
                else None
            ),
            confidence=result.confidence,
        )

    except ImageValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@router.post(
    "/verify",
    response_model=VerifyResponse,
    summary="Verify a medicine name",
    description=(
        "Check whether a predicted medicine name exists in the "
        "medicine database and return its generic name when available."
    ),
)
def verify_medicine(
    request: VerifyRequest,
    db: Session = Depends(get_db),
) -> VerifyResponse:
    medicine = (
        db.query(Medicine)
        .filter(Medicine.name == request.medicine_name)
        .first()
    )

    if medicine is None:
        return VerifyResponse(
            medicine=request.medicine_name,
            exists=False,
            generic_name=None,
        )

    return VerifyResponse(
        medicine=medicine.name,
        exists=True,
        generic_name=medicine.generic_name,
    )