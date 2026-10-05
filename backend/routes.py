from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import Medicine
from backend.inference import MockInferenceEngine
from backend.schemas import PredictResponse, VerifyRequest, VerifyResponse
from backend.services.prediction_service import PredictionService
from backend.utils import ImageValidationError


router = APIRouter()

# Predict the medicine from the uploaded image and return its name, generic name, and confidence score.

@router.post(
    "/predict",
    response_model=PredictResponse,
)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> PredictResponse:

    # Read uploaded file into memory.
    image_bytes = await file.read()

    # The inference interface currently expects a path.
    # We create a temporary file for the current mock implementation.
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
            inference_engine=MockInferenceEngine(),
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


#  Verify if a medicine exists in the database and return its generic name if it does.

@router.post(
    "/verify",
    response_model=VerifyResponse,
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