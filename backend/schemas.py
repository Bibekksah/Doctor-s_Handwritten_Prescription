from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    medicine: str
    generic_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class VerifyRequest(BaseModel):
    model_config = {
        "str_strip_whitespace": True,
    }

    medicine_name: str = Field(min_length=1)


class VerifyResponse(BaseModel):
    medicine: str
    exists: bool
    generic_name: str | None = None


class HealthResponse(BaseModel):
    status: str
    database: str
    model: str