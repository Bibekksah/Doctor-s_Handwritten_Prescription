from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from sqlalchemy import text

from backend.database.database import engine
from backend.exception_handlers import global_exception_handler
from backend.logging_config import configure_logging
from backend.ml003_inference import ML003InferenceEngine
from backend.routes import router
from backend.schemas import HealthResponse

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model once when the application starts.
    app.state.inference_engine = ML003InferenceEngine()

    yield

    # Release the model/session when the application shuts down.
    app.state.inference_engine = None


app = FastAPI(
    title="Doctor's Handwritten Prescription API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_exception_handler(Exception, global_exception_handler)


def check_database() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    database_ok = check_database()

    inference_engine = getattr(request.app.state, "inference_engine", None)

    model_ok = False

    if inference_engine is not None:
        try:
            model_info = inference_engine.model_info
            model_ok = bool(
                model_info.version
                and model_info.framework
                and model_info.model_path
            )
        except Exception:
            model_ok = False

    return HealthResponse(
        status="ok" if database_ok and model_ok else "degraded",
        database="ok" if database_ok else "error",
        model="ok" if model_ok else "error",
    )


app.include_router(router)