from sqlalchemy import text
from fastapi import FastAPI

from backend.database.database import engine
from backend.logging_config import configure_logging
from backend.routes import router
from backend.schemas import HealthResponse
from backend.exception_handlers import global_exception_handler


configure_logging()


app = FastAPI(
    title="Doctor's Handwritten Prescription API",
    version="0.1.0",
)

app.add_exception_handler(
    Exception,
    global_exception_handler,
)


def check_database() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return True

    except Exception:
        return False


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    database_ok = check_database()

    return HealthResponse(
        status="ok" if database_ok else "degraded",
        database="ok" if database_ok else "error",
        model="ok",
    )


app.include_router(router)