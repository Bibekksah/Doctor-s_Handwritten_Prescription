import logging

from fastapi import Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Handle unexpected application exceptions.

    The complete exception is logged server-side while the
    client receives a safe, generic response.
    """

    logger.exception(
        "Unhandled exception: method=%s path=%s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred.",
        },
    )
