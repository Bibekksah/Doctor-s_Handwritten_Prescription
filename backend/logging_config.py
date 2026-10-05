import logging
import sys


def configure_logging() -> None:
    """
    Configure application-wide logging.

    This function should be called once when the FastAPI
    application starts.
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        stream=sys.stdout,
        force=True,
    )