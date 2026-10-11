from sqlalchemy import text

from .database import engine


def main() -> None:
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            value = result.scalar_one()

        print("Database connection: OK")
        print(f"SELECT 1 result: {value}")
        print(f"Database URL: {engine.url}")

    except Exception as exc:
        print("Database connection: FAILED")
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()