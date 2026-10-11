from backend.database.database import SessionLocal
from backend.database.models import Medicine


def main():
    db = SessionLocal()

    try:
        medicine = Medicine(
            name="Aceta",
            generic_name="Paracetamol",
        )

        db.add(medicine)
        db.commit()
        db.refresh(medicine)

        print(f"Created medicine:")
        print(f"  id={medicine.id}")
        print(f"  name={medicine.name}")
        print(f"  generic_name={medicine.generic_name}")

        result = (
            db.query(Medicine)
            .filter(Medicine.name == "Aceta")
            .first()
        )

        print("\nRetrieved medicine:")
        print(f"  id={result.id}")
        print(f"  name={result.name}")
        print(f"  generic_name={result.generic_name}")

    finally:
        db.close()


if __name__ == "__main__":
    main()