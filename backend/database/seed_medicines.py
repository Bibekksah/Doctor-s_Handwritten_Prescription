
import pandas as pd
from sqlalchemy import select

from backend.database.database import Base, engine, SessionLocal
from backend.database.models import Medicine

CSV_PATH = "data/metadata/dataset.csv"


def seed_medicines():
    df = pd.read_csv(CSV_PATH)

    required_columns = {"medicine_name", "generic_name"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}"
        )

    # Keep one record per medicine and generic-name mapping.
    mappings = df[["medicine_name", "generic_name"]].drop_duplicates()
    mappings = mappings.dropna(subset=["medicine_name"])

    # Check that one medicine doesn't map to multiple generics.
    conflicts = mappings.groupby("medicine_name")["generic_name"].nunique()
    conflicts = conflicts[conflicts > 1]

    if not conflicts.empty:
        raise ValueError(
            f"Conflicting generic names found: {conflicts.to_dict()}"
        )

    inserted = 0
    updated = 0

    with SessionLocal() as db:
        for row in mappings.to_dict(orient="records"):
            name = str(row["medicine_name"]).strip()
            if not name:
                continue

            generic = row["generic_name"]
            generic = str(generic).strip() if pd.notna(generic) else None
            if generic == "":
                generic = None

            medicine = db.scalar(
                select(Medicine).where(Medicine.name == name)
            )

            if medicine is None:
                db.add(Medicine(name=name, generic_name=generic))
                inserted += 1
            elif medicine.generic_name != generic:
                medicine.generic_name = generic
                updated += 1

        db.commit()

    print(f"New medicines inserted: {inserted}")
    print(f"Existing medicines updated: {updated}")

    with SessionLocal() as db:
        print(f"Total medicines in database: {db.query(Medicine).count()}")

        aceta = db.query(Medicine).filter(
            Medicine.name == "Aceta"
        ).first()

        if aceta:
            print(f"Aceta generic name: {aceta.generic_name}")
        else:
            print("WARNING: Aceta was not found after seeding.")


if __name__ == "__main__":
    seed_medicines()