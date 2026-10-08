from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backend.database.database import SessionLocal
from backend.database.models import Medicine


DATASET_PATH = Path("data/metadata/dataset.csv")


def seed_medicines() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"medicine_name", "generic_name"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    medicines = (
        df[["medicine_name", "generic_name"]]
        .dropna(subset=["medicine_name"])
        .assign(
            medicine_name=lambda x: x["medicine_name"].astype(str).str.strip(),
            generic_name=lambda x: x["generic_name"].astype(str).str.strip(),
        )
        .drop_duplicates()
    )

    grouped = medicines.groupby("medicine_name")["generic_name"].agg(
        lambda values: sorted(set(values))
    )

    conflicts = grouped[grouped.apply(len) > 1]

    if not conflicts.empty:
        raise ValueError(
            "Conflicting generic names found:\n"
            + conflicts.to_string()
        )

    db = SessionLocal()

    try:
        now = datetime.now(timezone.utc)

        inserted = 0
        updated = 0

        for medicine_name, generic_names in grouped.items():
            generic_name = generic_names[0] if generic_names else None

            existing = (
                db.query(Medicine)
                .filter(Medicine.name == medicine_name)
                .first()
            )

            if existing is None:
                db.add(
                    Medicine(
                        name=medicine_name,
                        generic_name=generic_name,
                        created_at=now,
                        updated_at=now,
                    )
                )
                inserted += 1
            elif existing.generic_name != generic_name:
                existing.generic_name = generic_name
                existing.updated_at = now
                updated += 1

        db.commit()

        print(f"Medicines in dataset: {len(grouped)}")
        print(f"Inserted: {inserted}")
        print(f"Updated: {updated}")

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


if __name__ == "__main__":
    seed_medicines()