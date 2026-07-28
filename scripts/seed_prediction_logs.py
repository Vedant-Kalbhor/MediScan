"""Seed sample MediScan prediction rows into PostgreSQL or SQLite.

Usage:
    python scripts/seed_prediction_logs.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from database import PredictionLog, SessionLocal, init_db


SAMPLE_ROWS = [
    {"organ": "brain", "prediction": "glioma", "confidence": 0.94},
    {"organ": "brain", "prediction": "no tumor", "confidence": 0.98},
    {"organ": "chest", "prediction": "normal", "confidence": 0.91},
    {"organ": "breast", "prediction": "malignant", "confidence": 0.88},
    {"organ": "kidney", "prediction": "stone", "confidence": 0.86},
    {"organ": "bone", "prediction": "fracture", "confidence": 0.97},
]


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        existing = session.query(PredictionLog).count()
        if existing > 0:
            print(f"Prediction log table already has {existing} rows. Nothing to seed.")
            return

        now = datetime.now(timezone.utc)
        for index, row in enumerate(SAMPLE_ROWS):
            session.add(
                PredictionLog(
                    timestamp=now - timedelta(minutes=index * 5),
                    organ=row["organ"],
                    prediction=row["prediction"],
                    confidence=row["confidence"],
                )
            )
        session.commit()
        print(f"Seeded {len(SAMPLE_ROWS)} sample prediction rows.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
