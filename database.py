"""Database helpers for storing MediScan predictions."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List, Optional

from sqlalchemy import DateTime, Float, Integer, String, create_engine, desc, func
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql://") and "+psycopg2" not in url:
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url


DATABASE_URL = _normalize_database_url(
    os.getenv("DATABASE_URL", "sqlite:///./mediscan.db")
)


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    organ: Mapped[str] = mapped_column(String(50), nullable=False)
    prediction: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)


engine_kwargs = {"future": True}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Keep hosted deployments responsive if a Postgres connection is slow or unavailable.
    engine_kwargs["connect_args"] = {"connect_timeout": 5}
    engine_kwargs["pool_pre_ping"] = True

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)


def init_db(retries: int = 5, delay_seconds: float = 2.0) -> None:
    for attempt in range(1, retries + 1):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except SQLAlchemyError as exc:
            if attempt == retries:
                print(f"Warning: database initialization failed after {retries} attempts: {exc}")
                return
            print(
                f"Warning: database not ready yet (attempt {attempt}/{retries}). "
                f"Retrying in {delay_seconds} seconds..."
            )
            time.sleep(delay_seconds)


@contextmanager
def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def log_prediction(organ: str, prediction: str, confidence: float) -> Optional[PredictionLog]:
    try:
        with get_session() as session:
            row = PredictionLog(
                organ=organ,
                prediction=prediction,
                confidence=confidence,
            )
            session.add(row)
            session.flush()
            session.refresh(row)
            return row
    except (SQLAlchemyError, OperationalError) as exc:
        print(f"Warning: failed to log prediction to database: {exc}")
        return None


def fetch_recent_predictions(limit: int = 50) -> List[PredictionLog]:
    try:
        with get_session() as session:
            return (
                session.query(PredictionLog)
                .order_by(desc(PredictionLog.timestamp))
                .limit(limit)
                .all()
            )
    except (SQLAlchemyError, OperationalError) as exc:
        print(f"Warning: failed to fetch recent predictions: {exc}")
        return []
