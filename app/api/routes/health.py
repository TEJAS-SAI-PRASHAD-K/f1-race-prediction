"""Liveness endpoint that actually checks the database."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from app.db.deps import get_db
from app.db.models import Lap
from app.schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(db: OrmSession = Depends(get_db)):
    """A health check that does not touch the DB is close to useless, so this
    runs a real query."""
    try:
        laps = db.scalar(select(func.count()).select_from(Lap))
        return HealthOut(status="ok", database="connected", laps_ingested=laps)
    except Exception:  # noqa: BLE001
        return HealthOut(status="degraded", database="unavailable")
