"""Read endpoints for the ingested catalogue: seasons, events, drivers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import joinedload

from app.db.deps import get_db
from app.db.models import Driver, Event, Season, Session
from app.schemas import DriverOut, EventOut, SeasonOut, SessionOut

router = APIRouter(tags=["catalog"])


@router.get("/seasons", response_model=list[SeasonOut])
def list_seasons(db: OrmSession = Depends(get_db)):
    """Every season that has ingested data, newest first.

    A season appears here only once at least one of its sessions has been
    loaded, so this is a catalogue of what the database actually holds --
    not the F1 history book.
    """
    return db.scalars(select(Season).order_by(Season.year.desc())).all()


@router.get("/seasons/{year}/events", response_model=list[EventOut])
def list_events(year: int, db: OrmSession = Depends(get_db)):
    """The calendar for one season, in round order.

    A 404 means nothing has been ingested for that year, not that the season
    does not exist. Run `python -m app.ingestion.cli season <year>` first.
    """
    events = db.scalars(
        select(Event)
        .options(joinedload(Event.circuit))
        .where(Event.season_year == year)
        .order_by(Event.round)
    ).all()
    if not events:
        raise HTTPException(404, f"No events ingested for {year}")
    return events


@router.get("/events/{event_id}/sessions", response_model=list[SessionOut])
def list_sessions(event_id: int, db: OrmSession = Depends(get_db)):
    """Every ingested session of one event, in chronological order.

    `session_type` is one of `FP1`/`FP2`/`FP3`, `Q`, `S` (sprint),
    `SQ` (sprint qualifying) or `R`. Which of them are present depends on
    what was ingested -- `cli.py season <year> --only R,Q` skips practice
    entirely, so a missing session type means "not loaded", not "not held".
    """
    sessions = db.scalars(
        select(Session).where(Session.event_id == event_id).order_by(Session.session_date)
    ).all()
    if not sessions:
        raise HTTPException(404, f"No sessions for event {event_id}")
    return sessions


@router.get("/drivers", response_model=list[DriverOut])
def list_drivers(db: OrmSession = Depends(get_db)):
    """Every driver seen in any ingested session, by three-letter code.

    `abbreviation` is unique across the table and is the key the rest of the
    API accepts -- the `driver` filter on the lap endpoints and the `driver`
    field of a prediction response both use it.
    """
    return db.scalars(select(Driver).order_by(Driver.abbreviation)).all()
