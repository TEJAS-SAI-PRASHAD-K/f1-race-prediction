"""Idempotent loading of FastF1 data into Postgres.

Everything here is safe to re-run. Each table has a natural unique key and we
use INSERT ... ON CONFLICT DO UPDATE, so ingesting the same session twice
updates rows in place rather than duplicating them. That matters because
FastF1 backfills and corrects session data for a while after a race.

On bulk loading: laps are the only high-volume table (~150k rows per season).
We insert them with executemany in chunks rather than the ORM's per-object
flush. True COPY is faster still, but COPY cannot express ON CONFLICT, and
re-runnability is worth more here than raw speed -- a season loads in seconds
either way, and the network fetch dominates regardless.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session as OrmSession

from app.db.models import (
    Circuit,
    Driver,
    Event,
    Lap,
    PitStop,
    Season,
    Session,
    SessionResult,
    Team,
    Weather,
)
from app.ingestion import transform
from app.ingestion.fastf1_client import SessionData

log = logging.getLogger(__name__)

LAP_CHUNK_SIZE = 5_000


# --------------------------------------------------------------------------
# dimension upserts (small tables, return the id we need for foreign keys)
# --------------------------------------------------------------------------


def upsert_season(db: OrmSession, year: int) -> int:
    db.execute(pg_insert(Season).values(year=year).on_conflict_do_nothing())
    return year


def upsert_circuit(db: OrmSession, payload: dict[str, Any]) -> int:
    stmt = (
        pg_insert(Circuit)
        .values(**payload)
        .on_conflict_do_update(
            index_elements=["name"],
            set_={
                "country": payload.get("country"),
                "locality": payload.get("locality"),
            },
        )
        .returning(Circuit.id)
    )
    return db.execute(stmt).scalar_one()


def upsert_event(db: OrmSession, payload: dict[str, Any], circuit_id: int | None) -> int:
    values = {**payload, "circuit_id": circuit_id}
    stmt = (
        pg_insert(Event)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["season_year", "round"],
            set_={
                k: values[k]
                for k in ("event_name", "official_name", "event_date", "event_format", "circuit_id")
            },
        )
        .returning(Event.id)
    )
    return db.execute(stmt).scalar_one()


def upsert_session(
    db: OrmSession,
    event_id: int,
    session_type: str,
    session_date: Any,
    total_laps: int | None,
) -> int:
    values = {
        "event_id": event_id,
        "session_type": session_type,
        "session_date": session_date,
        "total_laps": total_laps,
    }
    stmt = (
        pg_insert(Session)
        .values(**values)
        .on_conflict_do_update(
            index_elements=["event_id", "session_type"],
            set_={"session_date": session_date, "total_laps": total_laps},
        )
        .returning(Session.id)
    )
    return db.execute(stmt).scalar_one()


def upsert_drivers(db: OrmSession, data: SessionData) -> dict[str, int]:
    """Upsert every driver seen in this session; return abbreviation -> id.

    Driver identity comes from `results` where available (it carries full name
    and country) and falls back to the abbreviations present in `laps`, since
    some older sessions have laps but no usable results.
    """
    payloads: dict[str, dict[str, Any]] = {}

    if data.results is not None and not data.results.empty:
        for res in data.results.itertuples(index=False):
            abbr = transform.to_str(getattr(res, "Abbreviation", None), 5)
            if not abbr:
                continue
            payloads[abbr] = {
                "abbreviation": abbr,
                "driver_number": transform.to_int(getattr(res, "DriverNumber", None)),
                "full_name": transform.to_str(getattr(res, "FullName", None), 128),
                "country_code": transform.to_str(getattr(res, "CountryCode", None), 8),
            }

    if data.laps is not None and not data.laps.empty:
        for abbr in data.laps["Driver"].dropna().unique():
            key = transform.to_str(abbr, 5)
            if key and key not in payloads:
                payloads[key] = {
                    "abbreviation": key,
                    "driver_number": None,
                    "full_name": None,
                    "country_code": None,
                }

    if not payloads:
        return {}

    for payload in payloads.values():
        # COALESCE so a later session with sparser data cannot blank out a
        # name or country we already know.
        stmt = pg_insert(Driver).values(**payload)
        db.execute(
            stmt.on_conflict_do_update(
                index_elements=["abbreviation"],
                set_={
                    "driver_number": func.coalesce(
                        stmt.excluded.driver_number, Driver.driver_number
                    ),
                    "full_name": func.coalesce(stmt.excluded.full_name, Driver.full_name),
                    "country_code": func.coalesce(
                        stmt.excluded.country_code, Driver.country_code
                    ),
                },
            )
        )

    rows = db.execute(
        select(Driver.abbreviation, Driver.id).where(Driver.abbreviation.in_(payloads))
    ).all()
    return {abbr: did for abbr, did in rows}


def upsert_teams(db: OrmSession, data: SessionData, year: int) -> dict[str, int]:
    """Upsert teams for this season; return team name -> id."""
    names: set[str] = set()
    for frame, col in ((data.results, "TeamName"), (data.laps, "Team")):
        if frame is not None and not frame.empty and col in frame.columns:
            for name in frame[col].dropna().unique():
                cleaned = transform.to_str(name, 128)
                if cleaned:
                    names.add(cleaned)

    if not names:
        return {}

    for name in names:
        db.execute(
            pg_insert(Team)
            .values(name=name, season_year=year)
            .on_conflict_do_nothing(index_elements=["name", "season_year"])
        )

    rows = db.execute(
        select(Team.name, Team.id).where(Team.season_year == year, Team.name.in_(names))
    ).all()
    return {name: tid for name, tid in rows}


# --------------------------------------------------------------------------
# fact tables
# --------------------------------------------------------------------------

_LAP_UPDATE_COLUMNS = (
    "team_id",
    "lap_time_ms",
    "sector1_ms",
    "sector2_ms",
    "sector3_ms",
    "compound",
    "tyre_life",
    "is_fresh_tyre",
    "stint_number",
    "pit_in_ms",
    "pit_out_ms",
    "track_status",
    "position",
    "speed_i1",
    "speed_i2",
    "speed_fl",
    "speed_st",
    "is_accurate",
    "is_deleted",
    "deleted_reason",
)


def load_laps(db: OrmSession, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    stmt = pg_insert(Lap)
    stmt = stmt.on_conflict_do_update(
        index_elements=["session_id", "driver_id", "lap_number"],
        set_={col: getattr(stmt.excluded, col) for col in _LAP_UPDATE_COLUMNS},
    )
    for start in range(0, len(rows), LAP_CHUNK_SIZE):
        db.execute(stmt, rows[start : start + LAP_CHUNK_SIZE])
    return len(rows)


def load_results(db: OrmSession, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    stmt = pg_insert(SessionResult)
    db.execute(
        stmt.on_conflict_do_update(
            index_elements=["session_id", "driver_id"],
            set_={
                col: getattr(stmt.excluded, col)
                for col in ("team_id", "position", "grid_position", "status", "points", "time_ms")
            },
        ),
        rows,
    )
    return len(rows)


def load_weather(db: OrmSession, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    stmt = pg_insert(Weather)
    db.execute(
        stmt.on_conflict_do_update(
            index_elements=["session_id", "time_offset_ms"],
            set_={
                col: getattr(stmt.excluded, col)
                for col in (
                    "air_temp",
                    "track_temp",
                    "humidity",
                    "pressure",
                    "rainfall",
                    "wind_speed",
                    "wind_direction",
                )
            },
        ),
        rows,
    )
    return len(rows)


def load_pit_stops(db: OrmSession, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    stmt = pg_insert(PitStop)
    db.execute(
        stmt.on_conflict_do_update(
            index_elements=["session_id", "driver_id", "stop_number"],
            set_={
                col: getattr(stmt.excluded, col)
                for col in ("lap_number", "duration_ms_est", "compound_in", "compound_out")
            },
        ),
        rows,
    )
    return len(rows)


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------


def ingest_session_data(db: OrmSession, data: SessionData) -> dict[str, int]:
    """Load one session end to end. Caller commits."""
    upsert_season(db, data.year)

    circuit_id = upsert_circuit(db, transform.transform_circuit(data.event))
    event_id = upsert_event(db, transform.transform_event(data.event, data.year), circuit_id)
    session_id = upsert_session(
        db,
        event_id=event_id,
        session_type=data.session_type,
        session_date=data.session_date,
        total_laps=data.total_laps,
    )

    driver_ids = upsert_drivers(db, data)
    team_ids = upsert_teams(db, data, data.year)

    counts = {
        "laps": load_laps(
            db, transform.transform_laps(data.laps, session_id, driver_ids, team_ids)
        ),
        "results": load_results(
            db, transform.transform_results(data.results, session_id, driver_ids, team_ids)
        ),
        "weather": load_weather(db, transform.transform_weather(data.weather, session_id)),
        "pit_stops": load_pit_stops(
            db, transform.derive_pit_stops(data.laps, session_id, driver_ids)
        ),
    }
    counts["session_id"] = session_id
    return counts
