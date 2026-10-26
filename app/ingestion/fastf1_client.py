"""Thin wrapper around FastF1.

Two things this module exists to enforce:

1. `telemetry=False` on every load. Telemetry is per-car data at 4-10 Hz and is
   enormous; none of this project's four features need it. Loading without it
   is the difference between a ~3s and a ~40s session load.

2. A single place where FastF1's display session names ("Practice 1",
   "Sprint Qualifying") are mapped to our short codes (FP1, SQ).
"""

from __future__ import annotations

import datetime as dt
import logging
import warnings
from dataclasses import dataclass

import fastf1
import pandas as pd

from app.config import get_settings

log = logging.getLogger(__name__)

# FastF1 emits a lot of noise about missing/partial data on older sessions.
warnings.filterwarnings("ignore", module="fastf1")

# FastF1 schedule session names -> our session_type codes.
# "Sprint Shootout" was the 2023 name; "Sprint Qualifying" is the 2024+ name.
SESSION_NAME_TO_TYPE: dict[str, str] = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Sprint": "S",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SQ",
    "Race": "R",
}


def _as_utc(value) -> dt.datetime | None:
    """Attach UTC to FastF1's naive session timestamp.

    `session.date` is naive but ALREADY in UTC -- verified against the
    schedule: China 2024 gives Timestamp('2024-04-21 07:00:00') with
    tzinfo=None, while the schedule reports Session5Date 15:00+08:00 and
    Session5DateUtc 07:00.

    This conversion is not cosmetic. Handing a naive datetime to a TIMESTAMPTZ
    column makes Postgres interpret it in the server's local timezone, so the
    stored instant is wrong by the local UTC offset -- silently, and by a
    different amount depending on where the code runs. The feature view orders races by
    time for walk-forward backtesting, so a shifted timestamp is a correctness
    problem, not a display one.
    """
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(dt.UTC)
    else:
        stamp = stamp.tz_convert(dt.UTC)
    return stamp.to_pydatetime()


@dataclass
class SessionData:
    """Everything we ingest from one session, as raw DataFrames."""

    year: int
    round_number: int
    session_type: str
    session_name: str
    session_date: pd.Timestamp | None
    total_laps: int | None
    event: pd.Series
    laps: pd.DataFrame
    results: pd.DataFrame
    weather: pd.DataFrame


_cache_enabled = False


def enable_cache() -> None:
    """Enable FastF1's HTTP cache. Idempotent."""
    global _cache_enabled
    if not _cache_enabled:
        fastf1.Cache.enable_cache(str(get_settings().fastf1_cache_path))
        _cache_enabled = True


def get_event_schedule(year: int) -> pd.DataFrame:
    """Race weekends for a season, excluding pre-season testing."""
    enable_cache()
    return fastf1.get_event_schedule(year, include_testing=False)


def session_types_for_event(event: pd.Series) -> list[str]:
    """Which of our session codes exist for this weekend.

    The schedule exposes Session1..Session5 as display names; a sprint weekend
    has a different set than a conventional one, so we read rather than assume.
    """
    types: list[str] = []
    for i in range(1, 6):
        name = event.get(f"Session{i}")
        if not name or pd.isna(name):
            continue
        code = SESSION_NAME_TO_TYPE.get(str(name).strip())
        if code:
            types.append(code)
        else:
            log.warning("Unmapped session name %r -- skipping", name)
    return types


def load_session(year: int, round_number: int, session_type: str) -> SessionData:
    """Load one session's laps, results and weather.

    Raises whatever FastF1 raises when a session has no data (common for very
    old or cancelled sessions); callers are expected to handle that.
    """
    enable_cache()
    session = fastf1.get_session(year, round_number, session_type)
    session.load(telemetry=False, weather=True, messages=False)

    return SessionData(
        year=year,
        round_number=round_number,
        session_type=session_type,
        session_name=session.name,
        session_date=_as_utc(session.date),
        total_laps=getattr(session, "total_laps", None),
        event=session.event,
        laps=session.laps,
        results=session.results,
        weather=session.weather_data
        if session.weather_data is not None
        else pd.DataFrame(),
    )
