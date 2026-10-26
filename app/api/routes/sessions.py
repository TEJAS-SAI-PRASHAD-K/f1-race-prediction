"""Per-session data: laps, results, stints, pit stops."""

from __future__ import annotations

import statistics

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from app.db.deps import get_db
from app.db.models import Driver, Lap, PitStop, Session, SessionResult, Team
from app.schemas import LapOut, PitStopOut, ResultOut, StintOut

router = APIRouter(prefix="/sessions", tags=["sessions"])

GREEN = [1]


def _require_session(db: OrmSession, session_id: int) -> Session:
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(404, f"Session {session_id} not found")
    return session


@router.get("/{session_id}/laps", response_model=list[LapOut])
def get_laps(
    session_id: int,
    driver: str | None = Query(None, description="Driver abbreviation, e.g. VER"),
    compound: str | None = Query(None, description="SOFT/MEDIUM/HARD/INTERMEDIATE/WET"),
    accurate_only: bool = Query(False, description="FastF1's own clean-lap flag"),
    green_only: bool = Query(
        False, description="Only strictly clear laps (track_status == [1])"
    ),
    limit: int = Query(2000, le=10000),
    db: OrmSession = Depends(get_db),
):
    """Laps for one session, optionally filtered.

    `accurate_only` and `green_only` are two different quality filters, and
    neither implies the other. `accurate_only` is FastF1's own flag, which
    drops in-laps, out-laps and timing dropouts. `green_only` keeps a lap
    only if its `track_status` is exactly `[1]`, so a lap that was partly
    under a yellow is excluded -- drivers lift for yellows, which makes the
    time unrepresentative of pace.

    **Both together are still not enough for pace modeling.** A standing
    restart after a red flag passes both filters: Mexico 2023 lap 36 records
    148.5s against an ~83s norm, because it includes time stationary on the
    grid. Anything fitting a curve to these lap times needs a pace-based
    outlier filter of its own; `cli.py doctor` reports the count.

    `lap_time_ms` is NULL where FastF1 had no time for the lap. NULL is not
    zero, and is deliberately kept distinct from it.
    """
    _require_session(db, session_id)

    stmt = (
        select(Lap, Driver.abbreviation, Team.name)
        .join(Driver, Driver.id == Lap.driver_id)
        .outerjoin(Team, Team.id == Lap.team_id)
        .where(Lap.session_id == session_id)
    )
    if driver:
        stmt = stmt.where(Driver.abbreviation == driver.upper())
    if compound:
        stmt = stmt.where(Lap.compound == compound.upper())
    if accurate_only:
        stmt = stmt.where(Lap.is_accurate.is_(True))
    if green_only:
        stmt = stmt.where(Lap.track_status == GREEN)

    rows = db.execute(stmt.order_by(Driver.abbreviation, Lap.lap_number).limit(limit)).all()

    return [
        LapOut(
            **{
                col: getattr(lap, col)
                for col in (
                    "lap_number",
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
                    "is_accurate",
                )
            },
            driver=abbr,
            team=team,
        )
        for lap, abbr, team in rows
    ]


@router.get("/{session_id}/results", response_model=list[ResultOut])
def get_results(session_id: int, db: OrmSession = Depends(get_db)):
    """Classified finishing order for one session.

    `position` is the *classified* position, which every starter receives.
    A retirement still carries one, so position alone does not tell you who
    saw the flag -- read `status` for that.

    `time_ms` is present for `Finished` and `Lapped`, and NULL for `+N Lap`
    and for every retirement status. A NULL time therefore does not mean the
    driver failed to finish; check `status` rather than inferring it.

    For qualifying sessions `status` is always NULL, where it has no
    meaning. `points` covers races only: sprints are not ingested, so on a
    sprint weekend these totals fall short of the official championship
    numbers.
    """
    _require_session(db, session_id)
    rows = db.execute(
        select(SessionResult, Driver, Team.name)
        .join(Driver, Driver.id == SessionResult.driver_id)
        .outerjoin(Team, Team.id == SessionResult.team_id)
        .where(SessionResult.session_id == session_id)
        .order_by(SessionResult.position.nulls_last())
    ).all()
    return [
        ResultOut(
            driver=drv.abbreviation,
            full_name=drv.full_name,
            team=team,
            position=res.position,
            grid_position=res.grid_position,
            status=res.status,
            points=res.points,
            time_ms=res.time_ms,
        )
        for res, drv, team in rows
    ]


@router.get("/{session_id}/stints", response_model=list[StintOut])
def get_stints(session_id: int, db: OrmSession = Depends(get_db)):
    """Aggregate laps into stints -- the unit tire degradation is measured over.

    `median_green_lap_time_ms` uses only strictly-clear accurate laps. Median
    rather than mean because a single traffic-compromised lap skews a mean
    badly, and a stint has few enough laps for that to matter.
    """
    _require_session(db, session_id)

    rows = db.execute(
        select(Lap, Driver.abbreviation)
        .join(Driver, Driver.id == Lap.driver_id)
        .where(Lap.session_id == session_id, Lap.stint_number.isnot(None))
        .order_by(Driver.abbreviation, Lap.stint_number, Lap.lap_number)
    ).all()

    grouped: dict[tuple[str, int], list[Lap]] = {}
    for lap, abbr in rows:
        grouped.setdefault((abbr, lap.stint_number), []).append(lap)

    stints: list[StintOut] = []
    for (abbr, stint_number), laps in grouped.items():
        green_times = [
            lap.lap_time_ms
            for lap in laps
            if lap.lap_time_ms is not None
            and lap.is_accurate
            and lap.track_status == GREEN
        ]
        tyre_lives = [float(lap.tyre_life) for lap in laps if lap.tyre_life is not None]
        stints.append(
            StintOut(
                driver=abbr,
                stint_number=stint_number,
                compound=laps[0].compound,
                lap_start=laps[0].lap_number,
                lap_end=laps[-1].lap_number,
                laps_run=len(laps),
                tyre_life_start=min(tyre_lives) if tyre_lives else None,
                tyre_life_end=max(tyre_lives) if tyre_lives else None,
                is_fresh_tyre=laps[0].is_fresh_tyre,
                median_green_lap_time_ms=(
                    int(statistics.median(green_times)) if green_times else None
                ),
                green_laps=len(green_times),
            )
        )
    return sorted(stints, key=lambda s: (s.driver, s.stint_number))


@router.get("/{session_id}/pit-stops", response_model=list[PitStopOut])
def get_pit_stops(session_id: int, db: OrmSession = Depends(get_db)):
    """Derived pit stops.

    `duration_ms_est` is NULL where the pit-entry-to-exit delta was
    implausible -- a red-flag suspension or a car stopping in the pit lane
    rather than a pit stop. Treat NULL as "not a measurable stop".
    """
    _require_session(db, session_id)
    rows = db.execute(
        select(PitStop, Driver.abbreviation)
        .join(Driver, Driver.id == PitStop.driver_id)
        .where(PitStop.session_id == session_id)
        .order_by(Driver.abbreviation, PitStop.stop_number)
    ).all()
    return [
        PitStopOut(
            driver=abbr,
            stop_number=ps.stop_number,
            lap_number=ps.lap_number,
            duration_ms_est=ps.duration_ms_est,
            compound_in=ps.compound_in,
            compound_out=ps.compound_out,
        )
        for ps, abbr in rows
    ]
