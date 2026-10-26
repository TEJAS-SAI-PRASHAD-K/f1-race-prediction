"""Ingestion CLI.

    python -m app.ingestion.cli session 2024 Monza R
    python -m app.ingestion.cli season 2024
    python -m app.ingestion.cli season 2024 --only R,Q
    python -m app.ingestion.cli status

Every command is safe to re-run: the loader upserts on natural keys.
"""

from __future__ import annotations

import logging
import sys
import time

import typer
from sqlalchemy import func, select, text

from app.db.base import SessionLocal
from app.db.models import Driver, Event, Lap, PitStop, Season, Session, Weather
from app.ingestion import fastf1_client
from app.ingestion.loader import ingest_session_data

app = typer.Typer(add_completion=False, help="Ingest F1 data from FastF1 into Postgres.")
log = logging.getLogger("ingest")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stdout,
    )
    # FastF1 is extremely chatty at INFO.
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    for name in ("fastf1.core", "fastf1.api", "fastf1._api", "fastf1.req"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _ingest_one(year: int, round_number: int, session_type: str, label: str) -> bool:
    """Returns True on success. Missing-data errors are logged, not raised."""
    started = time.perf_counter()
    try:
        data = fastf1_client.load_session(year, round_number, session_type)
    except Exception as exc:  # noqa: BLE001 - FastF1 raises many error types
        log.warning("  %-26s SKIP  (%s)", label, str(exc)[:90])
        return False

    if data.laps is None or data.laps.empty:
        log.warning("  %-26s SKIP  (no lap data)", label)
        return False

    with SessionLocal() as db:
        counts = ingest_session_data(db, data)
        db.commit()

    log.info(
        "  %-26s OK    laps=%-5d results=%-3d weather=%-4d pits=%-3d  (%.1fs)",
        label,
        counts["laps"],
        counts["results"],
        counts["weather"],
        counts["pit_stops"],
        time.perf_counter() - started,
    )
    return True


@app.command()
def session(
    year: int = typer.Argument(..., help="Season, e.g. 2024"),
    event: str = typer.Argument(..., help="Round number or event name, e.g. 14 or Monza"),
    session_type: str = typer.Argument(..., help="FP1/FP2/FP3/Q/SQ/S/R"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ingest a single session."""
    _setup_logging(verbose)
    schedule = fastf1_client.get_event_schedule(year)

    if event.isdigit():
        round_number = int(event)
    else:
        matches = schedule[
            schedule["EventName"].str.contains(event, case=False, na=False)
            | schedule["Location"].str.contains(event, case=False, na=False)
            | schedule["Country"].str.contains(event, case=False, na=False)
        ]
        if matches.empty:
            typer.secho(f"No {year} event matching {event!r}", fg="red")
            raise typer.Exit(1)
        round_number = int(matches.iloc[0]["RoundNumber"])

    name = schedule[schedule.RoundNumber == round_number].iloc[0]["EventName"]
    log.info("%d R%02d %s", year, round_number, name)
    ok = _ingest_one(year, round_number, session_type.upper(), session_type.upper())
    raise typer.Exit(0 if ok else 1)


@app.command()
def season(
    year: int = typer.Argument(..., help="Season, e.g. 2024"),
    only: str = typer.Option(
        None, "--only", help="Comma-separated session types, e.g. R,Q. Default: all."
    ),
    from_round: int = typer.Option(1, "--from-round"),
    to_round: int = typer.Option(99, "--to-round"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ingest a whole season.

    First run is network-bound and slow; FastF1's cache makes re-runs fast.
    Use --only R to get just the races when iterating on modeling.
    """
    _setup_logging(verbose)
    wanted = {s.strip().upper() for s in only.split(",")} if only else None

    schedule = fastf1_client.get_event_schedule(year)
    schedule = schedule[
        (schedule.RoundNumber >= from_round) & (schedule.RoundNumber <= to_round)
    ]

    started = time.perf_counter()
    ok = failed = 0
    for event in schedule.itertuples(index=False):
        round_number = int(event.RoundNumber)
        event_row = schedule[schedule.RoundNumber == round_number].iloc[0]
        types = fastf1_client.session_types_for_event(event_row)
        if wanted:
            types = [t for t in types if t in wanted]
        if not types:
            continue

        log.info("%d R%02d %s", year, round_number, event.EventName)
        for session_type in types:
            if _ingest_one(year, round_number, session_type, session_type):
                ok += 1
            else:
                failed += 1

    log.info(
        "Done: %d sessions ingested, %d skipped, %.0fs total",
        ok,
        failed,
        time.perf_counter() - started,
    )


@app.command()
def status() -> None:
    """Show what is currently in the database."""
    _setup_logging(False)
    with SessionLocal() as db:
        seasons = db.scalars(select(Season.year).order_by(Season.year)).all()
        counts = {
            "events": db.scalar(select(func.count()).select_from(Event)),
            "sessions": db.scalar(select(func.count()).select_from(Session)),
            "drivers": db.scalar(select(func.count()).select_from(Driver)),
            "laps": db.scalar(select(func.count()).select_from(Lap)),
            "weather": db.scalar(select(func.count()).select_from(Weather)),
            "pit_stops": db.scalar(select(func.count()).select_from(PitStop)),
        }
        accurate = db.scalar(
            select(func.count()).select_from(Lap).where(Lap.is_accurate.is_(True))
        )

    typer.echo(f"seasons   : {', '.join(map(str, seasons)) or '(none)'}")
    for key, value in counts.items():
        typer.echo(f"{key:10}: {value:,}")
    if counts["laps"]:
        typer.echo(f"{'accurate':10}: {accurate:,} ({accurate / counts['laps']:.0%} of laps)")


@app.command()
def doctor() -> None:
    """Run data-quality checks against the ingested data.

    These are not unit tests -- they interrogate the real database for the
    kinds of corruption that would pass silently into a model. Run it after
    every backfill.
    """
    _setup_logging(False)

    checks: list[tuple[str, str, str]] = [
        (
            # Checks the CONSEQUENCE, not the symptom. NULL track_status does
            # occur -- 23 laps from 2020 races, where FastF1 supplied no status
            # at all -- and that is harmless, because every one of them is also
            # is_accurate = false and so is excluded anyway.
            #
            # The first version of this check asserted zero NULLs outright and
            # started failing when the 2019-2022 backfill landed. It was
            # conflating "unusual" with "harmful". What actually matters is
            # whether an unknown-status lap can reach a model, so that is what
            # is asserted now.
            "unknown-status laps reaching the modeling filter",
            """SELECT count(*) FROM raw.laps
               WHERE track_status IS NULL
                 AND is_accurate
                 AND lap_time_ms IS NOT NULL""",
            "must be zero -- a lap we cannot vouch for must not reach a model",
        ),
        (
            "orphaned laps",
            """SELECT count(*) FROM raw.laps l
               LEFT JOIN raw.sessions s ON s.id = l.session_id
               WHERE s.id IS NULL""",
            "must be zero -- foreign keys should make this impossible",
        ),
        (
            "duplicate natural keys",
            """SELECT count(*) FROM (
                 SELECT 1 FROM raw.laps
                 GROUP BY session_id, driver_id, lap_number HAVING count(*) > 1
               ) d""",
            "must be zero -- proves ingestion is idempotent",
        ),
        (
            "implausible lap times LEAKING into models",
            """SELECT count(*) FROM raw.laps
               WHERE lap_time_ms IS NOT NULL
                 AND (lap_time_ms < 50000 OR lap_time_ms > 300000)
                 AND is_accurate
                 AND track_status = ARRAY[1]::smallint[]""",
            "must be zero -- red-flag laps must not reach the models",
        ),
        (
            "pit stops with implausible kept duration",
            """SELECT count(*) FROM raw.pit_stops
               WHERE duration_ms_est IS NOT NULL
                 AND (duration_ms_est < 8000 OR duration_ms_est > 180000)""",
            "must be zero -- the clamp in derive_pit_stops should hold",
        ),
        (
            "invalid compounds",
            """SELECT count(*) FROM raw.laps
               WHERE compound IS NOT NULL AND compound NOT IN
                 ('SOFT','MEDIUM','HARD','INTERMEDIATE','WET','TEST_UNKNOWN','UNKNOWN')""",
            "must be zero -- the CHECK constraint should prevent it",
        ),
    ]

    with SessionLocal() as db:
        failures = 0
        for label, sql, note in checks:
            count = db.execute(text(sql)).scalar() or 0
            ok = count == 0
            mark = "PASS" if ok else "FAIL"
            color = typer.colors.GREEN if ok else typer.colors.RED
            typer.secho(f"[{mark}] {label}: {count:,}", fg=color)
            if not ok:
                typer.echo(f"       {note}")
                failures += 1

        # --- advisory, not pass/fail ---
        # These laps are CORRECT in the raw data. The point is that neither
        # is_accurate nor the green-flag filter excludes them, so the view
        # needs a third, pace-based filter of its own.
        outliers = db.execute(
            text(
                """
                WITH clean AS (
                  SELECT l.session_id, l.lap_time_ms
                  FROM raw.laps l
                  JOIN raw.sessions s ON s.id = l.session_id AND s.session_type = 'R'
                  WHERE l.is_accurate
                    AND l.track_status = ARRAY[1]::smallint[]
                    AND l.lap_time_ms IS NOT NULL
                ),
                medians AS (
                  SELECT session_id,
                         percentile_cont(0.5) WITHIN GROUP (ORDER BY lap_time_ms)
                           AS session_median
                  FROM clean GROUP BY session_id
                )
                SELECT count(*)
                FROM clean c JOIN medians m USING (session_id)
                WHERE c.lap_time_ms > m.session_median * 1.5
                """
            )
        ).scalar() or 0

        if outliers:
            typer.echo()
            typer.secho(
                f"[WARN] {outliers:,} race laps are >150% of their session median "
                "yet pass BOTH is_accurate and the green-flag filter",
                fg=typer.colors.YELLOW,
            )
            typer.echo(
                "       These are real recorded times -- mostly standing-restart laps\n"
                "       after a red flag, which include time spent stationary on the\n"
                "       grid while track status already reads clear. The view adds\n"
                "       a pace-based outlier filter; the two existing filters do not\n"
                "       catch these. See docs/data-foundation.md."
            )

        # Informational, not pass/fail.
        typer.echo()
        unknown_status = db.execute(
            text("SELECT count(*) FROM raw.laps WHERE track_status IS NULL")
        ).scalar() or 0
        if unknown_status:
            typer.echo(
                f"note: {unknown_status:,} laps have no track_status at all "
                "(FastF1 supplied none). All are is_accurate = false, so they are "
                "excluded from modeling regardless."
            )

        rows = db.execute(
            text(
                """SELECT count(*) AS total,
                          count(*) FILTER (WHERE is_accurate) AS accurate,
                          count(*) FILTER (WHERE is_accurate
                                 AND track_status = ARRAY[1]::smallint[]
                                 AND lap_time_ms IS NOT NULL) AS clean_green
                   FROM raw.laps"""
            )
        ).one()
        typer.echo(f"laps total        : {rows.total:,}")
        typer.echo(f"accurate          : {rows.accurate:,}")
        typer.echo(f"clean + green     : {rows.clean_green:,}  <- usable for modeling")

    if failures:
        typer.secho(f"\n{failures} check(s) failed", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("\nAll data-quality checks passed", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
