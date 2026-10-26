"""Prediction CLI -- the same model the API serves, printed as a table.

    python -m app.ml.prediction.cli events
    python -m app.ml.prediction.cli event 247
    python -m app.ml.prediction.cli grid "Mexico City" NOR:1,PIA:2,LEC:3,VER:4
    python -m app.ml.prediction.cli upcoming 2026 19

Every command reads the ACTIVE registered model, so output here and output from
`/api/v1/prediction/*` cannot disagree -- there is no second code path. Train one
first if nothing is registered:

    python -m app.ml.prediction.train

WHICH COMMAND IS HONEST ABOUT WHAT
----------------------------------
`event` predicts a race already ingested, using the as-of features the view
computed for it. Those numbers are IN-SAMPLE -- the active model was fitted on
every season including that race -- so the actual finishing position is printed
beside the prediction and the warning is not suppressed. It is for inspecting
reasoning, not for claiming accuracy.

`grid` and `upcoming` predict a race that has not happened, from a starting
order you supply or that is fetched from qualifying. As-of features come from
each driver's most recent COMPLETED race, which is the honest best available
before a race runs. Drivers with no history are reported, never imputed in
silence.
"""

from __future__ import annotations

import json as jsonlib
import logging
import sys

import typer
from sqlalchemy import text

from app.db.base import SessionLocal, engine
from app.ml.prediction import predict as race_predict

app = typer.Typer(
    add_completion=False,
    help="Print race predictions from the active model as a table.",
)
log = logging.getLogger("predict")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _print_table(result, *, show_actual: bool) -> None:
    """One row per driver, ranked. `*` marks a driver missing some features."""
    header = f"{'P':>3}  {'DRV':<5}{'TEAM':<16}{'GRID':>5}{'PODIUM':>8}"
    if show_actual:
        header += f"{'ACTUAL':>8}{'DELTA':>7}"
    typer.echo(header)
    typer.echo("-" * len(header))

    for d in result.drivers:
        grid = "-" if d.grid_position is None else f"{int(d.grid_position)}"
        row = (
            f"{d.predicted_rank:>3}  {d.driver:<5}{(d.team or '-')[:15]:<16}"
            f"{grid:>5}{d.podium_probability:>8.1%}"
        )
        if show_actual:
            if d.actual_position is None:
                row += f"{'-':>8}{'-':>7}"
            else:
                delta = d.predicted_rank - int(d.actual_position)
                row += f"{int(d.actual_position):>8}{delta:>+7}"
        if d.missing_features:
            row += "  *"
        typer.echo(row)

    incomplete = [d for d in result.drivers if d.missing_features]
    if incomplete:
        typer.echo(
            f"\n* {len(incomplete)} driver(s) have missing features. "
            "Their prediction leans on grid position:"
        )
        for d in incomplete:
            typer.echo(f"    {d.driver:<5} missing: {', '.join(d.missing_features)}")


def _print_header(result) -> None:
    where = result.circuit
    if result.season_year is not None:
        where = f"{result.season_year} R{result.round} {result.circuit}"
    typer.secho(f"\n{where}", bold=True)
    typer.echo(
        f"model {result.model_version} ({result.model_algorithm}), "
        f"{len(result.drivers)} drivers"
    )


def _print_warnings(result) -> None:
    if not result.warnings:
        return
    typer.echo()
    for w in result.warnings:
        typer.secho(f"! {w}", fg=typer.colors.YELLOW)


def _emit(result, *, as_json: bool, show_actual: bool) -> None:
    if as_json:
        typer.echo(
            jsonlib.dumps(
                {
                    "circuit": result.circuit,
                    "event_id": result.event_id,
                    "season_year": result.season_year,
                    "round": result.round,
                    "model_version": result.model_version,
                    "model_algorithm": result.model_algorithm,
                    "warnings": result.warnings,
                    "drivers": [
                        {
                            "predicted_rank": d.predicted_rank,
                            "driver": d.driver,
                            "team": d.team,
                            "grid_position": d.grid_position,
                            "predicted_position": d.predicted_position,
                            "podium_probability": d.podium_probability,
                            "actual_position": d.actual_position,
                            "missing_features": d.missing_features,
                        }
                        for d in result.drivers
                    ],
                },
                indent=2,
            )
        )
        return
    _print_header(result)
    typer.echo()
    _print_table(result, show_actual=show_actual)
    _print_warnings(result)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _no_model_hint(exc: Exception) -> None:
    _fail(
        f"{exc}\n\nNo usable model. Train one with:\n"
        "    python -m app.ml.prediction.train"
    )


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


@app.command()
def events(
    season: int = typer.Option(None, "--season", help="Limit to one season."),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List ingested races and their event_id, for use with `event`."""
    sql = """
        SELECT e.id, e.season_year, e.round, c.name, e.event_date,
               count(rf.driver) AS drivers
        FROM raw.events e
        JOIN raw.circuits c ON c.id = e.circuit_id
        LEFT JOIN features.race_features rf ON rf.event_id = e.id
        {where}
        GROUP BY e.id, e.season_year, e.round, c.name, e.event_date
        HAVING count(rf.driver) > 0
        ORDER BY e.event_date DESC
        LIMIT :limit
    """.format(where="WHERE e.season_year = :season" if season else "")
    params = {"limit": limit}
    if season:
        params["season"] = season

    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).all()

    if not rows:
        _fail(
            "No ingested races with features. Run:\n"
            "    python -m app.ingestion.cli season 2024 --only R,Q"
        )

    typer.echo(f"{'ID':>5}  {'SEASON':>6} {'RND':>4}  {'CIRCUIT':<20}{'DATE':<12}{'DRV':>4}")
    typer.echo("-" * 60)
    for event_id, year, rnd, circuit, date, drivers in rows:
        typer.echo(f"{event_id:>5}  {year:>6} {rnd:>4}  {circuit:<20}{str(date):<12}{drivers:>4}")


@app.command()
def event(
    event_id: int = typer.Argument(..., help="From `events`."),
    json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Predict a race already in the database, beside what actually happened.

    IN-SAMPLE. The active model was trained on every season including this race,
    so the agreement shown here is not an accuracy estimate. Use
    `python -m app.ml.prediction.train` output, or /prediction/backtest, for the
    walk-forward number.
    """
    try:
        with SessionLocal() as db:
            result = race_predict.predict_event(engine, event_id, db=db)
    except FileNotFoundError as exc:
        _no_model_hint(exc)
    except LookupError as exc:
        if "no active model" in str(exc):
            _no_model_hint(exc)
        _fail(f"{exc}\n\nList available races with: python -m app.ml.prediction.cli events")
    _emit(result, as_json=json, show_actual=True)


@app.command()
def grid(
    circuit: str = typer.Argument(..., help="Circuit name as stored, e.g. 'Mexico City'."),
    starting_grid: str = typer.Argument(
        ..., metavar="GRID", help="Comma-separated DRV:POS, e.g. NOR:1,PIA:2,LEC:3"
    ),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Predict a race from a starting grid you supply.

    As-of features come from each driver's most recent completed race, so this
    is off by one race -- their state going into their LAST race, not this one.
    That is the honest best available before a race has happened.
    """
    entries: dict[str, int] = {}
    for chunk in starting_grid.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            _fail(f"Bad grid entry {chunk!r}. Expected DRV:POS, e.g. VER:1")
        driver, _, position = chunk.partition(":")
        driver = driver.strip().upper()
        try:
            entries[driver] = int(position)
        except ValueError:
            _fail(f"Bad grid position in {chunk!r}: {position!r} is not an integer")

    if len(entries) < 2:
        _fail("Need at least two drivers in the grid.")

    _check_circuit(circuit)

    try:
        with SessionLocal() as db:
            result = race_predict.predict_from_grid(engine, circuit, entries, db=db)
    except (LookupError, FileNotFoundError) as exc:
        _no_model_hint(exc)
    _emit(result, as_json=json, show_actual=False)


@app.command()
def upcoming(
    year: int = typer.Argument(..., help="Season, e.g. 2026"),
    round_number: int = typer.Argument(..., metavar="ROUND", help="Round number"),
    circuit: str = typer.Option(
        None, "--circuit", help="Override the circuit name match."
    ),
    json: bool = typer.Option(False, "--json"),
) -> None:
    """Predict a race from its QUALIFYING order, fetched live.

    Qualifying order is used as the starting grid. They are not the same thing --
    penalties and pit-lane starts move cars -- but before the race it is the
    best public estimate of the grid, and it is what is knowable.

    Needs the results API to have qualifying for this round. If qualifying has
    not run yet, nothing can be fetched; supply a projected grid with the `grid`
    command instead.
    """
    try:
        from fastf1.ergast import Ergast

        from app.ingestion.fastf1_client import enable_cache
    except ImportError:  # pragma: no cover - fastf1 is a hard dependency
        _fail("fastf1 is not installed")

    # Reuse the project's cache rather than FastF1's default in ~/Library, so
    # repeated runs hit the same ~1 GB store the ingest already filled.
    enable_cache()
    ergast = Ergast()
    try:
        quali = ergast.get_qualifying_results(season=year, round=round_number)
        frame = quali.content[0] if quali.content else None
    except Exception as exc:  # noqa: BLE001 - the client raises several types
        _fail(f"Could not fetch qualifying for {year} round {round_number}: {exc}")

    if frame is None or frame.empty:
        _fail(
            f"No qualifying results for {year} round {round_number} yet.\n"
            "If qualifying has not run, supply a projected grid:\n"
            '    python -m app.ml.prediction.cli grid "<circuit>" DRV:1,DRV:2,...'
        )

    entries = {
        str(row.driverCode).upper(): int(row.position)
        for row in frame.itertuples(index=False)
    }

    if circuit is None:
        circuit = _circuit_for_round(year, round_number)
    _check_circuit(circuit)

    typer.echo(
        f"Qualifying order for {year} round {round_number} "
        f"({len(entries)} drivers) -> circuit {circuit!r}"
    )

    try:
        with SessionLocal() as db:
            result = race_predict.predict_from_grid(engine, circuit, entries, db=db)
    except (LookupError, FileNotFoundError) as exc:
        _no_model_hint(exc)

    result.warnings.insert(
        0,
        "Grid taken from qualifying order; penalties and pit-lane starts are not "
        "applied. As-of features come from each driver's most recent completed "
        "race in the database.",
    )
    _emit(result, as_json=json, show_actual=False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _known_circuits() -> list[str]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text("SELECT name FROM raw.circuits ORDER BY name"))]


def _check_circuit(circuit: str) -> None:
    """Warn -- not fail -- on an unknown circuit.

    An unknown circuit is legitimate: a new venue has no history, and the model
    degrades to missing circuit features rather than refusing. But a TYPO is
    indistinguishable from a new venue at this layer and would silently cost two
    features, so the known names are printed.
    """
    known = _known_circuits()
    if circuit in known:
        return
    typer.secho(
        f"! {circuit!r} is not an ingested circuit, so its circuit features will be "
        "missing.\n  If this is a typo, the known names are:",
        fg=typer.colors.YELLOW,
    )
    typer.echo("    " + ", ".join(known))


def _circuit_for_round(year: int, round_number: int) -> str:
    """Match the results API's `locality` to a stored circuit name.

    `locality` is used because that is what FastF1 reports as `Location`, which
    is what the ingest stores -- 'Mexico City', 'Austin', 'Monza'. Falling back
    to the raw locality keeps the failure visible: an unmatched name produces
    missing circuit features and a warning, not a wrong circuit's history.
    """
    from fastf1.ergast import Ergast

    from app.ingestion.fastf1_client import enable_cache

    try:
        enable_cache()
        schedule = Ergast().get_race_schedule(season=year)
        row = schedule[schedule["round"] == round_number]
        if row.empty:
            return ""
        locality = str(row.iloc[0]["locality"])
    except Exception:  # noqa: BLE001
        return ""

    known = _known_circuits()
    if locality in known:
        return locality
    for name in known:
        if name.lower() == locality.lower():
            return name
    return locality


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)-7s %(message)s",
        stream=sys.stdout,
    )
    for name in ("fastf1", "fastf1.req", "fastf1.core", "req", "transport"):
        logging.getLogger(name).setLevel(logging.ERROR)
    app()
