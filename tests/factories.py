"""Synthetic seasons, so the test suite needs no ingested data.

WHY THIS EXISTS
---------------
`features.race_features` is built from rolling windows over past races, so it
cannot be exercised by a fixture that seeds one race. The windows need history:
a driver's form over 3 and 5 previous races, a team's form, a circuit's mean
position change, and `walk_forward_splits` needs more than `min_train_races`
races before it yields a single fold.

The alternative -- which this replaces -- was for the prediction tests to read
the developer's real database. That makes `pytest` pass or fail depending on
what happens to be ingested locally, which is no test at all for anyone who
clones the repo.

WHAT IS INJECTED, AND WHY IT MATTERS
------------------------------------
The generator knows the truth it planted:

  * each driver has a fixed skill, so finishing order is predictable from grid
    position plus skill -- which means "Ridge beats the grid baseline" becomes a
    statement about RECOVERING a known signal rather than a hope about real data;
  * each circuit has a fixed degradation slope, so the per-stint OLS fit has
    something real to find, and `circuit_degradation_s_per_lap` can be checked
    against the planted value rather than merely asserted non-null;
  * every team runs two cars, which is what makes the `GROUPS` vs `ROWS` leakage
    test meaningful -- with one car per team, "1 row preceding" and "1 group
    preceding" are the same frame and the bug is invisible.

Everything is seeded from `default_rng(SEED)`, so the data is identical on every
machine and a failure is reproducible.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from sqlalchemy import insert
from sqlalchemy.engine import Engine

from app.db.models import (
    Circuit,
    Driver,
    Event,
    Lap,
    Season,
    Session,
    SessionResult,
    Team,
)

SEED = 42

N_SEASONS = 4
N_ROUNDS = 22         # 88 races: above walk-forward's min_train_races=60
N_TEAMS = 10
CARS_PER_TEAM = 2      # 20 drivers; two cars per team is load-bearing, see above
LAPS_PER_STINT = 10
STINTS_PER_RACE = 2
RACE_LAPS = LAPS_PER_STINT * STINTS_PER_RACE

FIRST_SEASON = 2019
BASE_LAP_MS = 84_000
COMPOUNDS = ("MEDIUM", "HARD")

# Degradation slopes planted per circuit, in seconds per lap of tyre life.
# The range brackets what real circuits show (Monaco ~0.01, Sakhir ~0.11).
SLOPE_MIN_S, SLOPE_MAX_S = 0.02, 0.12


def _abbr(i: int) -> str:
    """D00..D19 -- deliberately not real driver codes, so nothing reads as real."""
    return f"D{i:02d}"


def seed_synthetic_seasons(engine: Engine) -> dict:
    """Write complete synthetic seasons and COMMIT them.

    Returns the planted ground truth, so tests can assert recovery rather than
    merely assert non-emptiness.
    """
    rng = np.random.default_rng(SEED)
    n_drivers = N_TEAMS * CARS_PER_TEAM

    driver_skill = rng.permutation(np.linspace(-4.0, 4.0, n_drivers))
    circuit_slope = dict(
        zip(
            [f"Circuit {c:02d}" for c in range(N_ROUNDS)],
            rng.uniform(SLOPE_MIN_S, SLOPE_MAX_S, N_ROUNDS),
            strict=True,
        )
    )

    with engine.begin() as conn:
        conn.execute(
            insert(Season),
            [{"year": FIRST_SEASON + s} for s in range(N_SEASONS)],
        )
        circuit_ids = [
            conn.execute(
                insert(Circuit).returning(Circuit.id),
                [{"name": name, "country": "Testland"}],
            ).scalar_one()
            for name in circuit_slope
        ]
        driver_ids = [
            conn.execute(
                insert(Driver).returning(Driver.id),
                [{"abbreviation": _abbr(i), "driver_number": i + 1,
                  "full_name": f"Driver {i:02d}"}],
            ).scalar_one()
            for i in range(n_drivers)
        ]
        # Teams are per-season in the schema, so one row per (team, season).
        team_ids: dict[tuple[int, int], int] = {}
        for s in range(N_SEASONS):
            for t in range(N_TEAMS):
                team_ids[(s, t)] = conn.execute(
                    insert(Team).returning(Team.id),
                    [{"name": f"Team {t:02d}", "season_year": FIRST_SEASON + s}],
                ).scalar_one()

        events, sessions, results, laps = [], [], [], []

        for s in range(N_SEASONS):
            season_year = FIRST_SEASON + s
            for r in range(N_ROUNDS):
                circuit_name = f"Circuit {r:02d}"
                slope_s = circuit_slope[circuit_name]
                event_date = dt.date(season_year, 3, 1) + dt.timedelta(days=14 * r)

                event_id = conn.execute(
                    insert(Event).returning(Event.id),
                    [{"season_year": season_year, "round": r + 1,
                      "event_name": f"{circuit_name} Grand Prix",
                      "circuit_id": circuit_ids[r], "event_date": event_date,
                      "event_format": "conventional"}],
                ).scalar_one()
                events.append(event_id)

                q_id, race_id = (
                    conn.execute(
                        insert(Session).returning(Session.id),
                        [{"event_id": event_id, "session_type": st,
                          "session_date": dt.datetime(
                              event_date.year, event_date.month, event_date.day,
                              hour, 0, tzinfo=dt.UTC),
                          "total_laps": RACE_LAPS if st == "R" else None}],
                    ).scalar_one()
                    for st, hour in (("Q", 13), ("R", 14))
                )
                sessions += [q_id, race_id]

                # --- qualifying: a noisy read on skill; grid follows from it ---
                q_pace = driver_skill + rng.normal(0, 1.6, n_drivers)
                grid_order = np.argsort(q_pace)           # fastest first
                grid_position = np.empty(n_drivers, dtype=int)
                grid_position[grid_order] = np.arange(1, n_drivers + 1)

                for i in range(n_drivers):
                    laps.append({
                        "session_id": q_id, "driver_id": driver_ids[i],
                        "team_id": team_ids[(s, i // CARS_PER_TEAM)],
                        "lap_number": 1,
                        "lap_time_ms": int(BASE_LAP_MS + q_pace[i] * 150),
                        "compound": "SOFT", "tyre_life": 2.0, "is_fresh_tyre": True,
                        "stint_number": 1, "track_status": [1],
                        "position": float(grid_position[i]), "is_accurate": True,
                    })

                # --- race: grid is the strongest term, but skill still shows
                # through, which is the signal the form features can recover ---
                race_pace = (
                    grid_position * 0.7
                    + driver_skill * 0.8
                    + rng.normal(0, 1.2, n_drivers)
                )
                retired = rng.random(n_drivers) < 0.08
                # Retirements are classified at the back, which is what the real
                # data does -- and why finish_position conflates "last" and "DNF".
                race_pace = np.where(retired, race_pace + 100, race_pace)
                finish_order = np.argsort(race_pace)
                finish_position = np.empty(n_drivers, dtype=int)
                finish_position[finish_order] = np.arange(1, n_drivers + 1)

                points_table = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0,
                                6: 8.0, 7: 6.0, 8: 4.0, 9: 2.0, 10: 1.0}

                for i in range(n_drivers):
                    results.append({
                        "session_id": race_id, "driver_id": driver_ids[i],
                        "team_id": team_ids[(s, i // CARS_PER_TEAM)],
                        "position": float(finish_position[i]),
                        "grid_position": float(grid_position[i]),
                        "status": "Retired" if retired[i] else "Finished",
                        "points": 0.0 if retired[i]
                                  else points_table.get(int(finish_position[i]), 0.0),
                        "time_ms": None if retired[i] else 90 * 60 * 1000,
                    })

                    # --- race laps: the degradation signal lives here ---
                    for stint in range(1, STINTS_PER_RACE + 1):
                        for age in range(1, LAPS_PER_STINT + 1):
                            lap_number = (stint - 1) * LAPS_PER_STINT + age
                            fuel_ms = -(lap_number * 30)       # car lightens
                            deg_ms = slope_s * age * 1000.0
                            laps.append({
                                "session_id": race_id, "driver_id": driver_ids[i],
                                "team_id": team_ids[(s, i // CARS_PER_TEAM)],
                                "lap_number": lap_number,
                                "lap_time_ms": int(
                                    BASE_LAP_MS + driver_skill[i] * 50 + fuel_ms
                                    + deg_ms + rng.normal(0, 60)
                                ),
                                "compound": COMPOUNDS[stint - 1],
                                "tyre_life": float(age), "is_fresh_tyre": True,
                                "stint_number": stint, "track_status": [1],
                                "position": float(finish_position[i]),
                                "is_accurate": True,
                            })

        conn.execute(insert(SessionResult), results)
        # One bulk insert: ~25k rows one at a time would dominate the suite.
        conn.execute(insert(Lap), laps)

    return {
        "n_events": len(events),
        "n_drivers": n_drivers,
        "driver_skill": {_abbr(i): float(driver_skill[i]) for i in range(n_drivers)},
        "circuit_slope_s": {k: float(v) for k, v in circuit_slope.items()},
        "first_season": FIRST_SEASON,
        "last_season": FIRST_SEASON + N_SEASONS - 1,
    }
