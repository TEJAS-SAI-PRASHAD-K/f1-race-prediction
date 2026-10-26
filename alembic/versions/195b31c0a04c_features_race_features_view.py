"""features.race_features view

One row per (race, driver) with the outcome and every feature that was knowable
BEFORE the race started.

THE LEAKAGE GUARD
-----------------
Every rolling feature uses a window frame ending at `1 PRECEDING`:

    AVG(finish_position) OVER (
        PARTITION BY driver_id ORDER BY event_date
        GROUPS BETWEEN 3 PRECEDING AND 1 PRECEDING
    )

That `1 PRECEDING` is the entire leakage defence, and it is declarative: the
current race physically cannot enter its own feature. The pandas equivalent is a
`shift(1).rolling(3)` chain that is easy to get off by one and impossible to
notice when you do -- the model just gets mysteriously good.

WHY `GROUPS` AND NOT `ROWS` -- a real bug this prevents
------------------------------------------------------
`ROWS` counts physical rows. `GROUPS` counts distinct ORDER BY values, treating
rows that tie as one peer group.

For a driver-level window that distinction is invisible: a driver has exactly one
row per race, so 3 rows and 3 races are the same thing.

For a TEAM-level or CIRCUIT-level window it is the difference between correct and
leaking. A team has two cars in every race and a circuit has twenty, so with
`ROWS ... AND 1 PRECEDING` the "previous row" is that driver's own TEAMMATE IN
THE SAME RACE -- a result from the race being predicted. `GROUPS` makes all rows
sharing an `event_date` one group, so `1 PRECEDING` means the previous RACE.

This is the subtlest leak in the project. It produces no error, no warning, and a
model that looks better than it is.

WHAT IS DELIBERATELY ABSENT
---------------------------
Nothing measured during the race: no fastest lap, no pit count, no final gap, no
lap times. `grid_position` and the qualifying gap are included because both are
settled before the race starts.

Expected tire degradation (a genuinely useful feature) is NOT here, because it
is fitted per stint in Python -- see `app.ml.prediction.dataset`. The view
stays pure SQL over `raw`.

Revision ID: 195b31c0a04c
Revises: 819c4ea913e1
"""

from collections.abc import Sequence

from alembic import op

revision: str = "195b31c0a04c"
down_revision: str | None = "819c4ea913e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

VIEW = """
CREATE OR REPLACE VIEW features.race_features AS
WITH results AS (
    SELECT
        e.id                AS event_id,
        e.season_year,
        e.round,
        e.event_date,
        c.name              AS circuit,
        d.id                AS driver_id,
        d.abbreviation      AS driver,
        t.name              AS team,
        sr.position::float  AS finish_position,
        sr.grid_position::float,
        sr.status,
        sr.points::float,
        -- A retirement is still CLASSIFIED at the back by FastF1, so
        -- finish_position is populated for almost every row. `classified`
        -- separates "reached the end" from "was given a position anyway",
        -- which is what a DNF-rate feature needs.
        (sr.status = 'Finished' OR sr.status LIKE '%Lap%') AS classified
    FROM raw.session_results sr
    JOIN raw.sessions s ON s.id = sr.session_id AND s.session_type = 'R'
    JOIN raw.events   e ON e.id = s.event_id
    JOIN raw.drivers  d ON d.id = sr.driver_id
    LEFT JOIN raw.teams    t ON t.id = sr.team_id
    LEFT JOIN raw.circuits c ON c.id = e.circuit_id
    WHERE sr.position IS NOT NULL
      AND sr.grid_position IS NOT NULL
),
quali AS (
    -- Best accurate qualifying lap per driver, and the gap to that weekend's
    -- pole. Derived from Q-session LAPS rather than results, because the
    -- schema stores session results without the Q1/Q2/Q3 split.
    SELECT
        q.event_id,
        q.driver_id,
        q.best_ms,
        (q.best_ms - p.pole_ms) / 1000.0 AS quali_gap_to_pole_s,
        rank() OVER (PARTITION BY q.event_id ORDER BY q.best_ms) AS quali_position
    FROM (
        SELECT s.event_id, l.driver_id, min(l.lap_time_ms) AS best_ms
        FROM raw.laps l
        JOIN raw.sessions s ON s.id = l.session_id AND s.session_type = 'Q'
        WHERE l.lap_time_ms IS NOT NULL AND l.is_accurate
        GROUP BY s.event_id, l.driver_id
    ) q
    JOIN (
        SELECT s.event_id, min(l.lap_time_ms) AS pole_ms
        FROM raw.laps l
        JOIN raw.sessions s ON s.id = l.session_id AND s.session_type = 'Q'
        WHERE l.lap_time_ms IS NOT NULL AND l.is_accurate
        GROUP BY s.event_id
    ) p ON p.event_id = q.event_id
),
as_of AS (
    SELECT
        r.*,

        -- ---- DRIVER form. One row per driver per race, so GROUPS and ROWS
        -- ---- would agree here; GROUPS is used for consistency.
        avg(r.finish_position) OVER (
            PARTITION BY r.driver_id ORDER BY r.event_date
            GROUPS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS driver_form_3,
        avg(r.finish_position) OVER (
            PARTITION BY r.driver_id ORDER BY r.event_date
            GROUPS BETWEEN 5 PRECEDING AND 1 PRECEDING
        ) AS driver_form_5,
        avg((NOT r.classified)::int::float) OVER (
            PARTITION BY r.driver_id ORDER BY r.event_date
            GROUPS BETWEEN 9 PRECEDING AND 1 PRECEDING
        ) AS driver_dnf_rate_10,
        count(*) OVER (
            PARTITION BY r.driver_id ORDER BY r.event_date
            GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS driver_races_to_date,

        -- ---- TEAM form. GROUPS IS ESSENTIAL: a team has two cars per race, so
        -- ---- `ROWS ... AND 1 PRECEDING` would reach the driver's own teammate
        -- ---- in the race being predicted.
        avg(r.finish_position) OVER (
            PARTITION BY r.team ORDER BY r.event_date
            GROUPS BETWEEN 3 PRECEDING AND 1 PRECEDING
        ) AS team_form_3,
        avg(r.points) OVER (
            PARTITION BY r.team, r.season_year ORDER BY r.event_date
            GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS team_points_per_entry_to_date,
        avg(r.points) OVER (
            PARTITION BY r.driver_id, r.season_year ORDER BY r.event_date
            GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS driver_points_per_race_to_date,

        -- ---- This driver at THIS circuit, previous visits only.
        avg(r.finish_position) OVER (
            PARTITION BY r.driver_id, r.circuit ORDER BY r.event_date
            GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS driver_circuit_mean_finish,

        -- ---- CIRCUIT overtaking difficulty: the mean absolute grid-to-finish
        -- ---- change at this circuit over PREVIOUS races. A low value means
        -- ---- the grid order survives, i.e. overtaking is hard.
        -- ---- GROUPS IS ESSENTIAL here too -- twenty rows share each event.
        avg(abs(r.grid_position - r.finish_position)) OVER (
            PARTITION BY r.circuit ORDER BY r.event_date
            GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS circuit_mean_abs_position_change
    FROM results r
)
SELECT
    a.event_id,
    a.season_year,
    a.round,
    a.event_date,
    a.circuit,
    a.driver_id,
    a.driver,
    a.team,

    -- ---- pre-race, known ----
    a.grid_position,
    q.quali_gap_to_pole_s,
    q.quali_position::float,

    -- ---- as-of rolling features ----
    a.driver_form_3,
    a.driver_form_5,
    a.driver_dnf_rate_10,
    a.driver_races_to_date::float,
    a.team_form_3,
    a.team_points_per_entry_to_date,
    a.driver_points_per_race_to_date,
    a.driver_circuit_mean_finish,
    a.circuit_mean_abs_position_change,

    -- ---- targets ----
    a.finish_position,
    (a.finish_position <= 3)  AS podium,
    (a.finish_position <= 10) AS points_finish,
    a.classified,
    a.status
FROM as_of a
LEFT JOIN quali q ON q.event_id = a.event_id AND q.driver_id = a.driver_id
"""


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS features")
    op.execute(VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS features.race_features")
