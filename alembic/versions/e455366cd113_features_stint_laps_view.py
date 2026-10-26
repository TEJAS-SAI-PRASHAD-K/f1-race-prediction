"""features.stint_laps view

The clean per-stint lap dataset, defined ONCE in SQL so the notebook,
the training code and the API all read exactly the same rows. If this lived in
Python it would drift between callers; as a view it cannot.

Four filters are applied here, and the third is the one the data-quality
checks showed was missing:

  1. is_accurate          -- FastF1's own clean-lap flag
  2. track_status = {1}   -- strictly clear conditions (a yellow-flag lap means
                             drivers lifted, so its time is not representative)
  3. PACE FILTER          -- lap_time <= 1.25 x the stint median
  4. dry compounds only   -- SOFT/MEDIUM/HARD

Filter 3 exists because filters 1 and 2 TOGETHER are still insufficient. Mexico
City 2023 lap 36 records 148.5s against an ~83s median while carrying
track_status={1} and is_accurate=true: it is the standing-restart lap after a
red flag, including time spent stationary on the grid. Only 17 such laps exist
in 46 races, but least squares minimises SQUARED error, so one of them inside a
34-lap stint moves the fitted slope by a factor of 48.

The median is used as the reference rather than the mean precisely because it is
robust to those outliers -- computing it on the unfiltered green laps and then
filtering against it is safe, where a mean would be dragged by the very laps we
are trying to exclude.

Filter 4 controls track evolution rather than tires. A drying or rubbering-in
track gets faster independently of tire age, and on the ingested data the median
raw slope is -0.386 s/lap for wet-compound stints versus +0.002 s/lap for dry
ones. Restricting to dry compounds makes evolution a second-order effect instead
of the dominant one.

Note what this view deliberately does NOT do: it applies no fuel correction.
Fuel is a modeling assumption (110 kg at 0.03 s/kg, see app/ml/fuel.py) and
belongs in code where it is documented and unit-tested, not baked into a view.
The view supplies the raw facts plus `race_laps` so the fuel model can be
applied -- and varied -- downstream.

Revision ID: e455366cd113
Revises: 552eea404de7
"""

from collections.abc import Sequence

from alembic import op

revision: str = "e455366cd113"
down_revision: str | None = "552eea404de7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Laps slower than this multiple of their stint median are discarded. 1.25 is
# loose enough to keep a lap compromised by traffic (a few percent off) and
# tight enough to reject a restart lap (nearly 80% off).
PACE_FILTER_MULTIPLE = 1.25

# A stint needs enough clean laps for a slope to mean anything. Below this a
# single lap dominates the fit.
MIN_GREEN_LAPS_PER_STINT = 5

CREATE_VIEW = f"""
CREATE OR REPLACE VIEW features.stint_laps AS
WITH green AS (
    -- Filters 1, 2 and 4. Race sessions only: practice and qualifying laps are
    -- push/prep laps, not representative race running.
    SELECT
        l.id                AS lap_id,
        l.session_id,
        l.driver_id,
        l.stint_number,
        l.lap_number,
        l.lap_time_ms,
        l.compound,
        l.tyre_life::float  AS tyre_life,
        l.is_fresh_tyre,
        s.total_laps        AS race_laps,
        e.season_year,
        e.round,
        e.event_date,
        c.name              AS circuit,
        d.abbreviation      AS driver,
        t.name              AS team
    FROM raw.laps l
    JOIN raw.sessions s ON s.id = l.session_id AND s.session_type = 'R'
    JOIN raw.events   e ON e.id = s.event_id
    JOIN raw.drivers  d ON d.id = l.driver_id
    LEFT JOIN raw.circuits c ON c.id = e.circuit_id
    LEFT JOIN raw.teams    t ON t.id = l.team_id
    WHERE l.is_accurate
      AND l.track_status = ARRAY[1]::smallint[]
      AND l.lap_time_ms IS NOT NULL
      AND l.tyre_life   IS NOT NULL
      AND l.stint_number IS NOT NULL
      AND s.total_laps   IS NOT NULL
      AND l.compound IN ('SOFT', 'MEDIUM', 'HARD')
),
stint_stats AS (
    -- One row per (session, driver, stint). percentile_cont is an ordered-set
    -- aggregate, so it needs WITHIN GROUP and a GROUP BY -- it cannot be
    -- written as a plain window function.
    SELECT
        session_id,
        driver_id,
        stint_number,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY lap_time_ms) AS stint_median_ms,
        count(*)                                                 AS stint_green_laps,
        min(lap_number)                                          AS stint_first_lap,
        max(lap_number)                                          AS stint_last_lap
    FROM green
    GROUP BY session_id, driver_id, stint_number
),
session_weather AS (
    -- Session-level aggregate, NOT per-lap. raw.laps stores only durations
    -- (lap_time_ms, sector*_ms), never a session-time offset, so there is no
    -- key on which to join a lap to a weather sample. Track temperature can
    -- move 5-10 C across a race, so this is a real loss of resolution; fixing
    -- it would mean storing FastF1's LapStartTime and re-ingesting.
    SELECT
        session_id,
        avg(air_temp)::float   AS air_temp,
        avg(track_temp)::float AS track_temp,
        bool_or(rainfall)      AS any_rainfall
    FROM raw.weather
    GROUP BY session_id
)
SELECT
    g.lap_id,
    g.session_id,
    g.driver_id,
    g.season_year,
    g.round,
    g.event_date,
    g.circuit,
    g.driver,
    g.team,
    g.compound,
    g.stint_number,
    g.lap_number,
    g.tyre_life,
    g.is_fresh_tyre,
    g.race_laps,
    g.lap_time_ms,
    ss.stint_median_ms,
    ss.stint_green_laps,
    ss.stint_first_lap,
    ss.stint_last_lap,
    w.air_temp,
    w.track_temp,
    w.any_rainfall
FROM green g
JOIN stint_stats ss
  ON  ss.session_id   = g.session_id
  AND ss.driver_id    = g.driver_id
  AND ss.stint_number = g.stint_number
LEFT JOIN session_weather w ON w.session_id = g.session_id
-- Filter 3: the pace filter the standing-restart laps proved was necessary.
WHERE g.lap_time_ms <= ss.stint_median_ms * {PACE_FILTER_MULTIPLE}
  AND ss.stint_green_laps >= {MIN_GREEN_LAPS_PER_STINT}
"""


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS features")
    op.execute(CREATE_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS features.stint_laps")
