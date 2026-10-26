# F1 Race Prediction

Predicts Formula 1 finishing positions and podium probability from information
available **before the race starts** — grid position, driver and team form,
qualifying pace and circuit history — using FastF1 timing data from 2019 onward.

Over an 88-race walk-forward backtest it beats the "finishes in grid order"
baseline by **12.0%** on mean absolute error.

## Results

Walk-forward: train on every race before race N, predict race N, step forward.
88 tested races, 2,556 driver-race rows.

| | MAE (positions) | Spearman | top-3 | top-1 |
|---|---|---|---|---|
| baseline: grid order | 3.573 | +0.593 | 60.2% | 44.3% |
| baseline: recent form | 3.801 | +0.534 | 49.2% | 34.1% |
| **Ridge** | **3.143** | **+0.674** | **66.3%** | **60.2%** |
| Random forest | 3.227 | +0.672 | 59.8% | 55.7% |
| XGBoost | 3.228 | +0.670 | 62.5% | 53.4% |

Podium probability is well calibrated — Brier **0.0683** against a base rate of
0.1277, **+46.5% skill**:

| predicted | n | observed |
|---|---|---|
| 0.03 | 1302 | 0.02 |
| 0.29 | 165 | 0.32 |
| 0.50 | 127 | 0.52 |
| 0.70 | 100 | 0.72 |
| 0.86 | 62 | 0.82 |

**Ridge wins, and that is the interesting result.** On 2,556 rows the linear
model beats both tree ensembles. The same feature pipeline run at lap level —
roughly 110,000 rows — rewards a gradient booster instead. The amount of data
decides the model, not the model's reputation.

## What makes this hard

**The baseline is strong.** Overtaking in modern F1 is difficult, so "the race
finishes in grid order" already scores an MAE of 3.57 positions and a Spearman
of +0.59. Any model that cannot beat that has learned nothing qualifying did
not already say. Every number above is reported against it, because an MAE of
3.14 is meaningless on its own.

**The dataset is small and correlated.** 128 races is 2,556 rows, and rows
within a race are not independent — one safety car moves twenty finishing
positions at once. A random train/test split would put the same race on both
sides and report a number that cannot be reproduced in use.

**Most of the work is not modeling.** It is making sure no feature knows
something it could not have known before the race.

## The leakage guards

Three separate leaks, each of which produces a better-looking model and no error.

**1. Rolling windows must end before the current race.** Every form feature uses
a frame ending at `1 PRECEDING`, so a driver's own result never contributes to
the feature predicting it.

**2. `GROUPS`, not `ROWS` — the subtle one.** A team runs two cars. With a `ROWS`
frame, "one row preceding" a driver is *their own teammate in the race being
predicted*, so a team-level feature silently imports the current race's result.
`GROUPS` steps by distinct ordering value — by race — and does not. The test
`test_teammates_agree_on_team_form` asserts both of a team's cars see an
identical `team_form_3`; with `ROWS` it fails.

**3. Fitted features need an as-of boundary too.** `circuit_degradation_s_per_lap`
is not counted, it is *fitted* — a per-stint OLS slope on fuel-corrected lap
times. Fitting it on all seasons and joining the result would leak, because the
rate for a 2024 race would be measured partly from 2024 races. It is computed
from seasons **strictly before** the race's own, so the earliest season has no
value at all, which is correct: nothing was known yet.

Details in [docs/race-prediction.md](docs/race-prediction.md).

## The features

Twelve come from `features.race_features`, a SQL view over the raw tables. One
is fitted in Python.

| feature | source | as-of |
|---|---|---|
| `grid_position` | view | the race itself (known before the start) |
| `quali_gap_to_pole_s`, `quali_position` | view | that weekend's qualifying |
| `driver_form_3`, `driver_form_5` | view | previous 3 / 5 races |
| `driver_dnf_rate_10` | view | previous 10 races |
| `driver_races_to_date` | view | all previous races |
| `team_form_3` | view | previous 3 races |
| `team_points_per_entry_to_date` | view | season to date |
| `driver_points_per_race_to_date` | view | season to date |
| `driver_circuit_mean_finish` | view | previous visits to this circuit |
| `circuit_mean_abs_position_change` | view | previous races at this circuit |
| `circuit_degradation_s_per_lap` | **fitted**, `app/ml/features/` | seasons strictly before this one |

Driver, team and circuit identities are deliberately **not** one-hot encoded by
default. With 2,556 rows, 36 drivers and 30 circuits would add ~78 columns — a
3% row-to-column ratio, which is where a linear model starts memorising names.
`driver_form_3` already encodes "is this driver fast right now" more usefully,
and generalises to a driver the model has never seen.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev,notebook]"
createdb f1_race_prediction && createdb f1_race_prediction_test
cp .env.example .env            # edit if your Postgres user differs
.venv/bin/alembic upgrade head  # creates the schemas, tables and views
```

Ingest, train, serve:

```bash
.venv/bin/python -m app.ingestion.cli season 2024 --only R,Q
.venv/bin/python -m app.ml.prediction.train
.venv/bin/uvicorn app.main:app --reload
```

One season is enough to serve predictions but **not** to train: the walk-forward
split needs more than 40 races before it yields a single fold, and the earliest
season has no degradation feature by construction. To reproduce the numbers
above, ingest 2019–2024:

```bash
for y in 2019 2020 2021 2022 2023 2024; do
  .venv/bin/python -m app.ingestion.cli season $y --only R,Q
done
```

That is 128 races and takes a while on first run; FastF1 responses are cached
under `data/fastf1_cache`, so re-runs are fast. Practice sessions are never
read — `race_features` uses only `R` and `Q`.

Every ingest command is safe to re-run: the loader upserts on natural keys,
which matters because FastF1 revises timing data for a while after a session.

## API

```
GET  /health
GET  /api/v1/seasons
GET  /api/v1/seasons/{year}/events
GET  /api/v1/events/{event_id}/sessions
GET  /api/v1/sessions/{id}/laps        ?driver=&compound=&accurate_only=&green_only=
GET  /api/v1/sessions/{id}/results
GET  /api/v1/sessions/{id}/stints
GET  /api/v1/sessions/{id}/pit-stops
GET  /api/v1/drivers

GET  /api/v1/prediction/race/{event_id}  What the model says about a past race
POST /api/v1/prediction/race             Predict from a grid you supply
GET  /api/v1/prediction/backtest         Walk-forward metrics beside the baselines
```

Interactive docs at <http://localhost:8000/docs>.

```bash
curl localhost:8000/api/v1/prediction/race/239
```

```json
{ "circuit": "Monza", "model_algorithm": "ridge",
  "drivers": [
    {"predicted_rank": 1, "driver": "NOR", "grid_position": 1.0,
     "actual_position": 3.0, "podium_probability": 0.79},
    {"predicted_rank": 2, "driver": "PIA", "grid_position": 2.0,
     "actual_position": 2.0, "podium_probability": 0.67}
  ],
  "warnings": ["IN-SAMPLE: the active model was trained on this race"] }
```

`/prediction/race/{id}` reports **in-sample** numbers and says so — the active
model was fitted on every season including that race. `/prediction/backtest` has
the honest number.

A driver with no history is not silently imputed: their missing features are
listed and a warning returned. A rookie's prediction rests almost entirely on
grid position, which is the correct answer and worth saying out loud.

## Tests

```bash
.venv/bin/python -m pytest
```

181 tests, and **they need no ingested data**. The suite builds
`f1_race_prediction_test` by running the migrations — so the views exist and the
migrations themselves are exercised — then seeds four synthetic seasons with a
known signal planted in them: fixed per-driver skill and fixed per-circuit
degradation. That turns "the model beats the baseline" into a statement about
recovering a known truth rather than a hope about whatever happens to be on the
developer's machine.

Each test runs inside a transaction that is rolled back. Tests run against real
Postgres, never SQLite, so array types, `ON CONFLICT` and CHECK constraints are
genuinely exercised.

## Architecture

```
app/
  config.py            pydantic-settings
  db/                  engine, session factory, SQLAlchemy ORM
  ingestion/
    fastf1_client.py   FastF1 wrapper (always telemetry=False)
    transform.py       DataFrame -> rows; every data gotcha handled here
    loader.py          idempotent upserts
    cli.py             typer commands
  ml/
    categories.py      pinned categorical vocabularies (unseen -> NaN)
    registry.py        XGBoost->JSON / sklearn->joblib + ml.model_versions
    features/          the fitted degradation feature: fuel, laps, slopes
    prediction/        dataset, train, predict
  api/routes/          FastAPI endpoints
```

| schema | holds |
|---|---|
| `raw` | ingested FastF1 data, never mutated by modeling code |
| `features` | `stint_laps` and `race_features` views |
| `ml` | model metadata, versions, metrics |

A modeling bug cannot corrupt ingested data, and re-deriving features is always
safe.

## Things the data will do to you

Behaviours of FastF1 data that produce working code with wrong answers. All are
handled in `app/ingestion/transform.py`.

1. **`TrackStatus` is a concatenated flag string, not a number.** `'14'` means
   the lap was partly clear *and* partly under safety car. `int('1267')` yields
   1267, which is nonsense that raises no error. Parsed to a sorted `SMALLINT[]`,
   so `'12'` and `'21'` become the same canonical `{1,2}` rather than two
   distinct arrays that split a `GROUP BY`.

2. **`LapTime` is a `Timedelta` and often `NaT`.** Stored as nullable `BIGINT`
   milliseconds — exact, correctly sortable, and NULL stays distinct from zero.

3. **`Compound` is sometimes the literal string `"None"`.** Not Python `None`.
   `pd.isna("None")` is `False`, so it slips through a naive `str()` and becomes
   a phantom compound category.

4. **`session.date` is naive but already UTC.** Passing it to `TIMESTAMPTZ` makes
   Postgres read it as server-local time, shifting every session date by the
   local UTC offset — by a different amount on a different machine. The feature
   view orders races by time, so this silently reorders history.

5. **`is_accurate` and the green-flag filter together are still not enough.**
   Mexico 2023 lap 36 records **148.5 s** against an ~83 s norm with
   `track_status = {1}` and `is_accurate = true` — it is the standing-restart lap
   after a red flag, including time stationary on the grid. One such lap inside a
   stint wrecks a linear fit, so `features.stint_laps` adds a pace filter and
   `cli.py doctor` reports the count.

6. **One circuit, two names.** FastF1 reports `Location` inconsistently across
   seasons: "Monte Carlo" in 2019 and 2021, "Monaco" from 2022. Every per-circuit
   statistic then splits in two. Canonicalised at ingest, and a migration repairs
   rows stored before that fix.

## Known limitations

- **128 races is a small dataset.** Confidence intervals on a 12% improvement
  are wider than the headline suggests.
- **`finish_position` conflates "finished 18th" with "retired on lap 3".**
  Retirements are classified at the back, so the target mixes pace with
  reliability. `status` distinguishes them; the model does not use it.
- **`POST /prediction/race` is off by one race.** As-of features come from each
  driver's most recent *completed* race, which is the honest best available
  before a race happens.
- **No weather forecast, no sprint races, no in-race information.**

## Data source

[FastF1](https://docs.fastf1.dev) 3.8.3. Full timing data is available from 2018
onward and appears 30–120 minutes after a session ends. Unofficial; not
affiliated with Formula 1.
