# Data Foundation

## What this phase is for

No ML. The point is a trustworthy database and an honest understanding of what the data
actually contains. Every later phase reads from here, so a quiet error introduced now shows up
as an inexplicable model result — by which time you will have no idea where it came
from.

The lesson of this phase: **raw data is never model-ready, and the ways it misleads you do not
raise exceptions.**

## What was built

| Component | File | Does |
|---|---|---|
| Config | `app/config.py` | pydantic-settings; `DATABASE_URL`, cache dir |
| ORM | `app/db/models.py` | 11 tables across `raw` and `ml` schemas |
| Migrations | `alembic/` | multi-schema aware |
| FastF1 wrapper | `app/ingestion/fastf1_client.py` | session loading, name mapping |
| Transform | `app/ingestion/transform.py` | all type coercion and gotcha handling |
| Loader | `app/ingestion/loader.py` | idempotent upserts |
| CLI | `app/ingestion/cli.py` | `session`, `season`, `status` |
| API | `app/api/routes/` | 9 read endpoints |
| Tests | `tests/` | 67 tests against real Postgres |
| Notebook | `notebooks/01_explore_fastf1.ipynb` | the gotchas, demonstrated |

## Design decisions and why

### Integer milliseconds for every duration

`LapTime` arrives as a `pandas.Timedelta`. We store `BIGINT` milliseconds.

Not float seconds: floats accumulate rounding error when summed across 50+ laps, which downstream pit-loss work
does on every simulated race, and float equality is unreliable. Not Timedelta: not a database
type. Integer ms is exact and sorts correctly.

NULL matters as much as the value. A missing lap time is *unknown*, not zero — storing 0 would
silently drag every average down.

### `track_status` as `SMALLINT[]`

The decision with the highest payoff in the phase.

FastF1 concatenates every flag that applied during a lap into one string. Real values from the
2024 Brazilian GP: `'1'`, `'12'`, `'45'`, `'124'`, `'671'`, `'1267'`. The digits mean clear (1),
yellow (2), safety car (4), red flag (5), VSC deployed (6), VSC ending (7).

`int('1267')` returns 1267. No error, no warning — just a plausible integer that is meaningless,
poisoning every model downstream.

Parsing once at the ingest boundary means no query ever has to think about it again:

```sql
WHERE track_status = ARRAY[1]::smallint[]          -- strictly clear
WHERE NOT (track_status && ARRAY[4,5,6,7]::smallint[])  -- no SC/red/VSC
```

**The flags are sorted.** FastF1 emits both `'12'` and `'21'` for the same pair. Unsorted, those
become the distinct arrays `{1,2}` and `{2,1}`, and a `GROUP BY` splits one real condition across
two buckets. This was not hypothetical: it accounted for 26 miscounted laps at São Paulo before
the fix.

### "Green flag" means exactly `[1]`

A yellow-flag lap (`[1,2]`) is *not* green. Drivers lift for a local caution, so the lap time is
not representative. Standard practice for F1 pace analysis is fully-clear laps only, and for
degradation modeling a contaminated lap is worse than a missing one.

This costs data — about a third of a chaotic race — and that is the correct answer rather than a
problem to engineer around.

### CHECK constraints, not Postgres ENUMs

Same validation, but altering a CHECK in Alembic is a one-line migration while altering an enum
type is genuinely awkward. Compound and session type both use CHECK.

### Upsert on natural keys

Every table has a natural unique key and ingestion uses `INSERT ... ON CONFLICT DO UPDATE`.
Re-running is safe and *expected*: FastF1 revises timing data for a while after a session, so
re-ingestion is routine maintenance.

Driver fields use `COALESCE(excluded.x, drivers.x)` so a later session with sparser data cannot
blank out a name learned earlier.

One documented limitation: **upsert never deletes.** If a later fetch returns fewer laps, the
earlier rows remain. Acceptable because FastF1 corrects laps rather than removing them, but
`tests/test_loader.py::test_fewer_laps_on_re_ingest_leaves_old_rows` pins the behaviour so it is
a known property and not a surprise.

### Partial index on the pace query

```sql
CREATE INDEX ON raw.laps (compound, tyre_life) WHERE is_accurate;
```

Pace queries filter on `is_accurate` every time. The `WHERE` clause keeps the index small by
skipping the laps that will never be selected.

### Tests against real Postgres

The test suite runs on `f1_intelligence_test`, not SQLite — a direct benefit of choosing one
database everywhere. Array round-trips, `ON CONFLICT` semantics and CHECK constraints are
genuinely exercised, so a passing test means something. Each test runs in a transaction that is
rolled back, so the suite stays fast and leaves nothing behind.

## Things discovered by looking, not assuming

Three of these contradicted reasonable expectations. All were found by exploring real sessions
before writing the loader — the order the plan insisted on, and it paid for itself.

1. **The event schedule carries no latitude/longitude.** Those columns stay NULL. No feature
   needs them.

2. **`results.Time` is cumulative for P1 but gap-to-leader for everyone else.** Stored verbatim
   and documented. The feature view uses `position`, not `time_ms`, partly for this reason.

3. **`PitOutTime - PitInTime` is sometimes not a pit stop at all.** At Brazil 2024 a red flag
   fell with cars in the pit lane, making the difference 23 minutes. Durations outside 8–180 s
   are stored as NULL. Monza, a clean race, keeps 30 of 31 stops with a ~25 s median, which is
   correct for F1; São Paulo keeps 18 of 36, which is the honest result for that race.

4. **Qualifying sessions produce many "pit stops."** Drivers return to the pits between runs, so
   `raw.pit_stops` for a Q session is really a list of in/out laps. Not a bug, but **downstream pit-loss work
   must estimate pit loss from race sessions only.**

5. **`Deleted` has dtype `object` holding both `None` and `NaN`.** `bool(NaN)` is `True`, so
   naive coercion would mark laps as deleted that were not.

6. **A red-flagged lap can record a 42-minute lap time.** Monaco 2024 was stopped on lap 1 after
   the Perez/Magnussen/Hulkenberg crash. Cars sat in the pit lane while the clock ran, so all 16
   classified cars have a lap-1 time of roughly 2,500 seconds.

   The data is not wrong -- that genuinely was the elapsed time. What matters is whether the
   filters catch it, and **both do, independently**: all 16 laps have `is_accurate = false` *and*
   `track_status = {1,2,4,5}`. Zero of them reach the modeling filter.

   That redundancy is the point. Either filter alone would have sufficed here, but relying on one
   means a session where it happens to be absent leaks a 42-minute lap into a lap-time
   regression. `cli.py doctor` now asserts this stays true as more data is ingested.

7. **`Compound` is sometimes the literal string `"None"`.** Not Python `None` -- the four
   characters. Observed in 2023 sessions.

   `pd.isna("None")` is `False`, so a naive `str()` passes it through as a real value. The
   compound CHECK constraint rejected it and the backfill failed loudly, which is exactly what a
   constraint is for: **without it, `"None"` becomes a phantom compound category and the model
   trains on it without complaint.** `to_str` now maps the usual sentinel strings (`none`, `nan`,
   `nat`, `na`, `<na>`, `null`) to `NULL`.

   The general lesson: let the database reject bad data. A constraint that fires during a backfill
   costs you ten minutes; the same bad data flowing silently into a model costs you a week of
   wondering why the numbers are odd.

8. **`session.date` is naive but already UTC.** FastF1 returns
   `Timestamp('2024-04-21 07:00:00')` with `tzinfo=None` for the Chinese GP, while the schedule
   reports `Session5Date` as `15:00+08:00` and `Session5DateUtc` as `07:00`.

   Handing a naive datetime to a `TIMESTAMPTZ` column makes Postgres interpret it in the
   *server's* local timezone. On a machine set to UTC+05:30 every session date was stored 5.5
   hours off -- and the error would differ on another machine, which is worse than being
   consistently wrong.

   This is not cosmetic. The feature view orders races by time for walk-forward backtesting, so a shifted
   timestamp is a correctness problem. `_as_utc()` now localises explicitly, with regression tests
   asserting the instant does not move.

9. **`is_accurate` and the green-flag filter are not sufficient on their own.** The most important
   finding in the phase, and it only appeared once two full seasons were loaded.

   Mexico City 2023, Piastri, stint 3, lap 36:

   | field | value |
   |---|---|
   | lap time | **148.5 s** (normal is ~83 s) |
   | `track_status` | `{1}` -- strictly clear |
   | `is_accurate` | `true` -- FastF1 vouches for it |
   | out-lap? | no |

   **Both filters pass a lap 65 seconds slower than normal.** It is the standing-restart lap after
   the red flag: the recorded time includes sitting stationary on the grid, by which point track
   status had already reverted to clear. The data is correct; the filters simply do not model this
   situation.

   The damage is out of all proportion to the count. There are only 17 such laps in 46 races
   (0.03% of the data), but least squares minimises *squared* error, so a point 65 s off the trend
   has enormous leverage. Measured on this exact stint in notebook 01:

   | fit | slope |
   |---|---|
   | all 34 laps | −0.3113 s/lap |
   | dropping the one restart lap | −0.0065 s/lap |

   **One lap out of 34 changes the answer by a factor of 48.** This is also what produced the
   −3.5 s/lap "degradation" outliers in the raw per-stint table before it was understood.

   **the stint view adds a third, pace-based filter**, for example discarding laps more than ~25%
   above the stint median, or a median-absolute-deviation rule. `cli.py doctor` emits a `[WARN]`
   with the current count so this cannot be quietly forgotten.

   The general lesson, and the best one in this phase: **two independent filters agreeing is not
   proof that data is clean.** Both encode assumptions, and both assumptions failed here in the
   same way. Checking the distribution of your target variable catches what categorical filters
   cannot.

## Verification

```bash
.venv/bin/python -m pytest                                  # 67 tests
.venv/bin/python -m app.ingestion.cli session 2024 Monza R   # twice -- no duplicates
.venv/bin/python -m app.ingestion.cli status
.venv/bin/python -m app.ingestion.cli doctor                 # data-quality checks
```

`doctor` is the one to run after every backfill. Unit tests check logic against fixtures;
`doctor` interrogates the real database for corruption that would otherwise pass silently into a
model -- duplicate keys, orphaned rows, invalid compounds, and implausible lap times or pit
durations surviving the filters.

The strongest check is against reality. For the 2024 Italian GP the pipeline reproduces the
actual race: Leclerc winning from P4 on a one-stop (MEDIUM 15 laps → HARD 38), ahead of Piastri
and Norris, with Verstappen on a two-stop. Lap times land at 81–85 s, correct for Monza.

If a pipeline cannot reproduce a race you can look up, it does not work — regardless of what the
tests say.

## What the modeling layer inherits

- `raw.laps` with clean tire state: `compound`, `tyre_life`, `is_fresh_tyre`, `stint_number`
- A reliable green-flag filter
- Per-session weather for temperature features
- `ml.model_versions`, ready for the registry

49,622 clean green-flag race laps across 2023–2024 are available for modeling.

### Three things the modeling layer must handle first

**1. Add a pace-based outlier filter.** Non-negotiable, per discovery 9 above. `is_accurate` plus
the green-flag filter still admits standing-restart laps 65 s off the pace.

**2. Correct for fuel burn.** A car burns ~2 kg/lap at ~0.03 s/kg, so it gains roughly 0.06 s/lap
purely from getting lighter — working directly against degradation. Fit without correcting and you
measure the *net* while calling it degradation, underestimating tire wear with no warning sign.

**3. Separate track evolution.** A drying or rubbering-in track gets faster independently of tires.
Measured on the ingested data, the median raw slope is −0.386 s/lap for wet-compound stints versus
+0.002 s/lap for dry ones. Restricting to dry compounds makes this a second-order effect rather
than the dominant one.

All three are the same underlying problem: **a model cannot tell you about a variable you never
gave it.** Lap time is the sum of several effects, and a naive fit attributes all of them to
whichever one you happened to put on the x-axis. That is a statistics lesson, not an F1 one.
