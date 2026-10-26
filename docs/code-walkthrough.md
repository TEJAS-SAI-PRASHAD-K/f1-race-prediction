# Code Walkthrough

Every function: what it receives, what it does, what it returns, and who
calls it next. Line numbers refer to the current files.

```
PATH F — Race prediction
  api/routes/prediction.py
    → predict.py    → registry.py                        (the saved model)
                    → features.race_features             (as-of features)
    → train.py      → dataset.py → features.race_features
                                 → degradation (as-of per-circuit slopes)
                    → walk-forward backtest → registry.py
```

3 modules, 1 SQL view, 3 endpoints, 39 tests.

---

# Part 0 — The view

## `alembic/versions/195b31c0a04c_phase4_features_race_features_view.py`

Read the docstring (`1–39`). The section "WHY `GROUPS` AND NOT `ROWS`" describes the
subtlest bug in the project.

| Line | What |
|---|---|
| `48` | `VIEW` — the whole view as one statement |
| `51–77` | `results` CTE — one row per (race, driver) with outcome, grid and `classified` |
| `78–101` | `quali` CTE — best Q lap per driver and the gap to pole |
| `102–163` | `as_of` CTE — **every rolling feature** |
| `165–197` | final projection, including the targets |

### The `results` CTE
`68–72` is worth reading: FastF1 classifies a retirement at the back, so
`finish_position` is populated for almost every row. `classified` separates "reached
the end" from "was given a position anyway", which is what a DNF-rate feature needs.

### The `quali` CTE
Derived from Q-session **laps**, not results, because the schema stores
session results without the Q1/Q2/Q3 split. `80–82` notes that. The subquery at
`87–92` takes each driver's best accurate lap; `93–98` takes the weekend's pole time;
`85` computes the gap.

### The `as_of` CTE — the leakage guard
Three groups of windows, and the frame type differs by design:

**Driver-level** (`107–127`) — `driver_form_3`, `driver_form_5`,
`driver_dnf_rate_10`, `driver_races_to_date`. One row per driver per race, so `GROUPS`
and `ROWS` would agree; `GROUPS` is used for consistency.

**Team-level** (`129–145`) — read the comment at `129–131`. **`GROUPS` is essential
here.** A team has two cars per race, so `ROWS ... AND 1 PRECEDING` reaches the
driver's own teammate in the race being predicted.

**Circuit-level** (`155–162`) — `circuit_mean_abs_position_change`, the overtaking
difficulty proxy. A low value means the grid order survives, i.e. overtaking is hard.
Twenty rows share each event, so `GROUPS` is essential again.

Every frame ends at `1 PRECEDING`. That is the whole defence.

### What is deliberately absent
`30–38` of the docstring: nothing measured during the race. `grid_position` and the
qualifying gap are in because both settle before the start. Expected tire degradation
is **not** here — it is fitted in Python, so the view stays pure
SQL over `raw`.

---

# Part F — The prediction path

## F1. `app/ml/prediction/dataset.py` (254 lines)

Read `1–35`. The "HONEST EXPECTATIONS" section is there because the numbers only make
sense with the sample size in mind.

| Line | What |
|---|---|
| `37–51` | `PRE_RACE_FEATURES` — the 12 columns from the view |
| `53` | `DEGRADATION_FEATURE` — added in Python |
| `55` | `NUMERIC_FEATURES` |
| `65` | `IDENTITY_FEATURES` — **excluded by default**; `57–64` explains the row-to-column maths |
| `67–68` | `TARGET_POSITION`, `TARGET_PODIUM` |

### `load_race_features(engine)` — `75–79`
One `read_sql`, ordered by date then grid so downstream slicing is deterministic.

### `add_degradation_feature(df, engine)` — `82–114`
- **Does:** `106–113` loops over seasons; for each, fits the slopes on seasons
  **strictly before** it (`107`), takes the per-circuit median slope (`111`), and maps
  it onto that season's rows (`113–115`).
- **Why the loop** (`89–97`): the naive version fits on everything and joins
  by circuit. A rate measured partly from 2024 races cannot be a feature for a 2024
  race. The earliest season ends with no rate, which is correct — nothing was known
  yet.
- **Returns:** a copy with `circuit_degradation_s_per_lap`.

### `build_dataset(engine, *, with_degradation)` — `117–128`
The pipeline the notebook, training and prediction all call. `126–127` casts the
identity columns to `category` even when unused, so the dtype is stable.

### `feature_matrix(df, include_identity)` — `135–140`
Features in a **fixed order**, pinned because a saved model
expects its training layout.

### `baseline_grid(df)` — `146–153`
One line: `df["grid_position"]`. **The baseline that matters**, strong because
overtaking is hard. Any model that cannot beat it has learned nothing beyond what
qualifying already said.

### `baseline_driver_form(df)` — `156–166`
Falls back to grid where form is unknown (`165–166`), so it never returns NaN. Two
baselines separate two claims: beating form says the model uses qualifying; beating
grid says it adds something qualifying did not contain.

### `time_split(df, test_season)` — `173–192`
- **Read `176–190`.** Two reasons a random split is invalid, and the second is
  specific to this phase: the features are built from past races, so training on a
  2024 race and testing on a 2023 one means the training row's features were computed
  from data including the test row's race. **The leak runs backwards through the
  feature construction**, which shuffling cannot fix.

### `walk_forward_splits(df, *, min_train_races)` — `195–221`
- **Does:** `205–210` lists events in date order; `211–220` yields one `(train, test)`
  pair per race, training on everything strictly before its date.
- **Returns:** ~81 folds on this data.
- The only evaluation that mirrors real use, and strictly harder than a season
  holdout because early folds train on very little.

### `audit_leakage(df)` — `224–254` ← run before every training
- **Checks two structural properties** (`227–235`):
  1. A team-level feature over previous races must be **identical for both of a
     team's cars** in the same race. Disagreement means the window reached into the
     current race — exactly what `ROWS` does and `GROUPS` does not.
  2. A driver's first-ever race must have **no form**. A value there means the window
     included the race itself.
- **Returns:** a dict with counts and a `passes` flag.
- `train_all` **refuses to train if it fails**. An advisory leakage check gets ignored.

---

## F2. `app/ml/prediction/train.py` (454 lines)

Read `1–33`. "MODEL ORDER IS DELIBERATE, AND OPPOSITE TO PHASE 2" is the headline.

### `spearman_by_race(df, predicted)` — `62–79`
- **Does:** groups by `event_id` (`72`), computes Spearman per race, averages (`79`).
- **Why per race and not pooled** (`65–71`): pooling is inflated by a good driver
  beating a backmarker in every race. The model would score well for knowing
  Verstappen beats a Haas, which is not the question. Ranking 20 cars *within* one
  race is.

### `top_k_hit_rate(df, predicted, k)` — `82–92`
Fraction of the actual top-k captured by the predicted top-k, per race. `88–90` uses
`nsmallest` on both sides — positions are better when lower.

### `calibration_table(probabilities, outcomes, bins)` — `107–127`
- **Why it exists** (`110–116`): a model can rank well and still be badly calibrated,
  and a miscalibrated probability is dangerous to act on. Brier score alone will not
  show it; a table of predicted-vs-observed will.

### `_preprocessor(include_identity)` — `130–157`
- **Does:** median imputation then scaling (`144–148`); one-hot identity columns if
  requested (`150–157`).
- **Read `134–142`:** imputation is *required* because Ridge and sklearn's
  RandomForest cannot accept NaN. That is a real cost —
  `driver_circuit_mean_finish` is missing 35% of the time (a driver's first visit to a
  circuit) and imputing the median asserts "this driver performs averagely here",
  which is a guess. XGBoost learns a split direction for missing instead, which is one
  concrete advantage it has on this data.

### `build_ridge(include_identity)` — `160–169`
`alpha=10.0`, not the usual 1.0: with 2,556 rows and 13 features the penalty has to
work harder. Chosen on the walk-forward backtest rather than guessed.

### `build_random_forest(include_identity)` — `172–190`
`max_depth=6`, `min_samples_leaf=20` — shallow, because the dataset is small enough
that a deep tree memorises driver names.

### `build_xgboost(include_identity)` — `193–216`
**`max_depth=2` is not a typo** (`196–198`, set at `206`). On 2,556 rows anything deeper overfits —
the same force seen at 100× the sample size, stronger here.

### `build_podium_classifier(include_identity)` — `219–232`
- **Podium, not win** (`222–226`): ~360 podiums against ~121 wins, so three times the
  positive class. "Predicting wins directly from 121 examples is not a modelling
  problem, it is a wish."

### `walk_forward_backtest(builder, df, *, ...)` — `247–269`
- **Takes a *factory*,** not a model, so each fold gets a fresh estimator.
- `256–265` per fold: rebuild, fit on the past, predict the one test race.
- **Returns:** aggregate metrics **plus the per-race predictions**, so they can be
  inspected rather than taken on trust.

### `backtest_baselines(df, min_train_races)` — `272–284`
Scores the baselines on **exactly the same races** as the models (`280`). Comparing
against a baseline measured on a different set would be meaningless.

### `train_all(*, include_identity, save, min_train_races)` — `287–433`
| Line | Step |
|---|---|
| `293–295` | build the dataset |
| `295–300` | **run the leakage audit and RAISE if it fails** |
| `305–310` | print the row count and the "this is a SMALL dataset" warning |
| `312–325` | compute and print both baselines *before* any model runs |
| `330–347` | per model: walk-forward backtest, print metrics and the delta vs grid |
| `350–377` | the podium classifier, Brier score and calibration table |
| `379–432` | refit the winner on everything and register it plus the podium model |

The baseline block matters: baselines are printed first, with "Grid order is a STRONG baseline.
Beating it is the whole task." An MAE of 3.14 is unreadable without 3.52 beside it.

---

## F3. `app/ml/prediction/predict.py` (315 lines)

Read `1–20`. Two modes, and the difference matters.

| Line | What |
|---|---|
| `41–47` | `_load` — caller's session or its own |
| `49–67` | `_load_with_session` — reads the active version fresh, caches the artifact |
| `71–80` | `DriverPrediction` — includes `missing_features` |
| `94–137` | `_rank_and_package` — ranks, detects missing features per driver |
| `140–156` | `_as_str` — **a real bug fix**, see below |

### `_as_str(value)` — `140–156`
pandas turns a missing string into `NaN`, which is a **float**. Without this an
unknown driver's `team` arrives as `float('nan')` and Pydantic rejects it for a
`str | None` field. The rookie test is what caught it.

### `predict_event(engine, event_id, *, db)` — `168–211`
- **Does:** loads both models, reads that race's stored as-of features (`180–186`),
  attaches degradation (`190`), reindexes to the saved feature order (`194`), predicts
  position and podium probability.
- **Read `172–177`:** the served model was fitted on **all** seasons including this
  race, so these numbers are **in-sample and not a performance estimate**. The returned
  `warnings` say so. It is for inspecting reasoning, not claiming accuracy.

### `predict_from_grid(engine, circuit, grid, *, db)` — `214–305`
- **Does:** `231–237` takes each driver's most recent row as their as-of state;
  `244–266` builds a row per grid entry, warning and leaving NaN for an unknown driver;
  `268–269` overrides grid and quali position, the one thing known about *this* race.
- **Read `221–229`:** features are **one race stale** by construction — strictly they
  should be the state going into *this* race. That is the honest best available before
  the race happens, and it is stated rather than hidden.
- A driver with no history is **reported, not imputed** (`248–252`). A rookie's
  prediction resting on grid position alone is the right answer.

---

## F4. `app/schemas/prediction.py` (92) and `app/api/routes/prediction.py` (180)

| Line | What |
|---|---|
| `schemas:18–43` | `DriverPredictionOut` — note `podium_probability` (`25–32`) says it is NOT derived from `predicted_position` |
| `schemas:65–92` | `BacktestOut` — baselines are required fields, not optional extras |
| `routes:21–31` | `HONEST_NOTES` — returned on every backtest response |

### `GET /race/{event_id}` — `routes:67–85`
Docstring `69–79` leads with **"These numbers are IN-SAMPLE"** in bold.

### `POST /race` — `routes:87–106`
`103` rejects a duplicate driver before predicting. Docstring `89–98` explains the
one-race-stale approximation and the rookie behaviour.

### `GET /backtest` — `routes:108–180`
- `146–151` 500s if the registered model has no baseline metrics, rather than
  reporting an MAE with nothing to compare it to.
- `165–179` assembles walk-forward metrics, both baselines, the improvement fraction,
  every candidate's numbers, and the podium skill.

---

# Part G — Tests (39)

| File | Tests | Covers |
|---|---|---|
| `test_prediction.py` | 21 | **The leakage suite**, splitting, baselines, metric semantics, models on real data |
| `test_prediction_api.py` | 18 | All 3 endpoints, in-sample warning, rookie handling, backtest fields |

Four worth reading as examples:

**`TestLeakageGuards::test_teammates_agree_on_team_form`** — asserts over the whole
dataset that a team-level feature is identical for both of a team's cars in the same
race. The single most valuable test here.

**`TestLeakageGuards::test_no_in_race_measurements_among_the_features`** — reviews the
feature list **by name** against a banned set. The obvious leak, which is also the
easiest to commit by accident while joining tables.

**`TestMetrics::test_spearman_is_averaged_per_race_not_pooled`** — constructs two
races, one perfectly ordered and one exactly reversed, and asserts the mean is 0.0.
Pins the semantics, not just the plumbing.

**`test_prediction_api.py::fitted`** — module-scoped because `build_dataset` runs
the slopes per season. Rebuilding it per test took that file from 15s to 207s; only the
registry row, which must live in each test's rolled-back transaction, is per-test.

```bash
.venv/bin/python -m pytest tests/test_prediction.py tests/test_prediction_api.py -q
```

---

# Reading order

1. **The migration's docstring** (`1–39`) — the `GROUPS` vs `ROWS` argument
2. **`dataset.py:224–254`** — `audit_leakage`, the two structural properties
3. **`dataset.py:173–192`** — `time_split`, and why the leak runs backwards
4. **`dataset.py:82–114`** — `add_degradation_feature`, as-of by season
5. **`train.py:62–79`** — `spearman_by_race`, and why per-race not pooled
6. **`train.py:193–216`** — `build_xgboost`, and why `max_depth=2`
7. **`train.py:295–300`** — the audit gate that refuses to train

Then reproduce the Ferrari table from the findings doc yourself. It is the clearest
demonstration of leakage in the project, and it takes one query.
