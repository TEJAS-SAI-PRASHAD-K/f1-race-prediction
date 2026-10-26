# Race Prediction

## What this phase is for

Predict finishing order from pre-race information. It is the hardest phase, and
the plan said so from the start — not because the modelling is exotic, but because
the dataset is small, the baseline is strong, and the features are built from past
races, which is the single most leak-prone setup in the project.

**The lesson of this phase:** leakage is the failure mode that produces no error. A
leaking model does not crash or warn — it just looks better than it is.

## Honest expectations, set before any numbers

There are ~24 races a year. The ingested data has **128 races and 2,556 driver-race
rows**, and rows within a race are strongly dependent — only one car can win.

The baseline to beat is **"the race finishes in grid order"**, and it is
embarrassingly strong because overtaking in modern F1 is hard. Beating it by a
meaningful margin is a real achievement; failing to would have been informative
rather than embarrassing.

## What was built

| Component | File | Does |
|---|---|---|
| As-of features | `alembic/.../195b31c0a04c_*.py` | `features.race_features`, with the leakage guard |
| Dataset | `app/ml/prediction/dataset.py` | Loads, adds the degradation feature, splits, audits |
| Training | `app/ml/prediction/train.py` | Ridge / RF / XGBoost, walk-forward backtest, podium classifier |
| Prediction | `app/ml/prediction/predict.py` | Past races and upcoming grids |
| API | `app/api/routes/prediction.py` | 3 endpoints |
| Tests | `tests/test_prediction.py`, `test_prediction_api.py` | 39 tests, mostly leakage |

---

## Results

Walk-forward backtest: train on every race before race N, predict race N, step
forward. 88 races tested.

| | MAE (positions) | Spearman | top-3 | top-1 |
|---|---|---|---|---|
| baseline: grid order | 3.573 | +0.593 | 60.2% | 44.3% |
| baseline: recent form | 3.801 | +0.534 | 49.2% | 34.1% |
| **Ridge** | **3.143** | **+0.674** | **66.3%** | **60.2%** |
| Random forest | 3.227 | +0.672 | 59.8% | 55.7% |
| XGBoost | 3.228 | +0.670 | 62.5% | 53.4% |

**+12.0% MAE over the grid baseline**, +0.081 Spearman, and top-1 accuracy up from
44.3% to 60.2%. Modest and real — which is the honest characterisation of this
problem.

### Ridge wins, and that is the finding

At lap level the same pipeline has ~110,000 rows and a gradient booster wins.
Here there are 2,556 and the
linear model wins outright. **The amount of data decides the model, not the model's
reputation.**

Regularisation already mattered more than model choice at that scale; here the
same force pushes past regularisation and into model *family*.

### Podium probability is well calibrated

A separate logistic classifier predicts P(podium). Podium rather than win because
there are ~360 podium finishes against ~121 wins — three times the positive class,
and a far better-conditioned problem. Predicting wins from 121 examples is not a
modelling problem, it is a wish.

| predicted | n | observed |
|---|---|---|
| 0.03 | 1197 | 0.02 |
| 0.29 | 155 | 0.33 |
| 0.50 | 127 | 0.52 |
| 0.70 | 100 | 0.72 |
| 0.86 | 62 | 0.82 |

Brier 0.0683 against a base-rate Brier of 0.1277 — **+46.5% skill**. "70%" means
70%, which matters because a model that ranks well but is miscalibrated is
dangerous to act on.

---

## The leakage guard

Every rolling feature uses a window frame ending at `1 PRECEDING`:

```sql
AVG(finish_position) OVER (
    PARTITION BY driver_id ORDER BY event_date
    GROUPS BETWEEN 3 PRECEDING AND 1 PRECEDING
)
```

That `1 PRECEDING` is the entire defence, and it is **declarative**: the current
race physically cannot enter its own feature. The pandas equivalent is a
`shift(1).rolling(3)` chain that is easy to get off by one and impossible to notice
when you do.

### `GROUPS` not `ROWS` — the subtlest bug in the project

`ROWS` counts physical rows. `GROUPS` counts distinct `ORDER BY` values, treating
ties as one peer group.

For a **driver**-level window the distinction is invisible: one row per driver per
race, so 3 rows and 3 races are the same thing.

For a **team** or **circuit** window it is the difference between correct and
leaking. A team has two cars per race, so with `ROWS ... AND 1 PRECEDING` the
"previous row" is **the driver's own teammate in the race being predicted**.

Measured on Ferrari, 2024:

```
 round | driver | finish | ROWS (leaks) | GROUPS (correct)
-------+--------+--------+--------------+------------------
     1 | LEC    |      4 |              |
     1 | SAI    |      3 |         4.00 |                      <- LEC's result, same race
     2 | BEA    |      7 |         3.33 |             3.50
     2 | LEC    |      3 |         3.50 |             3.50
     3 | LEC    |      2 |         4.33 |             4.25
     3 | SAI    |      1 |         4.00 |             4.25
```

Two tells, both structural:

1. At round 1, `ROWS` gives SAI a team form of **4.00** — Leclerc's finishing
   position in a race that had not happened yet. `GROUPS` correctly gives NULL.
2. At rounds 3 and 4, `ROWS` gives the **two teammates different values** for a
   team-level feature in the same race. A "previous races" team feature must be
   identical for both cars. That inequality is a check anyone can apply without
   knowing anything about window functions.

`tests/test_prediction.py::TestLeakageGuards::test_teammates_agree_on_team_form`
asserts it over the whole dataset.

### A third leak, in the degradation feature

Attaching the circuit degradation rate is useful and the naive version leaks:
fit the slopes on all seasons, join by circuit. A rate measured partly from 2024 races
cannot be a feature for predicting a 2024 race.

`add_degradation_feature` therefore measures the rate for season Y from seasons
**strictly before Y**. The earliest season has no rate at all, which is correct —
nothing was known yet. Both model families handle the resulting NaN.

### The audit

`dataset.audit_leakage` checks two properties that cannot hold if a frame is wrong:

| check | result |
|---|---|
| teammates disagreeing on `team_form_3` | **0** |
| first-ever races carrying form | **0** |

`train_all` runs it before training and **refuses to proceed if it fails**. A
leakage check that is advisory gets ignored.

### Why a random split is invalid here — two reasons

Rows within a race are dependent.

The second is worse and specific to this phase: **the features are built from past
races.** Training on a 2024 race and testing on a 2023 one means the training row's
features were computed from data that includes the test row's race. The leak runs
*backwards through the feature construction*, which no amount of row shuffling can
fix.

---

## Decisions worth understanding

### Identity columns are excluded by default, and it was measured

One-hot encoding 36 drivers, 30 circuits and 12 teams adds ~78 columns to 2,556
rows — a 3% row-to-column ratio, where a linear model starts memorising names.

Measured rather than asserted:

| model | without identity | with identity | change |
|---|---|---|---|
| ridge | 3.143 | 3.156 | +0.013 |
| random forest | 3.227 | 3.227 | +0.000 |
| xgboost | 3.228 | 3.243 | +0.015 |

Consistently **worse**. The rolling features already carry that signal:
`driver_form_3` encodes "is this driver fast right now" more usefully than a driver
dummy, and it generalises to a driver the model has never seen. `--include-identity`
enables them so the cost can be re-measured rather than taken on trust.

### Rank metrics, not just MAE

Position MAE treats "predicted 1st, finished 3rd" and "predicted 18th, finished
20th" as the same 2-position error. They are not remotely the same mistake.

Spearman is computed **per race and then averaged**, never pooled. Pooling would be
inflated by a good driver beating a backmarker in every race — the model would score
well for knowing Verstappen beats a Haas, which is not the question. Ranking 20 cars
within one race is.

### Unseen drivers are reported, not imputed

A rookie has no form history. `predict_from_grid` returns their missing features
explicitly and adds a warning, rather than substituting a median:

```
  1. VER        grid= 1 pred=  3.36 P(podium)=  82%  missing=0
  4. ROOKIE_X   grid=20 pred= 15.19 P(podium)=   0%  missing=10
WARN: ROOKIE_X: no race history. Their prediction rests on grid position alone.
```

That *is* the right answer for a rookie, and saying it out loud is better than hiding
it behind an imputed median that looks like knowledge.

---

## A bug the tests caught

`predict_from_grid` returned `team` as pandas `NaN` for an unknown driver, and
Pydantic rejected it for a `str | None` field. pandas turns a missing string into a
float, so `None` going in does not mean `None` coming out. Fixed with an explicit
`_as_str` coercion; the rookie test is what surfaced it.

---

## Known limitations

1. **The dataset is small.** 2,556 rows, 128 races, strongly correlated within races.
   Every number here should be read with that in mind, and a 0.1-position MAE
   difference between models is not meaningful.
2. **`predict_from_grid` features are one race stale.** They come from each driver's
   most recent race, not their state going into the race being predicted. That is the
   honest best available before the race happens, and it is an approximation.
3. **No weather forecast.** The plan listed rain probability as a feature; the
   ingested weather is observed, not forecast, so using it would be leakage. A real
   forecast feed would fix this.
4. **DNFs are classified at the back.** FastF1 assigns a position to retirements, so
   `finish_position` mixes "finished 18th" with "retired on lap 3". `driver_dnf_rate_10`
   gives the model a handle on it, but the target itself conflates the two.
5. **Sprint races are not ingested**, so sprint-weekend form is partly invisible.
6. **`/prediction/race/{event_id}` is in-sample.** The registered model was fitted on
   all seasons including that race. The response says so; `/prediction/backtest` has
   the honest number.

---

## Verification

```bash
.venv/bin/alembic upgrade head
.venv/bin/python -m app.ml.prediction.train          # runs the leakage audit first
.venv/bin/python -m pytest tests/test_prediction.py -q
```

```bash
curl -s localhost:8000/api/v1/prediction/backtest | python -m json.tool
curl -s -X POST localhost:8000/api/v1/prediction/race \
  -H 'content-type: application/json' \
  -d '{"circuit":"Monza","grid":[{"driver":"VER","grid_position":1},
       {"driver":"LEC","grid_position":2},{"driver":"NOR","grid_position":3}]}'
```

Check: the leakage audit passes, walk-forward MAE beats the grid baseline, and the
podium calibration table tracks the diagonal.

The strongest single check is the Ferrari `ROWS`-vs-`GROUPS` table above. If you
cannot reproduce that, you cannot be sure the features are as-of.
