"""Build the per-stint lap dataset behind the circuit degradation feature.

Reads `features.stint_laps`, applies
the fuel correction, and constructs the target. Everything here is deterministic
and side-effect free, so the notebook, training and the API all get identical
rows from identical inputs.

TARGET DESIGN
-------------
We do NOT predict raw lap time. Raw lap time is dominated by circuit and car:
the model would spend its capacity learning that Monaco is slower than Monza
instead of learning anything about tires.

Instead we predict the lap's deviation from its OWN STINT's fuel-corrected
median:

    target_ms = corrected_lap_ms - median(corrected_lap_ms over that stint)

Subtracting a within-stint baseline removes, in one step, every effect that is
constant inside a stint: driver skill, car performance, circuit, fuel strategy,
ambient conditions, even the compound's absolute grip level. What survives is
the part that VARIES with tire age, which is exactly degradation.

A reasonable worry: does subtracting the stint median also remove the
degradation signal? No. It removes the stint's mean LEVEL but preserves its
SLOPE. A stint running from -1.0s to +1.0s around its median still has the same
gradient against tyre_life. The level is a nuisance parameter; the slope is the
quantity of interest.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.ml.features.fuel import fuel_effect_seconds_per_lap

# Columns the model treats as categorical. XGBoost needs these as pandas
# `category` dtype with `enable_categorical=True`; sklearn's HistGBR accepts the
# same dtype via `categorical_features="from_dtype"`.
CATEGORICAL_FEATURES = ["compound", "circuit", "driver", "team"]

NUMERIC_FEATURES = [
    "tyre_life",
    "tyre_life_sq",
    "lap_number",
    "race_laps",
    "stint_number",
    "air_temp",
    "track_temp",
    "is_fresh_tyre_int",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target_ms"

# Hard cap on |target|. Measured on the ingested data, REAL degradation has a
# total swing of 1.18s at the median stint, 3.09s at p95 and 5.45s at p99.
# Target is a deviation from the stint MEDIAN, so it spans roughly half that
# swing: about +/-2.7s even for a p99 stint. Anything beyond 4s is therefore
# not tire degradation.
#
# This filter exists because the view's dry-compound filter does NOT fully
# control track evolution -- the same lesson `is_accurate` and `track_status`
# teach. A driver on slicks when rain arrives sets a lap 19-24s off
# their stint median that is still flagged accurate, still strictly green, and
# still on a dry compound. Observed at Zandvoort 2023, Monaco 2023 and
# Silverstone 2022, all races with changeable conditions.
#
# Least squares minimises SQUARED error, so a single +24s row carries roughly
# 80x the leverage of a typical +2.7s one. Keeping it would let one lap in
# 50,000 dominate a circuit's fitted curve.
MAX_ABS_TARGET_MS = 4_000.0

# Identifier columns kept for grouping and splitting but never fed to a model.
# `session_id` is the GroupKFold key; `season_year` drives the time-based split.
ID_COLUMNS = [
    "lap_id",
    "session_id",
    "driver_id",
    "season_year",
    "round",
    "event_date",
    "stint_key",
]

_LOAD_SQL = text("SELECT * FROM features.stint_laps")


def load_stint_laps(
    engine: Engine,
    *,
    circuits: Sequence[str] | None = None,
    before_season: int | None = None,
) -> pd.DataFrame:
    """Read the view. One row per clean, dry, green-flag race lap.

    The optional filters are applied in SQL, for callers that need a slice
    rather than the whole 104k-row view. They are a pure row filter: every
    downstream step in `build_dataset` is per-row or per-stint (the target's
    baseline is a within-`stint_key` median, and each slope is fitted one stint
    at a time), so a filtered read produces bit-identical rows for the stints it
    keeps. Nothing here is normalised against a dataset-wide statistic, which
    is what would make filtering change the numbers.

    `circuits=None` and `before_season=None` mean no filter, so the default is
    the full view as before.
    """
    clauses: list[str] = []
    params: dict[str, object] = {}
    if circuits is not None:
        clauses.append("circuit = ANY(:circuits)")
        params["circuits"] = list(circuits)
    if before_season is not None:
        clauses.append("season_year < :before_season")
        params["before_season"] = int(before_season)

    if not clauses:
        return pd.read_sql(_LOAD_SQL, engine)

    sql = text(
        "SELECT * FROM features.stint_laps WHERE " + " AND ".join(clauses)
    )
    return pd.read_sql(sql, engine, params=params)


def add_fuel_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Add `fuel_penalty_ms` and `lap_time_fuel_corr_ms`.

    Vectorised equivalent of `app.ml.features.fuel.fuel_penalty_ms`, computed per row
    from that lap's `lap_number` and its race's `race_laps`. The scalar version
    stays the readable reference and the tests assert the two agree.
    """
    out = df.copy()
    effect_s_per_lap = (
        out["race_laps"].map(fuel_effect_seconds_per_lap).astype(float)
    )
    laps_of_fuel_remaining = (out["race_laps"] - out["lap_number"]).clip(lower=0)
    out["fuel_penalty_ms"] = laps_of_fuel_remaining * effect_s_per_lap * 1000.0
    out["lap_time_fuel_corr_ms"] = out["lap_time_ms"] - out["fuel_penalty_ms"]
    return out


def _stint_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["session_id"].astype(str)
        + "_"
        + df["driver_id"].astype(str)
        + "_"
        + df["stint_number"].astype(str)
    )


def add_target(
    df: pd.DataFrame, max_abs_target_ms: float = MAX_ABS_TARGET_MS
) -> pd.DataFrame:
    """Add `stint_key` and the target: delta from the stint's corrected median.

    Two passes, which is the whole point:

      pass 1  compute each stint's median fuel-corrected lap time and the
              resulting target, then DROP rows whose |target| exceeds
              `max_abs_target_ms` (see the constant for why 4s)
      pass 2  recompute the median on the survivors and rebuild the target

    The second pass matters. The median is robust, so dropping outliers barely
    moves it -- but "barely" is not "not at all", and the baseline the model
    regresses against should be computed from the rows the model actually sees.
    Skipping pass 2 leaves every target in a contaminated stint biased by a lap
    that is no longer in the dataset.

    The baseline is built on FUEL-CORRECTED times rather than reusing the view's
    `stint_median_ms`, which is a raw-time median. The view's median exists to
    drive the view's own coarse pace filter; the model's baseline must live in
    the same units as the model's input.

    Filtering responsibility is split deliberately:
      features.stint_laps  -- STRUCTURAL validity (green, accurate, dry, a
                              stint with enough laps, no restart laps)
      this function        -- PACE validity, in target units, where it is
                              unit-tested and can be varied from a notebook
    """
    out = df.copy()
    out["stint_key"] = _stint_key(out)

    # pass 1
    first_median = out.groupby("stint_key")["lap_time_fuel_corr_ms"].transform("median")
    provisional_target = out["lap_time_fuel_corr_ms"] - first_median
    out = out[provisional_target.abs() <= max_abs_target_ms].copy()

    # pass 2
    corrected_median = out.groupby("stint_key")["lap_time_fuel_corr_ms"].transform(
        "median"
    )
    out["stint_corr_median_ms"] = corrected_median
    out[TARGET] = out["lap_time_fuel_corr_ms"] - corrected_median
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model's feature columns and set categorical dtypes.

    `tyre_life_sq` gives a LINEAR model the ability to bend. Degradation is
    rarely a straight line -- tires often fall off a cliff -- and without a
    quadratic term a linear model could only ever fit a slope. Tree models find
    curvature on their own and do not need it, but they are not harmed by it
    either.
    """
    out = df.copy()
    out["tyre_life_sq"] = out["tyre_life"] ** 2
    out["is_fresh_tyre_int"] = out["is_fresh_tyre"].fillna(False).astype(int)

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("category")

    # Session-mean weather is missing for a few older sessions; leave it as NaN.
    # Both XGBoost and HistGBR handle NaN natively by learning a default
    # direction at each split, so imputing a mean here would only add a
    # fabricated value and discard the information that it was missing.
    for col in ("air_temp", "track_temp"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def build_dataset(
    engine: Engine,
    *,
    circuits: Sequence[str] | None = None,
    before_season: int | None = None,
) -> pd.DataFrame:
    """Full pipeline: view -> fuel correction -> target -> features.

    `circuits` / `before_season` are passed through to `load_stint_laps`; see
    there for why filtering the read does not change the rows that survive.
    """
    df = load_stint_laps(engine, circuits=circuits, before_season=before_season)
    if df.empty:
        return df
    df = add_fuel_correction(df)
    df = add_target(df)
    df = add_features(df)
    return df


def time_split(
    df: pd.DataFrame, test_season: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by SEASON, never randomly.

    A random split is invalid here and the reason is worth internalising:
    consecutive laps in one stint are near-duplicates, so a random split puts
    almost-identical rows on both sides. The model is then scored on rows it has
    effectively already seen, and the test MAE looks far better than the model
    really is. This is the single most common way a beginner ML project fools
    itself.

    Splitting on season answers the question that actually matters -- "would
    this model have worked on a season it never saw?" -- which is also how it
    would be used.

    `test_season` defaults to the most recent season present.
    """
    if test_season is None:
        test_season = int(df["season_year"].max())
    train = df[df["season_year"] < test_season].copy()
    test = df[df["season_year"] == test_season].copy()
    return train, test


def baseline_mae_ms(df: pd.DataFrame) -> float:
    """MAE of the do-nothing model: "this lap equals its stint median".

    Because the target is already a deviation from the stint median, that
    prediction is literally zero, so the baseline MAE is mean(|target|).

    Any model must beat this. If it cannot, it has learned nothing about tire
    age -- it has only learned the average spread of lap times within a stint.
    """
    return float(df[TARGET].abs().mean())


def align_categories(
    df: pd.DataFrame, categories: dict[str, list[str]]
) -> pd.DataFrame:
    """Re-apply a pinned set of categories, mapping unseen values to NaN.

    Required for XGBoost, which treats a category absent from training as a
    hard error rather than a fallback. See app/ml/categories.py.
    """
    out = df.copy()
    for col, allowed in categories.items():
        if col not in out.columns:
            continue
        out[col] = pd.Categorical(
            out[col].astype(object).where(out[col].astype(object).isin(allowed)),
            categories=allowed,
        )
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Just the feature columns, in a fixed order.

    Column ORDER matters: a saved XGBoost booster expects the same layout it
    was trained on, and a silently reordered frame produces predictions that
    look plausible and are wrong.
    """
    return df[FEATURES]


def target_vector(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET].to_numpy(dtype=float)
