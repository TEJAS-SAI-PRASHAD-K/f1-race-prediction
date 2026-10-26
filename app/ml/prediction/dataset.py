"""Build the race-prediction dataset.

HONEST EXPECTATIONS, FIRST
--------------------------
This is the hardest phase, and the reason is sample size. There are ~24 races a
year; the ingested data has **128 races and 2,556 driver-race rows**, and rows
within a race are strongly dependent (only one car can win). That is a small,
correlated dataset by any standard, and overfitting it is effortless.

The baseline to beat is "finishing position = grid position". It is
embarrassingly strong, because F1 overtaking is hard. Beating it by a meaningful
margin is a genuine achievement; failing to is informative rather than
embarrassing. Knowing that going in is the difference between learning something
and concluding you are bad at this.

WHERE THE FEATURES COME FROM
----------------------------
Almost all of them from `features.race_features`, whose window frames end at
`1 PRECEDING` so the current race cannot enter its own features. That view's
docstring explains why the frames are `GROUPS` and not `ROWS` -- the subtlest
leak in the project.

One feature is added here instead of in SQL: expected tire degradation at the
circuit, fitted per stint from prior seasons. It is a modeling artefact rather than a
raw fact, so it lives in Python, and it is computed from seasons STRICTLY BEFORE
the race being predicted -- see `add_degradation_feature`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Everything in the view that is knowable before the race starts.
PRE_RACE_FEATURES = [
    "grid_position",
    "quali_gap_to_pole_s",
    "quali_position",
    "driver_form_3",
    "driver_form_5",
    "driver_dnf_rate_10",
    "driver_races_to_date",
    "team_form_3",
    "team_points_per_entry_to_date",
    "driver_points_per_race_to_date",
    "driver_circuit_mean_finish",
    "circuit_mean_abs_position_change",
]

# Added in Python: fitted per circuit from prior seasons' stints.
DEGRADATION_FEATURE = "circuit_degradation_s_per_lap"

NUMERIC_FEATURES = [*PRE_RACE_FEATURES, DEGRADATION_FEATURE]

# Held out of the default feature set on purpose. With 2,556 rows, one-hot
# encoding 36 drivers, 30 circuits and ~12 teams adds ~78 columns -- a 3%
# row-to-column ratio, which is where a linear model starts memorising names.
# The rolling features already carry most of that signal: `driver_form_3`
# encodes "is this driver fast right now" far more usefully than a driver dummy,
# and it generalises to a driver the model has never seen.
# `include_identity=True` enables them so the cost can be measured rather than
# asserted; notebook 04 does exactly that.
IDENTITY_FEATURES = ["driver", "team", "circuit"]

TARGET_POSITION = "finish_position"
TARGET_PODIUM = "podium"

ID_COLUMNS = ["event_id", "season_year", "round", "event_date", "driver", "team", "circuit"]

_LOAD_SQL = text("SELECT * FROM features.race_features ORDER BY event_date, grid_position")


def load_race_features(engine: Engine) -> pd.DataFrame:
    """Read the view. One row per (race, driver)."""
    df = pd.read_sql(_LOAD_SQL, engine)
    df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def add_degradation_feature(df: pd.DataFrame, engine: Engine) -> pd.DataFrame:
    """Attach expected tire degradation per circuit, computed AS-OF each season.

    The naive version -- fit the slopes on all seasons and join the result -- would
    leak. A degradation rate measured partly from 2024 races cannot be a feature
    for predicting a 2024 race, because it contains information from the race
    being predicted and from its future.

    So the rate for a race in season Y is measured only from seasons < Y. The
    first season in the data therefore has no rate at all, which is correct:
    nothing was known yet. Both model families handle the resulting NaN.

    Only the circuits and prior seasons `df` actually asks about are read, and
    the filter is pushed into SQL. Reading the whole view and discarding most of
    it is what made a single-race request take ~8.7s: predicting ONE race needs
    one circuit's prior stints (~2.8k laps), not all 104k. For a full training frame
    the filter spans every circuit and so loads the same rows as before; the
    only rows it drops there are the newest season's, which an as-of feature
    can never use. Slopes are unchanged either way, because each stint is fitted
    on its own -- see `app.ml.features.laps.load_stint_laps`.
    """
    from app.ml.features import degradation
    from app.ml.features import laps as lap_data

    seasons = sorted(df["season_year"].unique())
    circuits = sorted({str(c) for c in df["circuit"].dropna().unique()})
    laps = lap_data.build_dataset(
        engine,
        circuits=circuits,
        before_season=int(max(seasons)) if seasons else None,
    )
    out = df.copy()
    out[DEGRADATION_FEATURE] = np.nan

    # No prior stints for these circuits at all -- the first season in the data,
    # or a circuit never raced before. The feature stays NaN, which is the same
    # answer the unfiltered read gave, reached without the loop.
    if laps.empty:
        return out

    for season in sorted(out["season_year"].unique()):
        prior = laps[laps["season_year"] < season]
        if prior.empty:
            continue
        fits = degradation.fit_all_stints(prior, use_fuel_correction=True)
        if fits.empty:
            continue
        by_circuit = fits.groupby("circuit", observed=True)["slope"].median()
        mask = out["season_year"] == season
        out.loc[mask, DEGRADATION_FEATURE] = (
            out.loc[mask, "circuit"].map(by_circuit).astype(float)
        )

    return out


def build_dataset(engine: Engine, *, with_degradation: bool = True) -> pd.DataFrame:
    """Load the view and attach the circuit degradation feature."""
    df = load_race_features(engine)
    if df.empty:
        return df
    if with_degradation:
        df = add_degradation_feature(df, engine)
    else:
        df[DEGRADATION_FEATURE] = np.nan
    for col in IDENTITY_FEATURES:
        df[col] = df[col].astype("category")
    return df


def feature_columns(include_identity: bool = False) -> list[str]:
    return [*NUMERIC_FEATURES, *IDENTITY_FEATURES] if include_identity else list(NUMERIC_FEATURES)


def feature_matrix(df: pd.DataFrame, include_identity: bool = False) -> pd.DataFrame:
    """Features in a fixed order. Order is pinned because a saved model expects
    its training layout."""
    return df[feature_columns(include_identity)]


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def baseline_grid(df: pd.DataFrame) -> np.ndarray:
    """"The race finishes in grid order."

    The baseline that matters. It is strong because overtaking in modern F1 is
    hard, and any model that cannot beat it has learned nothing beyond what
    qualifying already told you.
    """
    return df["grid_position"].to_numpy(dtype=float)


def baseline_driver_form(df: pd.DataFrame) -> np.ndarray:
    """"The driver finishes where they have been finishing lately."

    A second baseline, to separate two different claims: beating this says the
    model uses qualifying; beating the grid baseline says it adds something
    qualifying did not already contain.
    """
    form = df["driver_form_3"].to_numpy(dtype=float)
    grid = baseline_grid(df)
    return np.where(np.isnan(form), grid, form)


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


def time_split(
    df: pd.DataFrame, test_season: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season. A random split is invalid here for two reasons.

    First: rows within a race are dependent, so a random split puts the same
    race on both sides.

    Second, and worse here: the features are built from PAST races. Training on
    a 2024 race and testing on a 2023 one means the training row's features were
    computed from data that includes the test row's race. The leak runs backwards
    through the feature construction, which no amount of row-level shuffling can
    fix.
    """
    if test_season is None:
        test_season = int(df["season_year"].max())
    return (
        df[df["season_year"] < test_season].copy(),
        df[df["season_year"] == test_season].copy(),
    )


def walk_forward_splits(
    df: pd.DataFrame, *, min_train_races: int = 40
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Train on every race before race N, predict race N. Step forward. Repeat.

    This is the only evaluation that mirrors how the model would really be used,
    and it is strictly harder than a single season holdout: early folds train on
    very little. `min_train_races` sets how much history the first fold gets.

    Returns one (train, test) pair per race, so ~80 folds on this data.
    """
    events = (
        df[["event_id", "event_date"]]
        .drop_duplicates()
        .sort_values("event_date")
        .reset_index(drop=True)
    )
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(min_train_races, len(events)):
        cutoff = events.loc[i, "event_date"]
        target_event = events.loc[i, "event_id"]
        train = df[df["event_date"] < cutoff]
        test = df[df["event_id"] == target_event]
        if train.empty or test.empty:
            continue
        splits.append((train.copy(), test.copy()))
    return splits


def audit_leakage(df: pd.DataFrame) -> dict[str, object]:
    """Structural checks that the as-of features really are as-of.

    Two properties that must hold, and would not if a window frame were wrong:

    1. A team-level feature computed over PREVIOUS races must be IDENTICAL for
       both of a team's cars in the same race. If the two teammates disagree,
       the window reached into the current race -- which is exactly what a
       `ROWS` frame does instead of `GROUPS`.
    2. A driver's first-ever race must have no form features. A non-null value
       there means the window included the race itself.
    """
    report: dict[str, object] = {}

    teammate_disagreements = 0
    for (_, _), group in df.groupby(["event_id", "team"], observed=True):
        if len(group) < 2:
            continue
        values = group["team_form_3"].dropna().unique()
        if len(values) > 1:
            teammate_disagreements += 1
    report["teammate_team_form_disagreements"] = teammate_disagreements

    first_races = df.sort_values("event_date").groupby("driver", observed=True).head(1)
    report["first_race_rows"] = len(first_races)
    report["first_race_with_form"] = int(first_races["driver_form_3"].notna().sum())

    report["passes"] = (
        teammate_disagreements == 0 and int(first_races["driver_form_3"].notna().sum()) == 0
    )
    return report
