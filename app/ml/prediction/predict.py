"""Serve race predictions.

Two modes, and the difference matters:

`predict_event` predicts a race that is ALREADY IN THE DATA, using the as-of
features the view computed for it. That is a backtest view -- useful for
inspecting what the model would have said, and the only mode that can be scored.

`predict_from_grid` predicts an upcoming race from a supplied grid, assembling
as-of features from each driver's history up to now. That is the real
application, and it is where the unseen-category and missing-history problems
bite: a rookie has no form, and a brand-new circuit has no history.

Both return a RANKED order plus a podium probability. The position model gives
order; the classifier gives calibrated probability. Converting positions to
probabilities directly would be wrong -- a regression output of 3.4 is not
"34% chance of a podium".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session as OrmSession

from app.db.base import SessionLocal
from app.ml import registry
from app.ml.prediction import dataset as ds
from app.ml.prediction.train import MODEL_NAME, PODIUM_MODEL_NAME

_CACHE: dict[tuple[str, str], registry.LoadedModel] = {}


def clear_cache() -> None:
    _CACHE.clear()


def _load(name: str, db: OrmSession | None = None) -> registry.LoadedModel:
    """Read the active version fresh, reuse the cached artifact when it matches."""
    if db is not None:
        return _load_with_session(name, db)
    with SessionLocal() as own:
        return _load_with_session(name, own)


def _load_with_session(name: str, db: OrmSession) -> registry.LoadedModel:
    from sqlalchemy import select

    from app.db.models import ModelVersion

    version = db.execute(
        select(ModelVersion.version).where(
            ModelVersion.name == name, ModelVersion.is_active.is_(True)
        )
    ).scalar_one_or_none()
    if version is None:
        raise LookupError(f"no active model registered as {name!r}")

    cached = _CACHE.get((name, version))
    if cached is not None:
        return cached
    loaded = registry.load_model(db, name, version=version)
    _CACHE[(name, version)] = loaded
    return loaded


@dataclass
class DriverPrediction:
    driver: str
    team: str | None
    grid_position: float | None
    predicted_position: float
    predicted_rank: int
    podium_probability: float
    actual_position: float | None = None
    missing_features: list[str] = field(default_factory=list)


@dataclass
class RacePrediction:
    event_id: int | None
    season_year: int | None
    round: int | None
    circuit: str
    model_version: str
    model_algorithm: str
    drivers: list[DriverPrediction]
    warnings: list[str] = field(default_factory=list)


def _rank_and_package(
    frame: pd.DataFrame,
    positions: np.ndarray,
    podium: np.ndarray,
    loaded: registry.LoadedModel,
    *,
    circuit: str,
    event_id: int | None,
    season_year: int | None,
    round_number: int | None,
    warnings: list[str],
) -> RacePrediction:
    order = np.argsort(positions, kind="stable")
    ranks = np.empty(len(positions), dtype=int)
    ranks[order] = np.arange(1, len(positions) + 1)

    feature_cols = [c for c in loaded.feature_names if c in frame.columns]
    drivers: list[DriverPrediction] = []
    for i, row in enumerate(frame.itertuples(index=False)):
        missing = [c for c in feature_cols if pd.isna(getattr(row, c, np.nan))]
        drivers.append(
            DriverPrediction(
                driver=row.driver,
                team=_as_str(getattr(row, "team", None)),
                grid_position=_as_float(getattr(row, "grid_position", None)),
                predicted_position=round(float(positions[i]), 3),
                predicted_rank=int(ranks[i]),
                podium_probability=round(float(podium[i]), 4),
                actual_position=_as_float(getattr(row, ds.TARGET_POSITION, None)),
                missing_features=missing,
            )
        )

    drivers.sort(key=lambda d: d.predicted_rank)
    return RacePrediction(
        event_id=event_id,
        season_year=season_year,
        round=round_number,
        circuit=circuit,
        model_version=loaded.version,
        model_algorithm=str(loaded.metrics.get("algorithm", loaded.backend)),
        drivers=drivers,
        warnings=warnings,
    )


def _as_str(value) -> str | None:
    """pandas turns a missing string into NaN, which is a float.

    Without this an unknown driver's `team` arrives as float('nan') and Pydantic
    rejects it for a `str | None` field -- a real bug the rookie test caught.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _as_float(value) -> float | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(out) else out


def predict_event(
    engine, event_id: int, *, db: OrmSession | None = None
) -> RacePrediction:
    """Predict a race already in the data, using its stored as-of features.

    IMPORTANT: the model served here was fitted on ALL seasons including this
    race, so these numbers are in-sample and NOT a performance estimate. The
    honest number is the walk-forward backtest in `/prediction/backtest`. This
    endpoint is for inspecting reasoning, not for claiming accuracy.
    """
    position_model = _load(MODEL_NAME, db)
    podium_model = _load(PODIUM_MODEL_NAME, db)

    frame = pd.read_sql(
        text("SELECT * FROM features.race_features WHERE event_id = :eid"),
        engine,
        params={"eid": event_id},
    )
    if frame.empty:
        raise LookupError(f"no race features for event_id {event_id}")

    frame = ds.add_degradation_feature(frame, engine)
    for col in ds.IDENTITY_FEATURES:
        frame[col] = frame[col].astype("category")

    X = frame[position_model.feature_names]
    positions = np.asarray(position_model.model.predict(X), dtype=float)
    podium = np.asarray(podium_model.model.predict_proba(X)[:, 1], dtype=float)

    first = frame.iloc[0]
    return _rank_and_package(
        frame,
        positions,
        podium,
        position_model,
        circuit=str(first["circuit"]),
        event_id=int(first["event_id"]),
        season_year=int(first["season_year"]),
        round_number=int(first["round"]),
        warnings=[
            "In-sample: the active model was trained on all seasons including this "
            "race. For an honest accuracy estimate use /prediction/backtest."
        ],
    )


def predict_from_grid(
    engine,
    circuit: str,
    grid: dict[str, int],
    *,
    db: OrmSession | None = None,
) -> RacePrediction:
    """Predict an upcoming race from a grid of {driver_abbreviation: position}.

    As-of features come from each driver's most recent row in the view, i.e.
    their state going into their last race. That is an approximation -- strictly
    it should be their state going into THIS race, which differs by one race --
    and it is the honest best available without the race having happened.

    A driver with no history gets NaN features and is reported in
    `missing_features`, not silently imputed. A rookie's prediction rests almost
    entirely on grid position, which is the right answer and worth saying out
    loud rather than hiding behind a median.
    """
    position_model = _load(MODEL_NAME, db)
    podium_model = _load(PODIUM_MODEL_NAME, db)

    history = pd.read_sql(
        text("SELECT * FROM features.race_features ORDER BY event_date"), engine
    )
    warnings: list[str] = []

    latest = (
        history.sort_values("event_date").groupby("driver", observed=True).tail(1)
        if not history.empty
        else history
    )

    rows: list[dict] = []
    for driver, position in grid.items():
        driver = driver.upper()
        match = latest[latest["driver"] == driver]
        if match.empty:
            warnings.append(
                f"{driver}: no race history. Their prediction rests on grid position alone."
            )
            row = {col: np.nan for col in ds.NUMERIC_FEATURES}
            row.update({"driver": driver, "team": None, "circuit": circuit})
        else:
            source = match.iloc[0]
            row = {col: source.get(col, np.nan) for col in ds.PRE_RACE_FEATURES}
            row.update(
                {
                    "driver": driver,
                    "team": source.get("team"),
                    "circuit": circuit,
                    ds.DEGRADATION_FEATURE: np.nan,
                }
            )
        # Grid and quali are the one thing we know about THIS race.
        row["grid_position"] = float(position)
        row["quali_position"] = float(position)
        rows.append(row)

    frame = pd.DataFrame(rows)

    # Degradation for this circuit, from everything known so far.
    frame["season_year"] = int(history["season_year"].max()) + 1 if not history.empty else 0
    try:
        frame = ds.add_degradation_feature(frame, engine)
    except Exception:  # noqa: BLE001
        frame[ds.DEGRADATION_FEATURE] = np.nan

    if circuit not in set(history["circuit"].astype(str)):
        warnings.append(
            f"{circuit!r} has no race history; circuit features fall back to missing."
        )

    for col in ds.IDENTITY_FEATURES:
        if col in frame.columns:
            frame[col] = frame[col].astype("category")

    X = frame.reindex(columns=position_model.feature_names)
    positions = np.asarray(position_model.model.predict(X), dtype=float)
    podium = np.asarray(podium_model.model.predict_proba(X)[:, 1], dtype=float)

    return _rank_and_package(
        frame,
        positions,
        podium,
        position_model,
        circuit=circuit,
        event_id=None,
        season_year=None,
        round_number=None,
        warnings=warnings,
    )


def backtest_summary(db: OrmSession | None = None) -> dict:
    """The registered model's walk-forward metrics and its baselines.

    Returns the stored numbers rather than recomputing: a walk-forward backtest
    is 81 model fits and takes minutes, which is not an API request.
    """
    loaded = _load(MODEL_NAME, db)
    return loaded.metrics
