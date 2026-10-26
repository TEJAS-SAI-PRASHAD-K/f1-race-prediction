"""Race prediction endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as OrmSession

from app.db.deps import get_db, get_engine
from app.ml.prediction import predict as race_predict
from app.schemas.prediction import (
    BacktestOut,
    DriverPredictionOut,
    MetricsOut,
    PredictFromGridRequest,
    RacePredictionOut,
)

router = APIRouter(prefix="/prediction", tags=["prediction"])

HONEST_NOTES = [
    "Trained on 2,556 driver-race rows from 128 races. That is a SMALL, "
    "strongly-correlated dataset and overfitting it is easy.",
    "'Finishing position = grid order' is a strong baseline because overtaking in "
    "modern F1 is hard. Every metric here is reported against it.",
    "Ridge beats both tree models on this data. At 100x the rows a booster wins "
    "instead -- the amount of data decides the model.",
    "Walk-forward backtest: trained on every race before race N, predicted race N, "
    "stepped forward. The only evaluation that mirrors real use.",
]


def _no_model() -> HTTPException:
    return HTTPException(
        503,
        "No race-prediction model is registered yet. "
        "Train one with: python -m app.ml.prediction.train",
    )


def _driver_out(d) -> DriverPredictionOut:
    return DriverPredictionOut(
        predicted_rank=d.predicted_rank,
        driver=d.driver,
        team=d.team,
        grid_position=d.grid_position,
        predicted_position=d.predicted_position,
        podium_probability=d.podium_probability,
        actual_position=d.actual_position,
        missing_features=d.missing_features,
    )


def _race_out(r) -> RacePredictionOut:
    return RacePredictionOut(
        circuit=r.circuit,
        event_id=r.event_id,
        season_year=r.season_year,
        round=r.round,
        model_version=r.model_version,
        model_algorithm=r.model_algorithm,
        drivers=[_driver_out(d) for d in r.drivers],
        warnings=r.warnings,
    )


@router.get("/race/{event_id}", response_model=RacePredictionOut)
def predict_race(
    event_id: int,
    db: OrmSession = Depends(get_db),
    engine: Engine = Depends(get_engine),
):
    """What the model says about a race already in the data.

    **These numbers are IN-SAMPLE.** The active model was fitted on every season
    including this race, so its accuracy here is not an estimate of anything.
    The response says so in `warnings`, and `/prediction/backtest` has the honest
    number.

    Useful for inspecting reasoning -- comparing predicted order against what
    actually happened, driver by driver -- not for claiming performance.
    """
    try:
        return _race_out(race_predict.predict_event(engine, event_id, db=db))
    except (LookupError, FileNotFoundError) as exc:
        if "no active model" in str(exc) or isinstance(exc, FileNotFoundError):
            raise _no_model() from exc
        raise HTTPException(404, str(exc)) from exc


@router.post("/race", response_model=RacePredictionOut)
def predict_upcoming(
    request: PredictFromGridRequest,
    db: OrmSession = Depends(get_db),
    engine: Engine = Depends(get_engine),
):
    """Predict an upcoming race from a grid you supply.

    As-of features come from each driver's most recent race. That is an
    approximation by one race, and it is the honest best available before the
    race happens.

    A driver with no history is NOT silently imputed: their missing features are
    listed and a warning is returned. A rookie's prediction rests almost entirely
    on grid position, which is the correct answer and worth saying out loud.
    """
    grid = {entry.driver.upper(): entry.grid_position for entry in request.grid}
    if len(grid) != len(request.grid):
        raise HTTPException(422, "duplicate driver in grid")
    try:
        return _race_out(race_predict.predict_from_grid(engine, request.circuit, grid, db=db))
    except (LookupError, FileNotFoundError) as exc:
        raise _no_model() from exc


@router.get("/backtest", response_model=BacktestOut)
def backtest(db: OrmSession = Depends(get_db)):
    """Walk-forward performance, reported beside the baselines.

    An MAE of 3.14 positions means nothing on its own. Predicting the grid order
    scores 3.52, so the model is ~11% better -- a real but modest gain, which is
    the honest characterisation of this problem.
    """
    try:
        metrics = race_predict.backtest_summary(db)
    except (LookupError, FileNotFoundError) as exc:
        raise _no_model() from exc

    from sqlalchemy import select

    from app.db.models import ModelVersion

    row = db.execute(
        select(ModelVersion).where(
            ModelVersion.name == "race_prediction", ModelVersion.is_active.is_(True)
        )
    ).scalar_one_or_none()
    if row is None:
        raise _no_model()

    baselines = metrics.get("baselines", {})
    grid_metrics = baselines.get("baseline_grid")
    if not grid_metrics:
        raise HTTPException(500, "registered model has no baseline metrics; retrain")

    podium = None
    all_models = metrics.get("all_models", {})
    try:
        podium_row = db.execute(
            select(ModelVersion).where(
                ModelVersion.name == "race_prediction_podium",
                ModelVersion.is_active.is_(True),
            )
        ).scalar_one_or_none()
        if podium_row and podium_row.metrics_json:
            import json

            podium = json.loads(podium_row.metrics_json)
    except Exception:  # noqa: BLE001
        podium = None

    walk_forward = MetricsOut(**{k: metrics[k] for k in MetricsOut.model_fields})

    return BacktestOut(
        model_version=row.version,
        algorithm=str(metrics.get("algorithm", row.backend)),
        train_season_min=row.train_season_min,
        train_season_max=row.train_season_max,
        n_train_rows=row.n_train_rows,
        walk_forward=walk_forward,
        baseline_grid=MetricsOut(**{k: grid_metrics[k] for k in MetricsOut.model_fields}),
        baseline_driver_form=MetricsOut(
            **{k: baselines["baseline_driver_form"][k] for k in MetricsOut.model_fields}
        ),
        improvement_over_grid_mae=round(
            1 - metrics["mae_positions"] / grid_metrics["mae_positions"], 5
        ),
        all_models={
            name: MetricsOut(**{k: vals[k] for k in MetricsOut.model_fields})
            for name, vals in all_models.items()
        },
        podium_brier=round(podium["brier"], 5) if podium else None,
        podium_brier_base_rate=round(podium["brier_base_rate"], 5) if podium else None,
        podium_skill=(
            round(1 - podium["brier"] / podium["brier_base_rate"], 5) if podium else None
        ),
        notes=HONEST_NOTES,
    )
