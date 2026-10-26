"""Request and response models for the race-prediction endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GridEntry(BaseModel):
    driver: str = Field(..., description="Driver abbreviation, e.g. VER")
    grid_position: int = Field(..., ge=1, le=24)


class PredictFromGridRequest(BaseModel):
    circuit: str = Field(..., description="Circuit name as stored, e.g. 'Monza'")
    grid: list[GridEntry] = Field(..., min_length=2, max_length=24)


class DriverPredictionOut(BaseModel):
    predicted_rank: int
    driver: str
    team: str | None = None
    grid_position: float | None = None
    predicted_position: float = Field(
        ..., description="Raw regression output; use predicted_rank for the order"
    )
    podium_probability: float = Field(
        ...,
        description=(
            "From a separate calibrated classifier, not derived from "
            "predicted_position. A regression output of 3.4 is not '34%'."
        ),
    )
    actual_position: float | None = Field(
        None, description="Populated only when predicting a race already in the data"
    )
    missing_features: list[str] = Field(
        default_factory=list,
        description=(
            "Features with no data for this driver. A rookie has no form history, "
            "so their prediction rests on grid position alone."
        ),
    )


class RacePredictionOut(BaseModel):
    circuit: str
    event_id: int | None = None
    season_year: int | None = None
    round: int | None = None
    model_version: str
    model_algorithm: str
    drivers: list[DriverPredictionOut]
    warnings: list[str] = Field(default_factory=list)


class MetricsOut(BaseModel):
    mae_positions: float
    spearman: float = Field(..., description="Mean per-race rank correlation")
    top3_hit_rate: float
    top1_hit_rate: float
    n: int
    n_races: int


class BacktestOut(BaseModel):
    """Walk-forward results, reported beside the baselines on purpose.

    An MAE of 3.14 positions is unreadable without knowing that predicting the
    grid order scores 3.52.
    """

    model_version: str
    algorithm: str
    train_season_min: int | None = None
    train_season_max: int | None = None
    n_train_rows: int | None = None

    walk_forward: MetricsOut
    baseline_grid: MetricsOut
    baseline_driver_form: MetricsOut
    improvement_over_grid_mae: float = Field(
        ..., description="Fractional MAE reduction vs predicting grid order"
    )
    all_models: dict[str, MetricsOut] = Field(
        default_factory=dict, description="Every candidate's walk-forward metrics"
    )
    podium_brier: float | None = None
    podium_brier_base_rate: float | None = None
    podium_skill: float | None = Field(
        None, description="1 - brier/brier_base_rate; positive means better than the base rate"
    )
    notes: list[str] = Field(default_factory=list)
